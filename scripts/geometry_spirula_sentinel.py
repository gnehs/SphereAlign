"""Compile exact pinned Spirula CPU sampler functions and reproduce sentinel leaks.

Produces a minimal proposed upstream patch, never changes the source snapshot.
This is a host sampler test, NOT actual trainer gradient certification.
Requires a C++17 compiler (e.g. g++), no Python packages.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


def extract_function(source, begin, end):
    return source[source.index(begin):source.index(end, source.index(begin))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path(".work/geometry-spike/spirula"))
    p.add_argument("--output", type=Path, default=Path(".work/geometry-spike/sentinel"))
    p.add_argument("--patch", type=Path, default=Path("patches/spirula/geometry-sentinel.patch"))
    p.add_argument("--compiler", default="g++")
    args = p.parse_args()
    cpu_path = "src/data/DataManager.cpp"
    gpu_path = "src/core/Interpolation.cuh"
    cpu = (args.source / cpu_path).read_text(encoding="utf-8")
    gpu = (args.source / gpu_path).read_text(encoding="utf-8")
    expected = {
        cpu_path: "9cede3dd716693c9bfd96d63e6a5f37485164177c21f748c4685486b5bbd7006",
        gpu_path: "5b7796ae21116c1dbd0f5a7ca715e5f1ec5a19589cefe2a6410d0a37b589b334",
    }
    for path, sha in expected.items():
        if sha and hashlib.sha256((args.source / path).read_bytes()).hexdigest() != sha:
            raise ValueError(f"source hash mismatch: {path}; do not apply to another version")
    cpu_function = extract_function(cpu, "template<typename T, int C>\ninline void cpu_bilinear_resize", "// Exact box average")
    gpu_function = extract_function(gpu, "__device__ __forceinline__ bool revalidate_weights", "__device__ __forceinline__ bool gt_depth_valid")
    guard = """            if constexpr (Geometry) {
                auto valid = [](const T* p) {
                    if constexpr (C == 1) return p[0] != 0;
                    else {
                        float x = float(p[0]) * (2.0f / 255.0f) - 1.0f;
                        float y = float(p[1]) * (2.0f / 255.0f) - 1.0f;
                        float z = float(p[2]) * (2.0f / 255.0f) - 1.0f;
                        return x+y+z > -2.366f && x*x+y*y+z*z > 0.25f;
                    }
                };
                if ((w00 > 0 && !valid(p00)) || (w10 > 0 && !valid(p10)) ||
                    (w01 > 0 && !valid(p01)) || (w11 > 0 && !valid(p11))) {
                    for (int c = 0; c < C; ++c) q[c] = 0;
                    continue;
                }
            }
"""
    fixed_cpu_function = cpu_function.replace("template<typename T, int C>", "template<typename T, int C, bool Geometry = false>", 1)
    marker = "            for (int c = 0; c < C; ++c) {"
    fixed_cpu_function = fixed_cpu_function.replace(marker, guard + marker, 1)
    fixed_cpu = cpu.replace(cpu_function, fixed_cpu_function, 1)
    fixed_cpu = fixed_cpu.replace("cpu_bilinear_resize<stbi_us, 1>(img", "cpu_bilinear_resize<stbi_us, 1, true>(img")
    fixed_cpu = fixed_cpu.replace("cpu_bilinear_resize<stbi_uc, 3>(img", "cpu_bilinear_resize<stbi_uc, 3, true>(img")
    gpu_guard = """    // Strict prior support: do not extend a valid neighbour into a rejected tap.
    if ((!v00 && w00 > 0) || (!v10 && w10 > 0) ||
        (!v01 && w01 > 0) || (!v11 && w11 > 0)) return false;
"""
    fixed_gpu_function = gpu_function.replace("    if (!v00)", gpu_guard + "    if (!v00)", 1)
    fixed_gpu = gpu.replace(gpu_function, fixed_gpu_function, 1)
    patch = ""
    for path, old, new in [(cpu_path, cpu, fixed_cpu), (gpu_path, gpu, fixed_gpu)]:
        patch += "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile="a/"+path, tofile="b/"+path))
    args.patch.parent.mkdir(parents=True, exist_ok=True)
    args.patch.write_text(patch, encoding="utf-8")
    harness = """#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <cassert>
#include <iostream>
#define __device__
#define __forceinline__ inline
"""
    harness += "namespace stock {\n" + cpu_function + gpu_function + "}\n"
    harness += "namespace fixed {\n" + fixed_cpu_function + fixed_gpu_function + "}\n"
    harness += r'''
int main() {
    uint16_t depth[2] = {0, 1000}, a[4], b[4];
    stock::cpu_bilinear_resize<uint16_t,1>(depth,1,2,a,1,4);
    fixed::cpu_bilinear_resize<uint16_t,1,true>(depth,1,2,b,1,4);
    assert(a[1] == 250 && a[2] == 750);
    assert(b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 1000);
    uint8_t normal[6] = {0,0,0,128,128,255}, na[12], nb[12];
    stock::cpu_bilinear_resize<uint8_t,3>(normal,1,2,na,1,4);
    fixed::cpu_bilinear_resize<uint8_t,3,true>(normal,1,2,nb,1,4);
    assert(na[3] > 0 && nb[3] == 0 && nb[6] == 0);
    // RGB must retain its ordinary interpolation.
    uint8_t rgb[12];
    fixed::cpu_bilinear_resize<uint8_t,3>(normal,1,2,rgb,1,4);
    for(int i=0;i<12;i++) assert(rgb[i] == na[i]);
    float x=.75f,y=.25f,z=0,w=0;
    bool expanded = stock::revalidate_weights(false,true,false,false,x,y,z,w);
    assert(expanded && y == 1);
    x=.75f;y=.25f;z=w=0;
    assert(!fixed::revalidate_weights(false,true,false,false,x,y,z,w));
    // Zero-weight invalid taps cannot destroy a valid exact sample.
    x=1;y=z=w=0;
    assert(fixed::revalidate_weights(true,false,false,false,x,y,z,w));
    uint16_t zeros[16]={}, small[4];
    fixed::cpu_bilinear_resize<uint16_t,1,true>(zeros,4,4,small,2,2);
    for(auto q: small) assert(q == 0);
    // An isolated valid point cannot fill a downscaled missing region.
    zeros[5]=2000;
    fixed::cpu_bilinear_resize<uint16_t,1,true>(zeros,4,4,small,2,2);
    for(auto q: small) assert(q == 0);
    std::cout << "{\"stockCpuDepth\":[" << a[0] << "," << a[1] << "," << a[2] << "," << a[3]
      << "],\"strictCpuDepth\":[" << b[0] << "," << b[1] << "," << b[2] << "," << b[3]
      << "],\"stockGpuSupportExpanded\":true,\"strictTestsPassed\":true,\"rgbUnchanged\":true}";
}
'''
    args.output.mkdir(parents=True, exist_ok=True)
    cpp = args.output / "sentinel.cpp"
    exe = args.output / "sentinel.exe"
    cpp.write_text(harness, encoding="utf-8")
    subprocess.run([args.compiler, "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True)
    result = subprocess.run([str(exe.resolve())], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    report.update({"testScope": "exact_pinned_host_sampler_functions_not_trainer_gradients", "spirulaCommit": "bddda193ee09f03ea1b0aad44b5c6b96f81e249a",
                   "sourceHashes": {p: hashlib.sha256((args.source/p).read_bytes()).hexdigest() for p in expected},
                   "patchSha256": hashlib.sha256(args.patch.read_bytes()).hexdigest()})
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
