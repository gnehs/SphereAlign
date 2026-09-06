"""Measure a bounded native probe subprocess, without stopping other GPU jobs.

GPU values are device-wide observations, not process VRAM or a stage budget.
Windows working-set counters are process-specific. No third-party dependencies.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("command",nargs=argparse.REMAINDER)
    args=p.parse_args();command=args.command
    if command and command[0]=="--":command=command[1:]
    if not command:raise ValueError("a command argv is required")
    args.output.mkdir(parents=True,exist_ok=False)
    flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
    gpu=[];working=[];errors=[]
    if os.name=="nt":
        class Counters(ctypes.Structure):
            _fields_=[("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ["PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage","PrivateUsage"]]
        psapi=ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype=ctypes.c_int
    start=time.perf_counter()
    with (args.output/"stdout.txt").open("w",encoding="utf-8") as out,(args.output/"stderr.txt").open("w",encoding="utf-8") as err:
        process=subprocess.Popen(command,stdout=out,stderr=err,creationflags=flags)
        last_gpu=-2.0
        while process.poll() is None:
            elapsed=time.perf_counter()-start
            if os.name=="nt":
                memory=Counters();memory.cb=ctypes.sizeof(memory)
                if psapi.GetProcessMemoryInfo(int(process._handle),ctypes.byref(memory),memory.cb):
                    working.append({"seconds":elapsed,"peakWorkingSetBytes":memory.PeakWorkingSetSize,"privateBytes":memory.PrivateUsage})
            if elapsed-last_gpu>=1:
                try:
                    r=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3,creationflags=flags)
                    if r.returncode==0:gpu.append({"seconds":elapsed,"deviceWideMiB":r.stdout.strip()})
                except (OSError,subprocess.TimeoutExpired) as e:errors.append(str(e))
                last_gpu=elapsed
            time.sleep(.1)
        code=process.wait()
    report={"scope":"single_probe_process_not_full_geometry_stage", "argv":command,"exitCode":code,
            "elapsedSeconds":time.perf_counter()-start,"peakProcessWorkingSetBytes":max((v["peakWorkingSetBytes"] for v in working),default=None),
            "maxSampledProcessPrivateBytes":max((v["privateBytes"] for v in working),default=None),
            "processMemorySamples":len(working),"gpuSamples":gpu,"errors":errors,
            "limitations":["Device GPU memory includes unrelated jobs; not per-process peak VRAM.","Sampling can miss short peaks; this is not a full dataset stage."]}
    (args.output/"resources.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2));raise SystemExit(code)


if __name__=="__main__":main()
