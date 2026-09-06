"""Create a tiny analytic COLMAP fixture in a NEW directory (standard library).

Two translated physical lenses, duplicate basenames, tilted walls, native ray
range, independent RGB masks, missing maps, all-invalid maps and a slanted hole.
These are analytic priors; never report this as model-generated end-to-end data.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib


def png(path, width, height, channels, bits, rows):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    payload = b"".join(b"\0" + row for row in rows)
    result = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB",width,height,bits,2 if channels == 3 else 0,0,0,0))
    result += chunk(b"IDAT",zlib.compress(payload)) + chunk(b"IEND",b"")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(result)


def unit(v):
    length = math.sqrt(sum(x*x for x in v))
    return [x / length for x in v]


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--camera",choices=["PINHOLE","OPENCV_FISHEYE"],default="OPENCV_FISHEYE")
    args=p.parse_args()
    if args.output.exists():
        raise ValueError("fixture output must be a new directory")
    args.output.mkdir(parents=True)
    w=h=64
    focal=17.0 if args.camera == "OPENCV_FISHEYE" else 40.0
    params=f"{focal} {focal} 32 32" + (" 0 0 0 0" if args.camera == "OPENCV_FISHEYE" else "")
    cameras="\n".join(f"{i} {args.camera} 64 64 {params}" for i in [3,91])+"\n"
    image_lines=[]
    points={}
    frame_reports=[]
    # Each inward normal n describes n dot X = -3: a closed, tilted box.
    walls=[unit(v) for v in [(0.2,0.15,-1),(-.2,-.15,1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)]]
    for capture in range(4):
        for lens in range(2):
            image_id=101+capture*13+lens*4
            name=f"lens{lens}/frame{capture:03}.png"
            center=[(capture-1.5)*.18 + lens*.06,0,0]
            rotation=[1,1,1] if lens == 0 else [-1,1,-1]
            q="1 0 0 0" if lens == 0 else "0 0 1 0"
            translation=[-rotation[k]*center[k] for k in range(3)]
            image_lines.append(f"{image_id} {q} {' '.join(map(str,translation))} {3 if lens==0 else 91} {name}")
            rgb_rows=[];normal_rows=[];depth_rows=[];mask_rows=[];observations=[]
            for y in range(h):
                rgb=bytearray(); normal=bytearray(); depth=bytearray(); mask=bytearray()
                for x in range(w):
                    a,b=(x+.5-32)/focal,(y+.5-32)/focal
                    rho=math.hypot(a,b)
                    optical=math.hypot(x+.5-32,y+.5-32)<32
                    if args.camera == "OPENCV_FISHEYE":
                        ray=[a*math.sin(rho)/rho,b*math.sin(rho)/rho,math.cos(rho)] if rho else [0,0,1]
                    else: ray=unit([a,b,1]); optical=True
                    ray_world=[ray[k]*rotation[k] for k in range(3)]
                    hits=[((-3-dot(n,center))/dot(n,ray_world),n) for n in walls if dot(n,ray_world)<-1e-10]
                    distance,n=min(hits,key=lambda hit:hit[0])
                    point=[center[k]+distance*ray_world[k] for k in range(3)]
                    nc=[n[k]*rotation[k] for k in range(3)]
                    color=[int(80+(math.floor(point[k]*8)%2)*100) for k in range(3)]
                    rgb.extend(color if optical else [0,0,0]);mask.append(255 if optical else 0)
                    accepted=optical and capture!=1 and x>y*.45+8
                    normal.extend([round((v*.5+.5)*255) for v in nc] if accepted else [0,0,0])
                    quantized=round(distance/16*65535) if accepted and 0<distance<=16 else 0
                    depth.extend(struct.pack(">H",quantized))
                    # Deterministic sparse observations with non-contiguous IDs.
                    if optical and x%8==4 and y%8==4:
                        pid=10001+len(points)*7
                        points[pid]=f"{pid} {' '.join(map(str,point))} {' '.join(map(str,color))} 0 {image_id} {len(observations)}"
                        observations.append((x+.5,y+.5,pid))
                rgb_rows.append(bytes(rgb));normal_rows.append(bytes(normal));depth_rows.append(bytes(depth));mask_rows.append(bytes(mask))
            image_lines.append(" ".join(f"{x} {y} {pid}" for x,y,pid in observations))
            png(args.output/"images"/name,w,h,3,8,rgb_rows)
            png(args.output/"masks"/name,w,h,1,8,mask_rows)
            if capture!=2:  # missing prior maps, RGB remains present
                png(args.output/"normals"/name,w,h,3,8,normal_rows)
                png(args.output/"depths"/name,w,h,1,16,depth_rows)
            frame_reports.append({"imageId":image_id,"name":name,"centerWorld":center,"cameraFromWorldDiagonal":rotation,
                                  "priorCase":"all_invalid" if capture==1 else "missing" if capture==2 else "slanted_hole",
                                  "rangeScale":16,"quantizationStep":16/65535,"normalConvention":"OpenCV source camera, oriented toward interior/camera"})
    sparse=args.output/"sparse/0";sparse.mkdir(parents=True)
    (sparse/"cameras.txt").write_text(cameras,encoding="utf-8")
    (sparse/"images.txt").write_text("\n".join(image_lines)+"\n",encoding="utf-8")
    (sparse/"points3D.txt").write_text("\n".join(points.values())+"\n",encoding="utf-8")
    hashes={p.relative_to(args.output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.rglob("*")) if p.is_file()}
    report={"schemaVersion":1,"kind":"analytic_fixture_not_AI_inference","camera":args.camera,"frames":frame_reports,
            "depthRepresentation":"positive_ray_range","unit":"synthetic_scene_unit_not_metres", "sourceSize":[w,h],
            "trackLimitation":"one observation per point; importer fixture, intentionally insufficient for anchor certification",
            "hashes":hashes}
    (args.output/"fixture.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"images":len(frame_reports),"points":len(points),"hashes":len(hashes)}))


if __name__ == "__main__": main()
