"""Actual trainer smoke on analytic fixtures, NOT a quality/gradient certificate.

Captures binary identity, --help-all, exact argv and bounded subprocess results.
Only use isolated generated fixtures, never a user's live training dataset.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spirula",type=Path,required=True)
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    fixture=json.loads((args.fixture/"fixture.json").read_text(encoding="utf-8"))
    if fixture.get("kind")!="analytic_fixture_not_AI_inference":raise ValueError("requires generated analytic fixture")
    args.output.mkdir(parents=True,exist_ok=False)
    binary=args.spirula.resolve();flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
    with binary.open("rb") as f:digest=hashlib.file_digest(f,"sha256").hexdigest()
    help_result=subprocess.run([str(binary),"train","--help-all","--lang","en"],capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=30,creationflags=flags)
    help_text=help_result.stdout+help_result.stderr
    if help_result.returncode or "bools take 0/1" not in help_text:raise ValueError("unverified CLI boolean syntax")
    (args.output/"train-help-all.txt").write_text(help_text,encoding="utf-8")
    common=["--data",str(args.fixture.resolve()),"--data-format","colmap","--output-dir-prefix",str((args.output/"training").resolve()),
            "--num-iterations","3","--cap-max","512","--min-init-fraction","0","--disable-viewer","1","--eval-mode","all",
            "--median-normal-supervision-weight","0","--supervision-warmup","0","--input-depth-is-ray-depth","1",
            "--depth-unit-scale-factor",str(16/65535),"--sh-degree","0","--lang","en"]
    runs=[]
    for name,normal,depth,divisor,warp in [("normals",1,0,1,0),("depth",0,1,1,0),("priors-div2",1,1,2,0),("priors-warp-div2",1,1,2,1)]:
        argv=[str(binary),"train"]+common+["--output-dir-name",name,"--load-normals",str(normal),"--load-depths",str(depth),
             "--normal-supervision-weight",str(.0025 if normal else 0),"--depth-supervision-weight",str(.01 if depth else 0),
             "--train-resolution-divisor",str(divisor),"--warp-to-pinhole",str(warp)]
        for flag in argv[2::2]:
            if flag!="--lang" and flag not in help_text:raise ValueError(f"flag absent from actual help: {flag}")
        start=time.perf_counter();status="completed"
        with (args.output/(name+"-stdout.txt")).open("w",encoding="utf-8") as out,(args.output/(name+"-stderr.txt")).open("w",encoding="utf-8") as err:
            try:
                code=subprocess.run(argv,stdout=out,stderr=err,timeout=90,creationflags=flags).returncode
                status="completed" if code==0 else "failed"
            except subprocess.TimeoutExpired:code=None;status="timeout"
        runs.append({"name":name,"argv":argv,"exitCode":code,"status":status,"elapsedSeconds":time.perf_counter()-start})
        (args.output/"report.json").write_text(json.dumps({"scope":"actual_binary_analytic_smoke_not_quality_or_gradient_acceptance","binarySha256":digest,"runs":runs},indent=2),encoding="utf-8")
    changed=[p for p,h in fixture["hashes"].items() if hashlib.sha256((args.fixture/p).read_bytes()).hexdigest()!=h]
    report={"scope":"actual_binary_analytic_smoke_not_quality_or_gradient_acceptance","binarySha256":digest,"fixtureFilesChanged":changed,"runs":runs,
            "unverified":["normal sign","loader/rasterizer range units","per-pixel prior gradients","RGB gradients at rejected-prior pixels","multi-scale loss sentinel preservation","real capture A/B quality"]}
    (args.output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
    if changed or any(r["exitCode"]!=0 for r in runs):raise SystemExit(1)


if __name__=="__main__":main()
