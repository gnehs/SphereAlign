"""Compare native probe f32le heads against explicit CPU ONNX golden heads.

One fixture is numerical evidence, not model accuracy or provider certification.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference",type=Path)
    p.add_argument("native",type=Path)
    p.add_argument("--dependencies",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.dependencies.resolve()))
    import numpy as np
    ref=np.load(args.reference/"raw-heads.npz",allow_pickle=False)
    native={k:np.fromfile(args.native/(k+".f32le"),dtype="<f4").reshape(ref[k].shape) for k in ref.files}
    report={"scope":"single_synthetic_pattern_not_provider_certification","heads":{}}
    for k in ref.files:
        a,b=ref[k].astype(np.float64),native[k].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("nonfinite heads")
        report["heads"][k]={"relativeL2":float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12)),"maxAbs":float(np.max(np.abs(a-b)))}
    a,b=ref["normal"].astype(np.float64),native["normal"].astype(np.float64)
    a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True)
    angles=np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1),-1,1)))
    report["normalAngleDeg"]={"median":float(np.median(angles)),"p99":float(np.quantile(angles,.99)),"max":float(angles.max())}
    report["validityDisagreementFraction"]=float(np.mean((ref["mask"]>=.5)!=(native["mask"]>=.5)))
    report["nearThresholdPixels"]=int(np.count_nonzero(np.abs(ref["mask"]-.5)<.01))
    # Independent known-focal diagnostic using normalized camera coordinates.
    # This arbitrary K is common to both numerical inputs, not a claim that the
    # RGB pattern depicts a scene with meaningful recovered depths.
    def recover(points,mask):
        _,h,w,_=points.shape
        yy,xx=np.mgrid[0:h:12,0:w:12]
        pts=points[0,::12,::12].astype(np.float64).reshape(-1,3)
        uv=np.stack([(xx+.5-w/2)/320,(yy+.5-h/2)/320],axis=-1).reshape(-1,2)
        valid=mask[0,::12,::12].reshape(-1)>=.5
        pts,uv=pts[valid],uv[valid]
        lo=-float(pts[:,2].min())+1e-7;hi=float(np.median(pts[:,2]))*16
        def objective(s):return float(np.mean((pts[:,:2]/(pts[:,2:]+s)-uv)**2))
        grid=np.linspace(lo,hi,1025);best=int(np.argmin([objective(x) for x in grid]))
        if best in [0,len(grid)-1]:raise ValueError("unbracketed shift; no recovery certificate")
        a,b=grid[best-1],grid[best+1]
        for _ in range(80):
            c=a+(b-a)*.3819660112501051;d=a+(b-a)*.6180339887498949
            if objective(c)<objective(d):b=d
            else:a=c
        s=(a+b)/2
        z=points[...,2].astype(np.float64)+s
        return s,z
    sa,za=recover(ref["points"],ref["mask"]);sb,zb=recover(native["points"],native["mask"])
    valid=(za>0)&(zb>0)&(ref["mask"]>=.5)&(native["mask"]>=.5)
    report["knownFocalDiagnostic"]={"fx":320,"referenceShift":sa,"nativeShift":sb,
                                     "positiveZRelativeL2":float(np.linalg.norm((za-zb)[valid])/np.linalg.norm(za[valid]))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))


if __name__ == "__main__":main()
