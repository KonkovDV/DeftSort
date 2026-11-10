"""CPU shard: FK/IK roundtrip stress only (no plan_pick). No accelerator."""
import json, os, sys, time, traceback
from pathlib import Path
IN=Path("/kaggle/input/team048-tpu-cpu-inputs"); OUT=Path("/kaggle/working"); LOG=OUT/"fk.log"
N=int(os.environ.get("N_IK","2000"))
def log(m):
    line=f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {m}"
    print(line, flush=True); LOG.open("a").write(line+"\n")
def find_input():
    base = Path('/kaggle/input')
    if not base.exists():
        raise FileNotFoundError('no /kaggle/input')
    # New Kaggle layout nests user datasets under /kaggle/input/datasets/<user>/<slug>
    cands = []
    for p in base.rglob('features.json'):
        # perception/features.json -> dataset root is parent of perception
        if p.parent.name == 'perception':
            cands.append(p.parent.parent)
    for p in base.rglob('arm.py'):
        if p.parent.name == 'solution_pkg':
            cands.append(p.parent.parent)
    for p in base.rglob('cam_head_color.npy'):
        cands.append(p.parent.parent)
    # also accept direct slug mounts
    for p in [base/'team048-tpu-cpu-inputs', base/'datasets'/'dmitrijkv'/'team048-tpu-cpu-inputs']:
        if p.exists():
            cands.append(p)
    # unique preserve order
    seen=set(); out=[]
    for c in cands:
        s=str(c)
        if s not in seen:
            seen.add(s); out.append(c)
    if not out:
        listing=[]
        for p in base.rglob('*'):
            listing.append(str(p))
            if len(listing)>=40: break
        raise FileNotFoundError('dataset tree not found; sample='+str(listing))
    return out[0]

def materialize(IN, work):
    import shutil, zipfile
    work.mkdir(parents=True, exist_ok=True)
    for name in ('perception', 'solution_pkg', 'v5_frames', 'ik'):
        src = IN / name
        dst = work / name
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                dst.write_bytes(src.read_bytes())
    for z in list(IN.glob('*.zip')) + list(IN.glob('**/*.zip')):
        dest = work / z.stem
        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)
            zipfile.ZipFile(z).extractall(dest)
    return work


N_LAYOUTS=int(os.environ.get('N_LAYOUTS','120')); N_OBJ=8; POOL=3; SEED0=30_000

def log(m):
    line=f"{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())} {m}"; print(line,flush=True); LOG.open('a').write(line+'\n')

def build_pkg(sol):
    pkg=OUT/"team048_solution"; pkg.mkdir(exist_ok=True); (pkg/"__init__.py").write_text("")
    for fn in ("geometry.py","kinematics.py","planning.py","perception.py","grasp.py","arm.py"):
        src=sol/fn
        if src.exists(): (pkg/fn).write_text(src.read_text())
    datap=pkg/"data"; datap.mkdir(exist_ok=True)
    urdf=sol/"data"/"tiangong_pro_with_hands.urdf"
    if urdf.exists(): (datap/urdf.name).write_bytes(urdf.read_bytes())
    priors=sol/"data"/"priors.json"
    if priors.exists(): (datap/"priors.json").write_text(priors.read_text())
    if not (pkg/"arm.py").exists(): raise FileNotFoundError("arm.py missing")
    sys.path.insert(0,str(OUT)); os.environ["HACK_URDF"]=str(datap/"tiangong_pro_with_hands.urdf")
def main():
    LOG.write_text(""); rep={"ok":False}
    try:
        import numpy as np
        IN2=find_input(); log(f"IN={IN2}")
        work=materialize(IN2, OUT/"in")
        sol=work/"solution_pkg"
        if not (sol/"arm.py").exists():
            c=list(work.glob("**/arm.py")); sol=c[0].parent if c else sol
        build_pkg(sol)
        from team048_solution.arm import Arm
        A=Arm("l"); log(f"dof={len(A.chain.names)} vmax={list(map(float,A.chain.velocity))}")
        rng=np.random.default_rng(123); ok=0; errs=[]; fails=0; t0=time.time()
        for _ in range(N):
            q=rng.uniform(A.chain.lower,A.chain.upper)*0.4; T=A.chain.fk(q)
            try:
                qq=A.chain.ik(T,A.home,max_iter=300)
                e=float(np.linalg.norm(A.chain.fk(qq)[:3,3]-T[:3,3])); errs.append(e); ok += e<0.003
            except Exception:
                fails += 1
        rep["fk_ik"]={"n":N,"ok_lt_3mm":ok,"ok_frac":round(ok/N,4),"fail_exceptions":fails,
                      "err_median_m":float(np.median(errs)) if errs else None,
                      "err_p95_m":float(np.percentile(errs,95)) if errs else None,
                      "err_max_m":float(np.max(errs)) if errs else None,
                      "seconds":round(time.time()-t0,3)}
        # hand opening sanity
        h=A.hand; rep["hand"]={"opposed_open":float(h["opposed_open"]),"close_axis_norm":float(np.linalg.norm(h["close_axis_hand"]))}
        rep["ok"]=True; log(str(rep["fk_ik"])); log("ALL OK")
    except Exception as e:
        rep["error"]=repr(e); rep["traceback"]=traceback.format_exc(); log("FAIL "+repr(e))
    (OUT/"fk_results.json").write_text(json.dumps(rep, indent=2)); return 0 if rep.get("ok") else 1
if __name__=="__main__": raise SystemExit(main())
