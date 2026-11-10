"""CPU shard B: IK roundtrip stress + plan_pick over pool-3-style layouts. No accelerator."""
import json, os, sys, time, zipfile, traceback, shutil
from pathlib import Path
import importlib.util

IN=Path('/kaggle/input/team048-tpu-cpu-inputs'); OUT=Path('/kaggle/working'); LOG=OUT/'ik_p3.log'


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

def unpack():
    global IN
    IN = find_input(); log(f'IN={IN} listing={list(IN.iterdir())[:30]}')
    work = materialize(IN, OUT/'in')
    sol=work/'solution_pkg'
    if not (sol/'arm.py').exists():
        c=list(work.glob('**/arm.py')); sol=c[0].parent if c else sol
    return sol

def build_pkg(sol: Path):
    pkg=OUT/'team048_solution'; pkg.mkdir(exist_ok=True); (pkg/'__init__.py').write_text('')
    for fn in ('geometry.py','kinematics.py','planning.py','perception.py','grasp.py','arm.py'):
        src=sol/fn
        if src.exists(): (pkg/fn).write_text(src.read_text())
    datap=pkg/'data'; datap.mkdir(exist_ok=True)
    urdf=sol/'data'/'tiangong_pro_with_hands.urdf'
    if urdf.exists(): (datap/urdf.name).write_bytes(urdf.read_bytes())
    priors=sol/'data'/'priors.json'
    if priors.exists(): (datap/'priors.json').write_text(priors.read_text())
    if not (pkg/'arm.py').exists():
        raise FileNotFoundError('arm.py not copied from '+str(sol)+' files='+str(list(sol.glob('**/*'))[:40]))
    sys.path.insert(0,str(OUT)); os.environ['HACK_URDF']=str(datap/'tiangong_pro_with_hands.urdf')
    return datap/'tiangong_pro_with_hands.urdf'

def main():
    LOG.write_text(''); rep={'ok':False,'pool':POOL,'n_layouts':N_LAYOUTS}
    try:
        import numpy as np
        sol=unpack(); urdf=build_pkg(sol); log(f'sol={sol} urdf_exists={urdf.exists()}')
        from team048_solution.arm import Arm
        A=Arm('l'); table_z=0.70-0.88
        # FK/IK
        rng=np.random.default_rng(42); ok=0; n=300; errs=[]; t0=time.time()
        for _ in range(n):
            q=rng.uniform(A.chain.lower,A.chain.upper)*.4; T=A.chain.fk(q)
            try:
                qq=A.chain.ik(T,A.home,max_iter=300); e=float(np.linalg.norm(A.chain.fk(qq)[:3,3]-T[:3,3])); errs.append(e); ok+=e<0.003
            except Exception: pass
        rep['fk_ik']={'n':n,'ok_lt_3mm':ok,'ok_frac':round(ok/n,4),'err_median_m':float(np.median(errs)) if errs else None,
                      'err_p95_m':float(np.percentile(errs,95)) if errs else None,'seconds':round(time.time()-t0,3),
                      'dof':len(A.chain.names),'vmax':[float(x) for x in A.chain.velocity]}
        log(f'fk_ik={rep["fk_ik"]}')
        # plans
        t1=time.time(); success=0; rejected=0; speeds=[]; fails=[]
        for seed in range(N_LAYOUTS):
            rr=np.random.default_rng(SEED0+seed)
            for i in range(N_OBJ):
                xy=np.array([rr.uniform(0.26,0.50), rr.uniform(-0.52,0.52), table_z+0.04])
                width=float(rr.uniform(0.03,0.09))
                p=A.plan_pick(xy,np.array([0.,1.,0.]),np.array([1.,0.,0.]),width)
                if p:
                    success+=1; speeds.append(float(p.get('max_speed',0)))
                else:
                    rejected+=1
                    if len(fails)<20:
                        fails.append({'seed':seed,'i':i,'xy':xy.tolist(),'width':width})
        rep['plan']={'layouts':N_LAYOUTS,'attempts':N_LAYOUTS*N_OBJ,'success':success,'rejected':rejected,
                     'success_frac':round(success/max(N_LAYOUTS*N_OBJ,1),4),
                     'max_speed_p95':float(np.percentile(speeds,95)) if speeds else None,
                     'speed_cap_ok':bool(all(s<=0.8+1e-6 for s in speeds)),
                     'seconds':round(time.time()-t1,3),'fail_samples':fails}
        log(f'plan={ {k:v for k,v in rep["plan"].items() if k!="fail_samples"} }')
        rep['ok']=True; log('ALL OK')
    except Exception as e:
        rep['error']=repr(e); rep['traceback']=traceback.format_exc(); log(f'FAIL {e}'); log(rep['traceback'])
    (OUT/'ik_p3_results.json').write_text(json.dumps(rep,indent=2)); return 0 if rep.get('ok') else 1
if __name__=='__main__': raise SystemExit(main())
