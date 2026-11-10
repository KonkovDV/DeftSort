"""CPU shard: parse v5 caminfo + TF; report intrinsics/extrinsics facts. No accelerator."""
import json, os, sys, time, traceback, re
from pathlib import Path
IN=Path("/kaggle/input/team048-tpu-cpu-inputs"); OUT=Path("/kaggle/working"); LOG=OUT/"calib.log"
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

def main():
    LOG.write_text(""); rep={"ok":False}
    try:
        IN2=find_input(); log(f"IN={IN2}")
        work=materialize(IN2, OUT/"in")
        v5=work/"v5_frames"
        if not (v5/"cam_head_color.npy").exists():
            c=list(work.glob("**/cam_head_color.npy")); v5=c[0].parent if c else v5
        import numpy as np
        frames={}
        if (v5/"frames.json").exists():
            frames=json.load(open(v5/"frames.json"))
        shapes={}
        for name in ("cam_head_color","cam_head_depth","cam_torso_color","cam_torso_depth"):
            p=v5/f"{name}.npy"
            if p.exists():
                a=np.load(p); shapes[name]={"shape":list(a.shape),"dtype":str(a.dtype),"min":float(np.nanmin(a)),"max":float(np.nanmax(a)),"mean":float(np.nanmean(a))}
        logs=v5/"logs"
        texts={}
        if logs.exists():
            for p in sorted(logs.glob("*.txt")):
                texts[p.name]=p.read_text()[:4000]
        # naive parse of CameraInfo-like numbers
        parsed={}
        for name,t in texts.items():
            nums=[float(x) for x in re.findall(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?", t)]
            parsed[name]={"n_numbers":len(nums),"head_numbers":nums[:32], "text_head":t[:500]}
        rep.update({"shapes":shapes,"frames_meta_type":type(frames).__name__,
                    "frames_keys":list(frames.keys())[:30] if isinstance(frames,dict) else None,
                    "log_files":list(texts.keys()),"parsed":parsed,"ok":True})
        log("ALL OK")
    except Exception as e:
        rep["error"]=repr(e); rep["traceback"]=traceback.format_exc(); log("FAIL "+repr(e))
    (OUT/"calib_results.json").write_text(json.dumps(rep, indent=2)); return 0 if rep.get("ok") else 1
if __name__=="__main__": raise SystemExit(main())
