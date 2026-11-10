"""CPU shard A: RandomForest recalibration/eval on render features. No accelerator."""
import json, os, sys, time, collections, zipfile, traceback
from pathlib import Path

IN = Path('/kaggle/input/team048-tpu-cpu-inputs')
OUT = Path('/kaggle/working'); LOG = OUT/'rf.log'


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



def log(m):
    line=f"{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())} {m}"; print(line,flush=True); LOG.open('a').write(line+'\n')

def ensure():
    global IN
    IN = find_input(); log(f'IN={IN} listing={list(IN.iterdir())[:20]}')
    work = materialize(IN, OUT/'in')
    perc = work/'perception'
    if not (perc/'features.json').exists():
        cands=list(work.glob('**/features.json'))
        perc=cands[0].parent if cands else perc
    return perc

def main():
    LOG.write_text(''); rep={'ok':False}
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        perc=ensure(); log(f'perc={perc}')
        R=[]
        for n in ('features.json','features_hard.json'):
            p=perc/n
            if p.exists(): R.extend(json.load(open(p))); log(f'loaded {n} total={len(R)}')
        det=[r for r in R if 'f6' in r and r.get('category')]
        rep['n_rows']=len(R); rep['n_det']=len(det); rep['pools']=dict(collections.Counter(r.get('pool') for r in det))
        metrics={}
        for pool in (3,6):
            P=[r for r in det if r.get('pool')==pool]
            lab=lambda r: r['category'] if r.get('purity',0)>=0.9 else 'MERGED'
            seeds=sorted({r['seed'] for r in P}); tr=set(seeds[:len(seeds)*6//10])
            A=[r for r in P if r['seed'] in tr]; B=[r for r in P if r['seed'] not in tr]
            X=lambda S: np.array([r['f6']+r['hist']+[np.log(max(r.get('npix',1),1))] for r in S])
            y=lambda S: np.array([lab(r) for r in S])
            t0=time.time()
            if len(A)<5 or len(B)<5:
                raise RuntimeError(f'pool{pool} too few samples train={len(A)} test={len(B)} perc_files='+str(list(perc.glob("*"))))
            clf=RandomForestClassifier(400,random_state=0,min_samples_leaf=2,n_jobs=-1,
                class_weight={**{c:1 for c in set(y(A))},'MERGED':3}).fit(X(A),y(A))
            fit=time.time()-t0
            pr=clf.predict_proba(X(B)); cls=clf.classes_
            ths={}
            for th in (0.5,0.7,0.8,0.9):
                acc=pr.max(1)>=th; pred=cls[pr.argmax(1)]; yt=y(B)
                keep=acc&(pred!='MERGED'); score=keep&(yt!='MERGED')
                ths[str(th)]={'coverage':round(float(keep.mean()),4),
                              'selective_error':round(float((pred[score]!=yt[score]).mean()) if score.any() else 0.0,4),
                              'n_kept':int(score.sum())}
            path=OUT/f'forest_pool{pool}_cpu.joblib'; joblib.dump(clf,path)
            metrics[f'pool{pool}']={'fit_s':round(fit,3),'n_train':len(A),'n_test':len(B),'thresholds':ths,'model':path.name}
            log(f'pool{pool} {metrics[f"pool{pool}"]}')
        # shipped models @0.8
        shipped={}
        for pool in (3,6):
            p=perc/f'forest_pool{pool}.joblib'
            if not p.exists(): continue
            m=joblib.load(p); P=[r for r in det if r.get('pool')==pool]
            seeds=sorted({r['seed'] for r in P}); te=set(seeds[len(seeds)*6//10:])
            B=[r for r in P if r['seed'] in te]
            X=np.array([r['f6']+r['hist']+[np.log(max(r.get('npix',1),1))] for r in B])
            yt=np.array([r['category'] if r.get('purity',0)>=0.9 else 'MERGED' for r in B])
            pr=m.predict_proba(X); cls=m.classes_; acc=pr.max(1)>=0.8; pred=cls[pr.argmax(1)]
            keep=acc&(pred!='MERGED'); score=keep&(yt!='MERGED')
            shipped[f'pool{pool}']={'coverage':round(float(keep.mean()),4),
                'selective_error':round(float((pred[score]!=yt[score]).mean()) if score.any() else 0.0,4),'n_test':len(B)}
        rep['rf']=metrics; rep['shipped_thr0.8']=shipped; rep['ok']=True
        log(f'shipped={shipped}'); log('ALL OK')
    except Exception as e:
        rep['error']=repr(e); rep['traceback']=traceback.format_exc(); log(f'FAIL {e}'); log(rep['traceback'])
    (OUT/'rf_results.json').write_text(json.dumps(rep,indent=2))
    return 0 if rep.get('ok') else 1
if __name__=='__main__': raise SystemExit(main())
