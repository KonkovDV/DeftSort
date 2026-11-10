import json,numpy as np,collections
from sklearn.ensemble import RandomForestClassifier
E=json.load(open('features.json'));H=json.load(open('features_hard.json'))
seg=[r for r in H if 'n_det' in r];det=[r for r in H if 'f6' in r]
print('hard frames',len(seg),'visible/frame',np.mean([r['n_gt_visible'] for r in seg]),'dets/frame',np.mean([r['n_det'] for r in seg]),'plane_err',sum('plane_error' in r for r in H))
imp=[r for r in det if r['purity']<.9];unl=[r for r in det if not r['category']]
print('dets',len(det),'merged(impure<0.9)',len(imp),'unlabelled',len(unl))
lab=[r for r in det if r['category']];cnt=collections.Counter((r['file'],r['name']) for r in lab);print('fragments',sum(v-1 for v in cnt.values()))
miss=sum(r['n_gt_visible'] for r in seg)-len(cnt);print('visible objects without any segment',miss)
X=lambda S:np.array([r['f6']+r['hist'] for r in S]);y=lambda S:np.array([r['category'] for r in S])
def rep(name,A,B):
    clf=RandomForestClassifier(300,random_state=0,min_samples_leaf=2).fit(X(A),y(A));pr=clf.predict_proba(X(B));c=clf.classes_;pred=c[pr.argmax(1)];yt=y(B)
    for th in (.7,.8,.9):
        a=pr.max(1)>=th;print(f' {name} thr{th}: coverage {a.mean():.3f} sel_err {(pred[a]!=yt[a]).mean() if a.any() else 0:.3f} n={len(B)}')
    print('  confusions@0.8',dict(collections.Counter((p,q) for p,q,k in zip(yt,pred,pr.max(1)>=.8) if k and p!=q)))
for pool in (3,6):
    Ee=[r for r in E if 'f6' in r and r['category'] and r['pool']==pool];Hh=[r for r in lab if r['pool']==pool and r['purity']>=.9]
    rep(f'pool{pool} easy->hard',Ee,Hh)
    s=sorted({r['seed'] for r in Hh});tr=set(s[:len(s)*6//10])
    rep(f'pool{pool} easy+hard-train->hard-test',Ee+[r for r in Hh if r['seed'] in tr],[r for r in Hh if r['seed'] not in tr])
    imp_p=[r for r in imp if r['pool']==pool and r['category']]
    if imp_p:
        clf=RandomForestClassifier(300,random_state=0,min_samples_leaf=2).fit(X(Ee+Hh),y(Ee+Hh));pr=clf.predict_proba(X(imp_p))
        print(f'  pool{pool} merged segments: {len(imp_p)}, accepted@0.8 {(pr.max(1)>=.8).sum()} (each accepted = pick risk)')
