import json,numpy as np,collections
from sklearn.ensemble import RandomForestClassifier
R=json.load(open('features.json'))
seg=[r for r in R if 'n_det' in r];det=[r for r in R if 'f6' in r]
print('frames',len(seg),'mean visible',np.mean([r['n_gt_visible'] for r in seg]),'mean dets',np.mean([r['n_det'] for r in seg]))
print('plane errors',sum('plane_error' in r for r in R))
pure=[r for r in det if r['category'] and r['purity']>=.9];print('dets',len(det),'pure>=0.9',len(pure),'merged/impure',sum(r['purity']<.9 for r in det))
loc=[np.linalg.norm(np.array(r['centroid'][:2])-np.array(r['gt_xy'])) for r in pure];print('centroid xy err cm median/p95',np.median(loc)*100,np.percentile(loc,95)*100)
for pool in (3,6):
    P=[r for r in pure if r['pool']==pool];seeds=sorted({r['seed'] for r in P});tr=set(seeds[:len(seeds)*6//10])
    X=lambda S:np.array([r['f6']+r['hist'] for r in S]);y=lambda S:np.array([r['category'] for r in S])
    A=[r for r in P if r['seed'] in tr];B=[r for r in P if r['seed'] not in tr]
    if not A or not B:continue
    clf=RandomForestClassifier(300,random_state=0).fit(X(A),y(A));pr=clf.predict_proba(X(B));cls=clf.classes_
    for th in (.5,.7,.8,.9):
        acc=pr.max(1)>=th;pred=cls[pr.argmax(1)];yt=y(B)
        cov=acc.mean();err=(pred[acc]!=yt[acc]).mean() if acc.any() else 0
        print(f'pool{pool} thr{th}: coverage {cov:.3f} selective_error {err:.3f} n_test {len(B)}')
    print(' confusion@0.8',collections.Counter((a,b) for a,b,c in zip(y(B),cls[pr.argmax(1)],pr.max(1)>=.8) if c and a!=b))
