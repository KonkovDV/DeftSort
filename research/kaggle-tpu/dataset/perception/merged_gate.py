import json,numpy as np,collections
from sklearn.ensemble import RandomForestClassifier
E=json.load(open('features.json'));H=json.load(open('features_hard.json'))
X=lambda S:np.array([r['f6']+r['hist']+[np.log(r['npix'])] for r in S])
lab=lambda r:r['category'] if r['purity']>=.9 else 'MERGED'
for pool in (3,6):
    A=[r for r in E+H if 'f6' in r and r['category'] and r['pool']==pool]
    s=sorted({r['seed'] for r in A});tr=set(s[:len(s)*6//10])
    Tr=[r for r in A if r['seed'] in tr];Te=[r for r in A if r['seed'] not in tr]
    clf=RandomForestClassifier(400,random_state=0,min_samples_leaf=2,class_weight={**{c:1 for c in set(map(lab,Tr))},'MERGED':3}).fit(X(Tr),[lab(r) for r in Tr])
    pr=clf.predict_proba(X(Te));c=clf.classes_;pred=c[pr.argmax(1)];yt=np.array([lab(r) for r in Te])
    for th in (.7,.8):
        acc=(pr.max(1)>=th)&(pred!='MERGED')
        single=yt!='MERGED'
        wrong=acc&(pred!=yt)
        print(f'pool{pool} thr{th}: singles accepted {acc[single].mean():.3f}, merged accepted {acc[~single].sum()}/{(~single).sum()}, wrong class among accepted {wrong[single].sum()}/{acc.sum()}')
