import json,numpy as np,collections,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from scipy.stats import beta
P='/workspace/fixes/perception/'
E=[dict(r,set='easy') for r in json.load(open(P+'features.json')) if 'f6' in r and r['category']]
H=[dict(r,set='hard') for r in json.load(open(P+'features_hard.json')) if 'f6' in r and r['category']]
ALL=E+H
def cp(k,n,a=.05):
    lo=0 if k==0 else beta.ppf(a/2,k,n-k+1);hi=1 if k==n else beta.ppf(1-a/2,k+1,n-k);return lo,hi
FEAT={'f6':lambda r:r['f6'],'hist':lambda r:r['hist'],'f6+hist':lambda r:r['f6']+r['hist'],'f6+hist+npix':lambda r:r['f6']+r['hist']+[np.log(r['npix'])]}
lab=lambda r,m:(r['category'] if r['purity']>=.9 else 'MERGED') if m else r['category']
def oof(rows,feat,merged,k=5):
    X=np.array([FEAT[feat](r) for r in rows]);y=np.array([lab(r,merged) for r in rows]);g=np.array([r['seed'] for r in rows])
    pr=np.zeros(len(rows));pred=np.empty(len(rows),object)
    for tr,te in GroupKFold(k).split(X,y,g):
        cw={c:1 for c in set(y[tr])}
        if 'MERGED' in cw:cw['MERGED']=3
        m=RandomForestClassifier(300,random_state=0,min_samples_leaf=2,class_weight=cw,n_jobs=-1).fit(X[tr],y[tr]);p=m.predict_proba(X[te])
        pr[te]=p.max(1);pred[te]=m.classes_[p.argmax(1)]
    return pr,pred,y,g
def summarize(rows,pr,pred,y,g,th):
    single=np.array([r['purity']>=.9 for r in rows]);truecat=np.array([r['category'] for r in rows])
    acc=(pr>=th)&(pred!='MERGED')
    wrong=acc&((~single)|(pred!=truecat))   # merged accepted counts as error (pick risk)
    k,n=int(wrong.sum()),int(acc.sum());lo,hi=cp(k,n)
    # layout bootstrap CI for coverage of singles
    seeds=np.unique(g);rng=np.random.default_rng(0);cov=[]
    for _ in range(1000):
        s=rng.choice(seeds,len(seeds));idx=np.concatenate([np.where(g==q)[0] for q in s]);m=single[idx]
        cov.append(acc[idx][m].mean())
    return dict(th=th,coverage_single=float(acc[single].mean()),cov_ci=[float(np.percentile(cov,2.5)),float(np.percentile(cov,97.5))],
                accepted=n,errors=k,risk=k/max(n,1),risk_ci95=[float(lo),float(hi)],merged_accepted=int((acc&~single).sum()),merged_total=int((~single).sum()))
def ece(conf,correct,bins=15):
    e=0;edges=np.linspace(0,1,bins+1);rel=[]
    for a,b in zip(edges[:-1],edges[1:]):
        m=(conf>a)&(conf<=b)
        if m.any():e+=m.mean()*abs(correct[m].mean()-conf[m].mean());rel.append((conf[m].mean(),correct[m].mean(),m.sum()))
    return e,rel
OUT={}
for pool in (3,6):
    rows=[r for r in ALL if r['pool']==pool];OUT[pool]={}
    for feat in FEAT:
        for merged in (False,True):
            pr,pred,y,g=oof(rows,feat,merged)
            OUT[pool][f'{feat}|merged={merged}']=[summarize(rows,pr,pred,y,g,t) for t in (.6,.7,.8,.9)]
            if feat=='f6+hist+npix' and merged:
                single=np.array([r['purity']>=.9 for r in rows]);truecat=np.array([r['category'] for r in rows])
                correct=((pred==y)).astype(float);e,rel=ece(pr,correct);OUT[pool]['ece_main']=e
                fig,ax=plt.subplots(1,2,figsize=(9,4))
                c=np.array([x[0] for x in rel]);a=np.array([x[1] for x in rel])
                ax[0].plot([0,1],[0,1],'k--');ax[0].plot(c,a,'o-');ax[0].set_xlabel('уверенность');ax[0].set_ylabel('доля верных');ax[0].set_title(f'Надёжность, пул {pool}, ECE={e:.3f}')
                ths=np.linspace(.34,.99,60);cv=[];rk=[]
                for t in ths:
                    acc=(pr>=t)&(pred!='MERGED');w=acc&((~single)|(pred!=truecat));cv.append(acc[single].mean());rk.append(w.sum()/max(acc.sum(),1))
                ax[1].plot(cv,rk);ax[1].set_xlabel('coverage (одиночные)');ax[1].set_ylabel('selective risk');ax[1].set_title('Риск–покрытие')
                fig.tight_layout();fig.savefig(f'fig/calib_pool{pool}.png',dpi=110);plt.close(fig)
                OUT[pool]['rc_curve']=list(zip(map(float,ths),map(float,cv),map(float,rk)))
    # domain shift easy->hard (main config)
    tr=[r for r in rows if r['set']=='easy'];te=[r for r in rows if r['set']=='hard']
    X=lambda S:np.array([FEAT['f6+hist+npix'](r) for r in S]);yy=lambda S:np.array([lab(r,True) for r in S])
    m=RandomForestClassifier(300,random_state=0,min_samples_leaf=2,class_weight={**{c:1 for c in set(yy(tr))},'MERGED':3}).fit(X(tr),yy(tr));p=m.predict_proba(X(te))
    OUT[pool]['shift_easy_to_hard']=[summarize(te,p.max(1),m.classes_[p.argmax(1)],yy(te),np.array([r['seed'] for r in te]),t) for t in (.7,.8,.9)]
json.dump(OUT,open('stats.json','w'),indent=1)
for pool in (3,6):
    print('POOL',pool,'ECE',round(OUT[pool]['ece_main'],4))
    for k,v in OUT[pool].items():
        if '|' in k:
            s=[x for x in v if x['th']==.8][0];print(f" {k:28s} cov {s['coverage_single']:.3f} [{s['cov_ci'][0]:.3f},{s['cov_ci'][1]:.3f}] err {s['errors']}/{s['accepted']} risk95hi {s['risk_ci95'][1]:.4f} merged_acc {s['merged_accepted']}/{s['merged_total']}")
    for s in OUT[pool]['shift_easy_to_hard']:print(f" shift th{s['th']} cov {s['coverage_single']:.3f} err {s['errors']}/{s['accepted']} risk95hi {s['risk_ci95'][1]:.4f} merged_acc {s['merged_accepted']}/{s['merged_total']}")
