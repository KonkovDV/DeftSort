"""Build v4 RangeClassifier profiles (per-object feature envelopes, labelled by category) from training seeds,
evaluate on held-out seeds with the unmodified v4 classifier."""
import json,sys,numpy as np,collections
sys.path.insert(0,'/workspace/project/workspace/team048-standalone/src/team048_manipulation')
from team048_manipulation.perception import RangeClassifier,Detection
R=json.load(open(sys.argv[1] if len(sys.argv)>1 else 'features.json'));det=[r for r in R if 'f6' in r and r['category']]
MARGIN=np.array([.008,.008,.005,.04,.04,.04])
out={}
for pool,allowed in ((3,('dish','box','jar')),(6,('electronics','box'))):
    P=[r for r in det if r['pool']==pool];seeds=sorted({r['seed'] for r in P});tr=set(seeds[:len(seeds)*6//10])
    A=[r for r in P if r['seed'] in tr];B=[r for r in P if r['seed'] not in tr]
    by=collections.defaultdict(list)
    for r in A:by[(r['name'],r['category'])].append(r['f6'])
    profiles=[]
    for (n,c),F in by.items():
        F=np.array(F);lo=np.percentile(F,1,axis=0)-MARGIN;hi=np.percentile(F,99,axis=0)+MARGIN
        profiles.append({'category':c,'object':n,'min':lo.tolist(),'max':hi.tolist(),'validated_on_sensors':True,
                         'evidence':'Isaac render of organiser USDs, approx camera; NOT yet the real cam_head/cam_torso'})
    clf=RangeClassifier(profiles,allowed=allowed)
    ok=wrong=abst=0;conf=collections.Counter()
    for r in B:
        d=Detection(np.zeros(3),np.zeros(3),np.ones(3),np.zeros((1,2)),np.array(r['f6']));clf.classify(d)
        if d.category is None:abst+=1
        elif d.category==r['category']:ok+=1
        else:wrong+=1;conf[(r['category'],d.category)]+=1
    n=len(B);print(f'pool{pool}: v4-envelopes coverage {(ok+wrong)/n:.3f} selective_error {wrong/max(ok+wrong,1):.3f} abstain {abst/n:.3f} n={n}',dict(conf))
    out[str(pool)]=profiles
json.dump(out,open('v4_profiles.json','w'),indent=1)
