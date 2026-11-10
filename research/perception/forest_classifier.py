"""Abstaining RandomForest over v4 features + 4x4x4 RGB histogram of the segment pixels. Sensor data only.
Drop-in alongside v4 RangeClassifier: classify(detection, rgb) sets detection.category/reason."""
import json,numpy as np,joblib
from sklearn.ensemble import RandomForestClassifier
def hist(rgb,pixels):
    px=rgb[pixels[:,0],pixels[:,1]].astype(float)/255;h=np.histogramdd(px,bins=(4,4,4),range=((0,1),)*3)[0].ravel();return h/max(h.sum(),1)
class ForestClassifier:
    def __init__(self,model,threshold=.8,allowed=None):
        self.m=model;self.th=threshold;self.allowed=allowed
    @classmethod
    def train(cls,X,y,**kw):return cls(RandomForestClassifier(400,random_state=0,min_samples_leaf=2,class_weight={**{c:1 for c in set(y)},'MERGED':3}).fit(X,y),**kw)
    def classify(self,detection,rgb):
        x=np.r_[detection.features,hist(rgb,detection.pixels),np.log(len(detection.pixels))][None];p=self.m.predict_proba(x)[0];k=int(p.argmax());c=self.m.classes_[k]
        ok=p[k]>=self.th and c!='MERGED' and (self.allowed is None or c in self.allowed)
        detection.category=c if ok else None
        detection.reason=f'forest p={p[k]:.2f}'+('' if ok else ' < threshold/not allowed -> abstain')
        return detection
if __name__=='__main__':
    import sys;R=[r for f in sys.argv[1:] for r in json.load(open(f)) if 'f6' in r and r['category']]
    for pool in (3,6):
        P=[r for r in R if r['pool']==pool];X=np.array([r['f6']+r['hist']+[np.log(r['npix'])] for r in P]);y=np.array([r['category'] if r['purity']>=.9 else 'MERGED' for r in P])
        joblib.dump(ForestClassifier.train(X,y).m,f'forest_pool{pool}.joblib');print('saved pool',pool,len(P))
