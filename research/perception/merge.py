import numpy as np,os
OVR=float(os.environ.get('OVR','.3'));DH=float(os.environ.get('DH','.015'))
from team048_manipulation.perception import Detection
def merge_fragments(dets,P,rgb,plane,pad=.03,low=.025):
    """Merge segments of one object split by depth jumps: XY footprints overlap (+pad) and
    similar median colour, or one fragment is small. Sensor data only."""
    n,d=plane;dets=list(dets);changed=True
    def col(x):return x.features[3:6]
    while changed:
        changed=False
        for i in range(len(dets)):
            for j in range(i+1,len(dets)):
                a,b=dets[i],dets[j]
                ov=np.all(a.lower[:2]-pad<=b.upper[:2]) and np.all(b.lower[:2]-pad<=a.upper[:2])
                if not ov:continue
                # only flat appendages (cables, rims) join a taller neighbour; never two full objects
                ha,hb=a.features[2],b.features[2]
                il=np.maximum(a.lower[:2],b.lower[:2]);ih=np.minimum(a.upper[:2],b.upper[:2]);inter=np.prod(np.clip(ih-il,0,None))
                amin=min(np.prod(a.upper[:2]-a.lower[:2]),np.prod(b.upper[:2]-b.lower[:2]))
                flat=min(ha,hb)<low and max(ha,hb)>=low
                if not (flat or (inter>OVR*amin and abs(ha-hb)<DH)):continue
                pix=np.vstack([a.pixels,b.pixels]);cloud=P[pix[:,0],pix[:,1]]
                lo,hi=cloud.min(0),cloud.max(0);ext=hi-lo;c=np.median(rgb[pix[:,0],pix[:,1]],axis=0)/255.
                f=np.r_[np.sort(ext[:2]),np.quantile(cloud@np.asarray(n)+d,.98),c]
                dets[i]=Detection(np.median(cloud,axis=0),lo,hi,pix,f);dets.pop(j);changed=True;break
            if changed:break
    return dets
