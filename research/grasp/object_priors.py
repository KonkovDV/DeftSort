"""Offline (build-time) geometry priors from organiser object USDs: 2D convex hull of footprint, height,
min/max caliper width. Used at runtime only as plausibility priors per recognised class, never as pose."""
import json,yaml,numpy as np
from pxr import Usd,UsdGeom
from scipy.spatial import ConvexHull
A='/workspace/project2/hackathon/'
pools=yaml.safe_load(open(A+'config/pools.yaml'))['pools']
def verts(n):
    st=Usd.Stage.Open(f'{A}assets/objects/{n}/{n}.usd');mpu=UsdGeom.GetStageMetersPerUnit(st);up=UsdGeom.GetStageUpAxis(st);out=[]
    for p in Usd.PrimRange(st.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.Mesh):
            v=np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(),float);T=np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)).T
            out.append((np.c_[v,np.ones(len(v))]@T.T)[:,:3])
    P=np.vstack(out)*mpu
    if up==UsdGeom.Tokens.y:P=P[:,[0,2,1]]*np.array([1,-1,1])
    return P
def calipers(h2):
    best=(9,0);worst=0
    for a in np.linspace(0,np.pi,360,endpoint=False):
        d=h2@np.array([np.cos(a),np.sin(a)]);w=d.max()-d.min()
        if w<best[0]:best=(w,a)
        worst=max(worst,w)
    return best,worst
pri={}
for pool in ('3','6'):
    for cat,names in pools[pool].items():
        for n in names:
            P=verts(n);z0=P[:,2].min();h=P[:,2].max()-z0
            hull=P[ConvexHull(P[:,:2]).vertices,:2];(wmin,amin),wmax=calipers(hull)
            # width profile at 25/50/75% height (for side grasps)
            prof={}
            for f in (.25,.5,.75):
                s=P[np.abs(P[:,2]-(z0+f*h))<.004,:2]
                if len(s)>10:prof[str(f)]=float(calipers(s[ConvexHull(s).vertices])[0][0])
            pri[n]={'pool':int(pool),'category':cat,'height':float(h),'w_min':float(wmin),'w_min_angle':float(amin),'w_max':float(wmax),
                    'hull_xy':hull.tolist(),'z_min':float(z0),'width_profile':prof}
            print(f'{pool} {cat:11s} {n[:34]:34s} h={h*100:5.1f} wmin={wmin*100:5.1f} wmax={wmax*100:5.1f} prof={ {k:round(v*100,1) for k,v in prof.items()} }')
json.dump(pri,open('object_priors.json','w'))
