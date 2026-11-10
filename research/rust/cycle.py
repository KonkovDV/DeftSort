"""Realistic CPU workload for profiling/benchmarking (no ROS):
 perception: v7 Kaggle head frames -> point cloud -> table_plane -> tray locate -> segment -> identify features
 planning  : reach3-style synthetic objects -> candidates -> plan_pick (SIDE_DZ escalation) -> plan_place_yaw
Deterministic (fixed seeds)."""
import sys,glob,time,numpy as np
import os;ROOT=os.environ.get('T48_ROOT','/workspace/solution/nssr-sol-team-048/src/team048_solution')
sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+'/test');import test_node_stub  # installs ROS stubs
from team048_solution import arm as ARM, node as NODE, trays as TR
from team048_solution.grasp import candidates
from team048_solution.perception import table_plane,segment
PRI=NODE.PRI;n=np.array([-.0699,0,.9976]);d=.1788;PLANE=(n/np.linalg.norm(n),d/np.linalg.norm(n))
zt=lambda x,y:-(PLANE[1]+PLANE[0][0]*x+PLANE[0][1]*y)/PLANE[0][2]
def cloud(name,x,y,yaw,rng):
    p=PRI[name];h=p['height'];a,b=p['w_min'],p['w_max'];N=3000
    if p['category']=='box':
        u=rng.uniform(-.5,.5,(N,3))*[b,a,0];face=rng.integers(0,5,N);u[:,2]=rng.uniform(0,h,N)
        u[face==0,2]=h;u[face==1,0]=b/2;u[face==2,0]=-b/2;u[face==3,1]=a/2;u[face==4,1]=-a/2
    else:
        t=rng.uniform(0,2*np.pi,N);z=rng.uniform(0,h,N);u=np.c_[b/2*np.cos(t),a/2*np.sin(t),z]
        top=rng.random(N)<.3;rr=np.sqrt(rng.random(N));u[top]=np.c_[b/2*rr*np.cos(t),a/2*rr*np.sin(t),np.full(N,h)][top]
    c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1]]);P=u@R.T;P[:,0]+=x;P[:,1]+=y;P[:,2]+=zt(x,y);return P
NAMES=['Threshold_Ramekin_White_Porcelain','Crayola_Bonus_64_Crayons','006_mustard_bottle','Ecoforms_Plant_Container_B4_Har','Saccharomyces_Boulardii_MOS_Value_Size']
TRAY={'dish':'tray_left','box':'tray_right','jar':'tray_center'}
_arms=None
def arms():
    global _arms
    if _arms is None:
        _arms={k:ARM.Arm(k) for k in 'lr'}
        for A in _arms.values():A.set_scene(PLANE,list(TR.NOMINAL.values()))
    return _arms
def plan_object(nm,x,y,yaw,rng,out):
    A2=arms();W=min(A2['l'].hand['opposed_open']-.01,.11)
    cat=PRI[nm]['category'];tx,ty=TR.NOMINAL[TRAY[cat]];cl=cloud(nm,x,y,yaw,rng)
    f,C=candidates(cl,PLANE,cat,[],W_MAX=W)
    order='lr' if ty>.1 else ('rl' if ty<-.1 else ('lr' if y>=0 else 'rl'))
    for side in order:
        A=A2[side]
        for g in C:
            top=zt(*g['center']);ap=np.r_[g['approach'],0.] if g['kind']=='side' else np.array([0,0,-1.]);p=None
            for dz in (NODE.SIDE_DZ if g['kind']=='side' else (0.,)):
                if g['kind']=='side' and dz>0 and g['z']+dz>f['height']-.015:break
                p=A.plan_pick(np.array([*g['center'],top+g['z']+dz]),np.r_[g['close_axis'],0.],ap,g['width'],point='pinch' if g['kind']=='rim' else 'grip')
                if p is not None:break
            if p is None:continue
            yp=NODE.plan_place_yaw(A,p['q_lift'],tx,ty,PLANE,list(TR.NOMINAL.values()))
            if yp is None:continue
            out.append((nm,side,round(yp[0],6),np.round(p['q_grasp'],9).tolist()));return
    out.append((nm,None))
def planning(n_obj=10,seed=1):
    rng=np.random.default_rng(seed);out=[]
    for i in range(n_obj):
        nm=NAMES[i%len(NAMES)];x=(.30,.40,.48)[i%3];y=(-.4,-.15,.15,.4)[i%4];plan_object(nm,x,y,rng.uniform(0,np.pi),rng,out)
    return out
K=dict(fx=343.16,cx=320.,cy=180.)
FRAMES=sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/*/*closed*.npz'))
def frame_cloud(z):
    dd=z['depth'].astype(float);dd=dd[...,0] if dd.ndim==3 else dd
    T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])})
    v,u=np.mgrid[0:dd.shape[0],0:dd.shape[1]];Z=dd/1000.;X=(u-K['cx'])*Z/K['fx'];Y=(v-K['cy'])*Z/K['fx']
    P=(np.stack([X,Y,Z],-1).reshape(-1,3)@T[:3,:3].T+T[:3,3]).reshape(dd.shape+(3,));return P,dd>0
def perception(nf=6):
    out=[]
    for fn in FRAMES[:nf]:
        z=np.load(fn);P,valid=frame_cloud(z);rgb=z['rgb']
        tb0=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
        plane=table_plane(P,tb0)
        for name,col in (('tray_left','red'),('tray_center','green'),('tray_right','blue')):TR.locate(P,rgb,valid,name,col,plane)
        tb=tb0.copy()
        for (tx,ty) in TR.NOMINAL.values():tb&=~((np.abs(P[...,0]-tx)<.13)&(np.abs(P[...,1]-ty)<.18))
        dets=segment(P,rgb,tb,plane,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
        out.append((plane[0].round(9).tolist(),len(dets),[d.features.round(9).tolist() for d in dets]))
    return out
if __name__=='__main__':
    what=sys.argv[1] if len(sys.argv)>1 else 'all'
    t=time.perf_counter();arms();print('Arm init x2 %.3f s'%(time.perf_counter()-t))
    if what in('all','perc'):
        t=time.perf_counter();r=perception();print('perception %d frames %.3f s'%(len(r),time.perf_counter()-t),[x[1] for x in r])
    if what in('all','plan'):
        t=time.perf_counter();r=planning();print('planning 10 objects %.3f s'%(time.perf_counter()-t),[(x[0][:10],x[1]) for x in r])
def dump(path):
    import json
    json.dump(dict(perception=perception(),planning=planning()),open(path,'w'))
