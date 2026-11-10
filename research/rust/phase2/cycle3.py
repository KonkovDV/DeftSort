"""Phase-2 end-to-end CPU workload (no ROS) for the COMPLIANCE code (scene measured by camera), mirrors node.capture /
 measure_plane / locate_trays (trays.detect) / tray_mask / observe / identify / pick_place planning.
 perception: v7 head frames -> deproject -> table_region + table_plane -> trays.detect x3 -> tray_mask -> segment -> identify
 planning  : synthetic objects -> candidates (with 'others' clearance) -> plan_pick (SIDE_DZ) -> plan_place_yaw
Deterministic. `dump` writes a JSON fingerprint for equivalence across accel on/off and branches."""
import sys,glob,time,json,os,numpy as np
ROOT=os.environ.get('T48_ROOT','/workspace/solution/nssr-sol-team-048/src/team048_solution')
sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+'/test');import test_node_stub  # ROS stubs
import scene_fixture as SF   # synthetic tray poses for the planning part only
from team048_solution import arm as ARM, node as NODE, trays as TR
from team048_solution.grasp import candidates
from team048_solution.perception import table_plane,segment
PRI=NODE.PRI;n=np.array([-.0699,0,.9976]);d=.1788;PLANE=(n/np.linalg.norm(n),d/np.linalg.norm(n))
zt=lambda x,y:-(PLANE[1]+PLANE[0][0]*x+PLANE[0][1]*y)/PLANE[0][2]
def cloud(name,x,y,yaw,rng):
    p=PRI[name];h=p['height'];a,b=p['w_min'],p['w_max'];N=3000
    if p['category'] in('box','electronics'):
        u=rng.uniform(-.5,.5,(N,3))*[b,a,0];face=rng.integers(0,5,N);u[:,2]=rng.uniform(0,h,N)
        u[face==0,2]=h;u[face==1,0]=b/2;u[face==2,0]=-b/2;u[face==3,1]=a/2;u[face==4,1]=-a/2
    else:
        t=rng.uniform(0,2*np.pi,N);z=rng.uniform(0,h,N);u=np.c_[b/2*np.cos(t),a/2*np.sin(t),z]
        top=rng.random(N)<.3;rr=np.sqrt(rng.random(N));u[top]=np.c_[b/2*rr*np.cos(t),a/2*rr*np.sin(t),np.full(N,h)][top]
    c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1]]);P=u@R.T;P[:,0]+=x;P[:,1]+=y;P[:,2]+=zt(x,y);return P
NAMES=[n for n in ['Threshold_Ramekin_White_Porcelain','Crayola_Bonus_64_Crayons','006_mustard_bottle','Ecoforms_Plant_Container_B4_Har','Saccharomyces_Boulardii_MOS_Value_Size'] if n in PRI]
TRAY={'dish':'tray_left','box':'tray_right','jar':'tray_center','electronics':'tray_center'}
_arms=None
def arms():
    global _arms
    if _arms is None:
        _arms={k:ARM.Arm(k) for k in 'lr'}
        for A in _arms.values():A.set_scene(PLANE,list(SF.TRAYS.values()))
    return _arms
def plan_object(nm,cl,others,out):
    """Same branch structure as Node.pick_place up to the first executed motion."""
    A2=arms();W=NODE.self_w(type('S',(),{'arms':A2})())
    cat=PRI[nm]['category'];tx,ty=SF.TRAYS[TRAY[cat]];plane=PLANE
    f,C=candidates(cl,plane,cat,others,W_MAX=W,robot_xy=(0.,0.))
    order='lr' if ty>.1 else ('rl' if ty<-.1 else ('lr' if f['center'][1]>=0 else 'rl'))
    dy=TR.slot_offset(0,.5*f['length'],ty=ty)
    for side in order:
        A=A2[side]
        for g in C:
            if not g['feasible']:continue
            top=-(plane[1]+plane[0][:2]@g['center'])/plane[0][2]
            if g['kind']=='side':ap=np.r_[g['approach'],0.];ca=np.r_[g['close_axis'],0.];pt='grip'
            else:
                ap=np.asarray(g.get('approach3',[0.,0.,-1.]),float);ca=np.asarray(g['close_axis'],float);ca=np.r_[ca,0.] if ca.size==2 else ca;pt='pinch'
            p=None
            for dz in (NODE.SIDE_DZ if g['kind']=='side' else (0.,)):
                if g['kind']=='side' and dz>0 and g['z']+dz>f['height']-.015:break
                p=A.plan_pick(np.array([g['center'][0],g['center'][1],top+g['z']+dz]),ca,ap,g['width'],point=pt)
                if p is not None:break
            if p is None:continue
            yp=NODE.plan_place_yaw(A,p['q_lift'],tx,ty+dy,plane,list(SF.TRAYS.values()),fallback=(tx,ty) if dy else None)
            if yp is None:continue
            out.append([nm,side,g['kind'],round(float(yp[0]),9),np.round(p['q_grasp'],9).tolist(),np.round(yp[1][0],9).tolist()]);return
    out.append([nm,None])
def planning(n_obj=10,seed=1):
    rng=np.random.default_rng(seed);out=[];scene=[]
    for i in range(n_obj):
        nm=NAMES[i%len(NAMES)];x=(.30,.40,.48)[i%3];y=(-.4,-.15,.15,.4)[i%4];scene.append((nm,cloud(nm,x,y,rng.uniform(0,np.pi),rng)))
    for i,(nm,cl) in enumerate(scene):
        plan_object(nm,cl,[c for j,(_,c) in enumerate(scene) if j!=i and j<i+3 and j>i-3],out)
    return out
FRAMES=sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/*/*closed*.npz'))
Kc=np.array([[343.16,0,320.],[0,343.16,180.],[0,0,1]])
def deproject(dmm,T,K=Kc):   # node.observe (now perception.deproject)
    try:from team048_solution.perception import deproject as D
    except ImportError:return deproject_ref(dmm,T,K)
    return D(dmm,K,T)
def deproject_ref(dmm,T,K=Kc):
    h,w=dmm.shape;v,u=np.mgrid[0:h,0:w];Z=dmm/1000.
    X=(u-K[0,2])*Z/K[0,0];Y=(v-K[1,2])*Z/K[1,1]
    return (np.stack([X,Y,Z],-1).reshape(-1,3)@T[:3,:3].T+T[:3,3]).reshape(h,w,3)
class _S:   # minimal Node stand-in for identify/locate_trays/tray_mask
    def __init__(s):s.task={'trays':{'tray_left':'red','tray_center':'green','tray_right':'blue'}};s.trays={}
    def log(s,*a):pass
    tray_names=NODE.Sol.tray_names
def perception(nf=6,timers=None):
    try:from team048_solution.perception import deproject as D,table_region as TRG
    except ImportError:   # compliance branch before the merge: its inline node.capture / observe arithmetic
        D=lambda dmm,K,T:deproject_ref(dmm,T,K);TRG=lambda P,valid:valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
    out=[];tm=timers if timers is not None else {}
    def tick(k,t0):tm[k]=tm.get(k,0.)+time.perf_counter()-t0;return time.perf_counter()
    for fn in FRAMES[:nf]:
        S=_S()
        z=np.load(fn);dd=z['depth'];dd=dd[...,0] if dd.ndim==3 else dd;dmm=dd.astype(float);rgb=z['rgb']
        T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])})
        t=time.perf_counter();P=D(dmm,Kc,T);valid=dmm>0;t=tick('deproject',t)
        tb0=TRG(P,valid);t=tick('masks',t)
        plane=table_plane(P,tb0);ok=ARM.plane_plausible(plane);t=tick('table_plane',t)
        found=NODE.Sol.locate_trays(S,P,rgb,valid,plane);t=tick('trays.detect x3',t)
        v2=valid&~NODE.Sol.tray_mask(S,P);tb=TRG(P,v2);t=tick('masks',t)
        dets=segment(P,rgb,tb,plane,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40);t=tick('segment',t)
        ids=NODE.Sol.identify(S,dets,P,plane,list(PRI));t=tick('identify',t)
        tr={k:(None if v is None else {a:(list(map(float,b)) if isinstance(b,tuple) else b) for a,b in v.items()}) for k,v in sorted(found.items())}
        out.append([np.round(plane[0],12).tolist(),round(float(plane[1]),12),ok,tr,
                    [np.round(d.features,12).tolist()+[int(len(d.pixels))] for d in dets],[[x[1],round(x[3],9)] for x in ids]])
    return out
if __name__=='__main__':
    what=sys.argv[1] if len(sys.argv)>1 else 'all'
    t=time.perf_counter();arms();print('Arm init x2 %.3f s'%(time.perf_counter()-t))
    if what=='dump':
        json.dump(dict(perception=perception(),planning=planning()),open(sys.argv[2],'w'));sys.exit()
    tm={}
    t=time.perf_counter();r=perception(timers=tm);tp=time.perf_counter()-t
    print('perception %d frames %.3f s (%.1f ms/frame)'%(len(r),tp,1e3*tp/len(r)),{k:'%.1f ms/fr'%(1e3*v/len(r)) for k,v in tm.items()})
    t=time.perf_counter();r=planning();print('planning 10 objects %.3f s'%(time.perf_counter()-t),[(x[0][:8],x[1]) for x in r])
