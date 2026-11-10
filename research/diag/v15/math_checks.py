"""v15 math review numbers (MATH_REVIEW.md), all from recorded data:
 1. URDF FK camera_head_link vs the simulator /tf (tf2_echo pelvis camera_head_link, v5 logs);
 2. RANSAC: inlier fraction w of the table region on every recorded head frame (cmp_plans frame set), the
    probability that 120 iterations miss the plane, (1 - w^3)^120, and the share of candidates that pass n_z >= 0.96;
 3. min-area-rect / caliper angular quantisation bounds for the measured tray and object sizes.
Usage: python math_checks.py  (from the repo root; NumPy reference path, TEAM048_ACCEL=0 is fine)"""
import sys,glob,json,math,numpy as np
sys.path.insert(0,'src/team048_solution');sys.path.insert(0,'src/team048_solution/test')
import test_node_stub
from team048_solution import arm as ARM, perception as PE, node as N
out={}
# 1. TF chain
L=np.array([[0.000,-0.502,0.865,0.105],[-1.000,-0.000,0.000,0.011],[0.000,-0.865,-0.502,0.648]])  # tf2_echo t=62.17
out['tf_log_vs_quat_to_T']=float(np.abs(N.quat_to_T([0.105,0.011,0.648],[0.613,-0.613,0.353,-0.353])[:3]-L).max())
out['tf_log_vs_urdf_fk_pitch0']=float(np.abs(ARM.fk_link('camera_head_link',{'head_pitch_joint':0.})[:3]-L).max())
# 2. RANSAC on the recorded head frames
K=np.load('src/team048_solution/test/fixtures/v7_s102_start_frames.npz')['K'].reshape(3,3)
R='research/kaggle-sim'
frames=sorted(glob.glob(f'{R}/out_v7/trace/*/*.npz')+glob.glob(f'{R}/out_v8/trace/*/*.npz')+glob.glob(f'{R}/out_v9b_f525ed9/trace/*/*.npz')+glob.glob(f'{R}/out_v11b_497471e/trace/*/*.npz'))
W=[];acc=[]
for fn in frames:
    z=np.load(fn)
    T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});dep=z['depth'].astype(float)
    P=PE.deproject(dep,K,T);reg=PE.table_region(P,dep>0)
    try:n,d=PE.table_plane(P,reg)
    except ValueError:W.append(None);continue
    p=P[reg];p=p[np.isfinite(p).all(1)]
    W.append(float((np.abs(p@n+d)<.006).mean()))
    rng=np.random.default_rng(0);ok=0
    for _ in range(120):
        a,b,c=p[rng.choice(len(p),3,replace=False)];nn=np.cross(b-a,c-a);l=np.linalg.norm(nn)
        if l<1e-9:continue
        nn/=l;nn*=np.sign(nn[2]) or 1;ok+=nn[2]>=.96
    acc.append(ok/120)
w=[x for x in W if x is not None]
out['frames']=len(frames);out['plane_failures']=W.count(None)
out['inlier_fraction_min_median_max']=[min(w),float(np.median(w)),max(w)]
out['p_miss_120_at_min_w']=(1-min(w)**3)**120
out['iterations_for_p_miss_1e-6_at_min_w']=math.ceil(math.log(1e-6)/math.log(1-min(w)**3))
out['nz_pass_fraction_min_median']=[min(acc),float(np.median(acc))]
# 3. angular quantisation: 91 directions over [0, pi/2] (1 deg step) / 180 over [0, pi) (1 deg step): worst offset 0.5 deg
d=math.radians(.5)
out['rect_width_err_bound_m(L=0.212)']=.212*math.sin(d)
out['caliper_width_err_bound_m(L=0.20)']=.20*math.sin(d)
print(json.dumps(out,indent=1))
# 4. impact of 200 RANSAC iterations (p_miss <= 1e-6 at the worst w) on the planes of the same frames
dn=[];dd=[];dh=[]
for fn in frames:
    z=np.load(fn);T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});dep=z['depth'].astype(float)
    P=PE.deproject(dep,K,T);reg=PE.table_region(P,dep>0)
    (n1,d1),(n2,d2)=PE.table_plane(P,reg),PE.table_plane(P,reg,iterations=200)
    dn.append(math.degrees(math.acos(min(1.,float(n1@n2)))));dh.append(abs((-(d1+n1[0]*.4)/n1[2])-(-(d2+n2[0]*.4)/n2[2])))
print(json.dumps(dict(ransac200_frames_changed=int(sum(x>0 for x in dn)),normal_deg_max=max(dn),table_z_at_x04_max_m=max(dh))))
# 5. tray axis: the q=2% trimmed extents make the area flat within a few degrees of the true angle, and the scan keeps
#    the FIRST minimum -> compare with the centre of the near-minimum plateau and with the exact hull rectangle (q=0)
from team048_solution import trays as TR
from scipy.spatial import ConvexHull
cap=[];orig=TR.min_area_rect
def grab(xy,q=.02):
    cap.append(np.asarray(xy,float).copy());return orig(xy,q)
TR.min_area_rect=grab;rows=[]
for fn in frames:
    z=np.load(fn);T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});dep=z['depth'].astype(float)
    P=PE.deproject(dep,K,T);v=dep>0;pl=PE.table_plane(P,PE.table_region(P,v))
    for name,col in (('tray_left','red'),('tray_right','blue')):
        cap.clear();d=TR.detect(P,z['rgb'],v,name,col,pl)
        if d is None or not cap:continue
        xy=cap[0];ang=np.radians(np.arange(0,90,.05))
        A=[]
        for a in ang:
            u=np.array([math.cos(a),math.sin(a)]);vv=np.array([-u[1],u[0]])
            lu,hu=np.quantile(xy@u,[.02,.98]);lv,hv=np.quantile(xy@vv,[.02,.98]);A.append((hu-lu)*(hv-lv))
        A=np.array(A);band=ang[A<=A.min()*1.001]
        mid=math.atan2(np.sin(4*band).mean(),np.cos(4*band).mean())/4%(np.pi/2)   # circular mean, period pi/2
        hull=xy[ConvexHull(xy).vertices];best=None
        for a in ang:
            u=np.array([math.cos(a),math.sin(a)]);vv=np.array([-u[1],u[0]]);ar=np.ptp(hull@u)*np.ptp(hull@vv)
            if best is None or ar<best[0]:best=(ar,a)
        ax=np.array(d['axis']);a1=math.atan2(ax[1],ax[0])%(np.pi/2)
        dd=lambda x,y:abs((math.degrees(x-y)+45)%90-45)
        rows.append(dict(f=fn.split('/')[-1][:24],tray=name,partial=d['partial'],first_min_vs_plateau_mid_deg=round(dd(a1,mid),2),
                         plateau_width_deg=round(math.degrees(np.ptp(band)),2),first_min_vs_hull_deg=round(dd(a1,best[1]),2)))
TR.min_area_rect=orig
fm=[r['first_min_vs_plateau_mid_deg'] for r in rows];fh=[r['first_min_vs_hull_deg'] for r in rows];pw=[r['plateau_width_deg'] for r in rows]
print(json.dumps(dict(tray_detections=len(rows),partial=sum(r['partial'] for r in rows),axis_first_min_vs_plateau_mid_deg_median_max=[float(np.median(fm)),max(fm)],
    plateau_width_deg_median_max=[float(np.median(pw)),max(pw)],axis_first_min_vs_hull_deg_median_max=[float(np.median(fh)),max(fh)],
    lateral_shift_at_16cm_mm_max=1000*.16*math.sin(math.radians(max(fm))))))
# 6. the same on synthetic full trays of the documented size (uniform points, as a fully visible tray floor/walls):
#    axis error of the 1-degree first-minimum scan vs the true angle
rng=np.random.default_rng(0);E=[];S=[]
for _ in range(200):
    a,b=TR.DOC_SIZE;th=rng.uniform(0,np.pi);u=np.array([math.cos(th),math.sin(th)]);vv=np.array([-u[1],u[0]])
    st=rng.uniform(-.5,.5,size=(8000,2));xy=np.outer(st[:,0]*b,u)+np.outer(st[:,1]*a,vv)
    c,size,ax=TR.min_area_rect_py(xy);E.append(math.degrees(math.acos(min(1.,abs(ax@u)))));S.append(np.abs(size-[a,b]).max())
print(json.dumps(dict(doc_size=list(TR.DOC_SIZE),synthetic_axis_err_deg_median_p95_max=[float(np.median(E)),float(np.quantile(E,.95)),max(E)],
    size_err_mm_max=1000*max(S),lateral_shift_at_half_long_side_mm_max=1000*TR.DOC_SIZE[1]/2*math.sin(math.radians(max(E))))))
