"""Phase-2 Rust kernels vs their NumPy references (perception, grasp candidates, Arm.solve, scenes, certify skip).
Contract: bit-identical results, except values that pass through a NumPy BLAS dot product (the reference itself is
CPU-dependent there: OpenBLAS uses FMA on AVX2/AVX-512): <= 1e-12 absolute, with identical decisions.
Skipped when the extension (API >= 2) is not installed."""
import contextlib,glob,json,os,sys,unittest,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,HERE)
import test_node_stub  # noqa: F401  (installs the ROS stubs for node.py)
import scene_fixture as SF
from team048_solution import _accel, arm, perception, trays as TR, grasp, planning, node as NODE
V7_TRACE=os.environ.get('TEAM048_V7_TRACE','/nonexistent')   # optional directory of recorded run-v7 frames (not shipped): tests skip without it
FRAMES=sorted(glob.glob(os.path.join(V7_TRACE,'*','*.npz')))[:int(os.environ.get('T48_EQ_FRAMES',31))]
FIX=os.path.join(HERE,'fixtures','v5_head_frame.npz')
K=np.array([[343.16,0,320.],[0,343.16,180.],[0,0,1]])
# Target environment (ROS 2 Humble: Python 3.10, NumPy < 2): bit-identical. NumPy 2.x evaluates 2-vector products
# (xy@u) with a different kernel (FMA): there the reference itself moves by <= 1 ulp, so floats are compared to
# 1e-12 and every discrete result (None, flags, counts, kinds, chosen axis) must still match exactly.
EXACT=np.lib.NumpyVersion(np.__version__)<'2.0.0'
def same(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)) and len(a)==len(b) and not all(isinstance(x,(int,float,np.number)) for x in a):
        return all(same(x,y) for x,y in zip(a,b))
    if a is None or b is None or isinstance(a,(str,bool,np.bool_)) or isinstance(b,(str,bool,np.bool_)):return a==b
    A,B=np.asarray(a),np.asarray(b)
    if A.shape!=B.shape:return False
    if EXACT or A.dtype.kind not in 'fc':return np.array_equal(A,B)
    return bool(np.allclose(A,B,rtol=0,atol=1e-12))

@contextlib.contextmanager
def reference():
    """Force every phase-2 call site onto its NumPy reference (as with an API-1 wheel or a disabled extension)."""
    old=_accel.enabled2;_accel.enabled2=lambda:False
    try:yield
    finally:_accel.enabled2=old

def frames():
    out=[]
    for fn in FRAMES:
        z=np.load(fn);dd=z['depth'].astype(float);dd=dd[...,0] if dd.ndim==3 else dd
        T=arm.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});out.append((fn,dd,T,z['rgb']))
    return out

@unittest.skipUnless(_accel.API>=2,'team048_accel API>=2 not installed: '+_accel.REASON)
class Phase2(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        c.F=frames()
        c.arms={k:arm.Arm(k) for k in 'lr'}
        n=np.array([-.0699,0,.9976]);c.plane=(n/np.linalg.norm(n),.1788/np.linalg.norm(n))
        for A in c.arms.values():A.set_scene(c.plane,list(SF.TRAYS.values()))
    def setUp(s):
        if not s.F:s.skipTest('v7 frames not on this machine')
    # ---------------- perception ----------------
    def test_rng_stream_and_candidates_match_numpy(s):
        bad=0;cases=0
        for seed in range(12):
            for n in (3,5,100,999,10001,123457):
                P=np.random.default_rng(seed+100).normal(size=(n,1,3));P[...,2]=np.abs(P[...,2])+5
                r=_accel.ransac_plane(P,np.ones((n,1),bool),seed,60,.5);rng=np.random.default_rng(seed);p=P.reshape(-1,3);c=[]
                for _ in range(60):
                    a,b,cc=p[rng.choice(len(p),3,replace=False)];nr=np.cross(b-a,cc-a);L=np.linalg.norm(nr)
                    if L<1e-9:continue
                    nr/=L
                    if nr[2]<0:nr*=-1
                    if nr[2]<.96:continue
                    c.append(np.r_[nr,-nr@a])
                c=np.array(c).reshape(-1,4);cases+=1
                if c.shape!=r[1].shape or (c.size and np.abs(c-r[1]).max()>1e-13):bad+=1
        print(f'\nRANSAC stream: {cases} cases, {bad} mismatches');s.assertEqual(bad,0)
    def test_deproject_table_plane_region(s):
        dmax=0.
        for fn,dd,T,rgb in s.F:
            P=perception.deproject(dd,K,T);R=perception.deproject_py(dd,K,T);dmax=max(dmax,np.abs(P-R).max())
            valid=dd>0
            with reference():tb_ref=perception.table_region(R,valid);tr_ref=perception.table_region(R,valid,[(x,y,.13,.18) for x,y in SF.TRAYS.values()])
            s.assertTrue(np.array_equal(perception.table_region(R,valid),tb_ref))
            s.assertTrue(np.array_equal(perception.table_region(R,valid,[(x,y,.13,.18) for x,y in SF.TRAYS.values()]),tr_ref))
            for seed in (0,1,7):
                a=perception.table_plane(R,tb_ref,seed=seed);b=perception.table_plane_py(R,tb_ref,seed=seed)
                s.assertTrue(np.array_equal(a[0],b[0]) and a[1]==b[1],(fn,seed))
        print(f'\ndeproject max |diff| {dmax:.1e} over {len(s.F)} frames');s.assertLessEqual(dmax,1e-12)
    def test_tray_detect_exact(s):
        n=found=0
        for fn,dd,T,rgb in s.F:
            P=perception.deproject_py(dd,K,T);valid=dd>0;pl=perception.table_plane_py(P,perception.table_region(P,valid))
            for name in TR.SEARCH:
                for col in TR.CHANNEL:
                    a=TR.detect(P,rgb,valid,name,col,pl)
                    with reference():b=TR.detect(P,rgb,valid,name,col,pl)
                    s.assertTrue(same(a,b),(fn,name,col,a,b));n+=1;found+=a is not None
        print(f'\ntrays.detect: {n} calls identical ({found} detections)')
    def test_min_area_rect_and_rect_mask(s):
        rng=np.random.default_rng(0);bad=0
        for k in range(150):
            n=int(rng.integers(1,3000));A=rng.normal(size=(n,3))*rng.uniform(.01,.3)+rng.uniform(-1,1,3)
            for xy in (A[:,:2],np.ascontiguousarray(A[:,:2]),A[:,:2].round(3)):
                a=TR.min_area_rect(xy);b=TR.min_area_rect_py(xy);bad+=not all(same(x,y) for x,y in zip(a,b))
        s.assertEqual(bad,0)
        fn,dd,T,rgb=s.F[0];P=perception.deproject_py(dd,K,T).copy();P[::7,::5]=np.nan
        for th in np.linspace(0,3,7):
            u=np.array([np.cos(th),np.sin(th)]);v=np.array([-u[1],u[0]])
            R=[(.34,.6,u,v,.14,.19),(.5,-.1,v,u,.1,.3)]
            s.assertTrue(np.array_equal(perception.rect_mask(P,R),perception.rect_mask_py(P,[(x,y,a[0],a[1],b[0],b[1],c,d) for x,y,a,b,c,d in R])))
        class S:pass
        o=S();o.trays={'tray_left':dict(center=(.34,.69),size=(.22,.32),axis=(.1,.995),height=.068),'tray_center':(.53,0.)}
        m=NODE.Sol.tray_mask(o,P);ref=np.zeros(P.shape[:2],bool)
        for t in o.trays.values():   # the former inline expression
            x,y,ax,ay,a,b,_=arm.tray_geom(t);v=np.array([ax,ay]);u=np.array([v[1],-v[0]]);dx=P[...,0]-x;dy=P[...,1]-y
            ref|=(np.abs(dx*u[0]+dy*u[1])<a/2+NODE.TRAY_MASK_MARGIN)&(np.abs(dx*v[0]+dy*v[1])<b/2+NODE.TRAY_MASK_MARGIN)
        s.assertTrue(np.array_equal(m,ref) and m.any())
    def test_segment_statistics(s):
        nd=0;roi=(np.array([.18,-.62,-1.]),np.array([.62,.62,1.]))
        for fn,dd,T,rgb in s.F:
            P=perception.deproject_py(dd,K,T);tb=perception.table_region(P,dd>0);pl=perception.table_plane_py(P,tb)
            A=perception.segment(P,rgb,tb,pl,roi,min_pixels=40);B=perception.segment_py(P,rgb,tb,pl,roi,min_pixels=40)
            s.assertEqual(len(A),len(B))
            for x,y in zip(A,B):
                s.assertTrue(np.array_equal(x.pixels,y.pixels) and np.array_equal(x.centroid,y.centroid) and np.array_equal(x.lower,y.lower) and np.array_equal(x.upper,y.upper))
                s.assertTrue(np.array_equal(np.delete(x.features,2),np.delete(y.features,2)) and abs(x.features[2]-y.features[2])<=1e-12);nd+=1
        print(f'\nsegment: {nd} detections, pixels/centroid/box/colour exact, height q98 <= 1e-12')
    # ---------------- grasp ----------------
    def test_grasp_candidates(s):
        rng=np.random.default_rng(3);pl=s.plane;nc=0
        def cloud(cat,x,y,yaw):
            a,b,h={'box':(.05,.09,.06),'electronics':(.06,.08,.03),'jar':(.06,.06,.12),'dish':(.12,.12,.04)}[cat];N=2500
            if cat in('box','electronics'):
                u=rng.uniform(-.5,.5,(N,3))*[b,a,0];u[:,2]=rng.uniform(0,h,N)
            else:
                t=rng.uniform(0,2*np.pi,N);u=np.c_[b/2*np.cos(t),a/2*np.sin(t),rng.uniform(0,h,N)]
            c,s_=np.cos(yaw),np.sin(yaw);P=u@np.array([[c,-s_,0],[s_,c,0],[0,0,1]]).T;P[:,0]+=x;P[:,1]+=y
            P[:,2]+=-(pl[1]+pl[0][0]*x+pl[0][1]*y)/pl[0][2];return P
        for k in range(60):
            cat=('box','electronics','jar','dish')[k%4];cl=cloud(cat,rng.uniform(.3,.5),rng.uniform(-.4,.4),rng.uniform(0,np.pi))
            others=[cloud('box',rng.uniform(.3,.5),rng.uniform(-.4,.4),0.) for _ in range(k%4)]+([np.zeros((0,3))] if k%5==0 else [])
            fa,Ca=grasp.candidates(cl,pl,cat,others,W_MAX=.1)
            with reference():fb,Cb=grasp.candidates(cl,pl,cat,others,W_MAX=.1)
            for key in ('width','length','height','axis','center'):s.assertTrue(same(fa[key],fb[key]),(cat,key))
            s.assertEqual(len(Ca),len(Cb))
            for g,h in zip(Ca,Cb):
                s.assertEqual(sorted(g),sorted(h))
                for key in g:s.assertTrue(same(g[key],h[key]),(cat,key));nc+=1
        print(f'\ngrasp.candidates: 60 objects, {nc} candidate fields identical')
    # ---------------- planning ----------------
    def test_solve_matches_seed_loop(s):
        rng=np.random.default_rng(21);same=found=0
        for A in s.arms.values():
            for i in range(150):
                q=np.clip(rng.uniform(A.chain.lower,A.chain.upper),A.chain.lower,A.chain.upper);T=A.chain.fk_py(q)
                if i%5==0:T=T.copy();T[:3,3]+=rng.normal(size=3)*.4      # partly unreachable
                seed=None if i%3 else A.seeds[1]
                a=A.solve(T,seed);b=A.solve_py(T,seed)
                s.assertEqual(a is None,b is None)
                if a is not None:s.assertTrue(np.array_equal(a,b));found+=1
                same+=1
            bad=np.eye(4);bad[:3,:3]*=2
            s.assertIsNone(A.solve(bad));s.assertIsNone(A.solve_py(bad))
        print(f'\nArm.solve: {same} targets identical ({found} solved)')
    def test_scene_cache_equivalent(s):
        A=arm.Arm('l');rng=np.random.default_rng(4)
        dets=[dict(center=(.34,.69),size=(.22,.32),axis=(.1,.995),height=.068),(.53,0.),(.34,-.69,0.,1.,.21,.31,.07)]
        for th in (0.,.7,-1.3):
            R=NODE.rotz(-th);pl=(R@s.plane[0],s.plane[1]);tr=arm.rotate_trays(dets,R)
            for trays,tx in ((tr,None),([],None),(tr,(.25,1.)),(tr[:1],(.05,.9))):
                A.set_scene(pl,trays,table_x=tx);env=A.col.environment;C,Rr=A.col._env_centers.copy(),A.col._env_radii.copy()
                qs=[rng.uniform(A.chain.lower,A.chain.upper) for _ in range(80)];va=[A.col.valid(q) for q in qs]
                A.col.environment=A.body+arm.table_env(pl,getattr(A,'table_x',None))+(arm.tray_env(trays,pl) if trays else [])
                s.assertEqual(repr(env),repr(A.col.environment));s.assertTrue(np.array_equal(C,A.col._env_centers) and np.array_equal(Rr,A.col._env_radii))
                s.assertEqual(va,[A.col.valid(q) for q in qs])
        A.set_other(np.zeros(7));A.set_scene(s.plane,tr);e1=repr(A.col.environment)
        A.col.environment=A.body+arm.table_env(s.plane,A.table_x)+arm.tray_env(tr,s.plane);s.assertEqual(e1,repr(A.col.environment))
    def test_table_env_vectorised_exact(s):
        for pl,x,y in ((s.plane,(.12,1.),(-1.,1.)),((np.array([.03,-.02,.99]),.2),(.3,.85),(-.95,.95))):
            n,d=np.asarray(pl[0],float),float(pl[1]);nn=np.linalg.norm(n);n=n/nn;d=d/nn;ref=[];r=arm.TABLE_EDGE_R
            zt=lambda a,b:-(d+n[0]*a+n[1]*b)/n[2]
            # v9fix model (scalar reference): big spheres from edge+R, top lip band, front face column
            for a in np.arange(x[0]+.1,x[1]+1e-9,.03):
                for b in np.arange(y[0],y[1]+1e-9,.03):ref.append(planning.Sphere('pelvis',np.array([a,b,zt(a,b)])-.1*n,.1))
            for a in np.arange(x[0]+r,x[0]+.1+1e-9,.03):
                for b in np.arange(y[0],y[1]+1e-9,.03):ref.append(planning.Sphere('pelvis',np.array([a,b,zt(a,b)])-r*n,r))
            for b in np.arange(y[0],y[1]+1e-9,.03):
                for dz in np.arange(r,arm.TABLE_FRONT_DEPTH+1e-9,.03):
                    a=x[0]+r;ref.append(planning.Sphere('pelvis',np.array([a,b,zt(a,b)-dz]),r))
            got=arm.table_env_build(pl,x,y);s.assertEqual(len(got),len(ref))
            for g,q in zip(got,ref):s.assertTrue(g.link==q.link and np.allclose(g.center,q.center,atol=1e-12) and g.radius==q.radius and not g.center.flags.writeable)
    def test_tray_env_bulk_exact(s):
        dets=[dict(center=(.34,.69),size=(.22,.32),axis=(.1,.995),height=.068),(.53,0.)]
        for pl in (s.plane,(np.array([.03,-.02,.99]),.2)):
            got=arm.tray_env_build(dets,pl);n,d=np.asarray(pl[0],float),float(pl[1]);ref=[];h0=.07
            for t in dets:
                tx,ty,ax,ay,sa,sb,h0=arm.tray_geom(t,(.22,.32),.07)   # each tray: its measured height or the default
                v=np.array([ax,ay])/max(1e-9,np.hypot(ax,ay));u=np.array([v[1],-v[0]]);hx,hy=sa/2,sb/2;loc=[]
                for a in np.arange(-hx,hx+1e-9,.03):loc+=[(a,-hy),(a,hy)]
                for b in np.arange(-hy,hy+1e-9,.03):loc+=[(-hx,b),(hx,b)]
                for x,y in [(tx+a*u[0]+b*v[0],ty+a*u[1]+b*v[1]) for a,b in loc]:
                    z0=-(d+n[0]*x+n[1]*y)/n[2]
                    for hh in (h0-.02,max(.02,h0-.06)):ref.append(planning.Sphere('pelvis',np.array([x,y,z0+hh]),.02))
            s.assertEqual(repr(got),repr(ref))
            s.assertTrue(all(np.array_equal(g.center,r.center) for g,r in zip(got,ref)))
    def test_tray_env_default_height_not_carried_over(s):
        # regression: a tray without a measured height used to inherit the previous tray's measured height
        meas=dict(center=(.34,.69),size=(.22,.32),axis=(0.,1.),height=.05)
        a=arm.tray_env_build([meas,(.53,0.)],s.plane);b=arm.tray_env_build([(.53,0.)],s.plane)
        s.assertEqual(repr(a[len(a)-len(b):]),repr(b))
    def test_certify_skip_and_levers_bound(s):
        rng=np.random.default_rng(8);n=agree=nn=0;viol=0.
        for A in s.arms.values():
            col=A.col;fast=col._fast
            for i in range(400):
                a=rng.uniform(A.chain.lower,A.chain.upper)*.6;b=a+(rng.uniform(A.chain.lower,A.chain.upper)*.6-a)*rng.uniform(.05,1.)
                if i%3==0:a=A.home
                r=col.certify(a,b);fast.skip=False
                try:plain=col.certify(a,b)   # Rust, every sample
                finally:fast.skip=True
                s.assertEqual(r,plain,(i,a,b));n+=1;agree+=r
                if i%20==0:          # NumPy valid_vec over every sample (slow)
                    col._fast=None
                    try:ref=col.certify(a,b)
                    finally:col._fast=fast
                    s.assertEqual(r,ref,(i,a,b));nn+=1
            for _ in range(80):
                qa=rng.uniform(A.chain.lower,A.chain.upper);qb=np.clip(qa+rng.normal(size=7)*.2,A.chain.lower,A.chain.upper)
                Fa,Fb=A.chain.frames_py(qa),A.chain.frames_py(qb)
                for k,sp in enumerate(col.spheres):
                    pa=Fa[sp.link][:3,:3]@sp.center+Fa[sp.link][:3,3];pb=Fb[sp.link][:3,:3]@sp.center+Fb[sp.link][:3,3]
                    viol=max(viol,np.linalg.norm(pa-pb)-col.levers[k]@np.abs(qb-qa))
        print(f'\ncertify skip == every-sample Rust on {n} paths ({agree} free), == NumPy on {nn}; max(|dp| - levers.|dq|) = {viol:.2e}')
        s.assertLessEqual(viol,1e-12)
    def test_identify_vectorised_cost(s):
        fn,dd,T,rgb=s.F[0];P=perception.deproject_py(dd,K,T);tb=perception.table_region(P,dd>0);pl=perception.table_plane_py(P,tb)
        dets=perception.segment_py(P,rgb,tb,pl,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
        class S:pass
        got=NODE.Sol.identify(S(),dets,P,pl,list(NODE.PRI))
        # reference: the former scalar double loop
        from scipy.optimize import linear_sum_assignment
        clouds=[P[d.pixels[:,0],d.pixels[:,1]] for d in dets];feats=[]
        for c in clouds:
            h=np.quantile(c@pl[0]+pl[1],.98);xy=c[:,:2]-c[:,:2].mean(0);ev,V=np.linalg.eigh(np.cov(xy.T));pr=xy@V;e=np.sort(pr.max(0)-pr.min(0));feats.append((h,e[0],e[1]))
        N=[n for n in NODE.PRI];C=np.full((len(dets),len(N)+len(dets)),3.)
        for i,(h,w,l) in enumerate(feats):
            for j,n in enumerate(N):
                p=NODE.PRI[n];C[i,j]=abs(h-p['height'])/.012+max(0,w-p['w_min']-.01)/.012+max(0,p['w_min']-w-.02)/.02+max(0,l-p['w_max']-.015)/.015
        r,c=linear_sum_assignment(C);ref=[(i,N[j],C[i,j]) for i,j in zip(r,c) if j<len(N)]
        ids=[id(d) for d in dets];s.assertEqual([(ids.index(id(x[0])),x[1],x[5]) for x in got],[(int(i),n,float(v)) for i,n,v in ref])
    def test_hand_mesh_cache(s):
        d=json.load(open(arm.HAND_MESH))
        for side in 'lr':s.assertEqual(arm.hand_mesh(side),d[side])
    def test_phase2_failures_fall_back(s):
        fn,dd,T,rgb=s.F[0];P=perception.deproject_py(dd,K,T);valid=dd>0
        with reference():
            tb=perception.table_region(P,valid);pl=perception.table_plane(P,tb);loc=TR.detect(P,rgb,valid,'tray_center','green',pl)
            seg=perception.segment(P,rgb,tb,pl,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
        names=('ransac_plane','deproject','tray_window_mask','min_area_rect_scan','rot_rect_mask','segment_components','min_caliper','min_dist_xy','table_mask')
        class Boom:
            def __getattr__(self,k):
                if k in names:
                    def f(*a,**kw):raise RuntimeError('injected')
                    return f
                return getattr(saved_ext,k)
        saved_ext=_accel._ext;_accel._ext=Boom()
        try:
            s.assertTrue(np.array_equal(perception.deproject(dd,K,T),P))
            s.assertTrue(np.array_equal(perception.table_region(P,valid),tb));s.assertFalse(_accel.enabled2())
            _accel._disabled=False;a=perception.table_plane(P,tb);s.assertTrue(np.array_equal(a[0],pl[0]) and a[1]==pl[1])
            _accel._disabled=False;s.assertEqual(TR.detect(P,rgb,valid,'tray_center','green',pl),loc)
            _accel._disabled=False;g=perception.segment(P,rgb,tb,pl,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
            s.assertEqual([x.pixels.tolist() for x in g],[x.pixels.tolist() for x in seg])
        finally:
            _accel._ext=saved_ext;_accel._disabled=False
if __name__=='__main__':unittest.main()
