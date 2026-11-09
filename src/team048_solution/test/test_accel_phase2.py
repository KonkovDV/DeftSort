"""Phase-2 Rust kernels vs their NumPy references (perception, grasp candidates, Arm.solve, scenes, certify skip).
Contract: bit-identical results, except values that pass through a NumPy BLAS dot product (the reference itself is
CPU-dependent there: OpenBLAS uses FMA on AVX2/AVX-512): <= 1e-12 absolute, with identical decisions.
Skipped when the extension (API >= 2) is not installed."""
import contextlib,glob,json,os,sys,unittest,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,HERE)
import test_node_stub  # noqa: ROS stubs for node.py
from team048_solution import _accel, arm, perception, trays as TR, grasp, planning, node as NODE
FRAMES=sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/*/*.npz'))[:int(os.environ.get('T48_EQ_FRAMES',31))]
FIX=os.path.join(HERE,'fixtures','v5_head_frame.npz')
K=np.array([[343.16,0,320.],[0,343.16,180.],[0,0,1]])

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
        for A in c.arms.values():A.set_scene(c.plane,list(TR.NOMINAL.values()))
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
            with reference():tb_ref=perception.table_region(R,valid);tr_ref=perception.table_region(R,valid,[(x,y,.13,.18) for x,y in TR.NOMINAL.values()])
            s.assertTrue(np.array_equal(perception.table_region(R,valid),tb_ref))
            s.assertTrue(np.array_equal(perception.table_region(R,valid,[(x,y,.13,.18) for x,y in TR.NOMINAL.values()]),tr_ref))
            for seed in (0,1,7):
                a=perception.table_plane(R,tb_ref,seed=seed);b=perception.table_plane_py(R,tb_ref,seed=seed)
                s.assertTrue(np.array_equal(a[0],b[0]) and a[1]==b[1],(fn,seed))
        print(f'\ndeproject max |diff| {dmax:.1e} over {len(s.F)} frames');s.assertLessEqual(dmax,1e-12)
    def test_tray_locate_exact(s):
        n=0
        for fn,dd,T,rgb in s.F:
            P=perception.deproject_py(dd,K,T);valid=dd>0;pl=perception.table_plane_py(P,perception.table_region(P,valid))
            for name in TR.NOMINAL:
                for col in TR.CHANNEL:
                    for plane in (pl,None):
                        a=TR.locate(P,rgb,valid,name,col,plane)
                        with reference():b=TR.locate(P,rgb,valid,name,col,plane)
                        s.assertTrue(np.array_equal(a[0],b[0]) and a[1]==b[1],(fn,name,col));n+=1
        print(f'\ntrays.locate: {n} calls identical')
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
            for key in ('width','length','height'):s.assertEqual(fa[key],fb[key])
            s.assertTrue(np.array_equal(fa['axis'],fb['axis']) and np.array_equal(fa['center'],fb['center']))
            s.assertEqual(len(Ca),len(Cb))
            for g,h in zip(Ca,Cb):
                s.assertEqual(sorted(g),sorted(h))
                for key in g:s.assertTrue(np.array_equal(np.asarray(g[key]),np.asarray(h[key])),(cat,key));nc+=1
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
        for th in (0.,.7,-1.3):
            R=NODE.rotz(-th);pl=(R@s.plane[0],s.plane[1]);tr=[tuple((R@np.r_[x,y,0.])[:2]) for x,y in TR.NOMINAL.values()]
            for trays in (tr,[]):
                A.set_scene(pl,trays);env=A.col.environment;C,Rr=A.col._env_centers.copy(),A.col._env_radii.copy()
                qs=[rng.uniform(A.chain.lower,A.chain.upper) for _ in range(80)];va=[A.col.valid(q) for q in qs]
                A.col.environment=A.body+arm.table_env(plane=pl)+(arm.tray_env(trays,pl) if trays else [])
                s.assertEqual(repr(env),repr(A.col.environment));s.assertTrue(np.array_equal(C,A.col._env_centers) and np.array_equal(Rr,A.col._env_radii))
                s.assertEqual(va,[A.col.valid(q) for q in qs])
    def test_table_env_vectorised_exact(s):
        for pl in (s.plane,(np.array([.03,-.02,.99]),.2)):
            n,d=np.asarray(pl[0],float),float(pl[1]);nn=np.linalg.norm(n);n=n/nn;d=d/nn;ref=[]
            for a in np.arange(.32,.85+1e-9,.03):
                for b in np.arange(-.95,.95+1e-9,.03):
                    z=-(d+n[0]*a+n[1]*b)/n[2];ref.append(planning.Sphere('pelvis',np.array([a,b,z])-.1*n,.1))
            got=arm.table_env_build(plane=pl);s.assertEqual(len(got),len(ref))
            for g,r in zip(got,ref):s.assertTrue(g.link==r.link and np.array_equal(g.center,r.center) and g.radius==r.radius and not g.center.flags.writeable)
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
            tb=perception.table_region(P,valid);pl=perception.table_plane(P,tb);loc=TR.locate(P,rgb,valid,'tray_center','green',pl)
            seg=perception.segment(P,rgb,tb,pl,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
        names=('ransac_plane','deproject','tray_select','segment_components','min_caliper','min_dist_xy','table_mask')
        saved={k:getattr(_accel._ext,k) for k in names}
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
            _accel._disabled=False;b=TR.locate(P,rgb,valid,'tray_center','green',pl);s.assertTrue(np.array_equal(b[0],loc[0]) and b[1]==loc[1])
            _accel._disabled=False;g=perception.segment(P,rgb,tb,pl,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
            s.assertEqual([x.pixels.tolist() for x in g],[x.pixels.tolist() for x in seg])
        finally:
            _accel._ext=saved_ext;_accel._disabled=False
if __name__=='__main__':unittest.main()
