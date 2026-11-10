"""Rust extension (team048_accel) vs the NumPy reference on thousands of random inputs.
Skipped when the extension is not installed (the reference is then the only implementation)."""
import glob,os,sys,unittest,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'))
from team048_solution import _accel, arm, perception
N_FK=int(os.environ.get('T48_EQ_N',3000))
N_IK=int(os.environ.get('T48_EQ_NIK',1000))

@unittest.skipUnless(_accel.AVAILABLE,'team048_accel not installed: '+_accel.REASON)
class Equivalence(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        c.arms={k:arm.Arm(k) for k in 'lr'}
        n=np.array([-.0699,0,.9976]);c.plane=(n/np.linalg.norm(n),.1788/np.linalg.norm(n))
        for A in c.arms.values():A.set_scene(c.plane,[(.34,.69),(.34,-.69),(.53,0.)]);assert A.chain._fast is not None and A.col._fast is not None
    def rq(s,A,rng,scale=1.):
        lo,hi=A.chain.lower,A.chain.upper;m=(lo+hi)/2;return np.clip(m+(rng.uniform(lo,hi)-m)*scale,lo,hi)
    def test_fk_frames_jacobian(s):
        rng=np.random.default_rng(10);e=0.
        for A in s.arms.values():
            ch=A.chain
            for _ in range(N_FK):
                q=s.rq(A,rng)
                e=max(e,np.abs(ch.fk(q)-ch.fk_py(q)).max(),np.abs(ch.jacobian(q)-ch.jacobian_py(q)).max())
                F,P=ch.frames(q),ch.frames_py(q);s.assertEqual(list(F),list(P))
                e=max(e,max(np.abs(F[k]-P[k]).max() for k in F))
        print(f'\nFK/frames/Jacobian max |diff| {e:.2e}');s.assertLess(e,1e-12)
    def test_ik_matches_reference(s):
        rng=np.random.default_rng(11);agree=diverged=fails=0;emax=0.
        for side,A in s.arms.items():
            ch=A.chain
            for i in range(N_IK):
                T=ch.fk(s.rq(A,rng,.7));seed=A.seeds[i%len(A.seeds)];w=(0.,.1)[i%2]
                kw=dict(max_iter=150,q_pref=A.home if w else None,w_null=w)
                try:a=ch.ik(T,seed,**kw)
                except ValueError:a=None
                try:b=ch.ik_py(T,seed,**kw)
                except ValueError:b=None
                if (a is None)!=(b is None):diverged+=1;continue
                if a is None:fails+=1;agree+=1;continue
                d=np.abs(a-b).max();emax=max(emax,d)
                if d<=1e-9:agree+=1
                else:
                    diverged+=1   # fp-level branch flip: both must still be valid solutions
                    for q in (a,b):s.assertLess(np.linalg.norm(ch.fk(q)[:3,3]-T[:3,3]),.002)
        print(f'\nIK: {agree}/{2*N_IK} identical (<=1e-9, incl. {fails} both non-converged), {diverged} diverged, max diff {emax:.2e}')
        s.assertGreaterEqual(agree,0.995*2*N_IK)
    def test_valid_and_valid_vec_match_reference(s):
        rng=np.random.default_rng(12);n=0;hits=0
        for A in s.arms.values():
            for i in range(N_FK):
                q=s.rq(A,rng);inf=rng.uniform(0,.03,len(A.col.spheres)) if i%2 else 0.
                r=A.col.valid_py(q,inf);s.assertEqual(A.col.valid(q,inf),r);s.assertEqual(A.col.valid_vec(q,inf),r);n+=1;hits+=not r
        print(f'\nvalid: {n} configs identical, {hits} in collision');s.assertGreater(hits,n*.2);s.assertLess(hits,n)
    def test_certify_matches_reference(s):
        rng=np.random.default_rng(13);res=[]
        for A in s.arms.values():
            for i in range(60):
                a=s.rq(A,rng,.6);b=np.clip(a+rng.normal(0,(.05,.3)[i%2],7),A.chain.lower,A.chain.upper)
                r=A.col.certify_py(a,b);s.assertEqual(A.col.certify(a,b),r);res.append(r)
                lo,hi=np.minimum(a,b),np.maximum(a,b)
                s.assertEqual(A.col.certify_box(a,b),A.col.valid_py((lo+hi)/2,A.col.levers@((hi-lo)/2)))
        print(f'\ncertify: {len(res)} sweeps identical, {sum(res)} certified');s.assertTrue(any(res) and not all(res))
    def test_scene_and_margin_updates_propagate(s):
        A=s.arms['l'];rng=np.random.default_rng(14);m0=A.col.margin
        try:
            A.col.margin=.03
            for _ in range(200):q=s.rq(A,rng);s.assertEqual(A.col.valid(q),A.col.valid_py(q))
        finally:A.col.margin=m0
        A.set_scene((s.plane[0],s.plane[1]+.05),[])
        try:
            for _ in range(200):q=s.rq(A,rng);s.assertEqual(A.col.valid(q),A.col.valid_py(q))
        finally:A.set_scene(s.plane,[(.34,.69),(.34,-.69),(.53,0.)])
    def test_errors_preserved(s):
        A=s.arms['l'];ch=A.chain;T=ch.fk(A.home)
        with s.assertRaises(ValueError):ch.fk(ch.upper+.1)
        with s.assertRaises(ValueError):A.col.valid(ch.upper+.1)
        with s.assertRaises(ValueError):ch.ik(T,A.home,max_iter=0)
        B=T.copy();B[:3,:3]*=0.9997    # passes check_transform, fails rotation_error's det check (reference behaviour)
        with s.assertRaises(ValueError):ch.ik_py(B,A.home)
        with s.assertRaises(ValueError):ch.ik(B,A.home)
    def test_runtime_failure_falls_back(s):
        A=arm.Arm('r');q=A.home.copy()
        class Boom:
            def __getattr__(self,k):
                def f(*a,**kw):raise RuntimeError('boom')
                return f
        ref=(A.chain.fk_py(q),A.col.valid_py(q))
        saved=_accel._disabled
        try:
            A.chain._fast=Boom();A.col._fast=Boom()
            s.assertTrue(np.allclose(A.chain.fk(q),ref[0]));s.assertEqual(A.col.valid(q),ref[1])
            s.assertIsNone(A.chain._fast);s.assertIsNone(A.col._fast)
        finally:_accel._disabled=saved
    def test_perception_kernels_match_reference(s):
        frames=sorted(glob.glob(os.path.join(os.environ.get('TEAM048_V7_TRACE',os.path.join(HERE,'..','..','..','research','kaggle-sim','out_v7','trace')),'*','*.npz')))[:12]
        if not frames:s.skipTest('v7 frames not on this machine')
        K=dict(fx=343.16,cx=320.,cy=180.)
        for fn in frames:
            z=np.load(fn);dd=z['depth'].astype(float);dd=dd[...,0] if dd.ndim==3 else dd
            T=arm.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])})
            v,u=np.mgrid[0:dd.shape[0],0:dd.shape[1]];Z=dd/1000.;P=(np.stack([(u-K['cx'])*Z/K['fx'],(v-K['cy'])*Z/K['fx'],Z],-1).reshape(-1,3)@T[:3,:3].T+T[:3,3]).reshape(dd.shape+(3,))
            valid=dd>0;tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
            a=perception.table_plane(P,tb);b=perception.table_plane_py(P,tb)
            s.assertTrue(np.array_equal(a[0],b[0]) and a[1]==b[1],fn)
            roi=(np.array([.18,-.62,-1.]),np.array([.62,.62,1.]))
            A=perception.segment(P,z['rgb'],tb,b,roi,min_pixels=40);B=perception.segment_py(P,z['rgb'],tb,b,roi,min_pixels=40)
            s.assertEqual(len(A),len(B))
            for x,y in zip(A,B):
                s.assertTrue(np.array_equal(x.pixels,y.pixels) and np.array_equal(x.centroid,y.centroid) and np.array_equal(x.lower,y.lower) and np.array_equal(x.upper,y.upper))
                # height quantile: Rust sums the dot product left to right, NumPy's BLAS gemv may use FMA (CPU-dependent): <= 1 ulp
                s.assertTrue(np.allclose(x.features,y.features,rtol=0,atol=1e-12) and np.array_equal(np.delete(x.features,2),np.delete(y.features,2)))
if __name__=='__main__':unittest.main()
