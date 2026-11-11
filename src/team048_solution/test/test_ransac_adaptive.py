"""Adaptive number of RANSAC samples for the table plane (perception.RANSAC_MISS_MAX, Hartley & Zisserman 2004, eq. 4.18):
the formula, no change on the recorded frames, the extension on a low-inlier cloud, Rust == NumPy reference."""
import os,sys,math,unittest
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,HERE)
from team048_solution import perception as PE, arm as ARM
F=os.path.join(HERE,'fixtures')

def cloud_low_inliers(seed,w=.22,m=20000):
    """Synthetic organised cloud (1, m, 3): a fraction w on the plane z = -0.1 (2 mm noise), the rest spread 1 cm to 40 cm
    above it (clutter); x, y inside the node's table search window."""
    r=np.random.default_rng(1000+seed);ni=int(w*m)
    tab=np.c_[r.uniform(.2,.7,ni),r.uniform(-.6,.6,ni),.002*r.standard_normal(ni)-.1]
    out=np.c_[r.uniform(.2,.7,m-ni),r.uniform(-.6,.6,m-ni),r.uniform(-.09,.3,m-ni)]
    P=np.r_[tab,out];r.shuffle(P);return P.reshape(1,m,3),np.ones((1,m),bool)

class capped:
    """RANSAC_ITER_MAX = cap inside the block (cap = 120 switches the extension off)."""
    def __init__(s,cap):s.cap=cap
    def __enter__(s):s.old=PE.RANSAC_ITER_MAX;PE.RANSAC_ITER_MAX=s.cap
    def __exit__(s,*a):PE.RANSAC_ITER_MAX=s.old

def plane_error(pl):return abs(pl[1]-.1)+float(np.linalg.norm(pl[0]-[0,0,1]))

class Formula(unittest.TestCase):
    def test_eq_4_18(self):
        for w in (.3,.406,.5,.8):
            N=PE.ransac_iterations(w,cap=10**6)
            self.assertEqual(N,math.ceil(math.log(1e-3)/math.log(1-w**3)))
            self.assertLessEqual(PE.ransac_miss_probability(w,N),1e-3)
            self.assertGreater(PE.ransac_miss_probability(w,N-1),1e-3)
        self.assertEqual(PE.ransac_iterations(.5,cap=10**6),52)
        self.assertEqual(PE.ransac_iterations(0.),PE.RANSAC_ITER_MAX);self.assertEqual(PE.ransac_iterations(1.),1)
    def test_threshold_of_the_extension(self):
        self.assertIsNone(PE._ransac_more(406,1000,120))     # smallest inlier fraction of the 62 recorded frames
        self.assertIsNotNone(PE._ransac_more(380,1000,120))
        self.assertIsNone(PE._ransac_more(0,1000,PE.RANSAC_ITER_MAX))   # never beyond the cap
        self.assertEqual(PE._ransac_more(0,1000,120),PE.RANSAC_ITER_MAX)

class RecordedFrames(unittest.TestCase):
    def frames(self):
        out=[]
        z=np.load(os.path.join(F,'v7_s102_start_frames.npz'));K=z['K'].reshape(3,3)
        T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});d=z['head_depth'].astype(float)
        out.append((PE.deproject_py(d,K,T),d>0))
        return out
    def test_no_extension_and_same_plane(self):
        for P,v in self.frames():
            tb=PE.table_region(P,v);p=P[tb];p=p[np.isfinite(p).all(1)]
            a=PE.table_plane(P,tb)
            with capped(120):b=PE.table_plane(P,tb);c=PE.table_plane_py(P,tb)
            self.assertTrue(np.array_equal(a[0],b[0]) and a[1]==b[1])
            self.assertTrue(np.array_equal(PE.table_plane_py(P,tb)[0],c[0]))
            w=float(np.mean(np.abs(p@b[0]+b[1])<.006));self.assertGreater(w,.38)

class LowInlierCloud(unittest.TestCase):
    def test_extension_recovers_the_plane(self):
        bad120=bad=0
        for seed in range(12):
            P,v=cloud_low_inliers(seed)
            with capped(120):
                try:e120=plane_error(PE.table_plane(P,v))
                except ValueError:e120=9.
            try:e=plane_error(PE.table_plane(P,v))
            except ValueError:e=9.
            bad120+=e120>.01;bad+=e>.01
        self.assertGreaterEqual(bad120,4)     # 120 samples at w = 0.22: miss probability (1-0.22^3)^120 = 0.28
        self.assertLessEqual(bad,1)
    def test_extended_reference_equals_fast_path(self):
        for seed in range(6):
            P,v=cloud_low_inliers(seed)
            a=PE.table_plane(P,v);b=PE.table_plane_py(P,v)
            np.testing.assert_allclose(a[0],b[0],rtol=0,atol=1e-12);self.assertAlmostEqual(a[1],b[1],delta=1e-12)
    def test_prefix_of_the_stream_is_kept(self):
        # the extension continues the same candidate stream: its best count is never below the first pass
        for seed in range(6):
            P,v=cloud_low_inliers(seed);p=P[v]
            b1,s1=PE._ransac_ref(p,120,.006,0);b2,s2=PE._ransac_ref(p,PE.RANSAC_ITER_MAX,.006,0)
            self.assertGreaterEqual(s2,s1)

if __name__=='__main__':unittest.main()
