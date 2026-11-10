import unittest,numpy as np,arm
class T(unittest.TestCase):
    @classmethod
    def setUpClass(c):c.A=arm.Arm('l')
    def test_limits_from_urdf(s):
        self=s;ch=s.A.chain
        self.assertEqual(len(ch.names),7);self.assertTrue(np.all(ch.velocity<=.8))
        self.assertAlmostEqual(ch.lower[ch.names.index('shoulder_roll_l_joint')],-0.2617993877991494)
    def test_fk_ik_roundtrip(s):
        rng=np.random.default_rng(1);ok=0
        for _ in range(10):
            q=rng.uniform(s.A.chain.lower,s.A.chain.upper)*.4;T=s.A.chain.fk(q)
            try:qq=s.A.chain.ik(T,s.A.home,max_iter=300);ok+=np.linalg.norm(s.A.chain.fk(qq)[:3,3]-T[:3,3])<.003
            except ValueError:pass
        s.assertGreaterEqual(ok,7)
    def test_home_valid_and_self_collision_detected(s):
        s.assertTrue(s.A.col.valid(s.A.home))
        import dbg;rng=np.random.default_rng(0);hits=0
        for _ in range(200):
            q=rng.uniform(s.A.chain.lower,s.A.chain.upper)
            if not s.A.col.valid(q):
                w=dbg.why(s.A,q);hits+=any(x[1]!='ENV' or x[2][0]<.3 for x in w)   # self/body (not table)
        s.assertGreater(hits,0)
    def test_hand_opening(s):
        h=s.A.hand;s.assertTrue(.06<h['opposed_open']<.14);s.assertAlmostEqual(np.linalg.norm(h['close_axis_hand']),1,6)
    def test_grasp_pose_maps_axes(s):
        T=s.A.grasp_pose(np.array([.4,.2,-.15]),np.array([0,1.,0]),np.array([1.,0,0]))
        g=(T@np.r_[arm.J['left_hand_to_tcp']['T'][:3,:3]@s.A.hand['grip_center_hand'],1])[:3];s.assertLess(np.linalg.norm(g-[.4,.2,-.15]),1e-6)
    def test_speed_cap(s):
        p=s.A.plan_pick(np.array([.36,.25,arm.TABLE_TOP+.05]),np.array([0,1.,0]),np.array([1.,0,0]),.05)
        if p:s.assertLessEqual(p['max_speed'],.8+1e-6)
    def test_too_wide_rejected(s):
        s.assertIsNone(s.A.plan_pick(np.array([.36,.25,-.13]),np.array([0,1.,0]),np.array([1.,0,0]),.2))
if __name__=='__main__':unittest.main()
