"""Existing IK/collision acceptance tests (package import)."""
import os,sys,unittest,numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));import scene_fixture as SF
from team048_solution import arm
def why(A,q):
    fr=A.chain.frames(q);W=[(s,fr[s.link][:3,:3]@s.center+fr[s.link][:3,3]) for s in A.col.spheres];out=[]
    for s,p in W:
        for e in A.col.environment:
            d=np.linalg.norm(e.center-p)-s.radius-e.radius-A.col.margin
            if d<=0:out.append((s.link,'ENV',np.round(e.center,2),round(d,3)))
    for i,(s,p) in enumerate(W):
        for t,r in W[i+1:]:
            if s.link==t.link or frozenset((s.link,t.link)) in A.col.ignored:continue
            d=np.linalg.norm(p-r)-s.radius-t.radius-A.col.margin
            if d<=0:out.append((s.link,t.link,round(d,3)))
    return out
class T(unittest.TestCase):
    @classmethod
    def setUpClass(c):c.A=arm.Arm('l');c.A.set_scene(SF.plane())
    def test_limits_from_urdf(s):
        ch=s.A.chain
        s.assertEqual(len(ch.names),7);s.assertTrue(np.all(ch.velocity<=.8))
        s.assertAlmostEqual(ch.lower[ch.names.index('shoulder_roll_l_joint')],-0.2617993877991494)
    def test_fk_ik_roundtrip(s):
        rng=np.random.default_rng(1);ok=0
        for _ in range(10):
            q=rng.uniform(s.A.chain.lower,s.A.chain.upper)*.4;T=s.A.chain.fk(q)
            try:qq=s.A.chain.ik(T,s.A.home,max_iter=300);ok+=np.linalg.norm(s.A.chain.fk(qq)[:3,3]-T[:3,3])<.003
            except ValueError:pass
        s.assertGreaterEqual(ok,7)
    def test_home_valid_and_self_collision_detected(s):
        s.assertTrue(s.A.col.valid(s.A.home))
        rng=np.random.default_rng(0);hits=0
        for _ in range(200):
            q=rng.uniform(s.A.chain.lower,s.A.chain.upper)
            if not s.A.col.valid(q):
                w=why(s.A,q);hits+=any(x[1]!='ENV' or x[2][0]<.3 for x in w)   # self/body (not table)
        s.assertGreater(hits,0)
    def test_hand_opening(s):
        h=s.A.hand;s.assertTrue(.06<h['opposed_open']<.14);s.assertAlmostEqual(np.linalg.norm(h['close_axis_hand']),1,6)
    def test_grasp_pose_maps_axes(s):
        T=s.A.grasp_pose(np.array([.4,.2,-.15]),np.array([0,1.,0]),np.array([1.,0,0]))
        g=(T@np.r_[arm.J['left_hand_to_tcp']['T'][:3,:3]@s.A.hand['grip_center_hand'],1])[:3];s.assertLess(np.linalg.norm(g-[.4,.2,-.15]),1e-6)
    def test_speed_cap(s):
        p=s.A.plan_pick(np.array([.36,.25,SF.TABLE_TOP+.05]),np.array([0,1.,0]),np.array([1.,0,0]),.05)
        s.assertIsNotNone(p);s.assertLessEqual(p['max_speed'],.8+1e-6)
    def test_no_plan_before_measured_scene(s):
        A=arm.Arm('l');s.assertFalse(A.scene_ready)        # #645: no stored table -> no planning
        s.assertIsNone(A.plan_pick(np.array([.36,.25,SF.TABLE_TOP+.05]),np.array([0,1.,0]),np.array([1.,0,0]),.05))
        with s.assertRaises(ValueError):arm.table_env(None)
    def test_too_wide_rejected(s):
        s.assertIsNone(s.A.plan_pick(np.array([.36,.25,-.13]),np.array([0,1.,0]),np.array([1.,0,0]),.2))
if __name__=='__main__':unittest.main()
