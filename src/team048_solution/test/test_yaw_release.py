"""v7 evidence (sort3 s103, Hyaluronic_Acid): fingers fully closed (0.003, effort 0.004) after lift = empty hand; after
release 0.97, home 0.99 = open (INTERFACES.md: 1.0 open, 0.0 closed). Torso yaw (body_yaw_joint, motor 31) for placing."""
import sys,os,types,unittest,numpy as np
sys.path.insert(0,os.path.dirname(__file__));import test_node_stub
from team048_solution import node as N, grip as G, trays as TR, arm as ARM
SIDE=[0,0,0,0,0,0.]
class Release(unittest.TestCase):
    def test_v7_s103_hyaluronic_lifted_is_empty(self):
        self.assertIs(G.finger_verdict([.003,.003,.003,.003,.005,0],[.003,.003,.004,.004,.002,.003],SIDE),False)
    def test_v7_s102_hyaluronic_closed_on_nothing(self):
        self.assertIs(G.finger_verdict([-.001,-.001,-.001,0,.004,0],[.001,.001,.003,.005,.004,.003],SIDE),False)
    def test_fingers_open_semantics(self):
        s=types.SimpleNamespace(hstate={})
        self.assertIsNone(N.Sol.fingers_open(s,'l'))
        s.hstate['l']=types.SimpleNamespace(position=[.974,.972,.972,.973,.987,0.])   # v7 'released'
        self.assertTrue(N.Sol.fingers_open(s,'l'))
        s.hstate['l']=types.SimpleNamespace(position=[.006,.007,.007,.007,.015,0.])   # v7 'over_tray' (closed)
        self.assertFalse(N.Sol.fingers_open(s,'l'))
    def test_open_command_is_one(self):
        got={}
        s=types.SimpleNamespace(hand_set=lambda side,v:got.setdefault(side,v))
        N.Sol.hand_open(s,'l');self.assertEqual(list(got['l'][:5]),[1,1,1,1,1])
class Yaw(unittest.TestCase):
    def test_candidates_turn_toward_far_tray(self):
        tx,ty=TR.NOMINAL['tray_right']
        c=N.yaw_candidates('l',tx,ty);self.assertTrue(c);self.assertTrue(all(t<0 and abs(t)<=N.YAW_MAX for t in c))
        tx,ty=TR.NOMINAL['tray_left'];c=N.yaw_candidates('r',tx,ty);self.assertTrue(c);self.assertTrue(all(t>0 for t in c))
    def test_rotz_maps_tray_in_front(self):
        th=np.arctan2(.69,.34);p=N.rotz(-th)@np.r_[.34,.69,0];self.assertAlmostEqual(p[1],0,6);self.assertGreater(p[0],.7)
    def test_plan_place_yaw_restores_scene(self):
        A=ARM.Arm('l');n=np.array([-.0699,0,.9976]);pl=(n/np.linalg.norm(n),.1788/np.linalg.norm(n))
        A.set_scene(pl,list(TR.NOMINAL.values()));e0=repr(A.col.environment)
        N.plan_place_yaw(A,A.home,*TR.NOMINAL['tray_right'],pl,list(TR.NOMINAL.values()))
        self.assertEqual(e0,repr(A.col.environment))
class FarTray(unittest.TestCase):
    def test_far_tray_sides(self):
        L,R,C=TR.NOMINAL['tray_left'][1],TR.NOMINAL['tray_right'][1],TR.NOMINAL['tray_center'][1]
        self.assertTrue(N.far_tray('l',R));self.assertTrue(N.far_tray('r',L))
        self.assertFalse(N.far_tray('l',L));self.assertFalse(N.far_tray('r',R))
        self.assertFalse(N.far_tray('l',C));self.assertFalse(N.far_tray('r',C))
    def test_far_tray_skips_unyawed_search(self):
        calls=[];orig=N.plan_place
        def spy(A,q,tx,ty,top,fallback=None):calls.append(round(float(ty),3));return None
        N.plan_place=spy
        try:
            A=ARM.Arm('l');n=np.array([-.0699,0,.9976]);pl=(n/np.linalg.norm(n),.1788/np.linalg.norm(n))
            N.plan_place_yaw(A,A.home,*TR.NOMINAL['tray_right'],pl,list(TR.NOMINAL.values()))
        finally:N.plan_place=orig
        self.assertTrue(calls);self.assertNotIn(-.69,calls)    # every attempt is in a yawed (rotated) frame
class InfeasibleCache(unittest.TestCase):
    def test_replan_only_after_scene_change(self):
        nof={}
        self.assertTrue(N.replan_infeasible(nof,'a',set()))
        nof['a']=0;self.assertFalse(N.replan_infeasible(nof,'a',set()))
        self.assertTrue(N.replan_infeasible(nof,'a',{'b'}))       # an object was removed: obstacles changed
        self.assertTrue(N.replan_infeasible(nof,'c',set()))
if __name__=='__main__':unittest.main()
