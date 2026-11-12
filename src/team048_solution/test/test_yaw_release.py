"""v7 evidence (sort3 s103, Hyaluronic_Acid): fingers fully closed (0.003, effort 0.004) after lift = empty hand; after
release 0.97, home 0.99 = open (INTERFACES.md: 1.0 open, 0.0 closed). Torso yaw (body_yaw_joint, motor 31) for placing."""
import sys,os,types,unittest,numpy as np
sys.path.insert(0,os.path.dirname(__file__));import test_node_stub  # noqa: F401  (installs the ROS stubs)
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));import scene_fixture as SF
from team048_solution import node as N, grip as G, arm as ARM
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
        tx,ty=SF.TRAYS['tray_right']
        c=N.yaw_candidates('l',tx,ty);self.assertTrue(c);self.assertTrue(all(t<0 and abs(t)<=N.YAW_MAX for t in c))
        tx,ty=SF.TRAYS['tray_left'];c=N.yaw_candidates('r',tx,ty);self.assertTrue(c);self.assertTrue(all(t>0 for t in c))
    def test_rotz_maps_tray_in_front(self):
        th=np.arctan2(.69,.34);p=N.rotz(-th)@np.r_[.34,.69,0];self.assertAlmostEqual(p[1],0,6);self.assertGreater(p[0],.7)
    def test_plan_place_yaw_restores_scene(self):
        A=ARM.Arm('l');n=np.array([-.0699,0,.9976]);pl=(n/np.linalg.norm(n),.1788/np.linalg.norm(n))
        A.set_scene(pl,list(SF.TRAYS.values()));e0=repr(A.col.environment)
        N.plan_place_yaw(A,A.home,*SF.TRAYS['tray_right'],pl,list(SF.TRAYS.values()))
        self.assertEqual(e0,repr(A.col.environment))
    def test_turned_scene_keeps_the_table_edge_in_the_pelvis_frame(self):
        """body_yaw_joint turns the torso and the arms, not the table, the trays or the hips. Rebuilding the table from
        a rotated plane put its near edge in front of the turned torso: for a 2.2 rad turn the modelled table
        reached 0.24 m behind the real near edge, and the real table beside the arm was missing."""
        A=ARM.Arm('r');n=np.array([-.0699,0,.9976]);pl=(n/np.linalg.norm(n),.1788/np.linalg.norm(n));edge=.24
        trays=list(SF.TRAYS.values())
        A.set_scene(pl,trays,table_x=(edge,ARM.TABLE_X_FAR));C0=A.col._env_centers.copy();R0=A.col._env_radii.copy()
        nb=len(A.body);h=ARM.HIP_SPHERES;nt=len(ARM.table_env(pl,(edge,ARM.TABLE_X_FAR)))
        for th in (1.1,-2.2):
            A.set_scene(pl,trays,yaw=th);C=A.col._env_centers;Rt=N.rotz(-th)
            self.assertTrue(np.array_equal(A.col._env_radii,R0))
            self.assertTrue(np.array_equal(C[:nb-h],C0[:nb-h]))                       # torso, head, other arm turn along
            self.assertTrue(np.allclose(C[nb-h:],C0[nb-h:]@Rt.T,atol=1e-12))          # hips, table, trays stay
            back=C[nb:nb+nt]@N.rotz(th).T                                            # table, back in the pelvis frame
            self.assertGreaterEqual(back[:,0].min(),edge-1e-9)
        A.set_scene((N.rotz(-2.2)@pl[0],pl[1]),[])                                  # the former construction, yaw 2.2
        old=A.col._env_centers[nb:nb+nt]@N.rotz(2.2).T
        self.assertLess(old[:,0].min(),edge-.2)
        A.set_scene(pl,trays);self.assertTrue(np.array_equal(A.col._env_centers,C0))
    def test_at_yaw_restores_the_previous_scene(self):
        A=ARM.Arm('l');A.set_other(np.zeros(7));A.set_scene(SF.plane(),list(SF.TRAYS.values()));e0=A.col._env_centers.copy()
        with A.at_yaw(.8):
            self.assertAlmostEqual(A._yaw,.8)
            with A.at_yaw(.8):self.assertAlmostEqual(A._yaw,.8)
            A.set_other(np.r_[.2,np.zeros(6)]);self.assertAlmostEqual(A._yaw,.8)      # other-arm update keeps the yaw
        self.assertEqual(A._yaw,0.);A.set_other(np.zeros(7))
        self.assertTrue(np.array_equal(A.col._env_centers,e0))
class TurnSweep(unittest.TestCase):
    """The hold posture is carried through the whole torso turn. Five sampled yaws left 0.6 rad between checks, a
    tray wall band is about 0.15 rad wide at the hand's radius."""
    def setUp(s):
        s.pl=SF.plane();s.trays=list(SF.TRAYS.values())
    def valid_at(s,A,q,a):
        A.set_scene(s.pl,s.trays,yaw=a)
        try:return A.col.valid(q)
        finally:A.set_scene(s.pl,s.trays)
    def test_wall_between_samples_is_caught(s):
        A=ARM.Arm('l');A.set_scene(s.pl,s.trays);x,y=.42,.30
        q=A.solve(A.grasp_pose(np.array([x,y,ARM.plane_height(s.pl,x,y)+.05]),np.array([1.,0,0]),np.array([0,0,-1.]),'pinch'))
        s.assertIsNotNone(q);s.assertTrue(A.col.valid(q))
        s.assertTrue(all(s.valid_at(A,q,a) for a in np.linspace(0,-2.5,5)))   # the former 5-sample check passed
        s.assertFalse(s.valid_at(A,q,-.2))                                   # the hand meets the centre tray wall
        s.assertFalse(A.certify_turn(q,0.,-2.5))
    def test_home_turns_through_the_whole_yaw_range(s):
        # go_home runs before waist(0) after a yawed place: the turn back happens at the home posture
        for side in 'lr':
            A=ARM.Arm(side);A.set_scene(s.pl,s.trays)
            for th in (N.YAW_MAX,-N.YAW_MAX):s.assertTrue(A.certify_turn(A.home,0.,th),(side,th))
    def test_accepted_turn_is_clear_on_a_dense_sweep(s):
        A=ARM.Arm('r');A.set_scene(s.pl,s.trays);rng=np.random.default_rng(3);checked=0
        for _ in range(40):
            q=rng.uniform(A.chain.lower,A.chain.upper)*.6;th=float(rng.uniform(-N.YAW_MAX,N.YAW_MAX))
            if not A.col.valid(q) or not A.certify_turn(q,0.,th):continue
            checked+=1;s.assertTrue(all(s.valid_at(A,q,a) for a in np.linspace(0,th,300)))
        s.assertGreater(checked,3)
class CorrectionYaw(unittest.TestCase):
    """A TCP correction is certified in whatever scene is installed. During a yawed place the torso is turned and the
    scene on the arm is back at yaw 0, so the correction has to be asked for inside at_yaw(measured)."""
    def test_yaw0_correction_crosses_the_tray_wall_once_the_torso_is_turned(s):
        import inspect
        A=ARM.Arm('l');pl=SF.plane();trays=list(SF.TRAYS.values());A.set_scene(pl,trays)
        x,y=.42,.30
        q=A.solve(A.grasp_pose(np.array([x,y,ARM.plane_height(pl,x,y)+.05]),np.array([1.,0,0]),np.array([0,0,-1.]),'pinch'))
        qm=q.copy();qm[0]+=.04
        yaw0=A.correct(qm,q,max_step=.15);s.assertIsNotNone(yaw0)
        A.set_scene(pl,trays,yaw=-.2);s.assertFalse(A.col.certify(qm,yaw0))   # the yaw-0 command hits the wall on the way
        A.set_scene(pl,trays)
        with A.at_yaw(-.2):s.assertIsNone(A.correct(qm,q,max_step=.15))
        s.assertEqual(A._yaw,0.)
        src=inspect.getsource(N.Sol.pick_place)
        s.assertEqual(src.count('at_yaw(s.body_yaw())'),2)   # grasp correction and place correction
class FarTray(unittest.TestCase):
    def test_far_tray_sides(self):
        L,R,C=SF.TRAYS['tray_left'][1],SF.TRAYS['tray_right'][1],SF.TRAYS['tray_center'][1]
        self.assertTrue(N.far_tray('l',R));self.assertTrue(N.far_tray('r',L))
        self.assertFalse(N.far_tray('l',L));self.assertFalse(N.far_tray('r',R))
        self.assertFalse(N.far_tray('l',C));self.assertFalse(N.far_tray('r',C))
    def test_far_tray_skips_unyawed_search(self):
        calls=[];orig=N.plan_place
        def spy(A,q,tx,ty,top,fallback=None,tray_h=None):calls.append(round(float(ty),3));return None
        N.plan_place=spy
        try:
            A=ARM.Arm('l');n=np.array([-.0699,0,.9976]);pl=(n/np.linalg.norm(n),.1788/np.linalg.norm(n))
            N.plan_place_yaw(A,A.home,*SF.TRAYS['tray_right'],pl,list(SF.TRAYS.values()))
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
