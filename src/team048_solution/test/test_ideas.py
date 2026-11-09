"""Unit tests for the competitor-derived ideas (COMPETITORS_RU.md I1-I10). Pure Python, no ROS needed."""
import os,sys,json,struct,tempfile,types,unittest
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,os.path.join(HERE,'..','..','..','tools'))
from team048_solution import arm,budget,grip,trays,sensing
from team048_solution.geometry import quaternion_transform
V5=os.path.join(HERE,'fixtures','v5_head_frame.npz')

class Budget(unittest.TestCase):          # I2 (time budget)
    def test_remaining_and_gate(s):
        b=budget.TimeBudget(270,t0=10.)
        s.assertAlmostEqual(b.remaining(40.),240.)
        s.assertTrue(b.can_start(40.))
        s.assertFalse(b.can_start(10+270-budget.FIRST_CYCLE_S-budget.SAFETY_S+1))
    def test_planned_duration_gate(s):
        b=budget.TimeBudget(270,t0=0.)
        now=270-budget.SAFETY_S-budget.PLACE_TAIL_S-10
        s.assertTrue(b.can_start(now,planned=9.9));s.assertFalse(b.can_start(now,planned=10.1))
    def test_cycle_estimate_rises_fast_decays_slow(s):
        b=budget.TimeBudget(405,0.,first_cycle=30.)
        b.record_cycle(50.);s.assertGreaterEqual(b.cycle,50.)
        b.record_cycle(20.);s.assertTrue(20.<b.cycle<50.)
        b.record_cycle(float('nan'));b.record_cycle(-1);s.assertEqual(len(b.cycles),2)
    def test_start_time_counts_startup(s):
        s.assertAlmostEqual(budget.start_time(100.,5.,2.),97.)
        s.assertAlmostEqual(budget.start_time(100.,2.,5.),100.)   # clock skew never moves t0 later
    def test_limits_from_scoring(s):
        s.assertEqual(budget.LIMITS,{'sort2':270.,'sort3':405.})
    def test_invalid(s):
        with s.assertRaises(ValueError):budget.TimeBudget(0,0)

class Grip(unittest.TestCase):            # I1 plumbing + I4
    def test_uncalibrated_is_unknown(s):
        m=grip.GraspMonitor();s.assertFalse(m.calibrated);m.baseline([0,0,0])
        s.assertIsNone(m.verdict([0,0,-5.],[.3]*6,[.9]*6))
    def test_constants_are_todo(s):
        s.assertIsNone(grip.GRASP_DF_MIN_N);s.assertIsNone(grip.FINGER_BLOCKED_MIN);s.assertIsNone(grip.EFFORT_MIN)
        s.assertEqual(grip.HAND_FORCE_RATIO,1.0);s.assertEqual(grip.HAND_SPEED_RATIO,1.0)
    def test_force_vote(s):
        m=grip.GraspMonitor(df_min=1.);m.baseline([0,0,0])
        s.assertTrue(m.verdict([0,0,-2.]));s.assertFalse(m.verdict([0,0,-.2]))
        m.baseline(None);s.assertIsNone(m.verdict([0,0,-2.]))
    def test_finger_vote(s):
        m=grip.GraspMonitor(blocked_min=.2,effort_min=.3)
        s.assertTrue(m.verdict(hand_pos=[.4,.4,.4,.4,.5,0],hand_eff=[.5]*6))
        s.assertFalse(m.verdict(hand_pos=[0,0,0,0,0,0],hand_eff=[.5]*6))
        s.assertFalse(m.verdict(hand_pos=[.4]*6,hand_eff=[.0]*6))
    def test_settled(s):
        s.assertFalse(grip.fingers_settled([[0]*6]*2))
        s.assertTrue(grip.fingers_settled([[.5]*6,[0]*6,[.01]*6,[0]*6]))
        s.assertFalse(grip.fingers_settled([[0]*6,[0]*6,[.05]+[0]*5]))
    def test_force_vec(s):
        w=types.SimpleNamespace(force=types.SimpleNamespace(x=1.,y=2.,z=3.))
        s.assertTrue(np.allclose(grip.force_vec(w),[1,2,3]))

class Trays(unittest.TestCase):           # I5 + I6
    def test_slot_cycle_and_walls(s):
        offs=[trays.slot_offset(k,.03) for k in range(6)]
        s.assertEqual(offs[:3],[0.,.08,-.08]);s.assertEqual(offs[3:],offs[:3])
        big=trays.slot_offset(1,.12)          # wide dish: room = .16-.008-.01-.12 = .022
        s.assertAlmostEqual(big,.022,6)
        s.assertEqual(trays.slot_offset(1,.2),0.)   # wider than the tray half: centre only
        for k in range(3):
            for h in (0,.02,.05,.08,.1,.14):
                s.assertLessEqual(abs(trays.slot_offset(k,h))+h,trays.TRAY_SIZE[1]/2-trays.TRAY_WALL+1e-9)
    @unittest.skipUnless(os.path.exists(V5),'v5 fixture missing')
    def test_chroma_on_real_v5_frame(s):
        f=np.load(V5);rgb=f['rgb'];r0=0
        tray=rgb[340:358,300:350];table=rgb[260:300,450:600];s.assertGreater(table.size,0)
        s.assertGreater(trays.chroma_mask(tray,'green').mean(),.8)       # green tray (desaturated render)
        s.assertLess(trays.chroma_mask(table,'green').mean(),.02)        # neutral table
        s.assertLess(trays.chroma_mask(tray,'red').mean(),.02)
        s.assertLess(trays.chroma_mask(tray,'blue').mean(),.02)
    def test_locate_synthetic(s):
        H,W=60,80;yy,xx=np.mgrid[0:H,0:W]
        P=np.zeros((H,W,3));P[...,0]=.2+xx/W*.5;P[...,1]=.4+yy/H*.6;P[...,2]=-.18
        rgb=np.full((H,W,3),224,np.uint8)
        true=(.36,.70);m=(np.abs(P[...,0]-true[0])<.11)&(np.abs(P[...,1]-true[1])<.16)
        rgb[m]=(250,234,236)                  # red-dominant, desaturated like the v5 render
        plane=(np.array([0,0,1.]),.18)
        c,src=trays.locate(P,rgb,np.ones((H,W),bool),'tray_left','red',plane)
        s.assertEqual(src,'camera');s.assertLess(np.linalg.norm(c-true),.015)
        c2,src2=trays.locate(P,rgb,np.ones((H,W),bool),'tray_left','blue',plane)
        s.assertEqual(src2,'nominal');s.assertTrue(np.allclose(c2,trays.NOMINAL['tray_left']))
    def test_locate_rejects_partial_and_far(s):
        H,W=60,80;yy,xx=np.mgrid[0:H,0:W]
        P=np.zeros((H,W,3));P[...,0]=.2+xx/W*.5;P[...,1]=.4+yy/H*.6;P[...,2]=-.18
        rgb=np.full((H,W,3),224,np.uint8)
        m=(np.abs(P[...,0]-.34)<.11)&(P[...,1]>.80)&(P[...,1]<.85)   # thin strip only: y extent < 60 % of tray
        rgb[m]=(250,234,236)
        c,src=trays.locate(P,rgb,np.ones((H,W),bool),'tray_left','red',None,min_px=10)
        s.assertAlmostEqual(c[1],.69)      # y not refined from a partial view
    def test_unknown_colour(s):
        with s.assertRaises(ValueError):trays.chroma_mask(np.zeros((2,2,3),np.uint8),'purple')

class Sensing(unittest.TestCase):         # I8
    def msg(s,t):
        return types.SimpleNamespace(header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=int(t),nanosec=int(round((t-int(t))*1e9)))),t=t)
    def test_pair_same_stamp_and_freshness(s):
        b=sensing.PairBuffer()
        for t in (1.,1.05,1.1):b.add_rgb(s.msg(t));b.add_depth(s.msg(t))
        r,d=b.pair(1.02);s.assertEqual((r.t,d.t),(1.1,1.1))
        s.assertIsNone(b.pair(1.2))
    def test_mismatched_stamps(s):
        b=sensing.PairBuffer();b.add_rgb(s.msg(1.));b.add_depth(s.msg(1.05))
        s.assertIsNone(b.pair(0.,tol=.02));s.assertIsNotNone(b.pair(0.,tol=.06))

class Kinematics(unittest.TestCase):      # I7 + I9 + I10
    @classmethod
    def setUpClass(c):c.A=arm.Arm('l')
    def test_null_space_keeps_accuracy_and_prefers_home(s):
        A=s.A;rng=np.random.default_rng(3);d0=d1=0.;n=0
        for _ in range(12):
            q=np.clip(A.home+rng.normal(0,.35,7),A.chain.lower,A.chain.upper);T=A.chain.fk(q)
            seed=np.clip(q+rng.normal(0,.3,7),A.chain.lower,A.chain.upper)
            try:
                qa=A.chain.ik(T,seed,max_iter=300);qb=A.chain.ik(T,seed,max_iter=300,q_pref=A.home,w_null=.1)
            except ValueError:continue
            for qq in (qa,qb):s.assertLess(np.linalg.norm(A.chain.fk(qq)[:3,3]-T[:3,3]),.0021)
            d0+=np.linalg.norm(qa-A.home);d1+=np.linalg.norm(qb-A.home);n+=1
        s.assertGreaterEqual(n,6);s.assertLessEqual(d1,d0+1e-6)
    def test_null_weight_validated(s):
        with s.assertRaises(ValueError):s.A.chain.ik(s.A.chain.fk(s.A.home),s.A.home,w_null=2.)
    def test_lift_straight_up(s):
        qu=s.A.lift(s.A.home,.08);s.assertIsNotNone(qu)
        T0=s.A.tcp(s.A.home);T1=s.A.tcp(qu)
        s.assertTrue(np.allclose(T1[:3,3]-T0[:3,3],[0,0,.08],atol=.003));s.assertTrue(np.allclose(T1[:3,:3],T0[:3,:3],atol=.02))
    def test_correct_reduces_tcp_error(s):
        A=s.A;q_goal=A.home.copy();q_meas=q_goal.copy();q_meas[0]+=.04;q_meas[3]+=.03   # PD undershoot
        e0=A.position_error(q_meas,A.tcp(q_goal));s.assertGreater(e0,.01)
        qc=A.correct(q_meas,q_goal);s.assertIsNotNone(qc)
        q_after=q_meas+(qc-q_goal)                       # same undershoot applied to the corrected command
        s.assertLess(A.position_error(q_after,A.tcp(q_goal)),e0*.5)
    def test_head_fk_matches_sim_tf_v5(s):
        """Real data (Kaggle v5, tf2_echo pelvis camera_head_link): our URDF FK at head 0 agrees to < 5 mm."""
        T=quaternion_transform([0.105,0.011,0.648],[0.613,-0.613,0.353,-0.353])
        F=arm.fk_link('camera_head_link',{})
        s.assertLess(np.linalg.norm(F[:3,3]-T[:3,3]),.005);s.assertLess(np.abs(F[:3,:3]-T[:3,:3]).max(),.01)

class MeshSpheres(unittest.TestCase):     # I3 plumbing
    def test_disabled_by_default(s):
        s.assertFalse(arm.USE_MESH_SPHERES)
    def test_loader_requires_validation(s):
        with tempfile.TemporaryDirectory() as d:
            p=os.path.join(d,'s.json');json.dump(dict(validated=False,spheres=[dict(link='x',center=[0,0,0],radius=.1)]),open(p,'w'))
            s.assertIsNone(arm.mesh_spheres(p))
            json.dump(dict(validated=True,spheres=[dict(link='x',center=[0,0,0],radius=.1)]),open(p,'w'))
            S=arm.mesh_spheres(p);s.assertEqual(len(S),1);s.assertEqual(S[0].link,'x')
            json.dump(dict(validated=True,spheres=[dict(link='x',center=[0,0],radius=.1)]),open(p,'w'))
            s.assertIsNone(arm.mesh_spheres(p))
            s.assertIsNone(arm.mesh_spheres(os.path.join(d,'missing.json')))
    def test_cover_and_stl(s):
        import fit_spheres as FS
        rng=np.random.default_rng(0);P=rng.uniform([-.03,-.03,0],[.03,.03,.25],(3000,3))   # forearm-like box
        S=FS.cover(P,r_max=.05,pad=.005)
        C=np.array([c for c,_ in S]);R=np.array([r for _,r in S])
        d=np.linalg.norm(P[:,None]-C[None],axis=2)-R[None]
        s.assertTrue(np.all(d.min(axis=1)<=1e-9));s.assertTrue(np.all(R<=.055+1e-9))
        with tempfile.TemporaryDirectory() as dd:
            p=os.path.join(dd,'t.STL');tri=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
            open(p,'wb').write(b'\0'*80+struct.pack('<I',1)+np.zeros(3,np.float32).tobytes()+tri.tobytes()+b'\0\0')
            s.assertTrue(np.allclose(FS.read_stl(p),tri))

class Rules(unittest.TestCase):           # constraints: no debug topics, 0.8 cap kept
    def test_no_debug_topics_and_cap(s):
        src=open(os.path.join(HERE,'..','team048_solution','node.py')).read()
        import ast
        lits=[n.value for n in ast.walk(ast.parse(src)) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('/')]
        for t in ('/sim/task/check','/sim/objects/randomize','/sim/objects/set','/sim/task/result'):
            s.assertNotIn(t,lits,msg='debug topic used')
        s.assertIn('min(.8,',src)
if __name__=='__main__':unittest.main()

class V6Fixes(unittest.TestCase):
    def test_rim_grip_below_rim_top(self):
        from team048_solution.grasp import candidates,RIM_DEPTH
        rng=np.random.default_rng(0);t=rng.uniform(0,2*np.pi,2000);h=.053
        cl=np.c_[.4+.055*np.cos(t),.2+.055*np.sin(t),rng.uniform(-.18,-.18+h,2000)]
        f,C=candidates(cl,(np.array([0,0,1.]),.18),'dish')
        g=[c for c in C if c['kind']=='rim'][0]
        self.assertAlmostEqual(g['z'],f['height']-min(RIM_DEPTH,.3*f['height']),6);self.assertLess(g['z'],f['height'])
