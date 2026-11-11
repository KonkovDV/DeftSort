"""Unit tests of the core pieces: time budget, finger verdicts, tray detection, RGB-D pairing, IK and the sphere
models. Pure Python, no ROS needed."""
import os,sys,json,struct,tempfile,types,unittest
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,os.path.join(HERE,'..','..','..','tools'))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));import scene_fixture as SF
from team048_solution import arm,budget,grip,trays,sensing
from team048_solution.geometry import quaternion_transform
V5=os.path.join(HERE,'fixtures','v5_head_frame.npz')

class Budget(unittest.TestCase):
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

class Grip(unittest.TestCase):
    def test_force_not_used_finger_calibrated(s):
        s.assertIsNone(grip.GRASP_DF_MIN_N)          # F/T offsets 0.05-5.5 N empty (v7): not a decision signal
        s.assertEqual(grip.FINGER_DEV_HELD,.35);s.assertEqual(grip.FINGER_EFF_HELD,.67)
        s.assertTrue(.3<=grip.HAND_FORCE_RATIO<=.5)       # organiser #648: 0.3-0.5 instead of 1.0
        s.assertLess(grip.HAND_SPEED_RATIO,1.0)
    def test_finger_verdict_on_v7_frames(s):
        rim=[1,1,.1,.1,.2,0.];side=[0,0,0,0,0,0.]
        # real v7 'lifted' frames (position, effort)
        loaded=[([1.,.99,.78,.77,.2],[.06,.05,1.,1.,.06],rim),        # s101 Ecoforms (later released outside: arm not at tray)
                ([1.,.99,.79,.78,.19],[.05,.02,1.,1.,.03],rim),        # s103 Ecoforms (ended in tray_left)
                ([.99,.98,1.1,1.06,.22],[.74,.82,1.,1.,.37],rim)]      # s102 Threshold (lost during transport)
        empty=[([.98,.97,.12,.12,.23],[.12,.14,.07,.06,.04],rim),      # s101 Threshold
               ([.98,.97,.12,.12,.23],[.12,.14,.07,.06,.04],rim),      # s102 Utana
               ([0,0,0,.08,.03],[.01,0,.01,.33,.32],side),             # s101 mustard
               ([0,0,0,.08,.01],[.01,.02,.02,.33,.01],side),           # s102 Crayola
               ([0,0,0,0,0],[0,0,0,0,0],side),                         # s102 Hyaluronic
               ([0,0,0,.08,.01],[.01,.01,.01,.33,.01],side),           # s103 Perricone
               ([0,0,0,0,0],[0,0,0,0,0],side)]                         # s103 Hyaluronic
        for p,e,t in loaded:s.assertTrue(grip.finger_verdict(p,e,t))
        for p,e,t in empty:s.assertIs(grip.finger_verdict(p,e,t),False)
        s.assertIsNone(grip.finger_verdict([1,1,.9,.9,.2],[.1,.1,.1,.1,.1],rim))   # conflicting -> unknown
        s.assertIsNone(grip.finger_verdict(None,None,rim))
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

class Trays(unittest.TestCase):
    def test_slot_cycle_and_walls(s):
        offs=[trays.slot_offset(k,.03) for k in range(6)]
        s.assertEqual(offs[:3],[0.,.08,-.08]);s.assertEqual(offs[3:],offs[:3])
        big=trays.slot_offset(1,.12)          # wide dish: room = .16-.008-.01-.12 = .022
        s.assertAlmostEqual(big,.022,6)
        s.assertEqual(trays.slot_offset(1,.2),0.)   # wider than the tray half: centre only
        for k in range(3):
            for h in (0,.02,.05,.08,.1,.14):
                s.assertLessEqual(abs(trays.slot_offset(k,h))+h,trays.DOC_SIZE[1]/2-trays.TRAY_WALL+1e-9)
    @unittest.skipUnless(os.path.exists(V5),'v5 fixture missing')
    def test_chroma_on_real_v5_frame(s):
        f=np.load(V5);rgb=f['rgb']
        tray=rgb[340:358,300:350];table=rgb[260:300,450:600];s.assertGreater(table.size,0)
        s.assertGreater(trays.chroma_mask(tray,'green').mean(),.8)       # green tray (desaturated render)
        s.assertLess(trays.chroma_mask(table,'green').mean(),.02)        # neutral table
        s.assertLess(trays.chroma_mask(tray,'red').mean(),.02)
        s.assertLess(trays.chroma_mask(tray,'blue').mean(),.02)
    def _scene(s,true=(.36,.70),yaw=0.,H=120,W=160,x=(.05,.75),y=(.30,1.20)):
        yy,xx=np.mgrid[0:H,0:W]
        P=np.zeros((H,W,3));P[...,0]=x[0]+xx/W*(x[1]-x[0]);P[...,1]=y[0]+yy/H*(y[1]-y[0]);P[...,2]=-.18
        rgb=np.full((H,W,3),224,np.uint8)
        c,sn=np.cos(yaw),np.sin(yaw);dx=P[...,0]-true[0];dy=P[...,1]-true[1]
        u=dx*c+dy*sn;v=-dx*sn+dy*c
        m=(np.abs(u)<.11)&(np.abs(v)<.16)
        rgb[m]=(250,234,236)                  # red-dominant, desaturated like the v5/v7 renders
        P[m,2]+=.0                            # tray floor at table height
        return P,rgb,(np.array([0,0,1.]),.18)
    def test_detect_synthetic_full_tray(s):
        P,rgb,plane=s._scene()
        d=trays.detect(P,rgb,np.ones(P.shape[:2],bool),'tray_left','red',plane)
        s.assertIsNotNone(d);s.assertFalse(d['partial'])
        s.assertLess(np.linalg.norm(np.array(d['center'])-[.36,.70]),.012)
        s.assertAlmostEqual(d['size'][0],.22,delta=.015);s.assertAlmostEqual(d['size'][1],.32,delta=.015)
        s.assertGreater(abs(d['axis'][1]),.99)          # long axis along y
        s.assertIsNone(trays.detect(P,rgb,np.ones(P.shape[:2],bool),'tray_left','blue',plane))   # wrong colour
    def test_detect_measures_rotation_and_position_anywhere(s):
        P,rgb,plane=s._scene(true=(.42,.62),yaw=.3)      # not the simulator layout: found where it is
        d=trays.detect(P,rgb,np.ones(P.shape[:2],bool),'tray_left','red',plane)
        s.assertFalse(d['partial']);s.assertLess(np.linalg.norm(np.array(d['center'])-[.42,.62]),.015)
        ang=np.arctan2(d['axis'][1],d['axis'][0]);s.assertAlmostEqual(abs(ang),np.pi/2+.3,delta=.04)
    def test_detect_partial_when_cut_by_border(s):
        P,rgb,plane=s._scene(true=(.36,1.15))           # extends past the image border (y up to 1.20)
        d=trays.detect(P,rgb,np.ones(P.shape[:2],bool),'tray_left','red',plane)
        s.assertIsNotNone(d);s.assertTrue(d['partial'])  # centre unreliable -> caller re-aims, never uses it
    def test_no_nominal_positions_in_package(s):
        s.assertFalse(hasattr(trays,'NOMINAL'));s.assertFalse(hasattr(trays,'locate'))
        s.assertFalse(hasattr(arm,'TABLE_TOP'))
    def test_unknown_colour(s):
        with s.assertRaises(ValueError):trays.chroma_mask(np.zeros((2,2,3),np.uint8),'purple')

class Sensing(unittest.TestCase):
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

class Kinematics(unittest.TestCase):
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
        """Real data (run v5, tf2_echo pelvis camera_head_link): our URDF FK at head 0 agrees to < 5 mm."""
        T=quaternion_transform([0.105,0.011,0.648],[0.613,-0.613,0.353,-0.353])
        F=arm.fk_link('camera_head_link',{})
        s.assertLess(np.linalg.norm(F[:3,3]-T[:3,3]),.005);s.assertLess(np.abs(F[:3,:3]-T[:3,:3]).max(),.01)

class MeshSpheres(unittest.TestCase):
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

class V7Fixes(unittest.TestCase):
    def test_startup_charges_process_start_only(self):
        # node ready 3 s after import, sim clock first seen 200 s later at sim t=10: t0 = 10-3, not 10-203
        self.assertAlmostEqual(budget.start_time(10.,3.,0.),7.)
    def test_table_env_follows_measured_plane(self):
        n=np.array([-.0699,0,.9976]);d=.1788          # run v7 plane (pelvis frame)
        T=arm.table_env((n,d));nu=n/np.linalg.norm(n);du=d/np.linalg.norm(n)
        x0=min(sp.center[0]-sp.radius for sp in T);tangent=0
        for sp in T:
            sd=float(sp.center@nu+du)
            if abs(sd+sp.radius)<1e-6:tangent+=1        # top spheres: tangent to the measured plane
            else:                                        # front face: at the edge, below the top
                self.assertAlmostEqual(sp.center[0]-sp.radius,x0,6);self.assertLess(sd,-.9*sp.radius)
        self.assertGreater(tangent,len(T)//2)
        z=lambda x:-(d+n[0]*x)/n[2]
        self.assertGreater(z(.5)-SF.TABLE_TOP,.03)    # 3.5 cm above the flat model at x=0.5
    def test_tray_walls_are_obstacles(self):
        n=np.array([0,0,1.]);d=-SF.TABLE_TOP
        W=arm.tray_env([(.34,-.69)],(n,d))
        self.assertTrue(W);top=max(s.center[2]+s.radius for s in W)
        self.assertAlmostEqual(top,SF.TABLE_TOP+.07,6)
    def test_thumb_in_collision_envelope(self):
        A=arm.Arm('l');hm=arm.hand_mesh('l');self.assertIsNotNone(hm)
        tip=[s for s in A.col.spheres if s.link==arm.SIDE['l']['tip']]
        self.assertGreaterEqual(len(tip),4+len(hm['thumb_env_hand']))
        self.assertTrue(A.col.valid(A.home))
    def test_pinch_point_deeper_than_open_tip_midpoint(self):
        A=arm.Arm('l');hm=arm.hand_mesh('l')
        self.assertLess(hm['pad_gap_open'],A.hand['opposed_open'])   # 7.8 cm pads vs 10.5 cm tip estimate
        g=np.asarray(A.hand['grip_center_hand']);p=np.asarray(hm['pinch_point_hand'])
        self.assertLess(p[2],g[2]-.01)
    def test_slot_inner_first(self):
        self.assertLess(trays.slot_offset(1,.03,ty=.69),0);self.assertGreater(trays.slot_offset(1,.03,ty=-.69),0)

class GraspStudy(unittest.TestCase):
    """Offline grasp study evidence."""
    def test_dish_has_multi_angle_rim(self):
        from team048_solution.grasp import candidates,RIM_ANGLE_DEG
        rng=np.random.default_rng(1);t=rng.uniform(0,2*np.pi,2500);h=.07
        cl=np.c_[.4+.066*np.cos(t),.2+.066*np.sin(t),rng.uniform(-.18,-.18+h,2500)]
        f,C=candidates(cl,(np.array([0.,0.,1.]),.18),'dish',W_MAX=.095)
        rims=[c for c in C if c['kind']=='rim']
        self.assertEqual(len(rims),len(RIM_ANGLE_DEG))
        self.assertGreater(rims[0]['score'],rims[-1]['score'])   # near-robot preferred
    def test_box_topdown_when_side_too_wide(self):
        """Gelatin-like footprint: min caliper ~8.9 cm fails side (needs w+2cm <= W_MAX) but top fits W_MAX."""
        from team048_solution.grasp import candidates
        rng=np.random.default_rng(2);N=3000
        # axis-aligned box 8.9 x 6.0 x 3 cm on table z=-0.18
        u=rng.uniform(-.5,.5,(N,3))*[0.089,0.060,0];u[:,2]=rng.uniform(0,.03,N)
        face=rng.integers(0,5,N);u[face==0,2]=.03;u[face==1,0]=.0445;u[face==2,0]=-.0445
        cl=u+[0.40,0.10,-0.18]
        f,C=candidates(cl,(np.array([0.,0.,1.]),.18),'box',W_MAX=.095)
        kinds={c['kind'] for c in C}
        self.assertIn('top',kinds)
        # side may or may not exist on the short axis (6 cm + 2 cm margin = 8 cm <= 9.5)
        tops=[c for c in C if c['kind']=='top']
        self.assertTrue(any(c['width']<=.095 for c in tops))
    def test_razer_wide_still_no_candidate(self):
        """Razer min width 10 cm > W_MAX≈0.095: neither side nor top (study union ok=0)."""
        from team048_solution.grasp import candidates
        rng=np.random.default_rng(3);N=3000
        u=rng.uniform(-.5,.5,(N,3))*[0.100,0.120,0];u[:,2]=rng.uniform(0,.04,N)  # min caliper 10 cm (study)
        cl=u+[0.40,0.0,-0.18]
        f,C=candidates(cl,(np.array([0.,0.,1.]),.18),'electronics',W_MAX=.095)
        self.assertEqual(C,[])   # nothing fits the hand opening gate
