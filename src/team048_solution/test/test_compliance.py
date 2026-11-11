"""Organiser answers #595/#613/#625/#636/#638/#645/#648 (see COMPLIANCE.md): scene measured by cameras, hand force/speed/
thumb/close-past, end-of-run policy, /robot_description, debug topics."""
import os,sys,types,unittest,glob
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE)
import test_node_stub as TS                      # installs the ROS stubs
import scene_fixture as SF
from team048_solution import node as N, arm as ARM, trays as TR, grip as G, perception as PE
PKG=os.path.join(HERE,'..','team048_solution')
V7=os.path.join(HERE,'fixtures','v7_s102_start_frames.npz')

def cloud(T,dep,K):
    v,u=np.mgrid[0:dep.shape[0],0:dep.shape[1]];Z=dep/1000.;X=(u-K[2])*Z/K[0];Y=(v-K[5])*Z/K[4]
    return (np.stack([X,Y,Z],-1).reshape(-1,3)@T[:3,:3].T+T[:3,3]).reshape(dep.shape+(3,))

class NoSceneConstants(unittest.TestCase):
    def test_no_scene_yaml_positions_in_package(self):
        """#645: the simulator layout numbers (tray xy 0.34/0.69/0.53, pelvis height 0.88) are not numeric literals in
        the solution code (tests use scene_fixture.py, outside the package). Checked on the AST, not on comments."""
        import ast
        bad={.34,.69,.53,.88}
        for f in glob.glob(os.path.join(PKG,'*.py')):
            for node in ast.walk(ast.parse(open(f).read())):
                if isinstance(node,ast.Constant) and isinstance(node.value,float) and any(abs(node.value-b)<1e-9 for b in bad):
                    self.fail(f'{os.path.basename(f)}:{node.lineno}: {node.value}')
    def test_scene_fixture_not_imported_by_package(self):
        for f in glob.glob(os.path.join(PKG,'*.py')):
            self.assertNotIn('scene_fixture',open(f).read(),f)

class Table(unittest.TestCase):
    def test_plane_band_rejects_floor_and_accepts_table(self):
        self.assertTrue(ARM.plane_plausible(SF.plane()))
        self.assertFalse(ARM.plane_plausible((np.array([0,0,1.]),.88)))       # floor
        self.assertFalse(ARM.plane_plausible((np.array([.5,0,.866]),.18)))    # not horizontal
    def test_table_obstacle_starts_at_measured_edge(self):
        # no part of the table model is closer to the robot than the measured edge
        T=ARM.table_env(SF.plane(),(.24,1.))
        self.assertAlmostEqual(min(s.center[0]-s.radius for s in T),.24,6)
        T=ARM.table_env(SF.plane(),(.0,1.))                                     # never inside the body region
        self.assertAlmostEqual(min(s.center[0]-s.radius for s in T),ARM.TABLE_X_NEAR_MIN,6)
    @unittest.skipUnless(os.path.exists(V7),'v7 fixture missing')
    def test_v7_near_edge_torso_seen_head_not(self):
        f=np.load(V7);K=f['K']
        P=cloud(ARM.fk_link('camera_head_link',{'head_pitch_joint':float(f['head_pitch'])}),f['head_depth'].astype(float),K)
        valid=f['head_depth']>0
        pl=PE.table_plane(P,valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7))
        self.assertTrue(ARM.plane_plausible(pl))
        self.assertIsNone(PE.near_edge(P,valid,pl))          # head view does not contain the edge -> no guess
        Pt=cloud(ARM.fk_link('camera_body_front_link'),f['torso_depth'].astype(float),K)
        e=PE.near_edge(Pt,f['torso_depth']>0,pl)
        self.assertIsNotNone(e);self.assertTrue(.15<e<.30)
    @unittest.skipUnless(os.path.exists(V7),'v7 fixture missing')
    def test_v7_trays_center_full_sides_partial(self):
        f=np.load(V7);K=f['K']
        P=cloud(ARM.fk_link('camera_head_link',{'head_pitch_joint':float(f['head_pitch'])}),f['head_depth'].astype(float),K)
        valid=f['head_depth']>0;pl=PE.table_plane(P,valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7))
        c=TR.detect(P,f['head_rgb'],valid,'tray_center','green',pl)
        self.assertFalse(c['partial']);self.assertAlmostEqual(c['size'][0],.22,delta=.02);self.assertAlmostEqual(c['size'][1],.32,delta=.02)
        self.assertAlmostEqual(c['height'],.07,delta=.01)
        for n,col in (('tray_left','red'),('tray_right','blue')):
            d=TR.detect(P,f['head_rgb'],valid,n,col,pl)
            self.assertTrue(d is None or d['partial'],n)       # cut by the image border: never used as a pose

class Survey(unittest.TestCase):
    def mk(self):
        s=N.Sol();s.task={'task':'sort3','categories':{'dish':'tray_left','box':'tray_right','jar':'tray_center'},
                         'trays':{'tray_left':'red','tray_right':'blue','tray_center':'green'},'objects':{}}
        s.js={'head_pitch_joint':.4,'head_yaw_joint':0.};return s
    def test_missing_tray_never_falls_back(self):
        s=self.mk();looks=[]
        s.look=lambda pitch=.4,yaw=None,**k:(looks.append(yaw),TS.Clock.t)[1]
        s.capture=lambda *a,**k:(np.zeros((4,4,3)),np.zeros((4,4,3),np.uint8),np.ones((4,4),bool))
        s.measure_plane=lambda P,v:SF.plane()
        full=dict(center=(.5,0.),size=(.22,.32),axis=(0.,1.),height=.07,partial=False,centroid=(.5,0.),px=5000)
        part=dict(center=(.3,.6),size=(.1,.2),axis=(0.,1.),height=.07,partial=True,centroid=(.3,.62),px=900)
        seq=iter([{'tray_center':full,'tray_left':part,'tray_right':None}]+[{'tray_left':part,'tray_right':None}]*10)
        def loc(P,rgb,valid,plane,refine_only=False):
            out=next(seq)
            for n,d in out.items():
                if d is not None and not d['partial']:s.trays[n]=d
            return out
        s.locate_trays=loc;s.measure_near_edge=lambda t:None
        s.survey()
        self.assertIn('tray_center',s.trays);self.assertNotIn('tray_left',s.trays);self.assertNotIn('tray_right',s.trays)
        self.assertEqual(looks[:3],list(N.HEAD_SCAN_YAWS));self.assertEqual(looks[-1],0.)   # scan, re-aim, back to 0
        hy=ARM.fk_link('head_yaw_link')[:3,3]
        self.assertTrue(any(y is not None and abs(y-np.arctan2(.62-hy[1],.3-hy[0]))<1e-6 for y in looks))   # re-aimed
    def test_subscriptions(self):
        s=self.mk()
        for t in ('/robot_description','/cam_torso/depth/image_raw','/cam_torso/depth/camera_info','/cam_head/depth/camera_info'):
            self.assertIn(t,s.subs)
        self.assertFalse(any(x in t for t in list(s.subs)+list(s.pubs) for x in ('/sim/objects/set','/sim/objects/randomize','/sim/task/check')))
    def test_robot_description_bundled_is_accepted(self):
        x=open(ARM.URDF).read();ok,why=ARM.load_urdf(x);self.assertTrue(ok,why)
        ok,why=ARM.load_urdf('<robot name="x"><link name="pelvis"/></robot>');self.assertFalse(ok)
        self.assertIn('left_tcp_link',open(ARM.URDF).read());A=ARM.Arm('l');self.assertTrue(A.col.valid(A.home))

class Grasp648(unittest.TestCase):
    def test_defaults(self):
        self.assertTrue(.3<=G.HAND_FORCE_RATIO<=.5);self.assertLess(G.HAND_SPEED_RATIO,1.);self.assertGreater(G.CLOSE_PAST,0)
        v=G.force_vector();self.assertEqual(len(v),6);self.assertEqual(v[5],G.HAND_FORCE_THUMB_ROT)
        self.assertAlmostEqual(G.finger_stroke_time(.5),1.6)                  # INTERFACES.md: 0.5 -> 1.6 s
    def test_effort_threshold_rescaled(self):
        self.assertAlmostEqual(G.effort_held_threshold(1.),.667,3)             # v7 calibration (ratio 1.0)
        self.assertGreater(G.effort_held_threshold(.4),.9)                     # unloaded effort scales 1/r
        self.assertAlmostEqual(G.effort_held_threshold(.2),1.,6)               # saturated: effort vote useless
    def test_contact_target_and_verdicts_on_v7_frames(self):
        rim=[1,1,.1,.1,.2,0.]
        pc=[1.,.99,.78,.77,.2,0.]                                               # v7 s103 Ecoforms contact (loaded)
        self.assertFalse(G.contact_empty(pc,rim))
        t2=G.contact_target(pc,rim,.15);self.assertAlmostEqual(t2[2],.63);self.assertAlmostEqual(t2[3],.62)
        self.assertEqual(t2[4],.2);self.assertEqual(t2[:2],[1.,1.])            # thumb bend at target, open fingers kept
        self.assertTrue(G.past_verdict(pc,t2,rim,.15))                          # still blocked at contact
        self.assertFalse(G.past_verdict([1,.99,.63,.62,.2,0],t2,rim,.15))       # reached target2: object gone
        self.assertTrue(G.contact_empty([.98,.97,.12,.12,.23,0],rim))           # v7 s101 Threshold: closed on nothing
    def test_grasp_close_sequence(self):
        s=N.Sol();sent=[];s.hand_set=lambda side,v:sent.append(list(v))
        s.wait_fingers=lambda side,**k:True
        s.hstate={'l':types.SimpleNamespace(position=[1.,.99,.78,.77,.2,0.],effort=[0]*6,velocity=[0]*6)}
        r=s.grasp_close('l','rim')
        self.assertEqual(sent[0],[1,1,.1,.1,.2,0.]);self.assertEqual(len(sent),2)   # phase 1 then phase 2
        self.assertAlmostEqual(sent[1][2],.78-G.CLOSE_PAST);self.assertFalse(r['empty'])
        s.hstate['l']=types.SimpleNamespace(position=[0.,0.,0.,0.,.01,0.],effort=[0]*6,velocity=[0]*6);sent.clear()
        r=s.grasp_close('l','side');self.assertTrue(r['empty']);self.assertEqual(len(sent),1)   # no phase 2, no lift
    def test_thumb_first(self):
        s=N.Sol();s.hstate={'l':types.SimpleNamespace(position=[1,1,1,1,1,0.01])}
        self.assertTrue(s.wait_thumb('l',tmax=.1))
        s.hstate['l']=types.SimpleNamespace(position=[1,1,1,1,1,.6]);s.sleep=lambda dt:None
        import time as _t
        orig=_t.sleep
        try:
            N.time.sleep=lambda dt:setattr(TS.Clock,'t',TS.Clock.t+dt)
            self.assertFalse(s.wait_thumb('l',tmax=.2))
        finally:N.time.sleep=orig

class EndOfRun595(unittest.TestCase):
    def test_place_time_positive_and_includes_yaw(self):
        A=ARM.Arm('l');A.set_scene(SF.plane(),list(SF.TRAYS.values()))
        q=A.home;place=(q+.3,None,None)
        t0=N.place_time(A,q,place,0.);t1=N.place_time(A,q,place,1.)
        self.assertGreater(t0,G.finger_stroke_time());self.assertAlmostEqual(t1-t0,2/.8+1.,6)
    def test_no_lift_when_place_does_not_fit(self):
        """A lifted object outside a tray at the limit is -10, untouched 0: with time for the grasp but not for the
        transport, the hand opens at the object and the arm retreats without lifting."""
        from team048_solution import budget as B
        s=N.Sol();s.arms={k:ARM.Arm(k) for k in 'lr'};s.hand_settings_step()
        for A in s.arms.values():A.set_scene(SF.plane(),list(SF.TRAYS.values()))
        s.trays=dict(SF.TRAYS);moves=[];hand=[]
        s.move_arm=lambda side,q,**k:moves.append(np.asarray(q).copy()) or True
        s.hand_set=lambda side,v:hand.append(list(v))
        s.wait_fingers=lambda side,**k:True;s.wait_thumb=lambda side,**k:True;s.sleep=lambda dt:None
        s.retreat=lambda side:moves.append('retreat');s.trace=lambda *a,**k:None;s.position_error=None
        s.hstate={'l':types.SimpleNamespace(position=[.8,.8,.8,.8,.5,0.],effort=[1]*6,velocity=[0]*6),
                  'r':types.SimpleNamespace(position=[.8,.8,.8,.8,.5,0.],effort=[1]*6,velocity=[0]*6)}
        A=s.arms['l'];s.js=dict(zip(A.chain.names,A.home));s.js.update(zip(s.arms['r'].chain.names,s.arms['r'].home))
        rng=np.random.default_rng(0);cl=np.c_[rng.uniform(.33,.39,400),rng.uniform(.17,.23,400),rng.uniform(-.18,-.08,400)]
        captured={}
        orig_pp=N.plan_place_yaw
        def ppy(*a,**k):
            r=orig_pp(*a,**k);captured['r']=r;return r
        N.plan_place_yaw=ppy
        try:
            # budget: enough for can_start (pick duration) is checked first; make the place check fail by shrinking
            # remaining right after the grasp
            s.budget=B.TimeBudget(270,t0=TS.Clock.t-0.)
            orig_close=s.grasp_close
            def close_and_age(side,kind):
                r=orig_close(side,kind);TS.Clock.t+=268.;return r
            s.grasp_close=close_and_age
            s.q_now=lambda side:next((np.asarray(m) for m in reversed(moves) if not isinstance(m,str)),np.asarray(s.arms[side].home))
            N.Sol.position_error=None
            r=s.pick_place('004_sugar_box','box',cl,SF.plane(),[],'tray_left')
        finally:N.plan_place_yaw=orig_pp;TS.Clock.t=100.
        self.assertIsNone(r)                                    # stopped for lack of time
        self.assertTrue(any(isinstance(m,str) and m=='retreat' for m in moves))
        self.assertEqual(hand[-1][:5],[1,1,1,1,1])             # opened at the object

class TimeLimit595(unittest.TestCase):
    def test_launch_age_counts_process_start(self):
        from team048_solution import budget as B
        a=B.launch_age_s();self.assertIsNotNone(a);self.assertGreaterEqual(a,0.)
        self.assertLessEqual(N.WALL_LAUNCH,N.WALL_IMPORT)            # start-up before import is charged, never less
        self.assertAlmostEqual(B.start_time(50.,10.,4.),44.)          # 6 s wall of start-up at RTF<=1 -> 6 s sim
    def test_run_sh_uses_sim_time(self):
        self.assertIn('use_sim_time:=true',open(os.path.join(HERE,'..','..','..','run.sh')).read())
