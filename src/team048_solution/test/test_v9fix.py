"""v9fix (fixes/V9_ANALYSIS_RU.md): Kaggle v9a sort3 s103 scored 0 -- the start->home move before the table was
measured jammed both hands in the table front; torso envelope missing; guard threshold without margin; early stop."""
import os,sys,json,types,unittest
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,HERE)
import test_node_stub as TS
from test_node_stub import N,Clock
from team048_solution import arm as ARM, grip as G, perception as PE
from test_compliance import cloud,V7
FX=json.load(open(os.path.join(HERE,'fixtures','v9a_sort3_103.json')))
TR={(t['t'],t['phase']):t for t in FX['trace']}
RIM=[1,1,.1,.1,.2,0.];SIDE=[0,0,0,0,0,0.];OPEN=[1,1,1,1,1,0.]
def v7_scene():
    f=np.load(V7);K=f['K']
    P=cloud(ARM.fk_link('camera_head_link',{'head_pitch_joint':float(f['head_pitch'])}),f['head_depth'].astype(float),K)
    valid=f['head_depth']>0;pl=PE.table_plane(P,valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7))
    Pt=cloud(ARM.fk_link('camera_body_front_link'),f['torso_depth'].astype(float),K)
    return pl,PE.near_edge(Pt,f['torso_depth']>0,pl)

class LoggedFailure(unittest.TestCase):
    """The fixture reproduces what the log shows (regression anchors)."""
    def test_done_early_with_nothing_handled(self):
        done=[l for l in FX['log'] if l.startswith('done at')];self.assertEqual(done,['done at 124.1 handled []'])
        self.assertEqual(sum('uncertified path home' in l for l in FX['log']),4)
    def test_both_arms_jammed_before_their_first_pick(self):
        for key,side in (((23.19,'start'),'l'),((93.02,'start'),'r')):
            t=TR[key];self.assertEqual(t['side'],side)
            self.assertGreater(np.linalg.norm(t['F']),250.)                     # 258 N / 938 N: hand blocked
            self.assertLess(t['tcp'][2],-.15)                                   # hand at table height, not home
            A=ARM.Arm(side);self.assertGreater(np.abs(np.array(t['q'])-A.home).max(),.5)

@unittest.skipUnless(os.path.exists(V7),'v7 fixture missing')
class StartPath(unittest.TestCase):
    def setUp(self):self.pl,self.edge=v7_scene();self.assertTrue(.2<self.edge<.3)
    def arm(self,side):
        A=ARM.Arm(side);A.set_scene(self.pl,[],(self.edge,ARM.TABLE_X_FAR));return A
    def test_start_posture_is_inside_the_margin(self):
        for side in 'lr':
            A=self.arm(side);q0=np.clip(np.zeros(7),A.chain.lower,A.chain.upper)
            self.assertFalse(A.col.valid(q0))                     # forearm vs hip sphere (real mesh gap 3 mm)
            ge,gs=A.col.gaps(q0);self.assertGreater(ge.min(),-.03);self.assertTrue(len(gs)==0 or gs.min()>0)
    def test_straight_start_to_home_crosses_the_table(self):
        """v9a root cause: the straight joint path start->home is NOT collision-free against the measured table."""
        for side in 'lr':
            A=self.arm(side);q0=np.clip(np.zeros(7),A.chain.lower,A.chain.upper)
            ok=A.col.certify_escape(q0,A.home);self.assertFalse(ok)
            B=ARM.Arm(side);self.assertTrue(B.col.certify_escape(q0,B.home) is False or True)   # no table: undecided
    def test_certified_path_home_from_start(self):
        for side in 'lr':
            A=self.arm(side);q0=np.clip(np.zeros(7),A.chain.lower,A.chain.upper)
            p=A.path_home(q0);self.assertIsNotNone(p,side);self.assertTrue(np.allclose(p[-1],A.home))
            self.assertTrue(A.col.certify_escape(q0,p[0]))
            for a,b in zip(p[:-1],p[1:]):self.assertTrue(A.col.certify(a,b))
            # independent check: the TCP never enters the box in front of / below the measured table top
            Q=[q0]+list(p)
            for a,b in zip(Q[:-1],Q[1:]):
                for k in np.linspace(0,1,25):
                    x,y,z=A.tcp(a+(b-a)*k)[:3,3]
                    self.assertFalse(x>self.edge-.02 and z<ARM.plane_height(self.pl,x,y)+.02,(side,k,x,z))
    def test_escape_refuses_going_deeper(self):
        A=self.arm('l');q0=np.clip(np.zeros(7),A.chain.lower,A.chain.upper)
        w=q0.copy();w[1]=A.chain.lower[1]          # roll inward: toward the hip
        self.assertFalse(A.col.certify_escape(q0,w))
    def test_escape_equals_certify_when_start_valid(self):
        A=self.arm('l');a=A.home;b=A.home.copy();b[0]+=.3
        self.assertEqual(A.col.certify_escape(a,b),A.col.certify(a,b))

class TableModel(unittest.TestCase):
    def test_nothing_closer_than_edge_and_front_is_solid(self):
        pl=(np.array([0,0,1.]),.18);T=ARM.table_env(pl,(.24,1.))
        C=np.array([s.center for s in T]);R=np.array([s.radius for s in T])
        self.assertAlmostEqual((C[:,0]-R).min(),.24,6)
        front=(C[:,0]-R<.24+1e-6)                                  # spheres touching the edge plane
        self.assertLessEqual((C[front,2]-R[front]).min(),-.18-ARM.TABLE_FRONT_DEPTH+.03)
        # top surface covered right from the edge: a point 1 cm behind the edge, 1 cm below the top is inside
        p=np.array([.25,0.,-.19]);self.assertTrue(np.any(np.linalg.norm(C-p,axis=1)<R))

class Torso(unittest.TestCase):
    def test_torso_envelope_rejects_v9a_contact_pose_and_keeps_home(self):
        A=ARM.Arm('l');self.assertTrue(A.col.valid(A.home))
        q=np.clip(TR[(64.48,'pre')]['q'],A.chain.lower,A.chain.upper)   # Ecoforms pre-grasp, wrist |F| 581 N
        self.assertFalse(A.col.valid(q))
        self.assertGreater(len(ARM.torso_spheres('l')),20);self.assertGreater(len(ARM.torso_spheres('r')),20)
    def test_torso_spheres_clear_of_shoulder_mount(self):
        for side in 'lr':
            sh=ARM.fk_link(ARM.J[f'shoulder_pitch_{side}_joint']['child'])[:3,3]
            for s in ARM.torso_spheres(side):self.assertGreater(np.linalg.norm(s.center-sh),ARM.SHOULDER_LOCAL+s.radius)

class Guard(unittest.TestCase):
    def test_v9a_free_motion_does_not_trip(self):
        for key in ((77.63,'pre'),(77.66,'guard')):     # Utana 2nd attempt: thumb 1.094/1.103, wrist |F| 3 N
            t=TR[key];self.assertLess(np.linalg.norm(t['F']),5.)
            self.assertEqual(G.approach_deviation(t['fing'],OPEN,RIM)[1],[])
    def test_threshold_between_noise_and_contacts(self):
        self.assertTrue(.103<G.GUARD_DEV<.225)

class EndOfRun(unittest.TestCase):
    def test_retry_round_then_reason_logged(self):
        sol=N.Sol();sol.arms={k:ARM.Arm(k) for k in 'lr'}
        sol.task={'task':'sort2','categories':{'box':'tray_left'},'objects':{'a':'box'},'trays':{}}
        Clock.t=50.;logs=[]
        sol.log=lambda *a:logs.append(' '.join(str(x) for x in a))
        sol.survey=lambda *a,**k:Clock.t;sol.go_home=lambda k:None;sol.retreat=lambda k:None;sol.js={'x':0.}
        sol.robot_description='<robot/>';sol.trays={'tray_left':{'center':[.3,.6],'axis':[0,1],'size':[.2,.3],'height':.07}}
        d=types.SimpleNamespace(pixels=np.zeros((0,2),int))
        sol.observe=lambda nb=None:(None,None,None,[d])
        sol.identify=lambda *a:[(d,'a','box',.9,None,.1)]
        tries=[]
        def pp(*a):tries.append(1);Clock.t+=1.;return False
        sol.pick_place=pp
        import time;from unittest import mock
        with mock.patch.object(N,'WALL_LAUNCH',time.monotonic()):sol.run()   # suite wall time since import must not count as start-up
        self.assertEqual(len(tries),2*N.FAILS_MAX)                     # FAILS_MAX, reset once, FAILS_MAX again
        self.assertTrue(any(l.startswith('retry round 1') for l in logs))
        done=[l for l in logs if l.startswith('done at')][0];self.assertIn('reason: nothing pickable',done)
if __name__=='__main__':unittest.main()
