"""Grasp/time audit: place_time calibrated on the v8 Perricone TRACE (the old
model under-estimated the closed->released chain 6.6 s vs 12.0 s measured and left the lift out), light-first order."""
import os,sys,json,types,unittest,time
from unittest import mock
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,HERE)
import test_node_stub as TS  # noqa: F401  (installs the ROS stubs)
from test_node_stub import N,Clock
from team048_solution import arm as ARM, grip as G
P8={x['phase']:x for x in json.load(open(os.path.join(HERE,'fixtures','v8_perricone_place.json')))['phases']}
GT=json.load(open(os.path.join(HERE,'fixtures','v7v8_grasp_traces.json')))['attempts']
dt=lambda a,b:P8[b]['t']-P8[a]['t']

class PlaceTime(unittest.TestCase):
    def test_each_logged_move_within_model(self):
        self.assertGreaterEqual(N.move_time(P8['closed']['q'],P8['lifted']['q'])+.3,dt('closed','lifted'))
        for a,b in (('lifted','carry'),('carry','over_tray')):
            self.assertGreaterEqual(N.move_time(P8[a]['q'],P8[b]['q']),dt(a,b),(a,b))
            self.assertLess(N.move_time(P8[a]['q'],P8[b]['q'],settle=.3),dt(a,b))      # the old 0.3 s overhead did not
        self.assertGreaterEqual(G.finger_stroke_time()+.5,dt('over_tray','released'))
    def test_chain_covers_measured_and_includes_lift(self):
        A=ARM.Arm('r');q=lambda k:np.array(P8[k]['q'])
        place=(q('over_tray'),None,q('carry'))
        new=N.place_time(A,q('lifted'),place,0.,q_from=q('closed'))
        home=N.move_time(q('over_tray'),A.home)
        self.assertGreaterEqual(new-home,dt('closed','released'))
        old=N.place_time(A,q('lifted'),place,0.,settle=.3)-N.move_time(q('over_tray'),A.home,settle=.3)
        self.assertLess(old,dt('closed','released'))       # old model: 6.6 s < 12.0 s measured

class LightFirst(unittest.TestCase):
    def test_priors_carry_masses(self):
        for n,p in N.PRI.items():self.assertGreater(p.get('mass',0),0,n)
        self.assertTrue(N.heavy('006_mustard_bottle'));self.assertFalse(N.heavy('Perricone_MD_AcylGlutathione_Eye_Lid_Serum'))
    def test_logged_holds_were_all_light(self):
        held=[a['obj'] for a in GT if a['held_at_lift'] is True];heavy=[a for a in GT if N.heavy(a['obj'])]
        self.assertTrue(held and not any(N.heavy(n) for n in held))
        self.assertEqual(len(heavy),6);self.assertFalse(any(a['held_at_lift'] for a in heavy))
    def test_run_tries_light_object_first(self):
        sol=N.Sol();sol.arms={k:ARM.Arm(k) for k in 'lr'}
        sol.task={'task':'sort2','categories':{'jar':'tray_left'},'objects':{'006_mustard_bottle':'jar','Hyaluronic_Acid':'jar'},'trays':{}}
        Clock.t=50.;logs=[];sol.log=lambda *a:logs.append(' '.join(str(x) for x in a))
        sol.survey=lambda *a,**k:Clock.t;sol.go_home=lambda k:None;sol.retreat=lambda k:None;sol.js={'x':0.}
        sol.robot_description='<robot/>';sol.trays={'tray_left':{'center':[.3,.6],'axis':[0,1],'size':[.2,.3],'height':.07}}
        d1=types.SimpleNamespace(pixels=np.zeros((0,2),int));d2=types.SimpleNamespace(pixels=np.zeros((0,2),int))
        sol.observe=lambda nb=None:(np.zeros((1,1,3)),None,None,[d1,d2])
        sol.identify=lambda *a:[(d1,'006_mustard_bottle','jar',.95,None,.1),(d2,'Hyaluronic_Acid','jar',.8,None,.1)]
        tries=[]
        def pp(name,*a):tries.append(name);Clock.t+=1.;return False
        sol.pick_place=pp
        with mock.patch.object(N,'WALL_LAUNCH',time.monotonic()):sol.run()
        self.assertEqual(tries[0],'Hyaluronic_Acid')       # lighter object first although less confident
        self.assertIn('006_mustard_bottle',tries)          # the heavy one is still attempted
if __name__=='__main__':unittest.main()
