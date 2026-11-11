"""Release over the tray: the place check, the bounded re-approach, the release back on the table when the tray is not
reached, and the carry check that does not count an object that slipped out during transport."""
import os,sys,types,unittest
import numpy as np
sys.path.insert(0,os.path.dirname(__file__))
import test_node_stub as TS                      # installs the ROS stubs
import scene_fixture as SF
from team048_solution import node as N, arm as ARM, budget as B

HELD=types.SimpleNamespace(position=[.8,.8,.8,.8,.5,0.],effort=[1]*6,velocity=[0]*6)
EMPTY=types.SimpleNamespace(position=[0.]*6,effort=[0]*6,velocity=[0]*6)
OPEN=types.SimpleNamespace(position=[1.,1.,1.,1.,1.,0.],effort=[0]*6,velocity=[0]*6)
CLOUD=np.c_[np.random.default_rng(0).uniform(.33,.39,400),np.random.default_rng(1).uniform(.17,.23,400),
            np.random.default_rng(2).uniform(-.18,-.08,400)]

def run(ignore=lambda st,q:False,empty_after_lift=False):
    """pick_place of the sugar box into tray_left on the synthetic scene with stubbed motion. ignore(st, q) -> True
    makes move_arm leave the arm where it is (not reached)."""
    s=N.Sol();s.arms={k:ARM.Arm(k) for k in 'lr'};s.hand_settings_step()
    for A in s.arms.values():A.set_scene(SF.plane(),list(SF.TRAYS.values()))
    s.trays=dict(SF.TRAYS);s.budget=B.TimeBudget(270,t0=TS.Clock.t)
    st={'phase':'pre','q':{},'moves':[],'opens':[],'plan':None};tr=[]
    def move_arm(side,q,**k):
        q=np.asarray(q,float).copy();st['moves'].append((st['phase'],q))
        if ignore(st,q):return False
        st['q'][side]=q;return True
    def hand_set(side,v):
        s.hand_target[side]=list(v)
        if v[0]==1 and st['phase']!='pre':st['opens'].append((st['phase'],st['q'][side].copy()))
    def trace(phase,side,name,*a,**k):
        tr.append((phase,k))
        if phase=='at_grasp':s.hstate[side]=HELD
        if phase=='lifted':
            st['phase']='carry'
            if empty_after_lift:s.hstate[side]=EMPTY
    s.move_arm=move_arm;s.hand_set=hand_set;s.trace=trace
    s.wait_fingers=lambda side,**k:True;s.wait_thumb=lambda side,**k:True;s.sleep=lambda dt:None
    s.retreat=lambda side:None;s.go_home=lambda side:st['moves'].append(('home',None))
    s.waist=lambda th,**k:True;s.fingers_open=lambda side:True
    s.hstate={'l':OPEN,'r':OPEN}
    for k in 'lr':
        A=s.arms[k];st['q'][k]=np.asarray(A.home,float);s.js.update(zip(A.chain.names,A.home))
    s.q_now=lambda side:st['q'][side].copy()
    orig=N.plan_place_yaw
    def ppy(*a,**k):
        r=orig(*a,**k);st['plan']=r;return r
    N.plan_place_yaw=ppy
    try:r=s.pick_place('004_sugar_box','box',CLOUD,SF.plane(),[],'tray_left')
    finally:N.plan_place_yaw=orig
    return r,s,st,[p for p,_ in tr],tr

def is_place(st,q):
    return st['plan'] is not None and np.allclose(q,st['plan'][1][0])

class InTray(unittest.TestCase):
    def test_rectangle(self):
        t={'center':[.4,.3],'axis':[0.,1.],'size':[.22,.32]}
        self.assertTrue(N.in_tray((.4,.3),t));self.assertTrue(N.in_tray((.4+.10,.3+.15),t))
        self.assertFalse(N.in_tray((.4+.12,.3),t));self.assertFalse(N.in_tray((.4,.3+.17),t))
        r={'center':[.4,.3],'axis':[1.,0.],'size':[.22,.32]}       # long axis along x
        self.assertTrue(N.in_tray((.4+.15,.3),r));self.assertFalse(N.in_tray((.4,.3+.12),r))

class Release(unittest.TestCase):
    def test_reached(self):
        r,s,st,ph,_=run()
        self.assertIs(r,True);self.assertEqual(s.placed,{'tray_left':1})
        self.assertNotIn('place_retry',ph);self.assertNotIn('release_off_tray',ph);self.assertIn('released',ph)
        place=st['plan'][1][0];A=s.arms['l']
        self.assertTrue(N.in_tray(N.release_xy(A,place,None,st['plan'][0]),s.trays['tray_left']))
    def test_not_reached_then_retry(self):
        n={'c':0}
        def first_place_only(st,q):
            if st['phase']=='carry' and is_place(st,q):
                n['c']+=1;return n['c']==1
            return st['phase']=='carry' and n['c']==1 and not is_place(st,q)   # the correction step does not move either
        r,s,st,ph,_=run(first_place_only)
        self.assertIs(r,True);self.assertIn('place_retry',ph);self.assertNotIn('release_off_tray',ph)
        self.assertEqual(s.placed,{'tray_left':1});self.assertEqual(n['c'],2)
    def test_not_reached_released_on_the_table(self):
        q_lift={}
        def stuck_toward_tray(st,q):   # every move toward the tray fails; the way back to the lift pose works
            return st['phase']=='carry' and not ('l' in q_lift and np.allclose(q,q_lift['l']))
        orig=ARM.Arm.plan_pick
        def pp(self,*a,**k):
            p=orig(self,*a,**k)
            if p is not None:q_lift[self.side]=np.asarray(p['q_lift'],float)
            return p
        ARM.Arm.plan_pick=pp
        try:r,s,st,ph,tr=run(stuck_toward_tray)
        finally:ARM.Arm.plan_pick=orig
        self.assertIs(r,False);self.assertEqual(s.placed,{})
        self.assertIn('place_retry',ph);self.assertIn('release_off_tray',ph);self.assertNotIn('over_tray',ph)
        self.assertEqual(dict(tr)['release_off_tray']['where'],'pick_spot')
        self.assertEqual(len(st['opens']),1)                       # opened once, never left holding the object
        phase,q=st['opens'][0];A=s.arms['l']
        self.assertTrue(np.allclose(q,q_lift['l']))                 # above the spot it was picked from
        xy=N.release_xy(A,q,None,0.);self.assertFalse(N.in_tray(xy,s.trays['tray_left']))
        self.assertTrue(s.last_contact)
    def test_no_time_to_carry_back_releases_in_place(self):
        def stuck(st,q):return st['phase']=='carry'
        orig=N.Sol.release_off_tray
        def aged(self,*a,**k):
            TS.Clock.t+=self.budget.remaining(self.now())-1.;return orig(self,*a,**k)
        N.Sol.release_off_tray=aged
        try:r,s,st,ph,tr=run(stuck)
        finally:N.Sol.release_off_tray=orig;TS.Clock.t=100.
        self.assertIs(r,False);self.assertEqual(dict(tr)['release_off_tray']['where'],'in_place')
        self.assertEqual(len(st['opens']),1)

class CarryCheck(unittest.TestCase):
    def test_empty_over_tray_not_counted(self):
        r,s,st,ph,tr=run(empty_after_lift=True)
        self.assertIs(r,False);self.assertEqual(s.placed,{})
        self.assertIs(dict(tr)['over_tray']['verdict'],False);self.assertIn('released',ph)
        self.assertTrue(s.last_contact)

if __name__=='__main__':
    unittest.main()
