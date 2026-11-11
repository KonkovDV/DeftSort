"""Hand settings delivery and closed dish tops, regression tests from run v9b (sort2 seed 101, 10 points, 1 of 6 objects).
1. The left hand never received its SetForce request (fixture v9b_hand_settings.json): requests are retried until
   the service is discovered and has answered; an arm whose hand force was never delivered is not used.
2. The Threshold ramekin has a closed top in every recorded frame; the rim pinch put the thumb on it in 4 of 4
   approaches (fixture v9b_dish_tops.npz): no rim candidates on a dish without an open rim."""
import os,sys,json,unittest
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,HERE)
import test_node_stub as TS
from test_node_stub import N
from team048_solution import grip as G, grasp as GR
HS=json.load(open(os.path.join(HERE,'fixtures','v9b_hand_settings.json')))
DT=os.path.join(HERE,'fixtures','v9b_dish_tops.npz')

class LateService(TS.Cli):
    """Service that is discovered only after `late` readiness polls (v9b: the left hand at the first and only poll)."""
    def __init__(s,name,late=0,reply=True):super().__init__(name);s.late=late;s.polls=0;s.reply=TS.Msg() if reply else None
    def service_is_ready(s):s.polls+=1;return s.polls>s.late

def fresh(late_l=0,late_r=0,reply=True):
    s=N.Sol()
    for kind in ('force','speed'):
        cl={'l':LateService(f'{kind} l',late_l,reply),'r':LateService(f'{kind} r',late_r,reply)}
        setattr(s,'cli_force' if kind=='force' else 'cli_speed',cl)
    s.logs=[];s.log=lambda *a:s.logs.append(' '.join(str(x) for x in a));return s

class Fixture(unittest.TestCase):
    def test_left_force_line_missing_in_five_of_six_layouts(self):
        L=HS['layouts'];self.assertEqual(len(L),6)
        self.assertEqual(sum(x['right'] is not None for x in L),6)
        self.assertEqual(sum(x['left'] is None for x in L),5)
        self.assertEqual(HS['v9b_left_hand_closes'],5)

class HandSettings(unittest.TestCase):
    def test_one_shot_send_misses_a_late_service(self):
        """Why requests are retried: a single readiness poll at start leaves a late left-hand service without force."""
        s=fresh(late_l=3)
        for k in 'lr':
            if s.cli_force[k].service_is_ready():s.cli_force[k].call_async(object())
        self.assertEqual(len(s.cli_force['l'].calls),0);self.assertEqual(len(s.cli_force['r'].calls),1)
    def test_late_left_service_gets_force_and_speed(self):
        s=fresh(late_l=3);wall=0.
        for _ in range(10):
            if s.hand_settings_step(wall):break
            wall+=N.HAND_CFG_POLL_S
        for k in 'lr':
            self.assertEqual(s.hand_cfg[('force',k)]['state'],'confirmed',k)
            self.assertEqual(s.hand_cfg[('speed',k)]['state'],'confirmed',k)
            r=s.cli_force[k].calls[-1]
            self.assertEqual([getattr(r,f'force{i}_ratio') for i in range(6)],G.force_vector())
            self.assertTrue(all(.3<=getattr(r,f'force{i}_ratio')<=.5 for i in range(5)))   # #648
            self.assertEqual(r.force5_ratio,G.HAND_FORCE_THUMB_ROT)
            self.assertEqual(s.cli_speed[k].calls[-1].speed0_ratio,G.HAND_SPEED_RATIO)
        self.assertEqual(len(s.cli_force['l'].calls),1)          # sent once, when it was discovered
        self.assertTrue(any('hand settings left force confirmed' in l for l in s.logs))
        self.assertTrue(any('hand settings right force confirmed' in l for l in s.logs))
    def test_no_reply_is_resent_then_given_up(self):
        s=fresh(reply=False);wall=0.;n=0
        while not s.hand_settings_step(wall) and n<200:wall+=N.HAND_CFG_POLL_S;n+=1
        for key,c in s.hand_cfg.items():
            self.assertEqual(c['state'],'unconfirmed',key);self.assertEqual(c['sends'],N.HAND_CFG_SENDS_MAX)
        self.assertEqual(len(s.cli_force['l'].calls),N.HAND_CFG_SENDS_MAX)
        # bounded: the last send happens within SENDS_MAX reply windows (+1 poll)
        self.assertLessEqual(wall,N.HAND_CFG_SENDS_MAX*(N.HAND_CFG_REPLY_S+N.HAND_CFG_POLL_S)+N.HAND_CFG_POLL_S)
        self.assertTrue(s.hand_force_applied('l') and s.hand_force_applied('r'))   # delivered, only unanswered
    def test_never_discovered_hand_is_not_used(self):
        s=fresh(late_l=10**9)
        for i in range(30):s.hand_settings_step(i*N.HAND_CFG_POLL_S)
        self.assertEqual(s.hand_cfg[('force','l')]['state'],'waiting')
        self.assertFalse(s.hand_force_applied('l'));self.assertTrue(s.hand_force_applied('r'))
        self.assertEqual(s.hand_cfg_summary()['left_force'],'waiting')
    def test_background_thread_finishes(self):
        s=fresh(late_l=2);th=s.hand_settings();th.join(5.)
        self.assertFalse(th.is_alive());self.assertTrue(all(c['state']=='confirmed' for c in s.hand_cfg.values()))
    def test_pick_skips_an_arm_without_force(self):
        """pick_place never plans for an arm whose force request was not delivered; nothing is marked infeasible."""
        s=fresh(late_l=10**9,late_r=10**9);s.hand_settings_step(0.)
        s.trays={'tray_left':None}
        called=[]
        class Arm:
            hand={'opposed_open':.09}
            def set_other(self,q):called.append(q)
        s.arms={'l':Arm(),'r':Arm()};s.placed={};s.q_now=lambda side:np.zeros(7)
        orig=N.candidates;N.candidates=lambda *a,**k:({'center':np.array([.4,.3]),'length':.1},[])
        orig_tg=N.ARMMOD.tray_geom;N.ARMMOD.tray_geom=lambda t:(.35,.6,1.,0.,.22,.32,.07)
        try:ok=s.pick_place('x','dish',None,None,[],'tray_left')
        finally:N.candidates=orig;N.ARMMOD.tray_geom=orig_tg
        self.assertFalse(ok);self.assertEqual(called,[]);self.assertFalse(s.last_infeasible if hasattr(s,'last_infeasible') else False)
        self.assertTrue(any('hand force not applied on any arm' in l for l in s.logs))

@unittest.skipUnless(os.path.exists(DT),'v9b dish fixture missing')
class ClosedRim(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        d=np.load(DT);c.pl=(d['plane_n'],float(d['plane_d']));c.th=d['threshold'];c.ut=d['utana']
    def test_threshold_top_is_closed(self):
        f,C=GR.candidates(self.th,self.pl,'dish')
        self.assertTrue(f['rim_closed']);self.assertEqual([g for g in C if g['kind']=='rim'],[])
        self.assertAlmostEqual(f['height'],.052,delta=.002)       # 5.3 cm in objects.yaml
    def test_utana_rim_still_planned(self):
        f,C=GR.candidates(self.ut,self.pl,'dish')
        self.assertFalse(f['rim_closed']);self.assertEqual(sum(g['kind']=='rim' for g in C),len(GR.RIM_ANGLE_DEG))
    def test_interior_profile(self):
        """The numbers behind the rule: Threshold interior at the top height, Utana interior empty."""
        for cl,closed in ((self.th,True),(self.ut,False)):
            f=GR.footprint(cl,self.pl);c0,r0=GR.circle_fit(f['hull']);h=cl@self.pl[0]+self.pl[1]
            inner=np.linalg.norm(cl[:,:2]-c0,axis=1)<r0-GR.RIM_W
            if closed:self.assertGreater(inner.sum(),600);self.assertGreater(np.median(h[inner]),f['height']-.002)
            else:self.assertEqual(int(inner.sum()),0)

if __name__=='__main__':unittest.main()
