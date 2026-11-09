"""v8 fixes (fixes/V8_ANALYSIS_RU.md): held rule, jammed fingers / clamped targets, guarded approach, position stall,
re-observe after contact, 'no plan' / tray-source / accel logs. Evidence: real Kaggle v7+v8 TRACE fixtures
(fixtures/v7v8_grasp_traces.json, force 1.0, built by fixes/diag/make_v7v8_fixture.py)."""
import os,sys,types,unittest,json
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE)
import test_node_stub as TS                      # installs the ROS stubs
import scene_fixture as SF
from team048_solution import node as N, arm as ARM, grip as G
FX=json.load(open(os.path.join(HERE,'fixtures','v7v8_grasp_traces.json')))['attempts']
RIM=[1,1,.1,.1,.2,0.];SIDE=[0,0,0,0,0,0.];OPEN=[1,1,1,1,1,0.]
def t0(a):return SIDE if a['kind']=='side' else RIM
LIFTED=[a for a in FX if a['held_at_lift'] is not None]

class TraceFixture(unittest.TestCase):
    def test_fixture_shape(self):
        self.assertEqual(len(FX),16);self.assertEqual(len(LIFTED),15)
        self.assertEqual(sum(a['held_at_lift'] for a in LIFTED),4)

class HeldRule(unittest.TestCase):
    def test_15_of_15_as_recorded(self):
        """v7/v8 commanded the closed target during the lift -> command = closed target."""
        for a in LIFTED:
            L=a['phases']['lifted']
            self.assertEqual(G.held_rule(L['fing'],L['eff'],t0(a),t0(a)),a['held_at_lift'],(a['run'],a['seed'],a['obj']))
    def test_15_of_15_with_phase2_command(self):
        """Phase 2 commands target2 = contact - past (clamped, jammed not re-targeted). A position-controlled finger
        cannot pass target2, so the lifted position is max(recorded, target2) on re-targeted fingers."""
        for a in LIFTED:
            C=a['phases']['closed'];L=a['phases']['lifted']
            t2=G.contact_target(C['fing'],t0(a),.15)
            p=list(L['fing'])
            for i in range(5):
                if t2[i]>t0(a)[i]+1e-9:p[i]=max(p[i],t2[i])
            self.assertEqual(G.held_rule(p,L['eff'],t2,t0(a)),a['held_at_lift'],(a['run'],a['seed'],a['obj']))
    def test_old_criteria_would_fail(self):
        """Why past_verdict / finger_verdict are log-only: v8 Perricone (the only scored object of v8)."""
        a=next(x for x in LIFTED if x['run']=='v8' and x['obj'].startswith('Perricone'))
        C=a['phases']['closed'];L=a['phases']['lifted']
        self.assertFalse(G.past_verdict(L['fing'],G.contact_target(C['fing'],SIDE,.15),SIDE,.15))
        self.assertIsNone(G.finger_verdict(L['fing'],L['eff'],SIDE,eff_min=.67))
        self.assertTrue(G.held_rule(L['fing'],L['eff'],SIDE,SIDE))
    def test_missing_data(self):
        self.assertIsNone(G.held_rule(None,[1]*6,SIDE,SIDE));self.assertIsNone(G.held_rule([0]*6,[1]*6,SIDE,[1]*6))
    def test_node_hand_verdict_uses_rule_and_logs_old(self):
        s=N.Sol();s.hand_target['l']=[.36,.5,.5,.5,1.,0.]
        s.hstate={'l':types.SimpleNamespace(position=[.04,.04,.0,-.11,1.22,0.],effort=[.08,.09,.18,1.,1.,.6],velocity=[0]*6)}
        gr=dict(closed=SIDE,target2=[0.,.36,.5,.5,1.,0.])
        self.assertTrue(s.hand_verdict('l',gr))
        sh=s.hand_shadow('l',gr);self.assertIn('past',sh);self.assertIn('v7',sh)

class JammedAndClamp(unittest.TestCase):
    def test_targets_never_above_one(self):
        old_over=0
        for a in FX:
            C=a['phases']['closed']
            t2=G.contact_target(C['fing'],t0(a),.15)
            self.assertLessEqual(max(t2),1.0)
            jam=G.jammed(C['fing'],t0(a))
            for d in jam:self.assertEqual(t2[d-1],t0(a)[d-1])          # jammed: keeps the closed target
            old=[max(t,p-.15) for t,p in zip(t0(a)[:5],C['fing'][:5]) if t<.5]
            old_over+=max(old)>1.
        self.assertEqual(old_over,7)            # the pre-fix formula: 7 of 16 attempts asked for > 1.0
    def test_jammed_examples(self):
        self.assertEqual(G.jammed([.988,1.117,1.738,.123,.215,0],RIM),[3])      # v8 Threshold: middle bent back
        self.assertEqual(G.jammed([.988,1.117,1.738,.123,.215,0]),[2,3])         # all fingers
        self.assertEqual(G.jammed([1.807,.583,.69,.661,1.304,0],SIDE),[1,5])     # v8 mustard
    def test_contact_empty_never_drops_a_held_grasp(self):
        for a in FX:
            if a['held_at_lift']:self.assertFalse(G.contact_empty(a['phases']['closed']['fing'],t0(a)))
    def test_grasp_close_logs_jam_and_keeps_target(self):
        s=N.Sol();sent=[];logs=[];s.hand_set=lambda side,v:sent.append(list(v));s.log=lambda *a:logs.append(a)
        s.wait_fingers=lambda side,**k:True
        s.hstate={'l':types.SimpleNamespace(position=[1.807,.583,.69,.661,1.304,0.],effort=[1]*6,velocity=[0]*6)}
        r=s.grasp_close('l','side')
        self.assertEqual(r['jam'],[1,5]);self.assertEqual(sent[1][0],0.);self.assertEqual(sent[1][4],0.)
        self.assertAlmostEqual(sent[1][1],.583-G.CLOSE_PAST);self.assertTrue(any('jammed' in str(x[0]) for x in logs))

class GuardedApproach(unittest.TestCase):
    def trips(self,a):
        return any(G.approach_deviation(a['phases'][ph]['fing'],OPEN,t0(a))[1] for ph in ('pre','at_grasp','corrected') if ph in a['phases'])
    def test_no_held_grasp_trips(self):
        for a in FX:
            if a['held_at_lift']:self.assertFalse(self.trips(a),(a['run'],a['seed'],a['obj']))
    def test_most_failed_grasps_trip_before_closing(self):
        bad=[a for a in FX if not a['held_at_lift']]
        self.assertEqual(len(bad),12);self.assertGreaterEqual(sum(self.trips(a) for a in bad),10)   # v9fix GUARD_DEV .16: v7 Hyaluronic (0.109) no longer
    def test_all_finger_guard_would_abort_v7_success(self):
        """Why only the closing fingers are guarded: v7 Ecoforms (rim) ring finger 1.14/1.17 at the grasp pose."""
        eco=[a for a in FX if a['held_at_lift'] and a['obj'].startswith('Ecoforms')]
        self.assertEqual(len(eco),2)
        # ring finger 1.137 / 1.173 (non-closing for the rim pinch): above the v8 threshold 0.10 in both, and above the
        # v9fix threshold 0.16 in one -> guarding every finger would still abort a held grasp
        for a in eco:self.assertGreater(G.approach_deviation(a['phases']['at_grasp']['fing'],OPEN)[0],.10)
        self.assertTrue(any(G.approach_deviation(a['phases']['at_grasp']['fing'],OPEN)[1] for a in eco))
    def test_move_arm_stops_on_guard(self):
        s=N.Sol();s.arms={k:ARM.Arm(k) for k in 'lr'};A=s.arms['l']
        s.js=dict(zip(A.chain.names,A.home));sent=[]
        s.cmd=lambda g,ids,q,spd:sent.append(np.asarray(q).copy())
        orig=N.time.sleep
        try:
            N.time.sleep=lambda dt:setattr(TS.Clock,'t',TS.Clock.t+dt)
            n=[0]
            def guard():
                n[0]+=1;return {'dev':.3,'fingers':[3]} if n[0]>=3 else None
            ok=s.move_arm('l',A.home+.3,guard=guard)
        finally:N.time.sleep=orig
        self.assertFalse(ok);self.assertEqual(s.guard_trip['fingers'],[3])
        self.assertTrue(np.allclose(sent[-1],A.home))        # hold at the measured joints
    def test_guard_abort_marks_contact_and_does_not_close(self):
        s=N.Sol();s.arms={k:ARM.Arm(k) for k in 'lr'};calls=[]
        s.retreat=lambda side:calls.append('retreat');s.trace=lambda *a,**k:calls.append(('trace',a[0]))
        s.hand_log=lambda *a,**k:calls.append('hand');s.log=lambda *a:None
        s.last_contact=False
        self.assertFalse(s.guard_abort('l','x','approach',{'dev':.4,'fingers':[3]}))
        self.assertTrue(s.last_contact);self.assertIn('retreat',calls);self.assertIn(('trace','guard'),calls)

class PositionStall(unittest.TestCase):
    def test_stall_and_motion(self):
        h=[(10+i*.05,[.5-.1*i,.5,.5,.5,.5,0]) for i in range(5)]
        self.assertFalse(G.fingers_stalled(h,10.2))                            # still closing
        h=[(10+i*.05,[.3+1e-3*(i%2),.5,.5,.5,.5,0]) for i in range(5)]
        self.assertTrue(G.fingers_stalled(h,10.2))
        self.assertFalse(G.fingers_stalled(h,10.2,since=10.15))                # too few samples after the command
        self.assertFalse(G.fingers_stalled([],10.))
    def test_wait_fingers_uses_position(self):
        s=N.Sol();orig=N.time.sleep
        def tick(dt):
            TS.Clock.t+=.05;s.hpos['l'].append((TS.Clock.t,[.3,.3,.3,.3,.3,0.]))
        try:
            N.time.sleep=tick;s.hvel['l']=[[1.]*6]*10           # velocities never 'settled' (v8: 13/16)
            t=TS.Clock.t;self.assertTrue(s.wait_fingers('l',tmin=.3,tmax=2.))
            self.assertLess(TS.Clock.t-t,1.)
        finally:N.time.sleep=orig;TS.Clock.t=100.

class RunLoopReobserve(unittest.TestCase):
    def test_reobserve_after_contact_and_no_plan_logs(self):
        sol=N.Sol();sol.arms={k:ARM.Arm(k) for k in 'lr'};logs=[]
        sol.log=lambda *a:logs.append(' '.join(str(x) for x in a))
        sol.task={'task':'sort2','categories':{'box':'tray_left','dish':'tray_right'},'objects':{'a':'box','b':'box','c':'dish'},
                  'trays':{'tray_left':'red','tray_right':'blue'}}
        sol.trays={'tray_left':SF.TRAYS['tray_left']} if isinstance(SF.TRAYS['tray_left'],dict) else {'tray_left':{'center':[0,0]}}
        TS.Clock.t=50.;obs=[];picks=[]
        sol.look=lambda *a,**k:TS.Clock.t;sol.survey=lambda *a,**k:TS.Clock.t
        P=np.zeros((2,2,3));D=[types.SimpleNamespace(pixels=np.array([[0,0]])) for _ in range(3)]
        sol.observe=lambda nb=None:(obs.append(1),(P,None,None,D))[1]
        def identify(dets,P,plane,names):
            return [(d,n,'dish' if n=='c' else 'box',5.-i,None,0.) for i,(d,n) in enumerate(zip(dets,['a','b','c'])) if n in names]
        sol.identify=identify
        def pp(name,cat,cloud,plane,others,tray):
            picks.append(name);sol.last_contact=True;return False      # every attempt touches and fails
        sol.pick_place=pp
        sol.retreat=lambda k:None;sol.go_home=lambda k:None;sol.js={'x':0.};sol.robot_description='<robot/>'
        try:sol.run()
        finally:TS.Clock.t=100.
        # each failed contact attempt triggers a new observation; the object is retried at most twice per round,
        # v9fix: plus one retry round with reset counters while time is left (RETRY_ROUNDS=1)
        self.assertEqual(picks,['a','a','b','b']*(1+N.RETRY_ROUNDS))
        self.assertTrue(any(l.startswith('retry round 1') for l in logs))
        self.assertGreaterEqual(len(obs),5)
        self.assertTrue(any(l.startswith('re-observe after contact a') for l in logs))
        self.assertTrue(any(l.startswith('no plan: c') and 'tray not found' in l for l in logs))
        self.assertTrue(any(l.startswith('accel ') for l in logs));self.assertTrue(any(l.startswith('accel at end') for l in logs))
    def test_accel_status(self):
        self.assertTrue(callable(N.accel_status));st=N.accel_status();self.assertIn(st[0],('on','off'))

if __name__=='__main__':unittest.main()
