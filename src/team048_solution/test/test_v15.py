"""v15 fixes (CPU, no GPU): bounded waits for required topics, frame freshness in capture, explicit handling of
joints missing from /joint_states, atomic hand-state history, stable tray component order, offline tests of the
motion methods (go_home, retreat, waist, approach_guard, recover, end_safe) on a fake ROS interface."""
import os,sys,types,unittest,threading,time
from unittest import mock
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import test_node_stub as TS                      # installs the ROS stubs
import scene_fixture as SF
from team048_solution import node as N, arm as ARM, grip as G

def run_in_thread(sol,timeout=20.):
    """Sol.run() with a simulated clock (every sleep advances sim time by 1 s); returns True if it finished."""
    real_sleep=time.sleep
    def tick(dt):TS.Clock.t+=1.;real_sleep(0)
    th=threading.Thread(target=lambda:sol.run(),daemon=True)
    with mock.patch.object(N.time,'sleep',tick),mock.patch.object(N,'WALL_LAUNCH',sol.wall_ready):
        th.start();th.join(timeout)
    return not th.is_alive()

class RequiredTopics(unittest.TestCase):
    def test_no_joint_states_ends_with_done_inside_the_budget(s):
        sol=N.Sol();logs=[];sol.log=lambda *a:logs.append(' '.join(str(x) for x in a))
        sol.task={'task':'sort2','categories':{'box':'tray_left'},'objects':{'a':'box'},'trays':{'tray_left':'red'}}
        sol.robot_description='<robot/>';sol.survey=lambda *a,**k:TS.Clock.t;sol.js={}
        sol.retreat=lambda k:None;sol.hand_open=lambda k:None
        TS.Clock.t=50.
        try:
            s.assertTrue(run_in_thread(sol),'run() still waiting for /joint_states')
            s.assertEqual(len(sol.done.sent),1)
            el=sol.budget.elapsed(TS.Clock.t);s.assertLessEqual(el,sol.budget.limit)
            s.assertTrue(any('joint_states' in l for l in logs))
        finally:TS.Clock.t=100.

def sol_with_arms(js=None):
    sol=N.Sol();sol.arms={k:ARM.Arm(k) for k in 'lr'};logs=[];sol.log=lambda *a:logs.append(' '.join(str(x) for x in a))
    for A in sol.arms.values():A.set_scene(SF.plane(),list(SF.TRAYS.values()))
    if js is None:
        js={}
        for A in sol.arms.values():js.update(zip(A.chain.names,A.home))
    sol.js=dict(js);sol.plane=SF.plane();return sol,logs

def bounded(fn,timeout=20.):
    """fn() on a thread with a simulated clock (each sleep = +0.05 s sim); (finished, exception)."""
    real_sleep=time.sleep;out={}
    def tick(dt):TS.Clock.t+=.05;real_sleep(0)
    def body():
        try:out['r']=fn()
        except BaseException as e:out['e']=e
    th=threading.Thread(target=body,daemon=True)
    with mock.patch.object(N.time,'sleep',tick):th.start();th.join(timeout)
    return (not th.is_alive()),out.get('e')

class MissingJoint(unittest.TestCase):
    def test_missing_joint_is_not_zero_and_the_arm_is_not_moved(s):
        sol,logs=sol_with_arms();del sol.js['elbow_pitch_r_joint']
        ok,e=bounded(lambda:sol.q_now('r'));s.assertTrue(ok);s.assertIsInstance(e,getattr(N,'JointStateMissing',type(None)))
        ok,e=bounded(lambda:sol.retreat('r'));s.assertTrue(ok,'retreat moved/waited from an invented posture')
        s.assertIsInstance(e,getattr(N,'JointStateMissing',type(None)))
        s.assertEqual(sol.pub['arm'].sent,[],'an arm command was sent from an invented posture')
        s.assertTrue(any('elbow_pitch_r_joint' in l for l in logs))
    def test_complete_joint_states_unchanged(s):
        sol,_=sol_with_arms();A=sol.arms['l']
        s.assertTrue(np.array_equal(sol.q_now('l'),np.asarray(A.home,float)))
    def test_end_safe_without_joint_states_moves_nothing(s):
        sol,logs=sol_with_arms(js={})
        ok,e=bounded(sol.end_safe);s.assertTrue(ok);s.assertIsNone(e)
        s.assertEqual(sol.pub['arm'].sent,[]);s.assertEqual(sol.pub['waist'].sent,[])

def frame(enc,data,t,h=2,w=2):
    st=types.SimpleNamespace(sec=int(t),nanosec=int(round((t-int(t))*1e9)))
    return types.SimpleNamespace(encoding=enc,height=h,width=w,data=bytes(data),header=types.SimpleNamespace(stamp=st,frame_id='camera_head_link'))
class FrameFreshness(unittest.TestCase):
    def cap(s,t_frame,t_req):
        sol=N.Sol();logs=[];sol.log=lambda *a:logs.append(' '.join(str(x) for x in a))
        sol.frames.add_rgb(frame('rgb8',np.zeros((2,2,3),np.uint8).tobytes(),t_frame))
        sol.frames.add_depth(frame('16UC1',np.full((2,2),800,np.uint16).tobytes(),t_frame))
        sol.info=types.SimpleNamespace(k=[100.,0,1,0,100.,1,0,0,1]);sol.tf=lambda *a:np.eye(4)
        return sol.capture(not_before=t_req,wait=.05),logs
    def test_frame_from_before_the_motion_is_refused(s):
        with s.assertRaises(RuntimeError):s.cap(9.0,10.0)      # 1 s = 33 camera periods older than the request
    def test_frame_within_two_periods_is_used(s):
        (P,rgb,valid),logs=s.cap(10.0-1.5*N.CAMERA_PERIOD_S,10.0);s.assertEqual(P.shape,(2,2,3))
    def test_fresh_frame_unchanged(s):
        (P,rgb,valid),logs=s.cap(10.01,10.0);s.assertFalse(any('capture:' in l for l in logs))
    def test_period_constant_from_the_sim_log(s):
        s.assertEqual(N.CAMERA_PERIOD_S,.03);s.assertAlmostEqual(N.CAPTURE_MAX_AGE,.06)

class HandHistory(unittest.TestCase):
    def test_reader_sees_a_contiguous_history_while_the_callback_trims(s):
        """The hand-state callback runs on an executor thread; wait_fingers iterates s.hpos on the task thread."""
        sol=N.Sol();stop=[False];dt=.01;bad=0;reads=0
        m=types.SimpleNamespace(velocity=[0.]*6,position=[.5]*6)
        def feed():
            while not stop[0]:TS.Clock.t=round(TS.Clock.t+dt,6);sol.on_hand('l',m)
        old=sys.getswitchinterval();sys.setswitchinterval(1e-6);th=threading.Thread(target=feed,daemon=True);th.start()
        try:
            t0=time.time()
            while time.time()-t0<2. and bad==0:
                h=sol.hpos['l'];ts=[x[0] for x in h];reads+=1
                if len(ts)>1 and np.max(np.diff(ts))>1.5*dt:bad+=1
        finally:
            stop[0]=True;th.join();sys.setswitchinterval(old);TS.Clock.t=100.
        s.assertGreater(reads,100);s.assertEqual(bad,0,'a sample was skipped while the history was trimmed in place')

from team048_solution import trays as TRY
class StableComponents(unittest.TestCase):
    SIZES=[42,32,26,14,16,3,4,1,60,60,32,45,25,30,48,36,31]   # two largest tie (labels 9 and 10, raster order)
    def mask(s):
        m=np.zeros((2*len(s.SIZES),70),bool)
        for r,n in enumerate(s.SIZES):m[2*r,:n]=True            # one component per even row, labelled top to bottom
        return m
    def test_tie_keeps_raster_order(s):
        pix=TRY._largest_component(s.mask(),10)
        s.assertEqual(len(pix),60);s.assertEqual(int(pix[0,0]),2*8,'tie broken toward the later component')
    def test_components_order_stable(s):
        c=TRY._components(s.mask());s.assertEqual([int(x[0,0])//2 for x in c[:2]],[8,9])

class FakeRobot:
    """Fake ROS side: an arm/waist/head command moves the measured joints to the commanded positions (clipped to the
    URDF limits the simulator enforces), a hand command moves the hand state to its target; every sleep advances sim
    time by 0.02 s. Records every command for sequence checks."""
    def __init__(s,sol):
        s.sol=sol;s.log=[];s.ids={}
        for side,ids in N.IDS.items():
            for i,n in zip(ids,sol.arms[side].chain.names):s.ids[i]=(side,n)
        for g in ('arm','waist','head'):
            sol.pub[g].publish=lambda m,g=g:s.on_cmd(g,m)
        for k,p in sol.hand.items():p.publish=lambda m,k=k:s.on_hand(k,m)
        side_of={v:k for k,v in N.SIDE_NAME.items()}
        s.side_of=side_of
        for k in 'lr':sol.hstate[k]=types.SimpleNamespace(position=[1.]*5+[0.],velocity=[0.]*6,effort=[0.]*6)
    def on_cmd(s,g,m):
        for c in m.cmds:
            if g=='arm':
                side,n=s.ids[c.name];A=s.sol.arms[side];j=A.chain.names.index(n)
                s.log.append(('arm',side,n,c.pos,c.spd))
                with s.sol.lock:s.sol.js[n]=float(np.clip(c.pos,A.chain.lower[j],A.chain.upper[j]))
            elif g=='waist':
                s.log.append(('waist',c.name,c.pos))
                with s.sol.lock:s.sol.js['body_yaw_joint']=float(c.pos)
            else:s.log.append((g,c.name,c.pos))
    def on_hand(s,k,m):
        side=s.side_of[k];s.log.append(('hand',side,list(m.position)))
        s.sol.hstate[side]=types.SimpleNamespace(position=list(m.position),velocity=[0.]*6,effort=[0.]*6)
    def arm_cmds(s,side=None):return [x for x in s.log if x[0]=='arm' and (side is None or x[1]==side)]

def robot(js=None):
    sol,logs=sol_with_arms(js);sol.hand_settings_step();return sol,logs,FakeRobot(sol)
def in_limits(test,sol,fr):
    for _,side,n,pos,spd in fr.arm_cmds():
        A=sol.arms[side];j=A.chain.names.index(n)
        test.assertGreaterEqual(pos,A.chain.lower[j]-1e-9,n);test.assertLessEqual(pos,A.chain.upper[j]+1e-9,n)
        test.assertGreater(spd,0.);test.assertLessEqual(spd,.8)

class MotionMethods(unittest.TestCase):
    def test_go_home_from_home_and_from_a_raised_pose(s):
        sol,logs,fr=robot();A=sol.arms['l']
        ok,e=bounded(lambda:sol.go_home('l'));s.assertTrue(ok);s.assertIsNone(e)
        s.assertTrue(np.allclose(sol.q_now('l'),A.home,atol=.02));in_limits(s,sol,fr)
        q=np.asarray(A.home,float).copy();q[0]-=.3                     # shoulder pitched forward (arm raised)
        q=np.clip(q,A.chain.lower,A.chain.upper);sol.js.update(zip(A.chain.names,q));fr.log.clear()
        ok,e=bounded(lambda:sol.go_home('l'));s.assertTrue(ok);s.assertIsNone(e)
        s.assertTrue(np.allclose(sol.q_now('l'),A.home,atol=.05));in_limits(s,sol,fr)
        s.assertEqual({x[1] for x in fr.arm_cmds()},{'l'},'go_home moved the other arm')
    def test_retreat_lifts_first_when_low(s):
        sol,logs,fr=robot();A=sol.arms['r']
        lifted=[];orig=A.lift
        def lift(q,dz):lifted.append(dz);return orig(q,dz)
        A.lift=lift;sol.plane=(np.array([0,0,1.]),-10.)                # table far above: TCP counts as low
        ok,e=bounded(lambda:sol.retreat('r'));s.assertTrue(ok);s.assertIsNone(e)
        s.assertEqual(lifted,[N.RETREAT_DZ]);in_limits(s,sol,fr)
        s.assertTrue(np.allclose(sol.q_now('r'),A.home,atol=.05))
    def test_waist_settles_and_times_out(s):
        sol,logs,fr=robot();sol.js['body_yaw_joint']=.4
        ok,e=bounded(lambda:sol.waist(0.));s.assertTrue(ok);s.assertIsNone(e)
        s.assertEqual([x for x in fr.log if x[0]=='waist'],[('waist',31,0.)]);s.assertEqual(sol.js['body_yaw_joint'],0.)
        sol.pub['waist'].publish=lambda m:None;t0=TS.Clock.t                # waist does not answer
        out={}
        ok,e=bounded(lambda:out.setdefault('r',sol.waist(.5)));s.assertTrue(ok)
        s.assertFalse(out['r']);s.assertLessEqual(TS.Clock.t-t0,N.WAIST_TMAX+.1)
    def test_approach_guard_trips_on_a_deflected_finger_only(s):
        sol,logs,fr=robot();sol.hand_open('l')
        s.assertIsNone(sol.approach_guard('l','side'))
        h=sol.hstate['l'];p=list(h.position);p[3]=p[3]-.5                 # index finger pushed by contact
        sol.hstate['l']=types.SimpleNamespace(position=p,velocity=h.velocity,effort=h.effort)
        g=sol.approach_guard('l','side');s.assertIsNotNone(g);s.assertGreater(g['dev'],0.)
    def test_recover_opens_hands_and_homes_both_arms(s):
        sol,logs,fr=robot()
        for k in 'lr':sol.hand_set(k,[0,0,0,0,0,0.])                   # both hands closed (on an object)
        ok,e=bounded(sol.recover);s.assertTrue(ok);s.assertIsNone(e)
        for k in 'lr':
            s.assertTrue(sol.fingers_open(k));s.assertEqual(sol.hand_target[k],[1,1,1,1,1,0.])
            s.assertTrue(np.allclose(sol.q_now(k),sol.arms[k].home,atol=.05))
        first_arm=next(i for i,x in enumerate(fr.log) if x[0]=='arm');last_open=max(i for i,x in enumerate(fr.log) if x[0]=='hand')
        s.assertLess(last_open,first_arm,'an arm moved before both hands were opened');in_limits(s,sol,fr)
    def test_end_safe_hands_empty_arms_home_waist_zero(s):
        sol,logs,fr=robot();sol.js['body_yaw_joint']=.3
        ok,e=bounded(sol.end_safe);s.assertTrue(ok);s.assertIsNone(e)
        for k in 'lr':s.assertTrue(np.allclose(sol.q_now(k),sol.arms[k].home,atol=.05))
        s.assertEqual(sol.js['body_yaw_joint'],0.);in_limits(s,sol,fr)
    def test_force_requests_both_hands_in_range(s):
        sol,logs,fr=robot()
        for (kind,side),r in sol.hand_requests().items():
            if kind!='force':continue
            v=[getattr(r,f'force{i}_ratio') for i in range(6)]
            s.assertTrue(all(G.HAND_FORCE_MIN<=x<=G.HAND_FORCE_MAX for x in v[:5]),v);s.assertEqual(v[5],G.HAND_FORCE_THUMB_ROT)
        s.assertEqual({k for (kind,k) in sol.hand_requests() if kind=='force'},{'l','r'})
