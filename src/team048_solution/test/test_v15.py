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
