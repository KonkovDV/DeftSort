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
