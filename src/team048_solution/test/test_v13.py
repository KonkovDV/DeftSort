"""v13 audit fixes: thread-safe RGB-D pairing, exception safety of the run loop (release, home, /sim/task/done on
every path), one retry of a failed observation."""
import os,sys,types,unittest,threading,time,collections
from unittest import mock
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import test_node_stub as TS                      # installs the ROS stubs
import scene_fixture as SF
from team048_solution import node as N, arm as ARM
from team048_solution.sensing import PairBuffer

def msg(t):return types.SimpleNamespace(header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=int(t),nanosec=int(round((t%1)*1e9)))))

class Feeder:
    """Camera callbacks on another thread: append rgb/depth as fast as possible (worst case for the race)."""
    def __init__(s,add_rgb,add_depth):s.stop=False;s.a=add_rgb;s.b=add_depth;s.th=threading.Thread(target=s.run,daemon=True)
    def run(s):
        t=0.
        while not s.stop:t+=.033;s.a(msg(t));s.b(msg(t))
    def __enter__(s):s.old=sys.getswitchinterval();sys.setswitchinterval(1e-6);s.th.start();return s
    def __exit__(s,*a):s.stop=True;s.th.join();sys.setswitchinterval(s.old)

class PairRace(unittest.TestCase):
    def test_unlocked_deque_iteration_races(s):
        # the failure the lock removes: iterating a deque another thread appends to (the pre-v13 pair() loop)
        d=collections.deque(maxlen=6);err=0;t0=time.time()
        with Feeder(d.append,d.append):
            while err==0 and time.time()-t0<5.:
                try:
                    for _ in reversed(d):pass
                except RuntimeError:err+=1
        s.assertGreater(err,0)
    def test_pair_is_safe_under_concurrent_callbacks(s):
        b=PairBuffer();err=0;n=0;t0=time.time()
        with Feeder(b.add_rgb,b.add_depth):
            while time.time()-t0<1.:
                try:b.pair(0.);n+=1
                except RuntimeError:err+=1
        s.assertGreater(n,1000);s.assertEqual(err,0)
    def test_pair_result_unchanged(s):
        b=PairBuffer()
        for t in (1.,1.033,1.066):b.add_rgb(msg(t));b.add_depth(msg(t))
        b.add_rgb(msg(1.1))
        r,d=b.pair(1.01);s.assertAlmostEqual(d.header.stamp.sec+d.header.stamp.nanosec*1e-9,1.066)
        s.assertIsNone(b.pair(2.))

def run_stub(pick_place=None,observe=None,identify=None,retreat=None):
    """Sol.run() on the ROS stubs with two box objects in a found tray; returns (sol, logs, calls)."""
    sol=N.Sol();sol.arms={k:ARM.Arm(k) for k in 'lr'};logs=[];calls=[]
    sol.log=lambda *a:logs.append(' '.join(str(x) for x in a))
    sol.task={'task':'sort2','categories':{'box':'tray_left'},'objects':{'a':'box','b':'box'},'trays':{'tray_left':'red'}}
    sol.trays={'tray_left':SF.TRAYS['tray_left']}
    TS.Clock.t=50.
    sol.look=lambda *a,**k:TS.Clock.t;sol.survey=lambda *a,**k:TS.Clock.t
    P=np.zeros((2,2,3));D=[types.SimpleNamespace(pixels=np.array([[0,0]])) for _ in range(2)]
    sol.observe=observe or (lambda nb=None:(P,None,None,D))
    sol.identify=identify or (lambda dets,P,plane,names:[(d,n,'box',5.-i,None,0.) for i,(d,n) in enumerate(zip(dets,['a','b'])) if n in names])
    sol.pick_place=pick_place or (lambda *a:False)
    sol.retreat=retreat or (lambda k:calls.append(('retreat',k)))
    sol.go_home=lambda k:calls.append(('home',k));sol.hand_open=lambda k:calls.append(('open',k))
    sol.fingers_open=lambda k:True;sol.js={'x':0.};sol.robot_description='<robot/>'
    try:
        with mock.patch.object(N,'WALL_LAUNCH',sol.wall_ready):sol.run()
    finally:TS.Clock.t=100.
    return sol,logs,calls

class RunSafety(unittest.TestCase):
    def test_exception_in_pick_place_recovers_and_continues(s):
        picks=[]
        def pp(name,*a):
            picks.append(name)
            if name=='a':raise ValueError('boom')    # e.g. a degenerate cloud in the planner
            TS.Clock.t+=1.;return True
        sol,logs,calls=run_stub(pick_place=pp)
        s.assertIn('b',picks)                                      # the run went on with the next object
        s.assertTrue(any(l.startswith('pick failed with exception a ValueError') for l in logs))
        s.assertIn(('open','l'),calls);s.assertIn(('open','r'),calls)  # recover(): both hands opened
        s.assertTrue(sol.pubs['/sim/task/done'].sent)
        s.assertTrue(any(l.startswith('done at') and "'b'" in l for l in logs))
    def test_exception_outside_pick_place_still_ends_cleanly(s):
        def ident(*a):raise IndexError('bad segment')
        sol,logs,calls=run_stub(identify=ident)
        s.assertTrue(sol.pubs['/sim/task/done'].sent)
        s.assertIn(('retreat','l'),calls);s.assertIn(('retreat','r'),calls)
        s.assertTrue(any('reason: exception IndexError' in l for l in logs))
    def test_failing_retreat_does_not_block_done(s):
        def rt(k):
            if k=='l':raise RuntimeError('no path')
        sol,logs,calls=run_stub(retreat=rt)
        s.assertTrue(sol.pubs['/sim/task/done'].sent)
        s.assertTrue(any(l.startswith('end: retreat failed l') for l in logs))
    def test_observe_failure_is_retried_once(s):
        n=[0];P=np.zeros((2,2,3))
        def obs(nb=None):
            n[0]+=1
            if n[0]==1:raise RuntimeError('no synchronised RGB-D pair / camera_info')
            return P,None,None,[]
        sol,logs,calls=run_stub(observe=obs)
        s.assertGreaterEqual(n[0],1+N.EMPTY_OBS_MAX)                 # the run did not end at the first failure
        s.assertFalse(any('reason: observe failed' in l for l in logs))
    def test_two_failures_in_a_row_end_the_run_cleanly(s):
        def obs(nb=None):raise RuntimeError('camera gone')
        sol,logs,calls=run_stub(observe=obs)
        s.assertTrue(any('reason: observe failed' in l for l in logs));s.assertTrue(sol.pubs['/sim/task/done'].sent)

class StartupPin(unittest.TestCase):
    def test_flaky_cause(s):
        # why test_node_stub:150 / test_v8fixes:137 failed under load: the suite's wall time since import was charged
        # as start-up; past ~230 s no cycle fits in sort2's 270 s. The tests now pin WALL_LAUNCH to the node creation.
        from team048_solution import budget as B
        b=B.TimeBudget(270.,B.start_time(50.,1000.,1000.-240.))
        s.assertFalse(b.can_start(50.))
        b=B.TimeBudget(270.,B.start_time(50.,1000.,1000.));s.assertTrue(b.can_start(50.))
if __name__=='__main__':unittest.main()
