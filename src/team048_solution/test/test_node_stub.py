"""node.py wiring without ROS: rclpy/msg/tf2 replaced by minimal stubs, real run-v5 head frame as input."""
import os,sys,types,unittest
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'))
V5=os.path.join(HERE,'fixtures','v5_head_frame.npz')

class Msg:
    def __init__(s,**k):s.__dict__.update(k)
class Pub:
    def __init__(s,topic):s.topic=topic;s.sent=[]
    def publish(s,m):s.sent.append(m)
class Fut:
    """rclpy.task.Future stand-in: resolved immediately with `reply` (None = the service never answers)."""
    def __init__(s,reply):s.reply=reply
    def done(s):return s.reply is not None
    def result(s):return s.reply
class Cli:
    def __init__(s,name):s.name=name;s.calls=[];s.ready=True;s.reply=Msg()
    def service_is_ready(s):return s.ready
    def call_async(s,r):s.calls.append(r);return Fut(s.reply)
class Clock:
    t=100.
    def now(s):return types.SimpleNamespace(nanoseconds=int(Clock.t*1e9))
class FakeNode:
    def __init__(s,name):s.pubs={};s.subs={};s.clis={}
    def create_subscription(s,typ,topic,cb,qos):s.subs[topic]=cb
    def create_publisher(s,typ,topic,qos):p=Pub(topic);s.pubs[topic]=p;return p
    def create_client(s,typ,name):c=Cli(name);s.clis[name]=c;return c
    def get_clock(s):return Clock()
    def get_logger(s):return types.SimpleNamespace(info=lambda m:None)
def install_stubs():
    def mod(name,**attrs):
        m=types.ModuleType(name);m.__dict__.update(attrs);sys.modules[name]=m;return m
    mk=lambda n:type(n,(Msg,),{})
    rclpy=mod('rclpy',init=lambda:None,shutdown=lambda:None)
    mod('rclpy.node',Node=FakeNode)
    mod('rclpy.qos',QoSProfile=lambda **k:None,DurabilityPolicy=types.SimpleNamespace(TRANSIENT_LOCAL=1),
        ReliabilityPolicy=types.SimpleNamespace(RELIABLE=1),qos_profile_sensor_data=None)
    mod('rclpy.executors',MultiThreadedExecutor=object)
    rclpy.time=types.SimpleNamespace(Time=lambda **k:None);rclpy.duration=types.SimpleNamespace(Duration=lambda **k:None)
    mod('std_msgs');mod('std_msgs.msg',String=mk('String'),Empty=mk('Empty'))
    mod('sensor_msgs');mod('sensor_msgs.msg',JointState=mk('JointState'),Image=mk('Image'),CameraInfo=mk('CameraInfo'))
    mod('geometry_msgs');mod('geometry_msgs.msg',WrenchStamped=mk('WrenchStamped'))
    class Cmd(Msg):
        def __init__(s):s.cmds=[]
    class SetForce:Request=mk('SetForceReq')
    class SetSpeed:Request=mk('SetSpeedReq')
    mod('bodyctrl_msgs');mod('bodyctrl_msgs.msg',CmdSetMotorPosition=Cmd,SetMotorPosition=mk('SetMotorPosition'))
    mod('bodyctrl_msgs.srv',SetForce=SetForce,SetSpeed=SetSpeed)
    mod('tf2_ros',Buffer=lambda:None,TransformListener=lambda b,n:None)
install_stubs()
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));import scene_fixture as SF
from team048_solution import node as N, arm as ARM

def stamp(t):return types.SimpleNamespace(sec=int(t),nanosec=int(round((t-int(t))*1e9)))
class NodeWiring(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        c.s=N.Sol();c.s.arms={k:ARM.Arm(k) for k in 'lr'};c.s.hand_settings_step()   # hand force delivered (stub services)
        for A in c.s.arms.values():A.set_scene(SF.plane(),list(SF.TRAYS.values()))   # synthetic test scene
    def test_speed_cap_in_command(s):
        s.s.cmd('arm',[11,12],[0.,0.],[5.,0.])
        sent=s.s.pub['arm'].sent[-1].cmds
        s.assertTrue(all(0<c.spd<=.8 for c in sent))      # 0 would mean "max speed" in the sim: never sent
    def test_hand_settings_calls_services(s):
        s.s.hand_settings()
        from team048_solution import grip as G
        r=s.s.cli_force['l'].calls[-1];s.assertEqual(r.force0_ratio,G.HAND_FORCE_RATIO);s.assertTrue(.3<=r.force3_ratio<=.5)
        s.assertEqual(r.force5_ratio,G.HAND_FORCE_THUMB_ROT)          # thumb rotation: positioning, not squeezing
        r=s.s.cli_speed['r'].calls[-1];s.assertEqual(r.speed3_ratio,G.HAND_SPEED_RATIO);s.assertLess(r.speed3_ratio,1.)
    def test_subscribed_topics(s):
        subs=set(s.s.subs)
        for t in ('/arm_6dof_left','/arm_6dof_right','/inspire_hand/state/left_hand','/inspire_hand/state/right_hand',
                  '/cam_head/color/image_raw','/cam_head/depth/image_raw','/sim/task','/joint_states'):
            s.assertIn(t,subs)
        s.assertFalse(any('/sim/objects' in t or '/sim/task/check' in t for t in list(subs)+list(s.s.pubs)))
    def test_plan_place_slots_and_retreat(s):
        A=s.s.arms['l'];top=SF.TABLE_TOP;tx,ty=SF.TRAYS['tray_left']
        res=[N.plan_place(A,A.home,tx,ty+dy,top) for dy in (0.,.08,-.08)]
        ok=[r for r in res if r is not None];s.assertTrue(ok,'no feasible place pose for the left tray')
        for q,qu,_ in ok:
            if qu is not None:s.assertGreater(A.tcp(qu)[2,3],A.tcp(q)[2,3]+.07)
    @unittest.skipUnless(os.path.exists(V5),'v5 fixture missing')
    def test_observe_on_real_v5_frame(s):
        f=np.load(V5);sol=s.s;sol.task={'task':'sort3','trays':{'tray_left':'red','tray_right':'blue','tray_center':'green'}}
        st=float(f['stamp']);Clock.t=st-.01
        rgb,dep=f['rgb'],f['depth']
        sol.frames.add_rgb(Msg(header=Msg(stamp=stamp(st),frame_id='camera_head_link'),height=360,width=640,data=rgb.tobytes()))
        sol.frames.add_depth(Msg(header=Msg(stamp=stamp(st),frame_id='camera_head_link'),height=360,width=640,data=dep.astype(np.uint16).tobytes()))
        sol.info=Msg(k=list(f['K']))
        T=N.quat_to_T(list(f['tf_t']),list(f['tf_q']));sol.tf=lambda a,b,stamp=None:T
        sol.trays={}
        P,rgb2,plane,dets=sol.observe()
        s.assertEqual(P.shape,(360,640,3));s.assertGreater(plane[0][2],.96);s.assertTrue(ARM.plane_plausible(plane))
        s.assertEqual(sol.trays,{})       # observe() never invents trays; the survey adds only full detections
        valid=np.isfinite(P).all(-1)&(f['depth']>0)
        out=sol.locate_trays(P,rgb2,valid,plane)
        d=out.get('tray_center')          # v5 frame: the green tray is cut by the bottom image border
        s.assertIsNotNone(d);s.assertTrue(d['partial']);s.assertNotIn('tray_center',sol.trays)
        hy=ARM.fk_link('head_yaw_link')[:3,3];s.assertLess(abs(np.arctan2(d['centroid'][1]-hy[1],d['centroid'][0]-hy[0])),.2)
    def test_budget_skip_returns_none(s):
        sol=s.s;from team048_solution import budget as B
        sol.budget=B.TimeBudget(270,t0=Clock.t-265)     # 5 s left
        for A in sol.arms.values():sol.js.update(zip(A.chain.names,A.home))   # q_now needs measured joints (no implicit 0)
        rng=np.random.default_rng(0);cloud=np.c_[rng.uniform(.33,.39,400),rng.uniform(.17,.23,400),rng.uniform(-.18,-.10,400)]
        plane=(np.array([0,0,1.]),-SF.TABLE_TOP);sol.trays=dict(SF.TRAYS)
        n0=len(sol.pub['arm'].sent);n1=len(sol.hand['left'].sent)+len(sol.hand['right'].sent)
        r=sol.pick_place('004_sugar_box','box',cloud,plane,[],'tray_left')
        s.assertIsNone(r)                               # a plan exists but does not fit: never started
        s.assertEqual(len(sol.pub['arm'].sent),n0);s.assertEqual(len(sol.hand['left'].sent)+len(sol.hand['right'].sent),n1)
    def test_plan_place_right_tray_reachable(s):
        # v6 root cause: no release pose for tray_right (closing axis tried with one sign only) -> boxes never picked
        A=s.s.arms['r'];top=SF.TABLE_TOP;tx,ty=SF.TRAYS['tray_right']
        s.assertIsNotNone(N.plan_place(A,A.home,tx,ty,top))
        s.assertIsNotNone(N.plan_place(A,A.home,tx,ty-.08,top,fallback=(tx,ty)))
    def test_look_waits_for_head(s):
        sol=s.s;sol.js={'head_pitch_joint':0.};seq=iter([0.,.1,.25,.39,.398,.399,.399,.399]+[.399]*50)
        def fake_sleep(dt):
            Clock.t+=dt;sol.js['head_pitch_joint']=next(seq)
        sol.sleep=fake_sleep;t0=Clock.t
        t=sol.look(.4)
        s.assertGreaterEqual(t-t0,.5)                       # returned only after the head reached 0.4 and stopped
        s.assertAlmostEqual(sol.js['head_pitch_joint'],.399,3)
        s.assertEqual(sol.pub['head'].sent[-1].cmds[0].pos,.4)
        del sol.sleep
    def test_look_times_out_when_no_joint_state(s):
        sol=s.s;sol.js={}
        def fake_sleep(dt):Clock.t+=dt
        sol.sleep=fake_sleep;t0=Clock.t;t=sol.look(.4,tmax=1.)
        s.assertGreaterEqual(t-t0,1.-1e-6);del sol.sleep   # float sum of sleeps (order-independent)
    def test_tf_uses_image_stamp(s):
        sol=s.s;seen=[]
        class Tr:pass
        def lk(target,source,time,timeout=None):
            seen.append(time);tr=Tr();tr.transform=Msg(translation=Msg(x=0,y=0,z=0),rotation=Msg(x=0,y=0,z=0,w=1));return tr
        import rclpy
        old=rclpy.time.Time;rclpy.time.Time=lambda **k:k
        sol.tfb=types.SimpleNamespace(lookup_transform=lk)
        try:N.Sol.tf(sol,'pelvis','camera_head_link',stamp(12.5))
        finally:rclpy.time.Time=old
        s.assertEqual(seen[0],{'seconds':12,'nanoseconds':500000000})
    def test_trace_line_is_json(s):
        sol=s.s;from team048_solution import budget as B;import json
        sol.budget=B.TimeBudget(270,t0=Clock.t);logs=[]
        sol.get_logger=lambda:types.SimpleNamespace(info=logs.append)
        A=sol.arms['l'];sol.js=dict(zip(A.chain.names,A.home));sol.wrench['l']=np.array([0.,0.,1.5]);sol.monitor.baseline(np.zeros(3))
        sol.trace('lifted','l','x',sol.arms['l'].tcp(sol.arms['l'].home),verdict=None)
        r=json.loads(logs[-1].split('TRACE ',1)[1]);s.assertEqual(r['phase'],'lifted');s.assertAlmostEqual(r['dF'],1.5)
        s.assertAlmostEqual(r['tcp_err'],0.,3)
        del sol.get_logger
    def test_run_retries_empty_observation_then_done(s):
        # v6 seed 101: first observation empty -> loop ended at once. Therefore look() first, re-observe up to EMPTY_OBS_MAX.
        sol=N.Sol();sol.arms=s.s.arms
        sol.task={'task':'sort2','categories':{'box':'tray_left'},'objects':{'a':'box'},'trays':{}}
        calls=[];Clock.t=50.
        sol.look=lambda *a,**k:(calls.append('look'),Clock.t)[1]
        sol.observe=lambda nb=None:(calls.append('obs'),(None,None,None,[]))[1]
        sol.identify=lambda *a:[]
        sol.retreat=lambda k:None;sol.go_home=lambda k:calls.append('home');sol.js={'x':0.}
        sol.survey=lambda *a,**k:(calls.append('survey'),Clock.t)[1];sol.robot_description='<robot/>'
        from unittest import mock
        # start-up = wall time between import and node creation: in a long or loaded suite run it exceeded the 270 s limit
        # (no observation at all). Pin it to zero so the test does not depend on how long the suite has been running.
        with mock.patch.object(N,'WALL_LAUNCH',sol.wall_ready):sol.run()
        # survey first (arms in the start posture), then home along a path certified against the measured table
        s.assertEqual(calls[:3],['survey','home','home']);s.assertEqual(calls.count('obs'),N.EMPTY_OBS_MAX)
        s.assertTrue(sol.pubs['/sim/task/done'].sent)
if __name__=='__main__':unittest.main()
