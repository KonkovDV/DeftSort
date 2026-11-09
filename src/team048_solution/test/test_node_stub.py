"""node.py wiring without ROS: rclpy/msg/tf2 replaced by minimal stubs, real v5 head frame (Kaggle) as input."""
import os,sys,types,unittest,time
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'))
V5=os.path.join(HERE,'fixtures','v5_head_frame.npz')

class Msg:
    def __init__(s,**k):s.__dict__.update(k)
class Pub:
    def __init__(s,topic):s.topic=topic;s.sent=[]
    def publish(s,m):s.sent.append(m)
class Cli:
    def __init__(s,name):s.name=name;s.calls=[]
    def service_is_ready(s):return True
    def call_async(s,r):s.calls.append(r)
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
    rclpy.time=types.SimpleNamespace(Time=lambda:None);rclpy.duration=types.SimpleNamespace(Duration=lambda **k:None)
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
from team048_solution import node as N, arm as ARM, trays as TR

def stamp(t):return types.SimpleNamespace(sec=int(t),nanosec=int(round((t-int(t))*1e9)))
class NodeWiring(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        c.s=N.Sol();c.s.arms={k:ARM.Arm(k) for k in 'lr'}
    def test_speed_cap_in_command(s):
        s.s.cmd('arm',[11,12],[0.,0.],[5.,0.])
        sent=s.s.pub['arm'].sent[-1].cmds
        s.assertTrue(all(0<c.spd<=.8 for c in sent))      # 0 would mean "max speed" in the sim: never sent
    def test_hand_settings_calls_services(s):
        s.s.hand_settings()
        r=s.s.cli_force['l'].calls[-1];s.assertEqual(r.force0_ratio,1.0);s.assertEqual(r.force5_ratio,1.0)
        r=s.s.cli_speed['r'].calls[-1];s.assertEqual(r.speed3_ratio,1.0)
    def test_subscribed_topics(s):
        subs=set(s.s.subs)
        for t in ('/arm_6dof_left','/arm_6dof_right','/inspire_hand/state/left_hand','/inspire_hand/state/right_hand',
                  '/cam_head/color/image_raw','/cam_head/depth/image_raw','/sim/task','/joint_states'):
            s.assertIn(t,subs)
        s.assertFalse(any('/sim/objects' in t or '/sim/task/check' in t for t in list(subs)+list(s.s.pubs)))
    def test_plan_place_slots_and_retreat(s):
        A=s.s.arms['l'];top=ARM.TABLE_TOP;tx,ty=TR.NOMINAL['tray_left']
        res=[N.plan_place(A,A.home,tx,ty+dy,top) for dy in (0.,.08,-.08)]
        ok=[r for r in res if r is not None];s.assertTrue(ok,'no feasible place pose for the left tray')
        for q,qu in ok:
            if qu is not None:s.assertGreater(A.tcp(qu)[2,3],A.tcp(q)[2,3]+.07)
    @unittest.skipUnless(os.path.exists(V5),'v5 fixture missing')
    def test_observe_on_real_v5_frame(s):
        f=np.load(V5);sol=s.s;sol.task={'task':'sort3','trays':{'tray_left':'red','tray_right':'blue','tray_center':'green'}}
        st=float(f['stamp']);Clock.t=st-.01
        rgb,dep=f['rgb'],f['depth']
        sol.frames.add_rgb(Msg(header=Msg(stamp=stamp(st),frame_id='camera_head_link'),height=360,width=640,data=rgb.tobytes()))
        sol.frames.add_depth(Msg(header=Msg(stamp=stamp(st),frame_id='camera_head_link'),height=360,width=640,data=dep.astype(np.uint16).tobytes()))
        sol.info=Msg(k=list(f['K']))
        T=N.quat_to_T(list(f['tf_t']),list(f['tf_q']));sol.tf=lambda a,b:T
        P,rgb2,plane,dets=sol.observe()
        s.assertEqual(P.shape,(360,640,3));s.assertGreater(plane[0][2],.96)
        for name,(x,y) in sol.trays.items():
            s.assertLessEqual(np.hypot(x-TR.NOMINAL[name][0],y-TR.NOMINAL[name][1]),.10+1e-9)
    def test_budget_skip_returns_none(s):
        sol=s.s;from team048_solution import budget as B
        sol.budget=B.TimeBudget(270,t0=Clock.t-265)     # 5 s left
        rng=np.random.default_rng(0);cloud=np.c_[rng.uniform(.33,.39,400),rng.uniform(.17,.23,400),rng.uniform(-.18,-.10,400)]
        plane=(np.array([0,0,1.]),-ARM.TABLE_TOP)
        n0=len(sol.pub['arm'].sent);n1=len(sol.hand['left'].sent)+len(sol.hand['right'].sent)
        r=sol.pick_place('004_sugar_box','box',cloud,plane,[],'tray_left')
        s.assertIsNone(r)                               # a plan exists but does not fit: never started
        s.assertEqual(len(sol.pub['arm'].sent),n0);s.assertEqual(len(sol.hand['left'].sent)+len(sol.hand['right'].sent),n1)
if __name__=='__main__':unittest.main()
