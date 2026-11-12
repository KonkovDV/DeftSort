"""Team 048 solution node: sort2/sort3 (pool from /sim/task). Sensor-only perception (head RGB-D + TF),
object identity by assignment of segmented blobs to the task's object list using size priors, URDF IK with
sphere self-collision checks, 0.8 rad/s-capped synchronized joint moves, Inspire hand pinch/side grasps.
Per-layout time budget (startup included), grasp check from the hand state (grip.held_rule, approach guard; the
wrist F/T is logged only), slots inside trays, trays confirmed by camera colour, straight-up retreats, TCP check
before closing. Hand force/speed (#648) are sent to each hand as soon as its service appears, re-sent until the call
returns, and an arm whose hand never got the force limit is not used.
Tubes: not solved -> immediate /sim/task/done (0 points, allowed by SCORING).
Debug topics (/sim/task/check, /sim/objects/*) are never used.
Scene (#645): table plane + near edge and the trays are measured by the cameras at run time (head survey with head
yaw scan, torso camera for the near edge); only approximate search bounds and robot geometry are constants."""
import json,os,time,threading
WALL_IMPORT=time.monotonic()
from . import budget as _B0
_AGE=_B0.launch_age_s()                       # run.sh -> ros2 run -> python start-up before this import (sim limit
WALL_LAUNCH=WALL_IMPORT-(_AGE if _AGE is not None and _AGE<120 else 0.)   # counts it: SCORING 4 Nov, #595.3/#638)
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String,Empty
from sensor_msgs.msg import JointState,Image,CameraInfo
from geometry_msgs.msg import WrenchStamped
from bodyctrl_msgs.msg import CmdSetMotorPosition,SetMotorPosition
from bodyctrl_msgs.srv import SetForce,SetSpeed
import tf2_ros
from scipy.optimize import linear_sum_assignment
from . import arm as ARMMOD
from . import budget as BUD, grip as GRIP, trays as TR
from .sensing import PairBuffer
from .perception import table_plane,segment,near_edge,deproject,table_region,rect_mask
from .grasp import candidates,demote_failed
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
PRI=json.load(open(os.path.join(D,'priors.json')))
TRAY_H=TR.DOC_H        # used only when a detected tray carries no measured wall height
IDS={'l':[11,12,13,14,15,16,17],'r':[21,22,23,24,25,26,27]}
SIDE_NAME={'l':'left','r':'right'}
APPROACH_VMAX=.3      # rad/s on the last pre-grasp -> grasp segment (do not knock the object); 0.8 elsewhere
TCP_TOL=.01           # m, measured TCP vs planned before closing; one correction if exceeded
RETREAT_DZ=.08        # m, straight-up lift after release / before going home from low poses
LOW_ABOVE_TABLE=.10    # m, TCP lower than this above the measured table -> lift straight up before going home
HEAD_PITCH=.4         # rad, look-down posture for the table (head_pitch_joint)
HEAD_TOL=.01          # rad, head considered on target
HEAD_TMAX=6.          # s sim, give up waiting for the head (logged)
# Tray survey (#645): generic look-left / look-right head yaws. Camera FOV geometry (camera_info: 640x360, 86 deg HFOV;
# URDF head chain): at pitch 0.4 the views at yaw 0/+0.6/-0.6 jointly cover bearings -83..+83 deg of the table in front
# of the robot -- the whole half-plane, not a stored tray position. A tray seen cut by the image border is re-aimed at
# its own measured centroid (TRAY_RELOOK_MAX times); a tray that is never fully seen is not used (its category is
# skipped -> 0 points, never -10).
HEAD_SCAN_YAWS=(0.,.6,-.6)
TRAY_RELOOK_MAX=2
TRAY_RESURVEY_MAX=1   # extra survey later in the run when a needed tray is still missing
TRAY_MASK_MARGIN=.03  # m, pixels within the measured tray rectangle + margin are not objects
ID_CONF_MIN=.7        # identify(): category margin; render-set study (pool 3, 625 segments): 0.7 -> 599 accepted / 7 wrong vs 1.0 -> 498 / 3
ID_COST_MAX=2.5
TRACE_DIR=os.environ.get('TEAM048_TRACE_DIR')   # offline test harness only: per-object JSON trace + RGB-D frames
SHADOW_DF_N=.5*.08*9.81   # not used for decisions: logged 'would-be' verdict, 1/2 weight of the lightest pool-3 object (sim cfg 0.08 kg)
PRE_TOL=.05           # m, pre-grasp TCP error above which the pick is aborted (v7 s101 mustard: 26.6 cm, then knocked)
RELEASE_OPEN_MIN=.8    # fingers 1-4 mean after release; v7: 0.96-0.97 released, 0.99 home
PLACE_TOL=.04         # m, release TCP error: one correction before opening (v7 s101 Ecoforms released 13.7 cm off)
PLACE_RETRIES=1       # re-approaches of the release pose when the release point is still off the tray
SIDE_DZ=(0.,.02,.04)  # m, raise a side grasp when the low one is not collision-free against the measured table
EMPTY_OBS_MAX=2       # consecutive observations with nothing to pick before giving up (an observation whose
                      # attempt touched the scene and was re-observed does not count as 'nothing to pick')
HEAVY_KG=.22          # kg (objects.yaml masses in priors.json). v7+v8 TRACE (force 1.0, 16 attempts): 4 of 10 attempts
                      # on objects <= 0.20 kg were held after the lift, 0 of 6 on objects >= 0.22 kg (Crunch 0.22, Utana and
                      # Crayola 0.35, mustard 0.43); failed attempts also push neighbours (v8 Crunch, v9a frames_end).
                      # -> light objects first within an observation, heavy ones afterwards (still attempted).
# move_arm / waist / release timeouts (sim s). They bound the worst case of every step while an object is in the hand
# (hold_time_bound) -- the #595 guarantee 'nothing in the hand at the limit' must hold even if every step times out.
MOVE_T_PAD=2.         # s added to the 1.6x speed-line time of the first move_arm attempt
REFINE_T=1.5          # s per refine re-command of move_arm
MOVE_RES=.06          # rad, largest residual move_arm accepts as reached (start of the next move may differ by this)
WAIST_TMAX=6.         # s, waist() waits at most this
RELEASE_WAIT=1.       # s, wait_fingers tmax after the release command (done at most twice: re-open)
HOLD_MARGIN_S=1.      # s, design margin for the solver's own sim time between the lift check and the release
                      # (IK correction, trace/log lines); not a measured value
MOVE_SETTLE_S=2.6     # s per arm move on top of the 1.6x speed line (move_arm: PD undershoot, refine). v8 Perricone TRACE:
                      # lifted->carry 2.8 s for |dq| 0.30 (line 0.6 s), carry->over_tray 4.9 s for |dq| 1.15 (line 2.3 s)
FAILS_MAX=2           # failed attempts per object before it is skipped (per round)
RETRY_ROUNDS=1        # extra round with reset failure counters when time is left (v9a: stop at 124 s of 405 s)
HAND_LOG_DT=.1        # s sim: hand-state log period during the guarded approach and the closing (10 Hz, for the guard and stall checks)
# SetForce / SetSpeed delivery (#648: finger force 0.3-0.5 on both hands). DDS discovery of the hand services can lag
# behind the node start, and a hand without a SetForce request closes at the default force 1.0 (the simulator logs
# '[hack] <side> hand force limits' for every request it applies; fixture v9b_hand_settings.json). Every request is
# therefore retried from a background thread until the service has answered;
# an arm whose hand never received its force request is not used for picking.
# The values below are wall-clock design choices (DDS discovery runs on wall time), not measured thresholds.
HAND_CFG_POLL_S=.2      # s wall between delivery passes
HAND_CFG_REPLY_S=2.     # s wall to wait for a reply before the request is sent again
HAND_CFG_SENDS_MAX=5    # sends per service; after that the request counts as delivered without a reply
# Camera frame period in sim time: '[hack] camera period 0.0300 s sim, render_interval 3' (simulator log of
# runs v9a, v9b, v11b; the bridge logged 37-41 frames per 5 s wall at RTF ~0.2). A frame up to two periods older than
# the request (one frame in flight plus one dropped) is still accepted when no newer one arrives within `wait`.
CAMERA_PERIOD_S=.03
CAPTURE_MAX_AGE=2*CAMERA_PERIOD_S
class JointStateMissing(RuntimeError):
    """An arm joint has never been reported on /joint_states (the arm is not moved from an invented posture)."""
def quat_to_T(t,q):
    """4x4 transform from translation t (m) and unit quaternion q = (x, y, z, w)."""
    x,y,z,w=q;T=np.eye(4)
    T[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    T[:3,3]=t;return T
class Sol(Node):
    """The ROS 2 node: subscriptions, task state machine, survey, pick and place, safe end."""
    def __init__(s):
        super().__init__('team048_solution')
        lat=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        s.task=None;s.js={};s.info=None;s.lock=threading.Lock();s.frames=PairBuffer()
        s.wrench={};s.hstate={};s.hvel={'l':(),'r':()};s.hpos={'l':(),'r':()};s.hand_log_t={}
        cb=s.guarded   # every callback body runs inside guarded(): a bad message is logged and dropped
        s.create_subscription(String,'/sim/task',cb(s.on_task),lat)
        s.create_subscription(JointState,'/joint_states',cb(s.on_js),20)
        s.create_subscription(Image,'/cam_head/color/image_raw',cb(s.frames.add_rgb),qos_profile_sensor_data)
        s.create_subscription(Image,'/cam_head/depth/image_raw',cb(s.frames.add_depth),qos_profile_sensor_data)
        s.create_subscription(CameraInfo,'/cam_head/depth/camera_info',cb(lambda m:setattr(s,'info',m)),10)
        # torso camera: only for the table near edge (#645: measured, not stored); intrinsics from its camera_info (#613)
        s.frames_torso=PairBuffer();s.info_torso=None
        s.create_subscription(Image,'/cam_torso/color/image_raw',cb(s.frames_torso.add_rgb),qos_profile_sensor_data)
        s.create_subscription(Image,'/cam_torso/depth/image_raw',cb(s.frames_torso.add_depth),qos_profile_sensor_data)
        s.create_subscription(CameraInfo,'/cam_torso/depth/camera_info',cb(lambda m:setattr(s,'info_torso',m)),10)
        # #595.4: /robot_description is published by robot_state_publisher, QoS TRANSIENT_LOCAL
        s.robot_description=None
        s.create_subscription(String,'/robot_description',cb(lambda m:setattr(s,'robot_description',m.data)),lat)
        for k,n in SIDE_NAME.items():
            s.create_subscription(WrenchStamped,f'/arm_6dof_{n}',cb(lambda m,k=k:s.wrench.__setitem__(k,GRIP.force_vec(m.wrench))),qos_profile_sensor_data)
            s.create_subscription(JointState,f'/inspire_hand/state/{n}_hand',cb(lambda m,k=k:s.on_hand(k,m)),10)
        s.pub={g:s.create_publisher(CmdSetMotorPosition,f'/{g}/cmd_pos',10) for g in ('arm','head','waist')}
        s.hand={k:s.create_publisher(JointState,f'/inspire_hand/ctrl/{k}_hand',10) for k in ('left','right')}
        s.cli_force={k:s.create_client(SetForce,f'/inspire_hand/set_force/{n}_hand') for k,n in SIDE_NAME.items()}
        s.cli_speed={k:s.create_client(SetSpeed,f'/inspire_hand/set_speed/{n}_hand') for k,n in SIDE_NAME.items()}
        s.done=s.create_publisher(Empty,'/sim/task/done',10)
        s.tfb=tf2_ros.Buffer();s.tfl=tf2_ros.TransformListener(s.tfb,s)
        s.wall_ready=time.monotonic()   # subscriptions exist from here: later waiting for /clock costs no sim time
        s.hand_target={'l':None,'r':None};s.t0=None;s.trays={};s.placed={};s.plane=None;s.table_x=None;s.monitor=GRIP.GraspMonitor()
    def guarded(s,fn):
        """Callback wrapper: an exception inside fn is logged and the message dropped, so the executor thread keeps
        spinning (rclpy re-raises callback exceptions out of spin, which would stop /clock and every other topic)."""
        def wrapped(*a):
            try:return fn(*a)
            except Exception as e:
                try:s.log(f'callback {getattr(fn,"__name__","?")} dropped a message: {type(e).__name__}: {e}')
                except Exception:pass
        return wrapped
    def on_task(s,m):
        """/sim/task callback: the task is accepted only as a JSON object; anything else is logged and ignored, and
        run() keeps waiting for a valid task."""
        t=json.loads(m.data)
        if not isinstance(t,dict):raise ValueError(f'/sim/task is not a JSON object: {type(t).__name__}')
        s.task=t
    def on_js(s,m):
        """/joint_states callback: store the latest position (rad) per joint name."""
        with s.lock:s.js.update(zip(m.name,m.position))
    def on_hand(s,k,m):
        """Hand state callback for side k: latest state and short position/velocity histories."""
        # The history is replaced by a new tuple (one atomic assignment), never appended + trimmed in place: a reader
        # on the task thread (wait_fingers -> fingers_stalled) iterating a list while this callback trims it
        # could skip samples (test_topics_motion.HandHistory).
        s.hstate[k]=m;s.hvel[k]=(tuple(s.hvel[k])+(tuple(m.velocity),))[-10:]
        try:t=s.now()
        except Exception:t=None
        if t is not None:s.hpos[k]=(tuple(s.hpos[k])+((t,tuple(m.position)),))[-60:]
    def hand_log(s,side,phase,force=False,**extra):
        """Hand state (position/velocity/effort + command) and the wrist force at <= 10 Hz (HAND_LOG_DT, sim time) during
        guarded phases. The force is logged only: v9b Ecoforms reached |F| 26 N before the finger guard tripped, and the
        approach force profile is needed before a force guard can be calibrated."""
        t=s.now()
        if not force and t-s.hand_log_t.get(side,-1e9)<HAND_LOG_DT:return
        s.hand_log_t[side]=t;h=s.hstate.get(side)
        r=lambda v:None if v is None else [round(float(x),3) for x in v]
        rec=dict(t=round(s.budget.elapsed(t),2) if getattr(s,'budget',None) else round(t,2),phase=phase,side=side,
                 pos=r(None if h is None else h.position),vel=r(None if h is None else h.velocity),
                 eff=r(None if h is None else h.effort),cmd=r(s.hand_target.get(side)),F=r(s.wrench.get(side)))   # F: log only
        rec.update(extra);s.log('HAND',json.dumps(rec))
    def now(s):
        """Current ROS time in seconds (sim time with use_sim_time)."""
        return s.get_clock().now().nanoseconds*1e-9
    def sleep(s,dt):
        """Sleep dt seconds of ROS time."""
        t=s.now()
        while s.now()-t<dt:time.sleep(.01)
    def log(s,*a):
        """Log the arguments joined by spaces at INFO level."""
        s.get_logger().info(' '.join(str(x) for x in a))
    # Actuation
    def cmd(s,group,ids,q,spd):
        """Publish one position command: motor ids, targets (rad), speeds clipped to [1e-3, 0.8] rad/s."""
        m=CmdSetMotorPosition()
        for i,p,v in zip(ids,q,spd):
            c=SetMotorPosition();c.name=int(i);c.pos=float(p);c.spd=float(min(.8,max(v,1e-3)));c.cur=1.0;m.cmds.append(c)
        s.pub[group].publish(m)
    def q_now(s,side):
        """Measured arm joints from /joint_states. A joint that never arrived must not read as 0 rad (the planner would
        start from an invented posture); it is an error (JointStateMissing, logged) and the caller does not move
        this arm. s.js keeps the last received value of every joint, so a joint missing in one message keeps its last
        known value."""
        names=[f'{n}_{side}_joint' for n in ARMMOD.ARM]
        with s.lock:
            miss=[n for n in names if n not in s.js];q=None if miss else np.array([float(s.js[n]) for n in names])
        if miss:
            s.log('joint_states: no value for',miss,'-> arm',side,'not moved')
            raise JointStateMissing(f'no /joint_states value for {miss}')
        return q
    def move_arm(s,side,q,tol=.02,timeout=None,refine=2,vmax=.8,guard=None):
        """guard: optional callable polled every cycle; a truthy return stops the arm where it is (hold command at the
        measured joints), stores it in s.guard_trip and returns False (guarded approach)."""
        vmax=min(.8,vmax)
        q=np.asarray(q,float);q0=s.q_now(side);dq=np.abs(q-q0);m=max(dq.max(),1e-3)
        spd=np.maximum(vmax*dq/m,.05*vmax/.8)     # synchronized: every joint arrives together (straight joint-space line)
        T=timeout or (m/vmax*1.6+MOVE_T_PAD)
        s.cmd('arm',IDS[side],q,spd);t=s.now();goal=q.copy()
        for k in range(refine+1):
            while s.now()-t<T:
                e=q-s.q_now(side)
                if np.abs(e).max()<tol:return True
                if guard is not None:
                    g=guard()
                    if g:
                        s.cmd('arm',IDS[side],s.q_now(side),np.full(7,.3));s.guard_trip=g;return False
                time.sleep(.02)
            e=q-s.q_now(side)
            if k<refine:   # PD undershoot under gravity: bias the target by the residual (bounded by joint limits)
                goal=np.clip(goal+e,s.arms[side].chain.lower,s.arms[side].chain.upper);s.cmd('arm',IDS[side],goal,np.full(7,.3));t=s.now();T=REFINE_T
        return np.abs(q-s.q_now(side)).max()<.06
    def hand_set(s,side,v):
        """Command the six hand DOF of one side (normalised positions, 1 = open)."""
        s.hand_target[side]=list(v);m=JointState();m.name=['1','2','3','4','5','6'];m.position=[float(x) for x in v];s.hand[SIDE_NAME[side]].publish(m)
    def hand_open(s,side):
        """Open the fingers and rotate the thumb into opposition (DOF 6 = 0)."""
        s.hand_set(side,[1,1,1,1,1,0.])      # fingers open, thumb rotated into opposition (DOF 6 = 0)
    def fingers_open(s,side,min_open=RELEASE_OPEN_MIN):
        """/inspire_hand/state position is a fraction, 1.0 open / 0.0 closed (INTERFACES.md); v7 releases read 0.96-0.97."""
        h=s.hstate.get(side)
        if h is None or len(h.position)<4:return None
        return bool(np.mean(np.asarray(h.position,float)[:4])>=min_open)
    @staticmethod
    def closed_target(kind):
        """Phase-1 closing command per grasp kind (DOF 1..6; 1 open, 0 closed; DOF 6 thumb rotation stays 0 =
        opposition). side: whole hand; rim/top pinch: index+middle against the thumb."""
        return [0,0,0,0,0,0.] if kind=='side' else [1,1,.1,.1,.2,0.]
    def wait_thumb(s,side,tmax=None):
        """#648: the thumb is rotated into opposition first (hand_open sets DOF 6), the other fingers close only
        after /inspire_hand/state shows DOF 6 there. Returns True/False (timeout) / None (no state)."""
        tmax=GRIP.finger_stroke_time()+.5 if tmax is None else tmax;t=s.now()
        while s.now()-t<tmax:
            h=s.hstate.get(side)
            r=GRIP.thumb_rotated(None if h is None else list(h.position))
            if r:return True
            time.sleep(.02)
        h=s.hstate.get(side);return None if h is None else False
    def grasp_close(s,side,kind):
        """#648 two-phase close: (1) towards the closed target at the reduced speed until the fingers stop on the
        object; (2) re-target 'slightly past' the measured contact (GRIP.CLOSE_PAST). Returns dict(contact, target2,
        empty, settled). empty=True: the fingers ran to the closed target (nothing grasped) -> caller opens without
        lifting (untouched object = 0 points)."""
        t0=s.closed_target(kind);s.hand_set(side,t0)
        settled=s.wait_fingers(side,log_phase='close1')
        h=s.hstate.get(side);pc=None if h is None else [float(x) for x in h.position]
        if pc is None:return dict(contact=None,target2=None,empty=None,settled=settled,closed=t0)
        empty=GRIP.contact_empty(pc,t0);jam=GRIP.jammed(pc,t0)
        if jam:s.log('jammed fingers (bent back past open, not re-targeted):',jam)
        t2=None
        if not empty and GRIP.CLOSE_PAST>0:
            t2=GRIP.contact_target(pc,t0);s.hand_set(side,t2);s.wait_fingers(side,tmin=.2,log_phase='close2')
        return dict(contact=pc,target2=t2,empty=empty,settled=settled,closed=t0,jam=jam)
    def wait_fingers(s,side,tmin=.3,tmax=None,log_phase=None):
        """Until the fingers stop (blocked by the object or at target), by position (a velocity criterion
        fired in only 3 of 16 v7+v8 closes): every finger moved < GRIP.STALL_TOL over GRIP.STALL_WINDOW s, on samples newer
        than the command. Default timeout = one full stroke at the configured finger speed + 0.5 s (speed 0.5 -> 2.1 s).
        log_phase: hand state logged at 10 Hz meanwhile."""
        tmax=GRIP.finger_stroke_time()+.5 if tmax is None else tmax
        t=s.now();s.hvel[side]=()
        while s.now()-t<tmax:
            if log_phase:s.hand_log(side,log_phase)
            if s.now()-t>=tmin and GRIP.fingers_stalled(s.hpos[side],s.now(),since=t):return True
            time.sleep(.02)
        return False
    def hand_requests(s):
        """(service kind, side) -> request with the configured ratios (bodyctrl_msgs/srv/SetForce, SetSpeed)."""
        out={}
        for k in 'lr':
            r=SetForce.Request()
            for i,v in enumerate(GRIP.force_vector()):setattr(r,f'force{i}_ratio',float(v))
            out[('force',k)]=r
            r=SetSpeed.Request()
            for i in range(6):setattr(r,f'speed{i}_ratio',float(GRIP.HAND_SPEED_RATIO))
            out[('speed',k)]=r
        return out
    def hand_settings_step(s,wall=None):
        """One non-blocking delivery pass over the four hand requests. State per (kind, side) in s.hand_cfg:
        'waiting' (service not discovered yet), 'sent' (request out, no reply yet), 'confirmed' (the service replied),
        'unconfirmed' (HAND_CFG_SENDS_MAX sends without a reply). Returns True when nothing is left to do."""
        wall=time.monotonic() if wall is None else wall
        if not hasattr(s,'hand_cfg'):
            s.hand_cfg={key:dict(state='waiting',sends=0,t_sent=None,future=None,t0=wall) for key in s.hand_requests()}
        reqs=s.hand_requests();busy=False
        for (kind,side),c in s.hand_cfg.items():
            if c['state'] in ('confirmed','unconfirmed'):continue
            cli=(s.cli_force if kind=='force' else s.cli_speed)[side];f=c['future']
            if f is not None and f.done():
                try:res=f.result()
                except Exception as e:res=None;s.log('hand settings',SIDE_NAME[side],kind,'call failed:',type(e).__name__)
                if res is not None:
                    c['state']='confirmed';c['future']=None
                    fields=getattr(res,'get_fields_and_field_types',None)
                    reply={n:getattr(res,n,None) for n in fields()} if callable(fields) else {}
                    s.log('hand settings',SIDE_NAME[side],kind,'confirmed after',c['sends'],'send(s),',
                          round(wall-c['t0'],1),'s wall, reply',json.dumps(reply,default=str))
                    continue
                c['future']=None
            if c['state']=='sent' and c['future'] is not None and wall-c['t_sent']<HAND_CFG_REPLY_S:busy=True;continue
            if c['sends']>=HAND_CFG_SENDS_MAX:
                c['state']='unconfirmed';s.log('hand settings',SIDE_NAME[side],kind,'sent',c['sends'],'times, no reply');continue
            busy=True
            if not cli.service_is_ready():continue
            c['future']=cli.call_async(reqs[(kind,side)]);c['sends']+=1;c['t_sent']=wall;c['state']='sent'
            vals=[round(getattr(reqs[(kind,side)],f'{kind}{i}_ratio'),3) for i in range(6)]
            s.log('hand settings',SIDE_NAME[side],kind,vals,'send',c['sends'],'service seen after',round(wall-c['t0'],1),'s wall')
        return not busy
    def hand_settings(s,stop=None):
        """Deliver SetForce/SetSpeed to both hands in the background: retry until every service has replied (or the
        send limit is reached), never blocking the run. stop: threading.Event that ends the thread."""
        s.hand_settings_step()
        stop=stop or threading.Event();s.hand_cfg_stop=stop
        def loop():
            while not stop.is_set():
                try:
                    if s.hand_settings_step():return
                except Exception as e:s.log('hand settings: pass failed',type(e).__name__,e)
                stop.wait(HAND_CFG_POLL_S)
        th=threading.Thread(target=loop,daemon=True);th.start();s.hand_cfg_thread=th;return th
    def hand_force_applied(s,side):
        """True when this hand's SetForce request reached the simulator's service (sent while the service was
        discovered, replied or not). v9b: the right hand's sim-side confirmation line followed such a send."""
        c=getattr(s,'hand_cfg',{}).get(('force',side))
        return c is not None and c['state']!='waiting'
    def hand_cfg_summary(s):
        """Delivery state of the set_force / set_speed requests per hand."""
        return {f"{SIDE_NAME[k]}_{kind}":c['state'] for (kind,k),c in sorted(getattr(s,'hand_cfg',{}).items())}
    def hand_verdict(s,side,grasp=None):
        """Held iff a closing finger deviates >= GRIP.HELD_DEV from its current command with effort >=
        GRIP.HELD_EFF (15/15 on v7+v8). Wrist F/T, past_verdict and the v7 finger_verdict are log-only
        (hand_shadow)."""
        h=s.hstate.get(side)
        if h is None:return None
        t0=(grasp or {}).get('closed') or s.closed_target('side')
        return GRIP.held_rule(list(h.position),list(h.effort),s.hand_target.get(side),t0)
    def hand_shadow(s,side,grasp=None):
        """Superseded verdicts, logged only for comparison: past_verdict (after the second closing step) and finger_verdict."""
        h=s.hstate.get(side)
        if h is None:return dict(past=None,v7=None)
        t0=(grasp or {}).get('closed') or s.hand_target.get(side)
        past=None if not grasp or grasp.get('target2') is None else GRIP.past_verdict(list(h.position),grasp['target2'],grasp['closed'])
        return dict(past=past,v7=GRIP.finger_verdict(list(h.position),list(h.effort),t0))
    def look(s,pitch=HEAD_PITCH,tol=HEAD_TOL,tmax=HEAD_TMAX,yaw=None):
        """Command the head pitch (and yaw, motor 3, if given) and WAIT until /joint_states shows it there and still
        (v6: the first observation was taken while the head was still rising from 0 -> objects outside the view / TF
        mismatch -> nothing picked). Returns the sim time at which the head was settled (frames must be newer)."""
        if yaw is None:s.cmd('head',[2],[pitch],[.8])
        else:yaw=float(np.clip(yaw,-1.5,1.5));s.cmd('head',[2,3],[pitch,yaw],[.8,.8])
        t=s.now();prev=None
        def cur():
            with s.lock:return s.js.get('head_pitch_joint'),s.js.get('head_yaw_joint')
        while s.now()-t<tmax:
            q,qy=cur()
            ok=q is not None and abs(q-pitch)<tol and prev is not None and abs(q-prev[0])<.002
            if yaw is not None:ok=ok and qy is not None and abs(qy-yaw)<2*tol and prev[1] is not None and abs(qy-prev[1])<.002
            if ok:
                s.log('head settled',round(q,3),'' if yaw is None else f'yaw {round(qy,3)}','after',round(s.now()-t,2),'s');return s.now()
            prev=(q,qy);s.sleep(.1)
        q,qy=cur()
        s.log('head NOT settled',q,qy,'target',pitch,yaw);return s.now()
    def trace(s,phase,side,name,T_plan=None,**extra):
        """One JSON line per phase: sim time, wrist force (hand weight already subtracted by the sim),
        finger position/effort, measured vs planned TCP. Saved frames only when TRACE_DIR is set (offline tests)."""
        A=s.arms[side];qm=np.clip(s.q_now(side),A.chain.lower,A.chain.upper);tcp=A.tcp(qm)[:3,3]
        F=s.wrench.get(side);h=s.hstate.get(side)
        rec=dict(t=round(s.budget.elapsed(s.now()),2),phase=phase,side=side,obj=name,
                 F=None if F is None else [round(float(x),3) for x in F],
                 dF=None if F is None or s.monitor.f0 is None else round(float(np.linalg.norm(F-s.monitor.f0)),3),
                 fing=None if h is None else [round(float(x),3) for x in h.position],
                 eff=None if h is None else [round(float(x),3) for x in h.effort],
                 tcp=[round(float(x),4) for x in tcp],q=[round(float(x),3) for x in qm])
        if T_plan is not None:rec['tcp_plan']=[round(float(x),4) for x in T_plan[:3,3]];rec['tcp_err']=round(float(np.linalg.norm(tcp-T_plan[:3,3])),4)
        rec.update(extra);s.log('TRACE',json.dumps(rec))
        if TRACE_DIR and phase in ('closed','lifted','released'):
            try:
                pair=s.frames.pair(-1e18)
                if pair is not None:
                    r,d=pair;os.makedirs(TRACE_DIR,exist_ok=True)
                    np.savez_compressed(os.path.join(TRACE_DIR,f"{rec['t']:07.2f}_{phase}_{side}_{name[:20]}.npz"),
                        rgb=np.frombuffer(bytes(r.data),np.uint8).reshape(r.height,r.width,3),
                        depth=np.frombuffer(bytes(d.data),np.uint16).reshape(d.height,d.width),
                        head_pitch=s.js.get('head_pitch_joint',np.nan),q=qm,rec=json.dumps(rec))
            except Exception as e:s.log('trace save failed',e)
    # Perception
    def tf(s,target,source,stamp=None):
        """Transform at the image stamp (head may move); latest available if that time is not buffered."""
        to=rclpy.duration.Duration(seconds=1.)
        tr=None
        if stamp is not None:
            try:tr=s.tfb.lookup_transform(target,source,rclpy.time.Time(seconds=stamp.sec,nanoseconds=stamp.nanosec),timeout=to)
            except Exception as e:s.log('tf at stamp failed, using latest:',type(e).__name__)
        if tr is None:tr=s.tfb.lookup_transform(target,source,rclpy.time.Time(),timeout=to)
        t=tr.transform.translation;r=tr.transform.rotation
        return quat_to_T([t.x,t.y,t.z],[r.x,r.y,r.z,r.w])
    def capture(s,buf=None,info_attr='info',default_frame='camera_head_link',not_before=None,wait=5.):
        """Newest synchronised RGB-D pair newer than not_before -> (P pelvis-frame HxWx3, rgb, valid). Intrinsics
        from camera_info only (#613), extrinsics from /tf at the image stamp."""
        buf=s.frames if buf is None else buf
        t_req=s.now() if not_before is None else not_before;t=time.monotonic();pair=None
        while time.monotonic()-t<wait and (pair is None or getattr(s,info_attr) is None):
            pair=buf.pair(t_req);time.sleep(.02)
        if pair is None:
            # The newest pair of any age could predate the last head/arm motion, so
            # only a pair at most CAPTURE_MAX_AGE older than the request is accepted; otherwise the capture fails and
            # observe_retry captures again.
            pair=buf.pair(t_req-CAPTURE_MAX_AGE)
            if pair is not None:s.log('capture: no pair newer than the request, using one <=',CAPTURE_MAX_AGE,'s older')
            else:s.log('capture: no RGB-D pair within',CAPTURE_MAX_AGE,'s of the request (stale or no camera)')
        info=getattr(s,info_attr)
        if pair is None or info is None:raise RuntimeError('no synchronised RGB-D pair / camera_info')
        rgbm,dm=pair
        rgb=np.frombuffer(bytes(rgbm.data),np.uint8).reshape(rgbm.height,rgbm.width,3)
        if getattr(rgbm,'encoding','')=='bgr8':rgb=rgb[...,::-1]   # tray colour (red/blue) must not swap; rgb8 unchanged
        dmm=np.frombuffer(bytes(dm.data),np.uint16).reshape(dm.height,dm.width).astype(float)
        K=np.array(info.k).reshape(3,3);T=s.tf('pelvis',dm.header.frame_id or default_frame,dm.header.stamp)
        P=deproject(dmm,K,T)   # X=(u-cx)Z/fx, Y=(v-cy)Z/fy, Z=mm/1000, pelvis = R@[X,Y,Z]+t (Rust kernel or NumPy reference)
        return P,rgb,dmm>0
    def measure_plane(s,P,valid):
        """Table plane in the search window (approximate bounds, #645) + height-band plausibility."""
        tb0=table_region(P,valid)   # valid & .15<x<.75 & |y|<.7
        plane=table_plane(P,tb0)
        if not ARMMOD.plane_plausible(plane):raise ValueError('plane outside the table height search band')
        return plane
    def tray_mask(s,P):
        """Pixels inside any detected tray rectangle (+margin): trays are not objects."""
        R=[]
        for t in s.trays.values():
            x,y,ax,ay,a,b,_=ARMMOD.tray_geom(t);v=np.array([ax,ay]);u=np.array([v[1],-v[0]])
            R.append((x,y,u,v,a/2+TRAY_MASK_MARGIN,b/2+TRAY_MASK_MARGIN))
        return rect_mask(P,R)   # |dx*u0+dy*u1| < a/2+margin & |dx*v0+dy*v1| < b/2+margin, any tray (Rust or NumPy)
    def observe(s,not_before=None):
        """One observation: capture, table plane, tray refinement, segments. Returns (P, rgb, plane, detections)."""
        P,rgb,valid=s.capture(not_before=not_before)
        plane=s.measure_plane(P,valid);s.plane=plane
        s.locate_trays(P,rgb,valid,plane,refine_only=True)
        valid=valid&~s.tray_mask(P)
        tb=table_region(P,valid)
        roi=(np.array([.18,-.62,-1.]),np.array([.62,.62,1.]))
        dets=segment(P,rgb,tb,plane,roi,min_pixels=40)
        return P,rgb,plane,dets
    def tray_names(s):
        """Trays the task uses (categories -> tray) that have a known colour (TASKS.md / /sim/task['trays'])."""
        t=s.task or {};colours=t.get('trays',{})
        need=set((t.get('categories') or {}).values()) or set(colours)
        return {n:colours.get(n) for n in need if colours.get(n) in TR.CHANNEL}
    def locate_trays(s,P,rgb,valid,plane,refine_only=False):
        """Detect trays in this frame. Only full (non-partial) detections are stored. refine_only: update trays
        already found (the main view) without adding new ones from a possibly occluded view.
        Returns {name: detection-or-None} for this frame."""
        out={}
        for name,col in s.tray_names().items():
            if refine_only and name not in s.trays:continue
            d=TR.detect(P,rgb,valid,name,col,plane);out[name]=d
            if d is not None and not d['partial']:
                prev=s.trays.get(name)
                if prev is None or d['px']>=.8*prev.get('px',0):
                    if prev is None:s.log('tray',name,col,'camera',[round(x,3) for x in d['center']],'size',[round(x,3) for x in d['size']])
                    s.trays[name]=d
        return out
    def survey(s,pitch=HEAD_PITCH,yaws=HEAD_SCAN_YAWS):
        """#645 scene survey: table plane + near edge (head + torso cameras) and every needed tray, by head yaw scan
        and re-aiming at partial detections. Returns the sim time when the head is back at yaw 0."""
        need=s.tray_names();partial={}
        for yaw in yaws:
            if all(n in s.trays for n in need):break
            s.look(pitch,yaw=yaw)
            try:
                P,rgb,valid=s.capture(not_before=s.now());plane=s.measure_plane(P,valid)
                if s.plane is None or abs(yaw)<1e-6:s.plane=plane
                for n,d in s.locate_trays(P,rgb,valid,plane).items():
                    if d is not None and d['partial'] and n not in s.trays:partial[n]=d
            except Exception as e:s.log('survey view failed',yaw,e)
        for _ in range(TRAY_RELOOK_MAX):
            todo=[n for n in need if n not in s.trays and n in partial]
            if not todo:break
            n=todo[0];cx,cy=partial.pop(n)['centroid']
            hy=ARMMOD.fk_link('head_yaw_link')[:3,3]     # robot geometry: the head yaw axis position in pelvis
            s.look(pitch,yaw=float(np.arctan2(cy-hy[1],cx-hy[0])))
            try:
                P,rgb,valid=s.capture(not_before=s.now());plane=s.measure_plane(P,valid)
                for m,d in s.locate_trays(P,rgb,valid,plane).items():
                    if d is not None and d['partial'] and m not in s.trays:partial[m]=d
            except Exception as e:s.log('tray re-look failed',n,e)
        missing=[n for n in need if n not in s.trays]
        if missing:s.log('trays NOT found (their categories are skipped, never a stored pose):',missing)
        s.log('tray sources',json.dumps({n:({'source':'camera','center':[round(float(x),3) for x in s.trays[n]['center']]}
                                          if n in s.trays else {'source':'missing'}) for n in sorted(need)}))
        t=s.look(pitch,yaw=0.)
        s.measure_near_edge(t)
        return t
    def measure_near_edge(s,not_before):
        """Table near edge from the torso camera (+ head) plane inliers; conservative bound when not seen."""
        if s.plane is None:return
        edges=[]
        for buf,attr,fr in ((s.frames_torso,'info_torso','camera_body_front_link'),(s.frames,'info','camera_head_link')):
            try:
                P,_,valid=s.capture(buf,attr,fr,not_before=not_before,wait=1.)
                e=near_edge(P,valid,s.plane)
                if e is not None:edges.append(e)
            except Exception as ex:s.log('near edge',attr,type(ex).__name__)
        near=min(edges) if edges else ARMMOD.TABLE_X_NEAR_MIN
        s.table_x=(near,ARMMOD.TABLE_X_FAR);s.log('table near edge',round(near,3),'measured' if edges else 'NOT seen -> conservative bound')
    def identify(s,dets,P,plane,names):
        """Blob<->object assignment (Hungarian) with geometric priors; returns list of (det,name,category,conf,cloud)."""
        if not dets:return []
        clouds=[P[d.pixels[:,0],d.pixels[:,1]] for d in dets]
        feats=[]
        for c in clouds:
            h=np.quantile(c@plane[0]+plane[1],.98);xy=c[:,:2]-c[:,:2].mean(0)
            ev,V=np.linalg.eigh(np.cov(xy.T));pr=xy@V;e=np.sort(pr.max(0)-pr.min(0))
            feats.append((h,e[0],e[1]))
        N=[n for n in names if n in PRI]
        C=np.full((len(dets),len(N)+len(dets)),3.)   # dummy columns: unassigned (cost 3)
        if N:   # element-wise identical to the scalar double loop (same operation order; max(0,x) == np.maximum(0,x) here)
            F=np.array(feats,float).reshape(-1,3);h,w,l=F[:,:1],F[:,1:2],F[:,2:3]
            ph=np.array([PRI[n]['height'] for n in N],float);pw=np.array([PRI[n]['w_min'] for n in N],float);pl=np.array([PRI[n]['w_max'] for n in N],float)
            C[:,:len(N)]=np.abs(h-ph)/.012+np.maximum(0,w-pw-.01)/.012+np.maximum(0,pw-w-.02)/.02+np.maximum(0,l-pl-.015)/.015
        r,c=linear_sum_assignment(C);out=[]
        for i,j in zip(r,c):
            if j>=len(N):continue
            cat=PRI[N[j]]['category'];other=[C[i,k] for k in range(len(N)) if PRI[N[k]]['category']!=cat]
            conf=(min(other) if other else 9)-C[i,j]
            out.append((dets[i],N[j],cat,float(conf),clouds[i],float(C[i,j])))
        return out
    # Task
    def run(s):
        """Whole task: hand settings, wait for /sim/task and /clock, survey, pick-and-place loop, safe end, done."""
        s.hand_settings()     # background delivery starts before the task arrives (service discovery takes wall time)
        while s.task is None or s.now()<=0:time.sleep(.05)
        # v7 s101: 'startup 206.9 s' -- the wait for the (not yet running) sim clock was counted as sim time.
        # Only process start-up before the node could receive /clock is charged (pessimistically, RTF<=1).
        t0=BUD.start_time(s.now(),getattr(s,'wall_ready',time.monotonic()),WALL_LAUNCH)
        task=s.task;s.log('task',task.get('task'),task.get('objects_from'))
        if task.get('task') not in BUD.LIMITS:
            s.publish_done();return
        s.budget=BUD.TimeBudget(BUD.LIMITS[task['task']],t0)
        handled=set();fails={};why='time budget';s.arms={}
        s.failed_grasps={}   # name -> [(kind, centre xy, closing axis xy)] of executed attempts that failed (grasp.demote_failed); kept across retry rounds
        try:
            # Inside the guarded block: a malformed task message (no 'categories'/'objects') or an arm model that
            # fails to build must not end the process before /sim/task/done (the robot has not moved yet)
            cats=task['categories'];objs=task['objects']
            s.use_robot_description()
            s.arms={k:ARMMOD.Arm(k) for k in 'lr'}
            s.log('startup',round(s.budget.elapsed(s.now()),1),'s of',s.budget.limit)
            s.log('accel',*accel_status())
            # /joint_states is required before any motion, but an unbounded wait would never publish /sim/task/done. It
            # ends as soon as no object cycle could be finished any more (the same can_start rule as the task loop),
            # which goes to end_safe (no arm motion without joint states) and publishes done inside the budget.
            while not s.js:
                if not s.budget.can_start(s.now()):raise RuntimeError('no /joint_states while an object cycle still fit the budget')
                time.sleep(.05)
            for k in 'lr':s.hand_open(k)
            # Survey first with the arms in the start posture (v7 s102 start frames: table plane, near edge and
            # trays are measured with the arms hanging), then go home along a path certified against the measured
            # table. v9a: going home before the table was known commanded the straight start->home joint path, which
            # sweeps the hands through the table front -> both arms jammed (|F| 258 N / 938 N) and 4 of 7 picks were lost.
            t_head=s.survey()
            if s.plane is not None:
                for A in s.arms.values():A.set_scene(s.plane,list(s.trays.values()),getattr(s,'table_x',None))
            for k in 'lr':s.go_home(k)     # from the start posture along a certified path (planning assumes home)
            t_head=max(t_head,s.now())
            empty=0;nofeas={};resurveys=0   # nofeas[name]=len(handled) when no pick+place plan existed
            retries=0
            s.last_contact=False
            while s.budget.can_start(s.now()):
                try:P,rgb,plane,dets=s.observe_retry(t_head)
                except Exception as e:s.log('observe failed',e);why='observe failed';break
                t_head=s.now()                   # later observations: frames newer than the end of the last motion
                if plane is not None:
                    for A in s.arms.values():A.set_scene(plane,list(s.trays.values()),s.table_x)   # measured table + near edge + tray walls
                allid=s.identify(dets,P,plane,[n for n in objs if n not in handled])
                s.log('observe t',round(s.budget.elapsed(s.now()),1),'dets',len(dets),'assigned',
                      [(x[1][:18],round(x[3],2),round(x[5],2)) for x in allid])
                ids=[x for x in allid if x[3]>ID_CONF_MIN and x[5]<ID_COST_MAX]
                ids.sort(key=lambda x:(heavy(x[1]),-x[3]))     # light objects first (HEAVY_KG), then most confident
                did=False;stop=False;touched=False
                for det,name,cat,conf,cloud,cost in ids:
                    if fails.get(name,0)>=FAILS_MAX:continue
                    if not replan_infeasible(nofeas,name,handled):   # same scene: the deterministic planner fails again
                        s.log('no plan:',name,json.dumps({'reason':'unchanged scene since the last failed planning'}));continue
                    cat=objs.get(name,cat)        # category from /sim/task is authoritative
                    tray=cats.get(cat)
                    if tray not in s.trays:
                        s.log('no plan:',name,json.dumps({'reason':'tray not found by the cameras','tray':tray}));continue
                    others=[P[d.pixels[:,0],d.pixels[:,1]] for d in dets if d is not det]   # every segment is an obstacle
                    t=s.now();s.last_infeasible=False;s.last_contact=False;s.last_attempt=None
                    try:ok=s.pick_place(name,cat,cloud,plane,others,tray)
                    except Exception as e:      # recover (hands open, arms home) and go on with a fresh observation
                        s.log('pick failed with exception',name,type(e).__name__,e);s.recover();ok=False;s.last_contact=True
                    remember_failure(s.failed_grasps,name,ok,s.last_attempt)
                    if ok is False and s.last_infeasible:nofeas[name]=len(handled)
                    if ok is None:stop=True;break           # budget: no time to finish another object
                    if ok:handled.add(name);did=True;s.budget.record_cycle(s.now()-t);break
                    fails[name]=fails.get(name,0)+1
                    if s.last_contact:           # the hand touched the scene -> the other clouds are stale
                        s.log('re-observe after contact',name);touched=True;break
                if stop:why='time budget (next object would not finish)';break
                if not did and resurveys<TRAY_RESURVEY_MAX and any(cats.get(objs.get(x[1],x[2])) not in s.trays for x in ids):
                    resurveys+=1;s.log('needed tray missing: re-survey');t_head=s.survey();continue
                empty=0 if (did or touched) else empty+1
                if empty>=EMPTY_OBS_MAX:
                    # v9a: the run ended at 124 s of 405 s because every remaining object had 2 failures (mostly the
                    # jammed-arm aborts) or was not identified. One more round with fresh counters while a whole cycle
                    # still fits (the -10 end-of-run rule is enforced per object by can_start / place_time anyway).
                    left=[n for n in objs if n not in handled and fails.get(n,0)>=FAILS_MAX]
                    if retries<RETRY_ROUNDS and left and s.budget.can_start(s.now()):
                        retries+=1;s.log('retry round',retries,'objects',[n[:24] for n in left]);fails={};nofeas={};empty=0;continue
                    why='nothing pickable in %d observations'%empty;break
        except Exception as e:   # an unexpected error must not skip the end-of-run release/retreat/done below
            why=f'exception {type(e).__name__}: {e}';s.log('run loop failed:',why)
        s.end_safe()
        if getattr(s,'hand_cfg_stop',None) is not None:s.hand_cfg_stop.set()
        s.publish_done();s.log('done at',round(s.budget.elapsed(s.now()),1),'handled',sorted(handled),'reason:',why,
                                      'fails',json.dumps({k[:24]:v for k,v in fails.items()}))
        s.log('accel at end',*accel_status())
        s.log('hand settings at end',json.dumps(s.hand_cfg_summary()))
    def publish_done(s):
        """Publish /sim/task/done at most once per run. Returns True when this call published it (main() calls it
        again as a last resort after an error outside run()'s guarded task loop)."""
        if getattr(s,'done_sent',False):return False
        s.done.publish(Empty());s.done_sent=True;return True
    def end_safe(s):
        """End of run: both arms up and home, waist back to 0. Every step is guarded on its own, so a failing step
        (planner, TF, a missing arm model) cannot keep /sim/task/done from being published."""
        for k in 'lr':
            if k not in (getattr(s,'arms',None) or {}):continue   # no arm model built (failure before the robot moved)
            try:s.retreat(k)
            except Exception as e:s.log('end: retreat failed',k,type(e).__name__,e)
        try:
            if abs(s.js.get('body_yaw_joint',0.))>.05:s.waist(0.)
        except Exception as e:s.log('end: waist failed',type(e).__name__,e)
    def recover(s):
        """After an exception inside pick_place: open both hands, arms up and home, waist back. The object state is
        unknown; an object still in the hand at the end or lifted and left outside a tray scores -10 either way, an
        object closed on but not lifted scores 0 only if the hand opens."""
        for k in 'lr':
            try:s.hand_open(k)
            except Exception as e:s.log('recover: open failed',k,type(e).__name__,e)
        for k in 'lr':
            try:
                if s.fingers_open(k) is False:s.wait_fingers(k,tmin=.4,tmax=RELEASE_WAIT)
            except Exception as e:s.log('recover: wait failed',k,type(e).__name__,e)
        s.end_safe()
    def observe_retry(s,not_before):
        """observe(), once more on a failure (a single failed capture or plane fit must not end the whole run)."""
        try:return s.observe(not_before)
        except Exception as e:s.log('observe failed, retrying once:',type(e).__name__,e)
        return s.observe(s.now())
    def use_robot_description(s,wait=1.):
        """#595.4: prefer the URDF the simulator publishes on /robot_description (TRANSIENT_LOCAL, latched). It is
        used only if it contains every link the planner needs; otherwise the bundled copy of the same organiser URDF
        stays (logged). Robot geometry is explicitly allowed (#645)."""
        t=time.monotonic()
        while s.robot_description is None and time.monotonic()-t<wait:time.sleep(.02)
        x=s.robot_description
        if not x:s.log('robot_description: not received, bundled URDF');return False
        try:
            ok,why=ARMMOD.load_urdf(x)
            s.log('robot_description:', 'in use' if ok else f'bundled URDF kept ({why})');return ok
        except Exception as e:
            s.log('robot_description: parse failed, bundled URDF kept',type(e).__name__);return False
    def waist(s,theta,tol=.02,tmax=WAIST_TMAX):
        """body_yaw_joint (motor 31) to theta at <=0.8 rad/s; waits on /joint_states."""
        s.cmd('waist',[31],[theta],[.8]);t=s.now()
        while s.now()-t<tmax:
            with s.lock:q=s.js.get('body_yaw_joint')
            if q is not None and abs(q-theta)<tol:return True
            time.sleep(.02)
        s.log('waist not settled',s.js.get('body_yaw_joint'),'target',theta);return False
    def go_home(s,side):
        """#625: home only along a certified (self-, body-, other-arm- and table-collision-free) joint path from the
        measured posture: straight, or straight up first. If none exists the arm is moved anyway (it cannot stay in
        a low pose) and the event is logged for the trajectory review."""
        A=s.arms[side];o='r' if side=='l' else 'l'
        if o in s.arms:A.set_other(s.q_now(o))
        q=s.q_now(side);path=A.path_home(q)
        if path is None:s.log('WARNING uncertified path home',side,'from',[round(float(x),3) for x in q]);path=[A.home]
        elif len(path)>1:s.log('path home',side,'via',len(path)-1,'waypoint(s)')
        for q in path[:-1]:s.move_arm(side,q,tol=.03)
        s.move_arm(side,path[-1],tol=.05)
    def retreat(s,side):
        """Lift straight up first if the TCP is low (near table/tray walls), then go home."""
        A=s.arms[side];q=s.q_now(side)
        try:
            p=A.tcp(np.clip(q,A.chain.lower,A.chain.upper))[:3,3]
            low=True if s.plane is None else bool(p[2]<ARMMOD.plane_height(s.plane,p[0],p[1])+LOW_ABOVE_TABLE)
        except ValueError:low=True    # unknown -> the safe choice: lift first
        if low:
            qu=A.lift(np.clip(q,A.chain.lower,A.chain.upper),RETREAT_DZ)
            if qu is not None:s.move_arm(side,qu,tol=.03)
        s.go_home(side)
    def finger_dev(s,side,kind):
        """Deviation of this grasp kind's closing fingers from the current (open) command."""
        h=s.hstate.get(side)
        return GRIP.approach_deviation(None if h is None else list(h.position),s.hand_target.get(side),s.closed_target(kind))
    def approach_guard(s,side,kind):
        """Polled by move_arm on the pre-grasp -> grasp segment: hand state at 10 Hz; trip on a deflected finger."""
        s.hand_log(side,'approach')
        dev,fl=s.finger_dev(side,kind)
        return dict(dev=round(dev,3),fingers=fl) if fl else None
    def place_reached(s,A,side,Tq,tray,obj_off,yaw):
        """Release check: the TCP is within PLACE_TOL of the planned release pose, or the measured release point (object
        centre or grip point, pelvis frame) lies inside the measured tray rectangle."""
        qm=np.clip(s.q_now(side),A.chain.lower,A.chain.upper)
        if A.position_error(qm,Tq)<=PLACE_TOL:return True
        return in_tray(release_xy(A,qm,obj_off,yaw),tray)
    def release_off_tray(s,A,side,name,q_lift,q_carry,yaw,gr):
        """The release point is still off the tray after the re-approach. The object must not stay in the hand, and it
        must come down on the table, not on the floor: carry it back along the planned path (reverse order) to the lift
        pose above the spot it was picked from and open there. Without time for the way back, open where the hand is."""
        qm=np.clip(s.q_now(side),A.chain.lower,A.chain.upper)
        back=(move_time(qm,q_carry)+move_time(q_carry,q_lift) if q_carry is not None else move_time(qm,q_lift))
        back+=(2*abs(yaw)/.8+1. if yaw else 0.)+GRIP.finger_stroke_time()
        if s.hand_verdict(side,gr) is False:
            s.log('place not reached, hand already empty: open here',name);where='empty_hand'
        elif s.budget.remaining(s.now())-s.budget.safety>=back:
            s.log('place not reached: back to the pick spot to release on the table',name)
            if q_carry is not None:s.move_arm(side,q_carry,tol=.03)
            elif yaw:s.move_arm(side,q_lift,tol=.03)
            if yaw:s.waist(0.)
            s.move_arm(side,q_lift,tol=.03);where='pick_spot'
        else:
            s.log('place not reached and no time to carry back: release here',name);where='in_place'
        s.trace('release_off_tray',side,name,A.tcp(np.clip(s.q_now(side),A.chain.lower,A.chain.upper)),where=where,
                verdict=s.hand_verdict(side,gr))
        s.hand_open(side);s.wait_fingers(side,tmin=.4,tmax=RELEASE_WAIT)
        if s.fingers_open(side) is False:s.hand_open(side);s.wait_fingers(side,tmin=.4,tmax=RELEASE_WAIT)
        s.go_home(side)
        if yaw:s.waist(0.)
        s.last_contact=True;return False
    def guard_abort(s,side,name,where,info):
        """Guarded approach tripped: hand stays open, lift straight up and home, mark contact -> re-observe."""
        s.hand_log(side,'guard_'+where,force=True,**info);s.trace('guard',side,name,where=where,**info)
        s.log('guard: finger contact during',where,name,json.dumps(info),'-> no close, re-observe')
        s.last_contact=True;s.retreat(side);return False
    def pick_place(s,name,cat,cloud,plane,others,tray):
        """True placed / False failed (safe, nothing lifted left outside) / None not started for lack of time."""
        f,C=candidates(cloud,plane,cat,others,W_MAX=self_w(s),robot_xy=(0.,0.),mass=PRI.get(name,{}).get('mass'))
        C=demote_failed(C,getattr(s,'failed_grasps',{}).get(name,()))   # a grasp that already failed is tried last
        tx,ty,ax,ay,ts,tl,th=ARMMOD.tray_geom(s.trays[tray])
        order='lr' if ty>.1 else ('rl' if ty<-.1 else ('lr' if f['center'][1]>=0 else 'rl'))
        k=s.placed.get(tray,0);dl=TR.slot_offset(k,.5*f['length'],size=(ts,tl),ty=ty)   # along the measured long axis
        sx,sy=tx+dl*ax,ty+dl*ay
        why={'candidates':len(C),'feasible':int(sum(1 for g in C if g['feasible'])),'pick_ik':{},'place':{},'hand_clear':{}}   # why no plan: logged per object
        if 'rim_closed' in f:why['rim_closed']=f['rim_closed']
        if f.get('rim_torque') is not None:why['rim_torque']=round(f['rim_torque'],3)
        for side in order:
            if not s.hand_force_applied(side):   # #648: never grasp at the default force 1.0
                why.setdefault('hand_force_not_applied',[]).append(side);continue
            A=s.arms[side];A.set_other(s.q_now('r' if side=='l' else 'l'))   # #625 arm-to-arm at the measured posture
            for g in C:
                if not g['feasible']:continue
                top=-(plane[1]+plane[0][:2]@g['center'])/plane[0][2]
                if g['kind']=='side':
                    ap=np.r_[g['approach'],0.];ca=np.r_[g['close_axis'],0.];pt='grip'
                elif g['kind']=='top':
                    ap=np.asarray(g.get('approach3',[0.,0.,-1.]),float)
                    ca=np.asarray(g['close_axis'],float);ca=np.r_[ca,0.] if ca.size==2 else ca;pt='pinch'
                else:  # rim
                    ap=np.asarray(g.get('approach3',[0.,0.,-1.]),float)
                    ca=np.asarray(g['close_axis'],float);ca=np.r_[ca,0.] if ca.size==2 else ca;pt='pinch'
                p=None
                for dz in (SIDE_DZ if g['kind']=='side' else (0.,)):
                    if g['kind']=='side' and dz>0 and g['z']+dz>f['height']-.015:break
                    c3=np.array([g['center'][0],g['center'][1],top+g['z']+dz])
                    p=A.plan_pick(c3,ca,ap,g['width'],point=pt,obj=None if 'box' not in g else dict(g['box'],z0=top))
                    if p is None and getattr(A,'last_clear_fail',0):why['hand_clear'][side]=why['hand_clear'].get(side,0)+A.last_clear_fail
                    if p is not None:break
                if p is None:
                    why['pick_ik'][side]=why['pick_ik'].get(side,0)+1;continue
                yp=plan_place_yaw(A,p['q_lift'],sx,sy,plane,list(s.trays.values()),fallback=(tx,ty) if dl else None,tray_h=th,
                                  obj_off=held_offset(A,p['q_grasp'],g,c3))
                if yp is None:
                    why['place'][side]=why['place'].get(side,0)+1;continue
                yaw,place=yp
                if not s.budget.can_start(s.now(),p['duration']+(2*abs(yaw)/.8+1. if yaw else 0.)):   # waist out and back at 0.8 rad/s
                    s.log('skip',name,'remaining',round(s.budget.remaining(s.now()),1),'s');return None
                cl=p.get('clear') or {}
                s.log('pick',name,cat,'->',tray,'slot',k,'arm',side,g['kind'],'w',round(g['width'],3),'z',round(g['z'],3),
                      'clear',json.dumps({k_:(None if cl.get(k_) is None else round(cl[k_],3)) for k_ in ('du','dr','margin')}))
                s.last_attempt=(g['kind'],np.asarray(g['center'],float)[:2].copy(),np.asarray(g['close_axis'],float)[:2].copy())   # executed from here (failure memory)
                Tg=A.tcp(p['q_grasp']);s.trace('start',side,name,kind=g['kind'],center=[round(float(x),3) for x in c3])
                s.hand_open(side);s.move_arm(side,p['q_pre']);Tp=A.tcp(p['q_pre']);s.trace('pre',side,name,Tp)
                thumb=s.wait_thumb(side)      # #648: thumb in opposition before anything closes (usually already there)
                if not thumb:s.log('thumb not in opposition before the close (non-fatal)',name,side)
                if A.position_error(np.clip(s.q_now(side),A.chain.lower,A.chain.upper),Tp)>PRE_TOL:
                    s.log('pre-grasp not reached, abort',name);s.retreat(side);return False
                # Guarded approach: the open fingers must follow their command; a finger off by > GRIP.GUARD_DEV
                # is touching something -> stop, do not close, re-observe. At pre-grasp the open command is re-sent once
                # first (a deflection that survives it is a contact, not motion transient).
                dev,fl=s.finger_dev(side,g['kind'])
                if fl:
                    s.hand_log(side,'pre',force=True,dev=dev,fingers=fl);s.hand_open(side);s.wait_fingers(side,tmin=.2,tmax=1.,log_phase='pre_reopen')
                    dev,fl=s.finger_dev(side,g['kind'])
                if fl:return s.guard_abort(side,name,'pre',dict(dev=dev,fingers=fl))
                s.guard_trip=None
                s.move_arm(side,p['q_grasp'],tol=.015,vmax=APPROACH_VMAX,guard=lambda side=side,kind=g['kind']:s.approach_guard(side,kind))
                if s.guard_trip:return s.guard_abort(side,name,'approach',s.guard_trip)
                qm=s.q_now(side);s.trace('at_grasp',side,name,Tg)
                if A.position_error(np.clip(qm,A.chain.lower,A.chain.upper),Tg)>TCP_TOL:
                    qc=A.correct(np.clip(qm,A.chain.lower,A.chain.upper),p['q_grasp'])
                    if qc is None:s.log('TCP correction: no solution',name,'err',round(float(A.position_error(np.clip(qm,A.chain.lower,A.chain.upper),Tg)),4))
                    else:
                        s.move_arm(side,qc,tol=.01,vmax=APPROACH_VMAX,refine=0,guard=lambda side=side,kind=g['kind']:s.approach_guard(side,kind))
                        if s.guard_trip:return s.guard_abort(side,name,'correction',s.guard_trip)
                        s.trace('corrected',side,name,Tg)
                s.monitor.baseline(s.wrench.get(side))
                gr=s.grasp_close(side,g['kind'])
                s.trace('closed',side,name,Tg,settled=gr['settled'],thumb=thumb,contact=gr['contact'],target2=gr['target2'],
                        empty=gr['empty'],jam=gr.get('jam'),force=GRIP.HAND_FORCE_RATIO,speed=GRIP.HAND_SPEED_RATIO,past=GRIP.CLOSE_PAST)
                if gr.get('jam') or not gr['empty']:s.last_contact=True   # the hand touched the scene
                if gr['empty']:       # fingers ran to the closed target: nothing between them -> do not lift (0, not -10)
                    s.log('grasp check (contact): empty',name);s.hand_open(side);s.retreat(side);return False
                # #595 end-of-run policy: a lifted object outside a tray (or in the hand) at the limit is -10, an
                # untouched one 0 -> lift only if lift + transport + release + retreat still fit before the limit.
                need=place_time(A,p['q_lift'],place,yaw,q_from=p['q_grasp']);hold=hold_time_bound(p['q_grasp'],p['q_lift'],place,yaw)
                if not lift_allowed(s.budget.remaining(s.now()),need,hold,s.budget.safety):
                    s.log('deadline: not lifting',name,'need',round(need,1),'hold bound',round(hold,1),'s left',round(s.budget.remaining(s.now()),1))
                    s.hand_open(side);s.wait_fingers(side,tmin=.3);s.retreat(side);return None
                s.move_arm(side,p['q_lift']);s.sleep(.3)
                held=s.hand_verdict(side,gr);old=s.hand_shadow(side,gr)
                F=s.wrench.get(side);f0=s.monitor.f0
                shadow=None if F is None or f0 is None else bool(np.linalg.norm(F-f0)>=SHADOW_DF_N)
                s.trace('lifted',side,name,A.tcp(p['q_lift']),verdict=held,shadow=shadow,verdict_past=old['past'],
                        verdict_v7=old['v7'],cmd=s.hand_target.get(side),eff_thr=round(GRIP.effort_held_threshold(),3))
                if held is None:s.log('no hand state after the lift: holding verdict unknown, carrying on',name)
                if held is False:      # nothing in the hand: open, retreat, re-observe (object untouched => 0, not -10)
                    s.log('grasp check: empty',name);s.hand_open(side);s.retreat(side);return False
                q_place,q_up,q_carry=place
                if q_carry is not None:s.move_arm(side,q_carry,tol=.03);s.trace('carry',side,name,A.tcp(q_carry))
                if yaw:s.waist(yaw);s.trace('yawed',side,name,yaw=round(yaw,3))
                Tq=A.tcp(q_place);off=held_offset(A,p['q_grasp'],g,c3)
                at_tray=lambda:s.place_reached(A,side,Tq,s.trays[tray],off,yaw)
                for attempt in range(1+PLACE_RETRIES):   # one bounded re-approach when the release point is off the tray
                    if attempt:s.log('place not reached, re-approach',name);s.trace('place_retry',side,name,Tq)
                    s.move_arm(side,q_place,tol=.03)
                    qm=np.clip(s.q_now(side),A.chain.lower,A.chain.upper)
                    if A.position_error(qm,Tq)>PLACE_TOL:
                        qc=A.correct(qm,q_place,max_step=.15)
                        if qc is not None:s.move_arm(side,qc,tol=.02,refine=1)
                    if at_tray():break
                else:
                    return s.release_off_tray(A,side,name,p['q_lift'],q_carry,yaw,gr)
                old=s.hand_shadow(side,gr);over=s.hand_verdict(side,gr)
                s.trace('over_tray',side,name,Tq,tray=tray,tray_xy=[round(sx,3),round(sy,3)],yaw=round(yaw,3),
                        verdict=over,verdict_past=old['past'],verdict_v7=old['v7'])   # slip during transport shows up here
                s.hand_open(side);s.wait_fingers(side,tmin=.4,tmax=RELEASE_WAIT)
                if s.fingers_open(side) is False:s.log('fingers not open after release, re-open',name);s.hand_open(side);s.wait_fingers(side,tmin=.4,tmax=RELEASE_WAIT)
                s.trace('released',side,name,A.tcp(q_place))
                if q_up is not None:s.move_arm(side,q_up,tol=.03)
                s.go_home(side)
                if yaw:s.waist(0.)
                s.trace('home',side,name)
                if over is False:   # the hand was empty over the tray: the object slipped out during transport
                    s.log('carry check: hand empty over the tray, not counted',name);s.last_contact=True;return False
                s.placed[tray]=k+1
                return True
        if len(why.get('hand_force_not_applied',[]))==len(order):   # not a planning failure: retry once a hand is ready
            why['reason']='hand force not applied on any arm';s.log('no plan:',name,json.dumps(why));return False
        s.last_infeasible=True   # nothing executed: no candidate had a collision-free pick + place plan
        why['reason']=('dish without an open rim (closed top measured)' if f.get('rim_closed') and not C else
                       'dish too heavy for a rim pinch (m*g*r above GRASP.RIM_TORQUE_MAX)' if f.get('rim_heavy') and not C else 'no grasp candidate' if not C else 'no feasible grasp candidate' if not why['feasible'] else
                       'no collision-free pick IK' if not why['place'] else 'no place plan (tray reach / collisions)')
        s.log('no plan:',name,json.dumps(why))
        return False
def remember_failure(memory,name,ok,attempt):
    """Failure memory (grasp.demote_failed): store attempt = (kind, centre xy, closing axis xy) under name when the attempt was executed
    (attempt is not None) and failed (ok is False). Placed (True), not started for time (None) or never executed
    (planning failure, attempt None) leave the memory unchanged. Returns the memory."""
    if ok is False and attempt is not None:memory.setdefault(name,[]).append(attempt)
    return memory
def accel_status():
    """('on'|'off', reason) of the optional Rust extension (logged at start and at the end of the run)."""
    try:
        from . import _accel as A
        on=A.enabled() if hasattr(A,'enabled') else bool(getattr(A,'AVAILABLE',False))
        return ('on' if on else 'off'),str(getattr(A,'REASON',''))
    except Exception as e:
        return 'off',f'{type(e).__name__}: {e}'
def replan_infeasible(nofeas,name,handled):
    """False when the last planning for `name` found no pick+place plan and no object has been removed since
    (obstacles unchanged -> the deterministic planner would fail again and only burn sim time; v7: observe->start
    planning 12.0 s sim mean). Offline episode Monte Carlo (pool 3 sort3, 100 layouts): +0.4..0.9 pts
    when planning costs 2x sim time."""
    return nofeas.get(name) is None or nofeas[name]!=len(handled)
PLACE_AXES=(np.array([1.,0,0]),np.array([0,1.,0]),np.array([-1.,0,0]),np.array([0,-1.,0]))
def held_offset(A,q_grasp,g,c3):
    """Centre of the held object in the TCP frame for grasps whose grip point is not over the object centre (rim pinch:
    the pinch is on the rim, the dish centre r0 away), else None. Check on recorded run data: FK of the measured at_grasp
    joints (v8, v9b, v11b, 5 rim attempts), pinch point + r0 * closing axis lies 10.4-12.9 mm from the dish centre
    measured in the head frame (TCP_TOL 1 cm plus the circle fit). Height = the pinch height (plan_place uses only the
    horizontal part)."""
    oc=g.get('obj_center')
    if oc is None:return None
    T=A.tcp(q_grasp);p=np.array([oc[0],oc[1],c3[2]],float)
    return T[:3,:3].T@(p-T[:3,3])
def release_xy(A,q,obj_off,yaw):
    """Pelvis-frame xy of the held object's centre at arm joints q: obj_off (rim pinch) or the grip point that
    plan_place puts over the slot, rotated from the yawed torso frame back to the pelvis."""
    T=A.tcp(q);o=np.asarray(obj_off,float) if obj_off is not None else A._hand_frame('grip')[1]
    return (rotz(yaw)@(T[:3,:3]@o+T[:3,3]))[:2]
def in_tray(xy,tray):
    """True when the point xy (pelvis frame, m) lies inside the measured tray rectangle."""
    x,y,ax,ay,a,b,_=ARMMOD.tray_geom(tray);d=np.asarray(xy,float)[:2]-(x,y)
    return abs(d@(ax,ay))<=b/2 and abs(d@(-ay,ax))<=a/2
def plan_place(A,q_from,tx,ty,top,fallback=None,tray_h=TRAY_H,obj_off=None):
    """Release pose over (tx,ty) at the lowest feasible height -> (q_place, q_up, q_carry) or None.
    q_carry: straight-up lift from q_from to the release height (carry high over the tray walls), None if not needed
    or not feasible (then q_from->q_place must be certified directly). q_up: straight-up retreat after release.
    Closing axis tried with both signs (v6: tray_right reachable only with the mirrored axis); fallback=(x,y) (tray
    centre) is tried when the slot is out of reach. obj_off (held_offset): the pose is shifted horizontally so that
    the object centre, not the hand's grip point, is over (x,y); without it a rim-held dish hangs r0 (5-6.6 cm) off the
    slot, toward or over the tray wall."""
    z_from=A.tcp(q_from)[2,3]
    for (x,y) in [(tx,ty)]+([fallback] if fallback is not None else []):
        for dz in (.20,.25,.30):            # unchanged heights; lowering needs payload-vs-tray-wall data
            c=np.array([x,y,top+tray_h+dz-.1])
            for ca in PLACE_AXES:
                T=A.grasp_pose(c,ca,np.array([0,0,-1.]))
                if obj_off is not None:
                    sh=T[:3,:3]@obj_off+T[:3,3]-c;sh[2]=0.;T=A.grasp_pose(c-sh,ca,np.array([0,0,-1.]))
                q=A.solve(T,q_from)
                if q is None:continue
                if not A.col.certify(q,A.home):continue
                zr=A.tcp(q)[2,3];qc=A.lift(q_from,zr-z_from) if zr-z_from>.02 else None
                if qc is not None and A.col.certify(qc,q):return q,A.lift(q,RETREAT_DZ),qc
                if A.col.certify(q_from,q):return q,A.lift(q,RETREAT_DZ),None
    return None
def rotz(a):
    """3x3 rotation about z by a (rad)."""
    c,s_=np.cos(a),np.sin(a);return np.array([[c,-s_,0],[s_,c,0],[0,0,1.]])
YAW_MAX=2.5           # rad, |body_yaw| used for placing (INTERFACES.md: body_yaw_joint motor 31, [-2.97, 3.14], plus = left)
# Candidate bearings (rad, torso frame, toward this arm's side) at which the arm can place: a robot-reach property
# (arm workspace beside the shoulder; offline reach study: placements found at 0.5-1.3 rad, densest ~1.1),
# applied to the tray position measured by the camera. Not a scene coordinate (#645).
YAW_AIMS=(1.1,.9,1.3,.7,.5)
def yaw_candidates(side,tx,ty):
    """Torso yaw angles that bring the tray in front of this arm's shoulder (rotation about pelvis z, joint at origin).
    Offline (reach study, 30 layouts): with body_yaw locked no object reached both trays with one arm."""
    phi=np.arctan2(ty,tx);out=[]
    sg=1. if side=='l' else -1.
    for aim in YAW_AIMS:    # bearing of the measured tray in the rotated torso frame (beside this arm's shoulder)
        th=float(phi-sg*aim)
        if abs(th)<=YAW_MAX and abs(th)>.15 and all(abs(th-o)>.1 for o in out):out.append(th)
    return out
FAR_TRAY_Y=.3
def far_tray(side,ty):
    """Tray on the other arm's side (tray_left for the right arm, tray_right for the left arm).
    Offline evidence: reach study, 30 layouts x 2 pools, 0 of 450 picked objects were placeable in the
    far tray with body_yaw=0; study r3s3 (current planner, organiser layout generator) 103/103 far-tray placements used yaw."""
    return (side=='l' and ty<-FAR_TRAY_Y) or (side=='r' and ty>FAR_TRAY_Y)
def plan_place_yaw(A,q_from,tx,ty,plane,trays,fallback=None,tray_h=TRAY_H,obj_off=None):
    """(theta, (q_place,q_up,q_carry)) or None. theta=0 first; otherwise the torso is yawed by theta while the arm holds
    q_carry (or q_from); the scene (table plane, trays) is expressed in the rotated torso frame. The hold posture is
    checked against the rotated scene at 5 intermediate angles (sampled, not swept)."""
    n,d=np.asarray(plane[0],float),float(plane[1])
    zt=lambda x,y,n=n:-(d+n[0]*x+n[1]*y)/n[2]
    kw={} if obj_off is None else {'obj_off':obj_off}    # side/top grasps: no object offset
    if not far_tray(A.side,ty):    # the unyawed search for the far tray always fails: skip it (planning costs sim time)
        r=plan_place(A,q_from,tx,ty,zt(tx,ty),fallback,tray_h,**kw)
        if r is not None:return 0.,r
    try:
        for th in yaw_candidates(A.side,tx,ty):
            R=rotz(-th);n2=R@n
            A.set_scene((n2,d),ARMMOD.rotate_trays(trays,R))
            t2=(R@np.r_[tx,ty,0.])[:2];fb=None if fallback is None else tuple((R@np.r_[fallback[0],fallback[1],0.])[:2])
            r=plan_place(A,q_from,t2[0],t2[1],zt(t2[0],t2[1],n2),fb,tray_h,**kw)
            if r is None:continue
            hold=r[2] if r[2] is not None else q_from
            ok=True
            for a in np.linspace(0,th,5):
                Ra=rotz(-a);A.set_scene((Ra@n,d),ARMMOD.rotate_trays(trays,Ra))
                if not A.col.valid(hold):ok=False;break
            if ok:return th,r
        return None
    finally:
        A.set_scene((n,d),list(trays))
def move_time(a,b,vmax=.8,settle=None):
    """Pessimistic sim seconds of one move_arm from a to b (see MOVE_SETTLE_S)."""
    return float(np.max(np.abs(np.asarray(b)-np.asarray(a))))/vmax*1.6+(MOVE_SETTLE_S if settle is None else settle)
def heavy(name):
    """True when the object mass in priors.json (from objects.yaml) is at least HEAVY_KG."""
    return float(PRI.get(name,{}).get('mass',0.))>=HEAVY_KG
def place_time(A,q_lift,place,yaw,vmax=.8,release_s=None,q_from=None,settle=None):
    """Pessimistic seconds from 'fingers closed' to 'arm home, waist back': joint-space moves at <= vmax (max joint
    delta / vmax, x1.6 PD margin as move_arm) plus MOVE_SETTLE_S each, the lift itself (q_from = q_grasp -> q_lift,
    + 0.3 s pause; v7/v8 closed->lifted 3.3-3.6 s), waist out and back, finger opening (one stroke) and settling."""
    release_s=GRIP.finger_stroke_time()+.5 if release_s is None else release_s
    settle=MOVE_SETTLE_S if settle is None else settle
    q_place,q_up,q_carry=place;seq=([q_from] if q_from is not None else [])+[q_lift]+[q for q in (q_carry,q_place,q_up) if q is not None]+[A.home]
    mv=sum(move_time(a,b,vmax,settle) for a,b in zip(seq,seq[1:]))+(.3 if q_from is not None else 0.)
    return mv+(2*abs(yaw)/vmax+1. if yaw else 0.)+release_s
def move_bound(a,b,vmax=.8,refine=2):
    """Worst-case sim seconds of one move_arm(a->b) as coded: first attempt T = 1.6 x (|dq|max + MOVE_RES)/vmax +
    MOVE_T_PAD (the measured start may be off the planned one by up to MOVE_RES), then `refine` re-commands of REFINE_T."""
    return (float(np.max(np.abs(np.asarray(b)-np.asarray(a))))+MOVE_RES)/vmax*1.6+MOVE_T_PAD+refine*REFINE_T
def hold_time_bound(q_grasp,q_lift,place,yaw,vmax=.8):
    """#595: upper bound of the sim time from the lift command to the end of the release when every step times out:
    lift (+0.3 s pause), carry, waist (WAIST_TMAX), place, one TCP correction (A.correct max_step 0.15 rad, refine 1),
    then PLACE_RETRIES re-approaches of the release pose (move_arm(q_place), refine 2, from the corrected command, which
    lies within max_step of q_place, plus the residual) each with its own correction, and the release with one re-open
    (2 x RELEASE_WAIT; release_off_tray opens the hand within the same waits when it releases in place). Lifting only
    when this fits guarantees no object in the hand at the limit regardless of the typical-time model place_time; the
    way back to the pick spot (release_off_tray) is started only when the remaining time covers it as well."""
    q_place,q_up,q_carry=place
    seq=[q_grasp,q_lift]+([q_carry] if q_carry is not None else [])+[q_place]
    t=sum(move_bound(a,b,vmax) for a,b in zip(seq,seq[1:]))+.3
    t+=WAIST_TMAX if yaw else 0.
    corr=(.15+2*MOVE_RES)/vmax*1.6+MOVE_T_PAD+REFINE_T   # correction move_arm(qc, refine=1): |qc-q_measured| <= residual + max_step
    reapproach=(.15+2*MOVE_RES)/vmax*1.6+MOVE_T_PAD+2*REFINE_T   # move_arm(q_place), refine 2: |q_place-q_now| <= max_step + residual
    t+=corr+PLACE_RETRIES*(reapproach+corr)
    return t+2*RELEASE_WAIT
def lift_allowed(remaining,need,hold,safety):
    """#595 lift decision: the typical-time model (place_time + safety: lift, transport, release, retreat home) AND the
    worst-case hold bound (every in-hand step timing out) must both fit in the remaining sim time."""
    return remaining>=max(need+safety,hold+HOLD_MARGIN_S)
def self_w(s):
    """Widest graspable object width (m) for the open hand, capped at 0.11 m."""
    return min(s.arms['l'].hand['opposed_open']-.01,.11)
def main():
    """Entry point: start the node on a multi-threaded executor and run one task."""
    rclpy.init();n=Sol();ex=MultiThreadedExecutor(4);ex.add_node(n)
    def spin():
        # callbacks are guarded; this loop also keeps the executor alive if spin itself raises
        while rclpy.ok():
            try:ex.spin()
            except Exception as e:
                try:n.get_logger().error(f'executor: {type(e).__name__}: {e}')
                except Exception:pass
                time.sleep(.01)
    th=threading.Thread(target=spin,daemon=True);th.start()
    try:n.run()
    except Exception as e:
        # run() guards its task loop itself. This is the last resort for an error outside it (before the loop starts
        # or in the end-of-run log lines): the harness still gets /sim/task/done instead of waiting for the limit,
        # and the error is re-raised so the process exits non-zero with its traceback.
        try:sent=n.publish_done()
        except Exception:sent=False
        try:n.get_logger().error(f'run failed: {type(e).__name__}: {e}'+('; /sim/task/done published' if sent else ''))
        except Exception:pass
        raise
    finally:
        time.sleep(.5);rclpy.shutdown()
