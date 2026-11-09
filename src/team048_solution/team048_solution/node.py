"""Team 048 solution node: sort2/sort3 (pool from /sim/task). Sensor-only perception (head RGB-D + TF),
object identity by assignment of segmented blobs to the task's object list using size priors, URDF IK with
sphere self-collision checks, 0.8 rad/s-capped synchronized joint moves, Inspire hand pinch/side grasps.
Per-layout time budget (startup included), sensor grasp check (wrist F/T + hand state; thresholds TODO until
calibrated), slots inside trays, trays confirmed by camera colour, straight-up retreats, TCP check before closing.
Tubes: not solved -> immediate /sim/task/done (0 points, allowed by SCORING).
Debug topics (/sim/task/check, /sim/objects/*) are never used.
Scene (#645): table plane + near edge and the trays are MEASURED by the cameras at run time (head survey with head
yaw scan, torso camera for the near edge); only approximate search bounds and robot geometry are constants."""
import json,os,time,threading,math
WALL_IMPORT=time.monotonic()
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
from .perception import table_plane,segment,plane_inliers,near_edge
from .grasp import candidates
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
PRI=json.load(open(os.path.join(D,'priors.json')))
TRAY_H=TR.DOC_H        # used only when a detected tray carries no measured wall height
IDS={'l':[11,12,13,14,15,16,17],'r':[21,22,23,24,25,26,27]}
SIDE_NAME={'l':'left','r':'right'}
APPROACH_VMAX=.3      # rad/s on the last pre-grasp -> grasp segment (do not knock the object); 0.8 elsewhere
TCP_TOL=.01           # m, measured TCP vs planned before closing; one correction if exceeded
RETREAT_DZ=.08        # m, straight-up lift after release / before going home from low poses
LOW_ABOVE_TABLE=.10    # m, TCP lower than this above the MEASURED table -> lift straight up before going home
HEAD_PITCH=.4         # rad, look-down posture for the table (head_pitch_joint)
HEAD_TOL=.01          # rad, head considered on target
HEAD_TMAX=6.          # s sim, give up waiting for the head (logged)
# Tray survey (#645): generic look-left / look-right head yaws. Camera FOV geometry (camera_info: 640x360, 86 deg HFOV;
# URDF head chain): at pitch 0.4 the views at yaw 0/+0.6/-0.6 jointly cover bearings -83..+83 deg of the table in front
# of the robot -- the whole half-plane, not a stored tray position. A tray seen cut by the image border is re-aimed at
# its own measured centroid (TRAY_RELOOK_MAX times); a tray that is never fully seen is NOT used (its category is
# skipped -> 0 points, never -10).
HEAD_SCAN_YAWS=(0.,.6,-.6)
TRAY_RELOOK_MAX=2
TRAY_RESURVEY_MAX=1   # extra survey later in the run when a needed tray is still missing
TRAY_MASK_MARGIN=.03  # m, pixels within the measured tray rectangle + margin are not objects
ID_CONF_MIN=.7        # identify(): category margin; render-set study 2026-11-09 (pool 3, 625 segments): 0.7 -> 599 accepted / 7 wrong vs 1.0 -> 498 / 3
ID_COST_MAX=2.5
TRACE_DIR=os.environ.get('TEAM048_TRACE_DIR')   # offline test harness only: per-object JSON trace + RGB-D frames
SHADOW_DF_N=.5*.08*9.81   # NOT used for decisions: logged 'would-be' verdict, 1/2 weight of the lightest pool-3 object (sim cfg 0.08 kg)
PRE_TOL=.05           # m, pre-grasp TCP error above which the pick is aborted (v7 s101 mustard: 26.6 cm, then knocked)
RELEASE_OPEN_MIN=.8    # fingers 1-4 mean after release; v7: 0.96-0.97 released, 0.99 home
PLACE_TOL=.04         # m, release TCP error: one correction before opening (v7 s101 Ecoforms released 13.7 cm off)
SIDE_DZ=(0.,.02,.04)  # m, raise a side grasp when the low one is not collision-free against the measured table
EMPTY_OBS_MAX=2       # consecutive observations with nothing to pick before giving up
def quat_to_T(t,q):
    x,y,z,w=q;T=np.eye(4)
    T[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    T[:3,3]=t;return T
class Sol(Node):
    def __init__(s):
        super().__init__('team048_solution')
        lat=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        s.task=None;s.js={};s.info=None;s.lock=threading.Lock();s.frames=PairBuffer()
        s.wrench={};s.hstate={};s.hvel={'l':[],'r':[]}
        s.create_subscription(String,'/sim/task',lambda m:setattr(s,'task',json.loads(m.data)),lat)
        s.create_subscription(JointState,'/joint_states',s.on_js,20)
        s.create_subscription(Image,'/cam_head/color/image_raw',s.frames.add_rgb,qos_profile_sensor_data)
        s.create_subscription(Image,'/cam_head/depth/image_raw',s.frames.add_depth,qos_profile_sensor_data)
        s.create_subscription(CameraInfo,'/cam_head/depth/camera_info',lambda m:setattr(s,'info',m),10)
        # torso camera: only for the table near edge (#645: measured, not stored); intrinsics from its camera_info (#613)
        s.frames_torso=PairBuffer();s.info_torso=None
        s.create_subscription(Image,'/cam_torso/color/image_raw',s.frames_torso.add_rgb,qos_profile_sensor_data)
        s.create_subscription(Image,'/cam_torso/depth/image_raw',s.frames_torso.add_depth,qos_profile_sensor_data)
        s.create_subscription(CameraInfo,'/cam_torso/depth/camera_info',lambda m:setattr(s,'info_torso',m),10)
        # #595.4: /robot_description is published by robot_state_publisher, QoS TRANSIENT_LOCAL
        s.robot_description=None
        s.create_subscription(String,'/robot_description',lambda m:setattr(s,'robot_description',m.data),lat)
        for k,n in SIDE_NAME.items():
            s.create_subscription(WrenchStamped,f'/arm_6dof_{n}',lambda m,k=k:s.wrench.__setitem__(k,GRIP.force_vec(m.wrench)),qos_profile_sensor_data)
            s.create_subscription(JointState,f'/inspire_hand/state/{n}_hand',lambda m,k=k:s.on_hand(k,m),10)
        s.pub={g:s.create_publisher(CmdSetMotorPosition,f'/{g}/cmd_pos',10) for g in ('arm','head','waist')}
        s.hand={k:s.create_publisher(JointState,f'/inspire_hand/ctrl/{k}_hand',10) for k in ('left','right')}
        s.cli_force={k:s.create_client(SetForce,f'/inspire_hand/set_force/{n}_hand') for k,n in SIDE_NAME.items()}
        s.cli_speed={k:s.create_client(SetSpeed,f'/inspire_hand/set_speed/{n}_hand') for k,n in SIDE_NAME.items()}
        s.done=s.create_publisher(Empty,'/sim/task/done',10)
        s.tfb=tf2_ros.Buffer();s.tfl=tf2_ros.TransformListener(s.tfb,s)
        s.wall_ready=time.monotonic()   # subscriptions exist from here: later waiting for /clock costs no sim time
        s.hand_target={'l':None,'r':None};s.t0=None;s.trays={};s.placed={};s.plane=None;s.table_x=None;s.monitor=GRIP.GraspMonitor()
    def on_js(s,m):
        with s.lock:s.js.update(zip(m.name,m.position))
    def on_hand(s,k,m):
        s.hstate[k]=m;h=s.hvel[k];h.append(list(m.velocity));del h[:-10]
    def now(s):return s.get_clock().now().nanoseconds*1e-9
    def sleep(s,dt):
        t=s.now()
        while s.now()-t<dt:time.sleep(.01)
    def log(s,*a):s.get_logger().info(' '.join(str(x) for x in a))
    # ---------------- actuation ----------------
    def cmd(s,group,ids,q,spd):
        m=CmdSetMotorPosition()
        for i,p,v in zip(ids,q,spd):
            c=SetMotorPosition();c.name=int(i);c.pos=float(p);c.spd=float(min(.8,max(v,1e-3)));c.cur=1.0;m.cmds.append(c)
        s.pub[group].publish(m)
    def q_now(s,side):
        names=[f'{n}_{side}_joint' for n in ARMMOD.ARM]
        with s.lock:return np.array([s.js.get(n,0.) for n in names])
    def move_arm(s,side,q,tol=.02,timeout=None,refine=2,vmax=.8):
        vmax=min(.8,vmax)
        q=np.asarray(q,float);q0=s.q_now(side);dq=np.abs(q-q0);m=max(dq.max(),1e-3)
        spd=np.maximum(vmax*dq/m,.05*vmax/.8)     # synchronized: every joint arrives together (straight joint-space line)
        T=timeout or (m/vmax*1.6+2.)
        s.cmd('arm',IDS[side],q,spd);t=s.now();goal=q.copy()
        for k in range(refine+1):
            while s.now()-t<T:
                e=q-s.q_now(side)
                if np.abs(e).max()<tol:return True
                time.sleep(.02)
            e=q-s.q_now(side)
            if k<refine:   # PD undershoot under gravity: bias the target by the residual (bounded by joint limits)
                goal=np.clip(goal+e,s.arms[side].chain.lower,s.arms[side].chain.upper);s.cmd('arm',IDS[side],goal,np.full(7,.3));t=s.now();T=1.5
        return np.abs(q-s.q_now(side)).max()<.06
    def hand_set(s,side,v):
        s.hand_target[side]=list(v);m=JointState();m.name=['1','2','3','4','5','6'];m.position=[float(x) for x in v];s.hand[SIDE_NAME[side]].publish(m)
    def hand_open(s,side):s.hand_set(side,[1,1,1,1,1,0.])      # fingers open, thumb rotated into opposition (DOF 6 = 0)
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
    def hand_close(s,side,kind):
        s.hand_set(side,s.closed_target(kind))
    def wait_thumb(s,side,tmax=None):
        """#648: the thumb is rotated into opposition FIRST (hand_open sets DOF 6), the other fingers close only
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
        settled=s.wait_fingers(side)
        h=s.hstate.get(side);pc=None if h is None else [float(x) for x in h.position]
        if pc is None:return dict(contact=None,target2=None,empty=None,settled=settled,closed=t0)
        empty=GRIP.contact_empty(pc,t0)
        t2=None
        if not empty and GRIP.CLOSE_PAST>0:
            t2=GRIP.contact_target(pc,t0);s.hand_set(side,t2);s.wait_fingers(side,tmin=.2)
        return dict(contact=pc,target2=t2,empty=empty,settled=settled,closed=t0)
    def wait_fingers(s,side,tmin=.3,tmax=None):
        """Until the fingers stop (blocked by the object or at target), from /inspire_hand/state velocities.
        Default timeout = one full stroke at the configured finger speed + 0.5 s (speed 0.5 -> 2.1 s)."""
        tmax=GRIP.finger_stroke_time()+.5 if tmax is None else tmax
        t=s.now();s.hvel[side].clear()
        while s.now()-t<tmax:
            if s.now()-t>=tmin and GRIP.fingers_settled(s.hvel[side]):return True
            time.sleep(.02)
        return False
    def hand_settings(s):
        """Explicit finger force/speed limits (bodyctrl_msgs/srv/SetForce, SetSpeed). Async, never blocks the run."""
        for k in 'lr':
            if s.cli_force[k].service_is_ready():
                r=SetForce.Request()
                for i,v in enumerate(GRIP.force_vector()):setattr(r,f'force{i}_ratio',float(v))
                s.cli_force[k].call_async(r)
            if s.cli_speed[k].service_is_ready():
                r=SetSpeed.Request()
                for i in range(6):setattr(r,f'speed{i}_ratio',float(GRIP.HAND_SPEED_RATIO))
                s.cli_speed[k].call_async(r)
    def hand_verdict(s,side,grasp=None):
        """Finger-based; wrist F/T only logged. After a two-phase close the 'past' criterion is used (fingers still
        blocked short of the phase-2 target); otherwise the v7-calibrated position+effort criterion with the effort
        threshold rescaled to the force ratio."""
        h=s.hstate.get(side)
        if h is None:return None
        if grasp and grasp.get('target2') is not None:
            return GRIP.past_verdict(list(h.position),grasp['target2'],grasp['closed'])
        return GRIP.finger_verdict(list(h.position),list(h.effort),(grasp or {}).get('closed') or s.hand_target.get(side))
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
    # ---------------- perception ----------------
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
            pair=buf.pair(-1e18);s.log('capture: no fresh pair, using newest')
        info=getattr(s,info_attr)
        if pair is None or info is None:raise RuntimeError('no synchronised RGB-D pair / camera_info')
        rgbm,dm=pair
        rgb=np.frombuffer(bytes(rgbm.data),np.uint8).reshape(rgbm.height,rgbm.width,3)
        dmm=np.frombuffer(bytes(dm.data),np.uint16).reshape(dm.height,dm.width).astype(float)
        K=np.array(info.k).reshape(3,3);T=s.tf('pelvis',dm.header.frame_id or default_frame,dm.header.stamp)
        v,u=np.mgrid[0:dm.height,0:dm.width];Z=dmm/1000.
        X=(u-K[0,2])*Z/K[0,0];Y=(v-K[1,2])*Z/K[1,1]
        P=(np.stack([X,Y,Z],-1).reshape(-1,3)@T[:3,:3].T+T[:3,3]).reshape(dm.height,dm.width,3)
        return P,rgb,dmm>0
    def measure_plane(s,P,valid):
        """Table plane in the search window (approximate bounds, #645) + height-band plausibility."""
        tb0=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
        plane=table_plane(P,tb0)
        if not ARMMOD.plane_plausible(plane):raise ValueError('plane outside the table height search band')
        return plane
    def tray_mask(s,P):
        """Pixels inside any detected tray rectangle (+margin): trays are not objects."""
        m=np.zeros(P.shape[:2],bool)
        for t in s.trays.values():
            x,y,ax,ay,a,b,_=ARMMOD.tray_geom(t);v=np.array([ax,ay]);u=np.array([v[1],-v[0]])
            dx=P[...,0]-x;dy=P[...,1]-y
            m|=(np.abs(dx*u[0]+dy*u[1])<a/2+TRAY_MASK_MARGIN)&(np.abs(dx*v[0]+dy*v[1])<b/2+TRAY_MASK_MARGIN)
        return m
    def observe(s,not_before=None):
        P,rgb,valid=s.capture(not_before=not_before)
        plane=s.measure_plane(P,valid);s.plane=plane
        s.locate_trays(P,rgb,valid,plane,refine_only=True)
        valid=valid&~s.tray_mask(P)
        tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
        roi=(np.array([.18,-.62,-1.]),np.array([.62,.62,1.]))
        dets=segment(P,rgb,tb,plane,roi,min_pixels=40)
        return P,rgb,plane,dets
    def tray_names(s):
        """Trays the task uses (categories -> tray) that have a known colour (TASKS.md / /sim/task['trays'])."""
        t=s.task or {};colours=t.get('trays',{})
        need=set((t.get('categories') or {}).values()) or set(colours)
        return {n:colours.get(n) for n in need if colours.get(n) in TR.CHANNEL}
    def locate_trays(s,P,rgb,valid,plane,refine_only=False):
        """Detect trays in this frame. Only FULL (non-partial) detections are stored. refine_only: update trays
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
        for i,(h,w,l) in enumerate(feats):
            for j,n in enumerate(N):
                p=PRI[n];C[i,j]=abs(h-p['height'])/.012+max(0,w-p['w_min']-.01)/.012+max(0,p['w_min']-w-.02)/.02+max(0,l-p['w_max']-.015)/.015
        r,c=linear_sum_assignment(C);out=[]
        for i,j in zip(r,c):
            if j>=len(N):continue
            cat=PRI[N[j]]['category'];other=[C[i,k] for k in range(len(N)) if PRI[N[k]]['category']!=cat]
            conf=(min(other) if other else 9)-C[i,j]
            out.append((dets[i],N[j],cat,float(conf),clouds[i],float(C[i,j])))
        return out
    # ---------------- task ----------------
    def run(s):
        while s.task is None or s.now()<=0:time.sleep(.05)
        # v7 s101: 'startup 206.9 s' -- the wait for the (not yet running) sim clock was counted as sim time.
        # Only process start-up before the node could receive /clock is charged (pessimistically, RTF<=1).
        t0=BUD.start_time(s.now(),getattr(s,'wall_ready',time.monotonic()),WALL_IMPORT)
        task=s.task;s.log('task',task.get('task'),task.get('objects_from'))
        if task.get('task') not in BUD.LIMITS:
            s.done.publish(Empty());return
        s.budget=BUD.TimeBudget(BUD.LIMITS[task['task']],t0)
        cats=task['categories'];objs=task['objects']
        s.use_robot_description()
        s.arms={k:ARMMOD.Arm(k) for k in 'lr'}
        s.log('startup',round(s.budget.elapsed(s.now()),1),'s of',s.budget.limit)
        s.hand_settings()
        while not s.js:time.sleep(.05)
        for k in 'lr':s.hand_open(k)
        for k in 'lr':s.go_home(k)     # from the start posture along a certified path (planning assumes home)
        t_head=s.survey()
        handled=set();fails={};empty=0;nofeas={};resurveys=0   # nofeas[name]=len(handled) when no pick+place plan existed
        while s.budget.can_start(s.now()):
            try:P,rgb,plane,dets=s.observe(t_head)
            except Exception as e:s.log('observe failed',e);break
            t_head=s.now()                   # later observations: frames newer than the end of the last motion
            if plane is not None:
                for A in s.arms.values():A.set_scene(plane,list(s.trays.values()),s.table_x)   # measured table + near edge + tray walls
            allid=s.identify(dets,P,plane,[n for n in objs if n not in handled])
            s.log('observe t',round(s.budget.elapsed(s.now()),1),'dets',len(dets),'assigned',
                  [(x[1][:18],round(x[3],2),round(x[5],2)) for x in allid])
            ids=[x for x in allid if x[3]>ID_CONF_MIN and x[5]<ID_COST_MAX]
            ids.sort(key=lambda x:-x[3])     # most confident first
            did=False;stop=False
            for det,name,cat,conf,cloud,cost in ids:
                if fails.get(name,0)>=2:continue
                if not replan_infeasible(nofeas,name,handled):continue   # same scene: the deterministic planner fails again
                cat=objs.get(name,cat)        # category from /sim/task is authoritative
                tray=cats.get(cat)
                if tray not in s.trays:continue
                others=[P[d.pixels[:,0],d.pixels[:,1]] for d in dets if d is not det]   # every segment is an obstacle
                t=s.now();s.last_infeasible=False;ok=s.pick_place(name,cat,cloud,plane,others,tray)
                if ok is False and s.last_infeasible:nofeas[name]=len(handled)
                if ok is None:stop=True;break           # budget: no time to finish another object
                if ok:handled.add(name);did=True;s.budget.record_cycle(s.now()-t);break
                fails[name]=fails.get(name,0)+1
            if stop:break
            if not did and resurveys<TRAY_RESURVEY_MAX and any(cats.get(objs.get(x[1],x[2])) not in s.trays for x in ids):
                resurveys+=1;s.log('needed tray missing: re-survey');t_head=s.survey();continue
            empty=0 if did else empty+1
            if empty>=EMPTY_OBS_MAX:break
        for k in 'lr':s.retreat(k)
        if abs(s.js.get('body_yaw_joint',0.))>.05:s.waist(0.)
        s.done.publish(Empty());s.log('done at',round(s.budget.elapsed(s.now()),1),'handled',sorted(handled))
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
    def waist(s,theta,tol=.02,tmax=6.):
        """body_yaw_joint (motor 31) to theta at <=0.8 rad/s; waits on /joint_states."""
        s.cmd('waist',[31],[theta],[.8]);t=s.now()
        while s.now()-t<tmax:
            with s.lock:q=s.js.get('body_yaw_joint')
            if q is not None and abs(q-theta)<tol:return True
            time.sleep(.02)
        s.log('waist not settled',s.js.get('body_yaw_joint'),'target',theta);return False
    def go_home(s,side):
        """#625: home only along a CERTIFIED (self-, body-, other-arm- and table-collision-free) joint path from the
        measured posture: straight, or straight up first. If none exists the arm is moved anyway (it cannot stay in
        a low pose) and the event is logged for the trajectory review."""
        A=s.arms[side];o='r' if side=='l' else 'l'
        if o in s.arms:A.set_other(s.q_now(o))
        path=A.path_home(s.q_now(side))
        if path is None:s.log('WARNING uncertified path home',side);path=[A.home]
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
    def pick_place(s,name,cat,cloud,plane,others,tray):
        """True placed / False failed (safe, nothing lifted left outside) / None not started for lack of time."""
        f,C=candidates(cloud,plane,cat,others,W_MAX=self_w(s),robot_xy=(0.,0.))
        tx,ty,ax,ay,ts,tl,th=ARMMOD.tray_geom(s.trays[tray])
        order='lr' if ty>.1 else ('rl' if ty<-.1 else ('lr' if f['center'][1]>=0 else 'rl'))
        k=s.placed.get(tray,0);dl=TR.slot_offset(k,.5*f['length'],size=(ts,tl),ty=ty)   # along the MEASURED long axis
        sx,sy=tx+dl*ax,ty+dl*ay
        for side in order:
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
                    p=A.plan_pick(c3,ca,ap,g['width'],point=pt)
                    if p is not None:break
                if p is None:continue
                yp=plan_place_yaw(A,p['q_lift'],sx,sy,plane,list(s.trays.values()),fallback=(tx,ty) if dl else None,tray_h=th)
                if yp is None:continue
                yaw,place=yp
                if not s.budget.can_start(s.now(),p['duration']+(2*abs(yaw)/.8+1. if yaw else 0.)):   # waist out and back at 0.8 rad/s
                    s.log('skip',name,'remaining',round(s.budget.remaining(s.now()),1),'s');return None
                s.log('pick',name,cat,'->',tray,'slot',k,'arm',side,g['kind'],'w',round(g['width'],3),'z',round(g['z'],3))
                Tg=A.tcp(p['q_grasp']);s.trace('start',side,name,kind=g['kind'],center=[round(float(x),3) for x in c3])
                s.hand_open(side);s.move_arm(side,p['q_pre']);Tp=A.tcp(p['q_pre']);s.trace('pre',side,name,Tp)
                thumb=s.wait_thumb(side)      # #648: thumb in opposition before anything closes (usually already there)
                if A.position_error(np.clip(s.q_now(side),A.chain.lower,A.chain.upper),Tp)>PRE_TOL:
                    s.log('pre-grasp not reached, abort',name);s.retreat(side);return False
                s.move_arm(side,p['q_grasp'],tol=.015,vmax=APPROACH_VMAX)
                qm=s.q_now(side);s.trace('at_grasp',side,name,Tg)
                if A.position_error(np.clip(qm,A.chain.lower,A.chain.upper),Tg)>TCP_TOL:
                    qc=A.correct(np.clip(qm,A.chain.lower,A.chain.upper),p['q_grasp'])
                    if qc is not None:s.move_arm(side,qc,tol=.01,vmax=APPROACH_VMAX,refine=0);s.trace('corrected',side,name,Tg)
                s.monitor.baseline(s.wrench.get(side))
                gr=s.grasp_close(side,g['kind'])
                s.trace('closed',side,name,Tg,settled=gr['settled'],thumb=thumb,contact=gr['contact'],target2=gr['target2'],
                        empty=gr['empty'],force=GRIP.HAND_FORCE_RATIO,speed=GRIP.HAND_SPEED_RATIO,past=GRIP.CLOSE_PAST)
                if gr['empty']:       # fingers ran to the closed target: nothing between them -> do not lift (0, not -10)
                    s.log('grasp check (contact): empty',name);s.hand_open(side);s.retreat(side);return False
                # #595 end-of-run policy: a lifted object outside a tray (or in the hand) at the limit is -10, an
                # untouched one 0 -> lift only if lift + transport + release + retreat still fit before the limit.
                need=place_time(A,p['q_lift'],place,yaw)
                if s.budget.remaining(s.now())<need+s.budget.safety:
                    s.log('deadline: not lifting',name,'need',round(need,1),'s left',round(s.budget.remaining(s.now()),1))
                    s.hand_open(side);s.wait_fingers(side,tmin=.3);s.retreat(side);return None
                s.move_arm(side,p['q_lift']);s.sleep(.3)
                held=s.hand_verdict(side,gr)
                F=s.wrench.get(side);f0=s.monitor.f0
                shadow=None if F is None or f0 is None else bool(np.linalg.norm(F-f0)>=SHADOW_DF_N)
                s.trace('lifted',side,name,A.tcp(p['q_lift']),verdict=held,shadow=shadow,
                        eff_thr=round(GRIP.effort_held_threshold(),3))
                if held is False:      # nothing in the hand: open, retreat, re-observe (object untouched => 0, not -10)
                    s.log('grasp check: empty',name);s.hand_open(side);s.retreat(side);return False
                q_place,q_up,q_carry=place
                if q_carry is not None:s.move_arm(side,q_carry,tol=.03);s.trace('carry',side,name,A.tcp(q_carry))
                if yaw:s.waist(yaw);s.trace('yawed',side,name,yaw=round(yaw,3))
                s.move_arm(side,q_place,tol=.03);Tq=A.tcp(q_place)
                qm=np.clip(s.q_now(side),A.chain.lower,A.chain.upper)
                if A.position_error(qm,Tq)>PLACE_TOL:
                    qc=A.correct(qm,q_place,max_step=.15)
                    if qc is not None:s.move_arm(side,qc,tol=.02,refine=1)
                s.trace('over_tray',side,name,Tq,tray=tray,tray_xy=[round(sx,3),round(sy,3)],yaw=round(yaw,3),
                        verdict=s.hand_verdict(side,gr))   # slip during transport shows up here (recalibration data)
                s.hand_open(side);s.wait_fingers(side,tmin=.4,tmax=1.0)
                if s.fingers_open(side) is False:s.log('fingers not open after release, re-open',name);s.hand_open(side);s.wait_fingers(side,tmin=.4,tmax=1.0)
                s.trace('released',side,name,A.tcp(q_place))
                if q_up is not None:s.move_arm(side,q_up,tol=.03)
                s.go_home(side)
                if yaw:s.waist(0.)
                s.trace('home',side,name)
                s.placed[tray]=k+1
                return True
        s.last_infeasible=True   # nothing executed: no candidate had a collision-free pick + place plan
        return False
def replan_infeasible(nofeas,name,handled):
    """False when the last planning for `name` found no pick+place plan and no object has been removed since
    (obstacles unchanged -> the deterministic planner would fail again and only burn sim time; v7: observe->start
    planning 12.0 s sim mean). Offline episode MC (fixes/budget/episode_mc.py, pool 3 sort3, 100 layouts): +0.4..0.9 pts
    when planning costs 2x sim time."""
    return nofeas.get(name) is None or nofeas[name]!=len(handled)
PLACE_AXES=(np.array([1.,0,0]),np.array([0,1.,0]),np.array([-1.,0,0]),np.array([0,-1.,0]))
def plan_place(A,q_from,tx,ty,top,fallback=None,tray_h=TRAY_H):
    """Release pose over (tx,ty) at the lowest feasible height -> (q_place, q_up, q_carry) or None.
    q_carry: straight-up lift from q_from to the release height (carry high over the tray walls), None if not needed
    or not feasible (then q_from->q_place must be certified directly). q_up: straight-up retreat after release.
    Closing axis tried with BOTH signs (v6: tray_right reachable only with the mirrored axis); fallback=(x,y) (tray
    centre) is tried when the slot is out of reach."""
    z_from=A.tcp(q_from)[2,3]
    for (x,y) in [(tx,ty)]+([fallback] if fallback is not None else []):
        for dz in (.20,.25,.30):            # unchanged heights; lowering needs payload-vs-tray-wall data (TODO)
            c=np.array([x,y,top+tray_h+dz-.1])
            for ca in PLACE_AXES:
                T=A.grasp_pose(c,ca,np.array([0,0,-1.]))
                q=A.solve(T,q_from)
                if q is None:continue
                if not A.col.certify(q,A.home):continue
                zr=A.tcp(q)[2,3];qc=A.lift(q_from,zr-z_from) if zr-z_from>.02 else None
                if qc is not None and A.col.certify(qc,q):return q,A.lift(q,RETREAT_DZ),qc
                if A.col.certify(q_from,q):return q,A.lift(q,RETREAT_DZ),None
    return None
def rotz(a):
    c,s_=np.cos(a),np.sin(a);return np.array([[c,-s_,0],[s_,c,0],[0,0,1.]])
YAW_MAX=2.5           # rad, |body_yaw| used for placing (INTERFACES.md: body_yaw_joint motor 31, [-2.97, 3.14], plus = left)
# Candidate bearings (rad, torso frame, toward this arm's side) at which the arm can place: a ROBOT-REACH property
# (arm workspace beside the shoulder; offline reach study 2026-11-09: placements found at 0.5-1.3 rad, densest ~1.1),
# applied to the tray position MEASURED by the camera. Not a scene coordinate (#645).
YAW_AIMS=(1.1,.9,1.3,.7,.5)
def yaw_candidates(side,tx,ty):
    """Torso yaw angles that bring the tray in front of this arm's shoulder (rotation about pelvis z, joint at origin).
    Offline (2026-11-09, reach-p3 30 layouts): with body_yaw locked no object reached both trays with one arm."""
    phi=np.arctan2(ty,tx);out=[]
    sg=1. if side=='l' else -1.
    for aim in YAW_AIMS:    # bearing of the MEASURED tray in the rotated torso frame (beside this arm's shoulder)
        th=float(phi-sg*aim)
        if abs(th)<=YAW_MAX and abs(th)>.15 and all(abs(th-o)>.1 for o in out):out.append(th)
    return out
FAR_TRAY_Y=.3
def far_tray(side,ty):
    """Tray on the OTHER arm's side (tray_left for the right arm, tray_right for the left arm).
    Offline evidence (2026-11-09): reach-p3/p6 Kaggle, 30 layouts x 2 pools, 0 of 450 picked objects were placeable in the
    far tray with body_yaw=0; study r3s3 (new planner, organiser layout generator) 103/103 far-tray placements used yaw."""
    return (side=='l' and ty<-FAR_TRAY_Y) or (side=='r' and ty>FAR_TRAY_Y)
def plan_place_yaw(A,q_from,tx,ty,plane,trays,fallback=None,tray_h=TRAY_H):
    """(theta, (q_place,q_up,q_carry)) or None. theta=0 first; otherwise the torso is yawed by theta while the arm holds
    q_carry (or q_from); the scene (table plane, trays) is expressed in the rotated torso frame. The hold posture is
    checked against the rotated scene at 5 intermediate angles (sampled, not swept)."""
    n,d=np.asarray(plane[0],float),float(plane[1])
    zt=lambda x,y,n=n:-(d+n[0]*x+n[1]*y)/n[2]
    if not far_tray(A.side,ty):    # the unyawed search for the far tray always fails: skip it (planning costs sim time)
        r=plan_place(A,q_from,tx,ty,zt(tx,ty),fallback,tray_h)
        if r is not None:return 0.,r
    try:
        for th in yaw_candidates(A.side,tx,ty):
            R=rotz(-th);n2=R@n
            A.set_scene((n2,d),ARMMOD.rotate_trays(trays,R))
            t2=(R@np.r_[tx,ty,0.])[:2];fb=None if fallback is None else tuple((R@np.r_[fallback[0],fallback[1],0.])[:2])
            r=plan_place(A,q_from,t2[0],t2[1],zt(t2[0],t2[1],n2),fb,tray_h)
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
def place_time(A,q_lift,place,yaw,vmax=.8,release_s=None):
    """Pessimistic seconds from 'fingers closed' to 'arm home, waist back': joint-space moves at <= vmax (max joint
    delta / vmax, x1.6 PD margin as move_arm), waist out and back, finger opening (one stroke) and settling."""
    release_s=GRIP.finger_stroke_time()+.5 if release_s is None else release_s
    q_place,q_up,q_carry=place;seq=[q_lift]+[q for q in (q_carry,q_place,q_up) if q is not None]+[A.home]
    mv=sum(float(np.max(np.abs(np.asarray(b)-np.asarray(a))))/vmax*1.6+.3 for a,b in zip(seq,seq[1:]))
    return mv+(2*abs(yaw)/vmax+1. if yaw else 0.)+release_s
def self_w(s):return min(s.arms['l'].hand['opposed_open']-.01,.11)
def main():
    rclpy.init();n=Sol();ex=MultiThreadedExecutor(4);ex.add_node(n)
    th=threading.Thread(target=ex.spin,daemon=True);th.start()
    try:n.run()
    finally:
        time.sleep(.5);rclpy.shutdown()
