"""Team 048 solution node: sort2/sort3 (pool from /sim/task). Sensor-only perception (head RGB-D + TF),
object identity by assignment of segmented blobs to the task's object list using size priors, URDF IK with
sphere self-collision checks, 0.8 rad/s-capped synchronized joint moves, Inspire hand pinch/side grasps.
Per-layout time budget (startup included), sensor grasp check (wrist F/T + hand state; thresholds TODO until
calibrated), slots inside trays, trays confirmed by camera colour, straight-up retreats, TCP check before closing.
Tubes: not solved -> immediate /sim/task/done (0 points, allowed by SCORING).
Debug topics (/sim/task/check, /sim/objects/*) are never used."""
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
from .perception import table_plane,segment
from .grasp import candidates
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
PRI=json.load(open(os.path.join(D,'priors.json')))
TRAY_HALF=(TR.TRAY_SIZE[0]/2,TR.TRAY_SIZE[1]/2);TRAY_H=TR.TRAY_H
IDS={'l':[11,12,13,14,15,16,17],'r':[21,22,23,24,25,26,27]}
SIDE_NAME={'l':'left','r':'right'}
APPROACH_VMAX=.3      # rad/s on the last pre-grasp -> grasp segment (do not knock the object); 0.8 elsewhere
TCP_TOL=.01           # m, measured TCP vs planned before closing; one correction if exceeded
RETREAT_DZ=.08        # m, straight-up lift after release / before going home from low poses
LOW_TCP=ARMMOD.TABLE_TOP+.10
HEAD_PITCH=.4         # rad, look-down posture for the table (head_pitch_joint)
HEAD_TOL=.01          # rad, head considered on target
HEAD_TMAX=6.          # s sim, give up waiting for the head (logged)
ID_CONF_MIN=.7        # identify(): category margin; render-set study 2026-11-09 (pool 3, 625 segments): 0.7 -> 599 accepted / 7 wrong vs 1.0 -> 498 / 3
ID_COST_MAX=2.5
TRACE_DIR=os.environ.get('TEAM048_TRACE_DIR')   # offline test harness only: per-object JSON trace + RGB-D frames
SHADOW_DF_N=.5*.08*9.81   # NOT used for decisions: logged 'would-be' verdict, 1/2 weight of the lightest pool-3 object (sim cfg 0.08 kg)
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
        for k,n in SIDE_NAME.items():
            s.create_subscription(WrenchStamped,f'/arm_6dof_{n}',lambda m,k=k:s.wrench.__setitem__(k,GRIP.force_vec(m.wrench)),qos_profile_sensor_data)
            s.create_subscription(JointState,f'/inspire_hand/state/{n}_hand',lambda m,k=k:s.on_hand(k,m),10)
        s.pub={g:s.create_publisher(CmdSetMotorPosition,f'/{g}/cmd_pos',10) for g in ('arm','head','waist')}
        s.hand={k:s.create_publisher(JointState,f'/inspire_hand/ctrl/{k}_hand',10) for k in ('left','right')}
        s.cli_force={k:s.create_client(SetForce,f'/inspire_hand/set_force/{n}_hand') for k,n in SIDE_NAME.items()}
        s.cli_speed={k:s.create_client(SetSpeed,f'/inspire_hand/set_speed/{n}_hand') for k,n in SIDE_NAME.items()}
        s.done=s.create_publisher(Empty,'/sim/task/done',10)
        s.tfb=tf2_ros.Buffer();s.tfl=tf2_ros.TransformListener(s.tfb,s)
        s.t0=None;s.trays=dict(TR.NOMINAL);s.placed={};s.monitor=GRIP.GraspMonitor()
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
        m=JointState();m.name=['1','2','3','4','5','6'];m.position=[float(x) for x in v];s.hand[SIDE_NAME[side]].publish(m)
    def hand_open(s,side):s.hand_set(side,[1,1,1,1,1,0.])      # fingers open, thumb rotated into opposition
    def hand_close(s,side,kind):
        s.hand_set(side,[0,0,0,0,0,0.] if kind=='side' else [1,1,.1,.1,.2,0.])
    def wait_fingers(s,side,tmin=.3,tmax=1.5):
        """Until the fingers stop (blocked by the object or at target), from /inspire_hand/state velocities."""
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
                for i in range(6):setattr(r,f'force{i}_ratio',float(GRIP.HAND_FORCE_RATIO))
                s.cli_force[k].call_async(r)
            if s.cli_speed[k].service_is_ready():
                r=SetSpeed.Request()
                for i in range(6):setattr(r,f'speed{i}_ratio',float(GRIP.HAND_SPEED_RATIO))
                s.cli_speed[k].call_async(r)
    def hand_verdict(s,side):
        h=s.hstate.get(side)
        return s.monitor.verdict(s.wrench.get(side),None if h is None else list(h.position),None if h is None else list(h.effort))
    def look(s,pitch=HEAD_PITCH,tol=HEAD_TOL,tmax=HEAD_TMAX):
        """Command the head pitch and WAIT until /joint_states shows it there and still (v6: the first observation
        was taken while the head was still rising from 0 -> objects outside the view / TF mismatch -> nothing picked).
        Returns the sim time at which the head was settled (frames must be newer)."""
        s.cmd('head',[2],[pitch],[.8]);t=s.now();prev=None
        while s.now()-t<tmax:
            with s.lock:q=s.js.get('head_pitch_joint')
            if q is not None and abs(q-pitch)<tol and prev is not None and abs(q-prev)<.002:
                s.log('head settled',round(q,3),'after',round(s.now()-t,2),'s');return s.now()
            prev=q;s.sleep(.1)
        with s.lock:q=s.js.get('head_pitch_joint')
        s.log('head NOT settled',q,'target',pitch);return s.now()
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
                 tcp=[round(float(x),4) for x in tcp])
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
    def observe(s,not_before=None):
        t_req=s.now() if not_before is None else not_before;t=time.monotonic();pair=None
        while time.monotonic()-t<5 and (pair is None or s.info is None):
            pair=s.frames.pair(t_req);time.sleep(.02)
        if pair is None:
            pair=s.frames.pair(-1e18);s.log('observe: no fresh pair, using newest')
        if pair is None or s.info is None:raise RuntimeError('no synchronised RGB-D pair / camera_info')
        rgbm,dm=pair;info=s.info
        rgb=np.frombuffer(bytes(rgbm.data),np.uint8).reshape(rgbm.height,rgbm.width,3)
        dmm=np.frombuffer(bytes(dm.data),np.uint16).reshape(dm.height,dm.width).astype(float)
        K=np.array(info.k).reshape(3,3);T=s.tf('pelvis',dm.header.frame_id or 'camera_head_link',dm.header.stamp)
        v,u=np.mgrid[0:dm.height,0:dm.width];Z=dmm/1000.
        X=(u-K[0,2])*Z/K[0,0];Y=(v-K[1,2])*Z/K[1,1]
        P=(np.stack([X,Y,Z],-1).reshape(-1,3)@T[:3,:3].T+T[:3,3]).reshape(dm.height,dm.width,3)
        valid=dmm>0
        tb0=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
        plane=table_plane(P,tb0)
        s.locate_trays(P,rgb,valid,plane)
        for (tx,ty) in s.trays.values():   # trays are not objects
            valid=valid&~((np.abs(P[...,0]-tx)<TRAY_HALF[0]+.02)&(np.abs(P[...,1]-ty)<TRAY_HALF[1]+.02))
        tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
        roi=(np.array([.18,-.62,-1.]),np.array([.62,.62,1.]))
        dets=segment(P,rgb,tb,plane,roi,min_pixels=40)
        return P,rgb,plane,dets
    def locate_trays(s,P,rgb,valid,plane):
        colours=(s.task or {}).get('trays',{})
        for name in TR.NOMINAL:
            col=colours.get(name)
            if col not in TR.CHANNEL:continue
            c,src=TR.locate(P,rgb,valid,name,col,plane)
            if src=='camera':s.log('tray',name,col,'camera',c.round(3))
            s.trays[name]=(float(c[0]),float(c[1]))
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
        t0=BUD.start_time(s.now(),time.monotonic(),WALL_IMPORT)
        task=s.task;s.log('task',task.get('task'),task.get('objects_from'))
        if task.get('task') not in BUD.LIMITS:
            s.done.publish(Empty());return
        s.budget=BUD.TimeBudget(BUD.LIMITS[task['task']],t0)
        cats=task['categories'];objs=task['objects']
        s.arms={k:ARMMOD.Arm(k) for k in 'lr'}
        s.log('startup',round(s.budget.elapsed(s.now()),1),'s of',s.budget.limit)
        s.hand_settings()
        while not s.js:time.sleep(.05)
        for k in 'lr':s.hand_open(k)
        t_head=s.look()
        handled=set();fails={};empty=0
        while s.budget.can_start(s.now()):
            try:P,rgb,plane,dets=s.observe(t_head)
            except Exception as e:s.log('observe failed',e);break
            t_head=s.now()                   # later observations: frames newer than the end of the last motion
            allid=s.identify(dets,P,plane,[n for n in objs if n not in handled])
            s.log('observe t',round(s.budget.elapsed(s.now()),1),'dets',len(dets),'assigned',
                  [(x[1][:18],round(x[3],2),round(x[5],2)) for x in allid])
            ids=[x for x in allid if x[3]>ID_CONF_MIN and x[5]<ID_COST_MAX]
            ids.sort(key=lambda x:-x[3])     # most confident first
            did=False;stop=False
            for det,name,cat,conf,cloud,cost in ids:
                if fails.get(name,0)>=2:continue
                cat=objs.get(name,cat)        # category from /sim/task is authoritative
                tray=cats.get(cat)
                if tray not in s.trays:continue
                others=[P[d.pixels[:,0],d.pixels[:,1]] for d in dets if d is not det]   # every segment is an obstacle
                t=s.now();ok=s.pick_place(name,cat,cloud,plane,others,tray)
                if ok is None:stop=True;break           # budget: no time to finish another object
                if ok:handled.add(name);did=True;s.budget.record_cycle(s.now()-t);break
                fails[name]=fails.get(name,0)+1
            if stop:break
            empty=0 if did else empty+1
            if empty>=EMPTY_OBS_MAX:break
        for k in 'lr':s.retreat(k)
        s.done.publish(Empty());s.log('done at',round(s.budget.elapsed(s.now()),1),'handled',sorted(handled))
    def go_home(s,side):
        s.move_arm(side,s.arms[side].home,tol=.05)
    def retreat(s,side):
        """Lift straight up first if the TCP is low (near table/tray walls), then go home."""
        A=s.arms[side];q=s.q_now(side)
        try:low=A.tcp(np.clip(q,A.chain.lower,A.chain.upper))[2,3]<LOW_TCP
        except ValueError:low=False
        if low:
            qu=A.lift(np.clip(q,A.chain.lower,A.chain.upper),RETREAT_DZ)
            if qu is not None:s.move_arm(side,qu,tol=.03)
        s.go_home(side)
    def pick_place(s,name,cat,cloud,plane,others,tray):
        """True placed / False failed (safe, nothing lifted left outside) / None not started for lack of time."""
        f,C=candidates(cloud,plane,cat,others,W_MAX=self_w(s),robot_xy=(0.,0.))
        tx,ty=s.trays[tray];order='lr' if ty>.1 else ('rl' if ty<-.1 else ('lr' if f['center'][1]>=0 else 'rl'))
        k=s.placed.get(tray,0);dy=TR.slot_offset(k,.5*f['length'])
        for side in order:
            A=s.arms[side]
            for g in C:
                if not g['feasible']:continue
                top=-(plane[1]+plane[0][:2]@g['center'])/plane[0][2]
                c3=np.array([g['center'][0],g['center'][1],top+g['z']])
                ap=np.r_[g['approach'],0.] if g['kind']=='side' else np.array([0,0,-1.])
                ca=np.r_[g['close_axis'],0.]
                p=A.plan_pick(c3,ca,ap,g['width'])
                if p is None:continue
                place=plan_place(A,p['q_lift'],tx,ty+dy,top,fallback=(tx,ty) if dy else None)
                if place is None:continue
                if not s.budget.can_start(s.now(),p['duration']):
                    s.log('skip',name,'remaining',round(s.budget.remaining(s.now()),1),'s');return None
                s.log('pick',name,cat,'->',tray,'slot',k,'arm',side,g['kind'],'w',round(g['width'],3),'z',round(g['z'],3))
                Tg=A.tcp(p['q_grasp']);s.trace('start',side,name,kind=g['kind'],center=[round(float(x),3) for x in c3])
                s.hand_open(side);s.move_arm(side,p['q_pre']);s.trace('pre',side,name,A.tcp(p['q_pre']))
                s.move_arm(side,p['q_grasp'],tol=.015,vmax=APPROACH_VMAX)
                qm=s.q_now(side);s.trace('at_grasp',side,name,Tg)
                if A.position_error(np.clip(qm,A.chain.lower,A.chain.upper),Tg)>TCP_TOL:
                    qc=A.correct(np.clip(qm,A.chain.lower,A.chain.upper),p['q_grasp'])
                    if qc is not None:s.move_arm(side,qc,tol=.01,vmax=APPROACH_VMAX,refine=0);s.trace('corrected',side,name,Tg)
                s.monitor.baseline(s.wrench.get(side))
                s.hand_close(side,g['kind']);settled=s.wait_fingers(side);s.trace('closed',side,name,Tg,settled=settled)
                s.move_arm(side,p['q_lift']);s.sleep(.3)
                held=s.hand_verdict(side)
                F=s.wrench.get(side);f0=s.monitor.f0
                shadow=None if F is None or f0 is None else bool(np.linalg.norm(F-f0)>=SHADOW_DF_N)
                s.trace('lifted',side,name,A.tcp(p['q_lift']),verdict=held,shadow=shadow)
                if held is False:      # nothing in the hand: open, retreat, re-observe (object untouched => 0, not -10)
                    s.log('grasp check: empty',name);s.hand_open(side);s.retreat(side);return False
                q_place,q_up=place
                s.move_arm(side,q_place,tol=.03);s.trace('over_tray',side,name,A.tcp(q_place),tray=tray,tray_xy=[round(tx,3),round(ty+dy,3)])
                s.hand_open(side);s.wait_fingers(side,tmin=.4,tmax=1.0);s.trace('released',side,name,A.tcp(q_place))
                if q_up is not None:s.move_arm(side,q_up,tol=.03)
                s.go_home(side);s.trace('home',side,name)
                s.placed[tray]=k+1
                return True
        return False
PLACE_AXES=(np.array([1.,0,0]),np.array([0,1.,0]),np.array([-1.,0,0]),np.array([0,-1.,0]))
def plan_place(A,q_from,tx,ty,top,fallback=None):
    """Release pose over (tx,ty) at the lowest feasible height, plus a straight-up retreat pose (or None).
    Closing axis tried with BOTH signs: the Inspire hand is not symmetric (thumb vs fingers), the right arm reaches
    tray_right only with the mirrored axis (offline 2026-11-09: +x/+y -> IK fails for every height, -y -> ok).
    v6 root cause: no box was ever picked because tray_right had no release pose. fallback=(x,y) (tray centre)
    is tried when the slot position is out of reach."""
    for (x,y) in [(tx,ty)]+([fallback] if fallback is not None else []):
        for dz in (.20,.25,.30):            # unchanged heights; lowering needs payload-vs-tray-wall data (TODO)
            c=np.array([x,y,top+TRAY_H+dz-.1])
            for ca in PLACE_AXES:
                T=A.grasp_pose(c,ca,np.array([0,0,-1.]))
                q=A.solve(T,q_from)
                if q is None:continue
                if A.col.certify(q_from,q) and A.col.certify(q,A.home):
                    return q,A.lift(q,RETREAT_DZ)
    return None
def self_w(s):return min(s.arms['l'].hand['opposed_open']-.01,.11)
def main():
    rclpy.init();n=Sol();ex=MultiThreadedExecutor(4);ex.add_node(n)
    th=threading.Thread(target=ex.spin,daemon=True);th.start()
    try:n.run()
    finally:
        time.sleep(.5);rclpy.shutdown()
