"""Team 048 solution node: sort2/sort3 (pool from /sim/task). Sensor-only perception (head RGB-D + TF),
object identity by assignment of segmented blobs to the task's object list using size priors, URDF IK with
sphere self-collision checks, 0.8 rad/s-capped synchronized joint moves, Inspire hand pinch/side grasps.
Tubes: not solved -> immediate /sim/task/done (0 points, allowed by SCORING)."""
import json,os,time,threading,math
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String,Empty
from sensor_msgs.msg import JointState,Image,CameraInfo
from bodyctrl_msgs.msg import CmdSetMotorPosition,SetMotorPosition
import tf2_ros
from scipy.optimize import linear_sum_assignment
from . import arm as ARMMOD
from .perception import table_plane,segment
from .grasp import candidates
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
PRI=json.load(open(os.path.join(D,'priors.json')))
TRAYS={'tray_left':(.34,.69),'tray_right':(.34,-.69),'tray_center':(.53,0.)}   # nominal; refined by camera if visible
TRAY_HALF=(.11,.16);TRAY_H=.07
IDS={'l':[11,12,13,14,15,16,17],'r':[21,22,23,24,25,26,27]}
def quat_to_T(t,q):
    x,y,z,w=q;T=np.eye(4)
    T[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    T[:3,3]=t;return T
class Sol(Node):
    def __init__(s):
        super().__init__('team048_solution')
        lat=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        s.task=None;s.js={};s.rgb=None;s.depth=None;s.info=None;s.lock=threading.Lock()
        s.create_subscription(String,'/sim/task',lambda m:setattr(s,'task',json.loads(m.data)),lat)
        s.create_subscription(JointState,'/joint_states',s.on_js,20)
        s.create_subscription(Image,'/cam_head/color/image_raw',lambda m:setattr(s,'rgb',m),qos_profile_sensor_data)
        s.create_subscription(Image,'/cam_head/depth/image_raw',lambda m:setattr(s,'depth',m),qos_profile_sensor_data)
        s.create_subscription(CameraInfo,'/cam_head/depth/camera_info',lambda m:setattr(s,'info',m),10)
        s.pub={g:s.create_publisher(CmdSetMotorPosition,f'/{g}/cmd_pos',10) for g in ('arm','head','waist')}
        s.hand={k:s.create_publisher(JointState,f'/inspire_hand/ctrl/{k}_hand',10) for k in ('left','right')}
        s.done=s.create_publisher(Empty,'/sim/task/done',10)
        s.tfb=tf2_ros.Buffer();s.tfl=tf2_ros.TransformListener(s.tfb,s)
        s.t0=None
    def on_js(s,m):
        with s.lock:s.js.update(zip(m.name,m.position))
    def now(s):return s.get_clock().now().nanoseconds*1e-9
    def sleep(s,dt):
        t=s.now()
        while s.now()-t<dt:time.sleep(.01)
    def log(s,*a):s.get_logger().info(' '.join(str(x) for x in a))
    # ---------------- actuation ----------------
    def cmd(s,group,ids,q,spd):
        m=CmdSetMotorPosition()
        for i,p,v in zip(ids,q,spd):
            c=SetMotorPosition();c.name=int(i);c.pos=float(p);c.spd=float(v);c.cur=1.0;m.cmds.append(c)
        s.pub[group].publish(m)
    def q_now(s,side):
        names=[f'{n}_{side}_joint' for n in ARMMOD.ARM]
        with s.lock:return np.array([s.js.get(n,0.) for n in names])
    def move_arm(s,side,q,tol=.02,timeout=None,refine=2):
        q=np.asarray(q,float);q0=s.q_now(side);dq=np.abs(q-q0);m=max(dq.max(),1e-3)
        spd=np.maximum(.8*dq/m,.05)               # synchronized: every joint arrives together (straight joint-space line)
        T=timeout or (m/.8*1.6+2.)
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
        m=JointState();m.name=['1','2','3','4','5','6'];m.position=[float(x) for x in v];s.hand['left' if side=='l' else 'right'].publish(m)
    def hand_open(s,side):s.hand_set(side,[1,1,1,1,1,0.])      # fingers open, thumb rotated into opposition
    def hand_close(s,side,kind):
        s.hand_set(side,[0,0,0,0,0,0.] if kind=='side' else [1,1,.1,.1,.2,0.])
    # ---------------- perception ----------------
    def tf(s,target,source):
        tr=s.tfb.lookup_transform(target,source,rclpy.time.Time(),timeout=rclpy.duration.Duration(seconds=1.))
        t=tr.transform.translation;r=tr.transform.rotation
        return quat_to_T([t.x,t.y,t.z],[r.x,r.y,r.z,r.w])
    def observe(s):
        s.rgb=s.depth=None
        t=s.now()
        while (s.rgb is None or s.depth is None or s.info is None) and s.now()-t<5:time.sleep(.02)
        rgbm,dm,info=s.rgb,s.depth,s.info
        rgb=np.frombuffer(bytes(rgbm.data),np.uint8).reshape(rgbm.height,rgbm.width,3)
        dmm=np.frombuffer(bytes(dm.data),np.uint16).reshape(dm.height,dm.width).astype(float)
        K=np.array(info.k).reshape(3,3);T=s.tf('pelvis',dm.header.frame_id or 'camera_head_link')
        v,u=np.mgrid[0:dm.height,0:dm.width];Z=dmm/1000.
        X=(u-K[0,2])*Z/K[0,0];Y=(v-K[1,2])*Z/K[1,1]
        P=(np.stack([X,Y,Z],-1).reshape(-1,3)@T[:3,:3].T+T[:3,3]).reshape(dm.height,dm.width,3)
        valid=dmm>0
        for (tx,ty) in TRAYS.values():   # trays are not objects
            valid&=~((np.abs(P[...,0]-tx)<TRAY_HALF[0]+.02)&(np.abs(P[...,1]-ty)<TRAY_HALF[1]+.02))
        tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
        plane=table_plane(P,tb)
        roi=(np.array([.18,-.62,-1.]),np.array([.62,.62,1.]))
        dets=segment(P,rgb,tb,plane,roi,min_pixels=40)
        return P,rgb,plane,dets
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
        while s.task is None:time.sleep(.05)
        s.t0=s.now();task=s.task;s.log('task',task.get('task'),task.get('objects_from'))
        if task.get('task') not in ('sort2','sort3'):
            s.done.publish(Empty());return
        limit={'sort2':270,'sort3':405}[task['task']]
        cats=task['categories'];objs=task['objects']
        s.arms={k:ARMMOD.Arm(k) for k in 'lr'}
        s.cmd('head',[2],[.4],[.8])
        while not s.js:time.sleep(.05)
        for k in 'lr':s.hand_open(k)
        handled=set();fails={}
        while s.now()-s.t0<limit-25:
            try:P,rgb,plane,dets=s.observe()
            except Exception as e:s.log('observe failed',e);break
            ids=[x for x in s.identify(dets,P,plane,[n for n in objs if n not in handled]) if x[3]>1.0 and x[5]<2.5]
            if not ids:break
            # order: most confident & isolated first
            ids.sort(key=lambda x:-x[3])
            did=False
            for det,name,cat,conf,cloud,cost in ids:
                if fails.get(name,0)>=2:continue
                tray=cats.get(cat)
                if tray not in TRAYS:continue
                others=[x[4] for x in ids if x[1]!=name]
                ok=s.pick_place(name,cat,cloud,plane,others,tray)
                if ok:handled.add(name);did=True;break
                fails[name]=fails.get(name,0)+1
            if not did:break
        for k in 'lr':s.go_home(k)
        s.done.publish(Empty());s.log('done at',round(s.now()-s.t0,1),'handled',sorted(handled))
    def go_home(s,side):
        s.move_arm(side,s.arms[side].home,tol=.05)
    def pick_place(s,name,cat,cloud,plane,others,tray):
        f,C=candidates(cloud,plane,cat,others,W_MAX=self_w(s),robot_xy=(0.,0.))
        tx,ty=TRAYS[tray];order='lr' if ty>.1 else ('rl' if ty<-.1 else ('lr' if f['center'][1]>=0 else 'rl'))
        for side in order:
            A=s.arms[side]
            for g in C:
                if not g['feasible']:continue
                z=-plane[1]/plane[0][2] if False else None
                top=-(plane[1]+plane[0][:2]@g['center'])/plane[0][2]
                c3=np.array([g['center'][0],g['center'][1],top+g['z']])
                ap=np.r_[g['approach'],0.] if g['kind']=='side' else np.array([0,0,-1.])
                ca=np.r_[g['close_axis'],0.]
                p=A.plan_pick(c3,ca,ap,g['width'])
                if p is None:continue
                place=s.plan_place(A,p['q_lift'],tx,ty,top)
                if place is None:continue
                s.log('pick',name,cat,'->',tray,'arm',side,g['kind'],'w',round(g['width'],3))
                s.hand_open(side);s.move_arm(side,p['q_pre']);s.move_arm(side,p['q_grasp'],tol=.015)
                s.hand_close(side,g['kind']);s.sleep(1.0)
                s.move_arm(side,p['q_lift'])
                for q in place:s.move_arm(side,q,tol=.03)
                s.hand_open(side);s.sleep(.8)
                s.go_home(side)
                return True
        return False
    def plan_place(s,A,q_from,tx,ty,top):
        for dz in (.20,.25,.30):
            c=np.array([tx,ty,top+TRAY_H+dz-.1])
            for ca in (np.array([1.,0,0]),np.array([0,1.,0])):
                T=A.grasp_pose(c,ca,np.array([0,0,-1.]))
                q=A.solve(T,q_from)
                if q is None:continue
                if A.col.certify(q_from,q) and A.col.certify(q,A.home):return [q]
        return None
def self_w(s):return min(s.arms['l'].hand['opposed_open']-.01,.11)
def main():
    rclpy.init();n=Sol();ex=MultiThreadedExecutor(4);ex.add_node(n)
    th=threading.Thread(target=ex.spin,daemon=True);th.start()
    try:n.run()
    finally:
        time.sleep(.5);rclpy.shutdown()
