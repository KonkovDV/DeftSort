"""TienKung 2.0 Pro arm IK + sphere self-collision built from the REAL organiser URDF (robot-only data).
Frames: pelvis, z up, x forward. Uses the v4 Chain (bounded DLS IK) and CollisionModel (swept sphere certification).
Limitations: the URDF STL meshes are absent, so link envelopes are spheres sized from the URDF inertial tensors
(equivalent-cylinder radius, inflated) -- an approximation, not mesh collision."""
import sys,os,numpy as np,xml.etree.ElementTree as ET

from .kinematics import Chain
from .planning import Sphere,CollisionModel,joint_path
from .geometry import pose,rotation
URDF=os.environ.get('HACK_URDF',os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','tiangong_pro_with_hands.urdf'))
import re
XML=re.sub(r'<axis\s+xyz="0 0 0"\s*/>','',open(URDF).read());ROOT=ET.fromstring(XML)
STANDING_PELVIS=.88;TABLE_TOP=.70-STANDING_PELVIS   # hackathon/config/scene.yaml (table.height, standing_pelvis_height)
SIDE={'l':dict(tip='left_tcp_link',hand='L_base_link',pre='left'),'r':dict(tip='right_tcp_link',hand='R_base_link',pre='right')}
ARM=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','elbow_yaw','wrist_pitch','wrist_roll']
def joints():
    out={}
    for j in ROOT.findall('joint'):
        o=j.find('origin');a=j.find('axis');l=j.find('limit')
        out[j.get('name')]=dict(type=j.get('type'),parent=j.find('parent').get('link'),child=j.find('child').get('link'),
            T=pose([float(x) for x in o.get('xyz').split()],[float(x) for x in o.get('rpy').split()]),
            axis=np.array([float(x) for x in a.get('xyz').split()]) if a is not None else None,
            lim=(float(l.get('lower')),float(l.get('upper')),float(l.get('velocity'))) if l is not None else None)
    return out
J=joints();BYCHILD={v['child']:k for k,v in J.items()}
def inertial(link):
    l=ROOT.find(f"link[@name='{link}']");i=l.find('inertial')
    m=float(i.find('mass').get('value'));c=np.array([float(x) for x in i.find('origin').get('xyz').split()])
    I=i.find('inertia').attrib;I=np.array([[float(I['ixx']),float(I['ixy']),float(I['ixz'])],[float(I['ixy']),float(I['iyy']),float(I['iyz'])],[float(I['ixz']),float(I['iyz']),float(I['izz'])]])
    ev=np.linalg.eigvalsh(I);return m,c,ev
def link_radius(link,inflate=1.25,lo=.025,hi=.09):
    m,c,ev=inertial(link);return float(np.clip(inflate*np.sqrt(2*ev[0]/max(m,1e-6)),lo,hi))   # solid-cylinder axial inertia
def fk_link(link,q=None):
    """pelvis->link with given joint values (default 0, clipped into limits)."""
    q=q or {};T=np.eye(4);path=[]
    while link!='pelvis':path.append(BYCHILD[link]);link=J[BYCHILD[link]]['parent']
    for n in reversed(path):
        j=J[n];T=T@j['T']
        if j['type']=='revolute':
            lo,hi,_=j['lim'];v=float(np.clip(q.get(n,0.),lo,hi));M=np.eye(4);M[:3,:3]=rotation(j['axis'],v);T=T@M
    return T
# ---------------- hand geometry (Inspire-type, from the URDF) ----------------
def fingertip(side,finger,q):
    """Tip estimate: distal link frame + 2x its inertial CoM offset (meshes absent)."""
    pre=SIDE[side]['pre'];distal=f'{pre}_{finger}_4' if finger=='thumb' else f'{pre}_{finger}_2'
    m,c,_=inertial(distal);T=fk_link(distal,q);hand=fk_link(SIDE[side]['hand'],q)
    return (np.linalg.inv(hand)@T@np.r_[2*c,1])[:3]   # in hand (X_base_link) frame
def hand_opening(side='l',n=9):
    """Max thumb-tip to index/middle-tip distance over thumb opposition angles, fingers open (all in hand frame)."""
    pre=SIDE[side]['pre'];best=(0,None)
    lo,hi,_=J[f'{pre}_thumb_1_joint']['lim']
    for a in np.linspace(lo,hi,n):
        q={f'{pre}_thumb_1_joint':a}
        t=fingertip(side,'thumb',q);f=(fingertip(side,'index',q)+fingertip(side,'middle',q))/2
        d=np.linalg.norm(t-f)
        if a==hi or d>best[0]:
            if a==hi:opp=(d,t,f)
            if d>best[0]:best=(d,a)
    d,t,f=opp
    return dict(max_open=float(best[0]),thumb1_at_max=float(best[1]),opposed_open=float(d),thumb_tip=t,finger_tip=f,
                close_axis_hand=(f-t)/np.linalg.norm(f-t),grip_center_hand=(t+f)/2)
# ---------------- arm chain + collision envelope ----------------
def arm_chain(side):
    tip=SIDE[side]['tip'];c=Chain.from_urdf(XML,'pelvis',tip,np.eye(4))
    return c.lock_joints({'body_yaw_joint':0.})
def arm_spheres(side,chain):
    S=[];s=side
    for n in ARM:
        child=J[f'{n}_{s}_joint']['child'];r=link_radius(child)
        # segment from this link frame to the next joint origin
        nxt=[k for k,v in J.items() if v['parent']==child and v['type']!='fixed' or (v['parent']==child and 'tcp' in k)]
        end=J[nxt[0]]['T'][:3,3] if nxt else np.zeros(3);L=np.linalg.norm(end);k=max(1,int(np.ceil(L/r)))
        for t in np.linspace(0,1,k+1):S.append(Sphere(child,end*t,r))
    tip=SIDE[s]['tip']   # hand + fingers: palm and finger region along hand z in tcp frame (hand frame == tcp up to rotation)
    R=J[f"{'left' if s=='l' else 'right'}_hand_to_tcp"]['T'][:3,:3]
    for z,r in ((.03,.045),(.08,.045),(.13,.04),(.17,.035)):S.append(Sphere(tip,R@np.array([0,0,z]),r))
    return S
def body_env(side,other_q=None):
    """Torso, head and the other arm (at its posture) as pelvis-frame spheres. Table added separately."""
    E=[];P=lambda link,p=np.zeros(3),q=None:(fk_link(link,q)@np.r_[p,1])[:3]
    w=fk_link('waist_yaw_link');shL=P('shoulder_pitch_l_link');shR=P('shoulder_pitch_r_link')
    # torso column: from pelvis up to shoulder height, width = shoulder separation minus arm clearance
    # trunk core kept inside the shoulder joints (|y| + r <= shoulder_y - 0.07); arm links mount on it
    r=min(.08,abs(shL[1])-.07)
    for z in np.arange(0.,shL[2]-.05,.06):
        for x in (-.03,.04):E.append(Sphere('pelvis',np.array([x,0.,z]),r))
    hp=P('head_pitch_link');E.append(Sphere('pelvis',hp+np.array([.02,0,.08]),.12))
    o='r' if side=='l' else 'l';q=other_q or {}
    for n in ARM:
        child=J[f'{n}_{o}_joint']['child'];E.append(Sphere('pelvis',P(child,q=q),link_radius(child)))
    E.append(Sphere('pelvis',P(SIDE[o]['tip'],q=q),.06))
    # legs/hips below the table are fixed: one sphere per hip
    for hs in ('l','r'):E.append(Sphere('pelvis',P(f'hip_pitch_{hs}_link'),.07))
    return E
def table_env(x=(.32,.85),y=(-.95,.95),R=.1,step=.03,top=TABLE_TOP):   # near edge x=.22 (table.front_gap)
    return [Sphere('pelvis',np.array([a,b,top-R]),R) for a in np.arange(x[0],x[1]+1e-9,step) for b in np.arange(y[0],y[1]+1e-9,step)]
class Arm:
    def __init__(self,side,margin=.015,table=True):
        self.side=side;self.chain=arm_chain(side);self.hand=hand_opening(side)
        env=body_env(side)+(table_env() if table else [])
        # ignore adjacent arm links and the shoulder link vs torso handled by env; within-chain adjacency:
        links=[J[f'{n}_{side}_joint']['child'] for n in ARM]+[SIDE[side]['tip']]
        ign=[(a,b) for i,a in enumerate(links) for b in links[i+1:i+3]]
        self.col=CollisionModel(self.chain,arm_spheres(side,self.chain),env,ign,audited_complete=True,margin=margin)
        self.seeds=[np.clip(np.zeros(len(self.chain.active)),self.chain.lower,self.chain.upper)]
        rng=np.random.default_rng(0)
        for _ in range(6):self.seeds.append(rng.uniform(self.chain.lower,self.chain.upper)*.5)
        e=[0.]*7;s=1 if side=='l' else -1   # a forward-reaching elbow-bent seed
        q=dict(zip([f'{n}_{side}_joint' for n in ARM],[-.6,.25*s,0.,-1.2,0.,0.,0.]))
        self.seeds.insert(0,np.clip([q.get(n,0.) for n in self.chain.names],self.chain.lower,self.chain.upper))
        self.home=self.seeds[0]
    def grasp_pose(self,center_xyz,close_axis,approach):
        """TCP pose so that the hand's thumb->finger closing axis maps to close_axis and fingers point along approach."""
        Rh=J[f"{'left' if self.side=='l' else 'right'}_hand_to_tcp"]['T'][:3,:3]   # hand frame in tcp frame
        c_t=Rh@self.hand['close_axis_hand'];z_t=Rh@np.array([0,0,1.]);g_t=Rh@self.hand['grip_center_hand']
        z_t=z_t-c_t*(z_t@c_t);z_t/=np.linalg.norm(z_t)
        u=close_axis/np.linalg.norm(close_axis);a=approach-u*(approach@u);a/=np.linalg.norm(a)
        A=np.c_[c_t,z_t,np.cross(c_t,z_t)];B=np.c_[u,a,np.cross(u,a)];R=B@A.T
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=center_xyz-R@g_t;return T
    def solve(self,T,seed=None):
        for s in ([seed] if seed is not None else [])+self.seeds:
            try:
                q=self.chain.ik(T,s,max_iter=150)
                if self.col.valid(q):return q
            except ValueError:pass
        return None
    def plan_pick(self,center,close_axis,approach,width,pre=.08,lift=.08,tilts=(0,.35,.6)):
        """Returns dict(q_pre,q_grasp,q_lift,traj) or None. Speed capped at 0.8 rad/s by joint_path/Chain.velocity."""
        if width>self.hand['opposed_open']-.01:return None
        import itertools
        for t,sg in itertools.product(tilts,(1.,-1.)):
            a=np.asarray(approach,float);u=sg*np.asarray(close_axis,float)   # a 2-finger-opposed grasp is symmetric in the closing axis
            if abs(a[2])<.9: a=a*np.cos(t)+np.array([0,0,-np.sin(t)])  # tilt fingers downward
            Tg=self.grasp_pose(center,u,a);Tp=Tg.copy();Tp[:3,3]-=pre*(a/np.linalg.norm(a));Tl=Tg.copy();Tl[2,3]+=lift
            qp=self.solve(Tp);
            if qp is None:continue
            qg=self.solve(Tg,qp)
            if qg is None:continue
            ql=self.solve(Tl,qg)
            if ql is None:continue
            try:
                # approach path: the hand spheres get close to the object on purpose; table/body are the obstacles
                tr=joint_path(self.chain,self.col,[self.home,qp,qg,ql],speed=.8)
            except ValueError:continue
            return dict(q_pre=qp,q_grasp=qg,q_lift=ql,traj=tr,tilt=t,duration=float(tr.times[-1]),
                        max_speed=float(np.max(np.abs(np.diff(tr.points,axis=0))/np.diff(tr.times)[:,None])))
        return None
