"""TienKung 2.0 Pro arm IK + sphere self-collision built from the REAL organiser URDF (robot-only data).
Frames: pelvis, z up, x forward. Uses the v4 Chain (bounded DLS IK) and CollisionModel (swept sphere certification).
Limitations: the URDF STL meshes are absent, so link envelopes are spheres sized from the URDF inertial tensors
(equivalent-cylinder radius, inflated) -- an approximation, not mesh collision."""
import sys,os,numpy as np,xml.etree.ElementTree as ET

from .kinematics import Chain
from .planning import Sphere,CollisionModel,joint_path,spheres_bulk
from . import _accel
from .geometry import pose,rotation,check_transform,check_rotation,finite
URDF=os.environ.get('HACK_URDF',os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','tiangong_pro_with_hands.urdf'))
import re
XML=re.sub(r'<axis\s+xyz="0 0 0"\s*/>','',open(URDF).read());ROOT=ET.fromstring(XML)
# #645 (2026-11-05): table, trays and racks are found by the cameras; no pre-measured scene coordinates here.
# Only APPROXIMATE SEARCH BOUNDS (allowed): the table plane must lie in this pelvis-z band (a wide "the table is roughly
# at this height" window, 40 cm), and the table obstacle extends from the MEASURED near edge to these far bounds
# (conservative: an obstacle where there may be no table only removes plans, never adds collisions).
TABLE_Z_BAND=(-.45,-.05)    # m, pelvis frame: accepted table-top height at x=0.4 (search window, not a measurement)
TABLE_X_FAR=1.0             # m, far bound of the table obstacle (search bound)
TABLE_Y_HALF=1.0            # m, half width of the table obstacle (search bound)
TABLE_X_NEAR_MIN=.12        # m, the table obstacle never starts closer than this (robot body region, robot geometry)
SIDE={'l':dict(tip='left_tcp_link',hand='L_base_link',pre='left'),'r':dict(tip='right_tcp_link',hand='R_base_link',pre='right')}
ARM=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','elbow_yaw','wrist_pitch','wrist_roll']
REQUIRED_LINKS=('pelvis','left_tcp_link','right_tcp_link','L_base_link','R_base_link','head_yaw_link','camera_head_link',
                'waist_yaw_link','shoulder_pitch_l_link','shoulder_pitch_r_link','hip_pitch_l_link','hip_pitch_r_link')
def load_urdf(xml_text):
    """Switch the kinematic model to `xml_text` (e.g. /robot_description) if it has every required link and the arm
    joints; returns (ok, reason). Must be called before Arm() objects are built."""
    global XML,ROOT,J,BYCHILD
    x=re.sub(r'<axis\s+xyz="0 0 0"\s*/>','',xml_text);r=ET.fromstring(x)
    links={l.get('name') for l in r.findall('link')}
    miss=[l for l in REQUIRED_LINKS if l not in links]
    if miss:return False,f'missing links {miss[:3]}'
    old=(XML,ROOT,J,BYCHILD)
    XML,ROOT=x,r
    try:
        J=joints();BYCHILD={v['child']:k for k,v in J.items()}
        need=[f'{n}_{s}_joint' for n in ARM for s in 'lr']+['left_hand_to_tcp','right_hand_to_tcp']
        if any(n not in J for n in need):raise KeyError('arm joints')
        for l in ('L_base_link','R_base_link'):inertial(l)
        hand_opening('l');hand_opening('r')     # finger links present (fingertip model)
    except Exception as e:
        XML,ROOT,J,BYCHILD=old;_ENV_CACHE.clear();return False,f'incomplete ({type(e).__name__})'
    _ENV_CACHE.clear();return True,'ok'
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
NULL_WEIGHT=.1   # IK null-space pull toward home; 0 disables (tested: accuracy unchanged, see tests/test_ideas.py)
# I3: mesh-fitted collision spheres (tools/fit_spheres.py on the UBTECH tiangong2pro_urdf STL meshes, Apache-2.0).
# TODO(calibrate): keep USE_MESH_SPHERES=False until the mesh frames are verified against the simulator /tf
# (FK vs TF for all arm links) and the JSON carries "validated": true. Inertia-derived spheres are used meanwhile.
USE_MESH_SPHERES=False
MESH_SPHERES=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','spheres_mesh.json')
def mesh_spheres(path=MESH_SPHERES,require_validated=True):
    """[{link,center,radius}] from JSON, or None when missing/unvalidated/malformed (never partially applied)."""
    import json
    try:d=json.load(open(path))
    except (OSError,ValueError):return None
    if require_validated and d.get('validated') is not True:return None
    try:return [Sphere(e['link'],np.asarray(e['center'],float),float(e['radius'])) for e in d['spheres']]
    except (KeyError,TypeError,ValueError):return None
HAND_MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','hand_mesh_geometry.json')
_HAND_MESH_CACHE={}
def hand_mesh(side):
    """Mesh-derived hand geometry (tools/hand_from_meshes.py) or None. Parsed once per file version (path, mtime,
    size): grasp_pose calls this for every candidate pose (was 248 JSON parses per 10 objects). Treat as read-only."""
    try:
        st=os.stat(HAND_MESH);key=(HAND_MESH,st.st_mtime_ns,st.st_size)
        d=_HAND_MESH_CACHE.get(key)
        if d is None:
            import json
            with open(HAND_MESH) as f:d=json.load(f)
            _HAND_MESH_CACHE.clear();_HAND_MESH_CACHE[key]=d
        return d[side]
    except Exception:return None
def arm_spheres(side,chain,mesh=True):
    if mesh and USE_MESH_SPHERES and chain is not None:
        M=mesh_spheres()
        links=set(chain.frames(np.clip(np.zeros(len(chain.active)),chain.lower,chain.upper)))
        if M:
            M=[x for x in M if x.link in links]
            if M:return M
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
    # thumb (off the hand axis by up to 9 cm, NOT covered by the axis spheres): v7 Kaggle side grasps pressed the
    # table with +11..+32 N (pelvis z) while the thumb tip was the lowest hand point (FK) -> mesh-derived thumb spheres
    hm=hand_mesh(s)
    if hm:
        for c in hm['thumb_env_hand']:S.append(Sphere(tip,R@np.asarray(c,float),max(.012,hm['thumb_env_radius'])))
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
    # #625: the OTHER arm at its (measured) posture with the same segment + hand envelope as the moving arm
    # (arm_spheres), mapped to pelvis by FK -- not one sphere per joint origin.
    for sp in arm_spheres(o,None,mesh=False):
        E.append(Sphere('pelvis',(fk_link(sp.link,q)@np.r_[sp.center,1])[:3],sp.radius))
    # legs/hips below the table are fixed: one sphere per hip
    for hs in ('l','r'):E.append(Sphere('pelvis',P(f'hip_pitch_{hs}_link'),.07))
    return E
_ENV_CACHE={}
def _cached_entry(key,build):
    """(tuple of Spheres, centers (n,3) read-only, radii (n,) read-only); scene sphere lists are pure functions of
    their arguments and Spheres are immutable: memoise (bounded)."""
    v=_ENV_CACHE.get(key)
    if v is None:
        if len(_ENV_CACHE)>256:_ENV_CACHE.clear()
        t=tuple(build());C=np.array([s.center for s in t],dtype=float).reshape(-1,3);R=np.array([s.radius for s in t],dtype=float)
        C.setflags(write=False);R.setflags(write=False);v=_ENV_CACHE[key]=(t,C,R)
    return v
def _cached(key,build):
    return list(_cached_entry(key,build)[0])
def _key(*a):
    return tuple(np.asarray(x,float).tobytes() if isinstance(x,(np.ndarray,list,tuple)) else x for x in a)
def plane_height(plane,x,y):
    """Table-top z (pelvis) at (x,y) for plane=(n,d), n.p+d=0."""
    n,d=np.asarray(plane[0],float),float(plane[1]);return float(-(d+n[0]*x+n[1]*y)/n[2])
def plane_plausible(plane,x=.4):
    """True if the measured plane is horizontal enough and inside the table height search band (#645 bounds)."""
    n=np.asarray(plane[0],float);n=n/np.linalg.norm(n)
    return bool(n[2]>.96 and TABLE_Z_BAND[0]<=plane_height(plane,x,0.)<=TABLE_Z_BAND[1])
def _table_args(plane,x,y):
    if plane is None:raise ValueError('table obstacle needs a measured plane (#645: no stored table height)')
    x=(TABLE_X_NEAR_MIN,TABLE_X_FAR) if x is None else (max(TABLE_X_NEAR_MIN,float(x[0])),float(x[1]))
    y=(-TABLE_Y_HALF,TABLE_Y_HALF) if y is None else y
    return x,y
def _table_entry(plane,x=None,y=None,R=.1,step=.03):
    x,y=_table_args(plane,x,y);pk=(np.asarray(plane[0],float).tobytes(),float(plane[1]))
    return _cached_entry(('table',_key(x,y,R,step),pk),lambda:table_env_build(plane,x,y,R,step))
def table_env(plane,x=None,y=None,R=.1,step=.03):
    """plane is REQUIRED (measured by the head camera); x=(near,far): near = measured near edge."""
    return list(_table_entry(plane,x,y,R,step)[0])
def table_env_build(plane,x,y,R=.1,step=.03):
    """Table as spheres tangent to the MEASURED top surface plane=(n,d) (pelvis frame, n.p+d=0, head camera).
    v7 Kaggle frames show the table tilted 4.0 deg in the pelvis frame (n_x=-0.070 in all 31 frames)."""
    n,d=np.asarray(plane[0],float),float(plane[1]);nn=np.linalg.norm(n);n=n/nn;d=d/nn
    # vectorised, element-wise identical to the scalar loop  z=-(d+n0*a+n1*b)/n2; centre=[a,b,z]-R*n  (a outer, b inner)
    A,B=np.meshgrid(np.arange(x[0],x[1]+1e-9,step),np.arange(y[0],y[1]+1e-9,step),indexing='ij');A=A.ravel();B=B.ravel()
    Z=-(d+n[0]*A+n[1]*B)/n[2]
    return spheres_bulk('pelvis',np.stack([A,B,Z],axis=1).reshape(-1,3)-R*n,R)
def tray_geom(t,size=(.22,.32),height=.07):
    """Normalise a tray to (x, y, ax, ay, short, long, height): a 2-tuple (x,y) (synthetic tests) or the dict of
    trays.detect() / a 7-tuple with the MEASURED long axis, size and wall height."""
    if isinstance(t,dict):
        return (float(t['center'][0]),float(t['center'][1]),float(t['axis'][0]),float(t['axis'][1]),
                float(t['size'][0]),float(t['size'][1]),float(t.get('height',height)))
    t=tuple(float(v) for v in t)
    return t if len(t)==7 else (t[0],t[1],0.,1.,float(size[0]),float(size[1]),float(height))
def rotate_trays(trays,R):
    """Trays expressed in a frame rotated by R (3x3 about z): centre and long axis rotate."""
    out=[]
    for t in trays:
        x,y,ax,ay,a,b,h=tray_geom(t);c=R@np.r_[x,y,0.];u=R@np.r_[ax,ay,0.];out.append((c[0],c[1],u[0],u[1],a,b,h))
    return out
def _tray_entry(trays,plane,size=(.22,.32),height=.07,r=.02,step=.03):
    G=tuple(tray_geom(t,size,height) for t in trays)
    key=('tray',np.asarray(G,float).tobytes(),np.asarray(plane[0],float).tobytes(),float(plane[1]),_key(r,step))
    return _cached_entry(key,lambda:tray_env_build(G,plane,size,height,r,step))
def tray_env(trays,plane,size=(.22,.32),height=.07,r=.02,step=.03):
    return list(_tray_entry(trays,plane,size,height,r,step)[0])
def tray_env_build(trays,plane,size=(.22,.32),height=.07,r=.02,step=.03):
    """Tray walls as spheres (centres r below the rim) from the MEASURED centre, long axis, size and wall height:
    v7 Kaggle s103 the hand was carried into the tray_right wall (82 N, TCP 14 cm below plan) when trays were not
    obstacles."""
    n,d=np.asarray(plane[0],float),float(plane[1]);out=[]
    for t in trays:
        tx,ty,ax,ay,sa,sb,height=tray_geom(t,size,height)
        v=np.array([ax,ay])/max(1e-9,np.hypot(ax,ay));u=np.array([v[1],-v[0]])   # u: short axis, v: long axis
        hx,hy=sa/2,sb/2;loc=[]
        for a in np.arange(-hx,hx+1e-9,step):loc+=[(a,-hy),(a,hy)]
        for b in np.arange(-hy,hy+1e-9,step):loc+=[(-hx,b),(hx,b)]
        pts=[(tx+a*u[0]+b*v[0],ty+a*u[1]+b*v[1]) for a,b in loc]
        for x,y in pts:
            z0=-(d+n[0]*x+n[1]*y)/n[2]
            for h in (height-r,max(r,height-3*r)):out.append((x,y,z0+h))
    return spheres_bulk('pelvis',out,r) if out else []
HOME_Q=(-.6,.25,0.,-1.2,0.,0.,0.)   # forward-reaching elbow-bent home posture (robot posture; y-sign mirrored per side)
def home_dict(side):
    s=1 if side=='l' else -1;v=list(HOME_Q);v[1]*=s
    return dict(zip([f'{n}_{side}_joint' for n in ARM],v))
def other_home(side):
    """The other arm at ITS home posture (the default while this arm plans; updated from /joint_states by set_other)."""
    return home_dict('r' if side=='l' else 'l')
class Arm:
    def __init__(self,side,margin=.015,table=True):
        self.side=side;self.chain=arm_chain(side);self.hand=hand_opening(side)
        # No table until the head camera measured it (#645): planning refuses to run before set_scene().
        self.body=body_env(side,other_home(side));env=list(self.body);self.scene_ready=False
        # ignore adjacent arm links and the shoulder link vs torso handled by env; within-chain adjacency:
        links=[J[f'{n}_{side}_joint']['child'] for n in ARM]+[SIDE[side]['tip']]
        ign=[(a,b) for i,a in enumerate(links) for b in links[i+1:i+3]]
        self.col=CollisionModel(self.chain,arm_spheres(side,self.chain),env,ign,audited_complete=True,margin=margin)
        self.seeds=[np.clip(np.zeros(len(self.chain.active)),self.chain.lower,self.chain.upper)]
        rng=np.random.default_rng(0)
        for _ in range(6):self.seeds.append(rng.uniform(self.chain.lower,self.chain.upper)*.5)
        q=home_dict(side)   # a forward-reaching elbow-bent seed
        self.seeds.insert(0,np.clip([q.get(n,0.) for n in self.chain.names],self.chain.lower,self.chain.upper))
        self.home=self.seeds[0]
    def set_scene(self,plane,trays=(),table_x=None):
        """Obstacles from the latest observation: measured table plane (+ measured near edge table_x=(near,far)) and
        the detected tray walls (body unchanged).
        Equivalent to  col.environment = body + table_env(plane, table_x) + tray_env(trays, plane)  (same spheres, same
        order); the assembled arrays and the Rust grid are memoised per scene (the torso-yaw search revisits the same
        rotated scenes for every object of an observation)."""
        if table_x is not None:self.table_x=table_x
        self._scene=(plane,list(trays))
        tx=getattr(self,'table_x',None)
        bt,bC,bR=self._body_entry()
        G=np.asarray([tray_geom(t) for t in trays],float).tobytes() if trays else None
        key=(np.asarray(plane[0],float).tobytes(),float(plane[1]),G,_key(*_table_args(plane,tx,None)),bC.tobytes(),bR.tobytes())
        cache=self.__dict__.setdefault('_scenes',{})
        e=cache.get(key)
        if e is None:
            parts=[(bt,bC,bR),_table_entry(plane,tx)]+([_tray_entry(trays,plane)] if trays else [])
            t=tuple(x for p in parts for x in p[0])
            if not t or any(s.link!=self.chain.base for s in t):raise ValueError("nonempty base-frame environment required")
            C=np.concatenate([p[1] for p in parts]);R=np.concatenate([p[2] for p in parts]);C.setflags(write=False);R.setflags(write=False)
            fe=None
            if self.col._fast is not None and _accel.enabled2():
                try:fe=_accel.fast_env(C,R)
                except BaseException as ex:_accel.guard(ex);_accel.failed('FastEnv',ex)
            if len(cache)>256:cache.clear()
            e=cache[key]=(t,C,R,fe)
        self.col.install_environment(*e)
        self.scene_ready=True
    def _body_entry(self):
        b=self.__dict__.get('_body_arr')
        if b is None or b[0] is not self.body or len(b[1])!=len(self.body):
            C=np.array([s.center for s in self.body],dtype=float).reshape(-1,3);R=np.array([s.radius for s in self.body],dtype=float)
            C.setflags(write=False);R.setflags(write=False);b=self._body_arr=(self.body,C,R)
        return (tuple(b[0]),b[1],b[2])
    def set_other(self,q_other):
        """#625 arm-to-arm: the other arm's envelope at its MEASURED joints (7-vector in ARM order or dict)."""
        o='r' if self.side=='l' else 'l'
        if not isinstance(q_other,dict):q_other=dict(zip([f'{n}_{o}_joint' for n in ARM],[float(x) for x in q_other]))
        key=tuple(round(q_other.get(f'{n}_{o}_joint',0.),3) for n in ARM)
        if key==getattr(self,'_other_key',None):return
        self._other_key=key
        self.body=_cached(('body',self.side,key),lambda:body_env(self.side,dict(zip([f'{n}_{o}_joint' for n in ARM],key))))
        if getattr(self,'_scene',None) is not None:self.set_scene(*self._scene)
        else:self.col.environment=list(self.body)
    def path_home(self,q):
        """Certified way home from q: [home] if the straight joint path is collision-free, else [lift(q), home] when
        that is, else None (#625: never an uncertified move when a certified one exists)."""
        q=np.clip(np.asarray(q,float),self.chain.lower,self.chain.upper)
        if self.col.certify(q,self.home):return [self.home]
        for dz in (.08,.15):
            qu=self.lift(q,dz)
            if qu is not None and self.col.certify(qu,self.home):return [qu,self.home]
        return None
    def tcp(self,q):
        return self.chain.fk(q)
    def lift(self,q,dz=.08):
        """Straight-up retreat target from q (same orientation, +dz in pelvis z), certified swept path, or None."""
        T=self.chain.fk(q).copy();T[2,3]+=dz
        try:qu=self.chain.ik(T,q,max_iter=150)
        except ValueError:return None
        return qu if self.col.valid(qu) and self.col.certify(q,qu) else None
    def correct(self,q_measured,q_goal,max_step=.05):
        """One position-only Jacobian correction of the commanded joints so the MEASURED TCP reaches the goal TCP
        (compensates PD undershoot under load). Returns a certified command or None."""
        e=self.chain.fk(q_goal)[:3,3]-self.chain.fk(q_measured)[:3,3]
        Jp=self.chain.jacobian(q_measured)[:3]
        dq=Jp.T@np.linalg.solve(Jp@Jp.T+1e-4*np.eye(3),e)
        n=np.abs(dq).max()
        if n>max_step:dq*=max_step/n
        qc=np.clip(np.asarray(q_goal,float)+dq,self.chain.lower,self.chain.upper)
        return qc if self.col.valid(qc) and self.col.certify(q_measured,qc) else None
    def position_error(self,q_measured,T_goal):
        """Distance between the TCP at the measured joints and the goal TCP (m)."""
        return float(np.linalg.norm(self.chain.fk(q_measured)[:3,3]-T_goal[:3,3]))
    def grasp_pose(self,center_xyz,close_axis,approach,point='grip'):
        """TCP pose so that the hand's thumb->finger closing axis maps to close_axis and fingers point along approach.
        point='grip': midpoint of the OPEN finger tips; 'pinch': where thumb and index/middle meshes first touch when
        closing (mesh-derived, 1.8 cm deeper toward the palm) -- used for rim pinches."""
        A,g_t=self._hand_frame(point)
        u=close_axis/np.linalg.norm(close_axis);a=approach-u*(approach@u);a/=np.linalg.norm(a)
        B=np.c_[u,a,np.cross(u,a)];R=B@A.T
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=center_xyz-R@g_t;return T
    def _hand_frame(self,point):
        """(A, g_t) of grasp_pose: hand closing/finger frame and grip point in the tcp frame. Pure function of the robot
        data and hand_mesh: memoised per (point, hand_mesh version); same arithmetic as the former inline code."""
        hm=hand_mesh(self.side);key=(point=='pinch' and bool(hm),next(iter(_HAND_MESH_CACHE),None))
        c=self.__dict__.setdefault('_hand_frames',{})
        v=c.get(key)
        if v is None:
            Rh=J[f"{'left' if self.side=='l' else 'right'}_hand_to_tcp"]['T'][:3,:3]   # hand frame in tcp frame
            gp=np.asarray(hm['pinch_point_hand'],float) if (point=='pinch' and hm) else self.hand['grip_center_hand']
            c_t=Rh@self.hand['close_axis_hand'];z_t=Rh@np.array([0,0,1.]);g_t=Rh@gp
            z_t=z_t-c_t*(z_t@c_t);z_t/=np.linalg.norm(z_t)
            A=np.c_[c_t,z_t,np.cross(c_t,z_t)];A.setflags(write=False);g_t.setflags(write=False)
            v=c[key]=(A,g_t)
        return v
    def solve(self,T,seed=None,w_null=NULL_WEIGHT):
        # 7-DOF redundancy: secondary pull toward the elbow-out home posture (away from the torso), see kinematics.ik
        seeds=([seed] if seed is not None else [])+self.seeds
        if self.col._fast is not None and self.chain._fast is not None and _accel.enabled2():
            r=self._solve_fast(T,seeds,w_null)
            if r is not NotImplemented:return r
        return self.solve_py(T,seed,w_null)
    def _solve_fast(self,T,seeds,w_null):
        """One Rust call for the whole seed loop of solve_py: IK per seed in order, first converged + collision-free q.
        Validation that solve_py repeats per seed but that does not depend on the seed (target rigid transform and
        rotation, preferred posture, parameters) is done once: if it raises, every seed raises -> None. Per-seed
        validation (finite, joint limits) happens in Rust, a rejected seed is skipped like the reference's ValueError."""
        ch=self.chain;n=len(ch.active)
        try:
            S=np.array(seeds,dtype=float)
            if S.shape!=(len(seeds),n):return NotImplemented
        except (ValueError,TypeError):return NotImplemented
        try:
            target=check_transform(T);qp=None
            if w_null>0:qp=ch.validate_q(np.clip(finite(self.home,(n,)),ch.lower,ch.upper))
            if not np.isfinite(w_null) or w_null<0 or w_null>1:raise ValueError('null-space weight')
            check_rotation(target[:3,:3])
        except ValueError:return None
        try:
            r=self.col._fast.solve(np.ascontiguousarray(target,float),np.ascontiguousarray(S),150,.002,.015,.25,.15,
                                   None if qp is None else np.ascontiguousarray(qp,float),float(w_null))
        except BaseException as e:
            _accel.guard(e);_accel.failed('solve',e);return NotImplemented
        return None if r is None else r[1]
    def solve_py(self,T,seed=None,w_null=NULL_WEIGHT):
        """Reference seed loop (Chain.ik + CollisionModel.valid per seed)."""
        for s in ([seed] if seed is not None else [])+self.seeds:
            try:
                q=self.chain.ik(T,s,max_iter=150,q_pref=self.home if w_null>0 else None,w_null=w_null)
                if self.col.valid(q):return q
            except ValueError:pass
        return None
    def plan_pick(self,center,close_axis,approach,width,pre=.08,lift=.08,tilts=(0,.35,.6),point='grip'):
        """Returns dict(q_pre,q_grasp,q_lift,traj) or None. Speed capped at 0.8 rad/s by joint_path/Chain.velocity."""
        if width>self.hand['opposed_open']-.01:return None
        if not self.scene_ready:return None   # no measured table yet: never plan against an assumed scene (#645)
        import itertools
        for t,sg in itertools.product(tilts,(1.,-1.)):
            a=np.asarray(approach,float);u=sg*np.asarray(close_axis,float)   # a 2-finger-opposed grasp is symmetric in the closing axis
            if abs(a[2])<.9: a=a*np.cos(t)+np.array([0,0,-np.sin(t)])  # tilt fingers downward
            Tg=self.grasp_pose(center,u,a,point);Tp=Tg.copy();Tp[:3,3]-=pre*(a/np.linalg.norm(a));Tl=Tg.copy();Tl[2,3]+=lift
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
