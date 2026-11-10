"""OFFLINE diagnosis (v9a): which robot links / table are in contact at the logged arm postures.
Meshes: UBTECH tiangong2pro_urdf STL (Apache-2.0, same joint names as the organiser URDF); FK: organiser URDF
(team048_solution.arm.fk_link). Vertex-to-vertex distances (approximate, mesh vertices only)."""
import sys,os,json,re,numpy as np,xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution')
from team048_solution import arm as A
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/tools')
from fit_spheres import read_stl,rpy_matrix
U='/workspace/competitors/UBTECH-Robot__tiangong2pro_tf/src/tiangong2pro_urdf'
R=ET.parse(U+'/urdf/tiangong2.0_pro_with_hands.urdf').getroot()
MESH={}
for l in R.findall('link'):
    c=l.find('collision') or l.find('visual')
    if c is None:continue
    m=c.find('geometry/mesh')
    if m is None:continue
    o=c.find('origin');xyz=np.array([float(x) for x in (o.get('xyz') if o is not None else '0 0 0').split()])
    rpy=[float(x) for x in (o.get('rpy') if o is not None else '0 0 0').split()]
    f=os.path.join(U,'meshes',os.path.basename(m.get('filename')))
    if not os.path.exists(f):continue
    V=np.unique(np.round(read_stl(f),4),axis=0)
    if len(V)>4000:V=V[np.random.default_rng(0).choice(len(V),4000,replace=False)]
    MESH[l.get('name')]=V@rpy_matrix(*rpy).T+xyz
ARMJ=A.ARM
def chain_links(s):
    pre='left' if s=='l' else 'right';H='L' if s=='l' else 'R'
    arm=[A.J[f'{n}_{s}_joint']['child'] for n in ARMJ]
    hand=[H+'_base_link']+[k for k in MESH if k.startswith(pre+'_') and k!=pre+'_tcp_link']
    return arm,hand
TORSO=['pelvis','waist_yaw_link','head_yaw_link','head_pitch_link','head_roll_link','hip_roll_l_link','hip_roll_r_link','hip_pitch_l_link','hip_pitch_r_link','hip_yaw_l_link','hip_yaw_r_link','knee_pitch_l_link','knee_pitch_r_link']
TORSO=[t for t in TORSO if t in MESH and t in A.BYCHILD or t=='pelvis']
def pts(link,qd):
    T=A.fk_link(link,qd);return MESH[link]@T[:3,:3].T+T[:3,3]
def qdict(side,q):return {f'{n}_{side}_joint':float(v) for n,v in zip(ARMJ,q)}
def report(side,q,other_q=None,table=(0.239,-0.180),label=''):
    o='r' if side=='l' else 'l'
    qd=qdict(side,q);qd.update(qdict(o,other_q if other_q is not None else np.zeros(7)))
    arm,hand=chain_links(side);oarm,ohand=chain_links(o)
    moving=arm+hand;P={k:pts(k,qd) for k in moving+TORSO+oarm+ohand if k in MESH}
    rows=[]
    def md(a,b):
        if a not in P or b not in P:return None
        return float(cKDTree(P[b]).query(P[a],k=1)[0].min())
    # same arm: non-adjacent (skip parent-child by URDF joint)
    adj=lambda a,b:(A.BYCHILD.get(a) and A.J[A.BYCHILD[a]]['parent']==b) or (A.BYCHILD.get(b) and A.J[A.BYCHILD[b]]['parent']==a)
    for i,a in enumerate(arm):
        for b in arm[i+2:]+hand[:1]:
            if a!=b and not adj(a,b):rows.append(('self',a,b,md(a,b)))
    hb=hand[0]
    for b in TORSO:
        for a in arm[2:]+[hb]:rows.append(('body',a,b,md(a,b)))
    for a in arm[3:]+[hb]:
        for b in oarm[3:]+[ohand[0]]:rows.append(('other',a,b,md(a,b)))
    # table: near edge x0, top z0 (pelvis frame, sim log 'table top -0.180 m vs pelvis', measured near edge)
    x0,z0=table;tab=[]
    for a in moving:
        if a not in P:continue
        p=P[a];inside=(p[:,0]>x0)&(p[:,2]<z0)
        if inside.any():tab.append((a,int(inside.sum()),round(float((z0-p[inside,2]).max()),3)))
    close=sorted([r for r in rows if r[3] is not None and r[3]<.012],key=lambda r:r[3])
    tcp=A.Arm(side).tcp(np.clip(q,*[A.Arm(side).chain.lower,A.Arm(side).chain.upper]))[:3,3] if False else None
    return dict(label=label,side=side,contacts=[(k,a,b,round(d,4)) for k,a,b,d in close[:8]],table_penetration=tab)
if __name__=='__main__':
    log=sys.argv[1];out=[]
    for line in open(log):
        if 'TRACE' not in line:continue
        d=json.loads(line.split('TRACE ',1)[1])
        if d['phase'] not in ('start','pre','guard'):continue
        r=report(d['side'],np.array(d['q']),label=f"{d['t']} {d['phase']} {d['obj'][:14]} |F|={np.linalg.norm(d['F']):.0f}N")
        out.append(r);print(json.dumps(r))
