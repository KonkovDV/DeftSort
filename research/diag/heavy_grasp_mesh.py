"""Open-hand mesh vs object box at the planned side-grasp pose (grasp frame, orientation independent).
Robot: UBTECH tiangong2pro meshes (same joints as the organiser URDF, tools/hand_from_meshes.py). Object: objects.yaml
dims via priors.json (w_min = caliper width across the closing axis, w_max = extent along the approach, height).
Reports, per object and plan tilt/sign, the hand vertices inside the object box at the grasp pose and along the last
8 cm approach (pre -> grasp), split by hand part (palm, thumb, fingers)."""
import sys,os,json,struct,numpy as np
R='/workspace/solution/nssr-sol-team-048/src/team048_solution'
sys.path.insert(0,R)
from team048_solution import arm as A
M='/workspace/competitors/UBTECH-Robot__tiangong2pro_tf/src/tiangong2pro_urdf/meshes/'
def stl(p):
    b=open(p,'rb').read();n=struct.unpack('<I',b[80:84])[0]
    a=np.frombuffer(b[84:84+n*50],dtype=np.dtype([('n','<3f4'),('v','<9f4'),('a','<u2')]))
    return np.unique(a['v'].reshape(-1,3).astype(float),axis=0)
def open_q(p):
    q={f'{p}_thumb_1_joint':A.J[f'{p}_thumb_1_joint']['lim'][1]}
    return q
def hand_pts(side):
    p='left' if side=='l' else 'right';q=open_q(p);tcp=A.fk_link(A.SIDE[side]['tip'],q);Ti=np.linalg.inv(tcp)
    out={}
    links=[A.SIDE[side]['hand']]+[f'{p}_{f}_{k}' for f in ('index','middle','ring','little') for k in (1,2)]+[f'{p}_thumb_{k}' for k in (1,2,3,4)]
    for l in links:
        V=stl(M+l+'.STL');T=Ti@A.fk_link(l,q);out[l]=(T@np.c_[V,np.ones(len(V))].T).T[:,:3]
    return out
def part(l):
    return 'palm' if 'base' in l else ('thumb' if 'thumb' in l else l.split('_')[1])
PRI=json.load(open(R+'/team048_solution/data/priors.json'))
def side_eval(arm,H,name,tilt,sg,shift=0.,pre=.08,steps=17,w=None,L=None):
    p=PRI[name];w=p['w_min'] if w is None else w;L=p['w_max'] if L is None else L;h=p['height']
    u=np.array([0,sg*1.,0]);a0=np.array([1.,0,0]);a=a0*np.cos(tilt)+np.array([0,0,-np.sin(tilt)])
    z=max(.5*h,min(.04,h-.01))
    c=np.array([shift,0,z])            # grip centre; object box centred at x=0, bottom z=0
    T=arm.grasp_pose(c,u,a,'grip')
    res={}
    for s in np.linspace(1,0,steps):   # s=1 pre, s=0 grasp
        Ts=T.copy();Ts[:3,3]-=s*pre*a/np.linalg.norm(a)
        for l,P in H.items():
            W=(Ts@np.c_[P,np.ones(len(P))].T).T[:,:3]
            ins=(np.abs(W[:,0])<L/2)&(np.abs(W[:,1])<w/2)&(W[:,2]>0)&(W[:,2]<h)
            if ins.any():
                d=np.min(np.c_[L/2-np.abs(W[ins,0]),w/2-np.abs(W[ins,1]),W[ins,2],h-W[ins,2]],axis=1).max()
                k=part(l);res.setdefault(k,[s,0.]);res[k][0]=min(res[k][0],s) if False else res[k][0];res[k][1]=max(res[k][1],d)
                if s==0:res.setdefault(k+'@grasp',d)
    return res
if __name__=='__main__':
    arm=A.Arm('l');H=hand_pts('l')
    for name in sys.argv[1:] or [n for n,v in PRI.items() if v['category'] in ('box','jar')]:
        for tilt in (0,.35,.6):
            for sg in (1,-1):
                r=side_eval(arm,H,name,tilt,sg)
                print(name[:24].ljust(24),PRI[name]['mass'],'w%.3f L%.3f h%.3f'%(PRI[name]['w_min'],PRI[name]['w_max'],PRI[name]['height']),'tilt',tilt,'sg',sg,{k:round(v[1] if isinstance(v,list) else v,4) for k,v in r.items()})
