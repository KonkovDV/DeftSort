"""Offline: Inspire-hand geometry from the UBTECH tiangong2pro_urdf STL meshes (Apache-2.0, robot assets; joint
origins of the hand verified identical to the organiser URDF: 32/32 joints). Writes data/hand_mesh_geometry.json:
open pad gap (thumb distal vs index/middle distal meshes), first-contact pinch point in the hand (X_base_link) frame,
and thumb surface sample points for the collision envelope. Mapping command->joint: angle=(1-cmd)*upper (1 open).
Usage: python tools/hand_from_meshes.py <meshes_dir>"""
import sys,os,json,struct,numpy as np
from scipy.spatial import cKDTree
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..','src','team048_solution'))
from team048_solution import arm as A
def stl(p):
    b=open(p,'rb').read();n=struct.unpack('<I',b[80:84])[0]
    a=np.frombuffer(b[84:84+n*50],dtype=np.dtype([('n','<3f4'),('v','<9f4'),('a','<u2')]))
    return np.unique(a['v'].reshape(-1,3).astype(float),axis=0)
def q_of(p,cf,ct):
    lim=lambda j:A.J[j]['lim'][1]
    q={f'{p}_thumb_1_joint':lim(f'{p}_thumb_1_joint'),f'{p}_thumb_2_joint':ct*lim(f'{p}_thumb_2_joint')}
    q[f'{p}_thumb_3_joint']=q[f'{p}_thumb_2_joint']*1.1425;q[f'{p}_thumb_4_joint']=q[f'{p}_thumb_3_joint']*.7508   # URDF mimic
    for f in ('index','middle','ring','little'):q[f'{p}_{f}_1_joint']=cf*lim(f'{p}_{f}_1_joint');q[f'{p}_{f}_2_joint']=q[f'{p}_{f}_1_joint']*1.1169
    return q
def main(M):
    out={}
    for side,p in (('l','left'),('r','right')):
        def pts(link,q):
            V=stl(os.path.join(M,link+'.STL'));T=A.fk_link(link,q);H=A.fk_link(A.SIDE[side]['hand'],q)
            return (np.linalg.inv(H)@T@np.c_[V,np.ones(len(V))].T).T[:,:3]
        res=[]
        for c in np.linspace(0,1,41):
            q=q_of(p,c,c);th=pts(f'{p}_thumb_4',q);F=np.vstack([pts(f'{p}_{f}_2',q) for f in ('index','middle')])
            d,i=cKDTree(F).query(th);k=int(np.argmin(d));res.append((float(c),float(d[k]),((th[k]+F[i[k]])/2).tolist()))
        first=next(r for r in res if r[1]<.004)
        q0=q_of(p,0,0);thumb=np.vstack([pts(f'{p}_thumb_{k}',q0) for k in (2,3,4)])
        # thumb envelope: 6 points along the thumb (k-means-free: quantiles along its principal axis)
        c0=thumb.mean(0);u=np.linalg.svd(thumb-c0)[2][0];s=(thumb-c0)@u
        env=[thumb[(s>=a)&(s<=b)].mean(0).tolist() for a,b in zip(np.quantile(s,np.linspace(0,1,7))[:-1],np.quantile(s,np.linspace(0,1,7))[1:])]
        out[side]=dict(pad_gap_open=res[0][1],pinch_closure=first[0],pinch_point_hand=first[2],thumb_env_hand=env,
                       thumb_env_radius=float(np.max(np.linalg.norm(thumb-c0-np.outer(s,u),axis=1))))
    out['source']='UBTECH-Robot/tiangong2pro_tf meshes (Apache-2.0), tools/hand_from_meshes.py'
    json.dump(out,open(os.path.join(HERE,'..','src','team048_solution','team048_solution','data','hand_mesh_geometry.json'),'w'),indent=1)
    print(json.dumps({k:(v if k=='source' else {a:b for a,b in v.items() if a!='thumb_env_hand'}) for k,v in out.items()},indent=1))
if __name__=='__main__':main(sys.argv[1])
