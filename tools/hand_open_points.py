"""Offline: surface sample of the open Inspire hand (fingers open, thumb rotated into opposition = the hand_open
command [1,1,1,1,1,0]) in the TCP frame, from the UBTECH tiangong2pro_urdf STL meshes (Apache-2.0, robot assets; hand
joint origins identical to the organiser URDF, see tools/hand_from_meshes.py). Used by arm.grasp_clear_shift to check
the open hand against the measured object footprint box at the grasp pose and along the approach.
Triangles are sampled at <= STEP spacing (vertices alone miss large flat faces), then voxel-downsampled to STEP.
Usage: python tools/hand_open_points.py <meshes_dir>"""
import sys,os,json,struct,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..','src','team048_solution'))
from team048_solution import arm as A
STEP=.004
def tris(p):
    b=open(p,'rb').read();n=struct.unpack('<I',b[80:84])[0]
    a=np.frombuffer(b[84:84+n*50],dtype=np.dtype([('n','<3f4'),('v','<9f4'),('a','<u2')]))
    return a['v'].reshape(-1,3,3).astype(float)
def sample(T,step=STEP):
    out=[T.reshape(-1,3)]
    for t in T:
        e=max(np.linalg.norm(t[1]-t[0]),np.linalg.norm(t[2]-t[0]),np.linalg.norm(t[2]-t[1]));k=int(np.ceil(e/step))
        if k<=1:continue
        i,j=np.meshgrid(np.arange(k+1),np.arange(k+1),indexing='ij');m=(i+j)<=k;i=i[m]/k;j=j[m]/k
        out.append(t[0]+np.outer(i,t[1]-t[0])+np.outer(j,t[2]-t[0]))
    return np.vstack(out)
def main(M):
    out={'source':'UBTECH-Robot/tiangong2pro_tf meshes (Apache-2.0), tools/hand_open_points.py','step':STEP,
         'posture':'fingers open (0 rad), thumb_1 at its upper limit (opposition), thumb bend 0'}
    for side,p in (('l','left'),('r','right')):
        q={f'{p}_thumb_1_joint':A.J[f'{p}_thumb_1_joint']['lim'][1]}
        Ti=np.linalg.inv(A.fk_link(A.SIDE[side]['tip'],q));parts={}
        links=[A.SIDE[side]['hand']]+[f'{p}_{f}_{k}' for f in ('index','middle','ring','little') for k in (1,2)]+[f'{p}_thumb_{k}' for k in (1,2,3,4)]
        for l in links:
            P=sample(tris(os.path.join(M,l+'.STL')));T=Ti@A.fk_link(l,q);W=(T@np.c_[P,np.ones(len(P))].T).T[:,:3]
            _,ix=np.unique(np.floor(W/STEP).astype(np.int64),axis=0,return_index=True)
            parts[l]=np.round(W[np.sort(ix)],4).tolist()
        out[side]={'points_tcp':[x for l in links for x in parts[l]],'parts':[[('palm' if 'base' in l else l.split('_',1)[1]),len(parts[l])] for l in links]}
        print(side,{l:len(v) for l,v in parts.items()},'total',len(out[side]['points_tcp']))
    json.dump(out,open(os.path.join(HERE,'..','src','team048_solution','team048_solution','data','hand_open_points.json'),'w'),separators=(',',':'))
if __name__=='__main__':main(sys.argv[1])
