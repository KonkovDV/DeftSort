"""Mesh check of the v9fix start path (start posture q=0 -> roadmap waypoints -> home) against the table (measured
plane + near edge of the v7 s102 start frames) and the robot body (UBTECH STL meshes)."""
import sys,numpy as np
exec(open('/workspace/fixes/diag/v9a_mesh_contacts.py').read().split("if __name__")[0])
P0=np.load('/tmp/startpaths.npy',allow_pickle=True)
import team048_solution.perception as PE
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution/test')
from test_compliance import cloud,V7
f=np.load(V7);K=f['K']
P=cloud(A.fk_link('camera_head_link',{'head_pitch_joint':float(f['head_pitch'])}),f['head_depth'].astype(float),K)
valid=f['head_depth']>0;pl=PE.table_plane(P,valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7))
Pt=cloud(A.fk_link('camera_body_front_link'),f['torso_depth'].astype(float),K);edge=PE.near_edge(Pt,f['torso_depth']>0,pl)
n,d=pl;top=lambda x,y:-(d+n[0]*x+n[1]*y)/n[2]
for si,side in enumerate('lr'):
    path=[np.zeros(7)]+list(P0[si]);arm,hand=chain_links(side);o='r' if side=='l' else 'l'
    worst_tab=0;min_body=9;min_hip=9
    for a,b in zip(path[:-1],path[1:]):
        for k in np.linspace(0,1,41):
            q=a+(b-a)*k;qd=qdict(side,q)
            Q=np.vstack([pts(l,qd) for l in arm+hand if l in MESH])
            inside=(Q[:,0]>edge)&(Q[:,2]<top(Q[:,0],Q[:,1]))
            worst_tab=max(worst_tab,int(inside.sum()))
            for b2 in ('waist_yaw_link','hip_pitch_l_link' if side=='l' else 'hip_pitch_r_link','hip_yaw_l_link' if side=='l' else 'hip_yaw_r_link'):
                dd=float(cKDTree(pts(b2,{})).query(np.vstack([pts(l,qd) for l in arm[2:]+hand if l in MESH]))[0].min())
                if b2=='waist_yaw_link':min_body=min(min_body,dd)
                else:min_hip=min(min_hip,dd)
    print(side,'edge %.3f'%edge,'waypoints',len(path)-1,'max mesh vertices inside table',worst_tab,'min dist torso %.4f hips %.4f'%(min_body,min_hip))
