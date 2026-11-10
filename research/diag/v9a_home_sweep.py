"""Sweep of the straight joint path start posture (q=0) -> home: hand points vs the table near edge/top.
Table from the v9a run: near edge 0.239 m (torso camera, 'table near edge 0.239 measured'), top -0.180 m vs pelvis
(sim log line '[hack] polygon ... table top ... -0.180 m vs pelvis'; used here for the offline analysis only)."""
import sys,numpy as np
sys.argv=[sys.argv[0],'x'];exec(open('v9a_mesh_contacts.py').read().split("if __name__")[0])
for s in 'lr':
    a=A.Arm(s);arm,hand=chain_links(s)
    for k in np.linspace(0,1,11):
        q=k*a.home;qd=qdict(s,q);P=np.vstack([pts(l,qd) for l in hand if l in MESH])
        under=(P[:,0]>.239)&(P[:,2]<-.180)
        print(s,round(k,1),'hand x max %.3f z min %.3f'%(P[:,0].max(),P[:,2].min()),'pts under/inside table',int(under.sum()))
