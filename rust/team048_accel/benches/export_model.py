"""Export the real left-arm model (URDF chain, collision spheres, measured-plane scene) + sample inputs for criterion.
Run with the team048_solution package importable:  python benches/export_model.py"""
import json,os,sys,glob,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,'..','..','..','src','team048_solution'))
from team048_solution import arm as ARM
A=ARM.Arm('l');n=np.array([-.0699,0,.9976]);A.set_scene((n/np.linalg.norm(n),.1788/np.linalg.norm(n)),[(.34,.69),(.34,-.69),(.53,0.)])
ch=A.chain;col=A.col;rng=np.random.default_rng(0)
index={ch.base:0};[index.__setitem__(j.child,i+1) for i,j in enumerate(ch.joints)]
S=col.spheres;pairs=[(i,k) for i in range(len(S)) for k in range(i+1,len(S)) if S[i].link!=S[k].link and frozenset((S[i].link,S[k].link)) not in col.ignored]
Q=[np.clip(rng.uniform(ch.lower,ch.upper)*.5,ch.lower,ch.upper) for _ in range(256)]
free=[q for q in (np.clip(rng.uniform(ch.lower,ch.upper)*.4,ch.lower,ch.upper) for _ in range(400)) if col.valid(q) and col.certify(A.home,q)][:16]
d=dict(origins=[j.origin.ravel().tolist() for j in ch.joints],axes=[j.axis.tolist() for j in ch.joints],movable=[bool(j.movable) for j in ch.joints],
       tcp=ch.tcp.ravel().tolist(),lower=ch.lower.tolist(),upper=ch.upper.tolist(),home=A.home.tolist(),
       frame=[index[s.link] for s in S],centers=[s.center.tolist() for s in S],radii=[float(s.radius) for s in S],pairs=pairs,
       levers=col.levers.tolist(),margin=col.margin,env_centers=col._env_centers.tolist(),env_radii=col._env_radii.tolist(),
       q=[q.tolist() for q in Q],targets=[ch.fk(q).ravel().tolist() for q in Q],free=[q.tolist() for q in free])
json.dump(d,open(os.path.join(HERE,'data','arm_l.json'),'w'))
RES=os.environ.get('TEAM048_RESEARCH',os.path.join(HERE,'..','..','..','research'))   # Kaggle frames + cycle.py (not in the submission)
fr=sorted(glob.glob(os.path.join(os.environ.get('TEAM048_V7_TRACE',os.path.join(RES,'kaggle-sim','out_v7','trace')),'*','*closed*.npz')))
if fr:
    sys.path.insert(0,os.path.join(RES,'rust'));import cycle
    z=np.load(fr[0]);P,valid=cycle.frame_cloud(z);tb=valid&(P[...,0]>.15)&(P[...,0]<.75)&(np.abs(P[...,1])<.7)
    p=P[tb];p=p[np.isfinite(p).all(1)];np.ascontiguousarray(p,'<f8').tofile(os.path.join(HERE,'data','plane_points.f64'))
    from team048_solution import perception as PE
    pl=PE.table_plane(P,tb);h=P@pl[0]+pl[1];m=tb&(h>.012)&(h<.35)
    np.ascontiguousarray(P,'<f8').tofile(os.path.join(HERE,'data','frame_points.f64'));np.ascontiguousarray(m).astype(np.uint8).tofile(os.path.join(HERE,'data','frame_mask.u8'))
    json.dump(dict(h=P.shape[0],w=P.shape[1]),open(os.path.join(HERE,'data','frame.json'),'w'))
print('exported',len(S),'spheres',len(col.environment),'env',len(free),'free sweeps')
