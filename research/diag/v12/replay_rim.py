"""CPU replay of every recorded rim attempt: rim candidates with the v12 torque gate, rim_closed, held_offset check."""
import json,sys,glob,os,numpy as np
R='/workspace/DeftSort/src/team048_solution';sys.path.insert(0,R);sys.path.insert(0,R+'/test');import test_node_stub
from team048_solution import arm as ARM, grasp as GR, perception as PE, node as N
K=np.load(R+'/test/fixtures/v7_s102_start_frames.npz')['K'].reshape(3,3)
KS='/workspace/DeftSort/research/kaggle-sim/'
RUNS=[('v7 s101',KS+'out_v7/logs/run_sort3_101.log',KS+'out_v7/trace/sort3_101'),('v7 s102',KS+'out_v7/logs/run_sort3_102.log',KS+'out_v7/trace/sort3_102'),
      ('v7 s103',KS+'out_v7/logs/run_sort3_103.log',KS+'out_v7/trace/sort3_103'),('v8',KS+'out_v8/logs/run_sort3_103.log',KS+'out_v8/trace/sort3_103'),
      ('v9b',KS+'out_v9b_f525ed9/logs/run_sort2_101.log',KS+'out_v9b_f525ed9/trace/sort2_101'),
      ('v11b','/workspace/kaggle-out/team048-sim-v11b/logs/run_sort2_101.log','/workspace/kaggle-out/team048-sim-v11b/trace/sort2_101')]
def scene(f):
    d=np.load(f);T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(d['head_pitch'])});P=PE.deproject(d['depth'],K,T);v=d['depth']>0
    reg=PE.table_region(P,v);pl=PE.table_plane(P,reg)
    dets=PE.segment(P,d['rgb'],reg,pl,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
    return P,pl,dets
out=[]
for run,log,tr in RUNS:
    frames={float(os.path.basename(f)[:7]):f for f in glob.glob(tr+'/*.npz')}
    L=open(log).read().split('\n');pend=None
    for i,l in enumerate(L):
        if 'TRACE ' not in l:continue
        d=json.loads(l.split('TRACE ',1)[1])
        if d['phase']=='start' and d.get('kind')=='rim':
            pend=(d,i+1)
            ag=None
            for l2 in L[i+1:i+40]:
                if 'TRACE ' in l2:
                    d2=json.loads(l2.split('TRACE ',1)[1])
                    if d2['phase']=='start':break
                    if d2['phase']=='at_grasp':ag=d2;break
            d=ag if ag else dict(t=d['t'],side=d['side'])
            st,ln=pend;pend=None;name=st['obj']
            before=[t for t in frames if t<st['t']];after=sorted(t for t in frames if t>d['t'] and 'lifted' in frames[t] and name[:20] in frames[t])
            tf=max(before) if before else (after[0] if after else None)
            if tf is None:out.append(dict(run=run,line=ln,obj=name,frame=None));continue
            P,pl,dets=scene(frames[tf]);pc=np.array(st['center'][:2]);best=None
            for s_ in dets:
                cl=P[s_.pixels[:,0],s_.pixels[:,1]]
                try:f0=GR.footprint(cl,pl);c0,r0=GR.circle_fit(f0['hull'])
                except Exception:continue
                e=abs(np.linalg.norm(pc-c0)-r0)
                if e<.03 and (best is None or e<best[0]):best=(e,cl,c0,r0)
            if best is None:out.append(dict(run=run,line=ln,obj=name,frame=tf,seg=None));continue
            _,cl,c0,r0=best;others=[P[s_.pixels[:,0],s_.pixels[:,1]] for s_ in dets if not np.shares_memory(P[s_.pixels[:,0],s_.pixels[:,1]],cl)]
            m=N.PRI.get(name,{}).get('mass')
            f_old,C_old=GR.candidates(cl,pl,'dish',(),mass=None);f,C=GR.candidates(cl,pl,'dish',(),mass=m)
            rec=dict(run=run,line=ln,obj=name[:14],frame=tf,r0=round(float(r0),4),mass=m,torque=None if f.get('rim_torque') is None else round(f['rim_torque'],3),
                     closed=f['rim_closed'],rim_old=sum(g['kind']=='rim' for g in C_old),rim_new=sum(g['kind']=='rim' for g in C))
            if d.get('q'):   # FK check of held_offset: pinch of the measured grasp pose + offset vs measured dish centre
                A=ARM.Arm(d['side']);q=np.clip(np.array(d['q']),A.chain.lower,A.chain.upper);T=A.tcp(q)
                Am,gt=A._hand_frame('pinch');pin=T[:3,:3]@gt+T[:3,3];u=T[:3,:3]@Am[:,0]
                rec['fk_centre_err_mm']=round(1000*float(np.linalg.norm(pin[:2]+r0*u[:2]-c0)),1)
            out.append(rec)
for r in out:print(json.dumps(r))
json.dump(out,open('/workspace/v12/replay_rim.json','w'),indent=1)
