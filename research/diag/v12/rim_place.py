import json,sys,numpy as np
R='/workspace/DeftSort/src/team048_solution';sys.path.insert(0,R)
from team048_solution import arm as ARM
runs={'v7s101':'/workspace/DeftSort/research/kaggle-sim/out_v7/logs/run_sort3_101.log','v7s102':'/workspace/DeftSort/research/kaggle-sim/out_v7/logs/run_sort3_102.log',
'v7s103':'/workspace/DeftSort/research/kaggle-sim/out_v7/logs/run_sort3_103.log','v8':'/workspace/DeftSort/research/kaggle-sim/out_v8/logs/run_sort3_103.log',
'v9b':'/workspace/DeftSort/research/kaggle-sim/out_v9b_f525ed9/logs/run_sort2_101.log','v11b':'/workspace/kaggle-out/team048-sim-v11b/logs/run_sort2_101.log'}
for run,fn in runs.items():
    L=open(fn).read().split('\n');start={}
    for i,l in enumerate(L):
        if 'TRACE ' not in l:continue
        d=json.loads(l.split('TRACE ',1)[1])
        if d['phase']=='start' and d.get('kind')=='rim':start[d['obj']]=(d,i+1)
        if d['phase'] in('at_grasp','over_tray','released') and d['obj'] in start and d.get('q'):
            A=ARM.Arm(d['side']);Am,gt=A._hand_frame('pinch')
            T=A.tcp(np.clip(np.array(d['q']),A.chain.lower,A.chain.upper))
            pin=T[:3,:3]@gt+T[:3,3];u=T[:3,:3]@Am[:,0]
            print(run,f"{fn.split('/')[-1]}:{i+1}",d['obj'][:10],d['phase'],d['t'],'pinch',np.round(pin,3),'u',np.round(u,2),'start c',start[d['obj']][0].get('center'))
