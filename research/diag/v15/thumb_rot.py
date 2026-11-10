"""Thumb rotation channel (DOF 6) in the Kaggle runs with HAND_FORCE_THUMB_ROT = 1.0 and finger force 0.4 (v9a, v9b,
v11b): HAND lines (10 Hz during guarded phases: approach guard, close1, close2, ...) and TRACE records.
Simulator effort = |tau| / (lim0 * r) clipped to 1 (grip.effort_held_threshold, hack_sim/robot_io.py), so at r = 1.0 the
logged effort IS the torque as a fraction of lim0 and a lower limit r would saturate only where effort >= r.
Usage: python thumb_rot.py [run_log ...]  (default: the three /workspace/kaggle-out runs + v7/v8 for comparison)"""
import sys,json,glob,re,collections
LOGS=sys.argv[1:] or (sorted(glob.glob('/workspace/kaggle-out/team048-sim-v*/logs/run_sort*_1*.log'))+
     sorted(glob.glob('/workspace/DeftSort/research/kaggle-sim/out_v[78]/logs/run_sort*_1*.log')))
out={}
for fn in LOGS:
    hand=[];trace=[]
    for line in open(fn,errors='replace'):
        m=re.search(r'\]: (HAND|TRACE) (\{.*\})\s*$',line)
        if not m:continue
        try:r=json.loads(m.group(2))
        except ValueError:continue
        (hand if m.group(1)=='HAND' else trace).append(r)
    ph=collections.defaultdict(lambda:dict(n=0,pos6_max_abs=0.,eff6_max=0.,eff6_at=None))
    for r in hand:
        if r.get('pos') is None or r.get('eff') is None or len(r['pos'])<6:continue
        d=ph[r['phase']];d['n']+=1;d['pos6_max_abs']=max(d['pos6_max_abs'],abs(r['pos'][5]))
        if r['eff'][5]>d['eff6_max']:d['eff6_max']=r['eff'][5];d['eff6_at']=(r['t'],r['side'])
    # first HAND sample of every close1 sequence: thumb already in opposition before the fingers squeeze?
    first=[];prev=None
    for r in hand:
        if r['phase']=='close1' and prev!='close1' and r.get('pos'):first.append((r['t'],r['side'],r['pos'][5],r.get('cmd',[None]*6)[5]))
        prev=r['phase']
    tr=[(r['t'],r['phase'],r['side'],r['obj'][:14],r['fing'][5],r['eff'][5],r.get('thumb')) for r in trace if r.get('fing') and r.get('eff')]
    out[fn.split('/')[-3]+'/'+fn.split('/')[-1]]=dict(hand_samples=len(hand),phases=dict(ph),close1_first=first,trace=tr,
        eff6_max_all=max([r['eff'][5] for r in hand if r.get('eff')]+[t[5] for t in tr]+[0.]),
        pos6_max_abs_all=max([abs(r['pos'][5]) for r in hand if r.get('pos')]+[abs(t[4]) for t in tr]+[0.]))
print(json.dumps(out,indent=1))
