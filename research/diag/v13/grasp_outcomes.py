"""Outcome of every pick attempt in the recorded Kaggle runs (v7, v8, v9a, v9b, v11b), by grasp kind.
Outcome = last TRACE phase reached + verdicts; 'placed' = released after over_tray. Output: grasp_outcomes.json."""
import json,re,glob,os,collections
R='/workspace/DeftSort/research/kaggle-sim'
RUNS=[('v7',f'{R}/out_v7/logs/run_sort3_10*.log'),('v8',f'{R}/out_v8/logs/run_sort3_103.log'),
      ('v9a',f'{R}/out_v9a_v1/logs/run_sort3_103.log'),('v9b',f'{R}/out_v9b_f525ed9/logs/run_sort2_101.log'),
      ('v11b',f'{R}/out_v11b_497471e/logs/run_sort2_101.log')]
rows=[]
for run,pat in RUNS:
    for fn in sorted(glob.glob(pat)):
        cur=None
        for i,l in enumerate(open(fn,errors='replace'),1):
            m=re.search(r'\]: pick (\S+) (\S+) -> (\S+) slot (\d+) arm (\w) (\w+)',l)
            if m:
                if cur:rows.append(cur)
                cur=dict(run=run,log=os.path.basename(fn),line=i,obj=m.group(1)[:24],cat=m.group(2),side=m.group(5),kind=m.group(6),phases=[],verdicts={})
                continue
            if cur and 'TRACE {' in l:
                j=json.loads(l[l.index('TRACE ')+6:])
                if j.get('obj','')[:24]!=cur['obj']:continue
                cur['phases'].append(j['phase'])
                if j['phase'] in('lifted','over_tray'):cur['verdicts'][j['phase']]=j.get('verdict')
                if j['phase']=='closed':cur['empty']=j.get('empty')
            if cur and ('grasp check' in l or 'guard:' in l or 'abort' in l or 'deadline' in l) and cur['obj'][:18] in l:
                cur.setdefault('notes',[]).append(f'{i}: '+l.split(']: ')[-1].strip()[:90])
        if cur:rows.append(cur)
def outcome(r):
    p=r['phases']
    if 'released' in p:return 'released_over_tray' if r['verdicts'].get('over_tray') is not False else 'released_verdict_empty'
    if 'lifted' in p:return 'lifted_empty' if r['verdicts'].get('lifted') is False else 'lifted_other'
    if 'guard' in p:return 'guard_abort'
    if 'closed' in p:return 'closed_empty' if r.get('empty') else 'closed_other'
    return 'not_closed'
tab=collections.defaultdict(collections.Counter)
for r in rows:r['outcome']=outcome(r);tab[(r['kind'],r['cat'])][r['outcome']]+=1
json.dump(dict(rows=rows,table={f'{k[0]}/{k[1]}':dict(v) for k,v in tab.items()}),open(os.path.join(os.path.dirname(__file__),'grasp_outcomes.json'),'w'),indent=1)
for k,v in sorted(tab.items()):print(k,dict(v))
for r in rows:
    if r['outcome'].startswith('released'):print(r['run'],r['log'],r['line'],r['obj'],r['kind'],r['outcome'])
