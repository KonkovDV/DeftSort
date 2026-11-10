"""Measured phase durations (sim s) from Kaggle v7 TRACE lines (old planner, RTF~0.23)."""
import re,json,glob
rows=[]
for f in sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/logs/run_sort3_10*.log')):
    obs=[float(m) for m in re.findall(r'observe t ([0-9.]+)',open(f).read())]
    ev=[]
    for line in open(f):
        m=re.search(r'TRACE (\{.*\})',line)
        if m:
            r=json.loads(m.group(1));ev.append((r['t'],r['phase'],r['obj']))
    st=re.search(r'startup ([0-9.]+) s',open(f).read()).group(1)
    cyc=[];cur=None
    for t,ph,o in ev:
        if ph=='start':cur={'obj':o,'start':t}; cyc.append(cur)
        elif cur is not None:cur[ph]=t
    for c in cyc:
        prev=[x for x in obs if x<=c['start']];c['plan']=round(c['start']-max(prev),2) if prev else None
        if 'lifted' in c:c['pick']=round(c['lifted']-c['start'],2)
        if 'home' in c and 'lifted' in c:c['place']=round(c['home']-c['lifted'],2)
        if 'pre' in c:c['to_pre']=round(c['pre']-c['start'],2)
        if 'closed' in c and 'pre' in c:c['approach_close']=round(c['closed']-c['pre'],2)
        if 'over_tray' in c and 'lifted' in c:c['carry']=round(c['over_tray']-c['lifted'],2)
        if 'released' in c and 'home' in c:c['retreat']=round(c['home']-c['released'],2)
    rows.append(dict(file=f.split('/')[-1],startup=float(st),cycles=cyc))
if __name__=='__main__':
    import numpy as np
    for r in rows:
        print(r['file'],'startup',r['startup'])
        for c in r['cycles']:print('  ',c['obj'][:22],{k:c.get(k) for k in ('plan','to_pre','approach_close','pick','carry','retreat','place')})
    allc=[c for r in rows for c in r['cycles']]
    for k in ('plan','to_pre','approach_close','pick','carry','retreat','place'):
        v=np.array([c[k] for c in allc if c.get(k) is not None]);print(k,len(v),'mean %.1f p50 %.1f p90 %.1f max %.1f'%(v.mean(),np.median(v),np.quantile(v,.9),v.max()))
