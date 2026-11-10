"""Build test/fixtures/v7v8_grasp_traces.json from the Kaggle v7/v8 TRACE lines (sort3 pool 3, force 1.0, speed 1.0).
held_at_lift labels: v7 calibration (grip.py: Ecoforms x2, Threshold s102 loaded) + v8 Perricone (placed, score 10)."""
import json,re,glob,sys
OUT=sys.argv[1]
HELD={('v7',101,'Ecoforms_Plant_Container_B4_Har'),('v7',103,'Ecoforms_Plant_Container_B4_Har'),
      ('v7',102,'Threshold_Ramekin_White_Porcelain'),('v8',103,'Perricone_MD_AcylGlutathione_Eye_Lid_Serum')}
att=[]
for run in ('v7','v8'):
    for f in sorted(glob.glob(f'/workspace/fixes/kaggle-sim/out_{run}/logs/run_sort3_*.log')):
        seed=int(re.search(r'_(\d+)\.log',f).group(1));cur=None
        for l in open(f,errors='ignore'):
            m=re.search(r'TRACE (\{.*\})',l)
            if not m:continue
            x=json.loads(m.group(1))
            if x['phase']=='start':
                cur=dict(run=run,seed=seed,obj=x['obj'],kind=x.get('kind'),side=x['side'],phases={});att.append(cur)
            if cur is not None and x['phase'] in('pre','at_grasp','corrected','closed','lifted','over_tray'):
                cur['phases'][x['phase']]=dict(t=x['t'],fing=x['fing'],eff=x['eff'],tcp_err=x.get('tcp_err'))
att=[a for a in att if 'closed' in a['phases']]
for a in att:a['held_at_lift']=(a['run'],a['seed'],a['obj']) in HELD if 'lifted' in a['phases'] else None
json.dump(dict(source='Kaggle out_v7/out_v8 run_sort3_*.log TRACE, force 1.0 speed 1.0, hand open cmd [1,1,1,1,1,0]',
               attempts=att),open(OUT,'w'),indent=0)
print(len(att),sum(1 for a in att if a['held_at_lift']),sum(1 for a in att if a['held_at_lift'] is False))
