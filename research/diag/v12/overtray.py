import json,sys,glob
sys.path.insert(0,'/workspace/DeftSort/src/team048_solution')
from team048_solution import grip as GRIP
KS='/workspace/DeftSort/research/kaggle-sim/'
for fn in sorted(glob.glob(KS+'out_v*/logs/run_sort*_1*.log'))+['/workspace/kaggle-out/team048-sim-v11b/logs/run_sort2_101.log']:
    L=open(fn).read().split('\n');last={}
    for i,l in enumerate(L):
        if 'TRACE ' not in l:continue
        try:d=json.loads(l.split('TRACE ',1)[1])
        except Exception:continue
        if d['phase']=='closed':last[d['obj']]=d
        if d['phase'] in('lifted','over_tray') and d.get('fing'):
            c=d.get('cmd');cl=last.get(d['obj'],{})
            print(fn.split('kaggle-sim/')[-1].split('/logs')[0][-14:],i+1,d['obj'][:12],d['phase'],'fing',d['fing'][:5],'eff',[round(x,2) for x in d['eff'][:5]],'cmd',c,'verdict',d.get('verdict'))
