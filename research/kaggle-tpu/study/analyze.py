import json,glob,collections,sys,numpy as np
B='/workspace/fixes/kaggle-tpu/study/out'
def rows(globs):
    R=[]
    for g in globs:
        for f in glob.glob(f'{B}/{g}/results.jsonl'):R+=[json.loads(l) for l in open(f)]
    return R
def reach(tag,globs):
    R=rows(globs);L=len(set(r['layout'] for r in R))
    if not R:print('\n== REACH',tag,'no rows yet');return
    print(f'\n== REACH {tag}: {len(R)} objects, {L} layouts')
    for cat in sorted(set(r['cat'] for r in R)):
        X=[r for r in R if r['cat']==cat];n=len(X);c=collections.Counter(r['result'] for r in X)
        ok=[r for r in X if r['result']=='ok'];y=sum(1 for r in ok if r['yaw'])
        print(f"  {cat:12s} tray={X[0]['tray']:11s} n={n:4d} ok={c['ok']/n:.3f} picked={sum(r['picked'] for r in X)/n:.3f} "
              f"| no_ik={c['no_ik']/n:.3f} no_place={c['no_place']/n:.3f} no_cand={c['no_candidate']/n:.3f} no_clear={c['no_clearance']/n:.3f} "
              f"| yaw={y/max(len(ok),1):.2f} pick_dur={np.mean([r['pick_dur'] for r in ok]) if ok else 0:.1f}s plan={np.mean([r['plan_s'] for r in X]):.1f}s")
        per=collections.defaultdict(lambda:[0,0])
        for r in X:per[r['name']][0]+=1;per[r['name']][1]+=r['result']=='ok'
        print('     per object:',{k[:24]:round(v[1]/v[0],2) for k,v in per.items()})
    # by position (|y| bins) for all
    ok=np.array([r['result']=='ok' for r in R]);x=np.array([r['pose'][0] for r in R]);y=np.array([r['pose'][1] for r in R])
    for lo,hi in ((0,.15),(.15,.3),(.3,.45),(.45,.6)):
        m=(np.abs(y)>=lo)&(np.abs(y)<hi);print(f'   |y| {lo:.2f}-{hi:.2f}: n={m.sum():4d} ok={ok[m].mean() if m.any() else 0:.3f}')
    for lo,hi in ((.25,.33),(.33,.41),(.41,.51)):
        m=(x>=lo)&(x<hi);print(f'   x {lo:.2f}-{hi:.2f}: n={m.sum():4d} ok={ok[m].mean() if m.any() else 0:.3f}')
    per_layout=collections.defaultdict(list)
    for r in R:per_layout[r['layout']].append(r['result']=='ok')
    v=np.array([sum(a) for a in per_layout.values()]);print(f'   feasible objects per layout: mean {v.mean():.2f} min {v.min()} max {v.max()} (upper bound score {10*v.mean():.0f})')
def grasp(tag,globs):
    R=rows(globs)
    if not R:print('\n== GRASP',tag,'no rows');return
    print(f'\n== GRASP {tag}: {len(R)} rows')
    for cat in sorted(set(r['cat'] for r in R)):
        X=[r for r in R if r['cat']==cat];keys=sorted(set((r['layout'],r['name']) for r in X));V=sorted(set(r['variant'] for r in X))
        tab={(r['layout'],r['name'],r['variant']):r for r in X}
        base={k:tab.get((*k,'base'),{}).get('result')=='ok' for k in keys}
        b=np.mean(list(base.values()));print(f'  {cat}: {len(keys)} object-instances, base ok={b:.3f}')
        uni={k:base[k] for k in keys}
        res=[]
        for v in V:
            o=[tab.get((*k,v),{}).get('result')=='ok' for k in keys if (*k,v) in tab]
            gain=[(tab.get((*k,v),{}).get('result')=='ok') and not base[k] for k in keys if (*k,v) in tab]
            loss=[(tab.get((*k,v),{}).get('result')!='ok') and base[k] for k in keys if (*k,v) in tab]
            pk=[tab[(*k,v)].get('pick_dur') for k in keys if (*k,v) in tab and tab[(*k,v)].get('pick_dur')]
            res.append((v,len(o),np.mean(o),np.sum(gain),np.sum(loss),np.mean(pk) if pk else 0))
            for k in keys:
                if tab.get((*k,v),{}).get('result')=='ok':uni[k]=True
        for v,n,m,g,l,pk in sorted(res,key=lambda t:-t[2]):print(f'    {v:18s} n={n:3d} ok={m:.3f} (+{g} new, -{l} lost vs base) pick_dur={pk:.1f}s')
        print(f'    UNION of all variants ok={np.mean(list(uni.values())):.3f}  (base {b:.3f})')
        per=collections.defaultdict(lambda:[0,0,0])
        for k in keys:per[k[1]][0]+=1;per[k[1]][1]+=base[k];per[k[1]][2]+=uni[k]
        print('    per object base->union:',{n[:22]:f'{a[1]/a[0]:.2f}->{a[2]/a[0]:.2f}' for n,a in per.items()})
def tubes():
    R=rows(['local-tubes','tubes'])
    if not R:return
    print('\n== TUBES')
    for kind in ('table_slot','hood_slot'):
        for rk in sorted(set(r['rack'] for r in R if r['kind']==kind)):
            X=[r for r in R if r['kind']==kind and r['rack']==rk]
            l=sum(1 for r in X if r.get('l'));rr=sum(1 for r in X if r.get('r'));any_=sum(1 for r in X if r.get('l') or r.get('r'))
            modes=collections.Counter((r.get('l') or {}).get('mode') for r in X if kind=='table_slot')
            yaws=collections.Counter((r.get('l') or {}).get('yaw') for r in X if kind=='hood_slot')
            print(f'  {kind:10s} {rk:14s} n={len(X)} left={l} right={rr} any={any_} {dict(modes) if kind=="table_slot" else dict(yaws)}')
if __name__=='__main__':
    reach('pool3 sort3',['local-r3s3','r3s3-*']);reach('pool3 sort2',['local-r3s2','r3s2-*']);reach('pool6 sort2',['local-r6s2','r6s2-*'])
    grasp('pool3 dish',['local-g3d','local-g3dx','g3d-*']);grasp('pool3 box',['local-g3b','local-g3bx','g3b-*']);grasp('pool6 electronics+box',['local-g6','local-g6x','g6-*']);tubes()
