import json,sys
O=json.load(open('/workspace/fixes/kaggle-tpu/cpu-shards/reach_p3/objects.json'))
P=json.load(open('/workspace/fixes/kaggle-tpu/cpu-shards/reach_p3/pools.json'))
O=O.get('categories',O);P=P.get('pools',P)
meta={o['name']:[c,[v/100 for v in o['dims_cm']]] for c,m in O.items() if isinstance(m,dict) for o in m['objects']}
def cfg(mode,pool,task,l0,l1,**k):
    pl=P[str(pool)];cats=list(pl)[:(2 if task=='sort2' else 3)]
    d=dict(mode=mode,task=task,pool={c:pl[c] for c in cats},cats=cats,l0=l0,l1=l1,meta={n:meta[n] for c in cats for n in pl[c]},seed_base=70000+1000*pool+(0 if task=='sort3' else 500))
    d.update(k);return d
def tubes_jobs():
    racks={'r50':dict(rows=2,cols=4,pitch=.046),'r15':dict(rows=2,cols=5,pitch=.030)}
    tubes={'falcon50':dict(d=.0292,length=.1144),'falcon15':dict(d=.0170,length=.1200)}
    def slots(rk,at,along,rows=None):
        r=racks[rk];rows=rows or r['rows'];xs=[(i-(r['cols']-1)/2)*r['pitch'] for i in range(r['cols'])];ys=[(j-(rows-1)/2)*r['pitch'] for j in range(rows)]
        S=[(x,y) for y in ys for x in xs]
        if along=='y':S=[(-sy,sx) for sx,sy in S]
        return [(at[0]+a,at[1]+b) for a,b in S]
    J=[]
    # standard: r50 table (.36,.13) -> hood (-.02,.60); r15 table (.36,-.14) -> hood (-.32,.60); easy/color use r50 at (.36,0)
    J.append(dict(rack='r50@.36,.13',slots=slots('r50',(.36,.13),'y'),hood_rack='r50@-.02,.60',hood_slots=slots('r50',(-.02,.60),'x',1),tube=tubes['falcon50'],tube_name='falcon50'))
    J.append(dict(rack='r15@.36,-.14',slots=slots('r15',(.36,-.14),'y'),hood_rack='r15@-.32,.60',hood_slots=slots('r15',(-.32,.60),'x',1),tube=tubes['falcon15'],tube_name='falcon15'))
    J.append(dict(rack='r50@.36,0',slots=slots('r50',(.36,0.),'y'),hood_rack='r50@-.30,.60',hood_slots=slots('r50',(-.30,.60),'x',1)+slots('r50',(-.08,.60),'x',1),tube=tubes['falcon50'],tube_name='falcon50'))
    return J
