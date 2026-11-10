import arm,numpy as np,json,time
R={};t0=time.time()
for side in 'lr':
    A=arm.Arm(side);s=1 if side=='l' else -1;res=[]
    for x in np.arange(.26,.501,.06):
        for y in np.arange(-.52,.521,.08):
            for z in (.03,.06):
                c=np.array([x,y,arm.TABLE_TOP+z])
                p=A.plan_pick(c,np.array([0,1.,0]),np.array([1.,0,0]),.05)   # side grasp, approach +x, close along y
                p2=None if p else A.plan_pick(c,np.array([1.,0,0]),np.array([0,-s*1.,0]) if False else np.array([0,-1.,0])*s*-1,.05)
                q=p or p2;res.append(dict(x=float(x),y=float(y),z=z,ok=q is not None,dur=q['duration'] if q else None,vmax=q['max_speed'] if q else None))
    R[side]=res;ok=[r for r in res if r['ok']]
    print(side,'reachable',len(ok),'/',len(res),'max joint speed',max([r['vmax'] for r in ok] or [0]),'median dur',np.median([r['dur'] for r in ok]) if ok else None,round(time.time()-t0),'s',flush=True)
json.dump(R,open('reach.json','w'))
