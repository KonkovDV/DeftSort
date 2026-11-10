import arm,numpy as np,json,time,collections
A=arm.Arm('l');t0=time.time();d=collections.defaultdict(int);T=[]
for x in (.3,.42):
    for y in np.arange(-.12,.53,.08):
        t=time.time();p=A.plan_pick(np.array([x,y,arm.TABLE_TOP+.04]),np.array([0,1.,0]),np.array([1.,0,0]),.05);T.append(time.time()-t)
        d[round(y,2)]+=p is not None
print(dict(d),'mean plan s',np.mean(T))
