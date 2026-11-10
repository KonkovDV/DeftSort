import sys,glob,json,os,numpy as np
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution')
from team048_solution import arm as ARM
arms={k:ARM.Arm(k,table=False) for k in 'lr'}
bad=0;n=0
for f in sorted(glob.glob('/workspace/fixes/kaggle-sim/out_v7/trace/*/*.npz')):
    side=os.path.basename(f).split('_')[2];z=np.load(f);A=arms[side];q=np.clip(z['q'],A.chain.lower,A.chain.upper)
    lim=np.any((z['q']<A.chain.lower-1e-3)|(z['q']>A.chain.upper+1e-3))
    v=A.col.valid(q);n+=1;bad+=not v
    if not v or lim:print(os.path.basename(f)[:40],'valid vs body',v,'outside limits',lim)
print('frames',n,'body-collision (model)',bad)
