"""Offline reachability of pool-3 objects with OUR planner (synthetic full-surface clouds from priors dims; best case for perception)."""
import sys,json,time,numpy as np,collections
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution')
from team048_solution import arm as ARM
from team048_solution.grasp import candidates
sys.path.insert(0,"/workspace/solution/nssr-sol-team-048/src/team048_solution/test");import test_node_stub
from team048_solution import node as NODE
PRI=json.load(open('/workspace/solution/nssr-sol-team-048/src/team048_solution/team048_solution/data/priors.json'))
TOP=ARM.TABLE_TOP;plane=(np.array([0,0,1.]),-TOP)
def cloud(name,x,y,yaw,rng):
    p=PRI[name];h=p['height'];a,b=p['w_min'],p['w_max']
    if p['category']=='box':
        u=rng.uniform(-.5,.5,(3000,3))*[b,a,0];face=rng.integers(0,5,3000)
        u[:,2]=rng.uniform(0,h,3000)
        u[face==0,2]=h; u[face==1,0]=b/2; u[face==2,0]=-b/2; u[face==3,1]=a/2; u[face==4,1]=-a/2
    else:
        r=b/2;t=rng.uniform(0,2*np.pi,3000);z=rng.uniform(0,h,3000);u=np.c_[r*np.cos(t),r*np.sin(t),z]
        top=rng.random(3000)<.3;rr=r*np.sqrt(rng.random(3000));u[top]=np.c_[rr*np.cos(t),rr*np.sin(t),np.full(3000,h)][top]
    c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return u@R.T+[x,y,TOP]
TR={'dish':'tray_left','box':'tray_right','jar':'tray_center'}
TXY={'tray_left':(.34,.69),'tray_right':(.34,-.69),'tray_center':(.53,0.)}
arms={k:ARM.Arm(k) for k in 'lr'};W=min(arms['l'].hand['opposed_open']-.01,.11);print('W_MAX',W,'opposed_open',arms['l'].hand['opposed_open'])
rng=np.random.default_rng(1);res=collections.defaultdict(collections.Counter)
names=[n for n in PRI if n in sys.argv[1:]] if len(sys.argv)>1 else ['Utana_5_Porcelain_Ramekin_Large','Threshold_Ramekin_White_Porcelain','Ecoforms_Plant_Container_B4_Har','Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box','Crayola_Bonus_64_Crayons','Perricone_MD_AcylGlutathione_Eye_Lid_Serum','006_mustard_bottle','Saccharomyces_Boulardii_MOS_Value_Size','Hyaluronic_Acid']
for n in names:
  cat=PRI[n]['category'];tray=TR[cat];tx,ty=TXY[tray]
  for x in (.30,.40,.48):
    for y in (-.4,-.15,.15,.4):
      yaw=rng.uniform(0,np.pi);cl=cloud(n,x,y,yaw,rng);t=time.time()
      f,C=candidates(cl,plane,cat,[],W_MAX=W)
      if not C:res[n]['no_candidate']+=1;continue
      out='no_ik'
      order='lr' if ty>.1 else ('rl' if ty<-.1 else ('lr' if y>=0 else 'rl'))
      for side in order:
        A=arms[side]
        for g in C:
          c3=np.array([g['center'][0],g['center'][1],TOP+g['z']])
          ap=np.r_[g['approach'],0.] if g['kind']=='side' else np.array([0,0,-1.])
          p=A.plan_pick(c3,np.r_[g['close_axis'],0.],ap,g['width'])
          if p is None:continue
          out='no_place'
          if NODE.plan_place(A,p['q_lift'],tx,ty,TOP) is None:continue
          out='ok_'+side;break
        if out.startswith('ok'):break
      res[n][out]+=1;print(n[:22],x,y,round(yaw,2),[ (g['kind'],round(g['width'],3)) for g in C],out,round(time.time()-t,1),flush=True)
for n in res:print(n[:30],dict(res[n]))
