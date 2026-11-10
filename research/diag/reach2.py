"""Offline reachability with the v8 planner: measured v7 table plane (tilted 4 deg in pelvis), tray walls, thumb spheres,
side-grasp height escalation, rim pinch point. Synthetic full-surface clouds from priors (best-case perception)."""
import sys,json,time,numpy as np,collections
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution');sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution/test');import test_node_stub
from team048_solution import arm as ARM, node as NODE, trays as TR
from team048_solution.grasp import candidates
PRI=NODE.PRI;n=np.array([-.0699,0,.9976]);d=.1788;plane=(n/np.linalg.norm(n),d/np.linalg.norm(n))
zt=lambda x,y:-(plane[1]+plane[0][0]*x+plane[0][1]*y)/plane[0][2]
def cloud(name,x,y,yaw,rng):
    p=PRI[name];h=p['height'];a,b=p['w_min'],p['w_max'];N=3000
    if p['category']=='box':
        u=rng.uniform(-.5,.5,(N,3))*[b,a,0];face=rng.integers(0,5,N);u[:,2]=rng.uniform(0,h,N)
        u[face==0,2]=h;u[face==1,0]=b/2;u[face==2,0]=-b/2;u[face==3,1]=a/2;u[face==4,1]=-a/2
    else:
        t=rng.uniform(0,2*np.pi,N);z=rng.uniform(0,h,N);u=np.c_[b/2*np.cos(t),a/2*np.sin(t),z]
        top=rng.random(N)<.3;rr=np.sqrt(rng.random(N));u[top]=np.c_[b/2*rr*np.cos(t),a/2*rr*np.sin(t),np.full(N,h)][top]
    c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1]]);P=u@R.T;P[:,0]+=x;P[:,1]+=y;P[:,2]+=zt(x,y);return P
TRY={'dish':'tray_left','box':'tray_right','jar':'tray_center'}
arms={k:ARM.Arm(k) for k in 'lr'}
for A in arms.values():A.set_scene(plane,list(TR.NOMINAL.values()))
W=min(arms['l'].hand['opposed_open']-.01,.11)
rng=np.random.default_rng(1);res=collections.defaultdict(collections.Counter)
names=['Utana_5_Porcelain_Ramekin_Large','Threshold_Ramekin_White_Porcelain','Ecoforms_Plant_Container_B4_Har','Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box','Crayola_Bonus_64_Crayons','Perricone_MD_AcylGlutathione_Eye_Lid_Serum','006_mustard_bottle','Saccharomyces_Boulardii_MOS_Value_Size','Hyaluronic_Acid']
for nm in names:
  cat=PRI[nm]['category'];tx,ty=TR.NOMINAL[TRY[cat]]
  for x in (.30,.40,.48):
    for y in (-.4,-.15,.15,.4):
      yaw=rng.uniform(0,np.pi);cl=cloud(nm,x,y,yaw,rng);t=time.time()
      f,C=candidates(cl,plane,cat,[],W_MAX=W)
      if not C:res[nm]['no_candidate']+=1;continue
      out='no_ik';order='lr' if ty>.1 else ('rl' if ty<-.1 else ('lr' if y>=0 else 'rl'))
      for side in order:
        A=arms[side]
        for g in C:
          top=zt(*g['center']);ap=np.r_[g['approach'],0.] if g['kind']=='side' else np.array([0,0,-1.]);p=None
          for dz in (NODE.SIDE_DZ if g['kind']=='side' else (0.,)):
            if g['kind']=='side' and dz>0 and g['z']+dz>f['height']-.015:break
            p=A.plan_pick(np.array([*g['center'],top+g['z']+dz]),np.r_[g['close_axis'],0.],ap,g['width'],point='pinch' if g['kind']=='rim' else 'grip')
            if p is not None:break
          if p is None:continue
          out='no_place'
          pl=NODE.plan_place(A,p['q_lift'],tx,ty,top)
          if pl is None:continue
          out='ok_'+side+('_carry' if pl[2] is not None else '')+'_dz%.2f'%dz;break
        if out.startswith('ok'):break
      res[nm][out.split('_dz')[0]]+=1;print(nm[:22],x,y,out,round(time.time()-t,1),flush=True)
for k in res:print(k[:30],dict(res[k]))
