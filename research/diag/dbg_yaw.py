exec(open('reach2.py').read().split("rng=np.random.default_rng(1)")[0])
rng=np.random.default_rng(3)
for nm,x,y,tray,side in (('Utana_5_Porcelain_Ramekin_Large',.4,.4,'tray_right','l'),('Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box',.4,-.4,'tray_left','r')):
  cat=PRI[nm]['category'];cl=cloud(nm,x,y,.3,rng);f,C=candidates(cl,plane,cat,[],W_MAX=W);A=arms[side];tx,ty=TR.NOMINAL[tray]
  for g in C:
    ap=np.r_[g['approach'],0.] if g['kind']=='side' else np.array([0,0,-1.])
    p=A.plan_pick(np.array([*g['center'],zt(*g['center'])+g['z']]),np.r_[g['close_axis'],0.],ap,g['width'],point='pinch' if g['kind']=='rim' else 'grip')
    if p:break
  print(nm[:10],'pick',p is not None)
  for th in np.arange(-2.4,2.41,.2):
    R=NODE.rotz(-th);A.set_scene((R@plane[0],plane[1]),[tuple((R@np.r_[a,b,0.])[:2]) for a,b in TR.NOMINAL.values()])
    t2=(R@np.r_[tx,ty,0.])[:2];r=NODE.plan_place(A,p['q_lift'],t2[0],t2[1],zt(tx,ty))
    print(' th %.1f'%th,np.round(t2,2),'ok' if r else '-')
  A.set_scene(plane,list(TR.NOMINAL.values()))
