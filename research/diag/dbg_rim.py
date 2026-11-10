exec(open('reach2.py').read().split("rng=np.random.default_rng(1)")[0])
rng=np.random.default_rng(5);nm='Threshold_Ramekin_White_Porcelain'
for (x,y) in ((.3,.15),(.4,.15),(.4,.4)):
  cl=cloud(nm,x,y,0.3,rng);f,C=candidates(cl,plane,'dish',[],W_MAX=W);g=C[0];top=zt(*g['center'])
  c3=np.array([*g['center'],top+g['z']]);A=arms['l']
  for lab,setup in (('full',lambda:A.set_scene(plane,list(TR.NOMINAL.values()))),('no_trays',lambda:A.set_scene(plane,[])),('flat',lambda:setattr(A.col,'environment',A.body+ARM.table_env()))):
    setup()
    for pt in ('pinch','grip'):
      p=A.plan_pick(c3,np.r_[g['close_axis'],0.],np.array([0,0,-1.]),g['width'],point=pt)
      print(x,y,lab,pt,'ok' if p else None)
  A.set_scene(plane,list(TR.NOMINAL.values()))
