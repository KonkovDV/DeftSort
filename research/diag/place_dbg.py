import sys,numpy as np
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution');sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution/test');import test_node_stub
from team048_solution import arm as ARM, node as N
TOP=ARM.TABLE_TOP
for side,(tx,ty) in (('l',(.34,.69)),('r',(.34,-.69)),('l',(.53,0)),('r',(.53,0))):
  A=ARM.Arm(side)
  for dz in (.2,.25,.3):
    c=np.array([tx,ty,TOP+N.TRAY_H+dz-.1])
    for ca in (np.array([1.,0,0]),np.array([0,1.,0]),np.array([-1.,0,0]),np.array([0,-1.,0])):
      T=A.grasp_pose(c,ca,np.array([0,0,-1.]))
      q=A.solve(T,A.home)
      if q is None:
        # why: try ik without collision
        try:
          qq=A.chain.ik(T,A.home,max_iter=300);print(side,tx,ty,dz,ca,'IK ok but collision',A.col.valid(qq))
        except ValueError as e:print(side,tx,ty,dz,ca,'IK fail',str(e)[:80])
        continue
      print(side,tx,ty,dz,ca,'solve ok','cert_home',A.col.certify(q,A.home),'lift',A.lift(q,.08) is not None)
