import sys,numpy as np
sys.path.insert(0,'/workspace/solution/nssr-sol-team-048/src/team048_solution')
from team048_solution import arm as A
def tips(side,cf,ct,t1=1.246165):
    p='left' if side=='l' else 'right';q={f'{p}_thumb_1_joint':t1,f'{p}_thumb_2_joint':ct*.48,f'{p}_thumb_3_joint':ct*.48*1.1425,f'{p}_thumb_4_joint':ct*.48*1.1425*.7508}
    for f in ('index','middle'):q[f'{p}_{f}_1_joint']=cf*1.333;q[f'{p}_{f}_2_joint']=cf*1.333*1.1169
    return A.fingertip(side,'thumb',q),(A.fingertip(side,'index',q)+A.fingertip(side,'middle',q))/2
for cf,ct in ((0,0),(.3,.3),(.5,.5),(.7,.6),(.9,.8),(1,1)):
    t,f=tips('l',cf,ct);print('cf',cf,'ct',ct,'thumb',t.round(3),'fing',f.round(3),'gap',round(np.linalg.norm(t-f),3),'mid',((t+f)/2).round(3))
# minimum gap over closure grid
best=min(((np.linalg.norm(np.subtract(*tips('l',a,b))),a,b) for a in np.linspace(0,1,21) for b in np.linspace(0,1,21)))
t,f=tips('l',best[1],best[2]);print('min gap',round(best[0],3),'at cf',best[1],'ct',best[2],'pinch point',((t+f)/2).round(3))
