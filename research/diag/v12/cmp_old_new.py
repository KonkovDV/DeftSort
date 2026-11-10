import sys,json,subprocess,numpy as np
code=r'''
import sys,json,numpy as np
root=sys.argv[1];sys.path.insert(0,root+'/src/team048_solution');sys.path.insert(0,root+'/src/team048_solution/test')
import test_node_stub
from team048_solution import arm as ARM, grasp as GR, perception as PE, node as N
import scene_fixture as SF
z=np.load('/workspace/DeftSort/src/team048_solution/test/fixtures/v7_s102_start_frames.npz');K=z['K'].reshape(3,3)
T=ARM.fk_link('camera_head_link',{'head_pitch_joint':float(z['head_pitch'])});P=PE.deproject(z['head_depth'],K,T);v=z['head_depth']>0
reg=PE.table_region(P,v);pl=PE.table_plane(P,reg)
dets=PE.segment(P,z['head_rgb'],reg,pl,(np.array([.18,-.62,-1.]),np.array([.62,.62,1.])),min_pixels=40)
A={s:ARM.Arm(s) for s in 'lr'}
for s in 'lr':A[s].set_scene(pl,[(0.348,0.69),(0.357,-0.709)])
out=[]
for i,s_ in enumerate(dets):
  cl=P[s_.pixels[:,0],s_.pixels[:,1]]
  for cat in ('box','jar','dish'):
    try:f,C=GR.candidates(cl,pl,cat)
    except Exception as e:out.append([i,cat,'err']);continue
    rec=[i,cat,[[g['kind'],np.round(g['center'],5).tolist(),round(g['score'],6)] for g in C]]
    # plan first feasible side/top pick + place with the node functions
    for g in C:
      if not g['feasible'] or g['kind']=='rim':continue
      side='l' if g['center'][1]>=0 else 'r';Aa=A[side];top=-(pl[1]+pl[0][:2]@g['center'])/pl[0][2]
      if g['kind']=='side':ap=np.r_[g['approach'],0.];ca=np.r_[g['close_axis'],0.];pt='grip'
      else:ap=np.array([0,0,-1.]);ca=np.asarray(g['close_axis'],float);pt='pinch'
      c3=np.array([g['center'][0],g['center'][1],top+g['z']]);p=Aa.plan_pick(c3,ca,ap,g['width'],point=pt,obj=dict(g['box'],z0=top))
      if p is None:continue
      kw=dict(obj_off=N.held_offset(Aa,p['q_grasp'],g,c3)) if hasattr(N,'held_offset') else {}
      tx,ty=(0.348,0.69) if side=='l' else (0.357,-0.709)
      yp=N.plan_place_yaw(Aa,p['q_lift'],tx,ty,pl,[(0.348,0.69),(0.357,-0.709)],**kw)
      rec.append([g['kind'],np.round(p['q_grasp'],6).tolist(),None if yp is None else [round(yp[0],6),np.round(yp[1][0],6).tolist()]]);break
    out.append(rec)
print(json.dumps(out))
'''
r={}
for k,root in (('old','/tmp/v12_main'),('new','/workspace/DeftSort')):
    r[k]=json.loads(subprocess.run([sys.executable,'-c',code,root],capture_output=True,text=True).stdout.strip().split('\n')[-1])
same=sum(a==b for a,b in zip(r['old'],r['new']));print('records',len(r['old']),'identical',same)
for a,b in zip(r['old'],r['new']):
    if a!=b:print('DIFF',a[:2])
print('plans compared',sum(1 for a in r['new'] if len(a)>3))
