# Pool-3/6 perception test set: organiser object USDs on a table, RGB + depth(mm) + instance masks, 640x360 HFOV 86.
import json,math,os,sys,time,random
from pathlib import Path
OUT=Path(os.environ['NEUTRAL_OUT']);OUT.mkdir(parents=True,exist_ok=True)
A=Path(os.environ['ASSETS']);rep_log=OUT/'render_report.json';R={'frames':0,'errors':[]}
def save():rep_log.write_text(json.dumps(R,indent=1))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False,'fast_shutdown':True,'width':640,'height':360},experience=sys.argv[1])
import numpy as np,yaml,omni.usd,omni.replicator.core as rep
from pxr import Usd,UsdGeom,UsdLux,Gf,Sdf
from PIL import Image
pools=yaml.safe_load(open(A/'pools.yaml'))['pools']
dims={o['name']:o['dims_cm'] for c in yaml.safe_load(open(A/'objects.yaml'))['categories'].values() for o in c['objects']}
def obj_fix(name):
 st=Usd.Stage.Open(str(A/'objects'/name/f'{name}.usd'));return UsdGeom.GetStageMetersPerUnit(st),UsdGeom.GetStageUpAxis(st)
FIX={}
TABLE_Z=0.75
HARD=os.environ.get('HARD')=='1';YS=.30 if HARD else .52;GAP=.80 if HARD else 1.0
def build(pool,seed):
 rng=random.Random(seed);ctx=omni.usd.get_context();ctx.new_stage();[app.update() for _ in range(2)]
 st=ctx.get_stage();UsdGeom.SetStageUpAxis(st,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(st,1.0)
 UsdGeom.Xform.Define(st,'/World')
 t=UsdGeom.Cube.Define(st,'/World/Table');t.CreateSizeAttr(1.0);x=UsdGeom.XformCommonAPI(t);x.SetScale(Gf.Vec3f(0.8,1.4,0.04));x.SetTranslate(Gf.Vec3d(0.4,0,TABLE_Z-0.02))
 g=rng.uniform(.35,.8);t.CreateDisplayColorAttr([Gf.Vec3f(g*rng.uniform(.9,1.1),g,g*rng.uniform(.85,1.05))])
 f=UsdGeom.Cube.Define(st,'/World/Floor');f.CreateSizeAttr(1.0);UsdGeom.XformCommonAPI(f).SetScale(Gf.Vec3f(8,8,.01));f.CreateDisplayColorAttr([Gf.Vec3f(.3,.3,.32)])
 d=UsdLux.DomeLight.Define(st,'/World/Dome');d.CreateIntensityAttr(rng.uniform(200,1500) if HARD else rng.uniform(500,1500))
 s=UsdLux.DistantLight.Define(st,'/World/Sun');s.CreateIntensityAttr(rng.uniform(1000,3500));UsdGeom.XformCommonAPI(s).SetRotate(Gf.Vec3f(rng.uniform(-40,40),rng.uniform(-40,40),0))
 cats=pools[str(pool)];names=[n for c in cats.values() for n in c];placed=[];labels={}
 for i,n in enumerate(names):
  r=0.5*max(dims[n][0],dims[n][1])/100+0.01
  for _ in range(200):
   px,py=rng.uniform(.26,.50),rng.uniform(-YS,YS)
   if all((px-a)**2+(py-b)**2>(GAP*(r+c))**2 for a,b,c in placed):break
  placed.append((px,py,r))
  if n not in FIX:FIX[n]=obj_fix(n)
  mpu,up=FIX[n];p=f'/World/obj_{i}';xf=UsdGeom.Xform.Define(st,p);yaw=rng.uniform(-180,180)
  xf.AddTranslateOp().Set(Gf.Vec3d(px,py,TABLE_Z));xf.AddRotateZOp().Set(yaw)
  h=UsdGeom.Xform.Define(st,p+'/geom')
  if up==UsdGeom.Tokens.y:h.AddRotateXOp().Set(90.0)
  if abs(mpu-1)>1e-6:h.AddScaleOp().Set(Gf.Vec3f(mpu,mpu,mpu))
  h.GetPrim().GetReferences().AddReference(str(A/'objects'/n/f'{n}.usd'))
  labels[p]={'name':n,'category':[c for c,v in cats.items() if n in v][0],'xy_yaw':[px,py,yaw]}
 [app.update() for _ in range(3)]
 # drop to table: shift each object so its bbox min z = table top
 bc=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
 for p in labels:
  pr=st.GetPrimAtPath(p);lo=bc.ComputeWorldBound(pr).ComputeAlignedRange().GetMin()
  op=UsdGeom.Xformable(pr).GetOrderedXformOps()[0];v=op.Get();op.Set(Gf.Vec3d(v[0],v[1],v[2]+(TABLE_Z-lo[2])))
 return st,labels,rng
cam_path='/World/Cam'
def camera(st,rng):
 c=UsdGeom.Camera.Define(st,cam_path);c.CreateClippingRangeAttr(Gf.Vec2f(.05,50));c.CreateHorizontalApertureAttr(20.955);c.CreateVerticalApertureAttr(20.955*360/640);c.CreateFocalLengthAttr(20.955/(2*math.tan(math.radians(43))))
 # approx. head camera: behind table edge, 0.45-0.75 m above table, looking at the spawn area
 eye=Gf.Vec3d(rng.uniform(-.10,.12),rng.uniform(-.25,.25),TABLE_Z+(rng.uniform(.25,.75) if HARD else rng.uniform(.45,.75)));tgt=Gf.Vec3d(rng.uniform(.3,.45),rng.uniform(-.15,.15),TABLE_Z)
 m=Gf.Matrix4d().SetLookAt(eye,tgt,Gf.Vec3d(0,0,1)).GetInverse();ops=UsdGeom.Xformable(c);ops.ClearXformOpOrder();ops.AddTransformOp().Set(m)
 return {'eye':list(eye),'target':list(tgt)}
NL=int(os.environ.get('N_LAYOUTS','40'));VIEWS=3 if HARD else 2;t0=time.monotonic()
for pool in [3,6]:
 for k in range(NL):
  seed=pool*100000+k+(50000 if HARD else 0)
  try:
   st,labels,rng=build(pool,seed)
   for v in range(VIEWS):
    pose=camera(st,rng);prod=rep.create.render_product(cam_path,(640,360))
    a_rgb=rep.AnnotatorRegistry.get_annotator('rgb');a_d=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane');a_i=rep.AnnotatorRegistry.get_annotator('instance_id_segmentation_fast')
    for a in (a_rgb,a_d,a_i):a.attach([prod])
    for _ in range(3):rep.orchestrator.step(rt_subframes=4)
    rgb=np.asarray(a_rgb.get_data())[...,:3];z=np.asarray(a_d.get_data()).squeeze();inst=a_i.get_data()
    ids=np.asarray(inst['data']).squeeze();id2=inst['info']['idToLabels']
    mask=np.zeros(ids.shape,np.uint8);objs=[]
    for j,(p,lab) in enumerate(labels.items(),1):
     sel=np.zeros(ids.shape,bool)
     for iid,path in id2.items():
      if str(path).startswith(p+'/') or str(path)==p:sel|=(ids==int(iid))
     mask[sel]=j;objs.append({**lab,'mask_id':j,'pixels':int(sel.sum())})
    good=np.isfinite(z)&(z>=.15)&(z<=6);dmm=np.zeros(z.shape,np.uint16);dmm[good]=np.rint(z[good]*1000)
    stem=f'p{pool}_s{seed}_v{v}';np.savez_compressed(OUT/f'{stem}.npz',rgb=rgb.astype(np.uint8),depth_mm=dmm,mask=mask)
    Image.fromarray(rgb.astype(np.uint8)).save(OUT/f'{stem}.jpg',quality=90)
    (OUT/f'{stem}.json').write_text(json.dumps({'pool':pool,'seed':seed,'camera':pose,'intrinsics':{'w':640,'h':360,'fx':320/math.tan(math.radians(43)),'cx':320,'cy':180},'objects':objs}))
    for a in (a_rgb,a_d,a_i):a.detach([prod.path])
    prod.destroy();R['frames']+=1
   R['elapsed']=time.monotonic()-t0;save();print('LAYOUT',pool,k,R['frames'],flush=True)
  except Exception as e:
   import traceback;R['errors'].append({'pool':pool,'k':k,'e':str(e),'tb':traceback.format_exc()[-1500:]});save()
   if len(R['errors'])>5:raise
R['done']=True;save();print('RENDER_DONE',R['frames'],flush=True)
app.close()
