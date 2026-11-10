def run_ros_exchange(app,tl,color,z,cam,stage,r,save):
 import os,sys,json,time,subprocess,hashlib,traceback,math
 from pathlib import Path
 import numpy as np
 OUT=Path('/kaggle/working');result={'bridge_enabled':False,'independent_dds_exchange':False};child=None;node=None;ctx=None;log=None;executor=None
 try:
  import omni.kit.app,omni.graph.core as og,rclpy
  from rclpy.context import Context
  from rclpy.executors import SingleThreadedExecutor
  from rclpy.parameter import Parameter
  from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy,ReliabilityPolicy
  from sensor_msgs.msg import Image,CameraInfo
  from tf2_msgs.msg import TFMessage
  from geometry_msgs.msg import TransformStamped
  from std_msgs.msg import Int32
  from builtin_interfaces.msg import Time
  from pxr import Gf,UsdGeom
  em=omni.kit.app.get_app().get_extension_manager();result['bridge_enabled']=em.is_extension_enabled('isaacsim.ros2.bridge')
  if not result['bridge_enabled']:raise RuntimeError('Official ROS2 bridge extension is not enabled')
  og.Controller.edit({'graph_path':'/NeutralClock','evaluator_name':'execution'},{og.Controller.Keys.CREATE_NODES:[('Tick','omni.graph.action.OnPlaybackTick'),('Time','isaacsim.core.nodes.IsaacReadSimulationTime'),('Clock','isaacsim.ros2.bridge.ROS2PublishClock'),('Context','isaacsim.ros2.bridge.ROS2Context')],og.Controller.Keys.CONNECT:[('Tick.outputs:tick','Clock.inputs:execIn'),('Time.outputs:simulationTime','Clock.inputs:timeStamp'),('Context.outputs:context','Clock.inputs:context')],og.Controller.Keys.SET_VALUES:[('Clock.inputs:topicName','/clock'),('Context.inputs:domain_id',146),('Context.inputs:useDomainIDEnvVar',True)]})
  ctx=Context();rclpy.init(context=ctx);node=rclpy.create_node('neutral_sim_publisher',context=ctx,parameter_overrides=[Parameter('use_sim_time',value=True)]);executor=SingleThreadedExecutor(context=ctx);executor.add_node(node);ack=[];acksub=node.create_subscription(Int32,'/neutral_probe/ack',lambda m:ack.append(m.data),10)
  rgbpub=node.create_publisher(Image,'/neutral_probe/rgb',qos_profile_sensor_data);depthpub=node.create_publisher(Image,'/neutral_probe/depth',qos_profile_sensor_data);infopub=node.create_publisher(CameraInfo,'/neutral_probe/camera_info',10);tfpub=node.create_publisher(TFMessage,'/neutral_probe/tf_static',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE))
  valid=np.isfinite(z)&(z>=.15)&(z<=6);d=np.zeros(z.shape,dtype='<u2');d[valid]=np.rint(z[valid]*1000).astype('<u2');rgb=np.ascontiguousarray(color[...,:3],dtype='uint8');d=np.ascontiguousarray(d)
  stamp_ns=int(tl.get_current_time()*1e9);stamp=Time(sec=stamp_ns//1000000000,nanosec=stamp_ns%1000000000);frame='neutral_camera_optical'
  def img(a,encoding,step):
   m=Image();m.header.stamp=stamp;m.header.frame_id=frame;m.height=360;m.width=640;m.encoding=encoding;m.is_bigendian=0;m.step=step;m.data=a.tobytes();return m
  rgbmsg=img(rgb,'rgb8',640*3);depthmsg=img(d,'16UC1',640*2);fx=320/math.tan(math.radians(43));ci=CameraInfo();ci.header=rgbmsg.header;ci.width=640;ci.height=360;ci.distortion_model='plumb_bob';ci.d=[0.]*5;ci.k=[fx,0.,320.,0.,fx,180.,0.,0.,1.];ci.r=[1.,0.,0.,0.,1.,0.,0.,0.,1.];ci.p=[fx,0.,320.,0.,0.,fx,180.,0.,0.,0.,1.,0.]
  optical=Gf.Matrix4d().SetScale(Gf.Vec3d(1,-1,-1))*UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0);p=optical.ExtractTranslation();q=optical.ExtractRotationQuat();v=q.GetImaginary();tr=TransformStamped();tr.header.stamp=stamp;tr.header.frame_id='neutral_world';tr.child_frame_id=frame;tr.transform.translation.x=float(p[0]);tr.transform.translation.y=float(p[1]);tr.transform.translation.z=float(p[2]);tr.transform.rotation.x=float(v[0]);tr.transform.rotation.y=float(v[1]);tr.transform.rotation.z=float(v[2]);tr.transform.rotation.w=float(q.GetReal());tf=TFMessage(transforms=[tr])
  expected={k:hashlib.sha256(bytes(m.data)).hexdigest() for k,m in [('rgb',rgbmsg),('depth',depthmsg)]};expected.update({'fx':fx,'frame_id':frame,'stamp_ns':stamp_ns});(OUT/'dds_expected.json').write_text(json.dumps(expected));log=(OUT/'ros_dds_receiver.log').open('w');child=subprocess.Popen([sys.executable,os.environ['NEUTRAL_RECEIVER']],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy());t=time.monotonic();tl.play()
  for i in range(900):
   if time.monotonic()-t>32:break
   app.update();executor.spin_once(timeout_sec=.02)
   if i%10==0:rgbpub.publish(rgbmsg);depthpub.publish(depthmsg);infopub.publish(ci);tfpub.publish(tf)
   if child.poll() is not None:break
  tl.pause()
  try:rc=child.wait(timeout=5)
  except subprocess.TimeoutExpired:child.terminate();rc=child.wait(timeout=5);result['receiver_timeout']=True
  result['receiver_exit_code']=rc;result['ack_received']=42 in ack;result['exchange_seconds']=time.monotonic()-t
  receiver_path=OUT/'ros_dds_receiver_report.json'
  if receiver_path.exists():result['receiver']=json.loads(receiver_path.read_text())
  result['independent_dds_exchange']=bool(result['ack_received'] and result.get('receiver',{}).get('received_all') and rc==0)
  result['clock_source']='IsaacReadSimulationTime -> official ROS2PublishClock OmniGraph; no manual Clock publisher';result['camera_scope']='one procedural camera; standard messages; static captured RGB/depth; not robot/pelvis TF or full two-camera system';result['optical_axes']='USD camera +X right,+Y up,-Z forward -> optical +X right,+Y down,+Z forward by 180deg X; no image flip'
 except Exception as e:result['exception']=str(e);result['traceback']=traceback.format_exc();traceback.print_exc()
 finally:
  if child is not None and child.poll() is None:
   child.terminate()
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
  if executor is not None:
   if node is not None:executor.remove_node(node)
   executor.shutdown(timeout_sec=2)
  if node is not None:node.destroy_node()
  if ctx is not None:
   try:rclpy.shutdown(context=ctx)
   except Exception:pass
  if log is not None:log.close()
  r['ros_bridge']=result;save();(OUT/'ros_bridge_report.json').write_text(json.dumps(result,indent=2))

def run_lab_context(app,r,save):
 import time,traceback,json,importlib.metadata as md
 from pathlib import Path
 result={'installed':False,'cpu_context_steps':0,'cuda_training_verified':False};sim=None;cls=None;t=time.monotonic()
 try:
  import torch,omni.usd
  result['version']=md.version('isaaclab');result['installed']=result['version']=='2.2.0'
  from isaaclab.sim import SimulationContext,SimulationCfg
  cls=SimulationContext;result['installed']=md.version('isaaclab')=='2.2.0';result['version']=md.version('isaaclab');result['torch_version']=torch.__version__;result['torch_cuda_available']=torch.cuda.is_available();result['device']='cpu';result['module']=SimulationContext.__module__
  omni.usd.get_context().new_stage();app.update();sim=SimulationContext(SimulationCfg(device='cpu',dt=1/60,use_fabric=False));sim.reset()
  for _ in range(3):sim.step(render=False);result['cpu_context_steps']+=1
  result['simulation_seconds']=float(sim.current_time);result['success']=result['installed'] and result['cpu_context_steps']==3 and result['simulation_seconds']>0
 except Exception as e:result['exception']=str(e);result['traceback']=traceback.format_exc();traceback.print_exc()
 finally:
  if sim is not None:sim.clear_all_callbacks()
  if cls is not None:cls.clear_instance()
  result['seconds']=time.monotonic()-t;r['isaaclab']=result;save();Path('/kaggle/working/lab_context_report.json').write_text(json.dumps(result,indent=2))

import json,time,traceback,os,sys,resource,subprocess,math
from pathlib import Path
FAST_SHUTDOWN='--neutral-fast-shutdown' in sys.argv;sys.argv=[x for x in sys.argv if x!='--neutral-fast-shutdown']
OUT=Path('/kaggle/working');r={'startup':False,'updates':0,'render':False,'physics':False,'ros_bridge':False};app=None

started=time.monotonic()
def telemetry(stage):
 u=resource.getrusage(resource.RUSAGE_SELF);item={'stage':stage,'elapsed':time.monotonic()-started,'cpu_user_seconds':u.ru_utime,'cpu_system_seconds':u.ru_stime,'max_rss_kib':u.ru_maxrss}
 try:item['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout
 except Exception as e:item['gpu_error']=str(e)
 r.setdefault('telemetry',[]).append(item)
def save(): (OUT/'isaac_scene_report.json').write_text(json.dumps(r,indent=2))
try:
 t=time.monotonic()
 from isaacsim import SimulationApp
 if SimulationApp is None:raise RuntimeError('SimulationApp API absent after resolved installation')
 app=SimulationApp({'headless':True,'width':640,'height':360,'renderer':'RayTracedLighting','multi_gpu':False,'active_gpu':0,'physics_gpu':0,'enable_crashreporter':False,'fast_shutdown':FAST_SHUTDOWN},experience=sys.argv[1]);r['startup_seconds']=time.monotonic()-t;r['startup']=True;telemetry('startup');save()
 for _ in range(3):app.update();r['updates']+=1
 import omni.usd,omni.timeline,omni.replicator.core as rep,numpy as np
 from pxr import UsdGeom,UsdPhysics,UsdLux,Gf
 stage=omni.usd.get_context().get_stage();UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.0)
 world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
 scene=UsdPhysics.Scene.Define(stage,'/World/Physics');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
 ground=UsdGeom.Cube.Define(stage,'/World/Ground');ground.CreateSizeAttr(2);UsdGeom.XformCommonAPI(ground).SetScale(Gf.Vec3f(3,3,.05));UsdGeom.XformCommonAPI(ground).SetTranslate(Gf.Vec3d(0,0,-.05));ground.CreateDisplayColorAttr([Gf.Vec3f(.15,.3,.55)]);UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
 cube=UsdGeom.Cube.Define(stage,'/World/Cube');cube.CreateSizeAttr(.4);UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(0,0,1));cube.CreateDisplayColorAttr([Gf.Vec3f(.8,.1,.05)]);UsdPhysics.CollisionAPI.Apply(cube.GetPrim());UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim());UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.)
 light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(1200.)
 cam=UsdGeom.Camera.Define(stage,'/World/Camera');cam.CreateClippingRangeAttr(Gf.Vec2f(.1,100));cam.CreateHorizontalApertureAttr(20.955);cam.CreateVerticalApertureAttr(20.955*360/640);cam.CreateFocalLengthAttr(20.955/(2*math.tan(math.radians(43))));UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(3,3,3),Gf.Vec3d(0,0,.3),Gf.Vec3d(0,0,1)).GetInverse())
 before=UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(0).ExtractTranslation()[2]
 tl=omni.timeline.get_timeline_interface();tl.set_target_framerate(60);tl.play();t=time.monotonic()
 for _ in range(120):app.update()
 after=UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(0).ExtractTranslation()[2];r['physics']={'before_z':before,'after_z':after,'timeline_seconds':tl.get_current_time(),'moved_down':after<before-.01,'update_seconds':time.monotonic()-t};tl.pause();save()
 t=time.monotonic();product=rep.create.render_product('/World/Camera',(640,360));rgb=rep.AnnotatorRegistry.get_annotator('rgb');depth=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane');rgb.attach([product]);depth.attach([product])
 for _ in range(4):rep.orchestrator.step(rt_subframes=2)
 color=np.asarray(rgb.get_data());z=np.asarray(depth.get_data());valid=z[np.isfinite(z)&(z>0)];r['render_seconds']=time.monotonic()-t
 r['rgb']={'shape':list(color.shape),'std':float(color[...,:3].std()) if color.size else 0};r['depth']={'shape':list(z.shape),'valid_count':int(valid.size),'min':float(valid.min()) if valid.size else None,'max':float(valid.max()) if valid.size else None};r['render']=color.shape[:2]==(360,640) and r['rgb']['std']>0 and valid.size>0
 if color.size:
  from PIL import Image
  Image.fromarray(color.astype('uint8')).save(OUT/'isaac_rgb.png')
 if valid.size:
  from PIL import Image
  nz=np.where(np.isfinite(z)&(z>0),z,0);nz=np.clip(nz/max(float(valid.max()),1e-6)*255,0,255).astype('uint8');Image.fromarray(nz.squeeze()).save(OUT/'isaac_depth_preview.png')
 good=np.isfinite(z)&(z>=.15)&(z<=6.);depth16=np.zeros(z.shape,dtype=np.uint16);depth16[good]=np.rint(z[good]*1000).astype(np.uint16);Image.fromarray(depth16.squeeze()).save(OUT/'isaac_depth_mm.png');r['depth16_mm']={'encoding':'16UC1','invalid_value':0,'range_m':[.15,6.0],'valid_pixels':int(good.sum()),'invalid_pixels':int((~good).sum()),'max_quantization_error_mm':float(np.max(np.abs(depth16[good].astype(float)-z[good]*1000))) if good.any() else None,'pixel_alignment':'same render product; no flips/transposes'};r['intrinsics']={'width':640,'height':360,'horizontal_fov_degrees':86,'fx':320/math.tan(math.radians(43)),'fy':320/math.tan(math.radians(43)),'cx':320,'cy':180};telemetry('rendered');save()
 run_ros_exchange(app,tl,color,z,cam,stage,r,save)
 rgb.detach([product.path]);depth.detach([product.path]);product.destroy();rep.orchestrator.stop();rep.orchestrator.set_capture_on_play(False);tl.stop();r['annotators_detached']=True;r['render_product_destroyed']=True;r['replicator_stopped']=True;r['asset_sources']='procedural primitives only';save()
 run_lab_context(app,r,save)
except Exception as e:
 r['exception_type']=type(e).__name__;r['exception']=str(e);r['traceback']=traceback.format_exc();save();traceback.print_exc()
finally:
 if app is not None:
  try:
   import asyncio,gc,omni.usd
   r['shutdown_mode']='official fast_shutdown=True' if FAST_SHUTDOWN else 'full per-extension teardown';r['close_requested']=True;save()
   # Release only this procedural scene's Python handles before the public async close.
   for name in ['stage','world','scene','ground','cube','cam','light','product','rgb','depth','color','z','valid','nz','depth16']:
    globals().pop(name,None)
   gc.collect();future=asyncio.ensure_future(omni.usd.get_context().close_stage_async());drain=time.monotonic()
   for _ in range(600):
    if future.done() or time.monotonic()-drain>20:break
    app.update()
   r['usd_async_close_done']=future.done()
   if future.done():r['usd_async_close_result']=str(future.result())
   for _ in range(3):app.update()
   save();t=time.monotonic();app.close(wait_for_replicator=False);r['close_seconds']=time.monotonic()-t;r['closed']=True;telemetry('closed')
  except Exception as e:r['close_error']=str(e)
 save();print(json.dumps(r,indent=2))

r['total_seconds']=time.monotonic()-started;r['acceptance_core']=bool(r.get('startup') and r.get('render') and r.get('physics',{}).get('moved_down') and r.get('closed') and not r.get('exception'));save();sys.exit(0 if r['acceptance_core'] else 1)
