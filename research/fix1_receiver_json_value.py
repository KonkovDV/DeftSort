import json,math,os,time
from pathlib import Path

def json_value(v):
 if v is None or isinstance(v,(str,bool,int)):return v
 if isinstance(v,float):return v if math.isfinite(v) else None
 if isinstance(v,dict):return {str(k):json_value(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [json_value(x) for x in v]
 if hasattr(v,'tolist'):return json_value(v.tolist())
 if hasattr(v,'item'):return json_value(v.item())
 raise TypeError('unsupported report value type: '+type(v).__name__)

def atomic_report(path,data):
 path=Path(path);text=json.dumps(json_value(data),indent=2,allow_nan=False);tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 try:
  with tmp.open('w') as f:f.write(text);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()
 return text

def full_acceptance(scene,child_rc,receiver):
 ros=scene.get('ros_bridge') or {};lab=scene.get('isaaclab') or {};rx=receiver or {};pred=rx.get('predicates',{})
 checks={'startup':scene.get('startup') is True,'rgb_depth':scene.get('render') is True,'physics':bool((scene.get('physics') or {}).get('moved_down')),'graceful_return':scene.get('closed') is True and scene.get('shutdown_mode')=='normal close; fast_shutdown=False','native_exit0':child_rc==0,'no_scene_exception':not scene.get('exception'),'receiver_report_finished':rx.get('phase')=='finished','receiver_exit0':ros.get('receiver_exit_code')==0 and rx.get('exit_code')==0,'receiver_no_error':not rx.get('exception') and not rx.get('cleanup_errors'),'actual_dds_ack_artifact':bool(ros.get('ack_queue')) and ros.get('ack_checkpoint_verified') is True,'independent_process':rx.get('pid') is not None and scene.get('process_pid') is not None and rx.get('pid')!=scene.get('process_pid'),'clock':pred.get('clock_progress') is True,'use_sim_time':pred.get('use_sim_time') is True,'rgb':pred.get('rgb') is True,'depth_mm':pred.get('depth_mm') is True,'camera_info':pred.get('camera_info') is True,'tf':pred.get('tf') is True,'stamps':pred.get('stamps') is True,'lab10steps':lab.get('steps',0)>=10 and lab.get('simulation_seconds',0)>0 and not lab.get('exception') and not lab.get('cleanup_errors')}
 return {'checks':checks,'passed':all(checks.values()),'criterion':'normal close must return; exit0/ACK alone never substitute for saved receiver report and real Lab steps'}

import sys,time,hashlib,traceback,math
OUT=Path(os.environ['NEUTRAL_OUT']);r={'pid':os.getpid(),'phase':'starting','predicates':{},'messages':{},'cleanup_errors':[],'ack_payloads':[]};node=None;started=time.monotonic()
def log(event,**data):print(json.dumps(json_value({'event':event,'elapsed':time.monotonic()-started,**data}),allow_nan=False),flush=True)
try:
 import numpy as np,rclpy
 from rclpy.parameter import Parameter
 from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy,ReliabilityPolicy
 from sensor_msgs.msg import Image,CameraInfo
 from rosgraph_msgs.msg import Clock
 from tf2_msgs.msg import TFMessage
 from std_msgs.msg import String
 expected=json.loads((OUT/'dds_expected.json').read_text());rclpy.init();node=rclpy.create_node('neutral_external_receiver_v2',parameter_overrides=[Parameter('use_sim_time',value=True)]);r['predicates']['use_sim_time']=bool(node.get_parameter('use_sim_time').value);pub=node.create_publisher(String,'/neutral_probe/ack_v2',10)
 def image(m,kind):
  raw=bytes(m.data);r['messages'][kind]={'encoding':m.encoding,'width':m.width,'height':m.height,'step':m.step,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'stamp_ns':int(m.header.stamp.sec)*10**9+int(m.header.stamp.nanosec),'frame_id':m.header.frame_id,'is_bigendian':m.is_bigendian}
  common=m.width==640 and m.height==360 and m.header.frame_id==expected['frame_id'] and r['messages'][kind]['sha256']==expected[kind]
  if kind=='rgb':r['predicates']['rgb']=bool(common and m.encoding=='rgb8' and m.step==1920 and len(raw)==691200)
  else:
   d=np.frombuffer(raw,dtype='<u2');nonzero=d[d>0];r['messages'][kind].update({'zero_pixels':int((d==0).sum()),'valid_pixels':int((d>0).sum()),'min_mm':int(nonzero.min()) if nonzero.size else None,'max_mm':int(nonzero.max()) if nonzero.size else None});r['predicates']['depth_mm']=bool(common and m.encoding=='16UC1' and m.step==1280 and len(raw)==460800 and not m.is_bigendian and nonzero.size and np.all((nonzero>=150)&(nonzero<=6000)) and int((d==0).sum())==expected['depth_invalid_pixels'])
  if not r.setdefault('logged_'+kind,False):log('image_received',kind=kind,details=r['messages'][kind]);r['logged_'+kind]=True
 def clock(m):
  t=int(m.clock.sec)*10**9+int(m.clock.nanosec);r.setdefault('first_clock_ns',t);r['last_clock_ns']=t;r['node_clock_ns']=int(node.get_clock().now().nanoseconds);r['predicates']['clock_progress']=bool(t>r['first_clock_ns'] and r['node_clock_ns']>0)
 def ci(m):
  fx=expected['fx'];r['messages']['camera_info']={'width':m.width,'height':m.height,'k':list(m.k),'p':list(m.p),'frame_id':m.header.frame_id,'stamp_ns':int(m.header.stamp.sec)*10**9+int(m.header.stamp.nanosec)};r['predicates']['camera_info']=bool(m.width==640 and m.height==360 and m.header.frame_id==expected['frame_id'] and np.allclose(m.k,[fx,0,320,0,fx,180,0,0,1],rtol=0,atol=1e-7) and np.allclose(m.p,[fx,0,320,0,0,fx,180,0,0,0,1,0],rtol=0,atol=1e-7) and np.allclose(m.r,np.eye(3).ravel()) and m.distortion_model=='plumb_bob' and np.allclose(m.d,0))
 def tf(m):
  values=[{'parent':x.header.frame_id,'child':x.child_frame_id,'translation':[x.transform.translation.x,x.transform.translation.y,x.transform.translation.z],'quaternion_xyzw':[x.transform.rotation.x,x.transform.rotation.y,x.transform.rotation.z,x.transform.rotation.w],'stamp_ns':int(x.header.stamp.sec)*10**9+int(x.header.stamp.nanosec)} for x in m.transforms];r['messages']['tf']=values;r['predicates']['tf']=bool(any(x['parent']=='neutral_world' and x['child']==expected['frame_id'] and all(math.isfinite(v) for v in x['translation']+x['quaternion_xyzw']) and abs(sum(v*v for v in x['quaternion_xyzw'])-1)<1e-6 for x in values))
 subs=[node.create_subscription(Image,'/neutral_probe/rgb',lambda m:image(m,'rgb'),qos_profile_sensor_data),node.create_subscription(Image,'/neutral_probe/depth',lambda m:image(m,'depth'),qos_profile_sensor_data),node.create_subscription(CameraInfo,'/neutral_probe/camera_info',ci,10),node.create_subscription(Clock,'/clock',clock,qos_profile_sensor_data),node.create_subscription(TFMessage,'/neutral_probe/tf_static',tf,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE))]
 r['phase']='subscribing';atomic_report(OUT/'dds_receiver_ready.json',{'ready':True,'pid':os.getpid(),'python':sys.version,'rclpy_module':rclpy.__file__});log('subscriptions_ready',use_sim_time=r['predicates']['use_sim_time'])
 for _ in range(3500):
  if time.monotonic()-started>35:break
  rclpy.spin_once(node,timeout_sec=.02);messages=r['messages'];r['predicates']['stamps']=bool(all(messages.get(k,{}).get('stamp_ns')==expected['stamp_ns'] for k in ['rgb','depth','camera_info']) and r.get('last_clock_ns',0)>=expected['stamp_ns'])
  required=['use_sim_time','clock_progress','rgb','depth_mm','camera_info','tf','stamps'];r['received_all']=bool(all(r['predicates'].get(k) is True for k in required))
  if r['received_all']:
   if not r['ack_payloads']:
    r['phase']='validated_before_ack';text=atomic_report(OUT/'ros_receiver_validated_checkpoint.json',r);payload={'kind':'neutral_ack_v2','seq':1,'pid':os.getpid(),'snapshot_stamp_ns':expected['stamp_ns'],'receiver_clock_ns':r['last_clock_ns'],'predicates':r['predicates'],'rgb_sha256':messages['rgb']['sha256'],'depth_sha256':messages['depth']['sha256'],'checkpoint_sha256':hashlib.sha256(text.encode()).hexdigest()};r['ack_payloads'].append(payload);r['first_ack_wall']=time.monotonic();log('all_predicates_validated',payload=payload)
   pub.publish(String(data=json.dumps(json_value(r['ack_payloads'][-1]),allow_nan=False)))
   if time.monotonic()-r['first_ack_wall']>1:break
except Exception as e:r['exception']=str(e);r['traceback']=traceback.format_exc();log('receiver_error',error=str(e));traceback.print_exc()
finally:
 if node is not None:
  try:node.destroy_node()
  except Exception as e:r['cleanup_errors'].append('node: '+str(e))
 try:
  if 'rclpy' in globals() and rclpy.ok():rclpy.shutdown()
 except Exception as e:r['cleanup_errors'].append('context: '+str(e))
 r['phase']='finished';r['exit_code']=0 if r.get('received_all') and not r.get('exception') and not r['cleanup_errors'] else 1;r['elapsed_seconds']=time.monotonic()-started;atomic_report(OUT/'ros_dds_receiver_report.json',r);log('finished',exit_code=r['exit_code'],predicates=r['predicates']);sys.exit(r['exit_code'])
