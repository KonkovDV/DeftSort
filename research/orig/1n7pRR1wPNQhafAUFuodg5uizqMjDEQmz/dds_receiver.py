import json,time,hashlib,traceback,sys,math,struct
from pathlib import Path
OUT=Path('/kaggle/working');r={'independent_process':True,'messages':{},'clock_progress':False,'use_sim_time':False};node=None
try:
 import rclpy
 from rclpy.parameter import Parameter
 from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy,ReliabilityPolicy
 from sensor_msgs.msg import Image,CameraInfo
 from rosgraph_msgs.msg import Clock
 from tf2_msgs.msg import TFMessage
 from std_msgs.msg import Int32
 rclpy.init();node=rclpy.create_node('neutral_external_receiver',parameter_overrides=[Parameter('use_sim_time',value=True)]);r['use_sim_time']=node.get_parameter('use_sim_time').value
 expected=json.loads((OUT/'dds_expected.json').read_text());pub=node.create_publisher(Int32,'/neutral_probe/ack',10)
 def image(msg,kind):
  r['messages'][kind]={'encoding':msg.encoding,'width':msg.width,'height':msg.height,'step':msg.step,'bytes':len(msg.data),'sha256':hashlib.sha256(bytes(msg.data)).hexdigest(),'stamp_ns':msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec,'frame_id':msg.header.frame_id}
 def clock(msg):
  t=msg.clock.sec*1000000000+msg.clock.nanosec
  r.setdefault('first_clock_ns',t);r['last_clock_ns']=t;r['clock_progress']=t>r['first_clock_ns'];r['node_clock_ns']=node.get_clock().now().nanoseconds
 def info(msg):r['messages']['camera_info']={'k':list(msg.k),'p':list(msg.p),'width':msg.width,'height':msg.height,'frame_id':msg.header.frame_id,'stamp_ns':msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec}
 def tf(msg):r['messages']['tf']=[{'parent':x.header.frame_id,'child':x.child_frame_id,'translation':[x.transform.translation.x,x.transform.translation.y,x.transform.translation.z],'rotation_xyzw':[x.transform.rotation.x,x.transform.rotation.y,x.transform.rotation.z,x.transform.rotation.w]} for x in msg.transforms]
 subs=[node.create_subscription(Image,'/neutral_probe/rgb',lambda m:image(m,'rgb'),qos_profile_sensor_data),node.create_subscription(Image,'/neutral_probe/depth',lambda m:image(m,'depth'),qos_profile_sensor_data),node.create_subscription(CameraInfo,'/neutral_probe/camera_info',info,10),node.create_subscription(Clock,'/clock',clock,qos_profile_sensor_data),node.create_subscription(TFMessage,'/neutral_probe/tf_static',tf,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE))]
 (OUT/'dds_receiver_ready.json').write_text(json.dumps({'ready':True,'python':sys.version,'rclpy_module':rclpy.__file__}))
 start=time.monotonic()
 for _ in range(3500):
  if time.monotonic()-start>32:break
  rclpy.spin_once(node,timeout_sec=.02)
  data=r['messages'];r['payloads_match']=all(data.get(k,{}).get('sha256')==expected[k] for k in ['rgb','depth'])
  ci=data.get('camera_info',{});r['metadata_matches']=all(data.get(k,{}).get('frame_id')==expected['frame_id'] and data.get(k,{}).get('stamp_ns')==expected['stamp_ns'] for k in ['rgb','depth','camera_info']) and ci.get('width')==640 and ci.get('height')==360 and len(ci.get('k',[]))==9 and abs(ci.get('k',[0])[0]-expected['fx'])<1e-7
  r['received_all']=r['metadata_matches'] and r['payloads_match'] and all(k in data for k in ['camera_info','tf']) and r['clock_progress'] and r.get('node_clock_ns',0)>0
  if r['received_all']:
   pub.publish(Int32(data=42));r['ack_published']=True
   if r.setdefault('first_ack_wall',time.monotonic())+1<time.monotonic():break
except Exception as e:r['exception']=str(e);r['traceback']=traceback.format_exc()
finally:
 if node is not None:node.destroy_node()
 try:rclpy.shutdown()
 except Exception:pass
 (OUT/'ros_dds_receiver_report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
