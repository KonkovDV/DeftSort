# save one RGB + depth frame from each camera (sensor data only) for offline diagnosis; GRAB_OUT = output dir
import rclpy,numpy as np,json,time,os
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
O=os.environ.get('GRAB_OUT','/kaggle/working');os.makedirs(O,exist_ok=True)
rclpy.init();n=Node('grab');got={}
def cb(k):
    def f(m):
        if k in got:return
        dt={'rgb8':np.uint8,'bgr8':np.uint8,'16UC1':np.uint16,'32FC1':np.float32}.get(m.encoding,np.uint8)
        a=np.frombuffer(bytes(m.data),dt);np.save(f'{O}/{k}.npy',a.reshape(m.height,m.width,-1));got[k]=dict(enc=m.encoding,frame=m.header.frame_id,stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9)
    return f
for cam in('cam_head','cam_torso'):
    for s in('color','depth'):
        n.create_subscription(Image,f'/{cam}/{s}/image_raw',cb(f'{cam}_{s}'),qos_profile_sensor_data)
t=time.time()
while time.time()-t<30 and len(got)<4:rclpy.spin_once(n,timeout_sec=.2)
json.dump(got,open(f'{O}/frames.json','w'),indent=1);print(sorted(got))
