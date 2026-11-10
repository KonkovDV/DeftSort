# save one RGB + depth frame from each camera with camera_info (sensor data only) for offline calibration/perception
import rclpy,numpy as np,json,time
from rclpy.node import Node
from sensor_msgs.msg import Image,CameraInfo
from rclpy.qos import qos_profile_sensor_data
rclpy.init();n=Node('grab');got={}
def cb(k):
    def f(m):
        if k in got:return
        if isinstance(m,CameraInfo):got[k]=dict(K=list(m.k),w=m.width,h=m.height,frame=m.header.frame_id,D=list(m.d))
        else:
            dt={'rgb8':np.uint8,'bgr8':np.uint8,'16UC1':np.uint16,'32FC1':np.float32}.get(m.encoding,np.uint8)
            a=np.frombuffer(bytes(m.data),dt);np.save(f'/kaggle/working/{k}.npy',a.reshape(m.height,m.width,-1));got[k]=dict(enc=m.encoding,frame=m.header.frame_id,stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9)
    return f
for cam in('cam_head','cam_torso'):
    for s in('color','depth'):
        n.create_subscription(Image,f'/{cam}/{s}/image_raw',cb(f'{cam}_{s}'),qos_profile_sensor_data)
        n.create_subscription(CameraInfo,f'/{cam}/{s}/camera_info',cb(f'{cam}_{s}_info'),10)
t=time.time()
while time.time()-t<40 and len(got)<8:rclpy.spin_once(n,timeout_sec=.2)
json.dump(got,open('/kaggle/working/frames.json','w'),indent=1);print(sorted(got))
