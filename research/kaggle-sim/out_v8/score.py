# debug-only scoring via /sim/task/check (disabled at final evaluation); also measures sim time
import rclpy,json,sys,time
from rclpy.node import Node
from std_msgs.msg import Int32,String
from rosgraph_msgs.msg import Clock
rclpy.init();n=Node('score');res={};clk={}
n.create_subscription(String,'/sim/task/result',lambda m:res.update(json.loads(m.data)),10)
n.create_subscription(Clock,'/clock',lambda m:clk.update(t=m.clock.sec+m.clock.nanosec*1e-9),10)
p=n.create_publisher(Int32,'/sim/task/check',10)
t=time.time()
while 't' not in clk and time.time()-t<20:rclpy.spin_once(n,timeout_sec=.1)
if sys.argv[1]=='start':print(json.dumps({'sim_t':clk.get('t')}));sys.exit()
for _ in range(30):
    p.publish(Int32(data=1));rclpy.spin_once(n,timeout_sec=.3)
    if res:break
r=res;sc=10*r.get('in_right_tray',0)-10*r.get('in_wrong_tray',0)
print(json.dumps({'task':sys.argv[2],'seed':int(sys.argv[3]),'score_lb':sc,'sim_t_end':clk.get('t'),**{k:r.get(k) for k in('in_right_tray','in_wrong_tray','not_in_tray','total')},'objects':r.get('objects')}))
