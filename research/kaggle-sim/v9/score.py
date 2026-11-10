# debug-only scoring via /sim/task/check (disabled at final evaluation); also measures sim time
# v12: v11b got no /sim/task/result (sim log without 'task check'): the check was published 30 x 0.3 s right after the
# node started, possibly before discovery matched. Now: wait until the sim subscribes to /sim/task/check and publishes
# /sim/task/result (up to 60 s), then re-send the check once per second for up to 60 s. QoS unchanged (v9b worked).
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
t=time.time()
while time.time()-t<60 and (p.get_subscription_count()==0 or n.count_publishers('/sim/task/result')==0):rclpy.spin_once(n,timeout_sec=.2)
match=dict(check_subs=p.get_subscription_count(),result_pubs=n.count_publishers('/sim/task/result'),wait_s=round(time.time()-t,1))
t=time.time();sent=0
while not res and time.time()-t<60:
    p.publish(Int32(data=1));sent+=1;t1=time.time()
    while not res and time.time()-t1<1.:rclpy.spin_once(n,timeout_sec=.1)
r=res;sc=10*r.get('in_right_tray',0)-10*r.get('in_wrong_tray',0)
print(json.dumps({'task':sys.argv[2],'seed':int(sys.argv[3]),'score_lb':sc if res else None,'sim_t_end':clk.get('t'),**{k:r.get(k) for k in('in_right_tray','in_wrong_tray','not_in_tray','total')},'objects':r.get('objects'),'check':dict(match,sent=sent,got=bool(res))}))
