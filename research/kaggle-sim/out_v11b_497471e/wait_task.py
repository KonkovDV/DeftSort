"""Harness only (Kaggle runs, never part of the solution): wait for the latched /sim/task in ONE long-lived ROS node.

Why: v9a (f525ed9, sort3 seed 103) polled with `timeout 8 ros2 topic echo --once` every ~14 s; camera frames
were flowing from bridge +1115 s (bridge_sort3.log:1295), yet 105 polls in 1512 s got nothing, so the run ended
as SIM NOT UP (run_sort_trace.log:786-788). Every poll started a fresh CLI process that had to discover the
publisher again inside its 8 s cap. v9b, same command, succeeded on the first poll after frames appeared
(+1086 s, run_sort_trace.log:583).
QoS: RELIABLE + TRANSIENT_LOCAL, depth 1. The solution node uses this profile and received /sim/task in every
run (run_sort2_101.log:1003), so it matches the publisher; nothing is assumed beyond that.
Every 60 s the script logs which /sim/task publishers it can see and their QoS, so the next run records the
publisher side as well.
Usage: wait_task.py <timeout_s> <out_file>; exit 0 once a message arrives, 1 on timeout."""
import sys,time,json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
tmax=float(sys.argv[1]);out=sys.argv[2]
rclpy.init();n=Node('team048_wait_task');got={}
lat=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
n.create_subscription(String,'/sim/task',lambda m:got.setdefault('task',m.data),lat)
n.create_subscription(Clock,'/clock',lambda m:got.setdefault('clock',time.time()),10)
t0=time.time();last=-1e9
def status(tag):
    pubs=[dict(node=p.node_name,durability=str(p.qos_profile.durability),reliability=str(p.qos_profile.reliability),
               depth=p.qos_profile.depth) for p in n.get_publishers_info_by_topic('/sim/task')]
    print(json.dumps(dict(tag=tag,t=round(time.time()-t0,1),clock_seen=('clock' in got),task_publishers=pubs)),flush=True)
while time.time()-t0<tmax and 'task' not in got:
    rclpy.spin_once(n,timeout_sec=.5)
    if time.time()-last>=60:last=time.time();status('waiting')
status('done' if 'task' in got else 'timeout')
if 'task' in got:
    open(out,'w').write('data: '+got['task']+'\n');print('sim task received after',round(time.time()-t0,1),'s',flush=True)
n.destroy_node();rclpy.shutdown();sys.exit(0 if 'task' in got else 1)
