"""Rust vs NumPy timing on the shipped inputs: perception on the two recorded head frames in test/fixtures
(v5 head frame, v7 s102 start frame) and grasp planning for 10 objects on the synthetic test scene.
Each mode runs in its own process (TEAM048_ACCEL=1 / 0); the table reports the median of N repeats.
Usage: python3 tools/bench_accel.py [N]      (from the repository root, team048_accel installed for Rust)"""
import os,sys,json,subprocess,platform,statistics
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG=os.path.join(ROOT,'src','team048_solution');TEST=os.path.join(PKG,'test')
def worker(n):
    sys.path[:0]=[PKG,TEST]
    import time,numpy as np
    import scene_fixture as SF
    from team048_solution import _accel,arm,perception
    fx=os.path.join(TEST,'fixtures')
    z5=np.load(os.path.join(fx,'v5_head_frame.npz'));z7=np.load(os.path.join(fx,'v7_s102_start_frames.npz'))
    K5=z5['K'].reshape(3,3);K7=z7['K'].reshape(3,3)
    T7=arm.fk_link('camera_head_link',{'head_pitch_joint':float(z7['head_pitch'])})
    q=z5['tf_q'];x,y,zz,w=q;R=np.array([[1-2*(y*y+zz*zz),2*(x*y-zz*w),2*(x*zz+y*w)],[2*(x*y+zz*w),1-2*(x*x+zz*zz),2*(y*zz-x*w)],[2*(x*zz-y*w),2*(y*zz+x*w),1-2*(x*x+y*y)]])
    T5=np.eye(4);T5[:3,:3]=R;T5[:3,3]=z5['tf_t']
    frames=[(z5['depth'].astype(float),K5,T5,z5['rgb']),(z7['head_depth'].astype(float),K7,T7,z7['head_rgb'])]
    ROI=(np.array([.18,-.62,-1.]),np.array([.62,.62,1.]))   # same window as node.observe
    def perceive():
        for d,K,T,rgb in frames:
            P=perception.deproject(d,K,T);valid=d>0
            plane=perception.table_plane(P,perception.table_region(P,valid))
            perception.segment(P,rgb,perception.table_region(P,valid),plane,ROI,min_pixels=40)
    arms={k:arm.Arm(k) for k in 'lr'};nrm=np.array([-.0699,0,.9976]);plane=(nrm/np.linalg.norm(nrm),.1788/np.linalg.norm(nrm))
    for A in arms.values():A.set_scene(plane,list(SF.TRAYS.values()))
    objs=[(x,y) for x in (.36,.44,.52) for y in (-.2,0.,.2)]+[(.40,.1)]
    def plan():
        for x,y in objs:
            z=arm.plane_height(plane,x,y)+.04
            arms['l' if y>=0 else 'r'].plan_pick(np.array([x,y,z]),np.array([0,1.,0]),np.array([1.,0,0]),.06)
    out={}
    for name,f in (('perception_ms_per_frame',perceive),('planning_s_10_objects',plan)):
        f();ts=[]
        for _ in range(n):
            t=time.perf_counter();f();ts.append(time.perf_counter()-t)
        out[name]=statistics.median(ts)*(1000/len(frames) if name.startswith('perception') else 1)
    out['rust']=bool(_accel.enabled());return out
if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[1]=='--worker':print(json.dumps(worker(int(sys.argv[2]))));sys.exit()
    n=int(sys.argv[1]) if len(sys.argv)>1 else 7;res={}
    for mode in ('0','1'):
        env=dict(os.environ,TEAM048_ACCEL=mode)
        res[mode]=json.loads(subprocess.check_output([sys.executable,__file__,'--worker',str(n)],env=env,text=True).strip().splitlines()[-1])
    if not res['1']['rust']:sys.exit('team048_accel is not installed: nothing to compare')
    cpu=next((l.split(':',1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')),platform.processor())
    print(f'{cpu}, {os.cpu_count()} logical CPUs, Python {platform.python_version()}, median of {n} runs')
    print('| Measurement | NumPy | Rust | Speed-up |\n|---|---:|---:|---:|')
    for k,label in (('perception_ms_per_frame','Perception, ms per frame'),('planning_s_10_objects','Grasp planning, 10 objects, s')):
        a,b=res['0'][k],res['1'][k];print(f'| {label} | {a:.3g} | {b:.3g} | {a/b:.1f}x |')
