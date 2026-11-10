"""v14 audit: squeeze-force range enforced for every env setting, rule checks on the AST of the solution package
(ROS interfaces, imports, forbidden simulator/USD names), run() setup failures still end with /sim/task/done,
explicit bgr8 handling of the colour image (tray colour must never swap)."""
import ast,glob,os,subprocess,sys,types,unittest
from unittest import mock
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE)
import test_node_stub as TS                      # installs the ROS stubs
import test_v13 as V13
from team048_solution import node as N, grip as G, arm as ARM
PKG=os.path.join(HERE,'..','team048_solution')

class ForceRange(unittest.TestCase):
    def ratio(s,env):
        e=dict(os.environ);e.pop('TEAM048_HAND_FORCE',None)
        if env is not None:e['TEAM048_HAND_FORCE']=env
        out=subprocess.run([sys.executable,'-c','from team048_solution import grip;print(repr(grip.HAND_FORCE_RATIO))'],
                           cwd=os.path.join(HERE,'..'),env=e,capture_output=True,text=True,check=True).stdout
        return float(out)
    def test_env_override_is_clamped_to_organiser_range(s):
        # before v14 the override was clamped to the simulator's [0.02, 1]: TEAM048_HAND_FORCE=1 gave the default force
        for env,want in ((None,.4),('0.45',.45),('0.3',.3),('0.5',.5),('1.0',.5),('0.02',.3),('-3',.3),('nan',.3),('inf',.5),('abc',.4)):
            got=s.ratio(env);s.assertTrue(.3<=got<=.5,(env,got));s.assertAlmostEqual(got,want,msg=env)
    def test_both_hands_squeeze_dofs_in_range_thumb_rotation_documented(s):
        sol=N.Sol();R=sol.hand_requests()
        for k in 'lr':
            r=R[('force',k)];v=[getattr(r,f'force{i}_ratio') for i in range(6)]
            s.assertTrue(all(.3<=x<=.5 for x in v[:5]),v)          # DOF 1-5 squeeze the object (#648)
            s.assertEqual(v[5],G.HAND_FORCE_THUMB_ROT);s.assertEqual(v[5],1.0)   # DOF 6 positions the thumb: COMPLIANCE.md
        doc=os.path.join(HERE,'..','..','..','docs','COMPLIANCE.md')
        if os.path.exists(doc):s.assertIn('HAND_FORCE_THUMB_ROT',open(doc).read())   # the assumption is documented

def package_ast():
    for f in sorted(glob.glob(os.path.join(PKG,'*.py'))):
        yield os.path.basename(f),ast.parse(open(f).read())

ALLOWED_IF={'/sim/task','/sim/task/done','/joint_states','/robot_description',
            '/cam_head/color/image_raw','/cam_head/depth/image_raw','/cam_head/depth/camera_info',
            '/cam_torso/color/image_raw','/cam_torso/depth/image_raw','/cam_torso/depth/camera_info',
            '/arm_6dof_{}','/inspire_hand/state/{}_hand','/inspire_hand/ctrl/{}_hand','/inspire_hand/set_force/{}_hand',
            '/inspire_hand/set_speed/{}_hand','/{}/cmd_pos'}
ALLOWED_IMPORTS={'numpy','scipy','rclpy','tf2_ros','std_msgs','sensor_msgs','geometry_msgs','bodyctrl_msgs','team048_accel',
                 'os','sys','re','json','math','time','threading','collections','dataclasses','heapq','itertools','xml'}
class Rules(unittest.TestCase):
    def test_ros_interfaces_whitelist(s):
        """Every topic/service name the package creates (string or f-string starting with '/' passed to
        create_subscription/create_publisher/create_client) is one of the organiser's public interfaces."""
        used=set()
        for f,t in package_ast():
            for n in ast.walk(t):
                if isinstance(n,ast.Call) and getattr(n.func,'attr','') in ('create_subscription','create_publisher','create_client'):
                    a=n.args[1]
                    if isinstance(a,ast.Constant):used.add(a.value)
                    elif isinstance(a,ast.JoinedStr):used.add(''.join(v.value if isinstance(v,ast.Constant) else '{}' for v in a.values))
                    else:s.fail(f'{f}:{n.lineno}: non-literal interface name')
        s.assertEqual(used-ALLOWED_IF,set());s.assertIn('/sim/task/done',used)
    def test_no_debug_topics_randomize_set_usd_or_sim_internals(s):
        bad=('/sim/task/check','/sim/objects','randomiz','.usd','pxr','omni.','isaacsim','isaaclab','omni_','get_prim')
        for f,t in package_ast():
            for n in ast.walk(t):
                if isinstance(n,ast.Constant) and isinstance(n.value,str):
                    v=n.value.lower()
                    if isinstance(n,ast.Constant) and len(v)<200 and not ' ' in v:   # names, not prose (docstrings mention them)
                        s.assertFalse(any(b in v for b in bad),f'{f}:{n.lineno}: {n.value!r}')
                        if v.startswith('/sim/'):s.assertIn(v,('/sim/task','/sim/task/done'),f'{f}:{n.lineno}')   # no /sim/*/set, check, randomize
                if isinstance(n,(ast.Import,ast.ImportFrom)):
                    mods=[a.name for a in n.names] if isinstance(n,ast.Import) else ([n.module] if n.level==0 else [])
                    for m in mods:s.assertIn(m.split('.')[0],ALLOWED_IMPORTS,f'{f}:{n.lineno}: import {m}')
    def test_no_scene_numbers_in_package_data(s):
        """data/*.json are robot geometry and object priors (category, size, mass): no positions."""
        import json
        d=json.load(open(os.path.join(PKG,'data','priors.json')))
        for k,v in d.items():s.assertEqual(set(v)-{'category','height','w_min','w_max','mass'},set(),k)

class SetupFailures(unittest.TestCase):
    def test_task_without_categories_still_publishes_done(s):
        sol=N.Sol();logs=[];sol.log=lambda *a:logs.append(' '.join(str(x) for x in a))
        sol.task={'task':'sort2','objects':{'a':'box'}};sol.js={'x':0.};TS.Clock.t=50.
        try:
            with mock.patch.object(N,'WALL_LAUNCH',sol.wall_ready):sol.run()
        finally:TS.Clock.t=100.
        s.assertTrue(sol.pubs['/sim/task/done'].sent);s.assertTrue(any("reason: exception KeyError" in l for l in logs))
        s.assertFalse(any(l.startswith('end: retreat failed') for l in logs))   # no arm model: nothing commanded
    def test_arm_model_failure_still_publishes_done(s):
        sol=N.Sol();logs=[];sol.log=lambda *a:logs.append(' '.join(str(x) for x in a))
        sol.task={'task':'sort2','categories':{'box':'tray_left'},'objects':{'a':'box'},'trays':{'tray_left':'red'}};sol.js={'x':0.}
        TS.Clock.t=50.
        def boom(k):raise ValueError('bad URDF')
        try:
            with mock.patch.object(N,'WALL_LAUNCH',sol.wall_ready),mock.patch.object(N.ARMMOD,'Arm',boom):sol.run()
        finally:TS.Clock.t=100.
        s.assertTrue(sol.pubs['/sim/task/done'].sent);s.assertTrue(any('reason: exception ValueError: bad URDF' in l for l in logs))
    def test_normal_run_unchanged(s):
        sol,logs,calls=V13.run_stub(pick_place=lambda *a:(setattr(TS.Clock,'t',TS.Clock.t+1.) or True))
        s.assertTrue(sol.pubs['/sim/task/done'].sent);s.assertTrue(any(l.startswith('done at') and "'a', 'b'" in l for l in logs))

def img(enc,h,w,data,t):
    st=types.SimpleNamespace(sec=int(t),nanosec=0)
    return types.SimpleNamespace(encoding=enc,height=h,width=w,data=bytes(data),header=types.SimpleNamespace(stamp=st,frame_id='camera_head_link'))
class ColourEncoding(unittest.TestCase):
    def cap(s,enc):
        sol=N.Sol();rgb=np.zeros((2,2,3),np.uint8);rgb[...,0]=200;rgb[...,2]=10     # red in rgb8 order
        sol.frames.add_rgb(img(enc,2,2,rgb.tobytes(),5));sol.frames.add_depth(img('16UC1',2,2,np.full((2,2),800,np.uint16).tobytes(),5))
        sol.info=types.SimpleNamespace(k=[100.,0,1,0,100.,1,0,0,1]);sol.tf=lambda *a:np.eye(4)
        P,out,valid=sol.capture(not_before=4.,wait=.1);return out
    def test_rgb8_unchanged(s):
        out=s.cap('rgb8');s.assertEqual(int(out[0,0,0]),200);s.assertEqual(int(out[0,0,2]),10)
    def test_bgr8_reordered(s):
        out=s.cap('bgr8');s.assertEqual(int(out[0,0,0]),10);s.assertEqual(int(out[0,0,2]),200)
if __name__=='__main__':unittest.main()

from team048_solution import trays as TRY
class TrayAxisSign(unittest.TestCase):
    def test_two_representations_of_x_axis_agree(s):
        # min_area_rect tie at 0 / 90 deg: u(0)=(1,0) or v(90 deg)=(-cos..,..)=(-1,6e-17); before v14 they kept opposite signs
        a=TRY.canonical_axis(np.array([1.,0.]));b=TRY.canonical_axis(np.array([-1.,6.123233995736766e-17]))
        s.assertGreater(a[0],0);s.assertGreater(b[0],0);s.assertTrue(np.allclose(a,b,atol=1e-12))
        s.assertTrue(np.allclose(TRY.canonical_axis(np.array([-1.,-1e-12])),[1.,1e-12]))
    def test_general_axes_unchanged(s):
        for ax in ([.927,.375],[-.3,.954],[0.,1.]):
            s.assertEqual(TRY.canonical_axis(np.array(ax)).tolist(),ax)              # y > 0: kept as before
        for ax in ([.927,-.375],[.3,-.954],[0.,-1.]):
            s.assertEqual(TRY.canonical_axis(np.array(ax)).tolist(),[-ax[0],-ax[1]])   # y < 0: flipped as before
    def test_detect_uses_canonical_sign(s):
        # synthetic x-aligned tray: whichever tie winner min_area_rect returns, detect reports axis x > 0
        P=np.zeros((60,80,3));P[...,0]=np.linspace(.2,.5,80)[None,:];P[...,1]=np.linspace(.5,.72,60)[:,None];P[...,2]=-.15
        rgb=np.zeros((60,80,3),np.uint8);rgb[...,0]=200
        valid=np.ones((60,80),bool);valid[0,:]=valid[-1,:]=valid[:,0]=valid[:,-1]=False
        plane=(np.array([0,0,1.]),.18)
        for forced in (np.array([1.,0.]),np.array([-1.,6.123233995736766e-17])):
            with mock.patch.object(TRY,'min_area_rect',lambda xy,q=.02,f=forced:(np.array([.35,.61]),np.array([.21,.29]),f)):
                d=TRY.detect(P,rgb,valid,'tray_left','red',plane)
            s.assertIsNotNone(d);s.assertGreater(d['axis'][0],0)


from team048_solution import perception as PER, _accel as ACC
class FfiEdges(unittest.TestCase):
    """research/diag/v14/ffi_edges.py: of 43 edge inputs only an empty depth image made a Rust kernel panic."""
    @unittest.skipUnless(ACC.enabled2(), 'needs the Rust module (API 2)')
    def test_empty_depth_image_keeps_rust_enabled(s):
        K=np.array([[343.16,0,320],[0,343.16,240],[0,0,1.]]);T=np.eye(4)
        try:
            P=PER.deproject(np.zeros((0,0)),K,T)
            s.assertEqual(P.shape,(0,0,3))
            s.assertTrue(ACC.enabled2(),'an empty image disabled the Rust module for the whole process')
        finally:ACC._disabled=False
