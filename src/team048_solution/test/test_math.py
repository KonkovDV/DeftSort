"""v15 math review (docs/MATH_PHYSICS.md): property tests of the formulas on random inputs and checks against the
simulator's own numbers (camera_info and /tf printed by tf2_echo in the Kaggle v5 logs, objects.yaml masses).
Nothing here changes a plan; every constant below names its source."""
import os,sys,math,unittest
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import test_node_stub as TS  # noqa: F401  (installs the ROS stubs)
from team048_solution import node as N, arm as ARM, geometry as GEO, perception as PE, trays as TR, grasp as GR

# camera_info of /camera/head and /camera/body_front (Kaggle v5 logs, frames.json): 640x360, D = 0
K_LOG=np.array([[343.1635437011719,0.,320.],[0.,343.1635437011719,180.],[0.,0.,1.]]);W_LOG,H_LOG=640,360
# tf2_echo pelvis camera_head_link at sim t = 62.17 s (Kaggle v5 logs): translation, quaternion xyzw, printed matrix
TF_T=[0.105,0.011,0.648];TF_Q=[0.613,-0.613,0.353,-0.353]
TF_M=np.array([[0.000,-0.502,0.865,0.105],[-1.000,-0.000,0.000,0.011],[0.000,-0.865,-0.502,0.648]])

def rquat(rng):
    q=rng.normal(size=4);return q/np.linalg.norm(q)
def qmul(a,b):
    x1,y1,z1,w1=a;x2,y2,z2,w2=b
    return np.array([w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2,w1*w2-x1*x2-y1*y2-z1*z2])
def inv(T):
    R=T[:3,:3];o=np.eye(4);o[:3,:3]=R.T;o[:3,3]=-R.T@T[:3,3];return o

class Transforms(unittest.TestCase):
    def test_quat_to_T_is_a_proper_rigid_transform(s):
        rng=np.random.default_rng(1)
        for _ in range(500):
            T=N.quat_to_T(rng.normal(size=3),rquat(rng));R=T[:3,:3]
            s.assertLess(np.abs(R.T@R-np.eye(3)).max(),1e-12);s.assertAlmostEqual(np.linalg.det(R),1.,places=12)
            GEO.check_transform(T)
    def test_axis_angle_sign_and_geometry_agree(s):
        rng=np.random.default_rng(2)
        for _ in range(300):
            a=rng.normal(size=3);a/=np.linalg.norm(a);th=rng.uniform(-np.pi,np.pi)
            q=np.r_[a*np.sin(th/2),np.cos(th/2)];T=N.quat_to_T([0,0,0],q)
            np.testing.assert_allclose(T[:3,:3],GEO.rotation(a,th),atol=1e-12)
            np.testing.assert_allclose(N.quat_to_T([0,0,0],-q),T,atol=1e-15)          # q and -q: same rotation
            np.testing.assert_allclose(GEO.quaternion_transform([0,0,0],q),T,atol=1e-12)
    def test_composition_and_inverse(s):
        rng=np.random.default_rng(3)
        for _ in range(300):
            t1,t2=rng.normal(size=3),rng.normal(size=3);q1,q2=rquat(rng),rquat(rng)
            A,B=N.quat_to_T(t1,q1),N.quat_to_T(t2,q2)
            np.testing.assert_allclose(A@B,N.quat_to_T(t1+A[:3,:3]@t2,qmul(q1,q2)),atol=1e-12)   # chain = product
            np.testing.assert_allclose(A@inv(A),np.eye(4),atol=1e-12);np.testing.assert_allclose(inv(A),np.linalg.inv(A),atol=1e-12)
            p=rng.normal(size=3);np.testing.assert_allclose((inv(A)@np.r_[A[:3,:3]@p+t1,1.])[:3],p,atol=1e-12)
    def test_simulator_tf_matches_quat_to_T_and_the_urdf_chain(s):
        T=N.quat_to_T(TF_T,TF_Q)
        s.assertLess(np.abs(T[:3]-TF_M).max(),1.5e-3)                  # printed to 3 decimals
        U=ARM.fk_link('camera_head_link',{'head_pitch_joint':0.})       # URDF camera -> head -> pelvis, head level
        s.assertLess(np.abs(U[:3]-TF_M).max(),2.5e-3)
        # optical convention (deproject: x right, y down, z along the view): view axis forward and 30 deg down,
        # image x = robot right (-y pelvis), image y = down
        z=U[:3,2];s.assertGreater(z[0],.8);s.assertAlmostEqual(math.degrees(math.asin(-z[2])),30.,delta=.5)
        np.testing.assert_allclose(U[:3,0],[0,-1,0],atol=1e-3);s.assertLess(U[2,1],-.8)

class Pinhole(unittest.TestCase):
    def test_fixture_intrinsics_are_the_logged_camera_info(s):
        F=os.path.join(os.path.dirname(os.path.abspath(__file__)),'fixtures')
        np.testing.assert_allclose(np.load(os.path.join(F,'v5_head_frame.npz'))['K'].reshape(3,3),K_LOG,rtol=0,atol=1e-9)
        np.testing.assert_allclose(np.load(os.path.join(F,'v7_s102_start_frames.npz'))['K'].reshape(3,3),K_LOG,rtol=0,atol=1e-3)  # stored to 3 decimals
        z=np.load(os.path.join(F,'v5_head_frame.npz'));np.testing.assert_allclose(z['tf_t'],TF_T);np.testing.assert_allclose(z['tf_q'],TF_Q)
        # principal point = image centre, square pixels; horizontal FOV 2*atan(320/fx) = 86.0 deg
        s.assertEqual((K_LOG[0,2],K_LOG[1,2]),(W_LOG/2,H_LOG/2));s.assertEqual(K_LOG[0,0],K_LOG[1,1])
        s.assertAlmostEqual(math.degrees(2*math.atan(W_LOG/2/K_LOG[0,0])),86.,delta=.01)
    def test_deproject_reprojects_onto_its_pixel(s):
        rng=np.random.default_rng(4)
        for k in range(20):
            h,w=int(rng.integers(5,40)),int(rng.integers(5,40))
            K=np.array([[rng.uniform(200,600),0,rng.uniform(0,w)],[0,rng.uniform(200,600),rng.uniform(0,h)],[0,0,1.]])
            T=N.quat_to_T(rng.normal(size=3),rquat(rng));dmm=rng.integers(150,6000,size=(h,w)).astype(float)
            for f in (PE.deproject,PE.deproject_py):
                P=f(dmm,K,T);C=(P.reshape(-1,3)-T[:3,3])@T[:3,:3]                # back to the camera frame
                v,u=np.mgrid[0:h,0:w]
                np.testing.assert_allclose(C[:,2],dmm.ravel()/1000.,atol=1e-9)
                np.testing.assert_allclose(K[0,0]*C[:,0]/C[:,2]+K[0,2],u.ravel(),atol=1e-7)
                np.testing.assert_allclose(K[1,1]*C[:,1]/C[:,2]+K[1,2],v.ravel(),atol=1e-7)

class Ransac(unittest.TestCase):
    def test_finds_the_table_at_the_lowest_measured_inlier_fraction(s):
        """62 recorded head frames: inlier fraction of the table region 0.41-0.63 (research/diag/v15/math_checks.py).
        Synthetic: 2 deg tilted plane, 40 % inliers (+-2 mm noise), 60 % object/tray points above it."""
        rng=np.random.default_rng(5);n0=np.array([math.sin(math.radians(2)),0,math.cos(math.radians(2))]);d0=.2
        for seed in range(10):
            m=6000;k=int(.4*m);xy=rng.uniform([.15,-.7],[.75,.7],size=(m,2))
            z=-(d0+n0[0]*xy[:,0])/n0[2];z[:k]+=rng.uniform(-.002,.002,k)/n0[2];z[k:]+=rng.uniform(.012,.35,m-k)
            P=np.c_[xy,z].reshape(1,m,3);n,d=PE.table_plane(P,np.ones((1,m),bool),seed=seed)
            s.assertLess(math.degrees(math.acos(min(1.,n@n0))),.5);s.assertLess(abs(d-d0),.002)

class Rectangles(unittest.TestCase):
    def test_min_area_rect_within_the_one_degree_scan_bound(s):
        rng=np.random.default_rng(6)
        for _ in range(30):
            a,b=sorted(rng.uniform(.08,.3,2));th=rng.uniform(0,np.pi);u=np.array([math.cos(th),math.sin(th)]);v=np.array([-u[1],u[0]])
            c=rng.uniform(-.5,.5,2);st=rng.uniform(-.5,.5,size=(6000,2))
            xy=c+np.outer(st[:,0]*b,u)+np.outer(st[:,1]*a,v)
            for f in (TR.min_area_rect,TR.min_area_rect_py):
                cc,size,ax=f(xy)
                bound=b*math.sin(math.radians(.5))+.004          # 0.5 deg worst scan offset + quantile sampling
                s.assertLess(abs(size[0]-a),bound);s.assertLess(abs(size[1]-b),bound)
                # axis: NOT within the 0.5 deg scan bound. The 2 % trimmed extents make the area flat over a few degrees
                # around the true angle and the scan keeps the first minimum: 1.7 deg median, 3.6 deg max on the documented
                # tray size (research/diag/v15/MATH_REVIEW.md 4, change deferred: it moves slot positions)
                s.assertLess(math.degrees(math.acos(min(1.,abs(ax@u)))),4.);s.assertLess(np.linalg.norm(cc-c),.004)
    def test_caliper_width_between_true_width_and_half_degree_bound(s):
        rng=np.random.default_rng(7)
        for _ in range(40):
            w,L=sorted(rng.uniform(.02,.2,2));th=rng.uniform(0,2*np.pi);u=np.array([math.cos(th),math.sin(th)]);v=np.array([-u[1],u[0]])
            st=rng.uniform(-.5,.5,size=(400,2));xy=np.outer(st[:,0]*w,u)+np.outer(st[:,1]*L,v)
            xy=np.r_[xy,[s_*w/2*u+t*L/2*v for s_ in (-1,1) for t in (-1,1)]]       # exact corners: true width = w
            cloud=np.c_[xy,rng.uniform(0,.05,len(xy))];f=GR.footprint(cloud,(np.array([0,0,1.]),0.))
            s.assertGreaterEqual(f['width'],w-1e-9);s.assertLessEqual(f['width'],w*math.cos(0)+L*math.sin(math.radians(.5))+1e-9)
    def test_kasa_circle_fit(s):
        rng=np.random.default_rng(8)
        for _ in range(30):
            c=rng.uniform(-.5,.5,2);r=rng.uniform(.02,.1);a=rng.uniform(0,2*np.pi,200)
            xy=c+r*np.c_[np.cos(a),np.sin(a)];c2,r2=GR.circle_fit(xy)
            np.testing.assert_allclose(c2,c,atol=1e-9);s.assertAlmostEqual(r2,r,places=9)

class RimTorque(unittest.TestCase):
    def dish(s,r,h=.07):
        a=np.linspace(0,2*np.pi,240,endpoint=False);z=np.linspace(.005,h,12)
        wall=np.array([[r*math.cos(x)+.4,r*math.sin(x)+.3,zz] for x in a for zz in z])
        floor=np.array([[.4+rr*math.cos(x),.3+rr*math.sin(x),.005] for x in a[::6] for rr in (.0,.01,.02)])   # open cavity
        return np.r_[wall,floor]
    def test_torque_is_m_g_r_in_newton_metres_and_gates_the_rim_pinch(s):
        pl=(np.array([0,0,1.]),0.)
        # objects.yaml: Utana 0.35 kg, 13.2 cm; Threshold 0.20 kg, 11.2 cm; Ecoforms plant container 0.10 kg, 10.1 cm
        for m,diam,heavy in ((.35,.132,True),(.20,.112,False),(.10,.101,False)):
            f,C=GR.candidates(s.dish(diam/2),pl,'dish',mass=m)
            s.assertAlmostEqual(f['rim_torque'],m*9.81*f['rim_r'],places=12);s.assertAlmostEqual(f['rim_r'],diam/2,delta=.002)
            s.assertEqual(f['rim_heavy'],heavy);s.assertEqual(any(g['kind']=='rim' for g in C),not heavy)
        s.assertAlmostEqual(.35*9.81*.066,.2266,places=4);s.assertAlmostEqual((.11+.23)/2,GR.RIM_TORQUE_MAX,places=12)

class Kinematics(unittest.TestCase):
    @classmethod
    def setUpClass(c):c.A=ARM.Arm('l')
    def test_rotation_error_is_the_so3_log(s):
        rng=np.random.default_rng(9)
        for th in list(rng.uniform(0,np.pi,200))+[1e-9,1e-6,np.pi-1e-7,np.pi]:
            a=rng.normal(size=3);a/=np.linalg.norm(a);R0=GEO.rotation(rng.normal(size=3),rng.uniform(0,3))
            e=GEO.rotation_error(GEO.rotation(a,th)@R0,R0)
            if np.pi-th<1e-5:np.testing.assert_allclose(np.abs(e@a),th,atol=1e-6);np.testing.assert_allclose(np.cross(e,a),0,atol=1e-6)
            else:np.testing.assert_allclose(e,a*th,atol=1e-9)
    def test_ik_converges_inside_urdf_limits(s):
        ch=s.A.chain;rng=np.random.default_rng(10);n=0
        for _ in range(25):
            q=rng.uniform(ch.lower,ch.upper);T=ch.fk(q);seed=np.clip(q+rng.uniform(-.25,.25,len(q)),ch.lower,ch.upper)
            r=ch.ik(T,seed);n+=1
            s.assertTrue(np.all(r>=ch.lower-1e-12) and np.all(r<=ch.upper+1e-12))
            F=ch.fk(r);s.assertLess(np.linalg.norm(F[:3,3]-T[:3,3]),.002);s.assertLess(np.linalg.norm(GEO.rotation_error(T[:3,:3],F[:3,:3])),.015)
        s.assertEqual(n,25)
    def test_collision_lever_bound_holds(s):
        """certify inflates every sphere by levers @ |dq| / steps: the true displacement of each sphere centre over a
        joint step dq must never exceed levers @ |dq| (otherwise the discrete check could skip a contact)."""
        col=s.A.col;ch=s.A.chain;rng=np.random.default_rng(11)
        def centres(q):
            F=ch.frames_py(q);return np.array([F[sp.link][:3,:3]@sp.center+F[sp.link][:3,3] for sp in col.spheres])
        for _ in range(100):
            q=rng.uniform(ch.lower,ch.upper);dq=rng.uniform(-.2,.2,len(q));q2=np.clip(q+dq,ch.lower,ch.upper)
            dist=np.linalg.norm(centres(q2)-centres(q),axis=1);bound=col.levers@np.abs(q2-q)
            s.assertTrue(np.all(dist<=bound+1e-12),float(np.max(dist-bound)))
