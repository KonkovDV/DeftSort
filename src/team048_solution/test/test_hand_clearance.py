"""Open-hand clearance: the open Inspire hand (UBTECH meshes, tools/hand_open_points.py)
must not overlap the MEASURED object box at the grasp pose or along the approach. Sim evidence: v7+v8 side grasps of
mustard / Crayola / Crunch / Perricone touched the object before closing (fixture v7v8_side_at_grasp.json)."""
import os,sys,json,unittest
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,HERE)
import test_node_stub as TS  # noqa: F401  (installs the ROS stubs)
from team048_solution import arm as ARM, grasp as GR
import scene_fixture as SF
PRI=json.load(open(os.path.join(HERE,'..','team048_solution','data','priors.json')))
SIDE=json.load(open(os.path.join(HERE,'fixtures','v7v8_side_at_grasp.json')))['attempts']

def prior_box(name,u=(0.,1.),swap=False,z0=SF.TABLE_TOP,center=(.36,.2)):
    p=PRI[name];w,L=(p['w_max'],p['w_min']) if swap else (p['w_min'],p['w_max'])
    cyl=p['category']=='jar' and abs(p['w_min']-p['w_max'])<.006
    return dict(center=center,u=u,w=w,L=L,h=p['height'],z0=z0,cyl=cyl)

def side_pose(A,box,tilt=0.,sg=1.):
    h=box['h'];z=max(.5*h,min(.04,h-.01));a=np.array([1.,0,0])*np.cos(tilt)+np.array([0,0,-np.sin(tilt)])
    c=np.array([box['center'][0],box['center'][1],box['z0']+z]);return A.grasp_pose(c,sg*np.r_[box['u'],0.],a,'grip'),a

def overlaps(A,T,a,box,pre=.08):
    return ARM.grasp_clear_shift(A.side,T,a,box,pre=pre,margins=(0.,),su=(0.,),sr=(0.,)) is None

class HandPoints(unittest.TestCase):
    def test_both_hands_loaded(self):
        for s in 'lr':
            P,parts=ARM.hand_open_points(s)
            self.assertEqual(P.shape[1],3);self.assertEqual(len(P),len(parts));self.assertGreater(len(P),5000)
            for k in ('palm','thumb_4','index_2','middle_2','little_1'):self.assertIn(k,set(parts))
    def test_open_gap_is_asymmetric_about_grip_point(self):
        # grasp frame (x approach, y closing axis, z up): distal finger pads at y >= +0.028, thumb tip at y <= -0.046,
        # the proximal finger phalanges reach y = +0.006 -> the unshifted grip point is NOT centred in the real pad gap
        A=ARM.Arm('l');T=A.grasp_pose(np.zeros(3),np.array([0,1.,0]),np.array([1.,0,0]),'grip')
        P,parts=ARM.hand_open_points('l');W=(T[:3,:3]@P.T).T+T[:3,3]
        self.assertGreater(W[parts=='index_2',1].min(),.02);self.assertLess(W[parts=='thumb_4',1].max(),-.04)
        self.assertLess(W[parts=='index_1',1].min(),.01);self.assertLess(W[parts=='palm',0].max(),-.05)

class ClearShift(unittest.TestCase):
    A=ARM.Arm('l')
    def test_logged_contacts_are_predicted(self):
        """Every v7/v8 side attempt on mustard/Crayola/Crunch touched the object before closing (|F| >= 5 N or a finger
        bent back past 1.1); the old zero-shift grasp overlaps their measured box at every tilt and sign."""
        hit={a['obj'] for a in SIDE if np.linalg.norm(a['at_grasp_F'])>=5 or max(a['at_grasp_fing'][:5])>1.1}
        for n in ('006_mustard_bottle','Crayola_Bonus_64_Crayons','Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box'):
            self.assertIn(n,hit)
            b=prior_box(n)
            for t in (0,.35,.6):
                for sg in (1.,-1.):
                    T,a=side_pose(self.A,b,t,sg);self.assertTrue(overlaps(self.A,T,a,b),(n,t,sg))
    def test_round_jar_needs_no_shift(self):
        b=prior_box('Hyaluronic_Acid');T,a=side_pose(self.A,b)
        self.assertFalse(overlaps(self.A,T,a,b))
        c=ARM.grasp_clear_shift('l',T,a,b);self.assertEqual((c['dr'],c['margin']),(0.,max(ARM.CLEAR_MARGINS)))
    def test_shift_clears_long_box_and_keeps_pads_on_it(self):
        n='Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box';b=prior_box(n)
        for t in (0,.35):
            T,a=side_pose(self.A,b,t);c=ARM.grasp_clear_shift('l',T,a,b)
            self.assertIsNotNone(c);self.assertGreater(c['dr'],0.)
            self.assertLessEqual(c['dr'],max(ARM.SHIFT_R))
            T2=T.copy();T2[:3,3]+=c['shift'];self.assertFalse(overlaps(self.A,T2,a,b),t)
            P,parts=ARM.hand_open_points('l');W=(T2[:3,:3]@P.T).T+T2[:3,3];B=ARM.box_coords(W,b)
            for k in ARM.PAD_PARTS:
                m=B[parts==k].mean(0);self.assertTrue(0<m[2]<b['h'] and abs(m[1])<b['L']/2,k)
    def test_minimal_retreat_at_the_largest_feasible_margin(self):
        # clearance first (contact before closing ended 5 of the 6 heavy v7/v8 attempts), then the smallest lever
        n='Crayola_Bonus_64_Crayons';b=prior_box(n);T,a=side_pose(self.A,b)
        c=ARM.grasp_clear_shift('l',T,a,b);self.assertEqual(c['margin'],max(ARM.CLEAR_MARGINS))
        for r in [r for r in ARM.SHIFT_R if r<c['dr']]:
            self.assertIsNone(ARM.grasp_clear_shift('l',T,a,b,margins=(c['margin'],),sr=(r,)))
    def test_too_wide_object_is_rejected(self):
        b=dict(center=(.36,.2),u=(0.,1.),w=.09,L=.09,h=.12,z0=SF.TABLE_TOP,cyl=False);T,a=side_pose(self.A,b)
        self.assertIsNone(ARM.grasp_clear_shift('l',T,a,b))
    def test_missing_hand_data_keeps_old_pose(self):
        old=ARM.HAND_OPEN_PTS;ARM.HAND_OPEN_PTS='/nonexistent.json'
        try:
            b=prior_box('Crayola_Bonus_64_Crayons');T,a=side_pose(self.A,b);c=ARM.grasp_clear_shift('l',T,a,b)
            self.assertEqual(c['dr'],0.);self.assertTrue(np.all(c['shift']==0))
        finally:ARM.HAND_OPEN_PTS=old

class PlanAndCandidates(unittest.TestCase):
    def test_candidates_carry_measured_box(self):
        rng=np.random.default_rng(0);dx,dy,h=.043,.147,.127
        u=rng.uniform(-.5,.5,(3000,3))*[dx,dy,0];face=rng.integers(0,5,3000);u[:,2]=rng.uniform(0,h,3000)
        u[face==0,2]=h;u[face==1,0]=dx/2;u[face==2,0]=-dx/2;u[face==3,1]=dy/2;u[face==4,1]=-dy/2
        cl=u+[.4,.1,SF.TABLE_TOP];f,C=GR.candidates(cl,SF.plane(),'box')
        for g in C:
            self.assertIn('box',g);bx=g['box'];self.assertAlmostEqual(bx['w'],g['width'])
            self.assertAlmostEqual(sorted([bx['w'],bx['L']])[1],max(f['width'],f['length']))
    def test_plan_pick_with_box_is_clear(self):
        A=ARM.Arm('l');A.set_scene(SF.plane(),[SF.TRAYS['tray_left']])
        b=prior_box('Perricone_MD_AcylGlutathione_Eye_Lid_Serum',center=(.36,.25))
        c3=np.array([.36,.25,SF.TABLE_TOP+.054])
        p=A.plan_pick(c3,np.array([0,1.,0]),np.array([1.,0,0]),b['w'],obj=b)
        self.assertIsNotNone(p);self.assertIsNotNone(p['clear'])
        a=np.array([1.,0,0])*np.cos(p['tilt'])+np.array([0,0,-np.sin(p['tilt'])])
        self.assertFalse(ARM.grasp_clear_shift('l',A.tcp(p['q_grasp']),a,b,margins=(-.002,),su=(0.,),sr=(0.,)) is None)
        p0=A.plan_pick(c3,np.array([0,1.,0]),np.array([1.,0,0]),b['w'])
        self.assertIsNone(p0['clear'])        # call without a box: no shift

if __name__=='__main__':unittest.main()
