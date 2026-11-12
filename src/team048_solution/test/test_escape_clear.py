"""certify_escape: a pair that is clear at the start stays clear on the whole segment, including one that
starts inside a single sample of the margin. A pair already inside the margin may not close by more than
ESCAPE_TOL. Self-pairs are still checked when every environment sphere is in that start set."""
import os,sys,unittest
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,HERE)
from team048_solution.kinematics import Chain,Joint
from team048_solution.planning import CollisionModel,Sphere,ESCAPE_TOL

def chain():
    z=np.array([0.,0.,1.]);o2=np.eye(4);o2[0,3]=.3
    j1=Joint('j1','l1',np.eye(4),z,-4.,4.,1.,True)
    j2=Joint('j2','tip',o2,z,-4.,4.,1.,True)
    return Chain([j1,j2],'base','tip',np.eye(4))

def tip_at(ch,q,center):
    T=ch.frames_py(q)['tip'];return T[:3,:3]@center+T[:3,3]

def model(spheres,env,margin=.015):
    return CollisionModel(chain(),spheres,env,audited_complete=True,margin=margin)

class ClearPair(unittest.TestCase):
    """Two revolute joints, one tip sphere. The arc -0.022 -> +0.022 passes a sphere that is clear at both
    ends of a sample but inside the margin in between, and starts close enough that the old check treated it
    as a contact (no-deeper by 1 mm) instead of a pair that must stay clear."""
    def setUp(self):
        self.center=np.array([.1,0.,0.]);self.rt=.02;self.re=.03;self.margin=.015
        self.thr=self.rt+self.re+self.margin
        self.q0=np.array([-.022,-.022]);self.q1=np.array([.022,.022])
        ch=chain();self.ch=ch
        pm=tip_at(ch,(self.q0+self.q1)/2,self.center);p0=tip_at(ch,self.q0,self.center)
        away=p0-pm;away/=np.linalg.norm(away)
        self.near=pm+(pm/np.linalg.norm(pm))*(self.thr-.0004)     # gap -0.4 mm at the arc midpoint
        self.contact=p0+away*(self.thr-.005)                      # 5 mm inside the margin at the start
    def build(self,env):
        tip=Sphere('tip',self.center,self.rt)
        return model([tip],[Sphere('base',c,self.re) for c in env],self.margin)
    def test_clear_pair_driven_into_the_margin_is_refused(self):
        M=self.build([self.contact,self.near]);q0,q1=self.q0,self.q1
        ge0,_=M.gaps(q0);steps=max(1,int(np.ceil(np.max(np.abs(q1-q0))/.003)));inf=M.levers@np.abs(q1-q0)/steps
        g_near=float(M._env_gap(q0,[1]).min());g_mid=float(M._env_gap((q0+q1)/2,[1]).min())
        self.assertGreater(g_near,0.);self.assertLessEqual(g_near,float(inf.max()))   # clear, but inside one sample
        self.assertLess(g_mid,0.);self.assertLess(float(ge0[:,0].min()),0.)           # the other sphere is a real contact
        self.assertFalse(M.valid(q0));self.assertTrue(M.valid(q1))
        self.assertFalse(M.certify_escape(q0,q1))
    def test_dip_between_samples_is_refused(self):
        # One sample from end to end: both endpoints clear of the near sphere, the midpoint is not.
        q0=np.array([-.3,-.3]);q1=np.array([.3,.3]);ch=self.ch
        pm=tip_at(ch,(q0+q1)/2,self.center);p0=tip_at(ch,q0,self.center)
        near=pm+(pm/np.linalg.norm(pm))*(self.thr-.01)
        away=p0-pm;away/=np.linalg.norm(away)
        contact=p0+away*(self.thr-.005)
        M=self.build([contact,near])
        self.assertGreater(float(M._env_gap(q0,[1]).min()),0.);self.assertGreater(float(M._env_gap(q1,[1]).min()),0.)
        self.assertLess(float(M._env_gap((q0+q1)/2,[1]).min()),0.)
        self.assertFalse(M.valid(q0));self.assertTrue(M.valid(q1))
        self.assertEqual(max(1,int(np.ceil(np.max(np.abs(q1-q0))/1.))),1)
        self.assertFalse(M.certify_escape(q0,q1,max_step=1.))
    def test_separating_a_true_contact_is_accepted(self):
        M=self.build([self.contact]);q0,q1=self.q0,self.q1
        g0=float(M.gaps(q0)[0].min());deepest=g0
        steps=max(1,int(np.ceil(np.max(np.abs(q1-q0))/.003)))
        for i in range(1,steps+1):
            deepest=min(deepest,float(M.gaps(q0+(q1-q0)*(i/steps))[0].min()))
        self.assertGreaterEqual(deepest,g0-ESCAPE_TOL)
        self.assertFalse(M.valid(q0));self.assertTrue(M.valid(q1));self.assertTrue(M.certify_escape(q0,q1))

class SelfPairs(unittest.TestCase):
    def test_self_collision_is_checked_when_every_env_sphere_is_in_the_start_set(self):
        # l1 sphere and the tip sphere meet at j2=pi and are clear at pi±0.8. The only environment sphere
        # is inside the margin at the start, so the start set contains the whole environment.
        ch=chain();margin=.005;rt=.005
        l1=Sphere('l1',np.array([.3,.05,0.]),rt);tip=Sphere('tip',np.array([0.,-.05,0.]),rt)
        def at(th):
            T=ch.frames_py(np.array([0.,th]))['tip'];return T[:3,:3]@tip.center+T[:3,3]
        q0=np.array([0.,np.pi-.8]);q1=np.array([0.,np.pi+.8]);p0=at(q0[1]);pmid=at(np.pi)
        thr=2*rt+margin
        self.assertGreater(np.linalg.norm(p0-l1.center),thr);self.assertLess(np.linalg.norm(pmid-l1.center),thr)
        away=p0-pmid;away/=np.linalg.norm(away)
        env=Sphere('base',p0+away*(rt+.01+margin-.005),.01)       # 5 mm inside the margin at the start
        M=CollisionModel(ch,[l1,tip],[env],audited_complete=True,margin=margin)
        self.assertFalse(M.valid(q0));self.assertTrue(M.valid(q1))
        ge0,gs0=M.gaps(q0);steps=max(1,int(np.ceil(np.max(np.abs(q1-q0))/.003)));inf=M.levers@np.abs(q1-q0)/steps
        self.assertTrue(np.any(ge0[:,0]<=inf))                         # the only env sphere is in the start set
        self.assertTrue(np.all(gs0>0))
        self.assertFalse(M.certify_escape(q0,q1))

if __name__=='__main__':unittest.main()
