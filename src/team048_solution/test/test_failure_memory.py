"""Failure memory (grasp.demote_failed, node.remember_failure): an executed grasp that failed is tried after every other
candidate of the same object, so a deterministic planner does not replay it on an unchanged scene.
Fixture: rim_dishes.npz (v7 sort3 s102 start head frame, Ecoforms segment, measured plane)."""
import os,sys,inspect,unittest
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,HERE)
import test_node_stub as TS  # noqa: F401  (installs the ROS stubs)
from team048_solution import grasp as GR, node as N
Z=np.load(os.path.join(HERE,'fixtures','rim_dishes.npz'))
def seg(k):return Z[k+'_cloud'].astype(float),(Z[k+'_n'],float(Z[k+'_d']))
def att(g):return (g['kind'],np.asarray(g['center'],float)[:2],np.asarray(g['close_axis'],float)[:2])
def ids(C):return [id(g) for g in C]

class DemoteFailed(unittest.TestCase):
    def setUp(s):
        cl,pl=seg('ecoforms_v7s102');s.f,s.rim=GR.candidates(cl,pl,'dish')
    def test_no_memory_keeps_order(s):
        for C in (s.rim,):
            s.assertEqual(ids(GR.demote_failed(C,())),ids(C))
    def test_failed_rim_point_goes_last_others_keep_order(s):
        C=s.rim;s.assertEqual(len(C),7)
        D=GR.demote_failed(C,[att(C[0])])
        s.assertEqual(ids(D),ids(C[1:])+[id(C[0])])           # stable: the remaining six in their original order
    def test_rim_points_of_smallest_dish_never_merge(s):
        c=np.array([g['center'] for g in s.rim])
        d=np.linalg.norm(c[:,None]-c[None],axis=2)+np.eye(len(c))
        s.assertGreater(d.min(),GR.FAIL_SAME_R)                # 30 deg rim spacing (r0 0.050 m) > 2 cm
    def test_all_failed_restores_original_order(s):
        D=GR.demote_failed(s.rim,[att(g) for g in s.rim]);s.assertEqual(ids(D),ids(s.rim))
    def test_moved_object_within_radius_still_demoted(s):
        k,c,a=att(s.rim[0]);D=GR.demote_failed(s.rim,[(k,c+np.array([.012,.003]),a)])   # v9b Utana: 1.24 cm apart
        s.assertEqual(id(D[-1]),id(s.rim[0]))
    def test_other_kind_or_far_point_not_demoted(s):
        k,c,a=att(s.rim[0])
        s.assertEqual(ids(GR.demote_failed(s.rim,[('side',c,a)])),ids(s.rim))
        s.assertEqual(ids(GR.demote_failed(s.rim,[(k,c+np.array([.03,0.]),a)])),ids(s.rim))
    def test_side_perpendicular_axis_is_a_different_grasp(s):
        # synthetic 4 x 6 x 8 cm box at (0.4, 0.1) on the plane z = 0: two side candidates share the centre, axes 90 deg apart
        x,y,z=np.meshgrid(np.linspace(-.02,.02,9),np.linspace(-.03,.03,13),np.linspace(.0,.08,9))
        cl=np.c_[x.ravel()+.4,y.ravel()+.1,z.ravel()];C=GR.candidates(cl,(np.array([0.,0.,1.]),0.),'box')[1]
        side=[g for g in C if g['kind']=='side'];s.assertEqual(len(side),2)
        s.assertAlmostEqual(float(np.asarray(side[0]['close_axis'])@np.asarray(side[1]['close_axis'])),0.,6)
        g0=C[0];D=GR.demote_failed(C,[att(g0)])
        s.assertEqual(id(D[-1]),id(g0));s.assertEqual(len(D),len(C))
        other=[g for g in C if g is not g0 and np.allclose(g['center'],g0['center'])];s.assertTrue(other)
        pos=[i for i,h in enumerate(D) if h is not g0];s.assertEqual(ids([D[i] for i in pos]),[id(h) for h in C if h is not g0])   # same centre, other axis or kind: order kept
    def test_candidates_are_not_modified(s):
        before=[(g['kind'],np.asarray(g['center']).tolist(),g['score'],g['feasible']) for g in s.rim]
        GR.demote_failed(s.rim,[att(s.rim[0])])
        s.assertEqual(before,[(g['kind'],np.asarray(g['center']).tolist(),g['score'],g['feasible']) for g in s.rim])

class RememberFailure(unittest.TestCase):
    def test_only_executed_failures_are_stored(s):
        a=('rim',np.zeros(2),np.array([1.,0.]))
        m={};N.remember_failure(m,'x',True,a);N.remember_failure(m,'x',None,a);N.remember_failure(m,'x',False,None)
        s.assertEqual(m,{})
        N.remember_failure(m,'x',False,a);N.remember_failure(m,'x',False,a);s.assertEqual(len(m['x']),2)
    def test_node_wiring(s):
        run=inspect.getsource(N.Sol.run);pp=inspect.getsource(N.Sol.pick_place)
        s.assertIn('s.failed_grasps={}',run)
        s.assertLess(run.index('ok=s.pick_place('),run.index('remember_failure(s.failed_grasps,name,ok,s.last_attempt)'))
        s.assertLess(run.index('remember_failure('),run.index('if ok:handled.add(name)'))
        s.assertLess(pp.index('C=demote_failed('),pp.index('for side in order:'))
        # the attempt is recorded before the first motion of the arm toward the object
        s.assertLess(pp.index('s.last_attempt=('),pp.index("s.move_arm(side,p['q_pre'])"))

if __name__=='__main__':unittest.main()
