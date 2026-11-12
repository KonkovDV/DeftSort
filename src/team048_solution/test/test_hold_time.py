"""#595 guarantee: lift only if the worst-case in-hand time (every move_arm / waist / release wait timing out, from
the node's own timeouts) fits, in addition to the typical-time model place_time + SAFETY_S."""
import os,sys,json,unittest
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,os.path.join(HERE,'..'));sys.path.insert(0,HERE)
import test_node_stub as TS   # noqa
from test_node_stub import N
from team048_solution import arm as ARM, budget as BUD
P8={x['phase']:x for x in json.load(open(os.path.join(HERE,'fixtures','v8_perricone_place.json')))['phases']}
q=lambda k:np.array(P8[k]['q'])

class HoldBound(unittest.TestCase):
    def test_bound_covers_measured_v8_chain(self):
        b=N.hold_time_bound(q('closed'),q('lifted'),(q('over_tray'),None,q('carry')),0.)
        self.assertGreaterEqual(b,P8['released']['t']-P8['closed']['t'])     # 12.05 s measured
    def test_move_bound_dominates_model_and_code_timeouts(self):
        for d in (0.,.05,.3,1.2):
            a=np.zeros(7);b=np.r_[d,np.zeros(6)]
            self.assertGreater(N.move_bound(a,b),N.move_time(a,b))
            self.assertGreaterEqual(N.move_bound(a,b),d/.8*1.6+N.MOVE_T_PAD+2*N.REFINE_T)   # move_arm: T + 2 refines
    def test_yaw_adds_the_waist_timeout(self):
        pl=(q('over_tray'),None,q('carry'))
        self.assertAlmostEqual(N.hold_time_bound(q('closed'),q('lifted'),pl,1.1)-N.hold_time_bound(q('closed'),q('lifted'),pl,0.),N.WAIST_TMAX)
    def test_strict_bound_binds_when_post_release_moves_are_short(self):
        # the typical model counts the retreat home after the release; with a short way home it can accept a lift whose
        # worst-case hold does not fit -> lift_allowed refuses it
        A=ARM.Arm('r');pl=(q('over_tray'),None,q('carry'))
        need=N.place_time(A,q('lifted'),pl,1.1,q_from=q('closed'));hold=N.hold_time_bound(q('closed'),q('lifted'),pl,1.1)
        rem=need+BUD.SAFETY_S
        self.assertEqual(N.lift_allowed(rem,need,hold,BUD.SAFETY_S),rem>=hold+N.HOLD_MARGIN_S)
        self.assertFalse(N.lift_allowed(hold+N.HOLD_MARGIN_S-.01,0.,hold,BUD.SAFETY_S))
        self.assertTrue(N.lift_allowed(max(need+BUD.SAFETY_S,hold+N.HOLD_MARGIN_S),need,hold,BUD.SAFETY_S))
    def test_constants_are_the_ones_the_code_waits_on(self):
        import inspect
        src=inspect.getsource(N.Sol.move_arm)
        self.assertIn('MOVE_T_PAD',src);self.assertIn('REFINE_T',src)
        self.assertEqual(inspect.signature(N.Sol.waist).parameters['tmax'].default,N.WAIST_TMAX)
        self.assertIn('tmax=RELEASE_WAIT',inspect.getsource(N.Sol.open_hand))
        self.assertIn('open_hand',inspect.getsource(N.Sol.pick_place))
class PlaceReapproachBound(unittest.TestCase):
    """The off-tray branch of pick_place (one more move_arm(q_place) and one more A.correct per PLACE_RETRIES) is inside
    hold_time_bound, with the node's own timeouts; the lift margins stay HOLD_MARGIN_S and SAFETY_S."""
    pl=(q('over_tray'),None,q('carry'))
    def bound(s,retries,yaw=0.):
        old=N.PLACE_RETRIES;N.PLACE_RETRIES=retries
        try:return N.hold_time_bound(q('closed'),q('lifted'),s.pl,yaw)
        finally:N.PLACE_RETRIES=old
    def corr(s):
        return (.15+2*N.MOVE_RES)/.8*1.6+N.MOVE_T_PAD+N.REFINE_T
    def test_each_retry_adds_one_place_move_and_one_correction(s):
        d=np.r_[.15+N.MOVE_RES,np.zeros(6)]   # start: the corrected command (within max_step 0.15 of q_place) + residual
        step=N.move_bound(q('over_tray')+d,q('over_tray'))+s.corr()
        for yaw in (0.,1.1):
            s.assertAlmostEqual(s.bound(1,yaw)-s.bound(0,yaw),step)
            s.assertAlmostEqual(s.bound(2,yaw)-s.bound(1,yaw),step)
    def test_bound_is_the_sum_of_the_coded_timeouts(s):
        seq=[q('closed'),q('lifted'),q('carry'),q('over_tray')]
        moves=sum(N.move_bound(a,b) for a,b in zip(seq,seq[1:]))+.3
        reapproach=(.15+2*N.MOVE_RES)/.8*1.6+N.MOVE_T_PAD+2*N.REFINE_T
        want=moves+s.corr()+N.PLACE_RETRIES*(reapproach+s.corr())+2*N.RELEASE_WAIT
        s.assertAlmostEqual(N.hold_time_bound(q('closed'),q('lifted'),s.pl,0.),want)
        s.assertAlmostEqual(N.hold_time_bound(q('closed'),q('lifted'),s.pl,1.1),want+N.WAIST_TMAX)
    def test_bound_mirrors_the_place_loop(s):
        import inspect
        src=inspect.getsource(N.Sol.pick_place)
        s.assertIn('range(1+PLACE_RETRIES)',src);s.assertIn('max_step=.15',src)
        s.assertEqual(N.PLACE_RETRIES,1)
    def test_way_back_is_the_sum_of_the_coded_timeouts(s):
        qm,ql,qc=q('over_tray'),q('lifted'),q('carry')
        s.assertAlmostEqual(N.return_hold_bound(qm,ql,None,0.),N.move_bound(qm,ql)+2*N.RELEASE_WAIT)
        s.assertAlmostEqual(N.return_hold_bound(qm,ql,qc,0.),N.move_bound(qm,qc)+N.move_bound(qc,ql)+2*N.RELEASE_WAIT)
        s.assertAlmostEqual(N.return_hold_bound(qm,ql,None,1.1),2*N.move_bound(qm,ql)+N.WAIST_TMAX+2*N.RELEASE_WAIT)
        s.assertAlmostEqual(N.return_hold_bound(qm,ql,qc,1.1),N.move_bound(qm,qc)+N.move_bound(qc,ql)+N.WAIST_TMAX+2*N.RELEASE_WAIT)
        s.assertGreater(N.return_hold_bound(qm,ql,qc,0.),N.move_time(qm,qc)+N.move_time(qc,ql))
    def test_margins_unchanged(s):
        s.assertEqual(N.HOLD_MARGIN_S,1.);s.assertEqual(BUD.SAFETY_S,8.)
        hold=N.hold_time_bound(q('closed'),q('lifted'),s.pl,0.)
        s.assertFalse(N.lift_allowed(hold+N.HOLD_MARGIN_S-.01,0.,hold,BUD.SAFETY_S))
        s.assertTrue(N.lift_allowed(hold+N.HOLD_MARGIN_S,0.,hold,BUD.SAFETY_S))
if __name__=='__main__':unittest.main()
