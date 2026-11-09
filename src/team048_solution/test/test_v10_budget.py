"""v10 #595 guarantee: lift only if the WORST-CASE in-hand time (every move_arm / waist / release wait timing out, from
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
        self.assertIn('tmax=RELEASE_WAIT',inspect.getsource(N.Sol.pick_place))
if __name__=='__main__':unittest.main()
