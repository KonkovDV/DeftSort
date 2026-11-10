"""Grasp verification and hand helpers from robot sensors only (INTERFACES.md):
  /arm_6dof_{left,right}       geometry_msgs/WrenchStamped, external load on the hand, hand weight subtracted
  /inspire_hand/state/{l,r}    sensor_msgs/JointState: position 0..1 (1 open), velocity = fraction of max speed,
                               effort = fraction of the force limit
Verified on Kaggle v5 (check_interfaces): both wrists |F| = 0.00 N with an empty hand; hand state has 6 x pos/vel/eff.
Loaded-hand values are NOT measured yet, so the decision thresholds below are TODO and the monitor answers
None ("unknown") until they are set; callers then keep the previous behaviour (assume held).
"""
import numpy as np

# TODO(calibrate from Kaggle sort runs with a lifted object): do not guess these.
GRASP_DF_MIN_N = None        # min |F - F_before_close| [N] after lift that means "object in hand"
FINGER_BLOCKED_MIN = None    # min mean(position) of fingers 1-4 after closing to 0 that means "blocked by an object"
EFFORT_MIN = None            # min max(effort) of fingers 1-5 that means "squeezing an object"
# Hand settings sent once at start (bodyctrl_msgs/srv/SetForce, SetSpeed; ratios 0..1, sim default 1.0 = 1000 g,
# full stroke in 0.8 s). TODO(calibrate): lower force only after slip tests; 1.0 keeps the simulator default.
HAND_FORCE_RATIO = 1.0
HAND_SPEED_RATIO = 1.0
SETTLE_VEL = .02             # |velocity| (fraction of max speed) below which a finger counts as stopped
SETTLE_COUNT = 3             # consecutive state messages (~20 Hz) with all fingers stopped


def force_vec(wrench):
    f = wrench.force
    return np.array([f.x, f.y, f.z], float)


class GraspMonitor:
    def __init__(self, df_min=None, blocked_min=None, effort_min=None):
        self.df_min = GRASP_DF_MIN_N if df_min is None else df_min
        self.blocked_min = FINGER_BLOCKED_MIN if blocked_min is None else blocked_min
        self.effort_min = EFFORT_MIN if effort_min is None else effort_min
        self.f0 = None

    @property
    def calibrated(self):
        return self.df_min is not None or (self.blocked_min is not None and self.effort_min is not None)

    def baseline(self, force):
        self.f0 = None if force is None else np.asarray(force, float)

    def verdict(self, force=None, hand_pos=None, hand_eff=None):
        """True held / False empty / None unknown (uncalibrated or missing data)."""
        votes = []
        if self.df_min is not None and force is not None and self.f0 is not None:
            votes.append(float(np.linalg.norm(np.asarray(force, float)-self.f0)) >= self.df_min)
        if self.blocked_min is not None and self.effort_min is not None and hand_pos is not None and hand_eff is not None:
            p = np.asarray(hand_pos, float); e = np.asarray(hand_eff, float)
            if len(p) >= 5 and len(e) >= 5:
                votes.append(bool(np.mean(p[:4]) >= self.blocked_min and np.max(np.abs(e[:5])) >= self.effort_min))
        if not votes:
            return None
        return any(votes)   # either sensor seeing the object counts; both empty => False


def fingers_settled(history, tol=SETTLE_VEL, n=SETTLE_COUNT):
    """history: list of velocity vectors (newest last). True when the last n all have every |v| < tol."""
    if len(history) < n:
        return False
    return all(np.all(np.abs(np.asarray(v, float)) < tol) for v in history[-n:])
