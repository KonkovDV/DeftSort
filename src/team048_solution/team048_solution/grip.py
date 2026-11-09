"""Grasp verification and hand helpers from robot sensors only (INTERFACES.md):
  /arm_6dof_{left,right}       geometry_msgs/WrenchStamped, external load on the hand, hand weight subtracted
  /inspire_hand/state/{l,r}    sensor_msgs/JointState: position 0..1 (1 open), velocity = fraction of max speed,
                               effort = fraction of the force limit
Verified on Kaggle v5 (check_interfaces): both wrists |F| = 0.00 N with an empty hand; hand state has 6 x pos/vel/eff.
Loaded-hand values are NOT measured yet, so the decision thresholds below are TODO and the monitor answers
None ("unknown") until they are set; callers then keep the previous behaviour (assume held).
"""
import numpy as np

# Wrist F/T is NOT used for the decision: v7 Kaggle (31 frames) shows empty-hand offsets of 0.05-5.5 N depending on
# arm pose and finger state (open hand ~4.5 N, closed fist ~0.1 N) -- as large as the objects' weight (0.8-4.2 N).
GRASP_DF_MIN_N = None
# Finger criterion, calibrated on v7 Kaggle 'lifted' frames (sort3 pool 3, 2026-11-09):
#   loaded (3: Ecoforms x2, Threshold): commanded-closing fingers stopped 0.67-0.98 above target, effort 1.00
#   empty  (7: 2 rim, 5 side): fingers within 0.02 of target, effort <= 0.334
# thresholds = midpoints between the classes.
FINGER_DEV_HELD = .35        # mean(position - target) over the commanded-closing fingers
FINGER_EFF_HELD = .67        # max effort over the commanded-closing fingers
FINGER_BLOCKED_MIN = FINGER_DEV_HELD   # backward-compatible names
EFFORT_MIN = FINGER_EFF_HELD
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


def finger_verdict(pos, eff, target, dev_min=FINGER_DEV_HELD, eff_min=FINGER_EFF_HELD):
    """True held / False empty / None (conflicting or missing). target = commanded hand vector (6, 1 open);
    'closing' fingers are the ones commanded below 0.5 among DOF 1-5."""
    if pos is None or eff is None or target is None:
        return None
    p = np.asarray(pos, float)[:5]; e = np.asarray(eff, float)[:5]; t = np.asarray(target, float)[:5]
    if len(p) < 5 or len(e) < 5:
        return None
    k = t < .5
    if not k.any():
        return None
    dev = float(np.mean(p[k]-t[k])); ef = float(np.max(np.abs(e[k])))
    a, b = dev >= dev_min, ef >= eff_min
    return True if (a and b) else (False if not (a or b) else None)


def fingers_settled(history, tol=SETTLE_VEL, n=SETTLE_COUNT):
    """history: list of velocity vectors (newest last). True when the last n all have every |v| < tol."""
    if len(history) < n:
        return False
    return all(np.all(np.abs(np.asarray(v, float)) < tol) for v in history[-n:])
