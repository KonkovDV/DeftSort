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
FINGER_DEV_HELD = .35        # mean(position - target) over the commanded-closing fingers (target = fully closed)
FINGER_EFF_HELD_R1 = .67     # max effort, midpoint 0.334 (empty) .. 1.00 (loaded) -- valid at force ratio 1.0 ONLY
FINGER_EFF_EMPTY_R1 = .334   # v7 empty-hand max effort at force ratio 1.0 (upper end of the 7 empty frames)
FINGER_BLOCKED_MIN = FINGER_DEV_HELD   # backward-compatible names
EFFORT_MIN = FINGER_EFF_HELD_R1
FINGER_EFF_HELD = FINGER_EFF_HELD_R1

# ---- Hand force / speed / closing, organiser answer #648 (2026-11-06) --------------------------------------------
# "Full-force squeezing makes the physics push the object out; lower /inspire_hand/set_force (e.g. 0.3-0.5 instead of
#  1.0), target slightly 'past' the object, lower the finger speed, rotate the thumb first, then close the rest."
# INTERFACES.md: set_force ratio 1.0 = 1000 g (default); set_speed 1.0 = full stroke in 0.8 s, 0.5 = 1.6 s.
# Defaults (env overrides for sim sweeps, TEAM048_HAND_FORCE / _SPEED / _CLOSE_PAST):
#   force 0.4  : middle of the organiser's 0.3-0.5 range. Heaviest pool-3 object 0.51 kg (004_sugar_box, objects.yaml).
#                TODO(sim): sweep 0.3/0.4/0.5 on the heaviest objects; TRACE logs slips (lifted vs carry verdict).
#   thumb rotation (DOF 6) keeps 1.0: it positions the thumb into opposition and does not squeeze the object; a weak
#                rotation actuator would let the contact force push the thumb out of opposition. TODO(sim) confirm.
#   speed 0.5  : the organiser's example value (1.6 s full stroke) -> wait_fingers timeouts scale with it.
#   close past : phase 1 closes towards the fully-closed target until the fingers stop on the object (contact
#                position p_c measured on /inspire_hand/state); phase 2 re-targets p_c - CLOSE_PAST, i.e. "slightly past
#                the object" by a MEASURED amount instead of an uncalibrated width->position map.
def _env_ratio(name, default, lo, hi):
    import os
    try:
        v = float(os.environ.get(name, default))
    except ValueError:
        v = default
    return min(hi, max(lo, v))
HAND_FORCE_RATIO = _env_ratio('TEAM048_HAND_FORCE', .4, .02, 1.)   # sim clamps to [0.02, 1] (robot_io.set_force)
HAND_FORCE_THUMB_ROT = 1.0
HAND_SPEED_RATIO = _env_ratio('TEAM048_HAND_SPEED', .5, .01, 1.)   # sim clamps to [0.01, 1] (robot_commands.srv_speed)
CLOSE_PAST = _env_ratio('TEAM048_CLOSE_PAST', .15, 0., .5)         # fraction of the finger stroke past measured contact
FULL_STROKE_S = .8           # INTERFACES.md: speed 1.0 = full stroke in 0.8 s
THUMB_ROT_TOL = .05          # DOF 6 position within this of its target = thumb in opposition (0.02 noise in v7 frames)
EMPTY_CONTACT_MAX = .10      # phase-1 mean(p - closed target) below this = closed on nothing. v7 'lifted' frames:
                             # empty rim/side <= 0.03, loaded rim >= 0.45 -> 0.10 sits near the empty class on purpose
                             # (a false 'empty' abandons a real grasp; a false 'held' only costs one lift + re-check)


def force_vector(ratio=None, thumb_rot=None):
    """6 SetForce ratios (DOF 1..6 = little, ring, middle, index, thumb bend, thumb rotation)."""
    r = HAND_FORCE_RATIO if ratio is None else ratio
    return [r]*5+[HAND_FORCE_THUMB_ROT if thumb_rot is None else thumb_rot]


def finger_stroke_time(speed=None):
    """Seconds for a full finger stroke at the configured speed ratio (INTERFACES.md)."""
    return FULL_STROKE_S/(HAND_SPEED_RATIO if speed is None else speed)


def effort_held_threshold(ratio=None, empty_r1=FINGER_EFF_EMPTY_R1, loaded=1.0):
    """Effort threshold at force ratio r. The simulator publishes effort = |tau| / (lim0 * r), clipped to 1
    (hack_sim/robot_io.py finger_effort): a finger BLOCKED by an object saturates at 1.0 for any r, while the torque of
    an unloaded finger is independent of r, so its normalised effort scales as 1/r (ASSUMPTION, to be confirmed by the
    next sim run's TRACE). Threshold = midpoint between min(1, empty_r1/r) and the loaded level. At r=0.4 this is
    0.92 with only a 0.165 gap -> the effort vote is weak at low force; the position criteria carry the decision."""
    r = HAND_FORCE_RATIO if ratio is None else ratio
    e = min(1., empty_r1/max(r, 1e-3))
    return (e+loaded)/2


def thumb_rotated(pos, target_rot=0., tol=THUMB_ROT_TOL):
    """DOF 6 (thumb rotation) reached its target (None if no data)."""
    if pos is None or len(pos) < 6:
        return None
    return bool(abs(float(pos[5])-target_rot) <= tol)


JAM_TOL = .02                # a finger reading above 1 + JAM_TOL is bent BACK past fully open by a contact ("jammed").
                             # v7+v8 'closed' frames: 8 of 15 attempts had such fingers (up to 1.81); none are pinching.
TARGET_MAX = 1.0             # hand command range [0, 1] (1 = open, INTERFACES.md)


def jammed(pos, closed_target=None, tol=JAM_TOL):
    """DOF numbers (1..5) of fingers bent back past fully open (position > 1 + tol). With closed_target given, only
    the commanded-closing fingers (target < .5) are reported."""
    if pos is None:
        return []
    p = np.asarray(pos, float)
    t = None if closed_target is None else np.asarray(closed_target, float)
    return [k+1 for k in range(min(5, len(p))) if p[k] > 1.+tol and (t is None or t[k] < .5)]


def contact_target(contact_pos, closed_target, past=None, tol=JAM_TOL):
    """Phase-2 command: for every commanded-closing finger (closed_target < .5, DOF 1-5) target = min(1, max(closed_target,
    p_contact - past)); other DOFs unchanged. 'Slightly past' the measured contact, never beyond the closed target and
    never above 1.0 (v8 fix 3: 7 of 15 v7+v8 attempts produced targets up to 1.66 = 'open further' for jammed fingers).
    Jammed fingers (contact position > 1 + tol, bent back by a contact, not pinching) are NOT re-targeted: they keep
    the closed target."""
    past = CLOSE_PAST if past is None else past
    p = np.asarray(contact_pos, float); t = np.asarray(closed_target, float).copy()
    for k in range(min(5, len(t), len(p))):
        if t[k] < .5 and p[k] <= 1.+tol:
            t[k] = min(TARGET_MAX, max(t[k], p[k]-past))
    return [float(min(TARGET_MAX, x)) for x in t]


# ---- Held rule (v8 fix 2, evidence: fixes/V8_ANALYSIS_RU.md section 5, fixtures test/fixtures/v7v8_grasp_traces.json)
# Held iff at least one commanded-closing finger (DOF 1-5) deviates >= HELD_DEV from its CURRENT command and its
# effort is >= HELD_EFF. 15/15 correct on the v7+v8 closed/lifted pairs (4 held, 11 empty, force 1.0), where
# past_verdict misses 2 of 4 held (v8 Perricone: the fingers travelled 0.46-0.76 past contact, only the jammed thumb
# held) and finger_verdict answers None for Perricone. Blocked fingers saturate the normalised effort (1.0 at any
# force ratio, sim robot_io finger_effort); unloaded fingers in v7/v8 read <= 0.334 at ratio 1.0.
HELD_DEV = .10
HELD_EFF = .9


def held_rule(pos, eff, command, closed_target, dev_min=HELD_DEV, eff_min=HELD_EFF):
    """True held / False empty / None (missing data). command = the hand vector currently commanded (phase-2 target
    or closed target); closed_target selects the closing fingers (< .5)."""
    if pos is None or eff is None or command is None or closed_target is None:
        return None
    p = np.asarray(pos, float); e = np.asarray(eff, float)
    c = np.asarray(command, float); t0 = np.asarray(closed_target, float)
    if min(len(p), len(e), len(c), len(t0)) < 5:
        return None
    k = [i for i in range(5) if t0[i] < .5]
    if not k:
        return None
    return bool(any(abs(p[i]-c[i]) >= dev_min and abs(e[i]) >= eff_min for i in k))


# ---- Guarded approach (v8 fix 4): v8 attempts 1-4 had fingers bent by contact BEFORE the close command
# (Utana middle/index -0.06/-0.10, Threshold index 0.16 with 16 N, Crunch 1.06-1.46 with 14.6 N, mustard little 1.60).
# Only the fingers that will CLOSE for this grasp kind are guarded (closed_target < .5): in both held v7 Ecoforms rim
# grasps the (non-closing) ring finger read 1.14/1.17 at the grasp pose, so guarding every finger would have aborted
# the only v7 success. With the closing set: 0 of 4 held v7+v8 grasps trip, 11 of 12 empty ones trip before closing
# (test/fixtures/v7v8_grasp_traces.json).
GUARD_DEV = .10


def approach_deviation(pos, command, closed_target=None, dofs=5):
    """(max |position - command| over the guarded DOFs, list of guarded DOF numbers above GUARD_DEV) or (None, []).
    Guarded DOFs: 1..dofs, restricted to the closing fingers (closed_target < .5) when closed_target is given."""
    if pos is None or command is None:
        return None, []
    p = np.asarray(pos, float)[:dofs]; c = np.asarray(command, float)[:dofs]
    if len(p) < dofs or len(c) < dofs:
        return None, []
    k = list(range(dofs)) if closed_target is None else [i for i in range(dofs) if float(closed_target[i]) < .5]
    if not k:
        return None, []
    d = np.abs(p-c)
    return float(max(d[i] for i in k)), [i+1 for i in k if d[i] > GUARD_DEV]


# ---- Finger stall by POSITION (v8 fix 5): the velocity criterion fired in 3 of 16 v7+v8 closes.
STALL_TOL = .005             # max |delta position| of every finger over the window
STALL_WINDOW = .15           # s sim


def fingers_stalled(hist, now, since=None, tol=STALL_TOL, window=STALL_WINDOW, dofs=6):
    """hist: list of (t, positions) (newest last, sim time). True when the samples of the last `window` seconds (all
    newer than `since`, the command time) span >= 80 % of the window, number >= 2, and every finger moved < tol."""
    if not hist:
        return False
    lo = now-window
    w = [(t, np.asarray(p, float)[:dofs]) for t, p in hist if t >= lo and (since is None or t >= since)]
    if len(w) < 2 or w[-1][0]-w[0][0] < .8*window:
        return False
    P = np.stack([p for _, p in w])
    return bool(np.all(P.max(0)-P.min(0) < tol))


def contact_empty(contact_pos, closed_target, empty_max=EMPTY_CONTACT_MAX):
    """True if in phase 1 the closing fingers ran (almost) to the closed target, i.e. nothing between them."""
    p = np.asarray(contact_pos, float)[:5]; t = np.asarray(closed_target, float)[:5]
    k = t < .5
    if not k.any() or len(p) < 5:
        return None
    return bool(np.mean(p[k]-t[k]) < empty_max)


def past_verdict(pos, target2, closed_target, past=None):
    """LOG-ONLY since v8 fix 2 (misses 2 of 4 v7+v8 held grasps). Held check after phase 2 (target2 = contact - past). Held: the fingers are still blocked short of target2 by
    at least half of `past` (the object holds them); empty: they reached target2 (object gone) or ran to the closed
    target. Threshold past/2 = 0.075 for past 0.15, vs 0.02 position noise of an unloaded finger in v7."""
    past = CLOSE_PAST if past is None else past
    if pos is None or target2 is None or closed_target is None:
        return None
    p = np.asarray(pos, float)[:5]; t2 = np.asarray(target2, float)[:5]; t0 = np.asarray(closed_target, float)[:5]
    if len(p) < 5:
        return None
    k = t2 > t0+1e-6          # only the fingers that were blocked in phase 1 and re-targeted to contact - past
    if not k.any() or past <= 1e-6:
        return None
    return bool(np.mean(p[k]-t2[k]) >= past/2)


SETTLE_VEL = .02             # |velocity| (fraction of max speed) below which a finger counts as stopped
SETTLE_COUNT = 3             # consecutive state messages (~20 Hz) with all fingers stopped


def force_vec(wrench):
    f = wrench.force
    return np.array([f.x, f.y, f.z], float)


class GraspMonitor:
    def __init__(self, df_min=None, blocked_min=None, effort_min=None):
        self.df_min = GRASP_DF_MIN_N if df_min is None else df_min
        self.blocked_min = FINGER_BLOCKED_MIN if blocked_min is None else blocked_min
        self.effort_min = effort_held_threshold() if effort_min is None else effort_min
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


def finger_verdict(pos, eff, target, dev_min=FINGER_DEV_HELD, eff_min=None):
    """True held / False empty / None (conflicting or missing). target = commanded hand vector (6, 1 open);
    'closing' fingers are the ones commanded below 0.5 among DOF 1-5. eff_min defaults to the force-ratio-normalised
    threshold effort_held_threshold() (v7 calibration at ratio 1.0 rescaled; see its docstring)."""
    eff_min = effort_held_threshold() if eff_min is None else eff_min
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
