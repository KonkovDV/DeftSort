"""Time budget per layout (SCORING.md: the limit is counted in SIM time from the moment run.sh starts; startup counts).

Rule: never start a pick that cannot be finished before the limit, because an object still in the hand (or set down
outside a tray) at the end scores -10, while an untouched object scores 0.
"""
import math

# TODO(calibrate from Kaggle v6+ sort logs): measured pick+place cycle and release/retreat times.
FIRST_CYCLE_S = 35.0   # prior for one pick+place cycle until the first real cycle is measured
PLACE_TAIL_S = 12.0    # transport to tray + release + retreat, added to the planned pick duration
SAFETY_S = 8.0         # final go-home + /sim/task/done margin
LIMITS = {'sort2': 270.0, 'sort3': 405.0}


class TimeBudget:
    def __init__(self, limit, t0, first_cycle=FIRST_CYCLE_S, safety=SAFETY_S, alpha=.5):
        if not (math.isfinite(limit) and limit > 0 and math.isfinite(t0)):
            raise ValueError('invalid budget')
        if not 0 < alpha <= 1:
            raise ValueError('alpha in (0,1]')
        self.limit = float(limit); self.t0 = float(t0)
        self.cycle = float(first_cycle); self.safety = float(safety); self.alpha = alpha
        self.cycles = []

    def elapsed(self, now):
        return now - self.t0

    def remaining(self, now):
        return self.limit - (now - self.t0)

    def record_cycle(self, seconds):
        """Update the per-object cycle estimate; the max of EMA and last cycle keeps it pessimistic."""
        if not (math.isfinite(seconds) and seconds > 0):
            return
        self.cycles.append(float(seconds))
        ema = self.alpha*seconds + (1-self.alpha)*self.cycle
        self.cycle = max(ema, float(seconds)) if seconds > self.cycle else ema   # rise at once, decay slowly

    def can_start(self, now, planned=None):
        """True if one more cycle (planned duration + place tail, or the estimate) still fits with the safety margin."""
        need = (planned + PLACE_TAIL_S) if planned is not None else self.cycle
        return self.remaining(now) >= need + self.safety


def start_time(sim_now_at_first_clock, wall_at_first_clock, wall_at_import, rtf_max=1.0):
    """Sim time at which run.sh effectively started.

    Process startup before rclpy sees /clock is not visible in sim time; bound it by the wall time elapsed since
    import scaled by an upper bound of the real-time factor (rtf_max=1: the sim is not faster than real time;
    Kaggle T4 measured ~24 Hz physics, i.e. slower). Pessimistic = earlier t0 = less remaining time.
    """
    lag = max(0.0, wall_at_first_clock - wall_at_import)
    return sim_now_at_first_clock - rtf_max*lag


def launch_age_s(marker='run.sh', max_up=6):
    """Seconds since the launch of this solution: the start of the oldest ancestor process (up to max_up levels) whose
    command line contains `marker` (run.sh), else of this process. Linux /proc only; None if unavailable.
    SCORING.md "Уточнения" 4 Nov / #638: the limit counts from the moment run.sh is started; start-up is included."""
    import os
    try:
        tck = os.sysconf('SC_CLK_TCK')
        now = time_boot()
        pid, best = os.getpid(), None
        for _ in range(max_up+1):
            with open(f'/proc/{pid}/stat') as f:
                st = f.read()
            rest = st[st.rindex(')')+2:].split()
            start = int(rest[19])/tck            # field 22 (starttime), seconds since boot
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read().replace(b'\0', b' ').decode(errors='ignore')
            if best is None or marker in cmd:
                best = start
            ppid = int(rest[1])
            if ppid <= 1:
                break
            pid = ppid
        return max(0.0, now-best) if best is not None else None
    except Exception:
        return None


def time_boot():
    import time
    return time.clock_gettime(time.CLOCK_BOOTTIME)
