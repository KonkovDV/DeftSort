"""RGB-D frame pairing. Run v5: colour and depth of cam_head carry the SAME stamp (70.039998434 s),
both 640x360, depth aligned to colour, so pairing by stamp is exact in the simulator; tolerance covers real robots."""
from collections import deque
import threading


def stamp_s(msg):
    """Header stamp of a ROS message in seconds."""
    st = msg.header.stamp
    return st.sec+st.nanosec*1e-9


class PairBuffer:
    """Thread-safe ring buffers of colour and depth messages for stamp-matched RGB-D pairs."""
    # The image callbacks run on executor threads (MultiThreadedExecutor) while pair() runs on the task thread.
    # Iterating a deque that another thread appends to raises 'RuntimeError: deque mutated during iteration'
    # (test_run_robustness stress test: 258 errors in 274739 unlocked calls); in observe() that error ended the whole run.
    # pair() therefore iterates snapshots taken under a lock.
    def __init__(self, n=6):
        self.rgb = deque(maxlen=n); self.depth = deque(maxlen=n); self.lock = threading.Lock()

    def add_rgb(self, m):
        """Append a colour image message."""
        with self.lock:
            self.rgb.append(m)

    def add_depth(self, m):
        """Append a depth image message."""
        with self.lock:
            self.depth.append(m)

    def snapshot(self):
        """Copies of both buffers taken under the lock."""
        with self.lock:
            return list(self.rgb), list(self.depth)

    def pair(self, not_before, tol=.02):
        """Newest (rgb, depth) with |dt| <= tol and both stamps >= not_before (frames taken after the arms stopped)."""
        best = None
        rgb, depth = self.snapshot()
        for d in reversed(depth):
            td = stamp_s(d)
            if td < not_before:
                continue
            for r in reversed(rgb):
                tr = stamp_s(r)
                if tr >= not_before and abs(tr-td) <= tol:
                    if best is None or td > best[2]:
                        best = (r, d, td)
                    break
        return None if best is None else best[:2]
