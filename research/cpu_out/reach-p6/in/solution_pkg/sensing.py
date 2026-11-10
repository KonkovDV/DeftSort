"""Frame selection for RGB-D (I8). Kaggle v5: colour and depth of cam_head carry the SAME stamp (70.039998434 s),
both 640x360, depth aligned to colour, so pairing by stamp is exact in the simulator; tolerance covers real robots."""
from collections import deque


def stamp_s(msg):
    st = msg.header.stamp
    return st.sec+st.nanosec*1e-9


class PairBuffer:
    def __init__(self, n=6):
        self.rgb = deque(maxlen=n); self.depth = deque(maxlen=n)

    def add_rgb(self, m):
        self.rgb.append(m)

    def add_depth(self, m):
        self.depth.append(m)

    def pair(self, not_before, tol=.02):
        """Newest (rgb, depth) with |dt| <= tol and both stamps >= not_before (frames taken after the arms stopped)."""
        best = None
        for d in reversed(self.depth):
            td = stamp_s(d)
            if td < not_before:
                continue
            for r in reversed(self.rgb):
                tr = stamp_s(r)
                if tr >= not_before and abs(tr-td) <= tol:
                    if best is None or td > best[2]:
                        best = (r, d, td)
                    break
        return None if best is None else best[:2]
