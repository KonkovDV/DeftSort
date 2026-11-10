"""Sensor-only plane removal and organized components. Classification abstains by default."""
from dataclasses import dataclass
import numpy as np
from .geometry import finite
from . import _accel


@dataclass
class Detection:
    centroid: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    pixels: np.ndarray
    features: np.ndarray
    category: str | None = None
    reason: str = 'no calibrated classifier'
    error_m: float = 0.
    feature_error: np.ndarray | None = None
    error_axes: np.ndarray | None = None
    anchor_error_axes: np.ndarray | None = None
    region_diagnostics: dict | None = None


def table_plane(points, valid, iterations=120, tolerance=.006, min_inliers=100, seed=0):
    """RANSAC horizontal plane. Same RNG stream and candidate arithmetic as table_plane_py; only the inlier COUNTING
    of all candidates is batched (Rust kernel, else one chunked BLAS product); the winner is the first candidate with
    the maximal count (== the reference's strict '>' scan) and its inlier set is recomputed exactly as before."""
    p = np.asarray(points)[np.asarray(valid, dtype=bool)]
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < min_inliers:
        raise ValueError('insufficient plane support')
    rng = np.random.default_rng(seed)  # algorithm RNG, never layout seed
    cands = []
    for _ in range(iterations):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        normal = np.cross(b-a, c-a); length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        if normal[2] < 0:
            normal *= -1
        if normal[2] < .96:  # table horizontal in pelvis, never floor by assumption alone
            continue
        cands.append((normal, -normal@a))
    best = None; score = 0
    if cands:
        planes = np.array([np.r_[n, o] for n, o in cands])
        counts = None
        if _accel.enabled():
            try: counts = np.asarray(_accel.plane_inlier_counts(p, planes, tolerance), dtype=np.int64)
            except BaseException as e: _accel.guard(e); _accel.failed('plane_inlier_counts', e)
        if counts is None:
            counts = np.zeros(len(cands), np.int64); Nm = planes[:, :3].T
            for k in range(0, len(p), 65536):
                counts += (np.abs(p[k:k+65536]@Nm+planes[:, 3]) < tolerance).sum(axis=0)
        k = int(np.argmax(counts))
        if counts[k] > 0:
            normal, offset = cands[k]; inliers = abs(p@normal+offset) < tolerance
            score = int(inliers.sum()); best = (normal, offset, inliers)
    return _refine_plane(p, best, score, min_inliers)


def _refine_plane(p, best, score, min_inliers):
    if best is None or score < min_inliers:
        raise ValueError('no supported horizontal plane')
    normal, offset, inliers = best
    cloud = p[inliers]; center = cloud.mean(axis=0)
    _, _, V = np.linalg.svd(cloud-center, full_matrices=False)
    normal = V[-1]
    if normal[2] < 0:
        normal *= -1
    if normal[2] < .96:
        raise ValueError('bad refined plane')
    return normal, -normal@center


def table_plane_py(points, valid, iterations=120, tolerance=.006, min_inliers=100, seed=0):
    p = np.asarray(points)[np.asarray(valid, dtype=bool)]
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < min_inliers:
        raise ValueError('insufficient plane support')
    rng = np.random.default_rng(seed)  # algorithm RNG, never layout seed
    best = None; score = 0
    for _ in range(iterations):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        normal = np.cross(b-a, c-a); length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        if normal[2] < 0:
            normal *= -1
        if normal[2] < .96:  # table horizontal in pelvis, never floor by assumption alone
            continue
        offset = -normal@a
        inliers = abs(p@normal+offset) < tolerance
        if inliers.sum() > score:
            score = inliers.sum(); best = (normal, offset, inliers)
    if best is None or score < min_inliers:
        raise ValueError('no supported horizontal plane')
    normal, offset, inliers = best
    cloud = p[inliers]; center = cloud.mean(axis=0)
    _, _, V = np.linalg.svd(cloud-center, full_matrices=False)
    normal = V[-1]
    if normal[2] < 0:
        normal *= -1
    if normal[2] < .96:
        raise ValueError('bad refined plane')
    return normal, -normal@center


def segment(points, rgb, valid, plane, roi, min_pixels=25, max_jump=.025, min_height=.012, max_height=.35):
    """Depth-continuity components above the plane. The flood fill (seed order, stack order, neighbour order and
    jump test identical to segment_py) runs in Rust when available; features are computed exactly as before."""
    p = np.asarray(points); rgb = np.asarray(rgb)
    if p.shape != rgb.shape or p.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError('RGB-D shape/type mismatch')
    low, high = finite(roi[0], (3,)), finite(roi[1], (3,))
    if np.any(high <= low):
        raise ValueError('ROI invalid')
    n, d = plane
    height = p@finite(n, (3,))+float(d)
    mask = valid & (height > min_height) & (height < max_height) & np.all((p >= low) & (p <= high), axis=2)
    comps = None
    if _accel.enabled():
        try:
            pix, off = _accel.flood_fill(mask, p, max_jump)
            comps = [pix[off[i]:off[i+1]] for i in range(len(off)-1)]
        except BaseException as e: _accel.guard(e); _accel.failed('flood_fill', e)
    if comps is None:
        comps = _components_py(p, mask, max_jump)
    result = []
    for pix in comps:
        if len(pix) < min_pixels:
            continue
        pix = np.asarray(pix); cloud = p[pix[:, 0], pix[:, 1]]
        lower, upper = np.min(cloud,axis=0), np.max(cloud,axis=0)
        color = np.median(rgb[pix[:, 0], pix[:, 1]], axis=0)/255.
        extent = upper-lower
        features = np.r_[np.sort(extent[:2]), np.quantile(cloud@np.asarray(n)+d, .98), color]
        result.append(Detection(np.median(cloud, axis=0), lower, upper, pix, features))
    return result


def _components_py(p, mask, max_jump):
    seen = np.zeros(mask.shape, bool); h, w = mask.shape; out = []
    for v, u in zip(*np.nonzero(mask)):
        if seen[v, u]:
            continue
        seen[v, u] = True; stack = [(v, u)]; pixels = []
        while stack:
            y, x = stack.pop(); pixels.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx] and np.linalg.norm(p[yy, xx]-p[y, x]) < max_jump:
                    seen[yy, xx] = True; stack.append((yy, xx))
        out.append(pixels)
    return out


def segment_py(points, rgb, valid, plane, roi, min_pixels=25, max_jump=.025, min_height=.012, max_height=.35):
    p = np.asarray(points); rgb = np.asarray(rgb)
    if p.shape != rgb.shape or p.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError('RGB-D shape/type mismatch')
    low, high = finite(roi[0], (3,)), finite(roi[1], (3,))
    if np.any(high <= low):
        raise ValueError('ROI invalid')
    n, d = plane
    height = p@finite(n, (3,))+float(d)
    mask = valid & (height > min_height) & (height < max_height) & np.all((p >= low) & (p <= high), axis=2)
    seen = np.zeros(mask.shape, bool); h, w = mask.shape; result = []
    for v, u in zip(*np.nonzero(mask)):
        if seen[v, u]:
            continue
        seen[v, u] = True; stack = [(v, u)]; pixels = []
        while stack:
            y, x = stack.pop(); pixels.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx] and np.linalg.norm(p[yy, xx]-p[y, x]) < max_jump:
                    seen[yy, xx] = True; stack.append((yy, xx))
        if len(pixels) < min_pixels:
            continue
        pix = np.asarray(pixels); cloud = p[pix[:, 0], pix[:, 1]]
        lower, upper = np.min(cloud,axis=0), np.max(cloud,axis=0)
        color = np.median(rgb[pix[:, 0], pix[:, 1]], axis=0)/255.
        extent = upper-lower
        features = np.r_[np.sort(extent[:2]), np.quantile(cloud@np.asarray(n)+d, .98), color]
        result.append(Detection(np.median(cloud, axis=0), lower, upper, pix, features))
    return result


class RangeClassifier:
    """User-supplied verified feature envelopes, not a pretrained model.

    Overlap or out-of-distribution => unknown. Envelopes are heuristic evidence,
    NOT calibrated probability and do not distinguish named instances.
    """
    def __init__(self, profiles=(), allowed=('dish', 'box', 'jar')):
        self.profiles = []
        for p in profiles:
            if p.get('category') not in allowed or p.get('validated_on_sensors') is not True:
                raise ValueError('profile not verified or wrong category')
            lo = finite(p['min'], (6,)); hi = finite(p['max'], (6,))
            if np.any(lo >= hi):
                raise ValueError('invalid feature envelope')
            self.profiles.append((p['category'], lo, hi))

    def classify(self, detection):
        f = finite(detection.features, (6,))
        error=np.zeros(6) if detection.feature_error is None else finite(detection.feature_error,(6,))
        if np.any(error<0): raise ValueError('negative feature uncertainty')
        compatible={c for c,lo,hi in self.profiles if np.all((f+error>=lo)&(f-error<=hi))}
        contained={c for c,lo,hi in self.profiles if np.all((f-error>=lo)&(f+error<=hi))}
        matches=contained if len(compatible)==1 else set()
        detection.category = next(iter(matches)) if len(matches) == 1 else None
        detection.reason = 'unique interval-contained envelope; heuristic, not probability' if len(matches) == 1 else 'unknown or ambiguous'
        return detection


def tray_color(rgb):
    """Coarse chroma vote only; not a cavity/placement detector."""
    a = finite(rgb)
    if a.size == 0 or a.shape[-1] != 3:
        return None
    c = np.median(a.reshape(-1, 3), axis=0)/255
    order = np.argsort(c)
    if c[order[-1]] < .25 or c[order[-1]]-c[order[-2]] < .18:
        return None
    return ('red', 'green', 'blue')[order[-1]]
