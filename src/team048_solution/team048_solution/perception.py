"""Sensor-only plane removal and organized components (object identity: node.identify, size priors)."""
from dataclasses import dataclass
import numpy as np
from .geometry import finite
from . import _accel


@dataclass
class Detection:
    """One segment above the table: centroid and bounds (pelvis, m), pixel indices, feature vector."""
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
    if _accel.enabled2() and np.ndim(points) == 3 and np.shape(valid) == np.shape(points)[:2] and type(seed) is int and seed >= 0:
        try:
            r = _accel.ransac_plane(points, valid, seed, iterations, tolerance)
        except BaseException as e:
            _accel.guard(e); _accel.failed('ransac_plane', e); r = None
        if r is not None:
            return _plane_from_best(*r, tolerance, min_inliers)
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


def _plane_from_best(p, planes, k, count, tolerance, min_inliers):
    """Rust path of table_plane: p = compacted cloud (same rows/order as NumPy), planes = candidates of the same RNG
    stream (normal, offset), k = first candidate with the maximal inlier count (-1: all counts 0; exact pruned scan).
    Inlier set and refinement exactly as in table_plane."""
    if len(p) < min_inliers:
        raise ValueError('insufficient plane support')
    best = None; score = 0
    if k >= 0:
        normal, offset = planes[k, :3].copy(), planes[k, 3]; inliers = abs(p@normal+offset) < tolerance
        score = int(inliers.sum()); best = (normal, offset, inliers)
    return _refine_plane(p, best, score, min_inliers)


def table_region(P, valid, rects=(), x=(.15, .75), y_abs=.7):
    """node.observe table mask: valid & x0 < X < x1 & |Y| < y_abs, minus every rectangle (cx, cy, hx, hy)
    (|X-cx| < hx & |Y-cy| < hy, the tray exclusion). Rust kernel when available (element-wise identical)."""
    if _accel.enabled2() and np.ndim(P) == 3 and np.shape(valid) == np.shape(P)[:2]:
        try: return _accel.table_mask(P, valid, x[0], x[1], y_abs, rects)
        except BaseException as e: _accel.guard(e); _accel.failed('table_mask', e)
    m = valid & (P[..., 0] > x[0]) & (P[..., 0] < x[1]) & (np.abs(P[..., 1]) < y_abs)
    for cx, cy, hx, hy in rects:
        m = m & ~((np.abs(P[..., 0]-cx) < hx) & (np.abs(P[..., 1]-cy) < hy))
    return m


def rect_mask(P, rects):
    """Pixels inside any rotated rectangle (x, y, u, v, ha, hb): |(X-x)u0+(Y-y)u1| < ha & |(X-x)v0+(Y-y)v1| < hb
    (u, v unit 2-vectors). Rust kernel when available (element-wise identical)."""
    rects = [(float(x), float(y), float(u[0]), float(u[1]), float(v[0]), float(v[1]), float(ha), float(hb)) for x, y, u, v, ha, hb in rects]
    if rects and _accel.enabled2() and np.ndim(P) == 3 and np.shape(P)[2] == 3:
        try: return _accel.rot_rect_mask(P, rects)
        except BaseException as e: _accel.guard(e); _accel.failed('rot_rect_mask', e)
    return rect_mask_py(P, rects)


def rect_mask_py(P, rects):
    """NumPy reference of the rotated-rectangle mask (rects in pelvis-frame m)."""
    m = np.zeros(np.shape(P)[:2], bool)
    for x, y, u0, u1, v0, v1, ha, hb in rects:
        dx = P[..., 0]-x; dy = P[..., 1]-y
        m |= (np.abs(dx*u0+dy*u1) < ha) & (np.abs(dx*v0+dy*v1) < hb)
    return m


def deproject(depth_mm, K, T):
    """Organized pelvis-frame cloud (h,w,3) from a depth image in mm (float) -- node.observe arithmetic:
    X = (u-cx)*Z/fx, Y = (v-cy)*Z/fy, Z = depth/1000, P = [X,Y,Z] @ R.T + t. Rust kernel when available
    (fixed evaluation order; NumPy's BLAS product may differ by <= 2 ulp depending on the CPU's OpenBLAS kernel)."""
    K = np.asarray(K, float); T = np.asarray(T, float); dmm = np.asarray(depth_mm, float)
    # v14: dmm.size -- an empty image (0 rows or columns) panicked in the Rust kernel (perc.rs chunk size 0), and the
    # panic switched the WHOLE process to the NumPy reference (2x slower planning); the reference returns (h, w, 3).
    if _accel.enabled2() and dmm.ndim == 2 and dmm.size and K.shape == (3, 3) and T.shape == (4, 4):
        try: return _accel.deproject(dmm, K[0, 0], K[1, 1], K[0, 2], K[1, 2], T)
        except BaseException as e: _accel.guard(e); _accel.failed('deproject', e)
    return deproject_py(dmm, K, T)


def deproject_py(dmm, K, T):
    """NumPy reference: depth (mm) -> organised cloud (H, W, 3) in m via K (px) and camera->pelvis T."""
    h, w = dmm.shape
    v, u = np.mgrid[0:h, 0:w]; Z = dmm/1000.
    X = (u-K[0, 2])*Z/K[0, 0]; Y = (v-K[1, 2])*Z/K[1, 1]
    return (np.stack([X, Y, Z], -1).reshape(-1, 3)@T[:3, :3].T+T[:3, 3]).reshape(h, w, 3)


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
    """NumPy reference RANSAC plane (n, d) with n.p + d = 0 (m) on the valid points; tolerance in m."""
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


def near_edge(points, valid, plane, y_half=.5, drop=.05, q=.1, min_cols=40, min_below=5):
    """Near (robot-side) table edge x from one view, or None if this view does not SEE the edge.
    Column-wise in the image: the lowest table-plane inlier pixel of a column is an edge pixel only if the pixels
    below it in that column (closer to the camera along the image) include >= min_below valid points at least `drop`
    under the plane (the view passes the edge and drops to the floor/crates). Otherwise the inlier boundary is just the
    image border and the true edge may be closer -> None (caller uses a conservative bound).
    Run v7 torso frames: edge pixels at x 0.21-0.25 m (pelvis), head frames: edge not visible (None)."""
    P = np.asarray(points, float)
    ok = np.asarray(valid, bool) & np.isfinite(P).all(axis=-1)
    n, d = np.asarray(plane[0], float), float(plane[1])
    h = np.where(ok, P@n+d, np.nan)
    inl = ok & (np.abs(h) < .008) & (np.abs(P[..., 1]) < y_half)
    H = inl.shape[0]
    xs = []
    for c in np.nonzero(inl.any(axis=0))[0]:
        r = int(np.nonzero(inl[:, c])[0].max())
        if r >= H-1:
            continue
        below = h[r+1:, c]
        if int(np.sum(below < -drop)) >= min_below:
            xs.append(P[r, c, 0])
    if len(xs) < min_cols:
        return None
    return float(np.quantile(xs, q))


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
    if _accel.enabled2() and p.ndim == 3 and np.shape(valid) == p.shape[:2]:
        try:
            pix, off, keep, rows = _accel.segment_components(p, rgb, np.asarray(valid, bool), finite(n, (3,)), float(d), min_height,
                                                             max_height, low, high, max_jump, min_pixels)
            # row: centroid(3) lower(3) upper(3) sorted xy extent(2) height q98 median rgb/255 (3)  -- perc.rs
            return [Detection(r[0:3].copy(), r[3:6].copy(), r[6:9].copy(), pix[off[c]:off[c+1]], r[9:15].copy())
                    for c, r in zip(keep, rows)]
        except BaseException as e: _accel.guard(e); _accel.failed('segment_components', e)
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
    """NumPy reference segmentation above the plane inside the ROI box (pelvis, m); heights and jumps in m."""
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

