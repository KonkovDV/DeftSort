"""Optional Rust acceleration (crate rust/team048_accel, module `team048_accel`).

Every accelerated function keeps its NumPy reference implementation in the same module (suffix `_py`); the Rust
path is used only when the extension imports, the model is supported (7 active joints, <=24 joints) and
TEAM048_ACCEL != "0". Any unexpected error inside the extension disables it for the process and the NumPy
reference takes over (logged once to stderr), so a missing or broken wheel can never stop the solution.
Equivalence (test/test_accel_equivalence.py, test/test_accel_kernels.py): FK/Jacobian/frames <= 1e-12, IK solutions <= 1e-9 on thousands of
random targets, collision verdicts identical, RANSAC plane and segments identical on the recorded simulator frames.
"""
import os, sys
import numpy as np

try:
    if os.environ.get('TEAM048_ACCEL', '1') == '0':
        raise ImportError('disabled by TEAM048_ACCEL=0')
    import team048_accel as _ext
    AVAILABLE = True; REASON = 'team048_accel %s' % getattr(_ext, '__version__', '?')
except Exception as e:      # ImportError, or an ABI mismatch surfacing as another error
    _ext = None; AVAILABLE = False; REASON = '%s: %s' % (type(e).__name__, e)

_disabled = False
# API level of the installed extension: 1 = FK/IK/collision/inlier counts/flood fill, 2 = additionally
# (Arm.solve, cached environments, deprojection, RANSAC stream, tray selection, segment statistics, grasp kernels).
# An API-1 wheel keeps working: the API-2 call sites then use their NumPy reference.
API = int(getattr(_ext, 'API', 1)) if AVAILABLE else 0
# rayon threads are opt-in: the simulator shares the evaluation machine's CPU, and on a machine with about two CPU cores
# parallel sweeps measured slower (certify 3.3 vs 2.7 ms). TEAM048_ACCEL_PARALLEL=1 enables them.
PARALLEL = os.environ.get('TEAM048_ACCEL_PARALLEL', '0') == '1'


def enabled():
    """True when the Rust module is importable and has not been disabled by a failure in this process."""
    return AVAILABLE and not _disabled


def enabled2():
    """True when the loaded Rust module exposes API >= 2 (perception and grasp kernels) and is not disabled."""
    return API >= 2 and not _disabled


def failed(where, exc):
    """Extension raised something unexpected: switch the whole process to the NumPy reference."""
    global _disabled
    if not _disabled:
        _disabled = True
        print('[team048_accel] disabled after %s in %s; using NumPy reference' % (type(exc).__name__, where), file=sys.stderr)


def guard(exc):
    """Re-raise interpreter-control exceptions; anything else from the extension is a fallback trigger."""
    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        raise exc


def fast_chain(chain):
    """Rust FK/IK chain mirroring a kinematics.Chain (7 active joints), or None when unavailable or unsupported."""
    if not enabled() or len(chain.active) != getattr(_ext, 'N_ACTIVE', 7) or len(chain.joints) > 24 or chain.joints[-1].child != chain.tip:
        return None
    try:
        return _ext.FastChain(np.ascontiguousarray([j.origin for j in chain.joints], float),
                              np.ascontiguousarray([j.axis for j in chain.joints], float),
                              [bool(j.movable) for j in chain.joints], np.ascontiguousarray(chain.tcp, float),
                              [float(x) for x in chain.lower], [float(x) for x in chain.upper])
    except BaseException as e:
        guard(e); failed('FastChain', e); return None


def fast_collision(model):
    """Rust sphere collision model mirroring a planning.CollisionModel whose chain has a Rust twin, or None."""
    fc = getattr(model.chain, '_fast', None)
    if fc is None or not enabled():
        return None
    try:
        index = {model.chain.base: 0}
        for i, j in enumerate(model.chain.joints):
            index[j.child] = i+1
        S = model.spheres
        pairs = [(i, k) for i in range(len(S)) for k in range(i+1, len(S))
                 if S[i].link != S[k].link and frozenset((S[i].link, S[k].link)) not in model.ignored]
        f = _ext.FastCollision(fc, [index[s.link] for s in S], np.ascontiguousarray([s.center for s in S], float),
                               [float(s.radius) for s in S], pairs, np.ascontiguousarray(model.levers, float),
                               float(model.margin), PARALLEL)
        f.set_env(np.ascontiguousarray(model._env_centers, float), [float(r) for r in model._env_radii])
        return f
    except BaseException as e:
        guard(e); failed('FastCollision', e); return None


def plane_inlier_counts(points, planes, tol):
    """Rust: inlier count per candidate plane (n_x, n_y, n_z, d) for points (N, 3) in m, |n.p + d| < tol (m)."""
    return _ext.plane_inlier_counts(np.ascontiguousarray(points, float), np.ascontiguousarray(planes, float), float(tol), PARALLEL)


def flood_fill(mask, points, max_jump):
    """Rust: connected components of `mask` over the organised cloud, neighbours joined when the 3-D jump is < max_jump (m)."""
    return _ext.flood_fill(np.ascontiguousarray(mask, bool), np.ascontiguousarray(points, float), float(max_jump))


# API 2 kernels
_PCG = {}


def pcg_state(seed):
    """(state_hi, state_lo, inc_hi, inc_lo, has_uint32, uinteger) of np.random.default_rng(seed)'s PCG64, from NumPy."""
    v = _PCG.get(seed)
    if v is None:
        st = np.random.PCG64(seed).state; M = (1 << 64)-1
        s, i = int(st['state']['state']), int(st['state']['inc'])
        v = _PCG[seed] = (s >> 64, s & M, i >> 64, i & M, bool(st['has_uint32']), int(st['uinteger']))
    return v


def ransac_plane(P, mask, seed, iterations, tol):
    """Rust: RANSAC table plane on the organised cloud P (H, W, 3) in m, same PCG64 stream as NumPy for `seed`."""
    return _ext.ransac_plane(np.ascontiguousarray(P, float), np.ascontiguousarray(mask, bool), pcg_state(seed),
                             int(iterations), float(tol), PARALLEL)


def deproject(depth, fx, fy, cx, cy, T):
    """Rust: depth image in mm -> organised cloud (H, W, 3) in m, pinhole K (px) then the 4x4 camera->pelvis transform T."""
    return _ext.deproject(np.ascontiguousarray(depth, float), float(fx), float(fy), float(cx), float(cy),
                          np.ascontiguousarray(T, float), PARALLEL)


def tray_window_mask(P, rgb, valid, h, window, band, channel, chroma_min):
    """Rust: tray colour/height mask inside the pelvis-frame search window (m), height band above the table (m)."""
    return _ext.tray_window_mask(np.ascontiguousarray(P, float), np.ascontiguousarray(rgb, np.uint8), np.ascontiguousarray(valid, bool),
                                 np.ascontiguousarray(h, float), tuple(float(x) for x in window), (float(band[0]), float(band[1])),
                                 int(channel), float(chroma_min))


def rot_rect_mask(P, rects):
    """Rust: pixels whose pelvis-frame xy lies in any rotated rectangle (x, y, u0, u1, v0, v1, half_a, half_b) in m."""
    return _ext.rot_rect_mask(np.ascontiguousarray(P, float), [tuple(float(x) for x in r) for r in rects])


def segment_components(P, rgb, valid, n, d, min_h, max_h, low, high, max_jump, min_pixels):
    """Rust: object segments above the table plane (n, d) inside the ROI box (m) with per-component statistics."""
    return _ext.segment_components(np.ascontiguousarray(P, float), np.ascontiguousarray(rgb, np.uint8), np.ascontiguousarray(valid, bool),
                                   [float(x) for x in n], float(d), float(min_h), float(max_h), [float(x) for x in low],
                                   [float(x) for x in high], float(max_jump), int(min_pixels))


def min_caliper(hull, dirs):
    """Rust: minimum convex-hull width (m) over the direction table `dirs` and the index of the best direction."""
    return _ext.min_caliper(np.ascontiguousarray(hull, float), np.ascontiguousarray(dirs, float))


def min_area_rect_scan(xy, dirs, q):
    """Rust: trimmed-quantile minimum-area rectangle scan over `dirs` for xy points in m (q = trimmed fraction)."""
    return _ext.min_area_rect_scan(np.ascontiguousarray(xy, float), np.ascontiguousarray(dirs, float), float(q))


def min_dist_xy(tips, pts):
    """Rust: smallest xy distance (m) from each fingertip point to the point set."""
    return _ext.min_dist_xy(np.ascontiguousarray(tips, float), np.ascontiguousarray(pts, float))


def fast_env(centers, radii):
    """Rust environment (pelvis-frame sphere centres in m and radii in m) for the collision model."""
    return _ext.FastEnv(np.ascontiguousarray(centers, float), np.ascontiguousarray(radii, float))


def table_mask(P, valid, x_lo, x_hi, y_abs, rects):
    """Rust: valid pixels on the table region x in [x_lo, x_hi], |y| < y_abs (m, pelvis) outside the given rectangles."""
    return _ext.table_mask(np.ascontiguousarray(P, float), np.ascontiguousarray(valid, bool), float(x_lo), float(x_hi), float(y_abs),
                           [tuple(float(v) for v in r) for r in rects])
