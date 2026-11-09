"""Optional Rust acceleration (crate rust/team048_accel, module `team048_accel`).

Every accelerated function keeps its NumPy reference implementation in the same module (suffix `_py`); the Rust
path is used only when the extension imports, the model is supported (7 active joints, <=24 joints) and
TEAM048_ACCEL != "0". Any unexpected error inside the extension disables it for the process and the NumPy
reference takes over (logged once to stderr), so a missing or broken wheel can never stop the solution.
Equivalence (tests/test_accel_equivalence.py): FK/Jacobian/frames <= 1e-12, IK solutions <= 1e-9 on thousands of
random targets, collision verdicts identical, RANSAC plane and segments identical on the Kaggle frames.
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


def enabled():
    return AVAILABLE and not _disabled


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
                               float(model.margin), os.environ.get('TEAM048_ACCEL_PARALLEL', '1') != '0')
        f.set_env(np.ascontiguousarray(model._env_centers, float), [float(r) for r in model._env_radii])
        return f
    except BaseException as e:
        guard(e); failed('FastCollision', e); return None


def plane_inlier_counts(points, planes, tol):
    return _ext.plane_inlier_counts(np.ascontiguousarray(points, float), np.ascontiguousarray(planes, float), float(tol))


def flood_fill(mask, points, max_jump):
    return _ext.flood_fill(np.ascontiguousarray(mask, bool), np.ascontiguousarray(points, float), float(max_jump))
