"""Сортировочные ящики: поиск камерой (цвет + глубина) и места внутри ящика.

Соответствие ответу организаторов #645 (2026-11-05): положение ящиков решение находит камерами, заранее измеренные
координаты не зашиваются. Здесь допустимы только
  * цвет ящика по имени: TASKS.md ("tray_left красный, tray_right синий, tray_center зелёный") и /sim/task["trays"];
  * ШИРОКИЕ окна поиска по смыслу имени (слева / справа / по центру от робота), а не измеренные позиции;
  * размер ящика как граница правдоподобия (DOC_SIZE, ±40 %); сам размер, высота стенок и ориентация МЕРЯЮТСЯ.
Если ящик не найден, возвращается None: вызывающий код переснимает (поворот головы) и никогда не подставляет позу.

Colour evidence (run v5/v7 head frames): green tray RGB ~(234,250,238), dominant-channel margin ~0.047; the table is
neutral ~(224,225,224), margin ~0.004 -> CHROMA_MIN=0.03 separates both. TASKS.md images show pastel red/blue trays.
"""
import numpy as np

from . import _accel

DOC_SIZE = (.22, .32)       # x, y extent [m]: plausibility bound only (organiser scene description), measured at runtime
DOC_H = .07                 # wall height [m]: plausibility bound only
SIZE_TOL = .40              # accepted measured size: DOC_SIZE*(1-SIZE_TOL) .. DOC_SIZE*(1+SIZE_TOL)
TRAY_WALL = .008
CHANNEL = {'red': 0, 'green': 1, 'blue': 2}
CHROMA_MIN = .03
SLOT_STEP = .08
# Search windows (pelvis frame, x forward, y left) -- semantic regions "left of / right of / in front of the robot"
# that cover the whole reachable table half-planes; NOT measured positions (#645: approximate search bounds allowed).
SEARCH = {'tray_left': ((.05, 1.10), (.20, 1.30)),
          'tray_right': ((.05, 1.10), (-1.30, -.20)),
          'tray_center': ((.20, 1.10), (-.45, .45))}
# Backward-compatible name: the code no longer has nominal positions.
TRAY_SIZE = DOC_SIZE
TRAY_H = DOC_H


def chroma_mask(rgb, colour, chroma_min=CHROMA_MIN):
    """Pixels whose dominant channel is `colour` by at least chroma_min (fractions of 255)."""
    k = CHANNEL.get(colour)
    if k is None:
        raise ValueError('unknown tray colour')
    c = np.asarray(rgb, float)/255.
    others = np.delete(c, k, axis=-1)
    return (c[..., k]-others.max(axis=-1)) >= chroma_min


def _components(mask):
    """4-connected components of a boolean image (pixel lists), largest first. Pure NumPy/scipy-free BFS on indices."""
    try:
        from scipy.ndimage import label
        lab, n = label(mask)
        if n == 0:
            return []
        sizes = np.bincount(lab.ravel())[1:]
        return [np.argwhere(lab == i+1) for i in np.argsort(-sizes, kind='stable')]
    except ImportError:   # pragma: no cover - scipy is a declared dependency
        return [np.argwhere(mask)] if mask.any() else []


def _largest_component(mask, min_px):
    """_components(mask) filtered to >= min_px pixels, first entry (or None) -- the only one detect() uses. Same
    labelling and the same size order (argsort of -sizes), but only that component's pixel list is extracted
    (extracting all of them was ~11 ms per call on v7 frames)."""
    try:
        from scipy.ndimage import label
    except ImportError:   # pragma: no cover - scipy is a declared dependency
        c = [x for x in _components(mask) if len(x) >= min_px]
        return c[0] if c else None
    lab, n = label(mask)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    # v15: kind='stable' -- equal sizes keep label (raster) order. The default introsort is not stable for >16
    # components: with sizes [..., 60, 60, ...] (labels 9 and 10) it returned label 10 (NumPy 1.26 and 2.x).
    i = np.argsort(-sizes, kind='stable')[0]
    return np.argwhere(lab == i+1) if sizes[i] >= min_px else None


_RECT_DIRS = np.array([np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, np.pi/2, 91)])   # the loop's u


def min_area_rect(xy, q=.02):
    """(centre(2), size(2) sorted short->long, long-axis unit vector(2)) of the minimum-area enclosing rectangle,
    using the q..1-q quantiles along each candidate axis (robust to depth outliers at edges); the extent is divided by
    (1-2q), the unbiased width of a uniformly covered interval. The 91-direction scan runs in Rust when available
    (same quantiles, same first-minimum rule: bit-identical); min_area_rect_py is the reference."""
    xy = np.asarray(xy, float)
    r = None
    if _accel.enabled2() and xy.ndim == 2 and xy.shape[1] == 2 and len(xy) and np.isfinite(xy).all():
        try:
            r = _accel.min_area_rect_scan(xy, _RECT_DIRS, q)
        except BaseException as e:
            _accel.guard(e); _accel.failed('min_area_rect_scan', e); r = None
    if r is None:
        return min_area_rect_py(xy, q)
    k, lu, hu, lv, hv = r
    u = _RECT_DIRS[k].copy(); v = np.array([-u[1], u[0]])
    return _rect_result(u, v, (lu+hu)/2, (lv+hv)/2, hu-lu, hv-lv, q)


def min_area_rect_py(xy, q=.02):
    """NumPy reference: trimmed-quantile minimum-area rectangle of xy points (m), 1 deg steps over 0-90 deg."""
    xy = np.asarray(xy, float)
    best = None
    for a in np.linspace(0, np.pi/2, 91):
        u = np.array([np.cos(a), np.sin(a)]); v = np.array([-u[1], u[0]])
        pu, pv = xy@u, xy@v
        lu, hu = np.quantile(pu, [q, 1-q]); lv, hv = np.quantile(pv, [q, 1-q])
        area = (hu-lu)*(hv-lv)
        if best is None or area < best[0]:
            best = (area, u, v, (lu+hu)/2, (lv+hv)/2, hu-lu, hv-lv)
    _, u, v, cu, cv, su, sv = best
    return _rect_result(u, v, cu, cv, su, sv, q)


def _rect_result(u, v, cu, cv, su, sv, q):
    c = cu*u+cv*v; su /= (1-2*q); sv /= (1-2*q)
    if su >= sv:
        return c, np.array([sv, su]), u
    return c, np.array([su, sv]), v


def detect(P, rgb, valid, name, colour, plane, min_px=150, border=2):
    """Tray in the current frame from colour + depth.

    Returns dict(center=(x,y), size=(short,long), axis=long-axis unit (x,y), height=wall top above the table,
    partial=bool, centroid=(x,y), px=int) or None when nothing tray-like of that colour is in the search window.
    partial=True: the component touches the image border or is smaller than the plausibility bound -> the centre is
    NOT reliable; the caller re-aims the head at `centroid` and looks again (never falls back to a stored pose).
    """
    if colour not in CHANNEL or name not in SEARCH:
        return None
    (x0, x1), (y0, y1) = SEARCH[name]
    P = np.asarray(P, float)
    n, d = np.asarray(plane[0], float), float(plane[1])
    h = P@n+d
    band = (-.01, DOC_H*(1+SIZE_TOL)+.02)
    m = None
    if (_accel.enabled2() and P.ndim == 3 and getattr(rgb, 'dtype', None) == np.uint8 and np.shape(rgb) == P.shape
            and np.shape(valid) == P.shape[:2] and np.shape(h) == P.shape[:2]):
        try:   # same selection, one pass (h stays NumPy's, so the band test sees the reference values)
            cnt, m = _accel.tray_window_mask(P, rgb, valid, h, (x0, x1, y0, y1), band, CHANNEL[colour], CHROMA_MIN)
        except BaseException as e:
            _accel.guard(e); _accel.failed('tray_window_mask', e); m = None
        if m is not None and cnt < min_px:
            return None
    if m is None:
        m = np.asarray(valid, bool) & np.isfinite(P).all(axis=-1)
        m &= (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
        m &= (h > band[0]) & (h < band[1])
        idx = np.nonzero(m)
        if len(idx[0]) < min_px:
            return None
        sel = chroma_mask(np.asarray(rgb)[idx], colour)
        m = np.zeros_like(m); m[tuple(i[sel] for i in idx)] = True
    pix = _largest_component(m, min_px)
    if pix is None:
        return None
    H, W = m.shape
    touches = bool((pix[:, 0] < border).any() or (pix[:, 0] >= H-border).any() or
                   (pix[:, 1] < border).any() or (pix[:, 1] >= W-border).any())
    cloud = P[pix[:, 0], pix[:, 1]]
    c, size, axis = min_area_rect(cloud[:, :2])
    lo = np.array(DOC_SIZE)*(1-SIZE_TOL); hi = np.array(DOC_SIZE)*(1+SIZE_TOL)
    if np.any(size > hi):
        return None           # larger than any tray: not a tray (or merged with something of the same colour)
    small = bool(np.any(size < lo))
    axis = canonical_axis(axis)
    return dict(center=(float(c[0]), float(c[1])), size=(float(size[0]), float(size[1])),
                axis=(float(axis[0]), float(axis[1])), height=float(np.quantile(h[pix[:, 0], pix[:, 1]], .98)),
                partial=touches or small, centroid=tuple(float(x) for x in cloud[:, :2].mean(0)), px=int(len(pix)))


def canonical_axis(axis, eps=1e-9):
    """Sign convention of a tray long axis: y component positive; for an axis along x (|y| <= eps) x positive.
    v14: the old rule (flip if y < 0) left the two representations of the x axis that min_area_rect returns for an
    exact 0 / 90 deg tie, u(0) = (1, 0) and v(90 deg) = (-1, 6e-17), with opposite signs. Which one wins the tie can
    change with 1-ulp rounding (NumPy 2 vs the Rust scan: 7 of 62 recorded frames, tray_left), and the sign selects the
    direction of the inner/outer slot (slot_offset * axis)."""
    axis = np.asarray(axis, float)
    if axis[1] < -eps or (abs(axis[1]) <= eps and axis[0] < 0):
        return -axis
    return axis


def slot_offset(k, half_extent, step=SLOT_STEP, size=DOC_SIZE, wall=TRAY_WALL, clearance=.01, ty=0.):
    """Offset ALONG THE TRAY LONG AXIS for the k-th object placed in this tray: 0, inner, outer, 0, ...
    inner = toward the robot midline (run v7: the outer slot of tray_left was not reached, TCP 13.7 cm off).
    `size` = MEASURED tray size (short, long) or (x, y) extent; s = min(step, room) keeps the object's footprint
    half-extent inside the walls with clearance."""
    room = max(size)/2-wall-clearance-max(0., float(half_extent))
    s = max(0., min(step, room))
    inner = -s if ty > 0 else s
    return (0., inner, -inner)[int(k) % 3]
