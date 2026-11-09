"""Сортировочные ящики: поиск камерой (цвет + глубина) и места внутри ящика.

Соответствие ответу организаторов #645 (2026-11-05): положение ящиков решение находит камерами, заранее измеренные
координаты не зашиваются. Здесь допустимы только
  * цвет ящика по имени: TASKS.md ("tray_left красный, tray_right синий, tray_center зелёный") и /sim/task["trays"];
  * ШИРОКИЕ окна поиска по смыслу имени (слева / справа / по центру от робота), а не измеренные позиции;
  * размер ящика как граница правдоподобия (DOC_SIZE, ±40 %); сам размер, высота стенок и ориентация МЕРЯЮТСЯ.
Если ящик не найден, возвращается None: вызывающий код переснимает (поворот головы) и никогда не подставляет позу.

Colour evidence (Kaggle v5/v7 head frames): green tray RGB ~(234,250,238), dominant-channel margin ~0.047; the table is
neutral ~(224,225,224), margin ~0.004 -> CHROMA_MIN=0.03 separates both. TASKS.md images show pastel red/blue trays.
"""
import numpy as np

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
        return [np.argwhere(lab == i+1) for i in np.argsort(-sizes)]
    except ImportError:   # pragma: no cover - scipy is a declared dependency
        return [np.argwhere(mask)] if mask.any() else []


def min_area_rect(xy, q=.02):
    """(centre(2), size(2) sorted short->long, long-axis unit vector(2)) of the minimum-area enclosing rectangle,
    using the q..1-q quantiles along each candidate axis (robust to depth outliers at edges); the extent is divided by
    (1-2q), the unbiased width of a uniformly covered interval."""
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
    m = np.asarray(valid, bool) & np.isfinite(P).all(axis=-1)
    m &= (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1)
    n, d = np.asarray(plane[0], float), float(plane[1])
    h = P@n+d
    m &= (h > -.01) & (h < DOC_H*(1+SIZE_TOL)+.02)
    idx = np.nonzero(m)
    if len(idx[0]) < min_px:
        return None
    sel = chroma_mask(np.asarray(rgb)[idx], colour)
    m = np.zeros_like(m); m[tuple(i[sel] for i in idx)] = True
    comps = [c for c in _components(m) if len(c) >= min_px]
    if not comps:
        return None
    pix = comps[0]
    H, W = m.shape
    touches = bool((pix[:, 0] < border).any() or (pix[:, 0] >= H-border).any() or
                   (pix[:, 1] < border).any() or (pix[:, 1] >= W-border).any())
    cloud = P[pix[:, 0], pix[:, 1]]
    c, size, axis = min_area_rect(cloud[:, :2])
    lo = np.array(DOC_SIZE)*(1-SIZE_TOL); hi = np.array(DOC_SIZE)*(1+SIZE_TOL)
    if np.any(size > hi):
        return None           # larger than any tray: not a tray (or merged with something of the same colour)
    small = bool(np.any(size < lo))
    if axis[1] < 0:
        axis = -axis          # canonical sign
    return dict(center=(float(c[0]), float(c[1])), size=(float(size[0]), float(size[1])),
                axis=(float(axis[0]), float(axis[1])), height=float(np.quantile(h[pix[:, 0], pix[:, 1]], .98)),
                partial=touches or small, centroid=tuple(float(x) for x in cloud[:, :2].mean(0)), px=int(len(pix)))


def slot_offset(k, half_extent, step=SLOT_STEP, size=DOC_SIZE, wall=TRAY_WALL, clearance=.01, ty=0.):
    """Offset ALONG THE TRAY LONG AXIS for the k-th object placed in this tray: 0, inner, outer, 0, ...
    inner = toward the robot midline (v7 Kaggle: the outer slot of tray_left was not reached, TCP 13.7 cm off).
    `size` = MEASURED tray size (short, long) or (x, y) extent; s = min(step, room) keeps the object's footprint
    half-extent inside the walls with clearance."""
    room = max(size)/2-wall-clearance-max(0., float(half_extent))
    s = max(0., min(step, room))
    inner = -s if ty > 0 else s
    return (0., inner, -inner)[int(k) % 3]
