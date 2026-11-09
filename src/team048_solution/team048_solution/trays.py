"""Trays: camera-based localisation by colour (I6) and in-tray slots (I5).

Nominal geometry comes from the task description of the hackathon (scene.yaml: size 0.22 x 0.32 x 0.07, wall 0.008,
xy of the three slots). The colour NAME of each tray comes from /sim/task["trays"] (an allowed topic). The camera
confirms/refines the tray centre; on any doubt the nominal value is kept.

Colour evidence (Kaggle v5 head frame, out_v5/cam_head_color.npy): the green centre tray renders very desaturated,
RGB ~(234,250,238), dominant-channel margin ~0.047; the table renders neutral ~(224,225,224), margin ~0.004.
CHROMA_MIN=0.03 separates both. Red/blue trays were not in view in v5: TODO(verify) on a frame that shows them.
"""
import numpy as np

TRAY_SIZE = (.22, .32)      # x, y extent [m]; long axis along pelvis y
TRAY_H = .07
TRAY_WALL = .008
NOMINAL = {'tray_left': (.34, .69), 'tray_right': (.34, -.69), 'tray_center': (.53, 0.)}
CHANNEL = {'red': 0, 'green': 1, 'blue': 2}
CHROMA_MIN = .03
SLOT_STEP = .08


def chroma_mask(rgb, colour, chroma_min=CHROMA_MIN):
    """Pixels whose dominant channel is `colour` by at least chroma_min (fractions of 255)."""
    k = CHANNEL.get(colour)
    if k is None:
        raise ValueError('unknown tray colour')
    c = np.asarray(rgb, float)/255.
    others = np.delete(c, k, axis=-1)
    return (c[..., k]-others.max(axis=-1)) >= chroma_min


def locate(P, rgb, valid, name, colour, plane=None, window=.15, min_px=150, max_shift=.03, min_cover=.6):
    # max_shift .10 -> .03: v7 Kaggle tray_left (cut by the image border) was 'located' 6 cm off (y .629 vs .69) in 3
    # observations while the fully visible tray_center stayed within 0.5 cm of nominal; scene.yaml positions are fixed.
    """Tray centre (x, y) in pelvis and the source ('camera'|'nominal').

    Uses only points in a window around the nominal slot, at most TRAY_H+2 cm above the table plane, with the tray
    colour. An axis is refined only if the visible colour extent covers >= min_cover of the tray size on that axis
    (a partially visible tray would bias the centre); shifts > max_shift are rejected.
    """
    nx, ny = NOMINAL[name]
    P = np.asarray(P, float); m = np.asarray(valid, bool) & np.isfinite(P).all(axis=-1)
    m &= (np.abs(P[..., 0]-nx) < TRAY_SIZE[0]/2+window) & (np.abs(P[..., 1]-ny) < TRAY_SIZE[1]/2+window)
    if plane is not None:
        h = P@np.asarray(plane[0], float)+float(plane[1])
        m &= (h > -.01) & (h < TRAY_H+.02)
    # chroma only on the pixels still selected (same per-pixel arithmetic as chroma_mask on the full image)
    idx = np.nonzero(m)
    if len(idx[0]):
        sel = chroma_mask(np.asarray(rgb)[idx], colour)
        m = np.zeros_like(m); m[tuple(i[sel] for i in idx)] = True
    if int(m.sum()) < min_px:
        return np.array([nx, ny]), 'nominal'
    xy = P[m][:, :2]
    lo, hi = np.quantile(xy, .03, axis=0), np.quantile(xy, .97, axis=0)
    c = np.array([nx, ny], float)
    for a in (0, 1):
        if hi[a]-lo[a] >= min_cover*TRAY_SIZE[a] and hi[a]-lo[a] <= TRAY_SIZE[a]+.04:
            c[a] = (lo[a]+hi[a])/2
    if np.linalg.norm(c-[nx, ny]) > max_shift:
        return np.array([nx, ny]), 'nominal'
    return c, ('camera' if not np.allclose(c, [nx, ny]) else 'nominal')


def slot_offset(k, half_extent, step=SLOT_STEP, size=TRAY_SIZE, wall=TRAY_WALL, clearance=.01, ty=0.):
    """Offset along the tray long axis (pelvis y) for the k-th object placed in this tray: 0, inner, outer, 0, ...
    inner = toward the robot midline (v7 Kaggle: the outer slot y=.77 of tray_left was not reached, TCP 13.7 cm off).

    s = min(step, room) where room keeps the object's footprint half-extent inside the walls with clearance.
    """
    room = size[1]/2-wall-clearance-max(0., float(half_extent))
    s = max(0., min(step, room))
    inner = -s if ty > 0 else s
    return (0., inner, -inner)[int(k) % 3]
