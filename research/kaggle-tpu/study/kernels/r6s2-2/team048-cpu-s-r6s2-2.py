import os,json
os.environ['STUDY_CFG']=json.dumps(json.loads(r'''{"mode": "reach", "task": "sort2", "pool": {"electronics": ["Razer_Taipan_White_Ambidextrous_Gaming_Mouse", "JBL_Charge_Speaker_portable_wireless_wired_Green", "Crosley_Alarm_Clock_Vintage_Metal"], "box": ["Nestle_Nips_Hard_Candy_Peanut_Butter", "Nips_Hard_Candy_Rich_Creamy_Butter_Rum_4_oz_1133_g", "009_gelatin_box"]}, "cats": ["electronics", "box"], "l0": 67, "l1": 100, "meta": {"Razer_Taipan_White_Ambidextrous_Gaming_Mouse": ["electronics", [0.1, 0.182, 0.04]], "JBL_Charge_Speaker_portable_wireless_wired_Green": ["electronics", [0.17600000000000002, 0.073, 0.075]], "Crosley_Alarm_Clock_Vintage_Metal": ["electronics", [0.109, 0.054000000000000006, 0.162]], "Nestle_Nips_Hard_Candy_Peanut_Butter": ["box", [0.08800000000000001, 0.042, 0.153]], "Nips_Hard_Candy_Rich_Creamy_Butter_Rum_4_oz_1133_g": ["box", [0.08900000000000001, 0.043, 0.153]], "009_gelatin_box": ["box", [0.08900000000000001, 0.10099999999999999, 0.03]]}, "seed_base": 76500, "deadline_s": 2880}'''))
os.environ.setdefault('OMP_NUM_THREADS','1')
"""Team048 offline CPU study on the NEW planner (commits 9387441/fe0a4d2): torso-yaw place, finger grasp check,
measured tilted table, rim pinch point. Modes: reach | grasp | tubes. Pure numpy/scipy, no ROS (node.py via test stubs).
Layouts: port of the organisers' hack_sim/objects.py::_sample_once (same rules: area, edge margin, gap, tray exclusion,
50% axis-aligned yaw), object half-extents from objects.yaml dims. Clouds: synthetic full-surface (best-case perception).
Checkpoints every result line to results.jsonl; stops submitting work at DEADLINE_S."""
import os, sys, json, time, math, random, traceback, collections, zlib, zipfile, multiprocessing as mp
from pathlib import Path
import numpy as np

CFG = json.loads(os.environ.get('STUDY_CFG', '{}'))
OUT = Path(CFG.get('out', '/kaggle/working')); OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT/'study.log'; RES = OUT/'results.jsonl'
DEADLINE_S = float(CFG.get('deadline_s', 50*60))
T_START = time.time()
PLANE_N = np.array([-.0699, 0, .9976]); PLANE_D = .1788   # v7 measured table plane in pelvis (tilted 4.0 deg)
TABLE = dict(front_gap=.22, depth=.76, width=2.44)
SPAWN = dict(area_x=(.26, .50), area_y=(-.52, .52), edge_margin=.03, gap=.02)
TRAYS = {'tray_left': (.34, .69), 'tray_right': (.34, -.69), 'tray_center': (.53, 0.)}
TRAY_SIZE = (.22, .32)
TASK_TRAYS = {'sort2': ['tray_left', 'tray_right'], 'sort3': ['tray_left', 'tray_right', 'tray_center']}

def log(m):
    line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z +{time.time()-T_START:6.0f}s {m}"
    print(line, flush=True)
    with LOG.open('a') as f: f.write(line+'\n')

def find_src():
    for base in [Path(CFG.get('src', '')), Path('/kaggle/input')]:
        if not str(base) or not base.exists(): continue
        for p in base.rglob('test_node_stub.py'):
            return p.parent.parent   # .../team048_solution (contains package + test)
    for z in sorted(Path('/kaggle/input').rglob('*.zip'), key=lambda z: 'sol_src' not in z.name) if Path('/kaggle/input').exists() else []:
        dest = OUT/'_src'/z.stem; dest.mkdir(parents=True, exist_ok=True); zipfile.ZipFile(z).extractall(dest)
        for p in dest.rglob('test_node_stub.py'): return p.parent.parent
    raise FileNotFoundError('sol_src not found: '+str([str(x) for _, x in zip(range(30), Path('/kaggle/input').rglob('*'))]))

def setup():
    src = find_src()
    sys.path.insert(0, str(src)); sys.path.insert(0, str(src/'test'))
    import test_node_stub  # noqa: installs rclpy/msg stubs, imports node
    return src

def zt(x, y, n=PLANE_N, d=PLANE_D):
    nn = np.linalg.norm(n); n = n/nn; d = d/nn
    return -(d+n[0]*x+n[1]*y)/n[2]

def plane():
    nn = np.linalg.norm(PLANE_N); return (PLANE_N/nn, PLANE_D/nn)

# ---------------- organisers' layout sampler (port) ----------------
def _obb(x, y, yaw, hx, hy):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x+c*u-s*v, y+s*u+c*v) for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
def _overlap(a, b):
    for poly in (a, b):
        for i in range(len(poly)):
            (x0, y0), (x1, y1) = poly[i], poly[(i+1) % len(poly)]
            nx, ny = y0-y1, x1-x0
            pa = [nx*x+ny*y for x, y in a]; pb = [nx*x+ny*y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa): return False
    return True
def sample_layout(names, dims, seed, trays):
    best = None
    for attempt in range(80):
        lay = _sample_once(names, dims, random.Random(int(seed)*1009+attempt), trays)
        miss = sum(1 for v in lay.values() if v == 'park')
        if best is None or miss < best[0]: best = (miss, lay)
        if miss == 0: break
    return best[1]
def _sample_once(names, dims, rng, trays):
    t = TABLE; sp = SPAWN
    tx0, tx1 = t['front_gap']+sp['edge_margin'], t['front_gap']+t['depth']-sp['edge_margin']
    ty = t['width']/2-sp['edge_margin']; ax, ay = sp['area_x'], sp['area_y']
    x0, x1, y0, y1 = max(ax[0], tx0), min(ax[1], tx1), max(ay[0], -ty), min(ay[1], ty)
    g = sp['gap']
    rects = [(x-TRAY_SIZE[0]/2-g, x+TRAY_SIZE[0]/2+g, y-TRAY_SIZE[1]/2-g, y+TRAY_SIZE[1]/2+g) for (x, y) in [TRAYS[k] for k in trays]]
    tpoly = [[(a, c), (b, c), (b, d), (a, d)] for a, b, c, d in rects]
    placed, out = [], {}
    size = lambda n: (dims[n][0]/2+g/2, dims[n][1]/2+g/2)
    for n in sorted(names, key=lambda k: -size(k)[0]*size(k)[1]):
        hx, hy = size(n)
        for _ in range(600):
            yaw = rng.uniform(-math.pi, math.pi)
            if rng.random() < .5: yaw = (math.pi/2 if hx > hy else 0.)+rng.uniform(-.3, .3)
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            poly = _obb(x, y, yaw, hx, hy)
            if any(not (x0 <= a <= x1 and y0 <= b <= y1) for a, b in poly): continue
            if any(_overlap(poly, q) for q in tpoly) or any(_overlap(poly, q) for q in placed): continue
            break
        else:
            out[n] = 'park'; continue
        placed.append(poly); out[n] = (x, y, yaw)
    return {n: out[n] for n in names}

def cloud(cat, dims, x, y, yaw, rng, N=2500):
    dx, dy, h = dims
    if cat in ('box', 'electronics'):
        u = rng.uniform(-.5, .5, (N, 3))*[dx, dy, 0]; face = rng.integers(0, 5, N); u[:, 2] = rng.uniform(0, h, N)
        u[face == 0, 2] = h; u[face == 1, 0] = dx/2; u[face == 2, 0] = -dx/2; u[face == 3, 1] = dy/2; u[face == 4, 1] = -dy/2
    else:
        t = rng.uniform(0, 2*np.pi, N); z = rng.uniform(0, h, N); u = np.c_[dx/2*np.cos(t), dy/2*np.sin(t), z]
        top = rng.random(N) < .3; rr = np.sqrt(rng.random(N))
        if cat == 'dish': rr = np.where(rng.random(N) < .5, 1., rr)   # open top: rim ring + bottom seen as the top cap
        u[top] = np.c_[dx/2*rr*np.cos(t), dy/2*rr*np.sin(t), np.full(N, h)][top]
    c, s = np.cos(yaw), np.sin(yaw); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    P = u@R.T; P[:, 0] += x; P[:, 1] += y; P[:, 2] += zt(x, y); return P

def load_meta(src):
    pools = json.load(open(Path(__file__).with_name('pools.json')))['pools'] if Path(__file__).with_name('pools.json').exists() else CFG['pools']
    return pools

# ---------------- worker ----------------
W = {}
def winit(src_str):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    sys.path.insert(0, src_str); sys.path.insert(0, src_str+'/test')
    import test_node_stub  # noqa
    from team048_solution import arm as ARM, node as NODE, trays as TR, grasp as GR
    W.update(ARM=ARM, NODE=NODE, TR=TR, GR=GR, arms={k: ARM.Arm(k) for k in 'lr'})

def rotz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

def try_pick(A, g, f, top, tilts=(0, .35, .6), point=None, side_dz=None, pre=.08):
    NODE = W['NODE']
    ap = np.r_[g['approach'], 0.] if g['kind'] == 'side' else np.asarray(g.get('approach3', [0, 0, -1.]), float)
    ca = np.r_[g['close_axis'], 0.] if len(g['close_axis']) == 2 else np.asarray(g['close_axis'], float)
    pt = point or ('pinch' if g['kind'] == 'rim' else 'grip')
    for dz in (side_dz if side_dz is not None else (NODE.SIDE_DZ if g['kind'] == 'side' else (0.,))):
        if g['kind'] == 'side' and dz > 0 and g['z']+dz > f['height']-.015: break
        c3 = np.array([g['center'][0], g['center'][1], top+g['z']+dz])
        p = A.plan_pick(c3, ca, ap, g['width'], point=pt, tilts=tilts, pre=pre)
        if p is not None: return p, dz
    return None, None

def eval_obj(job):
    """Node logic (pick_place without actuation) for one object of one layout -> outcome dict."""
    NODE, TR, GR = W['NODE'], W['TR'], W['GR']; arms = W['arms']
    rng = np.random.default_rng(job['seed'])
    pl = plane(); trays_xy = [TRAYS[k] for k in job['trays']]
    for A in arms.values(): A.set_scene(pl, trays_xy)
    C_all = {n: cloud(o['cat'], o['dims'], *o['pose'], rng) for n, o in job['objs'].items()}
    name = job['name']; o = job['objs'][name]
    others = [C_all[m] for m in job['objs'] if m != name]
    Wm = min(arms['l'].hand['opposed_open']-.01, .11)
    t0 = time.time()
    f, C = GR.candidates(C_all[name], pl, o['cat'], others, W_MAX=Wm, robot_xy=(0., 0.))
    out = dict(layout=job['layout'], name=name, cat=o['cat'], tray=job['tray'], pose=[round(v, 3) for v in o['pose']],
               n_cand=len(C), n_feas=sum(g['feasible'] for g in C))
    tx, ty = TRAYS[job['tray']]
    order = 'lr' if ty > .1 else ('rl' if ty < -.1 else ('lr' if f['center'][1] >= 0 else 'rl'))
    res = 'no_candidate' if not C else ('no_clearance' if not any(g['feasible'] for g in C) else 'no_ik')
    picked_any = False
    for side in order:
        A = arms[side]
        for g in C:
            if not g['feasible']: continue
            top = zt(*g['center'])
            p, dz = try_pick(A, g, f, top)
            if p is None: continue
            picked_any = True; res = 'no_place'
            yp = NODE.plan_place_yaw(A, p['q_lift'], tx, ty, pl, trays_xy)
            if yp is None: continue
            yaw, _ = yp
            res = 'ok'; out.update(side=side, kind=g['kind'], dz=dz, yaw=round(float(yaw), 3), tilt=p['tilt'],
                                   pick_dur=round(p['duration'], 2), width=round(g['width'], 3))
            break
        if res == 'ok': break
    out.update(result=res, picked=picked_any, plan_s=round(time.time()-t0, 2))
    return out

# ---------------- grasp variants (b) ----------------
def variants_for(cat):
    V = [dict(id='base')]
    V += [dict(id='tilt_wide', tilts=(0, .2, .35, .5, .6, .8, 1.0))]
    V += [dict(id='yawpick_%+.1f' % a, yaw_pick=a) for a in (-.6, -.3, .3, .6)]
    V += [dict(id='pre05', pre=.05)]
    if cat in ('box', 'electronics', 'jar'):
        V += [dict(id='topdown', topdown=True), dict(id='topdown_pinch', topdown=True, point='pinch'),
              dict(id='topdown_shallow', topdown=True, td_depth=.01), dict(id='topdown_wide', topdown=True, td_depth=.01, td_wmax=.097)]
        V += [dict(id='side_pinch', point='pinch')]
        V += [dict(id='side_z%.2f' % fz, zfrac=fz) for fz in (.3, .7)]
        V += [dict(id='roll%+d' % r, roll=r) for r in (-15, 15)]
    if cat == 'dish':
        V += [dict(id='rim_ang%+d' % a, rim_ang=a) for a in (-60, -30, 30, 60, 90, -90)]
        V += [dict(id='rim_tilt_in%.2f' % t, rim_tilt=t) for t in (.35, .6)]
        V += [dict(id='rim_tilt_out%.2f' % t, rim_tilt=-t) for t in (.35,)]
        V += [dict(id='rim_depth%.3f' % d, rim_depth=d) for d in (.008, .025)]
        V += [dict(id='rim_grip', point='grip')]
    return V

def make_variant_cands(v, f, C, cat, cloud_, pl):
    """Modify candidate list per variant; returns list of grasp dicts (kind/center/close_axis/approach/width/z)."""
    out = []
    if v.get('topdown'):
        for ax, w in ((f['axis'], f['width']), (np.array([-f['axis'][1], f['axis'][0]]), f['length'])):
            if w > v.get('td_wmax', .085): continue      # default: same 1 cm/side clearance as the side grasp (W_MAX-.01)
            out.append(dict(kind='top', center=f['center'], close_axis=np.r_[ax, 0.], approach3=[0, 0, -1.], width=w,
                            z=max(f['height']-v.get('td_depth', .03), .5*f['height']), feasible=True))
        return out
    for g in C:
        g = dict(g)
        if not g['feasible']: continue
        if 'zfrac' in v and g['kind'] == 'side': g['z'] = max(.02, v['zfrac']*f['height'])
        if 'roll' in v and g['kind'] == 'side':
            a = math.radians(v['roll']); ap = np.r_[g['approach'], 0.]; ca = np.r_[g['close_axis'], 0.]
            k = ap/np.linalg.norm(ap); ca = ca*np.cos(a)+np.cross(k, ca)*np.sin(a)+k*(k@ca)*(1-np.cos(a))
            g['close_axis'] = ca
        if g['kind'] == 'rim':
            hull = f['hull']; c0, r0 = W['GR'].circle_fit(hull) if len(hull) > 5 else (f['center'], .05)
            if 'rim_ang' in v:
                d0 = np.asarray(g['center'])-c0; a = math.atan2(d0[1], d0[0])+math.radians(v['rim_ang'])
                p = c0+r0*np.array([np.cos(a), np.sin(a)]); g['center'] = p
                rad = (p-c0)/np.linalg.norm(p-c0); g['close_axis'] = rad   # radial, like 'toward' for the near point
            if 'rim_depth' in v: g['z'] = f['height']-min(v['rim_depth'], .3*f['height'])
            if 'rim_tilt' in v:
                rad = np.asarray(g['center'])-c0; rad = rad/np.linalg.norm(rad)
                t = v['rim_tilt']; g['approach3'] = list(np.r_[-np.sin(t)*rad, -np.cos(t)])   # >0: fingers lean inward
        out.append(g)
    return out

def eval_grasp(job):
    NODE, TR, GR = W['NODE'], W['TR'], W['GR']; arms = W['arms']
    rng = np.random.default_rng(job['seed'])
    pl = plane(); trays_xy = [TRAYS[k] for k in job['trays']]
    C_all = {n: cloud(o['cat'], o['dims'], *o['pose'], rng) for n, o in job['objs'].items()}
    name = job['name']; o = job['objs'][name]; cat = o['cat']
    others = [C_all[m] for m in job['objs'] if m != name]
    Wm = min(arms['l'].hand['opposed_open']-.01, .11)
    f, C = GR.candidates(C_all[name], pl, cat, others, W_MAX=Wm, robot_xy=(0., 0.))
    tx, ty = TRAYS[job['tray']]
    rows = []
    for v in variants_for(cat):
        if time.time()-T_START > DEADLINE_S+300: break
        t0 = time.time(); res = 'no_candidate'; info = {}
        yp_ = v.get('yaw_pick', 0.)
        R = rotz(-yp_); pl_r = (R@pl[0], pl[1]); tr_r = [tuple((R@np.r_[x, y, 0.])[:2]) for x, y in trays_xy]
        for A in arms.values(): A.set_scene(pl_r, tr_r)
        cands = make_variant_cands(v, f, C, cat, C_all[name], pl)
        if cands: res = 'no_ik'
        order = 'lr' if ty > .1 else ('rl' if ty < -.1 else ('lr' if f['center'][1] >= 0 else 'rl'))
        for side in order:
            A = arms[side]
            for g in cands:
                g = dict(g)
                cen = np.asarray(g['center'], float); top = zt(*cen)
                if yp_:
                    cen2 = (R@np.r_[cen, 0.])[:2]; g['center'] = cen2
                    if g['kind'] == 'side':
                        g['approach'] = (R[:2, :2]@np.asarray(g['approach']))
                        ca = np.asarray(g['close_axis']); g['close_axis'] = (R@np.r_[ca[:2], ca[2] if len(ca) > 2 else 0.]) if len(ca) == 3 else R[:2, :2]@ca
                    else:
                        ca = np.asarray(g['close_axis']); ca3 = np.r_[ca, 0.] if len(ca) == 2 else ca; g['close_axis'] = R@ca3
                        if 'approach3' in g: g['approach3'] = list(R@np.asarray(g['approach3'], float))
                    top = zt(*cen2, n=pl_r[0], d=pl_r[1])
                if g['kind'] == 'top':
                    kind_tilts = (0,)
                else:
                    kind_tilts = v.get('tilts', (0, .35, .6))
                p, dz = try_pick(A, g, f, top, tilts=kind_tilts, point=v.get('point'), pre=v.get('pre', .08),
                                 side_dz=(0.,) if g['kind'] == 'top' else None)
                if p is None: continue
                res = 'no_place'
                # place: plan in the yawed frame then (if yaw_pick) the torso must also turn back/forward: same planner
                ypl = NODE.plan_place_yaw(A, p['q_lift'], *(R@np.r_[tx, ty, 0.])[:2], pl_r, tr_r)
                if ypl is None: continue
                res = 'ok'; info = dict(side=side, kind=g['kind'], pick_dur=round(p['duration'], 2), place_yaw=round(float(ypl[0]), 3), tilt=p['tilt'])
                break
            if res == 'ok': break
        rows.append(dict(layout=job['layout'], name=name, cat=cat, tray=job['tray'], variant=v['id'], result=res,
                         plan_s=round(time.time()-t0, 2), **info))
    for A in arms.values(): A.set_scene(pl, trays_xy)
    return rows

# ---------------- tubes (d) ----------------
def eval_tubes(job):
    """IK + collision feasibility: grasp a standing tube (side, closing across the body, 2 heights; top-down on cap),
    yaw torso left, place vertically over each hood rack hole (bottom 2 cm above the rack top plate)."""
    ARM, NODE = W['ARM'], W['NODE']; arms = W['arms']; from team048_solution.planning import Sphere
    pl = plane(); rows = []
    for A in arms.values(): A.set_scene(pl, [])
    tube = job['tube']; L, d = tube['length'], tube['d']
    for (sx, sy) in job['slots']:
        top = zt(sx, sy); base = top+.004   # rack plate on the table, tube bottom at base+0.0015
        rk_top = top+.072+.004
        rec = dict(kind='table_slot', rack=job['rack'], slot=[round(sx, 3), round(sy, 3)], tube=job['tube_name'])
        for side in 'lr':
            A = arms[side]; best = None
            for mode, zc, ap in (('side_hi', rk_top+.025, None), ('side_lo', rk_top+.012, None), ('top_cap', base+L+.01, [0, 0, -1.])):
                if mode.startswith('side'):
                    toward = -np.array([sx, sy])/np.hypot(sx, sy); appr = -toward
                    ca = np.r_[-appr[1], appr[0], 0.]; ap3 = np.r_[appr, 0.]
                else:
                    ca = np.array([0, 1., 0]); ap3 = np.array(ap)
                p = A.plan_pick(np.array([sx, sy, zc]), ca, ap3, d, point='grip' if mode != 'top_cap' else 'pinch',
                                tilts=(0, .2, .35) if mode.startswith('side') else (0,), lift=.15)
                if p is not None: best = (mode, p); break
            rec[side] = None if best is None else dict(mode=best[0], dur=round(best[1]['duration'], 2))
            if best is not None: rec['_q_'+side] = best[1]['q_lift']; rec['_mode_'+side] = best[0]
        rows.append(rec)
    # place: hood holes; torso yaw th>0 (left); scene: hood work surface z=+0.02 (pelvis) as plane in the region
    for (hx, hy) in job['hood_slots']:
        rec = dict(kind='hood_slot', rack=job['hood_rack'], slot=[round(hx, 3), round(hy, 3)], tube=job['tube_name'])
        for side in 'lr':
            A = arms[side]; ok = None
            for th in (1.57, 1.3, 1.8, 1.1, 2.0, 2.3):
                R = rotz(-th); p2 = (R@np.r_[hx, hy, 0.])[:2]
                env = list(A.body)
                # hood work surface (rotated into the torso frame) as spheres tangent to z=.02, plus rack top plate edges
                for a in np.arange(-.70, .18, .04):
                    for b in np.arange(.53, 1.28, .04):
                        q = R@np.r_[a, b, 0.]; env.append(Sphere('pelvis', np.array([q[0], q[1], .02-.06]), .06))
                A.col.environment = env
                zb = .02+.004+.072+.004+.02     # tube bottom 2 cm above the hood rack top plate
                zc = zb+.09                     # grip 9 cm above the bottom (above the cone, below the cap)
                appr = np.array([1., 0.]) if True else None
                for ang in (0., .4, -.4, .8):
                    a2 = np.array([np.cos(ang), np.sin(ang)])
                    T = A.grasp_pose(np.array([p2[0], p2[1], zc]), np.r_[-a2[1], a2[0], 0.], np.r_[a2, 0.])
                    q = A.solve(T)
                    if q is not None and A.col.certify(q, A.home):
                        ok = dict(yaw=th, appr_ang=ang); break
                if ok: break
            rec[side] = ok
        for A in arms.values(): A.set_scene(pl, [])
        rows.append(rec)
    for r in rows:
        for k in [k for k in r if k.startswith('_q_')]: r.pop(k)
    return rows

# ---------------- driver ----------------
def run():
    LOG.write_text('') if not LOG.exists() else None
    src = setup(); log(f'src={src} cfg={json.dumps({k: v for k, v in CFG.items() if k != "pools"})}')
    mode = CFG['mode']; jobs = []
    meta = CFG['meta']   # name -> [cat, [dx,dy,dz]]
    if mode in ('reach', 'grasp'):
        task = CFG['task']; trays = TASK_TRAYS[task]; cats = CFG['cats']   # category order == /sim/task categories order
        cat2tray = {c: trays[i] for i, c in enumerate(cats)}
        names = [n for c in cats for n in CFG['pool'][c]]
        dims = {n: meta[n][1] for n in names}
        for li in range(CFG['l0'], CFG['l1']):
            seed = CFG.get('seed_base', 70000)+li
            lay = sample_layout(names, dims, seed, trays)
            objs = {n: dict(cat=meta[n][0], dims=dims[n], pose=lay[n]) for n in names if lay[n] != 'park'}
            for n in objs:
                if mode == 'grasp' and objs[n]['cat'] not in CFG['grasp_cats']: continue
                jobs.append(dict(layout=li, seed=seed*31+zlib.crc32(n.encode()) % 1000, name=n, objs=objs, trays=trays, tray=cat2tray[objs[n]['cat']]))
        fn = eval_obj if mode == 'reach' else eval_grasp
    else:
        fn = eval_tubes; jobs = CFG['jobs']
    log(f'{len(jobs)} jobs, {mp.cpu_count()} cpus')
    done = 0
    with mp.get_context('fork').Pool(CFG.get('procs', mp.cpu_count()), initializer=winit, initargs=(str(src),)) as P:
        it = P.imap_unordered(fn, jobs)
        with RES.open('a') as fo:
            while True:
                if time.time()-T_START > DEADLINE_S:
                    log(f'DEADLINE after {done} jobs, terminating'); P.terminate(); break
                try: r = it.next(timeout=30)
                except mp.TimeoutError: continue
                except StopIteration: break
                for row in (r if isinstance(r, list) else [r]):
                    fo.write(json.dumps(row, default=float)+'\n')
                fo.flush(); done += 1
                if done % 10 == 0: log(f'{done}/{len(jobs)} done')
    (OUT/'status.json').write_text(json.dumps(dict(done=done, total=len(jobs), seconds=round(time.time()-T_START, 1),
                                                   complete=done == len(jobs))))
    log(f'END done={done}/{len(jobs)}')

if __name__ == '__main__':
    try: run()
    except Exception:
        log('FAIL '+traceback.format_exc()); (OUT/'status.json').write_text(json.dumps(dict(error=traceback.format_exc())))
