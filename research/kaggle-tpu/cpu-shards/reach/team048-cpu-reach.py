"""Reachability study on NEW solution (5117bec): pick + place(tray_left/right) per class. CPU only."""
import json, os, sys, time, traceback, collections
from pathlib import Path
import numpy as np

OUT = Path('/kaggle/working'); LOG = OUT/'reach.log'
POOL = int(os.environ.get('POOL', '3'))
N_LAYOUTS = int(os.environ.get('N_LAYOUTS', '40'))
SEED0 = int(os.environ.get('SEED0', str(100000*POOL)))
TABLE_TOP = 0.70 - 0.88  # relative to standing pelvis

def log(m):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {m}"
    print(line, flush=True)
    LOG.open('a').write(line+'\n')

def find_input():
    base = Path('/kaggle/input')
    cands = []
    for p in base.rglob('arm.py'):
        if p.parent.name == 'solution_pkg':
            cands.append(p.parent.parent)
    for p in [base/'team048-tpu-cpu-inputs', base/'datasets'/'dmitrijkv'/'team048-tpu-cpu-inputs']:
        if p.exists(): cands.append(p)
    if not cands:
        listing = [str(x) for i,x in zip(range(40), base.rglob('*'))]
        raise FileNotFoundError('no input: '+str(listing))
    return cands[0]

def materialize(IN, work):
    import shutil, zipfile
    work.mkdir(parents=True, exist_ok=True)
    for name in ('solution_pkg', 'perception', 'v5_frames', 'ik'):
        src, dst = IN/name, work/name
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst) if src.is_dir() else dst.write_bytes(src.read_bytes())
    for z in list(IN.glob('*.zip')):
        dest = work/z.stem
        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)
            zipfile.ZipFile(z).extractall(dest)
    for fn in ('pools.yaml', 'objects.yaml'):
        src = IN/fn
        if src.exists():
            (work/fn).write_text(src.read_text())
    return work

def load_yaml(path):
    try:
        import yaml
        return yaml.safe_load(open(path))
    except Exception:
        # minimal YAML subset for our files
        text = Path(path).read_text()
        # Prefer PyYAML from Kaggle image; if missing, pip not available (no internet). Bundle json instead.
        raise

def synth_cloud(xy, yaw, dims_m, category, n=400, rng=None):
    """Synthetic tabletop cloud in pelvis: x forward, z up. dims_m = (dx,dy,dz) object extents."""
    rng = rng or np.random.default_rng(0)
    dx, dy, dz = dims_m
    # footprint samples
    if category == 'jar':
        r = 0.5*min(dx, dy)
        th = rng.uniform(0, 2*np.pi, n)
        rr = r * np.sqrt(rng.uniform(0, 1, n))
        local = np.c_[rr*np.cos(th), rr*np.sin(th), rng.uniform(0.002, dz, n)]
    else:
        local = np.c_[rng.uniform(-dx/2, dx/2, n), rng.uniform(-dy/2, dy/2, n), rng.uniform(0.002, dz, n)]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    xy2 = (R @ local[:, :2].T).T + np.asarray(xy)
    return np.c_[xy2, local[:, 2] + TABLE_TOP]

def build_pkg(sol):
    pkg = OUT/'team048_solution'; pkg.mkdir(exist_ok=True)
    (pkg/'__init__.py').write_text('')
    for fn in ('geometry.py','kinematics.py','planning.py','perception.py','grasp.py','arm.py','trays.py','budget.py','grip.py','sensing.py'):
        src = sol/fn
        if src.exists(): (pkg/fn).write_text(src.read_text())
    # plan_place lives in node.py — extract without ROS deps
    node = (sol/'node.py').read_text()
    # keep only plan_place + constants it needs
    stub = '''import numpy as np
from team048_solution.trays import TRAY_H
RETREAT_DZ=.08
PLACE_AXES=(np.array([1.,0,0]),np.array([0,1.,0]),np.array([-1.,0,0]),np.array([0,-1.,0]))
'''
    # grab plan_place function body from node
    start = node.index('def plan_place')
    end = node.index('\ndef self_w') if '\ndef self_w' in node else node.index('\ndef main')
    stub += node[start:end]
    (pkg/'place.py').write_text(stub)
    datap = pkg/'data'; datap.mkdir(exist_ok=True)
    urdf = sol/'data'/'tiangong_pro_with_hands.urdf'
    if urdf.exists(): (datap/urdf.name).write_bytes(urdf.read_bytes())
    priors = sol/'data'/'priors.json'
    if priors.exists(): (datap/'priors.json').write_text(priors.read_text())
    sys.path.insert(0, str(OUT))
    os.environ['HACK_URDF'] = str(datap/'tiangong_pro_with_hands.urdf')

def main():
    LOG.write_text('')
    rep = {'ok': False, 'pool': POOL, 'n_layouts': N_LAYOUTS, 'commit_note': '5117bec-local'}
    try:
        IN = find_input(); log(f'IN={IN}')
        work = materialize(IN, OUT/'in')
        sol = work/'solution_pkg'
        if not (sol/'arm.py').exists():
            c = list(work.glob('**/arm.py')); sol = c[0].parent if c else sol
        # yaml paths
        pools_p = work/'pools.yaml'
        objs_p = work/'objects.yaml'
        if not pools_p.exists():
            c = list(work.glob('**/pools.yaml')); pools_p = c[0] if c else pools_p
        if not objs_p.exists():
            c = list(work.glob('**/objects.yaml')); objs_p = c[0] if c else objs_p
        # Parse yaml without PyYAML if needed
        try:
            import yaml
            pools = yaml.safe_load(open(pools_p))['pools'][str(POOL)]
            cats_meta = yaml.safe_load(open(objs_p))['categories']
        except ImportError:
            # fall back: ship a tiny json next to script — or use regex. Prefer json dump we embed.
            raise RuntimeError('PyYAML required')
        dims = {}
        for cat, meta in cats_meta.items():
            for o in meta['objects']:
                d = o['dims_cm']; dims[o['name']] = (cat, np.array(d)/100.)  # metres
        # flatten pool object list
        pool_objs = []
        for cat, names in pools.items():
            for n in names:
                pool_objs.append((n, cat, dims[n][1] if n in dims else np.array([.05,.05,.08])))
        log(f'pool_objs={[(n,c) for n,c,_ in pool_objs]}')

        build_pkg(sol)
        from team048_solution.arm import Arm
        from team048_solution.grasp import candidates
        from team048_solution.place import plan_place
        from team048_solution import trays as TR
        arms = {'l': Arm('l'), 'r': Arm('r')}
        plane = (np.array([0.,0.,1.]), -TABLE_TOP)  # z=TABLE_TOP
        W_MAX = min(arms['l'].hand['opposed_open']-.01, .11)

        # stats[class] = counters
        stats = collections.defaultdict(lambda: dict(
            n=0, pick_ok=0, pick_fail=0,
            place_left_ok=0, place_right_ok=0, place_both_ok=0,
            pick_place_left=0, pick_place_right=0,
            by_kind=collections.Counter()))
        t_wall = time.time()
        for li in range(N_LAYOUTS):
            rng = np.random.default_rng(SEED0 + li)
            # spawn all pool objects with random xy/yaw, simple non-overlap reject
            placed = []
            for name, cat, dim in pool_objs:
                for _ in range(40):
                    xy = np.array([rng.uniform(0.26, 0.50), rng.uniform(-0.52, 0.52)])
                    yaw = rng.uniform(-np.pi, np.pi)
                    if all(np.linalg.norm(xy-p['xy']) > 0.08 for p in placed):
                        placed.append(dict(name=name, cat=cat, dim=dim, xy=xy, yaw=yaw)); break
            clouds = {p['name']: synth_cloud(p['xy'], p['yaw'], p['dim'], p['cat'], rng=rng) for p in placed}
            for p in placed:
                st = stats[p['cat']]; st['n'] += 1
                others = [clouds[q['name']] for q in placed if q['name'] != p['name']]
                f, C = candidates(clouds[p['name']], plane, p['cat'], others, W_MAX=W_MAX, robot_xy=(0.,0.))
                # arm order like node: by object y
                order = 'lr' if p['xy'][1] >= 0 else 'rl'
                pick = None; side_used = None; kind = None
                for side in order:
                    A = arms[side]
                    for g in C:
                        if not g['feasible']: continue
                        top = TABLE_TOP
                        c3 = np.array([g['center'][0], g['center'][1], top + g['z']])
                        ap = np.r_[g['approach'], 0.] if g['kind'] == 'side' else np.array([0., 0., -1.])
                        ca = np.r_[g['close_axis'], 0.]
                        plan = A.plan_pick(c3, ca, ap, g['width'])
                        if plan is None: continue
                        pick = plan; side_used = side; kind = g['kind']; break
                    if pick is not None: break
                if pick is None:
                    st['pick_fail'] += 1; continue
                st['pick_ok'] += 1; st['by_kind'][kind] += 1
                A = arms[side_used]
                top = TABLE_TOP
                # place both trays, slot 0 + fallback centre
                left_xy = TR.NOMINAL['tray_left']; right_xy = TR.NOMINAL['tray_right']
                dy = TR.slot_offset(0, .5 * f['length'])
                pl = plan_place(A, pick['q_lift'], left_xy[0], left_xy[1]+dy, top, fallback=left_xy)
                pr = plan_place(A, pick['q_lift'], right_xy[0], right_xy[1]+dy, top, fallback=right_xy)
                if pl is not None: st['place_left_ok'] += 1; st['pick_place_left'] += 1
                if pr is not None: st['place_right_ok'] += 1; st['pick_place_right'] += 1
                if pl is not None and pr is not None: st['place_both_ok'] += 1
            if (li+1) % 5 == 0:
                log(f'layout {li+1}/{N_LAYOUTS} elapsed={time.time()-t_wall:.1f}s')

        # rates
        rates = {}
        for cat, st in stats.items():
            n = max(st['n'], 1)
            rates[cat] = {
                'n': st['n'],
                'pick_rate': round(st['pick_ok']/n, 4),
                'place_left_given_pick': round(st['place_left_ok']/max(st['pick_ok'],1), 4),
                'place_right_given_pick': round(st['place_right_ok']/max(st['pick_ok'],1), 4),
                'place_both_given_pick': round(st['place_both_ok']/max(st['pick_ok'],1), 4),
                'pick_and_place_left': round(st['pick_place_left']/n, 4),
                'pick_and_place_right': round(st['pick_place_right']/n, 4),
                'pick_kinds': dict(st['by_kind']),
            }
        rep.update({'rates': rates, 'seconds': round(time.time()-t_wall, 2), 'ok': True,
                    'hand_open': float(arms['l'].hand['opposed_open']),
                    'trays': TR.NOMINAL})
        log(json.dumps(rates, indent=2)); log('ALL OK')
    except Exception as e:
        rep['error'] = repr(e); rep['traceback'] = traceback.format_exc(); log('FAIL '+repr(e)); log(rep['traceback'])
    (OUT/'reach_results.json').write_text(json.dumps(rep, indent=2))
    return 0 if rep.get('ok') else 1

if __name__ == '__main__':
    raise SystemExit(main())
