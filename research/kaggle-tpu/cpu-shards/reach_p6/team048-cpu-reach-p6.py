"""Reachability study on NEW solution (5117bec): pick + place(tray_left/right) per class. CPU only."""
import json, os, sys, time, traceback, collections
from pathlib import Path
import numpy as np

OUT = Path('/kaggle/working'); LOG = OUT/'reach.log'
POOL = int(os.environ.get('POOL', '6'))
N_LAYOUTS = int(os.environ.get('N_LAYOUTS', '30'))
SEED0 = int(os.environ.get('SEED0', str(100000*POOL)))
TABLE_TOP = 0.70 - 0.88  # relative to standing pelvis
POOLS_JSON = json.loads(r'''{"pools": {"1": {"jar": ["Weston_No_22_Cajun_Jerky_Tonic_12_fl_oz_nLj64ZnGwDh", "005_tomato_soup_can", "YumYum_D3_Liquid"], "box": ["004_sugar_box", "Just_For_Men_Mustache_Beard_Brushin_Hair_Color_Gel_Kit_Jet_Black_M60", "Office_Depot_Canon_CLI_221BK_Ink_Cartridge_Black_2946B001"], "figure": ["3D_Dollhouse_Lamp", "3D_Dollhouse_TablePurple", "3D_Dollhouse_Happy_Brother"]}, "2": {"electronics": ["Deskstar_Desk_Top_Hard_Drive_1_TB", "Razer_Taipan_Black_Ambidextrous_Gaming_Mouse", "Ubisoft_RockSmith_Real_Tone_Cable_Xbox_360"], "jar": ["Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion", "Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold", "Beta_Glucan"], "figure": ["Schleich_S_Bayala_Unicorn_70432", "Schleich_Lion_Action_Figure", "Schleich_Bald_Eagle"]}, "3": {"dish": ["Utana_5_Porcelain_Ramekin_Large", "Threshold_Ramekin_White_Porcelain", "Ecoforms_Plant_Container_B4_Har"], "box": ["Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box", "Crayola_Bonus_64_Crayons", "Perricone_MD_AcylGlutathione_Eye_Lid_Serum"], "jar": ["006_mustard_bottle", "Saccharomyces_Boulardii_MOS_Value_Size", "Hyaluronic_Acid"]}, "4": {"figure": ["Schleich_African_Black_Rhino", "3D_Dollhouse_Sink", "LADYBUG_BEAD"], "dish": ["Threshold_Bamboo_Ceramic_Soap_Dish", "Cole_Hardware_Bowl_Scirocco_YellowBlue", "Ecoforms_Cup_B4_SAN"], "box": ["Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g", "Brother_Ink_Cartridge_Magenta_LC75M", "Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack"]}, "5": {"jar": ["Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original", "Black_Elderberry_Syrup_54_oz_Gaia_Herbs", "AllergenFree_JarroDophilus"], "figure": ["Dino_4", "Schleich_Hereford_Bull", "Android_Figure_Chrome"], "dish": ["Cole_Hardware_Saucer_Electric", "025_mug", "45oz_RAMEKIN_ASST_DEEP_COLORS"]}, "6": {"electronics": ["Razer_Taipan_White_Ambidextrous_Gaming_Mouse", "JBL_Charge_Speaker_portable_wireless_wired_Green", "Crosley_Alarm_Clock_Vintage_Metal"], "box": ["Nestle_Nips_Hard_Candy_Peanut_Butter", "Nips_Hard_Candy_Rich_Creamy_Butter_Rum_4_oz_1133_g", "009_gelatin_box"]}}}''')
OBJECTS_JSON = json.loads(r'''{"categories": {"jar": {"objects": [{"name": "5_HTP", "mass": 0.12, "dims_cm": [5.3, 5.2, 8.9]}, {"name": "Beta_Glucan", "mass": 0.12, "dims_cm": [5.4, 5.4, 9.8]}, {"name": "Hyaluronic_Acid", "mass": 0.12, "dims_cm": [5.4, 5.4, 9.8]}, {"name": "AllergenFree_JarroDophilus", "mass": 0.1, "dims_cm": [5.4, 5.4, 9.6]}, {"name": "Saccharomyces_Boulardii_MOS_Value_Size", "mass": 0.15, "dims_cm": [7.5, 7.4, 13.5]}, {"name": "YumYum_D3_Liquid", "mass": 0.06, "dims_cm": [3.3, 3.4, 10.0]}, {"name": "Black_Elderberry_Syrup_54_oz_Gaia_Herbs", "mass": 0.22, "dims_cm": [5.7, 5.9, 14.5]}, {"name": "Perricone_MD_Firming_Neck_Therapy_Treatment", "mass": 0.15, "dims_cm": [6.1, 6.1, 7.3]}, {"name": "Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold", "mass": 0.15, "dims_cm": [4.6, 4.6, 13.5]}, {"name": "005_tomato_soup_can", "mass": 0.35, "dims_cm": [6.8, 6.8, 10.2]}, {"name": "006_mustard_bottle", "mass": 0.43, "dims_cm": [9.7, 6.7, 19.1]}, {"name": "Weston_No_22_Cajun_Jerky_Tonic_12_fl_oz_nLj64ZnGwDh", "mass": 0.35, "dims_cm": [6.4, 6.2, 19.9]}, {"name": "Phillips_Milk_of_Magnesia_Saline_Laxative_Liquid_Original", "mass": 0.4, "dims_cm": [7.3, 5.7, 17.4]}, {"name": "Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion", "mass": 0.25, "dims_cm": [5.7, 5.1, 19.5]}]}, "box": {"objects": [{"name": "004_sugar_box", "mass": 0.51, "dims_cm": [5.0, 9.4, 17.6]}, {"name": "009_gelatin_box", "mass": 0.1, "dims_cm": [8.9, 10.1, 3.0]}, {"name": "Brother_Ink_Cartridge_Magenta_LC75M", "mass": 0.06, "dims_cm": [8.8, 3.1, 14.7]}, {"name": "Office_Depot_Canon_CLI_221BK_Ink_Cartridge_Black_2946B001", "mass": 0.05, "dims_cm": [8.5, 5.0, 11.8]}, {"name": "Phillips_Caplets_Size_24", "mass": 0.05, "dims_cm": [7.9, 4.5, 8.6]}, {"name": "Just_For_Men_Mustache_Beard_Brushin_Hair_Color_Gel_Kit_Jet_Black_M60", "mass": 0.1, "dims_cm": [8.0, 4.6, 14.4]}, {"name": "Perricone_MD_AcylGlutathione_Eye_Lid_Serum", "mass": 0.08, "dims_cm": [8.0, 5.6, 10.8]}, {"name": "Philips_EcoVantage_43_W_Light_Bulbs_Natural_Light_2_pack", "mass": 0.15, "dims_cm": [10.7, 6.7, 10.8]}, {"name": "Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g", "mass": 0.2, "dims_cm": [9.2, 6.4, 16.1]}, {"name": "Nestle_Nips_Hard_Candy_Peanut_Butter", "mass": 0.13, "dims_cm": [8.8, 4.2, 15.3]}, {"name": "Nips_Hard_Candy_Rich_Creamy_Butter_Rum_4_oz_1133_g", "mass": 0.13, "dims_cm": [8.9, 4.3, 15.3]}, {"name": "Crayola_Bonus_64_Crayons", "mass": 0.35, "dims_cm": [14.7, 4.3, 12.7]}, {"name": "Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box", "mass": 0.22, "dims_cm": [13.9, 6.3, 16.4]}]}, "dish": {"objects": [{"name": "BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup", "mass": 0.15, "dims_cm": [8.9, 8.9, 4.3]}, {"name": "45oz_RAMEKIN_ASST_DEEP_COLORS", "mass": 0.2, "dims_cm": [9.0, 9.0, 4.4]}, {"name": "Threshold_Ramekin_White_Porcelain", "mass": 0.2, "dims_cm": [11.2, 11.2, 5.3]}, {"name": "Utana_5_Porcelain_Ramekin_Large", "mass": 0.35, "dims_cm": [13.2, 13.2, 7.0]}, {"name": "Ecoforms_Cup_B4_SAN", "mass": 0.1, "dims_cm": [10.0, 10.0, 7.2]}, {"name": "Ecoforms_Plant_Container_B4_Har", "mass": 0.1, "dims_cm": [10.1, 10.1, 7.2]}, {"name": "Cole_Hardware_Bowl_Scirocco_YellowBlue", "mass": 0.25, "dims_cm": [11.3, 11.5, 6.2]}, {"name": "Threshold_Bamboo_Ceramic_Soap_Dish", "mass": 0.2, "dims_cm": [12.5, 9.8, 5.1]}, {"name": "Cole_Hardware_Saucer_Electric", "mass": 0.2, "dims_cm": [12.4, 12.4, 3.2]}, {"name": "025_mug", "mass": 0.12, "dims_cm": [11.7, 9.3, 8.1]}]}, "figure": {"objects": [{"name": "Schleich_Bald_Eagle", "mass": 0.06, "dims_cm": [7.6, 7.1, 9.0]}, {"name": "Schleich_African_Black_Rhino", "mass": 0.08, "dims_cm": [4.1, 13.8, 6.0]}, {"name": "Schleich_Hereford_Bull", "mass": 0.08, "dims_cm": [4.4, 13.4, 8.2]}, {"name": "Schleich_Lion_Action_Figure", "mass": 0.07, "dims_cm": [11.5, 4.2, 7.0]}, {"name": "Schleich_S_Bayala_Unicorn_70432", "mass": 0.07, "dims_cm": [5.5, 14.8, 10.2]}, {"name": "Android_Figure_Chrome", "mass": 0.05, "dims_cm": [6.9, 4.6, 8.2]}, {"name": "Android_Figure_Panda", "mass": 0.05, "dims_cm": [6.1, 4.6, 7.6]}, {"name": "3D_Dollhouse_Happy_Brother", "mass": 0.03, "dims_cm": [5.9, 3.1, 9.6]}, {"name": "3D_Dollhouse_Lamp", "mass": 0.04, "dims_cm": [4.8, 4.7, 16.7]}, {"name": "3D_Dollhouse_Sink", "mass": 0.05, "dims_cm": [8.5, 7.3, 13.1]}, {"name": "3D_Dollhouse_TablePurple", "mass": 0.05, "dims_cm": [9.8, 9.0, 5.7]}, {"name": "Dino_4", "mass": 0.08, "dims_cm": [13.7, 5.9, 9.3]}, {"name": "LADYBUG_BEAD", "mass": 0.03, "dims_cm": [5.7, 8.4, 3.0]}]}, "electronics": {"objects": [{"name": "Razer_Abyssus_Ambidextrous_Gaming_Mouse", "mass": 0.1, "dims_cm": [10.1, 14.6, 3.8]}, {"name": "Razer_Taipan_Black_Ambidextrous_Gaming_Mouse", "mass": 0.12, "dims_cm": [12.2, 17.4, 4.0]}, {"name": "Razer_Taipan_White_Ambidextrous_Gaming_Mouse", "mass": 0.12, "dims_cm": [10.0, 18.2, 4.0]}, {"name": "Nikon_1_AW1_w11275mm_Lens_Silver", "mass": 0.35, "dims_cm": [6.3, 6.3, 7.9]}, {"name": "Crosley_Alarm_Clock_Vintage_Metal", "mass": 0.3, "dims_cm": [10.9, 5.4, 16.2]}, {"name": "JBL_Charge_Speaker_portable_wireless_wired_Green", "mass": 0.6, "dims_cm": [17.6, 7.3, 7.5]}, {"name": "Deskstar_Desk_Top_Hard_Drive_1_TB", "mass": 0.65, "dims_cm": [15.5, 7.7, 19.9]}, {"name": "Ubisoft_RockSmith_Real_Tone_Cable_Xbox_360", "mass": 0.15, "dims_cm": [17.2, 14.0, 4.4]}]}}, "reserve": [{"name": "003_cracker_box", "mass": 0.41, "dims_cm": [7.2, 16.4, 21.3]}, {"name": "002_master_chef_can", "mass": 0.41, "dims_cm": [10.2, 10.2, 14.0]}, {"name": "010_potted_meat_can", "mass": 0.37, "dims_cm": [10.2, 6.0, 8.4]}, {"name": "NESCAFE_NESCAFE_TC_STKS_DECAF_6_CT", "mass": 0.08, "dims_cm": [8.1, 2.9, 13.1]}, {"name": "040_large_marker", "mass": 0.02, "dims_cm": [2.1, 12.1, 1.9]}, {"name": "Craftsman_Grip_Screwdriver_Phillips_Cushion", "mass": 0.15, "dims_cm": [26.0, 3.6, 3.8]}, {"name": "Cole_Hardware_Hammer_Black", "mass": 0.6, "dims_cm": [29.2, 6.0, 9.6]}, {"name": "OXO_Cookie_Spatula", "mass": 0.05, "dims_cm": [24.0, 6.8, 2.5]}, {"name": "Big_O_Sponges_Assorted_Cellulose_12_pack", "mass": 0.05, "dims_cm": [7.0, 12.6, 1.9]}], "sets": {"demo": {"jar": ["5_HTP", "005_tomato_soup_can", "YumYum_D3_Liquid"], "box": ["004_sugar_box", "Phillips_Caplets_Size_24", "Brother_Ink_Cartridge_Magenta_LC75M"], "figure": ["Schleich_Bald_Eagle", "Android_Figure_Panda", "Dino_4"]}, "demo2": {"dish": ["Threshold_Ramekin_White_Porcelain", "025_mug", "Ecoforms_Cup_B4_SAN"], "electronics": ["Razer_Abyssus_Ambidextrous_Gaming_Mouse", "Crosley_Alarm_Clock_Vintage_Metal", "Nikon_1_AW1_w11275mm_Lens_Silver"], "jar": ["Beta_Glucan", "006_mustard_bottle", "Black_Elderberry_Syrup_54_oz_Gaia_Herbs"]}}, "spawn": {"area_x": [0.26, 0.5], "area_y": [-0.52, 0.52], "edge_margin": 0.03, "gap": 0.02, "drop": 0.005, "random_yaw": true, "parking": {"x0": -1.6, "y0": -1.0, "dx": -0.3, "dy": 0.4, "per_row": 6}}}''')



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
        pools = POOLS_JSON['pools'][str(POOL)]
        cats_meta = OBJECTS_JSON['categories']
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
