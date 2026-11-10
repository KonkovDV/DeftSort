"""Heavy TPU-VM CPU workload for Team048. Private."""
import json, os, sys, time, zipfile, traceback, shutil
from pathlib import Path
IN = Path("/kaggle/input/team048-tpu-cpu-inputs")
OUT = Path("/kaggle/working")
LOG = OUT / "cpuwork.log"
N3 = int(os.environ.get("N_LAYOUTS_P3", "300"))
N6 = int(os.environ.get("N_LAYOUTS_P6", "300"))

def log(m):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {m}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")

def find_input():
    base = Path('/kaggle/input')
    if not base.exists():
        raise FileNotFoundError('no /kaggle/input')
    # New Kaggle layout nests user datasets under /kaggle/input/datasets/<user>/<slug>
    cands = []
    for p in base.rglob('features.json'):
        # perception/features.json -> dataset root is parent of perception
        if p.parent.name == 'perception':
            cands.append(p.parent.parent)
    for p in base.rglob('arm.py'):
        if p.parent.name == 'solution_pkg':
            cands.append(p.parent.parent)
    for p in base.rglob('cam_head_color.npy'):
        cands.append(p.parent.parent)
    # also accept direct slug mounts
    for p in [base/'team048-tpu-cpu-inputs', base/'datasets'/'dmitrijkv'/'team048-tpu-cpu-inputs']:
        if p.exists():
            cands.append(p)
    # unique preserve order
    seen=set(); out=[]
    for c in cands:
        s=str(c)
        if s not in seen:
            seen.add(s); out.append(c)
    if not out:
        listing=[]
        for p in base.rglob('*'):
            listing.append(str(p))
            if len(listing)>=40: break
        raise FileNotFoundError('dataset tree not found; sample='+str(listing))
    return out[0]

def materialize(IN, work):
    import shutil, zipfile
    work.mkdir(parents=True, exist_ok=True)
    for name in ('perception', 'solution_pkg', 'v5_frames', 'ik'):
        src = IN / name
        dst = work / name
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                dst.write_bytes(src.read_bytes())
    for z in list(IN.glob('*.zip')) + list(IN.glob('**/*.zip')):
        dest = work / z.stem
        if not dest.exists():
            dest.mkdir(parents=True, exist_ok=True)
            zipfile.ZipFile(z).extractall(dest)
    return work


N_LAYOUTS=int(os.environ.get('N_LAYOUTS','120')); N_OBJ=8; POOL=3; SEED0=30_000

def log(m):
    line=f"{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())} {m}"; print(line,flush=True); LOG.open('a').write(line+'\n')


def unpack():
    global IN
    IN = find_input()
    log(f"IN={IN}")
    work = materialize(IN, OUT / "in")
    sol = work / "solution_pkg"
    if not (sol / "arm.py").exists():
        c = list(work.glob("**/arm.py"))
        sol = c[0].parent if c else sol
    v5 = work / "v5_frames"
    if not (v5 / "cam_head_color.npy").exists():
        c = list(work.glob("**/cam_head_color.npy"))
        v5 = c[0].parent if c else v5
    return sol, v5

def build_pkg(sol):
    pkg = OUT / "team048_solution"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for fn in ("geometry.py", "kinematics.py", "planning.py", "perception.py", "grasp.py", "arm.py"):
        src = sol / fn
        if src.exists():
            (pkg / fn).write_text(src.read_text())
    datap = pkg / "data"
    datap.mkdir(exist_ok=True)
    urdf = sol / "data" / "tiangong_pro_with_hands.urdf"
    if urdf.exists():
        (datap / urdf.name).write_bytes(urdf.read_bytes())
    priors = sol / "data" / "priors.json"
    if priors.exists():
        (datap / "priors.json").write_text(priors.read_text())
    if not (pkg / "arm.py").exists():
        raise FileNotFoundError("arm.py missing")
    sys.path.insert(0, str(OUT))
    os.environ["HACK_URDF"] = str(datap / "tiangong_pro_with_hands.urdf")
    return datap / "tiangong_pro_with_hands.urdf"

def plan_pool(A, n_layouts, n_obj, seed0, table_z, np):
    t1 = time.time()
    success = 0
    rejected = 0
    speeds = []
    for seed in range(n_layouts):
        rr = np.random.default_rng(seed0 + seed)
        for i in range(n_obj):
            xy = np.array([rr.uniform(0.26, 0.50), rr.uniform(-0.52, 0.52), table_z + 0.04])
            p = A.plan_pick(xy, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), float(rr.uniform(0.03, 0.09)))
            if p:
                success += 1
                speeds.append(float(p.get("max_speed", 0)))
            else:
                rejected += 1
    return {
        "layouts": n_layouts,
        "attempts": n_layouts * n_obj,
        "success": success,
        "rejected": rejected,
        "success_frac": round(success / max(n_layouts * n_obj, 1), 4),
        "max_speed_p95": float(np.percentile(speeds, 95)) if speeds else None,
        "speed_cap_ok": bool(all(s <= 0.8 + 1e-6 for s in speeds)),
        "seconds": round(time.time() - t1, 3),
    }

def main():
    LOG.write_text("")
    rep = {"ok": False}
    try:
        import numpy as np
        sol, v5 = unpack()
        build_pkg(sol)
        from team048_solution.arm import Arm
        from team048_solution.perception import table_plane, segment
        A = Arm("l")
        table_z = 0.70 - 0.88
        rng = np.random.default_rng(42)
        ok = 0
        n = 500
        errs = []
        t0 = time.time()
        for _ in range(n):
            q = rng.uniform(A.chain.lower, A.chain.upper) * 0.4
            T = A.chain.fk(q)
            try:
                qq = A.chain.ik(T, A.home, max_iter=300)
                e = float(np.linalg.norm(A.chain.fk(qq)[:3, 3] - T[:3, 3]))
                errs.append(e)
                ok += e < 0.003
            except Exception:
                pass
        rep["fk_ik"] = {
            "n": n,
            "ok_lt_3mm": ok,
            "ok_frac": round(ok / n, 4),
            "err_median_m": float(np.median(errs)) if errs else None,
            "err_p95_m": float(np.percentile(errs, 95)) if errs else None,
            "seconds": round(time.time() - t0, 3),
        }
        log(str(rep["fk_ik"]))
        rep["plan_pool3"] = plan_pool(A, N3, 8, 30000, table_z, np)
        log(str(rep["plan_pool3"]))
        rep["plan_pool6"] = plan_pool(A, N6, 12, 60000, table_z, np)
        log(str(rep["plan_pool6"]))
        v5rep = {"present": (v5 / "cam_head_color.npy").exists()}
        if v5rep["present"]:
            color = np.load(v5 / "cam_head_color.npy")
            depth = np.load(v5 / "cam_head_depth.npy")
            d = depth.astype(np.float64)
            if d.ndim == 3 and d.shape[-1] == 1:
                d = d[..., 0]
            Z = d / 1000.0 if np.nanmedian(d[d > 0]) > 20 else d
            h, w = Z.shape
            fx = 0.5 * w / np.tan(np.deg2rad(86) / 2)
            cx, cy = (w - 1) / 2, (h - 1) / 2
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fx
            P = np.stack([Z, -X, -Y], -1)
            valid = np.isfinite(P).all(-1) & (Z > 0.15) & (Z < 3.0)
            rgb = color
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, -1)
            try:
                plane = table_plane(P, valid, iterations=80, tolerance=0.01, min_inliers=80)
                dets = segment(P, rgb, valid, plane, (np.array([-1.0, -1.0, -1.0]), np.array([2.0, 2.0, 2.0])), min_pixels=40)
                v5rep.update({"plane_ok": True, "n_dets": len(dets), "depth_valid_frac": float(valid.mean()), "shapes": [list(color.shape), list(depth.shape)]})
            except Exception as e:
                v5rep.update({"plane_ok": False, "plane_error": repr(e), "depth_valid_frac": float(valid.mean())})
        rep["v5"] = v5rep
        rep["ok"] = True
        log("ALL OK")
    except Exception as e:
        rep["error"] = repr(e)
        rep["traceback"] = traceback.format_exc()
        log("FAIL " + repr(e))
        log(rep["traceback"])
    (OUT / "cpuwork_results.json").write_text(json.dumps(rep, indent=2))
    return 0 if rep.get("ok") else 1

if __name__ == "__main__":
    raise SystemExit(main())
