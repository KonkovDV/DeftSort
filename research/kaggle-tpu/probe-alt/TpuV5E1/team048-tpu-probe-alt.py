"""Short TPU-VM probe: machine specs + pip deps + tiny RandomForest timing. Private, internet on."""
import json, os, platform, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working/tpu_probe_results.json")
LOG = Path("/kaggle/working/tpu_probe.log")


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sh(cmd: str) -> str:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")


def main() -> int:
    LOG.write_text("")
    result = {"ok": False, "steps": {}}
    try:
        nproc = int(sh("nproc").strip() or "0")
        mem = sh("free -h || true")
        mem_kb = sh("awk '/MemTotal/ {print $2}' /proc/meminfo").strip()
        disk = sh("df -h /kaggle / | tail -n +1 || true")
        uname = sh("uname -a").strip()
        py = sys.version
        accel = sh("ls -la /dev/accel* 2>/dev/null || ls -la /dev/tpm* 2>/dev/null || echo 'no /dev/accel*'")
        xla = sh("python -c 'import torch_xla; print(torch_xla.__version__)' 2>&1 || true")
        result["machine"] = {
            "nproc": nproc,
            "mem_total_kb": int(mem_kb) if mem_kb.isdigit() else mem_kb,
            "mem_free_h": mem,
            "disk": disk,
            "uname": uname,
            "python": py,
            "platform": platform.platform(),
            "accel_devs": accel,
            "torch_xla": xla.strip(),
            "KAGGLE_KERNEL_RUN_TYPE": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
            "TPU_NAME": os.environ.get("TPU_NAME"),
            "TPU_WORKER_ID": os.environ.get("TPU_WORKER_ID"),
        }
        log(f"nproc={nproc} mem_kb={mem_kb} python={py.split()[0]}")
        log(f"uname={uname}")
        log(f"accel={accel.strip()[:200]}")

        t0 = time.time()
        # Prefer binary wheels; keep internet use short.
        pip_cmd = (
            f"{sys.executable} -m pip install -q --upgrade pip && "
            f"{sys.executable} -m pip install -q "
            "numpy scipy scikit-learn opencv-python-headless joblib"
        )
        pip_out = sh(pip_cmd)
        result["steps"]["pip"] = {
            "seconds": round(time.time() - t0, 2),
            "tail": pip_out[-1500:],
        }
        log(f"pip done in {result['steps']['pip']['seconds']}s")

        import numpy as np
        import scipy
        import sklearn
        import cv2
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        result["steps"]["imports"] = {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "cv2": cv2.__version__,
            "joblib": joblib.__version__,
        }
        log(f"imports ok: {result['steps']['imports']}")

        # Tiny RF fit timing (CPU path; TPU chips unused on purpose).
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4000, 64))
        y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
        t1 = time.time()
        clf = RandomForestClassifier(
            n_estimators=200, random_state=0, n_jobs=-1, min_samples_leaf=2
        )
        clf.fit(X, y)
        pred = clf.predict(X[:200])
        result["steps"]["rf_fit"] = {
            "seconds": round(time.time() - t1, 3),
            "n_estimators": 200,
            "n_samples": 4000,
            "n_features": 64,
            "n_jobs": -1,
            "pred_sum": int(pred.sum()),
            "score_train": float(clf.score(X, y)),
        }
        log(f"rf_fit={result['steps']['rf_fit']}")

        # Light OpenCV op
        img = (rng.integers(0, 255, (480, 640, 3))).astype(np.uint8)
        t2 = time.time()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        result["steps"]["opencv"] = {
            "seconds": round(time.time() - t2, 4),
            "edges_nonzero": int(np.count_nonzero(edges)),
        }

        result["ok"] = True
        result["wall_seconds"] = round(time.time() - t0, 2)
        log(f"ALL OK wall={result['wall_seconds']}s")
    except Exception as e:
        result["error"] = repr(e)
        result["traceback"] = traceback.format_exc()
        log(f"FAIL {e}")
        log(result["traceback"])

    OUT.write_text(json.dumps(result, indent=2))
    log(f"wrote {OUT}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
