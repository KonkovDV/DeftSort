//! PyO3 bindings. All heavy calls release the GIL (`py.detach`, formerly `allow_threads`) so the ROS executor
//! threads keep running while planning.
use crate::col::{Collision, Env};
use crate::kin::{Chain, Rt, N, Q};
use crate::perc;
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn q_of(a: &PyReadonlyArray1<f64>) -> PyResult<Q> {
    let s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("q must be contiguous"))?;
    if s.len() != N {
        return Err(PyValueError::new_err("q length"));
    }
    Ok(Q::from_column_slice(s))
}
fn rt_of(a: &PyReadonlyArray2<f64>) -> PyResult<Rt> {
    let v = a.as_array();
    if v.shape() != [4, 4] {
        return Err(PyValueError::new_err("4x4 expected"));
    }
    let mut m = [0.; 16];
    for i in 0..4 {
        for j in 0..4 {
            m[4 * i + j] = v[[i, j]];
        }
    }
    Ok(Rt::from_rows(&m))
}
fn qs_of(a: &PyReadonlyArray2<f64>) -> PyResult<Vec<Q>> {
    let v = a.as_array();
    if v.shape()[1] != N {
        return Err(PyValueError::new_err("seeds (k,7)"));
    }
    Ok((0..v.shape()[0])
        .map(|i| Q::from_fn(|k, _| v[[i, k]]))
        .collect())
}
fn slice<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    a: &'a numpy::PyReadonlyArray<'_, T, D>,
    what: &str,
) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous")))
}
fn v3(v: Vec<f64>, what: &str) -> PyResult<[f64; 3]> {
    v.try_into()
        .map_err(|_| PyValueError::new_err(format!("{what}: 3 values")))
}
fn rt_to_py<'py>(py: Python<'py>, t: &Rt) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((4, 4), t.to_rows().to_vec())
        .unwrap()
        .into_pyarray(py)
}

#[pyclass(module = "team048_accel", frozen)]
struct FastChain {
    c: Chain,
}

#[pymethods]
impl FastChain {
    #[new]
    fn new(
        origins: PyReadonlyArray3<f64>,
        axes: PyReadonlyArray2<f64>,
        movable: Vec<bool>,
        tcp: PyReadonlyArray2<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> PyResult<Self> {
        let o = origins.as_array();
        let a = axes.as_array();
        if o.shape()[1..] != [4, 4] || a.shape()[1] != 3 || o.shape()[0] != a.shape()[0] {
            return Err(PyValueError::new_err("origins (n,4,4), axes (n,3)"));
        }
        let mut ov = vec![];
        let mut av = vec![];
        for j in 0..o.shape()[0] {
            let mut m = [0.; 16];
            for r in 0..4 {
                for c in 0..4 {
                    m[4 * r + c] = o[[j, r, c]];
                }
            }
            ov.push(m);
            av.push([a[[j, 0]], a[[j, 1]], a[[j, 2]]]);
        }
        let t = rt_of(&tcp)?.to_rows();
        Ok(FastChain {
            c: Chain::new(&ov, &av, &movable, &t, &lower, &upper).map_err(PyValueError::new_err)?,
        })
    }
    fn fk<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = q_of(&q)?;
        Ok(rt_to_py(py, &self.c.fk(&q)))
    }
    /// (n_joints+1, 4, 4): base then the child frame of every joint.
    fn frames<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
        let q = q_of(&q)?;
        let mut fr = [Rt::IDENTITY; crate::kin::MAXJ + 1];
        self.c.frames_into(&q, &mut fr);
        let n = self.c.joints.len() + 1;
        let mut v = Vec::with_capacity(n * 16);
        for f in fr.iter().take(n) {
            v.extend_from_slice(&f.to_rows());
        }
        Ok(numpy::ndarray::Array3::from_shape_vec((n, 4, 4), v)
            .unwrap()
            .into_pyarray(py))
    }
    fn jacobian<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = q_of(&q)?;
        let j = self.c.jacobian(&q);
        Ok(Array2::from_shape_fn((6, N), |(r, c)| j[(r, c)]).into_pyarray(py))
    }
    /// Returns q (converged) or None (did not converge). Arguments validated by the Python caller.
    #[pyo3(signature = (target, seed, max_iter, position_tol, angle_tol, length_scale, trust_radius, q_pref, w_null))]
    #[allow(clippy::too_many_arguments)]
    fn ik<'py>(
        &self,
        py: Python<'py>,
        target: PyReadonlyArray2<f64>,
        seed: PyReadonlyArray1<f64>,
        max_iter: usize,
        position_tol: f64,
        angle_tol: f64,
        length_scale: f64,
        trust_radius: f64,
        q_pref: Option<PyReadonlyArray1<f64>>,
        w_null: f64,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let t = rt_of(&target)?;
        let s = q_of(&seed)?;
        let qp = match q_pref {
            Some(a) => Some(q_of(&a)?),
            None => None,
        };
        let r = py.detach(|| {
            self.c.ik(
                &t,
                &s,
                max_iter,
                position_tol,
                angle_tol,
                length_scale,
                trust_radius,
                qp.as_ref(),
                w_null,
            )
        });
        Ok(r.map(|q| q.as_slice().to_vec().into_pyarray(py)))
    }
}

#[pyclass(module = "team048_accel")]
struct FastCollision {
    c: Collision,
    parallel: bool,
}

#[pymethods]
impl FastCollision {
    #[new]
    #[pyo3(signature = (chain, frame, centers, radii, pairs, levers, margin, parallel=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        chain: &FastChain,
        frame: Vec<usize>,
        centers: PyReadonlyArray2<f64>,
        radii: Vec<f64>,
        pairs: Vec<(u16, u16)>,
        levers: PyReadonlyArray2<f64>,
        margin: f64,
        parallel: bool,
    ) -> PyResult<Self> {
        let c = centers.as_array();
        let l = levers.as_array();
        if c.shape()[1] != 3 || l.shape()[1] != N || l.shape()[0] != c.shape()[0] {
            return Err(PyValueError::new_err("centers (s,3), levers (s,7)"));
        }
        let cv: Vec<[f64; 3]> = (0..c.shape()[0])
            .map(|i| [c[[i, 0]], c[[i, 1]], c[[i, 2]]])
            .collect();
        let lv: Vec<[f64; N]> = (0..l.shape()[0])
            .map(|i| std::array::from_fn(|k| l[[i, k]]))
            .collect();
        Ok(FastCollision {
            c: Collision::new(chain.c.clone(), frame, &cv, radii, pairs, lv, margin)
                .map_err(PyValueError::new_err)?,
            parallel,
        })
    }
    fn set_env(&mut self, centers: PyReadonlyArray2<f64>, radii: Vec<f64>) -> PyResult<()> {
        let c = centers.as_array();
        if c.shape()[1] != 3 || c.shape()[0] != radii.len() {
            return Err(PyValueError::new_err("env centers (e,3)"));
        }
        let cv: Vec<[f64; 3]> = (0..c.shape()[0])
            .map(|i| [c[[i, 0]], c[[i, 1]], c[[i, 2]]])
            .collect();
        self.c.env = std::sync::Arc::new(Env::new(&cv, &radii));
        Ok(())
    }
    /// Skip certificate on/off (diagnostics and equivalence tests; results are identical either way).
    #[getter]
    fn get_skip(&self) -> bool {
        self.c.skip
    }
    #[setter]
    fn set_skip(&mut self, v: bool) {
        self.c.skip = v;
    }
    /// Share a prebuilt (cached) environment grid: no copy, no rebuild.
    fn use_env(&mut self, env: &FastEnv) {
        self.c.env = env.e.clone();
    }
    /// Arm.solve: first seed whose IK converges to a collision-free q -> (seed index, q), else None.
    #[pyo3(signature = (target, seeds, max_iter, position_tol, angle_tol, length_scale, trust_radius, q_pref, w_null))]
    #[allow(clippy::too_many_arguments)]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        target: PyReadonlyArray2<f64>,
        seeds: PyReadonlyArray2<f64>,
        max_iter: usize,
        position_tol: f64,
        angle_tol: f64,
        length_scale: f64,
        trust_radius: f64,
        q_pref: Option<PyReadonlyArray1<f64>>,
        w_null: f64,
    ) -> PyResult<Option<(usize, Bound<'py, PyArray1<f64>>)>> {
        let t = rt_of(&target)?;
        let sv = qs_of(&seeds)?;
        let qp = match q_pref {
            Some(a) => Some(q_of(&a)?),
            None => None,
        };
        let r = py.detach(|| {
            self.c.solve(
                &t,
                &sv,
                max_iter,
                position_tol,
                angle_tol,
                length_scale,
                trust_radius,
                qp.as_ref(),
                w_null,
            )
        });
        Ok(r.map(|(k, q)| (k, q.as_slice().to_vec().into_pyarray(py))))
    }
    /// Many independent solves (offline studies). first: (m,7) extra first seed per target, NaN row = none.
    /// Returns (index (m,) with -1 = no solution, q (m,7) NaN where none). rayon over targets when parallel.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (targets, first, seeds, max_iter, position_tol, angle_tol, length_scale, trust_radius, q_pref, w_null, parallel=true))]
    #[allow(clippy::too_many_arguments)]
    fn solve_batch<'py>(
        &self,
        py: Python<'py>,
        targets: PyReadonlyArray3<f64>,
        first: PyReadonlyArray2<f64>,
        seeds: PyReadonlyArray2<f64>,
        max_iter: usize,
        position_tol: f64,
        angle_tol: f64,
        length_scale: f64,
        trust_radius: f64,
        q_pref: Option<PyReadonlyArray1<f64>>,
        w_null: f64,
        parallel: bool,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>)> {
        let tv = targets.as_array();
        if tv.shape()[1..] != [4, 4] || first.shape() != [tv.shape()[0], N] {
            return Err(PyValueError::new_err("targets (m,4,4), first (m,7)"));
        }
        let ts: Vec<Rt> = (0..tv.shape()[0])
            .map(|i| {
                Rt::from_rows(&std::array::from_fn::<f64, 16, _>(|k| {
                    tv[[i, k / 4, k % 4]]
                }))
            })
            .collect();
        let fv = first.as_array();
        let fs: Vec<Option<Q>> = (0..ts.len())
            .map(|i| {
                let q = Q::from_fn(|k, _| fv[[i, k]]);
                if q.iter().any(|x| x.is_nan()) {
                    None
                } else {
                    Some(q)
                }
            })
            .collect();
        let sv = qs_of(&seeds)?;
        let qp = match q_pref {
            Some(a) => Some(q_of(&a)?),
            None => None,
        };
        let r = py.detach(|| {
            self.c.solve_batch(
                &ts,
                &fs,
                &sv,
                max_iter,
                position_tol,
                angle_tol,
                length_scale,
                trust_radius,
                qp.as_ref(),
                w_null,
                parallel,
            )
        });
        let idx: Vec<i64> = r
            .iter()
            .map(|x| x.as_ref().map_or(-1, |(k, _)| *k as i64))
            .collect();
        let qv = Array2::from_shape_fn((r.len(), N), |(i, k)| {
            r[i].as_ref().map_or(f64::NAN, |(_, q)| q[k])
        });
        Ok((idx.into_pyarray(py), qv.into_pyarray(py)))
    }
    #[setter]
    fn set_margin(&mut self, m: f64) {
        self.c.margin = m;
    }
    fn valid(
        &self,
        py: Python<'_>,
        q: PyReadonlyArray1<f64>,
        inflation: PyReadonlyArray1<f64>,
    ) -> PyResult<bool> {
        let q = q_of(&q)?;
        let inf = inflation
            .as_slice()
            .map_err(|_| PyValueError::new_err("inflation contiguous"))?
            .to_vec();
        if inf.len() != self.c.n_spheres() {
            return Err(PyValueError::new_err("inflation length"));
        }
        Ok(py.detach(|| self.c.valid(&q, &inf)))
    }
    #[pyo3(signature = (a, b, max_step=0.003))]
    fn certify(
        &self,
        py: Python<'_>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
        max_step: f64,
    ) -> PyResult<bool> {
        let (a, b) = (q_of(&a)?, q_of(&b)?);
        let par = self.parallel;
        Ok(py.detach(|| self.c.certify(&a, &b, max_step, par)))
    }
}

#[pyfunction]
#[pyo3(signature = (points, planes, tol, parallel=false))]
fn plane_inlier_counts<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    planes: PyReadonlyArray2<f64>,
    tol: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    if points.shape()[1] != 3 || planes.shape()[1] != 4 {
        return Err(PyValueError::new_err("points (n,3), planes (m,4)"));
    }
    let p = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("points contiguous"))?;
    let pl = planes.as_array();
    let pv: Vec<[f64; 4]> = (0..pl.shape()[0])
        .map(|i| [pl[[i, 0]], pl[[i, 1]], pl[[i, 2]], pl[[i, 3]]])
        .collect();
    let c = py.detach(|| perc::plane_inlier_counts(p, &pv, tol, parallel));
    Ok(c.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::type_complexity)]
fn flood_fill<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray2<bool>,
    points: PyReadonlyArray3<f64>,
    max_jump: f64,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>)> {
    let (h, w) = (mask.shape()[0], mask.shape()[1]);
    if points.shape() != [h, w, 3] {
        return Err(PyValueError::new_err("points (h,w,3)"));
    }
    let m = mask
        .as_slice()
        .map_err(|_| PyValueError::new_err("mask contiguous"))?;
    let p = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("points contiguous"))?;
    let (pix, off) = py.detach(|| perc::flood_fill(m, p, h, w, max_jump));
    let n = pix.len() / 2;
    let a = Array2::from_shape_vec((n, 2), pix.into_iter().map(|v| v as i64).collect()).unwrap();
    Ok((
        a.into_pyarray(py),
        off.into_iter()
            .map(|v| v as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    ))
}

/// Prebuilt environment-sphere grid, shareable between collision models (cached per scene by the Python glue).
#[pyclass(module = "team048_accel", frozen)]
struct FastEnv {
    e: std::sync::Arc<Env>,
}
#[pymethods]
impl FastEnv {
    #[new]
    fn new(centers: PyReadonlyArray2<f64>, radii: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let c = centers.as_array();
        let r = slice(&radii, "radii")?;
        if c.shape()[1] != 3 || c.shape()[0] != r.len() {
            return Err(PyValueError::new_err("env centers (e,3), radii (e,)"));
        }
        let cv: Vec<[f64; 3]> = (0..c.shape()[0])
            .map(|i| [c[[i, 0]], c[[i, 1]], c[[i, 2]]])
            .collect();
        Ok(FastEnv {
            e: std::sync::Arc::new(Env::new(&cv, r)),
        })
    }
    fn __len__(&self) -> usize {
        self.e.len()
    }
}

/// node.observe back-projection: depth (h,w) mm as float, K entries, camera->pelvis T (4,4) -> P (h,w,3).
#[pyfunction]
#[pyo3(signature = (depth, fx, fy, cx, cy, t, parallel=false))]
#[allow(clippy::too_many_arguments)]
fn deproject<'py>(
    py: Python<'py>,
    depth: PyReadonlyArray2<f64>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    t: PyReadonlyArray2<f64>,
    parallel: bool,
) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
    let (h, w) = (depth.shape()[0], depth.shape()[1]);
    let d = slice(&depth, "depth")?;
    let tv = t.as_array();
    if tv.shape() != [4, 4] {
        return Err(PyValueError::new_err("T (4,4)"));
    }
    let r: [f64; 9] = std::array::from_fn(|k| tv[[k / 3, k % 3]]);
    let tr = [tv[[0, 3]], tv[[1, 3]], tv[[2, 3]]];
    let out = py.detach(|| perc::deproject(d, h, w, fx, fy, cx, cy, &r, &tr, parallel));
    Ok(numpy::ndarray::Array3::from_shape_vec((h, w, 3), out)
        .unwrap()
        .into_pyarray(py))
}

/// table_plane RANSAC: compaction (mask & finite, row-major), NumPy-stream candidates, inlier counts.
/// pcg = (state_hi, state_lo, inc_hi, inc_lo, has_uint32, uinteger) of np.random.PCG64(seed).state.
/// Returns (p (n,3), candidates (m,4) [normal, offset], index of the first maximal-count candidate or -1 when every
/// count is 0, its count) or None if NumPy would not take Floyd's branch.
#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (points, mask, pcg, iterations, tol, parallel=false))]
fn ransac_plane<'py>(
    py: Python<'py>,
    points: PyReadonlyArray3<f64>,
    mask: PyReadonlyArray2<bool>,
    pcg: (u64, u64, u64, u64, bool, u32),
    iterations: usize,
    tol: f64,
    parallel: bool,
) -> PyResult<
    Option<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        i64,
        u64,
    )>,
> {
    let (h, w) = (mask.shape()[0], mask.shape()[1]);
    if points.shape() != [h, w, 3] {
        return Err(PyValueError::new_err("points (h,w,3)"));
    }
    let p = slice(&points, "points")?;
    let m = slice(&mask, "mask")?;
    let r = py.detach(|| {
        let (x, y, z) = perc::compact_finite(p, m);
        if x.len() < 3 {
            return None;
        }
        let mut g = crate::rng::Pcg64::from_numpy(
            ((pcg.0 as u128) << 64) | pcg.1 as u128,
            ((pcg.2 as u128) << 64) | pcg.3 as u128,
            pcg.4,
            pcg.5,
        );
        let c = perc::ransac_candidates(&x, &y, &z, &mut g, iterations)?;
        let best = if parallel {
            // rayon over candidates (all counts), then the same first-maximum rule
            let counts = perc::plane_inlier_counts_soa(&x, &y, &z, &c, tol, true);
            let mut b = None;
            let mut sc = 0;
            for (k, &v) in counts.iter().enumerate() {
                if v > sc {
                    sc = v;
                    b = Some((k, v));
                }
            }
            b
        } else {
            perc::ransac_best(&x, &y, &z, &c, tol)
        };
        Some((x, y, z, c, best))
    });
    Ok(r.map(|(x, y, z, c, best)| {
        let n = x.len();
        let mut pa = Array2::<f64>::zeros((n, 3));
        for (i, mut row) in pa.rows_mut().into_iter().enumerate() {
            row[0] = x[i];
            row[1] = y[i];
            row[2] = z[i];
        }
        let ca = Array2::from_shape_fn((c.len(), 4), |(i, k)| c[i][k]);
        let (k, cnt) = best.map_or((-1i64, 0u64), |(k, v)| (k as i64, v));
        (pa.into_pyarray(py), ca.into_pyarray(py), k, cnt)
    }))
}

/// node.tray_mask: union of rotated rectangles (x, y, u0, u1, v0, v1, ha, hb) over an organised cloud.
#[pyfunction]
fn rot_rect_mask<'py>(
    py: Python<'py>,
    points: PyReadonlyArray3<f64>,
    rects: Vec<[f64; 8]>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let sh = points.shape();
    if sh[2] != 3 {
        return Err(PyValueError::new_err("points (h,w,3)"));
    }
    let (h, w) = (sh[0], sh[1]);
    let p = slice(&points, "points")?;
    let mut out = vec![false; h * w];
    py.detach(|| perc::rot_rect_mask(p, &rects, &mut out));
    Ok(Array2::from_shape_vec((h, w), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py))
}

/// trays.detect selection -> (count in window and height band, mask after chroma).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tray_window_mask<'py>(
    py: Python<'py>,
    points: PyReadonlyArray3<f64>,
    rgb: PyReadonlyArray3<u8>,
    valid: PyReadonlyArray2<bool>,
    height: PyReadonlyArray2<f64>,
    window: (f64, f64, f64, f64),
    band: (f64, f64),
    channel: usize,
    chroma_min: f64,
) -> PyResult<(usize, Bound<'py, PyArray2<bool>>)> {
    let (h, w) = (valid.shape()[0], valid.shape()[1]);
    if points.shape() != [h, w, 3]
        || rgb.shape() != [h, w, 3]
        || height.shape() != [h, w]
        || channel > 2
    {
        return Err(PyValueError::new_err(
            "points/rgb (h,w,3), height (h,w), channel 0..2",
        ));
    }
    let (p, c, v, hh) = (
        slice(&points, "points")?,
        slice(&rgb, "rgb")?,
        slice(&valid, "valid")?,
        slice(&height, "height")?,
    );
    let mut out = vec![false; h * w];
    let win = [window.0, window.1, window.2, window.3];
    let n =
        py.detach(|| perc::tray_window_mask(p, c, v, hh, win, band, channel, chroma_min, &mut out));
    let arr = numpy::ndarray::Array2::from_shape_vec((h, w), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((n, arr.into_pyarray(py)))
}

/// perception.segment core: mask, flood fill and the statistics of every component with >= min_pixels pixels.
/// Returns (pixels (n,2), offsets (c+1,), kept component indices (k,), rows (k,15)); row layout in perc.rs.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn segment_components<'py>(
    py: Python<'py>,
    points: PyReadonlyArray3<f64>,
    rgb: PyReadonlyArray3<u8>,
    valid: PyReadonlyArray2<bool>,
    n: Vec<f64>,
    d: f64,
    min_h: f64,
    max_h: f64,
    low: Vec<f64>,
    high: Vec<f64>,
    max_jump: f64,
    min_pixels: usize,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let (h, w) = (valid.shape()[0], valid.shape()[1]);
    if points.shape() != [h, w, 3] || rgb.shape() != [h, w, 3] {
        return Err(PyValueError::new_err("points/rgb (h,w,3)"));
    }
    let (p, c, v) = (
        slice(&points, "points")?,
        slice(&rgb, "rgb")?,
        slice(&valid, "valid")?,
    );
    let (n, low, high) = (v3(n, "n")?, v3(low, "low")?, v3(high, "high")?);
    let (pix, off, keep, rows) = py.detach(|| {
        let mask = perc::segment_mask(p, v, &n, d, min_h, max_h, &low, &high);
        let (pix, off) = perc::flood_fill(&mask, p, h, w, max_jump);
        let (keep, rows) = perc::component_features(p, c, w, &pix, &off, &n, d, min_pixels);
        (pix, off, keep, rows)
    });
    let np_ = pix.len() / 2;
    let a = Array2::from_shape_vec((np_, 2), pix.into_iter().map(|v| v as i64).collect()).unwrap();
    let ra = Array2::from_shape_fn((rows.len(), 15), |(i, k)| rows[i][k]);
    Ok((
        a.into_pyarray(py),
        off.into_iter()
            .map(|v| v as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
        keep.into_iter()
            .map(|v| v as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
        ra.into_pyarray(py),
    ))
}

/// node.observe table mask (see perc::table_mask). rects: list of (cx, cy, hx, hy).
#[pyfunction]
fn table_mask<'py>(
    py: Python<'py>,
    points: PyReadonlyArray3<f64>,
    valid: PyReadonlyArray2<bool>,
    x_lo: f64,
    x_hi: f64,
    y_abs: f64,
    rects: Vec<(f64, f64, f64, f64)>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let (h, w) = (valid.shape()[0], valid.shape()[1]);
    if points.shape() != [h, w, 3] {
        return Err(PyValueError::new_err("points (h,w,3)"));
    }
    let (p, v) = (slice(&points, "points")?, slice(&valid, "valid")?);
    let rv: Vec<[f64; 4]> = rects.iter().map(|r| [r.0, r.1, r.2, r.3]).collect();
    let m = py.detach(|| perc::table_mask(p, v, x_lo, x_hi, y_abs, &rv));
    Ok(Array2::from_shape_vec((h, w), m).unwrap().into_pyarray(py))
}

/// grasp.footprint caliper sweep: (min width, index of the first minimal direction).
#[pyfunction]
fn min_caliper(
    py: Python<'_>,
    hull: PyReadonlyArray2<f64>,
    dirs: PyReadonlyArray2<f64>,
) -> PyResult<(f64, usize)> {
    if hull.shape()[1] != 2 || dirs.shape()[1] != 2 {
        return Err(PyValueError::new_err("hull (m,2), dirs (a,2)"));
    }
    let hv = hull.as_array();
    let dv = dirs.as_array();
    let hs: Vec<[f64; 2]> = (0..hv.shape()[0])
        .map(|i| [hv[[i, 0]], hv[[i, 1]]])
        .collect();
    let ds: Vec<[f64; 2]> = (0..dv.shape()[0])
        .map(|i| [dv[[i, 0]], dv[[i, 1]]])
        .collect();
    Ok(py.detach(|| crate::grasp::min_caliper(&hs, &ds)))
}

/// trays.min_area_rect direction scan -> (index, lu, hu, lv, hv) or None.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn min_area_rect_scan(
    py: Python<'_>,
    xy: PyReadonlyArray2<f64>,
    dirs: PyReadonlyArray2<f64>,
    q: f64,
) -> PyResult<Option<(usize, f64, f64, f64, f64)>> {
    if xy.shape()[1] != 2 || dirs.shape()[1] != 2 {
        return Err(PyValueError::new_err("xy (n,2), dirs (a,2)"));
    }
    let xv = xy.as_array();
    let dv = dirs.as_array();
    let ps: Vec<[f64; 2]> = (0..xv.shape()[0])
        .map(|i| [xv[[i, 0]], xv[[i, 1]]])
        .collect();
    let ds: Vec<[f64; 2]> = (0..dv.shape()[0])
        .map(|i| [dv[[i, 0]], dv[[i, 1]]])
        .collect();
    Ok(py
        .detach(|| crate::grasp::min_area_rect_scan(&ps, &ds, q))
        .map(|(k, e)| (k, e[0], e[1], e[2], e[3])))
}

/// Finger-corridor clearance: per tip min xy distance to the points (inf when no points). pts (m, >=2).
#[pyfunction]
fn min_dist_xy<'py>(
    py: Python<'py>,
    tips: PyReadonlyArray2<f64>,
    pts: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if tips.shape()[1] != 2 || pts.shape()[1] < 2 {
        return Err(PyValueError::new_err("tips (t,2), pts (m,>=2)"));
    }
    let tv = tips.as_array();
    let pv = pts.as_array();
    let ts: Vec<[f64; 2]> = (0..tv.shape()[0])
        .map(|i| [tv[[i, 0]], tv[[i, 1]]])
        .collect();
    let xs: Vec<f64> = pv.column(0).iter().cloned().collect();
    let ys: Vec<f64> = pv.column(1).iter().cloned().collect();
    let r = py.detach(|| crate::grasp::min_dist_xy(&ts, &xs, &ys));
    Ok(r.into_pyarray(py))
}

#[pymodule]
fn team048_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastChain>()?;
    m.add_class::<FastCollision>()?;
    m.add_function(wrap_pyfunction!(plane_inlier_counts, m)?)?;
    m.add_function(wrap_pyfunction!(flood_fill, m)?)?;
    m.add_class::<FastEnv>()?;
    m.add_function(wrap_pyfunction!(deproject, m)?)?;
    m.add_function(wrap_pyfunction!(ransac_plane, m)?)?;
    m.add_function(wrap_pyfunction!(tray_window_mask, m)?)?;
    m.add_function(wrap_pyfunction!(rot_rect_mask, m)?)?;
    m.add_function(wrap_pyfunction!(segment_components, m)?)?;
    m.add_function(wrap_pyfunction!(min_caliper, m)?)?;
    m.add_function(wrap_pyfunction!(min_area_rect_scan, m)?)?;
    m.add_function(wrap_pyfunction!(table_mask, m)?)?;
    m.add_function(wrap_pyfunction!(min_dist_xy, m)?)?;
    m.add("API", 2)?;
    m.add("N_ACTIVE", N)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
