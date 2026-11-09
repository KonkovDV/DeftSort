//! PyO3 bindings. All heavy calls release the GIL (`py.detach`, formerly `allow_threads`) so the ROS executor
//! threads keep running while planning.
use crate::col::{Collision, Env};
use crate::kin::{Chain, Q, Rt, N};
use crate::perc;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn q_of(a: &PyReadonlyArray1<f64>) -> PyResult<Q> {
    let s = a.as_slice().map_err(|_| PyValueError::new_err("q must be contiguous"))?;
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
fn rt_to_py<'py>(py: Python<'py>, t: &Rt) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((4, 4), t.to_rows().to_vec()).unwrap().into_pyarray(py)
}

#[pyclass(module = "team048_accel", frozen)]
struct FastChain {
    c: Chain,
}

#[pymethods]
impl FastChain {
    #[new]
    fn new(origins: PyReadonlyArray3<f64>, axes: PyReadonlyArray2<f64>, movable: Vec<bool>, tcp: PyReadonlyArray2<f64>, lower: Vec<f64>, upper: Vec<f64>) -> PyResult<Self> {
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
        Ok(FastChain { c: Chain::new(&ov, &av, &movable, &t, &lower, &upper).map_err(PyValueError::new_err)? })
    }
    fn fk<'py>(&self, py: Python<'py>, q: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = q_of(&q)?;
        Ok(rt_to_py(py, &self.c.fk(&q)))
    }
    /// (n_joints+1, 4, 4): base then the child frame of every joint.
    fn frames<'py>(&self, py: Python<'py>, q: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
        let q = q_of(&q)?;
        let mut fr = [Rt::IDENTITY; crate::kin::MAXJ + 1];
        self.c.frames_into(&q, &mut fr);
        let n = self.c.joints.len() + 1;
        let mut v = Vec::with_capacity(n * 16);
        for f in fr.iter().take(n) {
            v.extend_from_slice(&f.to_rows());
        }
        Ok(numpy::ndarray::Array3::from_shape_vec((n, 4, 4), v).unwrap().into_pyarray(py))
    }
    fn jacobian<'py>(&self, py: Python<'py>, q: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let q = q_of(&q)?;
        let j = self.c.jacobian(&q);
        Ok(Array2::from_shape_fn((6, N), |(r, c)| j[(r, c)]).into_pyarray(py))
    }
    /// Returns q (converged) or None (did not converge). Arguments validated by the Python caller.
    #[pyo3(signature = (target, seed, max_iter, position_tol, angle_tol, length_scale, trust_radius, q_pref, w_null))]
    #[allow(clippy::too_many_arguments)]
    fn ik<'py>(&self, py: Python<'py>, target: PyReadonlyArray2<f64>, seed: PyReadonlyArray1<f64>, max_iter: usize, position_tol: f64, angle_tol: f64, length_scale: f64, trust_radius: f64, q_pref: Option<PyReadonlyArray1<f64>>, w_null: f64) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let t = rt_of(&target)?;
        let s = q_of(&seed)?;
        let qp = match q_pref {
            Some(a) => Some(q_of(&a)?),
            None => None,
        };
        let r = py.detach(|| self.c.ik(&t, &s, max_iter, position_tol, angle_tol, length_scale, trust_radius, qp.as_ref(), w_null));
        Ok(r.ok().map(|q| q.as_slice().to_vec().into_pyarray(py)))
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
    #[pyo3(signature = (chain, frame, centers, radii, pairs, levers, margin, parallel=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(chain: &FastChain, frame: Vec<usize>, centers: PyReadonlyArray2<f64>, radii: Vec<f64>, pairs: Vec<(u16, u16)>, levers: PyReadonlyArray2<f64>, margin: f64, parallel: bool) -> PyResult<Self> {
        let c = centers.as_array();
        let l = levers.as_array();
        if c.shape()[1] != 3 || l.shape()[1] != N || l.shape()[0] != c.shape()[0] {
            return Err(PyValueError::new_err("centers (s,3), levers (s,7)"));
        }
        let cv: Vec<[f64; 3]> = (0..c.shape()[0]).map(|i| [c[[i, 0]], c[[i, 1]], c[[i, 2]]]).collect();
        let lv: Vec<[f64; N]> = (0..l.shape()[0]).map(|i| std::array::from_fn(|k| l[[i, k]])).collect();
        Ok(FastCollision { c: Collision::new(chain.c.clone(), frame, &cv, radii, pairs, lv, margin).map_err(PyValueError::new_err)?, parallel })
    }
    fn set_env(&mut self, centers: PyReadonlyArray2<f64>, radii: Vec<f64>) -> PyResult<()> {
        let c = centers.as_array();
        if c.shape()[1] != 3 || c.shape()[0] != radii.len() {
            return Err(PyValueError::new_err("env centers (e,3)"));
        }
        let cv: Vec<[f64; 3]> = (0..c.shape()[0]).map(|i| [c[[i, 0]], c[[i, 1]], c[[i, 2]]]).collect();
        self.c.env = Env::new(&cv, &radii);
        Ok(())
    }
    #[setter]
    fn set_margin(&mut self, m: f64) {
        self.c.margin = m;
    }
    fn valid(&self, py: Python<'_>, q: PyReadonlyArray1<f64>, inflation: PyReadonlyArray1<f64>) -> PyResult<bool> {
        let q = q_of(&q)?;
        let inf = inflation.as_slice().map_err(|_| PyValueError::new_err("inflation contiguous"))?.to_vec();
        if inf.len() != self.c.n_spheres() {
            return Err(PyValueError::new_err("inflation length"));
        }
        Ok(py.detach(|| self.c.valid(&q, &inf)))
    }
    #[pyo3(signature = (a, b, max_step=0.003))]
    fn certify(&self, py: Python<'_>, a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>, max_step: f64) -> PyResult<bool> {
        let (a, b) = (q_of(&a)?, q_of(&b)?);
        let par = self.parallel;
        Ok(py.detach(|| self.c.certify(&a, &b, max_step, par)))
    }
}

#[pyfunction]
fn plane_inlier_counts<'py>(py: Python<'py>, points: PyReadonlyArray2<f64>, planes: PyReadonlyArray2<f64>, tol: f64) -> PyResult<Bound<'py, PyArray1<u64>>> {
    if points.shape()[1] != 3 || planes.shape()[1] != 4 {
        return Err(PyValueError::new_err("points (n,3), planes (m,4)"));
    }
    let p = points.as_slice().map_err(|_| PyValueError::new_err("points contiguous"))?;
    let pl = planes.as_array();
    let pv: Vec<[f64; 4]> = (0..pl.shape()[0]).map(|i| [pl[[i, 0]], pl[[i, 1]], pl[[i, 2]], pl[[i, 3]]]).collect();
    let c = py.detach(|| perc::plane_inlier_counts(p, &pv, tol));
    Ok(c.into_pyarray(py))
}

#[pyfunction]
fn flood_fill<'py>(py: Python<'py>, mask: PyReadonlyArray2<bool>, points: PyReadonlyArray3<f64>, max_jump: f64) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>)> {
    let (h, w) = (mask.shape()[0], mask.shape()[1]);
    if points.shape() != [h, w, 3] {
        return Err(PyValueError::new_err("points (h,w,3)"));
    }
    let m = mask.as_slice().map_err(|_| PyValueError::new_err("mask contiguous"))?;
    let p = points.as_slice().map_err(|_| PyValueError::new_err("points contiguous"))?;
    let (pix, off) = py.detach(|| perc::flood_fill(m, p, h, w, max_jump));
    let n = pix.len() / 2;
    let a = Array2::from_shape_vec((n, 2), pix.into_iter().map(|v| v as i64).collect()).unwrap();
    Ok((a.into_pyarray(py), off.into_iter().map(|v| v as i64).collect::<Vec<_>>().into_pyarray(py)))
}

#[pymodule]
fn team048_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastChain>()?;
    m.add_class::<FastCollision>()?;
    m.add_function(wrap_pyfunction!(plane_inlier_counts, m)?)?;
    m.add_function(wrap_pyfunction!(flood_fill, m)?)?;
    m.add("N_ACTIVE", N)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
