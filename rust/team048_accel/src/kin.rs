//! Serial-chain kinematics, a port of `team048_solution.kinematics.Chain` (fk / frames / jacobian / ik)
//! and `geometry.rotation_error`. Arithmetic follows the NumPy reference operation by operation; results agree to
//! ~1e-15 (BLAS may reorder 3-term sums), which the equivalence tests bound by 1e-9.
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

pub type M3 = Matrix3<f64>;
pub type V3 = Vector3<f64>;
/// Rigid transform kept as (R, t): the bottom row [0 0 0 1] of the 4x4 reference never changes the result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rt {
    pub r: M3,
    pub t: V3,
}
impl Rt {
    pub const IDENTITY: Rt = Rt {
        r: M3::new(1., 0., 0., 0., 1., 0., 0., 0., 1.),
        t: V3::new(0., 0., 0.),
    };
    #[inline(always)]
    pub fn mul(&self, o: &Rt) -> Rt {
        Rt {
            r: self.r * o.r,
            t: self.r * o.t + self.t,
        }
    }
    #[inline(always)]
    pub fn mul_rot(&self, r: &M3) -> Rt {
        Rt {
            r: self.r * r,
            t: self.t,
        }
    }
    #[inline(always)]
    pub fn apply(&self, p: &V3) -> V3 {
        self.r * p + self.t
    }
    pub fn from_rows(m: &[f64]) -> Rt {
        // row-major 4x4
        Rt {
            r: M3::new(m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]),
            t: V3::new(m[3], m[7], m[11]),
        }
    }
    pub fn to_rows(&self) -> [f64; 16] {
        let r = &self.r;
        let t = &self.t;
        [
            r[(0, 0)],
            r[(0, 1)],
            r[(0, 2)],
            t[0],
            r[(1, 0)],
            r[(1, 1)],
            r[(1, 2)],
            t[1],
            r[(2, 0)],
            r[(2, 1)],
            r[(2, 2)],
            t[2],
            0.,
            0.,
            0.,
            1.,
        ]
    }
}

/// Maximum number of joints (fixed + movable) supported without heap allocation.
pub const MAXJ: usize = 24;
/// Number of active joints the fixed-size IK is specialised for (the TienKung arm).
pub const N: usize = 7;
pub type Jac = SMatrix<f64, 6, N>;
pub type Q = SVector<f64, N>;

#[derive(Clone, Debug)]
pub struct Joint {
    pub origin: Rt,
    /// skew matrix K of the normalised axis and K@K: rotation(axis, a) = I + sin(a) K + (1-cos(a)) K@K
    pub k: M3,
    pub kk: M3,
    pub axis: V3,
    pub movable: bool,
}

#[derive(Clone, Debug)]
pub struct Chain {
    pub joints: Vec<Joint>,
    pub tcp: Rt,
    pub lower: Q,
    pub upper: Q,
    pub n_active: usize,
}

#[inline(always)]
fn skew_of(axis: &V3) -> (M3, M3) {
    // geometry.rotation: a/np.linalg.norm(a)
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let (x, y, z) = (axis[0] / norm, axis[1] / norm, axis[2] / norm);
    let k = M3::new(0., -z, y, z, 0., -x, -y, x, 0.);
    (k, k * k)
}

#[inline(always)]
pub fn rot(j: &Joint, angle: f64) -> M3 {
    // np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
    let s = angle.sin();
    let c1 = 1. - angle.cos();
    let mut r = M3::identity();
    for i in 0..9 {
        r[i] = (r[i] + s * j.k[i]) + c1 * j.kk[i];
    }
    r
}

impl Chain {
    /// origins: row-major 4x4 per joint; axes: per joint; movable flags. Lower/upper per ACTIVE joint.
    pub fn new(
        origins: &[[f64; 16]],
        axes: &[[f64; 3]],
        movable: &[bool],
        tcp: &[f64; 16],
        lower: &[f64],
        upper: &[f64],
    ) -> Result<Chain, String> {
        let nj = origins.len();
        if nj == 0 || nj > MAXJ || axes.len() != nj || movable.len() != nj {
            return Err("unsupported chain size".into());
        }
        let n_active = movable.iter().filter(|m| **m).count();
        if n_active != N || lower.len() != N || upper.len() != N {
            return Err(format!("fast path needs exactly {N} active joints"));
        }
        let joints = (0..nj)
            .map(|i| {
                let axis = V3::new(axes[i][0], axes[i][1], axes[i][2]);
                let (k, kk) = skew_of(&axis);
                Joint {
                    origin: Rt::from_rows(&origins[i]),
                    k,
                    kk,
                    axis,
                    movable: movable[i],
                }
            })
            .collect();
        Ok(Chain {
            joints,
            tcp: Rt::from_rows(tcp),
            lower: Q::from_column_slice(lower),
            upper: Q::from_column_slice(upper),
            n_active,
        })
    }

    /// frames[0] = base, frames[j+1] = child frame of joint j (the Python dict keyed by child link).
    #[inline]
    pub fn frames_into(&self, q: &Q, out: &mut [Rt; MAXJ + 1]) {
        let mut t = Rt::IDENTITY;
        out[0] = t;
        let mut i = 0;
        for (j, joint) in self.joints.iter().enumerate() {
            t = t.mul(&joint.origin);
            if joint.movable {
                t = t.mul_rot(&rot(joint, q[i]));
                i += 1;
            }
            out[j + 1] = t;
        }
    }

    #[inline]
    pub fn fk(&self, q: &Q) -> Rt {
        let mut t = Rt::IDENTITY;
        let mut i = 0;
        for joint in &self.joints {
            t = t.mul(&joint.origin);
            if joint.movable {
                t = t.mul_rot(&rot(joint, q[i]));
                i += 1;
            }
        }
        t.mul(&self.tcp)
    }

    /// World-frame geometric Jacobian, rows [linear; angular], also returns FK (shared traversal).
    #[inline]
    pub fn jacobian(&self, q: &Q) -> Jac {
        let mut t = Rt::IDENTITY;
        let mut origins = [V3::zeros(); N];
        let mut axes = [V3::zeros(); N];
        let mut i = 0;
        for joint in &self.joints {
            t = t.mul(&joint.origin);
            if joint.movable {
                origins[i] = t.t;
                axes[i] = t.r * joint.axis;
                t = t.mul_rot(&rot(joint, q[i]));
                i += 1;
            }
        }
        let point = t.mul(&self.tcp).t;
        let mut jac = Jac::zeros();
        for c in 0..N {
            let lin = axes[c].cross(&(point - origins[c]));
            for r in 0..3 {
                jac[(r, c)] = lin[r];
                jac[(r + 3, c)] = axes[c][r];
            }
        }
        jac
    }

    #[inline(always)]
    fn clip(&self, q: &Q) -> Q {
        let mut o = *q;
        for k in 0..N {
            // np.clip(x, lo, hi) == minimum(maximum(x, lo), hi)
            o[k] = q[k].max(self.lower[k]).min(self.upper[k]);
        }
        o
    }

    #[inline(always)]
    fn residual(&self, target: &Rt, q: &Q) -> SVector<f64, 6> {
        let t = self.fk(q);
        let dp = target.t - t.t;
        let w = rotation_error(&target.r, &t.r);
        SVector::<f64, 6>::new(dp[0], dp[1], dp[2], w[0], w[1], w[2])
    }

    /// Port of Chain.ik after its argument validation (done by the Python caller). Some(q) or None == "did not converge".
    #[allow(clippy::too_many_arguments)]
    pub fn ik(
        &self,
        target: &Rt,
        seed: &Q,
        max_iter: usize,
        position_tol: f64,
        angle_tol: f64,
        length_scale: f64,
        trust_radius: f64,
        q_pref: Option<&Q>,
        w_null: f64,
    ) -> Option<Q> {
        let mut q = *seed;
        let weights = SVector::<f64, 6>::new(1., 1., 1., length_scale, length_scale, length_scale);
        const SCALES: [f64; 6] = [1., 0.5, 0.25, 0.125, 0.0625, 0.015625];
        // residual of the accepted line-search candidate is reused (same q, same deterministic arithmetic)
        let mut cached: Option<SVector<f64, 6>> = None;
        for _ in 0..max_iter {
            let e = match cached.take() {
                Some(e) => e,
                None => self.residual(target, &q),
            };
            if norm3(e[0], e[1], e[2]) < position_tol && norm3(e[3], e[4], e[5]) < angle_tol {
                return Some(q);
            }
            let mut jw = self.jacobian(&q);
            for r in 0..6 {
                for c in 0..N {
                    jw[(r, c)] *= weights[r];
                }
            }
            let ew = weights.component_mul(&e);
            let (u, s, vt) = svd_sorted(&jw);
            let smin = s[5];
            let damping = 0.002 + 0.05 * (0f64.max(1. - smin / 0.04)).powi(2);
            let dd = damping * damping;
            // dq_task = Vt.T @ ((s/(s*s+d*d)) * (U.T @ ew))
            let ute = u.transpose() * ew;
            let mut coef = SVector::<f64, 6>::zeros();
            for k in 0..6 {
                coef[k] = (s[k] / (s[k] * s[k] + dd)) * ute[k];
            }
            let dq_task: Q = vt.transpose() * coef;
            let ew_norm = norm_n(ew.as_slice());
            let mut steps: [Q; 2] = [dq_task, dq_task];
            let mut nsteps = 1;
            if let Some(qp) = q_pref {
                if w_null > 0. {
                    let thr = 1e-6 * s[0];
                    let rank = s.iter().filter(|v| **v > thr).count();
                    // N = I - Vt[:rank].T @ Vt[:rank]
                    let mut nm = SMatrix::<f64, N, N>::identity();
                    let mut vr = SMatrix::<f64, 6, N>::zeros();
                    for r in 0..rank {
                        vr.set_row(r, &vt.row(r));
                    }
                    nm -= vr.transpose() * vr;
                    steps[0] = dq_task + w_null * (nm * (qp - q));
                    steps[1] = dq_task;
                    nsteps = 2;
                }
            }
            let mut improved = false;
            for dq0 in steps.iter().take(nsteps) {
                let mut dq = *dq0;
                let nrm = norm_n(dq.as_slice());
                if nrm > trust_radius {
                    dq *= trust_radius / nrm;
                }
                for sc in SCALES {
                    let cand = self.clip(&(q + sc * dq));
                    let rc = self.residual(target, &cand);
                    let r = weights.component_mul(&rc);
                    if norm_n(r.as_slice()) < ew_norm - 1e-12 {
                        q = cand;
                        cached = Some(rc);
                        improved = true;
                        break;
                    }
                }
                if improved {
                    break;
                }
            }
            if !improved {
                break;
            }
        }
        None
    }
}

#[inline(always)]
pub fn norm3(a: f64, b: f64, c: f64) -> f64 {
    ((a * a + b * b) + c * c).sqrt()
}
#[inline(always)]
pub fn norm_n(x: &[f64]) -> f64 {
    let mut s = 0.;
    for v in x {
        s += v * v;
    }
    s.sqrt()
}

/// Thin SVD of the 6x7 weighted Jacobian, singular values sorted descending (LAPACK/numpy convention).
/// One-sided Jacobi (Hestenes) on the 7x6 matrix A = J^T: column rotations until every column pair is orthogonal to
/// machine precision; high relative accuracy, fixed size, no allocation (nalgebra's bidiagonal SVD reconstructed
/// this matrix only to ~2.5e-10). The IK update only uses sign-invariant products, so sign conventions do not matter.
pub type Svd = (SMatrix<f64, 6, 6>, SVector<f64, 6>, SMatrix<f64, 6, N>);
pub fn svd_sorted(j: &Jac) -> Svd {
    let mut a: SMatrix<f64, N, 6> = j.transpose();
    let mut v = SMatrix::<f64, 6, 6>::identity();
    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..5 {
            for q in p + 1..6 {
                let (mut alpha, mut beta, mut gamma) = (0., 0., 0.);
                for r in 0..N {
                    let (x, y) = (a[(r, p)], a[(r, q)]);
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if gamma == 0. || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2. * gamma);
                let t = zeta.signum() / (zeta.abs() + (1. + zeta * zeta).sqrt());
                let t = if zeta == 0. { 1. } else { t };
                let c = 1. / (1. + t * t).sqrt();
                let s = c * t;
                for r in 0..N {
                    let (x, y) = (a[(r, p)], a[(r, q)]);
                    a[(r, p)] = c * x - s * y;
                    a[(r, q)] = s * x + c * y;
                }
                for r in 0..6 {
                    let (x, y) = (v[(r, p)], v[(r, q)]);
                    v[(r, p)] = c * x - s * y;
                    v[(r, q)] = s * x + c * y;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    // A = J^T V = U' S  =>  J^T = U' S V^T  =>  J = V S U'^T : U_J = V, Vt_J = U'^T
    let sv: [f64; 6] = std::array::from_fn(|k| norm_n(a.column(k).as_slice()));
    let mut idx = [0usize, 1, 2, 3, 4, 5];
    idx.sort_by(|x, y| {
        sv[*y]
            .partial_cmp(&sv[*x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut u = SMatrix::<f64, 6, 6>::zeros();
    let mut s = SVector::<f64, 6>::zeros();
    let mut vt = SMatrix::<f64, 6, N>::zeros();
    for (k, &i) in idx.iter().enumerate() {
        s[k] = sv[i];
        for r in 0..6 {
            u[(r, k)] = v[(r, i)];
        }
        if sv[i] > 0. {
            for c in 0..N {
                vt[(k, c)] = a[(c, i)] / sv[i];
            }
        }
    }
    (u, s, vt)
}

/// geometry.rotation_error (SO(3) log of target @ current.T), inputs already validated by the caller.
#[inline]
pub fn rotation_error(target: &M3, current: &M3) -> V3 {
    let r = target * current.transpose();
    let tr = (r[(0, 0)] + r[(1, 1)]) + r[(2, 2)];
    let theta = ((tr - 1.) / 2.).clamp(-1., 1.).acos();
    let v = V3::new(
        r[(2, 1)] - r[(1, 2)],
        r[(0, 2)] - r[(2, 0)],
        r[(1, 0)] - r[(0, 1)],
    );
    if theta < 1e-7 {
        return v * 0.5;
    }
    if std::f64::consts::PI - theta < 1e-5 {
        let sym = (r + r.transpose()) / 2.;
        let eig = nalgebra::linalg::SymmetricEigen::new(sym);
        // numpy eigh sorts ascending and argmax returns the first maximum -> pick the largest eigenvalue
        let mut best = 0;
        for k in 1..3 {
            if eig.eigenvalues[k] > eig.eigenvalues[best] {
                best = k;
            }
        }
        let mut axis: V3 = eig.eigenvectors.column(best).into_owned();
        if axis.dot(&v) < 0. {
            axis = -axis;
        }
        return axis * theta;
    }
    v * (theta / (2. * theta.sin()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;

    fn rpy(r: f64, p: f64, y: f64) -> M3 {
        let (cr, sr, cp, sp, cy, sy) = (r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin());
        M3::new(
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        )
    }
    pub fn test_chain() -> Chain {
        // 7-DOF arm-like chain with offsets; deterministic.
        let axes = [
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
        ];
        let offs = [
            [0., 0.2, 0.3],
            [0., 0.05, 0.],
            [0., 0., -0.12],
            [0., 0., -0.14],
            [0., 0., -0.1],
            [0., 0., -0.12],
            [0., 0., -0.05],
            [0., 0., -0.08],
        ];
        let mov = [true, true, true, true, true, true, true, false];
        let mut origins = vec![];
        for (k, o) in offs.iter().enumerate() {
            let r = rpy(0.1 * k as f64, -0.05 * k as f64, 0.02 * k as f64);
            let rt = Rt {
                r,
                t: V3::new(o[0], o[1], o[2]),
            };
            origins.push(rt.to_rows());
        }
        let lower = [-2.9, -0.3, -2.9, -2.2, -2.9, -1.0, -1.5];
        let upper = [2.9, 2.9, 2.9, 0.1, 2.9, 1.0, 1.5];
        Chain::new(
            &origins,
            &axes,
            &mov,
            &Rt::IDENTITY.to_rows(),
            &lower,
            &upper,
        )
        .unwrap()
    }
    fn q_in(c: &Chain, u: [f64; 7]) -> Q {
        Q::from_fn(|k, _| c.lower[k] + (c.upper[k] - c.lower[k]) * u[k])
    }
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]
        #[test]
        fn fk_rotation_is_orthonormal(u in proptest::array::uniform7(0f64..1.)) {
            let c = test_chain(); let t = c.fk(&q_in(&c, u));
            let e = (t.r.transpose() * t.r - M3::identity()).abs().max();
            prop_assert!(e < 1e-12); prop_assert!((t.r.determinant() - 1.).abs() < 1e-12);
        }
        #[test]
        fn jacobian_matches_finite_differences(u in proptest::array::uniform7(0.05f64..0.95)) {
            let c = test_chain(); let q = q_in(&c, u); let j = c.jacobian(&q); let h = 1e-6;
            for k in 0..N {
                let mut a = q; a[k] += h; let mut b = q; b[k] -= h;
                let (ta, tb) = (c.fk(&a), c.fk(&b));
                let dp = (ta.t - tb.t) / (2. * h);
                let dw = rotation_error(&ta.r, &tb.r) / (2. * h);
                for r in 0..3 { prop_assert!((dp[r] - j[(r, k)]).abs() < 1e-6); prop_assert!((dw[r] - j[(r + 3, k)]).abs() < 1e-6); }
            }
        }
        #[test]
        fn rotation_error_inverts_exp(ax in proptest::array::uniform3(-1f64..1.), th in 0.0f64..3.1) {
            let a = V3::new(ax[0], ax[1], ax[2]); prop_assume!(a.norm() > 1e-3);
            let j = Joint { origin: Rt::IDENTITY, k: skew_of(&a).0, kk: skew_of(&a).1, axis: a, movable: true };
            let r = rot(&j, th); let w = rotation_error(&r, &M3::identity());
            prop_assert!((w - a.normalize() * th).norm() < 1e-6);
        }
        #[test]
        fn ik_converges_from_nearby_seed(u in proptest::array::uniform7(0.2f64..0.8), du in proptest::array::uniform7(-0.03f64..0.03)) {
            let c = test_chain(); let q = q_in(&c, u); let target = c.fk(&q);
            let mut s = q; for k in 0..N { s[k] += du[k]; }
            let r = c.ik(&target, &s, 200, 0.002, 0.015, 0.25, 0.15, None, 0.);
            prop_assert!(r.is_some());
            let t = c.fk(&r.unwrap());
            prop_assert!((t.t - target.t).norm() < 0.002);
        }
        #[test]
        fn svd_reconstructs(u in proptest::array::uniform7(0f64..1.)) {
            let c = test_chain(); let j = c.jacobian(&q_in(&c, u)); let (uu, s, vt) = svd_sorted(&j);
            let rec = uu * SMatrix::<f64,6,6>::from_diagonal(&s) * vt;
            let err = (rec - j).abs().max(); prop_assert!(err < 1e-12, "err {}", err);
            for k in 1..6 { prop_assert!(s[k-1] >= s[k]); }
        }
    }
}
