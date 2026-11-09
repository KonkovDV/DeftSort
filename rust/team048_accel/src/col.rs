//! Port of `team048_solution.planning.CollisionModel.valid / certify / certify_box` (enclosing spheres).
//! Exactness: every per-pair test is the reference expression evaluated in the same order:
//!   environment: sqrt((dx*dx + dy*dy) + dz*dz) <= ((r + R_env) + margin) + inflation[k]     (d = env - p)
//!   self pairs : |p_i - p_k| <= (((r_i + r_k) + margin) + infl_i) + infl_k
//! Speed comes only from (a) no Python/NumPy dispatch, (b) an exact broad phase: environment spheres sorted by x,
//! a robot sphere only visits those with |dx| <= its largest possible threshold (+1e-9 m guard), and
//! (c) early-out sample ordering in `certify` (coarse-to-fine; the AND over all samples is order independent).
use crate::kin::{Chain, Q, Rt, MAXJ, N, V3};
use rayon::prelude::*;

/// Environment spheres in a uniform xy grid (exact broad phase): a query visits every cell overlapping the square
/// [p ± reach], reach = largest possible threshold + 1e-9 m. A sphere outside that square has |dx| or |dy| > reach,
/// so its distance exceeds every threshold and the reference would not report it either.
#[derive(Clone, Debug, Default)]
pub struct Env {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    r: Vec<f64>,
    rmax: f64,
    x0: f64,
    y0: f64,
    inv: f64,
    nx: usize,
    ny: usize,
    start: Vec<u32>, // CSR: cell c holds spheres start[c]..start[c+1] (spheres stored cell by cell)
}
pub const CELL: f64 = 0.08;
impl Env {
    pub fn new(c: &[[f64; 3]], r: &[f64]) -> Env {
        let n = c.len();
        let rmax = r.iter().cloned().fold(0., f64::max);
        if n == 0 {
            return Env { rmax, inv: 1. / CELL, start: vec![0, 0], nx: 1, ny: 1, ..Default::default() };
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in c {
            x0 = x0.min(p[0]);
            y0 = y0.min(p[1]);
            x1 = x1.max(p[0]);
            y1 = y1.max(p[1]);
        }
        let inv = 1. / CELL;
        let nx = (((x1 - x0) * inv).floor() as usize + 1).min(4096);
        let ny = (((y1 - y0) * inv).floor() as usize + 1).min(4096);
        let cell = |p: &[f64; 3]| -> usize {
            let i = (((p[0] - x0) * inv).floor().max(0.) as usize).min(nx - 1);
            let j = (((p[1] - y0) * inv).floor().max(0.) as usize).min(ny - 1);
            i * ny + j
        };
        let mut count = vec![0u32; nx * ny + 1];
        for p in c {
            count[cell(p) + 1] += 1;
        }
        for k in 0..nx * ny {
            count[k + 1] += count[k];
        }
        let start = count.clone();
        let mut fill = count;
        let mut e = Env { x: vec![0.; n], y: vec![0.; n], z: vec![0.; n], r: vec![0.; n], rmax, x0, y0, inv, nx, ny, start };
        for (i, p) in c.iter().enumerate() {
            let k = cell(p);
            let at = fill[k] as usize;
            fill[k] += 1;
            e.x[at] = p[0];
            e.y[at] = p[1];
            e.z[at] = p[2];
            e.r[at] = r[i];
        }
        e
    }
    pub fn len(&self) -> usize {
        self.x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
    #[inline(always)]
    fn range(&self, lo: f64, hi: f64, o: f64, n: usize) -> Option<(usize, usize)> {
        let a = ((lo - o) * self.inv).floor();
        let b = ((hi - o) * self.inv).floor();
        if b < 0. || a > (n - 1) as f64 {
            return None;
        }
        Some((a.max(0.) as usize, (b as usize).min(n - 1)))
    }
    /// true if sphere (p, r) with margin+inflation touches any environment sphere (reference `<=` semantics).
    #[inline(always)]
    fn hits(&self, p: &V3, r: f64, margin: f64, infl: f64) -> bool {
        if self.x.is_empty() {
            return false;
        }
        let reach = ((r + self.rmax) + margin) + infl + 1e-9;
        // floor((x - x0) / cell) is monotonic in x, so every sphere with x in [p-reach, p+reach] lies in [i0, i1].
        let (i0, i1) = match self.range(p[0] - reach, p[0] + reach, self.x0, self.nx) {
            Some(v) => v,
            None => return false,
        };
        let (j0, j1) = match self.range(p[1] - reach, p[1] + reach, self.y0, self.ny) {
            Some(v) => v,
            None => return false,
        };
        for i in i0..=i1 {
            let row = i * self.ny;
            let (s, e) = (self.start[row + j0] as usize, self.start[row + j1 + 1] as usize);
            for k in s..e {
                let dx = self.x[k] - p[0];
                let dy = self.y[k] - p[1];
                let dz = self.z[k] - p[2];
                let d = ((dx * dx + dy * dy) + dz * dz).sqrt();
                if d <= ((r + self.r[k]) + margin) + infl {
                    return true;
                }
            }
        }
        false
    }
}

#[derive(Clone, Debug)]
pub struct Collision {
    pub chain: Chain,
    frame: Vec<usize>,
    center: Vec<V3>,
    radius: Vec<f64>,
    pairs: Vec<(u16, u16)>,
    levers: Vec<[f64; N]>,
    pub margin: f64,
    pub env: Env,
}

impl Collision {
    #[allow(clippy::too_many_arguments)]
    pub fn new(chain: Chain, frame: Vec<usize>, center: &[[f64; 3]], radius: Vec<f64>, pairs: Vec<(u16, u16)>, levers: Vec<[f64; N]>, margin: f64) -> Result<Collision, String> {
        let s = frame.len();
        if s == 0 || center.len() != s || radius.len() != s || levers.len() != s || frame.iter().any(|f| *f > chain.joints.len()) || pairs.iter().any(|(a, b)| *a as usize >= s || *b as usize >= s) {
            return Err("inconsistent collision model".into());
        }
        Ok(Collision { chain, frame, center: center.iter().map(|c| V3::new(c[0], c[1], c[2])).collect(), radius, pairs, levers, margin, env: Env::default() })
    }
    pub fn n_spheres(&self) -> usize {
        self.frame.len()
    }

    #[inline]
    fn world(&self, q: &Q, out: &mut Vec<V3>) {
        let mut fr = [Rt::IDENTITY; MAXJ + 1];
        self.chain.frames_into(q, &mut fr);
        out.clear();
        for (k, c) in self.center.iter().enumerate() {
            out.push(fr[self.frame[k]].apply(c));
        }
    }

    /// valid(q, inflation): inflation per sphere.
    pub fn valid(&self, q: &Q, infl: &[f64]) -> bool {
        let mut w = Vec::with_capacity(self.center.len());
        self.valid_buf(q, infl, &mut w)
    }
    #[inline]
    pub fn valid_buf(&self, q: &Q, infl: &[f64], w: &mut Vec<V3>) -> bool {
        self.world(q, w);
        for k in 0..w.len() {
            if self.env.hits(&w[k], self.radius[k], self.margin, infl[k]) {
                return false;
            }
        }
        for &(i, k) in &self.pairs {
            let (i, k) = (i as usize, k as usize);
            let d = w[i] - w[k];
            let dist = ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt();
            if dist <= (((self.radius[i] + self.radius[k]) + self.margin) + infl[i]) + infl[k] {
                return false;
            }
        }
        true
    }

    /// Sample count and inflation exactly as the reference.
    fn sweep_setup(&self, a: &Q, b: &Q, max_step: f64) -> (usize, Vec<f64>, Q) {
        let mut dmax: f64 = 0.;
        let mut ad = Q::zeros();
        for k in 0..N {
            ad[k] = (b[k] - a[k]).abs();
            dmax = dmax.max(ad[k]);
        }
        let steps = ((dmax / max_step).ceil() as usize).max(1);
        let infl = self
            .levers
            .iter()
            .map(|row| {
                let mut s = 0.;
                for k in 0..N {
                    s += row[k] * ad[k];
                }
                s / steps as f64
            })
            .collect();
        (steps, infl, b - a)
    }

    /// certify(a, b): AND of valid(a + (b-a)*(i/steps), inflation) over i = 0..=steps (reference sample points).
    pub fn certify(&self, a: &Q, b: &Q, max_step: f64, parallel: bool) -> bool {
        let (steps, infl, d) = self.sweep_setup(a, b, max_step);
        let sample = |i: usize| -> Q { a + d * (i as f64 / steps as f64) };
        let order = coarse_to_fine(steps);
        if parallel && steps >= 128 {
            // chunks in coarse-to-fine order; rayon's all() stops the other workers at the first collision.
            let chunk = (order.len() / (4 * rayon::current_num_threads())).max(16);
            return order.par_chunks(chunk).all(|ch| {
                let mut w = Vec::with_capacity(self.center.len());
                ch.iter().all(|&i| self.valid_buf(&sample(i), &infl, &mut w))
            });
        }
        let mut w = Vec::with_capacity(self.center.len());
        order.iter().all(|&i| self.valid_buf(&sample(i), &infl, &mut w))
    }
}

/// 0, steps, then midpoints level by level: finds a colliding stretch early, visits every index exactly once.
pub fn coarse_to_fine(steps: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(steps + 1);
    let mut seen = vec![false; steps + 1];
    let mut stride = steps.next_power_of_two();
    while stride >= 1 {
        let mut i = 0;
        while i <= steps {
            if !seen[i] {
                seen[i] = true;
                out.push(i);
            }
            i += stride;
        }
        if !seen[steps] {
            seen[steps] = true;
            out.push(steps);
        }
        stride /= 2;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    fn brute(c: &Collision, q: &Q, infl: &[f64], envc: &[[f64; 3]], envr: &[f64]) -> bool {
        let mut w = vec![];
        c.world(q, &mut w);
        for k in 0..w.len() {
            for (e, er) in envc.iter().zip(envr) {
                let d = V3::new(e[0], e[1], e[2]) - w[k];
                if ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt() <= ((c.radius[k] + er) + c.margin) + infl[k] {
                    return false;
                }
            }
        }
        for &(i, k) in &c.pairs {
            let d = w[i as usize] - w[k as usize];
            if ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt() <= (((c.radius[i as usize] + c.radius[k as usize]) + c.margin) + infl[i as usize]) + infl[k as usize] {
                return false;
            }
        }
        true
    }
    fn model(seed: u64) -> (Collision, Vec<[f64; 3]>, Vec<f64>) {
        let chain = crate::kin::tests::test_chain();
        let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut rnd = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        let nj = chain.joints.len();
        let s = 20;
        let frame: Vec<usize> = (0..s).map(|k| 1 + k % nj).collect();
        let center: Vec<[f64; 3]> = (0..s).map(|_| [0.05 * (rnd() - 0.5), 0.05 * (rnd() - 0.5), -0.1 * rnd()]).collect();
        let radius: Vec<f64> = (0..s).map(|_| 0.02 + 0.03 * rnd()).collect();
        let mut pairs = vec![];
        for i in 0..s {
            for k in i + 1..s {
                if frame[k] > frame[i] + 2 {
                    pairs.push((i as u16, k as u16));
                }
            }
        }
        let levers = (0..s).map(|_| [0.5; N]).collect();
        let mut c = Collision::new(chain, frame, &center, radius, pairs, levers, 0.015).unwrap();
        let envc: Vec<[f64; 3]> = (0..400).map(|_| [rnd() * 1.2 - 0.3, rnd() * 1.4 - 0.7, rnd() * 0.8 - 0.6]).collect();
        let envr: Vec<f64> = (0..400).map(|_| 0.01 + 0.1 * rnd()).collect();
        c.env = Env::new(&envc, &envr);
        (c, envc, envr)
    }
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]
        #[test]
        fn broad_phase_is_exact(seed in 0u64..1000, u in proptest::array::uniform7(0f64..1.), inf in 0f64..0.05) {
            let (c, ec, er) = model(seed);
            let q = Q::from_fn(|k, _| c.chain.lower[k] + (c.chain.upper[k] - c.chain.lower[k]) * u[k]);
            let infl = vec![inf; c.n_spheres()];
            prop_assert_eq!(c.valid(&q, &infl), brute(&c, &q, &infl, &ec, &er));
        }
        #[test]
        fn certify_order_and_parallel_agree(seed in 0u64..200, u in proptest::array::uniform7(0f64..1.), v in proptest::array::uniform7(0f64..1.)) {
            let (c, _, _) = model(seed);
            let a = Q::from_fn(|k, _| c.chain.lower[k] + (c.chain.upper[k] - c.chain.lower[k]) * u[k]);
            let b = Q::from_fn(|k, _| c.chain.lower[k] + (c.chain.upper[k] - c.chain.lower[k]) * v[k]);
            let (steps, infl, d) = c.sweep_setup(&a, &b, 0.003);
            let seq = (0..=steps).all(|i| c.valid(&(a + d * (i as f64 / steps as f64)), &infl));
            prop_assert_eq!(c.certify(&a, &b, 0.003, false), seq);
            prop_assert_eq!(c.certify(&a, &b, 0.003, true), seq);
        }
        #[test]
        fn coarse_to_fine_is_permutation(steps in 1usize..3000) {
            let mut o = coarse_to_fine(steps); prop_assert_eq!(o[0], 0); o.sort(); prop_assert_eq!(o, (0..=steps).collect::<Vec<_>>());
        }
    }
}
