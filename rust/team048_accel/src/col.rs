//! Port of `team048_solution.planning.CollisionModel.valid / certify / certify_box` (enclosing spheres).
//! Exactness: every per-pair test is the reference expression evaluated in the same order:
//!   environment: sqrt((dx*dx + dy*dy) + dz*dz) <= ((r + R_env) + margin) + inflation[k]     (d = env - p)
//!   self pairs : |p_i - p_k| <= (((r_i + r_k) + margin) + infl_i) + infl_k
//! Speed comes only from (a) no Python/NumPy dispatch, (b) an exact broad phase: environment spheres sorted by x,
//! a robot sphere only visits those with |dx| <= its largest possible threshold (+1e-9 m guard), and
//! (c) early-out sample ordering in `certify` (coarse-to-fine; the AND over all samples is order independent).
use crate::kin::{Chain, Rt, MAXJ, N, Q, V3};
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
            return Env {
                rmax,
                inv: 1. / CELL,
                start: vec![0, 0],
                nx: 1,
                ny: 1,
                ..Default::default()
            };
        }
        let (mut x0, mut y0, mut x1, mut y1) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
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
        let mut e = Env {
            x: vec![0.; n],
            y: vec![0.; n],
            z: vec![0.; n],
            r: vec![0.; n],
            rmax,
            x0,
            y0,
            inv,
            nx,
            ny,
            start,
        };
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
    /// Like `hits`, plus a certificate: None if any sphere is hit (same verdict as `hits`), else the minimum over
    /// environment spheres of (distance - threshold), lower-bounded by `ext` for spheres not visited. The query square
    /// is widened by `ext`: an unvisited sphere has |dx| or |dy| > reach + ext, hence slack > ext.
    fn slack(&self, p: &V3, r: f64, margin: f64, infl: f64, ext: f64) -> Option<f64> {
        if self.x.is_empty() {
            return Some(f64::INFINITY);
        }
        let reach = (((r + self.rmax) + margin) + infl + 1e-9) + ext;
        let mut best = ext;
        let (i0, i1) = match self.range(p[0] - reach, p[0] + reach, self.x0, self.nx) {
            Some(v) => v,
            None => return Some(best),
        };
        let (j0, j1) = match self.range(p[1] - reach, p[1] + reach, self.y0, self.ny) {
            Some(v) => v,
            None => return Some(best),
        };
        for i in i0..=i1 {
            let row = i * self.ny;
            let (s, e) = (
                self.start[row + j0] as usize,
                self.start[row + j1 + 1] as usize,
            );
            for k in s..e {
                let dx = self.x[k] - p[0];
                let dy = self.y[k] - p[1];
                let dz = self.z[k] - p[2];
                let d = ((dx * dx + dy * dy) + dz * dz).sqrt();
                let thr = ((r + self.r[k]) + margin) + infl;
                if d <= thr {
                    return None;
                }
                best = best.min(d - thr);
            }
        }
        Some(best)
    }
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
            let (s, e) = (
                self.start[row + j0] as usize,
                self.start[row + j1 + 1] as usize,
            );
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
    pub env: std::sync::Arc<Env>,
    /// certify with the exact skip certificate (default); false = evaluate every sample (A/B, tests)
    pub skip: bool,
    /// per self pair: (active joints moving both spheres, active joints between them, downstream sphere)
    pair_meta: Vec<(u8, u8, u16)>,
}

impl Collision {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chain: Chain,
        frame: Vec<usize>,
        center: &[[f64; 3]],
        radius: Vec<f64>,
        pairs: Vec<(u16, u16)>,
        levers: Vec<[f64; N]>,
        margin: f64,
    ) -> Result<Collision, String> {
        let s = frame.len();
        if s == 0
            || center.len() != s
            || radius.len() != s
            || levers.len() != s
            || frame.iter().any(|f| *f > chain.joints.len())
            || pairs
                .iter()
                .any(|(a, b)| *a as usize >= s || *b as usize >= s)
        {
            return Err("inconsistent collision model".into());
        }
        // active-joint masks per self pair: joints upstream of both spheres rotate them rigidly together,
        // joints between the two links move only the downstream sphere
        let mut active_of = vec![usize::MAX; chain.joints.len()];
        let mut a = 0;
        for (idx, j) in chain.joints.iter().enumerate() {
            if j.movable {
                active_of[idx] = a;
                a += 1;
            }
        }
        let pair_meta = pairs
            .iter()
            .map(|&(i, k)| {
                let (fi, fk) = (frame[i as usize], frame[k as usize]);
                let (lo, hi, down) = if fi <= fk { (fi, fk, k) } else { (fk, fi, i) };
                let (mut up, mut between) = (0u8, 0u8);
                for (idx, &act) in active_of.iter().enumerate() {
                    if act == usize::MAX || act >= 8 {
                        continue;
                    }
                    if idx < lo {
                        up |= 1 << act;
                    } else if idx < hi {
                        between |= 1 << act;
                    }
                }
                (up, between, down)
            })
            .collect();
        Ok(Collision {
            chain,
            frame,
            center: center.iter().map(|c| V3::new(c[0], c[1], c[2])).collect(),
            radius,
            pairs,
            levers,
            margin,
            env: std::sync::Arc::new(Env::default()),
            skip: true,
            pair_meta,
        })
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

    /// Safety distance for the skip certificate (m): far above FK/sampling rounding (~1e-15 m).
    const SKIP_EPS: f64 = 1e-9;
    /// Environment look-ahead of the skip certificate (m).
    const SKIP_EXT: f64 = 0.06;
    /// valid(q, infl) with a certificate: None if invalid (identical verdict), else m >= 0 such that the next m
    /// samples of the sweep are provably valid too. Per step every sphere centre moves at most infl[k]
    /// (= levers[k] . |b-a| / steps, the reference's own swept-volume bound: |dp_k| <= sum_j lever_kj |dq_j|), so an
    /// environment test with slack s stays satisfied for m steps if m*infl[k] < s - eps, a self pair if
    /// m*(infl[i]+infl[k]) < s - eps.
    fn valid_skip(
        &self,
        q: &Q,
        infl: &[f64],
        w: &mut Vec<V3>,
        max_skip: usize,
        pr: &[(f64, f64)],
    ) -> Option<usize> {
        self.world(q, w);
        let mut m = max_skip as f64;
        let bound = |slack: f64, rate: f64, m: f64| -> f64 {
            if rate > 0. {
                m.min(((slack - Self::SKIP_EPS) / rate).floor().max(0.))
            } else if slack > Self::SKIP_EPS {
                m
            } else {
                0.
            }
        };
        for k in 0..w.len() {
            let s = self
                .env
                .slack(&w[k], self.radius[k], self.margin, infl[k], Self::SKIP_EXT)?;
            m = bound(s, infl[k], m);
        }
        for (p, &(i, k)) in self.pairs.iter().enumerate() {
            let (i, k) = (i as usize, k as usize);
            let d = w[i] - w[k];
            let dist = ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt();
            let thr = (((self.radius[i] + self.radius[k]) + self.margin) + infl[i]) + infl[k];
            if dist <= thr {
                return None;
            }
            let crude = infl[i] + infl[k];
            let mut mp = bound(dist - thr, crude, m);
            if mp < m {
                // finer relative-motion bound: |dr| <= n*(U*(|r0| + n*crude) + V) after n steps (U: joints moving both
                // spheres rotate r rigidly, V: joints between the links move the downstream sphere, crude bounds |r|)
                let (u, v) = pr[p];
                let rhs = dist - thr - Self::SKIP_EPS;
                let (qa, qb) = (u * crude, u * dist + v);
                let mut n = if qa > 0. {
                    ((-qb + (qb * qb + 4. * qa * rhs).sqrt()) / (2. * qa)).floor()
                } else if qb > 0. {
                    (rhs / qb).floor()
                } else {
                    m
                };
                // guard the rounding of the root: step down until the bound holds as evaluated
                while n > 0. && n * (u * (dist + n * crude) + v) >= rhs {
                    n -= 1.;
                }
                mp = mp.max(n.max(0.).min(m));
            }
            m = mp;
        }
        Some(m as usize)
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
                ch.iter()
                    .all(|&i| self.valid_buf(&sample(i), &infl, &mut w))
            });
        }
        let mut w = Vec::with_capacity(self.center.len());
        if steps < 16 || !self.skip {
            return order
                .iter()
                .all(|&i| self.valid_buf(&sample(i), &infl, &mut w));
        }
        // 1) coarse samples first (early exit on a colliding path, as before) ...
        let coarse = order.len().min(9);
        if !order[..coarse]
            .iter()
            .all(|&i| self.valid_buf(&sample(i), &infl, &mut w))
        {
            return false;
        }
        // 2) ... then every sample in order, skipping the ones the certificate of a checked sample proves valid.
        //    Same AND over the same samples: a skipped sample is valid by the bound, a checked one by evaluation.
        let mut ad = [0.; N];
        for (k, x) in ad.iter_mut().enumerate() {
            *x = d[k].abs() / steps as f64;
        }
        let pr: Vec<(f64, f64)> = self
            .pair_meta
            .iter()
            .map(|&(up, between, down)| {
                let (mut u, mut v) = (0., 0.);
                for (a, x) in ad.iter().enumerate() {
                    if up >> a & 1 == 1 {
                        u += x;
                    } else if between >> a & 1 == 1 {
                        v += self.levers[down as usize][a] * x;
                    }
                }
                (u, v)
            })
            .collect();
        let mut i = 0usize;
        while i <= steps {
            match self.valid_skip(&sample(i), &infl, &mut w, steps, &pr) {
                None => return false,
                Some(m) => i += m + 1,
            }
        }
        true
    }
    /// Reference-order certify without the skip certificate (tests / benchmarks).
    pub fn certify_plain(&self, a: &Q, b: &Q, max_step: f64) -> bool {
        let (steps, infl, d) = self.sweep_setup(a, b, max_step);
        let mut w = Vec::with_capacity(self.center.len());
        (0..=steps).all(|i| self.valid_buf(&(a + d * (i as f64 / steps as f64)), &infl, &mut w))
    }
}

impl Collision {
    /// `Arm.solve`: for each seed in order, IK (same kernel and arguments as `Chain.ik`), and the first converged q
    /// that is collision-free with zero inflation is returned with its seed index. A seed that `Chain.validate_q`
    /// would reject (non-finite, or outside [lower-1e-9, upper+1e-9]) is skipped like the reference's ValueError.
    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        &self,
        target: &Rt,
        seeds: &[Q],
        max_iter: usize,
        position_tol: f64,
        angle_tol: f64,
        length_scale: f64,
        trust_radius: f64,
        q_pref: Option<&Q>,
        w_null: f64,
    ) -> Option<(usize, Q)> {
        let zero = vec![0.; self.center.len()];
        let mut w = Vec::with_capacity(self.center.len());
        for (k, s) in seeds.iter().enumerate() {
            if !(0..N).all(|i| {
                s[i].is_finite()
                    && s[i] >= self.chain.lower[i] - 1e-9
                    && s[i] <= self.chain.upper[i] + 1e-9
            }) {
                continue;
            }
            if let Some(q) = self.chain.ik(
                target,
                s,
                max_iter,
                position_tol,
                angle_tol,
                length_scale,
                trust_radius,
                q_pref,
                w_null,
            ) {
                if self.valid_buf(&q, &zero, &mut w) {
                    return Some((k, q));
                }
            }
        }
        None
    }
    /// Independent `solve` calls (offline studies): one per target, each with its own optional extra first seed.
    /// rayon over targets when `parallel`; results are identical to sequential calls (no shared state).
    #[allow(clippy::too_many_arguments)]
    pub fn solve_batch(
        &self,
        targets: &[Rt],
        first: &[Option<Q>],
        seeds: &[Q],
        max_iter: usize,
        position_tol: f64,
        angle_tol: f64,
        length_scale: f64,
        trust_radius: f64,
        q_pref: Option<&Q>,
        w_null: f64,
        parallel: bool,
    ) -> Vec<Option<(usize, Q)>> {
        let one = |(t, f): (&Rt, &Option<Q>)| -> Option<(usize, Q)> {
            match f {
                Some(f0) => {
                    let mut all = Vec::with_capacity(seeds.len() + 1);
                    all.push(*f0);
                    all.extend_from_slice(seeds);
                    self.solve(
                        t,
                        &all,
                        max_iter,
                        position_tol,
                        angle_tol,
                        length_scale,
                        trust_radius,
                        q_pref,
                        w_null,
                    )
                }
                None => self.solve(
                    t,
                    seeds,
                    max_iter,
                    position_tol,
                    angle_tol,
                    length_scale,
                    trust_radius,
                    q_pref,
                    w_null,
                ),
            }
        };
        if parallel {
            targets.par_iter().zip(first.par_iter()).map(one).collect()
        } else {
            targets.iter().zip(first.iter()).map(one).collect()
        }
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
                if ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt()
                    <= ((c.radius[k] + er) + c.margin) + infl[k]
                {
                    return false;
                }
            }
        }
        for &(i, k) in &c.pairs {
            let d = w[i as usize] - w[k as usize];
            if ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]).sqrt()
                <= (((c.radius[i as usize] + c.radius[k as usize]) + c.margin) + infl[i as usize])
                    + infl[k as usize]
            {
                return false;
            }
        }
        true
    }
    fn model(seed: u64) -> (Collision, Vec<[f64; 3]>, Vec<f64>) {
        let chain = crate::kin::tests::test_chain();
        let mut rng = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut rnd = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        let nj = chain.joints.len();
        let s = 20;
        let frame: Vec<usize> = (0..s).map(|k| 1 + k % nj).collect();
        let center: Vec<[f64; 3]> = (0..s)
            .map(|_| [0.05 * (rnd() - 0.5), 0.05 * (rnd() - 0.5), -0.1 * rnd()])
            .collect();
        let radius: Vec<f64> = (0..s).map(|_| 0.02 + 0.03 * rnd()).collect();
        let mut pairs = vec![];
        for i in 0..s {
            for k in i + 1..s {
                if frame[k] > frame[i] + 2 {
                    pairs.push((i as u16, k as u16));
                }
            }
        }
        // rigorous swept-volume levers, as CollisionModel builds them: for every movable joint up to the sphere's
        // link, sum of the downstream origin translations + |centre| (the skip certificate relies on this bound)
        let levers: Vec<[f64; N]> = (0..s)
            .map(|k| {
                let stop = frame[k] - 1;
                let mut row = [0.; N];
                let mut a = 0;
                for (idx, j) in chain.joints.iter().enumerate() {
                    if j.movable {
                        if idx <= stop {
                            let tail: f64 = chain.joints[idx + 1..=stop]
                                .iter()
                                .map(|o| o.origin.t.norm())
                                .sum();
                            let c = center[k];
                            row[a] = tail + (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                        }
                        a += 1;
                    }
                }
                row
            })
            .collect();
        let mut c = Collision::new(chain, frame, &center, radius, pairs, levers, 0.015).unwrap();
        let envc: Vec<[f64; 3]> = (0..400)
            .map(|_| [rnd() * 1.2 - 0.3, rnd() * 1.4 - 0.7, rnd() * 0.8 - 0.6])
            .collect();
        let envr: Vec<f64> = (0..400).map(|_| 0.01 + 0.1 * rnd()).collect();
        c.env = std::sync::Arc::new(Env::new(&envc, &envr));
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
            prop_assert_eq!(c.certify_plain(&a, &b, 0.003), seq);
            prop_assert_eq!(c.certify(&a, &b, 0.003, true), seq);
        }
        #[test]
        fn skip_certificate_matches_every_sample(seed in 0u64..400, u in proptest::array::uniform7(0f64..1.), v in proptest::array::uniform7(0f64..1.), t in 0.02f64..1., m in 0.001f64..0.05) {
            let (mut c, _, _) = model(seed);
            c.margin = m;
            let a = Q::from_fn(|k, _| c.chain.lower[k] + (c.chain.upper[k] - c.chain.lower[k]) * u[k]);
            let b0 = Q::from_fn(|k, _| c.chain.lower[k] + (c.chain.upper[k] - c.chain.lower[k]) * v[k]);
            let b = a + (b0 - a) * t;
            prop_assert_eq!(c.certify(&a, &b, 0.003, false), c.certify_plain(&a, &b, 0.003));
        }
        #[test]
        fn levers_bound_sphere_motion(seed in 0u64..200, u in proptest::array::uniform7(0f64..1.), v in proptest::array::uniform7(0f64..1.)) {
            let (c, _, _) = model(seed);
            let a = Q::from_fn(|k, _| c.chain.lower[k] + (c.chain.upper[k] - c.chain.lower[k]) * u[k]);
            let b = Q::from_fn(|k, _| c.chain.lower[k] + (c.chain.upper[k] - c.chain.lower[k]) * v[k]);
            let (mut wa, mut wb) = (vec![], vec![]);
            c.world(&a, &mut wa);
            c.world(&b, &mut wb);
            for k in 0..wa.len() {
                let bound: f64 = (0..N).map(|j| c.levers[k][j] * (b[j] - a[j]).abs()).sum();
                prop_assert!((wa[k] - wb[k]).norm() <= bound + 1e-12);
            }
        }
        #[test]
        fn coarse_to_fine_is_permutation(steps in 1usize..3000) {
            let mut o = coarse_to_fine(steps); prop_assert_eq!(o[0], 0); o.sort(); prop_assert_eq!(o, (0..=steps).collect::<Vec<_>>());
        }
    }
}
