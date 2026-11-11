//! Grasp-candidate kernels of `grasp.footprint` / `grasp.candidates`.
//! * `min_caliper`: the width sweep over 180 directions (1 deg apart) of the convex hull. Directions u_a = (cos a, sin a) are
//!   computed by NumPy and passed in (so no libm differences); per angle w = max(p) - min(p), p_i = h_i0*u0 + h_i1*u1,
//!   and the first angle with the strictly smallest width wins (reference `if w < best[0]` scan, best starts at 9).
//! * `min_dist_xy`: finger-corridor clearance, min over cloud points of sqrt(dx*dx + dy*dy) per tip
//!   (== np.min(np.linalg.norm(o[:, :2] - t, axis=1)); sqrt is monotone and correctly rounded, so min-then-sqrt
//!   is bit-identical to sqrt-then-min).
pub fn min_caliper(hull: &[[f64; 2]], dirs: &[[f64; 2]]) -> (f64, usize) {
    let mut best = (9., 0usize);
    for (a, u) in dirs.iter().enumerate() {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for h in hull {
            let p = h[0] * u[0] + h[1] * u[1];
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let w = hi - lo;
        if w < best.0 {
            best = (w, a);
        }
    }
    best
}

/// Squared-distance minimum per tip over all points; returns sqrt(min) per tip (+inf when there are no points).
/// Points are processed in blocks against all tips so the cloud is streamed once.
pub fn min_dist_xy(tips: &[[f64; 2]], pts_x: &[f64], pts_y: &[f64]) -> Vec<f64> {
    let mut best = vec![f64::INFINITY; tips.len()];
    for (t, b) in tips.iter().zip(best.iter_mut()) {
        let mut m = f64::INFINITY;
        for (x, y) in pts_x.iter().zip(pts_y) {
            let dx = x - t[0];
            let dy = y - t[1];
            m = m.min(dx * dx + dy * dy);
        }
        *b = m.sqrt();
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn caliper_square_width(s in 0.01f64..1.0, a in 0f64..3.0) {
            let (c, si) = (a.cos(), a.sin());
            let hull: Vec<[f64;2]> = [(0.,0.),(s,0.),(s,s),(0.,s)].iter().map(|&(x,y)| [c*x - si*y, si*x + c*y]).collect();
            let dirs: Vec<[f64;2]> = (0..180).map(|k| { let t = std::f64::consts::PI * k as f64 / 180.; [t.cos(), t.sin()] }).collect();
            let (w, _) = min_caliper(&hull, &dirs);
            prop_assert!(w <= s * 1.01 + 1e-12 && w >= s - 1e-12); // 1-degree sampling: s(cos d + sin d), |d| <= .5 deg
        }
        #[test]
        fn dist_matches_brute(n in 0usize..300, tx in -1f64..1., ty in -1f64..1., seed in any::<u64>()) {
            let mut s = seed | 1; let mut r = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64<<53) as f64 };
            let xs: Vec<f64> = (0..n).map(|_| r()).collect(); let ys: Vec<f64> = (0..n).map(|_| r()).collect();
            let d = min_dist_xy(&[[tx, ty]], &xs, &ys)[0];
            let b = xs.iter().zip(&ys).map(|(x, y)| ((x-tx)*(x-tx) + (y-ty)*(y-ty)).sqrt()).fold(f64::INFINITY, f64::min);
            prop_assert_eq!(d.to_bits(), b.to_bits());
        }
    }
}

/// trays.min_area_rect scan: for every direction u (v = [-u1, u0]) the linear-interpolated q / (1-q) quantiles of
/// xy.u and xy.v (NumPy `quantile`, method 'linear'); returns the first direction with the strictly smallest
/// area (hu-lu)*(hv-lv) and its (lu, hu, lv, hv). Projections are x*u0 + y*u1 (two products, one sum: the same
/// value NumPy's matmul gives for a 2-vector). None if xy or dirs is empty.
pub fn min_area_rect_scan(xy: &[[f64; 2]], dirs: &[[f64; 2]], q: f64) -> Option<(usize, [f64; 4])> {
    if xy.is_empty() || dirs.is_empty() {
        return None;
    }
    let (q0, q1) = (q, 1. - q);
    let mut pu = vec![0.; xy.len()];
    let mut pv = vec![0.; xy.len()];
    let mut best: Option<(f64, usize, [f64; 4])> = None;
    for (k, u) in dirs.iter().enumerate() {
        let v = [-u[1], u[0]];
        for ((a, b), p) in pu.iter_mut().zip(pv.iter_mut()).zip(xy) {
            *a = p[0] * u[0] + p[1] * u[1];
            *b = p[0] * v[0] + p[1] * v[1];
        }
        let lu = crate::perc::quantile_linear(&mut pu, q0);
        let hu = crate::perc::quantile_linear(&mut pu, q1);
        let lv = crate::perc::quantile_linear(&mut pv, q0);
        let hv = crate::perc::quantile_linear(&mut pv, q1);
        let area = (hu - lu) * (hv - lv);
        if best.is_none_or(|b| area < b.0) {
            best = Some((area, k, [lu, hu, lv, hv]));
        }
    }
    best.map(|(_, k, e)| (k, e))
}

#[cfg(test)]
mod rect_tests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn rect_scan_covers_points(pts in proptest::collection::vec((-1.0f64..1.0, -1.0f64..1.0), 5..300)) {
            let xy: Vec<[f64; 2]> = pts.iter().map(|&(x, y)| [x, y]).collect();
            let dirs: Vec<[f64; 2]> = (0..91).map(|i| { let a = std::f64::consts::FRAC_PI_2 * i as f64 / 90.; [a.cos(), a.sin()] }).collect();
            let (k, [lu, hu, lv, hv]) = min_area_rect_scan(&xy, &dirs, 0.).unwrap();
            // q = 0: the extent along the chosen axes contains every point
            let u = dirs[k];
            for p in &xy {
                let a = p[0] * u[0] + p[1] * u[1];
                let b = -p[0] * u[1] + p[1] * u[0];
                prop_assert!(a >= lu - 1e-12 && a <= hu + 1e-12 && b >= lv - 1e-12 && b <= hv + 1e-12);
            }
        }
    }
}
