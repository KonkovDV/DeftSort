//! Perception kernels: RANSAC inlier counting for `perception.table_plane` and the 4-connected depth-continuity
//! flood fill of `perception.segment`. Candidate generation (the NumPy RNG stream) and all statistics stay in
//! Python, so the random sequence and the final plane/feature arithmetic are untouched.
use rayon::prelude::*;

/// For each candidate plane (n, offset) count points with |p.n + offset| < tol. Points: N x 3 row-major.
/// Returns the counts (same order as candidates). Parallel over candidates.
pub fn plane_inlier_counts(
    points: &[f64],
    planes: &[[f64; 4]],
    tol: f64,
    parallel: bool,
) -> Vec<u64> {
    let npts = points.len() / 3;
    let count = |pl: &[f64; 4]| -> u64 {
        let (a, b, c, d) = (pl[0], pl[1], pl[2], pl[3]);
        let mut cnt = 0u64;
        for i in 0..npts {
            let p = &points[3 * i..3 * i + 3];
            // (p @ n) + offset, 3-term dot evaluated left to right
            let h = ((p[0] * a + p[1] * b) + p[2] * c) + d;
            cnt += (h.abs() < tol) as u64;
        }
        cnt
    };
    if parallel {
        planes.par_iter().map(count).collect()
    } else {
        planes.iter().map(count).collect()
    }
}

/// Exact port of the DFS in perception.segment: seeds in row-major (np.nonzero) order, explicit stack, neighbour
/// order up, down, left, right; neighbour accepted if mask && !seen && |p_n - p| < max_jump.
/// Returns (pixels as flat [y0,x0,y1,x1,...] in visit order, component start offsets incl. final end).
pub fn flood_fill(
    mask: &[bool],
    p: &[f64],
    h: usize,
    w: usize,
    max_jump: f64,
) -> (Vec<u32>, Vec<usize>) {
    let mut seen = vec![false; h * w];
    let mut pix: Vec<u32> = Vec::new();
    let mut off = vec![0usize];
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for v in 0..h {
        for u in 0..w {
            let id = v * w + u;
            if !mask[id] || seen[id] {
                continue;
            }
            seen[id] = true;
            stack.push((v, u));
            while let Some((y, x)) = stack.pop() {
                pix.push(y as u32);
                pix.push(x as u32);
                let c = 3 * (y * w + x);
                let (px, py, pz) = (p[c], p[c + 1], p[c + 2]);
                let nb: [(isize, isize); 4] = [
                    (y as isize - 1, x as isize),
                    (y as isize + 1, x as isize),
                    (y as isize, x as isize - 1),
                    (y as isize, x as isize + 1),
                ];
                for (yy, xx) in nb {
                    if yy < 0 || xx < 0 || yy >= h as isize || xx >= w as isize {
                        continue;
                    }
                    let (yy, xx) = (yy as usize, xx as usize);
                    let nid = yy * w + xx;
                    if !mask[nid] || seen[nid] {
                        continue;
                    }
                    let k = 3 * nid;
                    let (dx, dy, dz) = (p[k] - px, p[k + 1] - py, p[k + 2] - pz);
                    if ((dx * dx + dy * dy) + dz * dz).sqrt() < max_jump {
                        seen[nid] = true;
                        stack.push((yy, xx));
                    }
                }
            }
            off.push(pix.len() / 2);
        }
    }
    (pix, off)
}

/// Inlier counts for many planes, structure-of-arrays input (x, y, z separate) and 4 planes per pass over the
/// points (one memory sweep feeds 4 accumulators; the inner loop is branch-free and vectorises with baseline SSE2).
/// Per point and plane the expression is exactly `plane_inlier_counts`: |((x*a + y*b) + z*c) + d| < tol.
pub fn plane_inlier_counts_soa(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    planes: &[[f64; 4]],
    tol: f64,
    parallel: bool,
) -> Vec<u64> {
    let block = |pl: &[[f64; 4]]| -> Vec<u64> {
        let mut acc = [0u64; 4];
        let k = pl.len();
        let p = |j: usize| {
            if j < k {
                pl[j]
            } else {
                [0., 0., 0., f64::INFINITY]
            }
        }; // padding plane: never an inlier
        let (p0, p1, p2, p3) = (p(0), p(1), p(2), p(3));
        for i in 0..x.len() {
            let (xi, yi, zi) = (x[i], y[i], z[i]);
            acc[0] += ((((xi * p0[0] + yi * p0[1]) + zi * p0[2]) + p0[3]).abs() < tol) as u64;
            acc[1] += ((((xi * p1[0] + yi * p1[1]) + zi * p1[2]) + p1[3]).abs() < tol) as u64;
            acc[2] += ((((xi * p2[0] + yi * p2[1]) + zi * p2[2]) + p2[3]).abs() < tol) as u64;
            acc[3] += ((((xi * p3[0] + yi * p3[1]) + zi * p3[2]) + p3[3]).abs() < tol) as u64;
        }
        acc[..k].to_vec()
    };
    if parallel {
        planes.par_chunks(4).flat_map_iter(block).collect()
    } else {
        planes.chunks(4).flat_map(block).collect()
    }
}

#[inline(always)]
fn count_chunk_generic(x: &[f64], y: &[f64], z: &[f64], p: [f64; 4], tol: f64) -> u64 {
    // same expression per point as plane_inlier_counts (Rust never contracts mul+add into FMA, so the count does
    // not depend on the instruction set); u32 lane accumulator keeps the compare/add packed
    let mut cnt = 0u32;
    for ((xi, yi), zi) in x.iter().zip(y).zip(z) {
        cnt += (((((xi * p[0]) + yi * p[1]) + zi * p[2]) + p[3]).abs() < tol) as u32;
    }
    cnt as u64
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_chunk_avx2(x: &[f64], y: &[f64], z: &[f64], p: [f64; 4], tol: f64) -> u64 {
    count_chunk_generic(x, y, z, p, tol)
}

/// Runtime dispatch: AVX2 when the CPU has it (detected once), else the baseline x86-64 (SSE2) build.
/// The wheel itself stays baseline x86-64; results are identical on both paths (same IEEE operations, no FMA).
#[inline]
fn count_chunk(x: &[f64], y: &[f64], z: &[f64], p: [f64; 4], tol: f64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
            && std::env::var_os("TEAM048_ACCEL_NO_AVX2").is_none()
        {
            // SAFETY: the CPU supports AVX2 (checked above).
            return unsafe { count_chunk_avx2(x, y, z, p, tol) };
        }
    }
    count_chunk_generic(x, y, z, p, tol)
}

/// First candidate with the maximal inlier count, if that count is > 0 (== `k = argmax(counts); counts[k] > 0`),
/// with exact pruning: candidates are scanned in order and a later one only wins with a strictly larger count, so
/// counting stops as soon as partial + remaining points <= best (it can no longer win). Returns (index, count).
pub fn ransac_best(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    planes: &[[f64; 4]],
    tol: f64,
) -> Option<(usize, u64)> {
    const CHUNK: usize = 2048;
    let n = x.len();
    let mut best: Option<(usize, u64)> = None;
    let mut score = 0u64;
    for (k, pl) in planes.iter().enumerate() {
        let (a, b, c, d) = (pl[0], pl[1], pl[2], pl[3]);
        let mut cnt = 0u64;
        let mut start = 0;
        let mut dead = false;
        while start < n {
            let end = (start + CHUNK).min(n);
            cnt += count_chunk(
                &x[start..end],
                &y[start..end],
                &z[start..end],
                [a, b, c, d],
                tol,
            );
            start = end;
            if cnt + (n - start) as u64 <= score {
                dead = true;
                break;
            }
        }
        if !dead && cnt > score {
            score = cnt;
            best = Some((k, cnt));
        }
    }
    best
}

/// node.observe table masks: valid & x_lo < x < x_hi & |y| < y_abs & not inside any exclusion rectangle
/// (|x - cx| < hx and |y - cy| < hy), element-wise identical to the NumPy expressions.
pub fn table_mask(
    p: &[f64],
    valid: &[bool],
    x_lo: f64,
    x_hi: f64,
    y_abs: f64,
    rects: &[[f64; 4]],
) -> Vec<bool> {
    valid
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let (px, py) = (p[3 * i], p[3 * i + 1]);
            v && px > x_lo
                && px < x_hi
                && py.abs() < y_abs
                && !rects
                    .iter()
                    .any(|r| (px - r[0]).abs() < r[2] && (py - r[1]).abs() < r[3])
        })
        .collect()
}

/// Rows of an (h*w, 3) cloud where mask is set and all three coordinates are finite, in row-major order
/// (== `p = P[valid]; p = p[np.isfinite(p).all(axis=1)]`), returned as SoA.
pub fn compact_finite(p: &[f64], mask: &[bool]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = mask.iter().filter(|&&m| m).count();
    let (mut x, mut y, mut z) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (i, &m) in mask.iter().enumerate() {
        if m {
            let (a, b, c) = (p[3 * i], p[3 * i + 1], p[3 * i + 2]);
            if a.is_finite() && b.is_finite() && c.is_finite() {
                x.push(a);
                y.push(b);
                z.push(c);
            }
        }
    }
    (x, y, z)
}

/// RANSAC candidate planes of `perception.table_plane_py` from the NumPy-compatible stream: per iteration
/// a, b, c = p[choice(n, 3, replace=False)], normal = cross(b-a, c-a) (NumPy's cross expression), skip if
/// |normal| < 1e-9, normalise, flip to n_z >= 0, skip if n_z < .96, offset = -(n . a).
/// Returns [n_x, n_y, n_z, offset] per accepted candidate (iteration order).
pub fn ransac_candidates(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    rng: &mut crate::rng::Pcg64,
    iterations: usize,
) -> Option<Vec<[f64; 4]>> {
    let n = x.len() as u64;
    let mut idx = [0u64; 3];
    let mut out = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        rng.choice_floyd(n, 3, &mut idx)?;
        let g = |k: usize| -> [f64; 3] {
            let i = idx[k] as usize;
            [x[i], y[i], z[i]]
        };
        let (a, b, c) = (g(0), g(1), g(2));
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let mut nr = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let len = ((nr[0] * nr[0] + nr[1] * nr[1]) + nr[2] * nr[2]).sqrt();
        if len < 1e-9 {
            continue;
        }
        for e in nr.iter_mut() {
            *e /= len;
        }
        if nr[2] < 0. {
            for e in nr.iter_mut() {
                *e = -*e;
            }
        }
        if nr[2] < 0.96 {
            continue;
        }
        let off = -((nr[0] * a[0] + nr[1] * a[1]) + nr[2] * a[2]);
        out.push([nr[0], nr[1], nr[2], off]);
    }
    Some(out)
}

/// Height above the plane, |n|-weighted, the expression used by every Rust mask: ((x*n0 + y*n1) + z*n2) + d.
#[inline(always)]
pub fn height(p: &[f64], n: &[f64; 3], d: f64) -> f64 {
    ((p[0] * n[0] + p[1] * n[1]) + p[2] * n[2]) + d
}

/// Back-projection of a depth image (mm, float) with pinhole K and the camera->pelvis transform (R row-major, t):
/// X = ((u - cx) * Z) / fx, Y = ((v - cy) * Z) / fy, Z = depth / 1000, P = R @ [X, Y, Z] + t  (node.observe).
#[allow(clippy::too_many_arguments)]
pub fn deproject(
    depth: &[f64],
    h: usize,
    w: usize,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    r: &[f64; 9],
    t: &[f64; 3],
    parallel: bool,
) -> Vec<f64> {
    let mut out = vec![0.; h * w * 3];
    let row = |v: usize, o: &mut [f64]| {
        let vf = v as f64;
        for u in 0..w {
            let zc = depth[v * w + u] / 1000.;
            let xc = ((u as f64 - cx) * zc) / fx;
            let yc = ((vf - cy) * zc) / fy;
            for k in 0..3 {
                o[3 * u + k] = ((xc * r[3 * k] + yc * r[3 * k + 1]) + zc * r[3 * k + 2]) + t[k];
            }
        }
    };
    if parallel {
        out.par_chunks_mut(3 * w)
            .enumerate()
            .for_each(|(v, o)| row(v, o));
    } else {
        out.chunks_mut(3 * w)
            .enumerate()
            .for_each(|(v, o)| row(v, o));
    }
    out
}

/// trays.locate pixel selection: valid & finite & inside the xy window & (optional) plane height band & tray chroma.
/// Returns the selected (x, y) in row-major order (== P[m][:, :2]).
#[allow(clippy::too_many_arguments)]
pub fn tray_select(
    p: &[f64],
    rgb: &[u8],
    valid: &[bool],
    nx: f64,
    ny: f64,
    wx: f64,
    wy: f64,
    plane: Option<(&[f64; 3], f64, f64, f64)>,
    k: usize,
    chroma_min: f64,
) -> Vec<f64> {
    let mut out = Vec::new();
    let (o1, o2) = match k {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    for i in 0..valid.len() {
        if !valid[i] {
            continue;
        }
        let q = &p[3 * i..3 * i + 3];
        if !(q[0].is_finite() && q[1].is_finite() && q[2].is_finite())
            || (q[0] - nx).abs() >= wx
            || (q[1] - ny).abs() >= wy
        {
            // coordinates finite here: same as !(.. < ..)
            continue;
        }
        if let Some((n, d, lo, hi)) = plane {
            let hh = height(q, n, d);
            if hh.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater)
                || hh.partial_cmp(&hi) != Some(std::cmp::Ordering::Less)
            {
                continue;
            }
        }
        let c = &rgb[3 * i..3 * i + 3];
        let ck = c[k] as f64 / 255.;
        let m = (c[o1] as f64 / 255.).max(c[o2] as f64 / 255.);
        if ck - m >= chroma_min {
            out.push(q[0]);
            out.push(q[1]);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
/// perception.segment mask: valid & min_h < height < max_h & low <= P <= high (all three axes).
pub fn segment_mask(
    p: &[f64],
    valid: &[bool],
    n: &[f64; 3],
    d: f64,
    min_h: f64,
    max_h: f64,
    low: &[f64; 3],
    high: &[f64; 3],
) -> Vec<bool> {
    valid
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let q = &p[3 * i..3 * i + 3];
            let hh = height(q, n, d);
            v && hh > min_h && hh < max_h && (0..3).all(|k| q[k] >= low[k] && q[k] <= high[k])
        })
        .collect()
}

/// NumPy `quantile(a, q)` (method 'linear', identical in NumPy 1.21 .. 2.x): virtual index (n-1)*q, weight
/// t = index - floor, lerp a + (b-a)*t, or b - (b-a)*(1-t) when t >= .5. `v` is reordered in place.
pub fn quantile_linear(v: &mut [f64], q: f64) -> f64 {
    let n = v.len();
    let vi = (n - 1) as f64 * q;
    let below = vi.floor();
    let i = below as usize;
    let j = (i + 1).min(n - 1);
    let t = vi - below;
    let (_, a, rest) = v.select_nth_unstable_by(i, f64::total_cmp);
    let a = *a;
    let b = if j == i {
        a
    } else {
        rest.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1. - t)
    } else {
        a + diff * t
    }
}

/// NumPy `median` of finite values: middle element, or (lo + hi) / 2 of the two middle elements.
pub fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    let m = n / 2;
    let (lo_part, mid, _) = v.select_nth_unstable_by(m, f64::total_cmp);
    let mid = *mid;
    if n % 2 == 1 {
        mid
    } else {
        let lo = lo_part.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (lo + mid) / 2.
    }
}

/// Per-component statistics of perception.segment for components with >= min_pixels pixels:
/// row = [centroid(3) = median(cloud), lower(3), upper(3), sort(extent_xy)(2), quantile(height, .98), median(rgb)/255 (3)].
/// Returns (kept component indices, rows).
#[allow(clippy::too_many_arguments)]
pub fn component_features(
    p: &[f64],
    rgb: &[u8],
    w: usize,
    pix: &[u32],
    off: &[usize],
    n: &[f64; 3],
    d: f64,
    min_pixels: usize,
) -> (Vec<usize>, Vec<[f64; 15]>) {
    let mut keep = vec![];
    let mut rows = vec![];
    let mut buf: Vec<f64> = vec![];
    for c in 0..off.len().saturating_sub(1) {
        let (s, e) = (off[c], off[c + 1]);
        if e - s < min_pixels {
            continue;
        }
        let idx = |k: usize| pix[2 * k] as usize * w + pix[2 * k + 1] as usize;
        let mut r = [0f64; 15];
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for k in s..e {
            let q = &p[3 * idx(k)..3 * idx(k) + 3];
            for a in 0..3 {
                lo[a] = lo[a].min(q[a]);
                hi[a] = hi[a].max(q[a]);
            }
        }
        for a in 0..3 {
            buf.clear();
            buf.extend((s..e).map(|k| p[3 * idx(k) + a]));
            r[a] = median(&mut buf);
            r[3 + a] = lo[a];
            r[6 + a] = hi[a];
        }
        let (e0, e1) = (hi[0] - lo[0], hi[1] - lo[1]);
        r[9] = e0.min(e1);
        r[10] = e0.max(e1);
        buf.clear();
        buf.extend((s..e).map(|k| height(&p[3 * idx(k)..3 * idx(k) + 3], n, d)));
        r[11] = quantile_linear(&mut buf, 0.98);
        for a in 0..3 {
            buf.clear();
            buf.extend((s..e).map(|k| rgb[3 * idx(k) + a] as f64));
            r[12 + a] = median(&mut buf) / 255.;
        }
        keep.push(c);
        rows.push(r);
    }
    (keep, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]
        #[test]
        fn flood_fill_partitions_mask(h in 1usize..24, w in 1usize..24, seed in 0u64..10000) {
            let mut s = seed | 1; let mut r = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64<<53) as f64 };
            let mask: Vec<bool> = (0..h*w).map(|_| r() < 0.6).collect();
            let p: Vec<f64> = (0..h*w*3).map(|_| r() * 0.05).collect();
            let (pix, off) = flood_fill(&mask, &p, h, w, 0.03);
            let mut cover = vec![0u8; h*w];
            for k in 0..pix.len()/2 { cover[pix[2*k] as usize * w + pix[2*k+1] as usize] += 1; }
            for i in 0..h*w { prop_assert_eq!(cover[i], mask[i] as u8); }
            prop_assert_eq!(*off.last().unwrap(), pix.len()/2);
        }
        #[test]
        fn ransac_best_is_first_argmax(n in 1usize..6000, m in 0usize..40, seed in 0u64..1000) {
            let mut s = seed | 1; let mut r = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64<<53) as f64 };
            let x: Vec<f64> = (0..n).map(|_| r() - 0.5).collect();
            let y: Vec<f64> = (0..n).map(|_| r() - 0.5).collect();
            let z: Vec<f64> = (0..n).map(|_| if r() < 0.6 { 0.01 * (r() - 0.5) } else { r() }).collect();
            let planes: Vec<[f64;4]> = (0..m).map(|k| if k % 3 == 0 { [0., 0., 1., 0.] } else { [0.1*(r()-0.5), 0.1*(r()-0.5), 1.0, -r()*0.5] }).collect();
            let counts = plane_inlier_counts_soa(&x, &y, &z, &planes, 0.006, false);
            let mut exp = None; let mut sc = 0;
            for (k, &c) in counts.iter().enumerate() { if c > sc { sc = c; exp = Some((k, c)); } }
            prop_assert_eq!(ransac_best(&x, &y, &z, &planes, 0.006), exp);
        }
        #[test]
        fn counts_match_scalar(n in 1usize..500, seed in 0u64..1000) {
            let mut s = seed | 1; let mut r = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64<<53) as f64 };
            let pts: Vec<f64> = (0..3*n).map(|_| r() - 0.5).collect();
            let planes: Vec<[f64;4]> = (0..5).map(|_| [r()-0.5, r()-0.5, 1.0, r()*0.1]).collect();
            let c = plane_inlier_counts(&pts, &planes, 0.05, seed % 2 == 0);
            for (k, pl) in planes.iter().enumerate() {
                let m = (0..n).filter(|&i| ((((pts[3*i]*pl[0]) + pts[3*i+1]*pl[1]) + pts[3*i+2]*pl[2]) + pl[3]).abs() < 0.05).count();
                prop_assert_eq!(c[k] as usize, m);
            }
        }
    }
}
