//! Perception kernels: RANSAC inlier counting for `perception.table_plane` and the 4-connected depth-continuity
//! flood fill of `perception.segment`. Candidate generation (the NumPy RNG stream) and all statistics stay in
//! Python, so the random sequence and the final plane/feature arithmetic are untouched.
use rayon::prelude::*;

/// For each candidate plane (n, offset) count points with |p.n + offset| < tol. Points: N x 3 row-major.
/// Returns the counts (same order as candidates). Parallel over candidates.
pub fn plane_inlier_counts(points: &[f64], planes: &[[f64; 4]], tol: f64) -> Vec<u64> {
    let npts = points.len() / 3;
    planes
        .par_iter()
        .map(|pl| {
            let (a, b, c, d) = (pl[0], pl[1], pl[2], pl[3]);
            let mut cnt = 0u64;
            for i in 0..npts {
                let p = &points[3 * i..3 * i + 3];
                // (p @ n) + offset, 3-term dot evaluated left to right
                let h = ((p[0] * a + p[1] * b) + p[2] * c) + d;
                cnt += (h.abs() < tol) as u64;
            }
            cnt
        })
        .collect()
}

/// Exact port of the DFS in perception.segment: seeds in row-major (np.nonzero) order, explicit stack, neighbour
/// order up, down, left, right; neighbour accepted if mask && !seen && |p_n - p| < max_jump.
/// Returns (pixels as flat [y0,x0,y1,x1,...] in visit order, component start offsets incl. final end).
pub fn flood_fill(mask: &[bool], p: &[f64], h: usize, w: usize, max_jump: f64) -> (Vec<u32>, Vec<usize>) {
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
                let nb: [(isize, isize); 4] = [(y as isize - 1, x as isize), (y as isize + 1, x as isize), (y as isize, x as isize - 1), (y as isize, x as isize + 1)];
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
        fn counts_match_scalar(n in 1usize..500, seed in 0u64..1000) {
            let mut s = seed | 1; let mut r = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64<<53) as f64 };
            let pts: Vec<f64> = (0..3*n).map(|_| r() - 0.5).collect();
            let planes: Vec<[f64;4]> = (0..5).map(|_| [r()-0.5, r()-0.5, 1.0, r()*0.1]).collect();
            let c = plane_inlier_counts(&pts, &planes, 0.05);
            for (k, pl) in planes.iter().enumerate() {
                let m = (0..n).filter(|&i| ((((pts[3*i]*pl[0]) + pts[3*i+1]*pl[1]) + pts[3*i+2]*pl[2]) + pl[3]).abs() < 0.05).count();
                prop_assert_eq!(c[k] as usize, m);
            }
        }
    }
}
