//! NumPy-compatible random stream for `np.random.default_rng(seed).choice(n, 3, replace=False)`.
//!
//! The PCG64 state (128-bit state and increment) is taken from NumPy itself (`PCG64(seed).state`), so NumPy's
//! SeedSequence hashing is not re-implemented. Re-implemented (NumPy `pcg64.h`, `distributions.c`,
//! `_generator.pyx`, stable since NumPy 1.17, verified bit-for-bit on 1.21.5 / 1.26.4 / 2.5 by
//! test_accel_equivalence.py):
//!   * PCG64 XSL-RR 128/64 step-then-output, and `next_uint32` (low half first, high half buffered);
//!   * `random_bounded_uint64(0, rng)` for rng <= 0xFFFFFFFF: Lemire's nearly-divisionless bounded 32-bit draw;
//!   * `Generator.choice(n, k, replace=False, shuffle=True)` for the Floyd branch (taken whenever
//!     not (n > 10000 and k > n // 50)), followed by the Fisher-Yates `_shuffle_int` of the k picks.
const MULT: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

#[derive(Clone, Debug)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has32: bool,
    buf32: u32,
}

impl Pcg64 {
    /// From NumPy's `bit_generator.state['state']` (state, inc) and the 32-bit buffer flags.
    pub fn from_numpy(state: u128, inc: u128, has32: bool, buf32: u32) -> Pcg64 {
        Pcg64 {
            state,
            inc,
            has32,
            buf32,
        }
    }
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }
    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        if self.has32 {
            self.has32 = false;
            return self.buf32;
        }
        let n = self.next_u64();
        self.has32 = true;
        self.buf32 = (n >> 32) as u32;
        n as u32
    }
    /// NumPy `random_bounded_uint64(bitgen, 0, rng, 0, use_masked=0)`: uniform integer in [0, rng].
    #[inline]
    pub fn bounded(&mut self, rng: u64) -> u64 {
        if rng == 0 {
            return 0;
        }
        if rng <= 0xFFFF_FFFF {
            if rng == 0xFFFF_FFFF {
                return self.next_u32() as u64;
            }
            let excl = (rng as u32).wrapping_add(1);
            let mut m = (self.next_u32() as u64) * (excl as u64);
            let mut left = m as u32;
            if left < excl {
                let threshold = (u32::MAX - rng as u32) % excl;
                while left < threshold {
                    m = (self.next_u32() as u64) * (excl as u64);
                    left = m as u32;
                }
            }
            return m >> 32;
        }
        // 64-bit Lemire (never reached for point clouds < 4e9 points, kept for completeness)
        if rng == u64::MAX {
            return self.next_u64();
        }
        let excl = rng + 1;
        let mut m = (self.next_u64() as u128) * (excl as u128);
        let mut left = m as u64;
        if left < excl {
            let threshold = (u64::MAX - rng) % excl;
            while left < threshold {
                m = (self.next_u64() as u128) * (excl as u128);
                left = m as u64;
            }
        }
        (m >> 64) as u64
    }
    /// `Generator.choice(n, k, replace=False)` (shuffle=True). None where NumPy would take another branch or raise.
    pub fn choice_floyd(&mut self, n: u64, k: usize, out: &mut [u64]) -> Option<()> {
        if k as u64 > n || out.len() < k || (n > 10000 && k as u64 > n / 50) {
            return None;
        }
        let set_size = (1.2 * k as f64) as u64;
        let mask = gen_mask(set_size);
        let mut hash = vec![u64::MAX; (mask + 1) as usize];
        for j in (n - k as u64)..n {
            let val = self.bounded(j);
            let mut loc = val & mask;
            while hash[loc as usize] != u64::MAX && hash[loc as usize] != val {
                loc = (loc + 1) & mask;
            }
            let at = (j - (n - k as u64)) as usize;
            if hash[loc as usize] == u64::MAX {
                hash[loc as usize] = val;
                out[at] = val;
            } else {
                let mut loc = j & mask;
                while hash[loc as usize] != u64::MAX {
                    loc = (loc + 1) & mask;
                }
                hash[loc as usize] = j;
                out[at] = j;
            }
        }
        // _shuffle_int(k, 1, idx): for i in reversed(range(1, k)): j = bounded(i); swap
        for i in (1..k).rev() {
            let j = self.bounded(i as u64) as usize;
            out.swap(i, j);
        }
        Some(())
    }
}

/// NumPy `_gen_mask`: smallest all-ones mask >= max.
fn gen_mask(max: u64) -> u64 {
    let mut m = max;
    m |= m >> 1;
    m |= m >> 2;
    m |= m >> 4;
    m |= m >> 8;
    m |= m >> 16;
    m |= m >> 32;
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn floyd_distinct_in_range(n in 3u64..200000, s in any::<u64>(), i in any::<u64>()) {
            let mut g = Pcg64::from_numpy(((s as u128) << 64) | i as u128, ((i as u128) << 1) | 1, false, 0);
            let mut o = [0u64; 3];
            g.choice_floyd(n, 3, &mut o).unwrap();
            prop_assert!(o.iter().all(|&v| v < n));
            prop_assert!(o[0] != o[1] && o[1] != o[2] && o[0] != o[2]);
        }
        #[test]
        fn bounded_in_range(r in 0u64..u64::MAX, s in any::<u64>()) {
            let mut g = Pcg64::from_numpy(s as u128, 0x2b, false, 0);
            prop_assert!(g.bounded(r) <= r);
        }
    }
}
