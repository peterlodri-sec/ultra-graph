//! aarch64 NEON dot product: `sum(a[i] * b[i])` over int8 lanes.
//!
//! Products widen to i16, pairwise-accumulate into i32, and only the tail is
//! scalar.  Every step is integer, so the result is identical to the scalar
//! kernel -- `tests/parity.rs` checks that across many lengths and values.
//!
//! Note: NEON is baseline on aarch64, so no runtime feature detection is needed.

use core::arch::aarch64::*;

pub fn dot_i8(a: &[i8], b: &[i8]) -> i64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let n = a.len();
    let mut i = 0usize;
    let wide: i32;

    // SAFETY: bounds checked by the `i + 16 <= n` loop guard; pointers stay in
    // the same allocation for 16 readable bytes.  NEON is always available on
    // aarch64.
    unsafe {
        let mut acc = vdupq_n_s32(0);
        while i + 16 <= n {
            let va = vld1q_s8(a.as_ptr().add(i));
            let vb = vld1q_s8(b.as_ptr().add(i));
            let lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
            let hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
            acc = vpadalq_s16(acc, lo);
            acc = vpadalq_s16(acc, hi);
            i += 16;
        }
        wide = vaddvq_s32(acc);
    }

    let mut tail = 0i64;
    while i < n {
        tail += a[i] as i64 * b[i] as i64;
        i += 1;
    }
    wide as i64 + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        assert_eq!(dot_i8(&[], &[]), 0);
    }

    #[test]
    fn matches_scalar_for_short_and_long() {
        for n in [1usize, 3, 15, 16, 17, 33, 64, 65, 257] {
            let a: Vec<i8> = (0..n).map(|k| ((k * 37) % 255) as i8).collect();
            let b: Vec<i8> = (0..n).map(|k| ((k * 91 + 7) % 255) as i8).collect();
            let want = crate::ternary::dot_i8_scalar(&a, &b);
            assert_eq!(dot_i8(&a, &b), want, "n={n}");
        }
    }
}
