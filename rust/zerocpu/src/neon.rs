//! aarch64 NEON dot product: `sum(a[i] * b[i])` over int8 lanes.
//!
//! Products widen to i16, pairwise-accumulate into i32, and only the tail is
//! scalar.  Every step is integer, so the result is identical to the scalar
//! kernel -- `tests/parity.rs` checks that across many lengths and values.
//!
//! Note: NEON is baseline on aarch64, so no runtime feature detection is needed.

use core::arch::aarch64::*;

/// The ARMv8.2 DotProd kernel: four int8 multiply-accumulates per instruction.
///
/// `vdotq_s32` is the whole inner loop -- 16 bytes in, one `sdot`, no widening
/// and no pairwise adds. It needs the `dotprod` target feature, so it is only
/// called after a runtime check, and only on aarch64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_i8_dotprod(a: &[i8], b: &[i8]) -> i64 {
    let n = a.len();
    let mut i = 0usize;
    let mut acc = vdupq_n_s32(0);
    while i + 16 <= n {
        let va = vld1q_s8(a.as_ptr().add(i));
        let vb = vld1q_s8(b.as_ptr().add(i));
        acc = vdotq_s32(acc, va, vb);
        i += 16;
    }
    let mut tail = 0i64;
    while i < n {
        tail += i64::from(a[i]) * i64::from(b[i]);
        i += 1;
    }
    vaddvq_s32(acc) as i64 + tail
}

/// True when this CPU implements ARMv8.2 DotProd (`sdot`).
///
/// Apple M-series and every Cortex-A55 or newer do; the check is cached by
/// `std`, so it costs a relaxed atomic load rather than a CPUID walk.
#[cfg(target_arch = "aarch64")]
pub fn has_dotprod() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
}

/// The dispatcher used by `ternary::dot_i8`.
///
/// Order matters and the middle step is the interesting one: DotProd if the CPU
/// has it, otherwise the plain scalar loop. The hand-written `vmull` kernel
/// below is *kept but not used* by the dispatcher, because LLVM auto-vectorises
/// the scalar loop into something faster than it (see `examples/dot_i8_bench.rs`:
/// 341 ns vs 449 ns over 4096 lanes). Keeping it preserves the aarch64 path for
/// anyone who wants the intrinsics and pins their parity with the scalar answer.
#[cfg(target_arch = "aarch64")]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    if has_dotprod() {
        // SAFETY: `has_dotprod` just confirmed the feature.
        return unsafe { dot_i8_dotprod(a, b) };
    }
    crate::ternary::dot_i8_scalar(a, b)
}

/// The original NEON intrinsics: `vmull_s8` widened, then pairwise-added.
///
/// Not on the dispatch path any more, but still compiled and still tested --
/// it is the reference for what "NEON by hand" costs on this machine.
pub fn dot_i8_neon(a: &[i8], b: &[i8]) -> i64 {
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

    /// The DotProd kernel must agree with the scalar loop for every length,
    /// including the ones that leave a tail shorter than a vector.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dotprod_matches_scalar_for_every_length() {
        for n in 0..80usize {
            let a: Vec<i8> = (0..n).map(|k| ((k * 37) % 255) as i8).collect();
            let b: Vec<i8> = (0..n).map(|k| ((k * 91 + 7) % 255) as i8).collect();
            assert_eq!(
                dot_i8(&a, &b),
                crate::ternary::dot_i8_scalar(&a, &b),
                "n={n}"
            );
        }
    }

    /// And it must agree with the hand-written NEON kernel too, so the two
    /// aarch64 paths cannot drift apart silently.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dotprod_matches_the_neon_kernel() {
        for n in [0usize, 1, 3, 15, 16, 17, 33, 64, 257] {
            let a: Vec<i8> = (0..n).map(|k| ((k * 37) % 255) as i8).collect();
            let b: Vec<i8> = (0..n).map(|k| ((k * 91 + 7) % 255) as i8).collect();
            let neon = dot_i8_neon(&a, &b);
            let dotprod = if has_dotprod() {
                unsafe { dot_i8_dotprod(&a, &b) }
            } else {
                neon
            };
            assert_eq!(dotprod, neon, "n={n}");
        }
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
