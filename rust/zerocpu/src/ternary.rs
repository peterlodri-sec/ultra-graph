//! 5-trit-per-byte packing and ternary dot products.
//!
//! `pack5`/`unpack5` are byte-identical to `ultragraph/pack.py`
//! (`3**5 = 243 < 256`, so five `{-1,0,+1}` values fit one byte: 1.6 bits each).

pub const TRITS_PER_BYTE: usize = 5;
const PLACE: [u16; TRITS_PER_BYTE] = [1, 3, 9, 27, 81];

/// Pack ternary values into bytes, 5 per byte, zero-padded (digit value 1).
pub fn pack5(q: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(q.len().div_ceil(TRITS_PER_BYTE));
    for chunk in q.chunks(TRITS_PER_BYTE) {
        let mut byte = 0u16;
        for (i, &w) in chunk.iter().enumerate() {
            assert!((-1..=1).contains(&w), "pack5 expects ternary, got {w}");
            byte += ((w as i16 + 1) as u16) * PLACE[i];
        }
        out.push(byte as u8);
    }
    out
}

/// Inverse of [`pack5`]; returns the first `n` values.
pub fn unpack5(packed: &[u8], n: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(n);
    for &byte in packed {
        let mut b = byte as u16;
        for _ in 0..TRITS_PER_BYTE {
            out.push((b % 3) as i8 - 1);
            b /= 3;
        }
    }
    out.truncate(n);
    out
}

/// Portable dot product of two int8 vectors into an i64 accumulator.
pub fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as i64) * (*y as i64))
        .sum()
}

/// Dot product of ternary weights against int8 activations.
pub fn dot_ternary(w: &[i8], x: &[i8]) -> i64 {
    assert_eq!(w.len(), x.len(), "length mismatch");
    debug_assert!(
        w.iter().all(|v| (-1..=1).contains(v)),
        "dot_ternary expects ternary weights"
    );
    dot_i8(w, x)
}

/// Accelerated dot product: NEON on aarch64, scalar elsewhere.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i64 {
    #[cfg(target_arch = "aarch64")]
    {
        crate::neon::dot_i8(a, b)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_i8_scalar(a, b)
    }
}
