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

/// Eight ternary lanes per machine word: the SWAR packing used by
/// [`ternary_dot_packed`].  Each lane is one two's-complement byte.
pub const PACKED_LANES: usize = 8;
/// Lowest bit of every byte in a packed word.
pub const LANE_ONE: i64 = 0x0101_0101_0101_0101;

/// `sum(a[i] * b[i])` over eight packed ternary lanes, without multiplying.
///
/// `+1 * +1` and `-1 * -1` are `+1`; a mismatched sign is `-1`; a zero lane is
/// zero.  So the whole dot product is two bit-AND masks per outcome and two
/// popcounts -- no multiply, no widening, no tables, and identical on every
/// architecture (this is the portable twin of `neon::dot_i8`).
///
/// The contract is ternary lanes: each byte must be `0x00`, `0x01` or `0xFF`.
/// Anything else is silently misread as one of those, so validate at the edge;
/// debug builds assert it here.
pub fn ternary_dot_packed(a: i64, b: i64) -> i64 {
    debug_assert!(is_ternary_packed(a), "lane out of ternary range: {a:#018x}");
    debug_assert!(is_ternary_packed(b), "lane out of ternary range: {b:#018x}");
    // `-1` is 0xFF, whose lowest bit is set too: clear the negative lanes
    // before reading a lane as +1, or every -1 counts as a +1 as well.
    let a_neg = (a >> 7) & LANE_ONE;
    let b_neg = (b >> 7) & LANE_ONE;
    let a_one = a & LANE_ONE & !a_neg;
    let b_one = b & LANE_ONE & !b_neg;
    let positive = (a_one & b_one) | (a_neg & b_neg);
    let negative = (a_one & b_neg) | (a_neg & b_one);
    i64::from(positive.count_ones()) - i64::from(negative.count_ones())
}

/// True when every byte of `word` is `0x00`, `0x01` or `0xFF`.
pub fn is_ternary_packed(word: i64) -> bool {
    for lane in 0..PACKED_LANES {
        let byte = ((word >> (8 * lane)) & 0xFF) as u8;
        if byte != 0x00 && byte != 0x01 && byte != 0xFF {
            return false;
        }
    }
    true
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
