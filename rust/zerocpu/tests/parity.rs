//! Parity with the Python reference implementation.
//!
//! Every constant here was produced by `ultragraph/zerocpu/` (and
//! `ultragraph/pack.py`) on the Python side.  If a kernel changes semantics,
//! one of these tests fails instead of the two runtimes drifting apart.

use zerocpu::{
    dot_i8, dot_i8_scalar, dot_ternary, from_int, from_ratio, fx_div, fx_dot, fx_mul, fx_sqrt,
    pack5, round_div, saturate_i8, to_int, unpack5, ONE,
};

#[test]
fn pack_matches_python_goldens() {
    assert_eq!(pack5(&[1, -1, 0, 1, 0]), vec![146u8]);
    assert_eq!(pack5(&[-1, 0, 1]), vec![21u8]);
    assert_eq!(pack5(&[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), vec![242u8, 121u8]);
}

#[test]
fn pack_unpack_roundtrip_and_truncate() {
    let q: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 0, 1, -1, 1, 1];
    let packed = pack5(&q);
    assert_eq!(packed.len(), q.len().div_ceil(5));
    assert_eq!(unpack5(&packed, q.len()), q);
    assert_eq!(unpack5(&packed, 3), q[..3].to_vec());
    assert_eq!(unpack5(&[], 0), Vec::<i8>::new());
}

#[test]
fn fixed_matches_python_goldens() {
    assert_eq!(from_ratio(1, 10), 6554);
    assert_eq!(fx_mul(6554, from_int(10)), 65540);
    assert_eq!(fx_div(ONE, from_int(3)), 21845);
    assert_eq!(from_ratio(1, 3), 21845);
    assert_eq!(fx_mul(from_ratio(3, 2), from_int(-2)), -196608);
    assert_eq!(to_int(from_ratio(5, 2)), 3);
    assert_eq!(to_int(from_ratio(-5, 2)), -3);
    assert_eq!(fx_sqrt(from_ratio(9, 4)), 98304);
}

#[test]
fn fixed_dot_matches_python_golden() {
    let a = [from_int(1), from_int(2), from_int(3)];
    let b = [from_int(4), from_int(-1), from_ratio(1, 2)];
    assert_eq!(fx_dot(&a, &b), 229376); // 3.5
}

#[test]
fn fixed_rounding_is_symmetric_about_zero() {
    let m = fx_mul(from_ratio(3, 2), from_int(2));
    assert_eq!(m, fx_mul(from_ratio(-3, 2), from_int(-2)));
    assert_eq!(
        fx_div(from_int(-1), from_int(3)),
        -fx_div(from_int(1), from_int(3))
    );
}

#[test]
#[should_panic(expected = "division by zero")]
fn fixed_div_by_zero_panics() {
    let _ = round_div(1, 0);
}

#[test]
fn saturate_clamps_to_int8() {
    assert_eq!(saturate_i8(200), 127);
    assert_eq!(saturate_i8(-300), -128);
    assert_eq!(saturate_i8(7), 7);
}

#[test]
fn dot_matches_python_golden() {
    let xs = [3i8, -7, 11, 0, -2, 5, 9, -4, 1, 6];
    let ws = [1i8, -1, 0, 1, 0, 1, 1, -1, 0, 1];
    assert_eq!(dot_i8(&xs, &ws), 34);
    assert_eq!(dot_ternary(&ws, &xs), 34);
}

fn lcg(state: &mut u64) -> i8 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as i8
}

#[test]
fn accelerated_dot_equals_scalar_over_random_vectors() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for n in [0usize, 1, 7, 15, 16, 17, 31, 32, 63, 64, 100, 255] {
        let a: Vec<i8> = (0..n).map(|_| lcg(&mut seed)).collect();
        let b: Vec<i8> = (0..n).map(|_| lcg(&mut seed)).collect();
        assert_eq!(dot_i8(&a, &b), dot_i8_scalar(&a, &b), "n={n}");
    }
}

#[test]
fn ternary_pack_roundtrip_after_neon_dot() {
    // The deployed path is: pack weights -> store -> unpack -> dot.
    let w: Vec<i8> = (0..37).map(|i| [1i8, -1, 0][i % 3]).collect();
    let x: Vec<i8> = (0..37).map(|i| (i as i8) - 18).collect();
    let restored = unpack5(&pack5(&w), w.len());
    assert_eq!(restored, w);
    assert_eq!(dot_i8(&restored, &x), dot_i8_scalar(&w, &x));
}
