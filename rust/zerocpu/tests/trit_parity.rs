//! Parity: `zerocpu::trit` against the golden vectors from the Python reference.

use zerocpu::trit::{
    bitplanes, delta_for_level, from_planes, plane_parity, quantize, tkand, tknot, tnand, tkor,
    trit_of, MAGNITUDE_MAX,
};

const VECTORS: &str = include_str!("fixtures/trit_vectors.txt");

#[test]
fn matches_python_bit_for_bit() {
    let mut section = "";
    let mut checked = 0usize;
    for raw in VECTORS.lines().map(str::trim) {
        if raw.is_empty() || raw.starts_with('#') {
            continue;
        }
        if raw.starts_with('[') && raw.ends_with(']') {
            section = raw;
            continue;
        }
        match section {
            "[levels]" => {
                let (level, delta) = raw.split_once(' ').expect("level delta");
                let level: usize = level.parse().expect("level");
                let delta: i64 = delta.trim().parse().expect("delta");
                assert_eq!(delta_for_level(level).unwrap(), delta, "{raw}");
            }
            "[quantize]" => {
                let nums: Vec<i64> = raw.split_whitespace().map(|v| v.parse().unwrap()).collect();
                assert_eq!(nums.len(), 3, "{raw}");
                assert_eq!(
                    i64::from(quantize(nums[0], nums[1]).unwrap()),
                    nums[2],
                    "{raw}"
                );
            }
            "[planes]" => {
                let parts: Vec<&str> = raw.split('|').map(str::trim).collect();
                assert_eq!(parts.len(), 3, "{raw}");
                let values: Vec<i64> = parts[0]
                    .split(',')
                    .map(|v| v.trim().parse().unwrap())
                    .collect();
                let planes: Vec<u16> = parts[1]
                    .split(',')
                    .map(|v| v.trim().parse().unwrap())
                    .collect();
                let parity: u16 = parts[2].parse().unwrap();
                let ours = bitplanes(&values).unwrap();
                assert_eq!(ours, planes, "planes for {raw}");
                assert_eq!(plane_parity(&ours), parity, "parity for {raw}");
                let expected: Vec<i64> = values
                    .iter()
                    .map(|value| {
                        let magnitude = value.unsigned_abs().min(MAGNITUDE_MAX) as i64;
                        if *value < 0 {
                            -magnitude
                        } else {
                            magnitude
                        }
                    })
                    .collect();
                let back = from_planes(&planes).unwrap();
                assert_eq!(&back[..values.len()], &expected[..], "round trip for {raw}");
            }
            "[kleene]" => {
                let n: Vec<i64> = raw.split_whitespace().map(|v| v.parse().unwrap()).collect();
                assert_eq!(n.len(), 6, "{raw}");
                assert_eq!(i64::from(tnand(n[0], n[1]).unwrap()), n[2], "{raw}");
                assert_eq!(i64::from(tkand(n[0], n[1]).unwrap()), n[3], "{raw}");
                assert_eq!(i64::from(tkor(n[0], n[1]).unwrap()), n[4], "{raw}");
                assert_eq!(i64::from(tknot(n[0]).unwrap()), n[5], "{raw}");
            }
            other => panic!("unknown section {other:?}"),
        }
        checked += 1;
    }
    assert_eq!(trit_of(-7), -1);
    assert_eq!(trit_of(9), 1);
    assert!(checked >= 35, "only checked {checked} rows");
}
