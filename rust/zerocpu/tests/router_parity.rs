//! Parity: `zerocpu::router` against the golden vectors from the Python reference.

use zerocpu::router::{features, route, route_name};

const VECTORS: &str = include_str!("fixtures/router_vectors.txt");

#[test]
fn matches_python_bit_for_bit() {
    let mut checked = 0usize;
    for raw in VECTORS.lines().filter(|line| !line.starts_with('#')) {
        if raw.is_empty() {
            continue;
        }
        let fields: Vec<&str> = raw.split('\t').collect();
        assert_eq!(fields.len(), 9, "expected 9 tab-separated fields: {raw:?}");
        let prompt = fields[8];
        let feats = features(prompt);
        assert_eq!(feats.chars, fields[0].parse().unwrap(), "{prompt:?}");
        assert_eq!(feats.words, fields[1].parse().unwrap(), "{prompt:?}");
        assert_eq!(feats.lines, fields[2].parse().unwrap(), "{prompt:?}");
        assert_eq!(feats.digits, fields[3].parse().unwrap(), "{prompt:?}");
        assert_eq!(feats.plugins, fields[4].parse().unwrap(), "{prompt:?}");
        assert_eq!(feats.marks, fields[5].parse().unwrap(), "{prompt:?}");
        assert_eq!(feats.heat, fields[6].parse().unwrap(), "{prompt:?}");
        let expected: u8 = fields[7].parse().unwrap();
        assert_eq!(route(&feats), expected, "{prompt:?}");
        assert!(route_name(expected).is_ok());
        checked += 1;
    }
    assert!(checked >= 12, "only checked {checked} rows");
}
