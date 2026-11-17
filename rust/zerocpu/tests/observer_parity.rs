//! Parity: `zerocpu::observer` against the golden vectors from Python.

use zerocpu::observer::{
    decay_default, decay_pade, pade_exp_neg, Mahakala, ObserverFrame, Witness,
};

const VECTORS: &str = include_str!("fixtures/observer_vectors.txt");

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
            "[decay]" => {
                let n: Vec<i64> = raw.split_whitespace().map(|v| v.parse().unwrap()).collect();
                assert_eq!(n.len(), 3, "{raw}");
                assert_eq!(decay_default(n[0], n[1]), n[2], "{raw}");
            }
            "[frames]" => {
                let n: Vec<i64> = raw.split_whitespace().map(|v| v.parse().unwrap()).collect();
                assert_eq!(n.len(), 4, "{raw}");
                let frame = ObserverFrame::new("f", n[0], n[1]).unwrap();
                assert_eq!(frame.local_time(n[2]), n[3], "{raw}");
            }
            "[mahakala]" => {
                let (text, matched) = raw.split_once('|').expect("text | matched");
                let guard = Mahakala::default();
                let ours = guard.raise_guard(text.trim());
                let expected: Vec<String> = matched
                    .split(',')
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(str::to_string)
                    .collect();
                assert_eq!(ours, expected, "{text:?}");
                assert_eq!(guard.protects(text.trim()), !expected.is_empty(), "{text:?}");
            }
            "[witness]" => {
                let n: Vec<i64> = raw.split_whitespace().map(|v| v.parse().unwrap()).collect();
                assert_eq!(n.len(), 5, "{raw}");
                let mut memory = Witness::default();
                memory.remember("k", n[2], n[0], n[1] != 0).unwrap();
                assert_eq!(memory.variance("k", n[3]), Some(n[4]), "{raw}");
                assert_eq!(memory.anchors().len(), usize::from(n[1] != 0), "{raw}");
            }
            "[pade]" => {
                let n: Vec<i64> = raw.split_whitespace().map(|v| v.parse().unwrap()).collect();
                assert_eq!(n.len(), 2, "{raw}");
                assert_eq!(pade_exp_neg(n[0]), n[1], "{raw}");
            }
            "[pade_decay]" => {
                let n: Vec<i64> = raw.split_whitespace().map(|v| v.parse().unwrap()).collect();
                assert_eq!(n.len(), 4, "{raw}");
                assert_eq!(decay_pade(n[0], n[1], n[2]).unwrap(), n[3], "{raw}");
            }
            other => panic!("unknown section {other:?}"),
        }
        checked += 1;
    }
    assert!(checked >= 30, "only checked {checked} rows");
}
