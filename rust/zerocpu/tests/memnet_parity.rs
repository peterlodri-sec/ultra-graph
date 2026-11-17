//! Parity: `zerocpu::memnet` against the golden vectors from the Python reference.

use zerocpu::memnet::MemNet;

const VECTORS: &str = include_str!("fixtures/memnet_vectors.txt");

#[test]
fn matches_python_bit_for_bit() {
    let mut net = MemNet::new();
    let mut section = "";
    let mut expected_hex: Option<&str> = None;
    let mut merge_rows: Vec<&str> = Vec::new();
    for raw in VECTORS.lines().map(str::trim) {
        if raw.is_empty() || raw.starts_with('#') {
            continue;
        }
        if raw.starts_with('[') && raw.ends_with(']') {
            section = raw;
            continue;
        }
        match section {
            "[put]" => {
                let (key, value) = raw.rsplit_once(' ').expect("key value");
                net.put(key, value.parse().expect("i32 value")).unwrap();
            }
            "[hex]" => expected_hex = Some(raw),
            "[merge]" => merge_rows.push(raw),
            other => panic!("unknown section {other:?}"),
        }
    }
    assert_eq!(net.to_hex(), expected_hex.expect("hex row"));
    assert_eq!(merge_rows.len(), 3, "merge needs a, b and the merged row");

    let a = MemNet::from_hex(merge_rows[0]).unwrap();
    let b = MemNet::from_hex(merge_rows[1]).unwrap();
    let mut ab = MemNet::from_hex(merge_rows[0]).unwrap();
    assert_eq!(ab.merge(&b), 2, "both right-hand keys are new");
    assert_eq!(ab.to_hex(), merge_rows[2]);

    let mut ba = MemNet::from_hex(merge_rows[1]).unwrap();
    ba.merge(&a);
    assert_eq!(ba.to_hex(), merge_rows[2], "merge must be order independent");
}
