//! Parity: `zerocpu::memnet` against the golden vectors from the Python reference.

use zerocpu::observer::{ObserverError, ObserverFrame};
use zerocpu::memnet::{MemNet, RingMemNet};

const VECTORS: &str = include_str!("fixtures/memnet_vectors.txt");

#[test]
fn matches_python_bit_for_bit() {
    let mut net = MemNet::new();
    let mut ring = RingMemNet::new();
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
            "[ring]" => {
                let n: Vec<i64> = raw.split_whitespace().map(|v| v.parse().unwrap()).collect();
                assert_eq!(n.len(), 3, "{raw}");
                assert!(ring.record(n[0], n[1], n[2] != 0).is_some(), "{raw}");
            }
            "[ring_expect]" => {
                let fields: Vec<&str> = raw.split_whitespace().collect();
                if fields.first() == Some(&"len") {
                    assert_eq!(fields.len(), 2, "{raw}");
                    assert_eq!(ring.len() as i64, fields[1].parse::<i64>().unwrap(), "{raw}");
                    continue;
                }
                assert_eq!(fields.len(), 5, "{raw}");
                let expected: i64 = fields[4].parse().unwrap();
                match fields[0] {
                    "variance" | "consensus" => {
                        let now: i64 = fields[1].parse().unwrap();
                        let half: i64 = fields[2].parse().unwrap();
                        let rate: i64 = fields[3].parse().unwrap();
                        let ours = if fields[0] == "variance" {
                            ring.variance(now, half, rate).unwrap()
                        } else {
                            i64::from(ring.consensus(now, half, rate).unwrap())
                        };
                        assert_eq!(ours, expected, "{raw}");
                    }
                    other => panic!("unknown ring expectation {other:?}"),
                }
            }
            "[world]" => {
                let f: Vec<&str> = raw.split_whitespace().collect();
                assert_eq!(f.len(), 7, "world row needs 7 fields: {raw}");
                let now: i64 = f[1].parse().unwrap();
                let half: i64 = f[2].parse().unwrap();
                let expected_world: i64 = f[3].parse().unwrap();
                let expected_trit: i8 = f[4].parse().unwrap();
                let rate: i64 = f[5].parse().unwrap();
                let expected_local: i64 = f[6].parse().unwrap();

                assert_eq!(ring.variance_world(now, half).unwrap(), expected_world, "{raw}");
                assert_eq!(ring.consensus_world(now, half).unwrap(), expected_trit, "{raw}");
                let frame = ObserverFrame::new("fixture", 0, rate).unwrap();
            assert_eq!(ring.local_variance(&frame, now, half).unwrap(), expected_local, "{raw}");
        }
            "[world_of]" => {
                let f: Vec<&str> = raw.split_whitespace().collect();
                assert_eq!(f.len(), 4, "world_of row needs 4 fields: {raw}");
                let origin: i64 = f[0].parse().unwrap();
                let local: i64 = f[1].parse().unwrap();
                let rate: i64 = f[2].parse().unwrap();
                let frame = ObserverFrame::new("fixture", origin, rate).unwrap();
                if f[3] == "ERR" {
                    assert_eq!(frame.world_of(local), Err(ObserverError::FrozenClock), "{raw}");
                } else {
                    let expected: i64 = f[3].parse().unwrap();
                    assert_eq!(frame.world_of(local).unwrap(), expected, "{raw}");
                    // round trip: world_of(local_time(t)) == t
                assert_eq!(frame.world_of(frame.local_time(expected)).unwrap(), expected, "{raw}");
            }
        }
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

    assert_eq!(ring.len(), 3, "ring fixture must record three slots");
    let mut ba = MemNet::from_hex(merge_rows[1]).unwrap();
    ba.merge(&a);
    assert_eq!(ba.to_hex(), merge_rows[2], "merge must be order independent");
}
