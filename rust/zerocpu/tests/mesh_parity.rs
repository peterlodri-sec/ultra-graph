//! Parity: `zerocpu::mesh` against the golden frames from the Python reference.

use zerocpu::mesh::{merge_peers, pack_announce, parse_announce, Peer};

const VECTORS: &str = include_str!("fixtures/mesh_vectors.txt");

fn unhex(text: &str) -> Vec<u8> {
    assert!(text.len().is_multiple_of(2), "odd hex length: {text}");
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks(2) {
        let hi = (pair[0] as char).to_digit(16).expect("hex digit") as u8;
        let lo = (pair[1] as char).to_digit(16).expect("hex digit") as u8;
        out.push((hi << 4) | lo);
    }
    out
}

#[test]
fn frames_match_python_bit_for_bit() {
    let mut peers: Vec<Peer> = Vec::new();
    let mut checked = 0usize;
    for raw in VECTORS.lines().filter(|line| !line.starts_with('#')) {
        if raw.is_empty() {
            continue;
        }
        let fields: Vec<&str> = raw.split('\t').collect();
        assert_eq!(fields.len(), 4, "expected 4 tab-separated fields: {raw:?}");
        let name = fields[0];
        let port: u16 = fields[1].parse().unwrap();
        let seq: u32 = fields[2].parse().unwrap();
        let expected = unhex(fields[3]);

        let frame = pack_announce(name, port, seq).unwrap();
        assert_eq!(frame, expected, "frame for {name}");

        let parsed = parse_announce(&expected).expect("frame must parse");
        assert_eq!(parsed, Peer::new(name, port, seq));

        for cut in 0..expected.len() {
            assert!(parse_announce(&expected[..cut]).is_none());
        }

        peers = merge_peers(&peers, &parsed);
        checked += 1;
    }
    assert!(checked >= 6, "only checked {checked} rows");
    let names: Vec<&str> = peers.iter().map(|peer| peer.name.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted, "merged peers must be sorted by name");
    assert_eq!(peers.len(), 6, "every distinct name must survive the merge");
}
