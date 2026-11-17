//! The LAN peer announce frame: twelve fixed header bytes plus a short name.
//!
//! ```text
//! "UGRD" | version u8 | port u16 | seq u32 | name_len u8 | name (utf-8)
//! ```
//!
//! Little-endian, fixed width, so the frame is byte-identical to the one
//! `ultragraph.zerocpu.mesh` produces.  [`parse_announce`] never panics and
//! returns `None` for anything that is not exactly one well-formed frame:
//! mystery datagrams are a normal condition on a LAN.

use std::collections::BTreeMap;
use std::fmt;

/// Frame magic.
pub const MAGIC: [u8; 4] = *b"UGRD";
/// Wire format version.
pub const PROTO_VERSION: u8 = 1;
/// Default announce port.
pub const PROTO_PORT: u16 = 8423;
/// Longest accepted name, in UTF-8 bytes.
pub const MAX_NAME: usize = 32;
/// The fixed part of the frame, excluding the name.
pub const HEADER_FIXED: usize = 12;

/// A node heard from.  `source` is runtime-only and never on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Peer {
    pub name: String,
    pub port: u16,
    pub seq: u32,
    pub source: String,
}

impl Peer {
    /// A peer with no source address yet.
    pub fn new(name: &str, port: u16, seq: u32) -> Self {
        Peer {
            name: name.to_string(),
            port,
            seq,
            source: String::new(),
        }
    }
}

/// Everything that can be wrong with a frame or its parts.
#[derive(Debug, PartialEq, Eq)]
pub enum MeshError {
    /// Name empty, longer than [`MAX_NAME`] bytes, or containing NUL.
    Name,
    /// Port outside `u16`.
    Port(u32),
    /// Frame length does not match the declared name length.
    Len,
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeshError::Name => write!(f, "name must be 1..{MAX_NAME} bytes without NUL"),
            MeshError::Port(port) => write!(f, "port must be 0..65535, got {port}"),
            MeshError::Len => write!(f, "frame length does not match its name length"),
        }
    }
}

impl std::error::Error for MeshError {}

/// Encode one announce frame.
pub fn pack_announce(name: &str, port: u16, seq: u32) -> Result<Vec<u8>, MeshError> {
    let raw = name.as_bytes();
    if raw.is_empty() || raw.len() > MAX_NAME || raw.contains(&0) {
        return Err(MeshError::Name);
    }
    let mut frame = Vec::with_capacity(HEADER_FIXED + raw.len());
    frame.extend_from_slice(&MAGIC);
    frame.push(PROTO_VERSION);
    frame.extend_from_slice(&port.to_le_bytes());
    frame.extend_from_slice(&seq.to_le_bytes());
    frame.push(raw.len() as u8);
    frame.extend_from_slice(raw);
    Ok(frame)
}

/// Decode one frame, or `None`.  Never panics on arbitrary input.
pub fn parse_announce(frame: &[u8]) -> Option<Peer> {
    if frame.len() < HEADER_FIXED {
        return None;
    }
    if frame[0..4] != MAGIC {
        return None;
    }
    if frame[4] != PROTO_VERSION {
        return None;
    }
    let port = u16::from_le_bytes([frame[5], frame[6]]);
    let seq = u32::from_le_bytes([frame[7], frame[8], frame[9], frame[10]]);
    let name_len = frame[11] as usize;
    if name_len == 0 || name_len > MAX_NAME {
        return None;
    }
    if frame.len() != HEADER_FIXED + name_len {
        return None;
    }
    let raw = &frame[HEADER_FIXED..];
    if raw.contains(&0) {
        return None;
    }
    let name = std::str::from_utf8(raw).ok()?.to_string();
    Some(Peer {
        name,
        port,
        seq,
        source: String::new(),
    })
}

/// Insert `incoming` into `current`; higher `seq` wins, sorted by name.
///
/// The returned map is ordered, so two nodes that heard the same peers in a
/// different order end up with the same list.
pub fn merge_peers(current: &[Peer], incoming: &Peer) -> Vec<Peer> {
    let mut merged: BTreeMap<String, Peer> = BTreeMap::new();
    for peer in current {
        let replace = match merged.get(&peer.name) {
            None => true,
            Some(existing) => peer.seq >= existing.seq,
        };
        if replace {
            merged.insert(peer.name.clone(), peer.clone());
        }
    }
    match merged.get(&incoming.name) {
        None => {
            merged.insert(incoming.name.clone(), incoming.clone());
        }
        Some(existing) => {
            if incoming.seq > existing.seq {
                merged.insert(incoming.name.clone(), incoming.clone());
            } else if incoming.seq == existing.seq
                && existing.source.is_empty()
                && !incoming.source.is_empty()
            {
                let mut filled = existing.clone();
                filled.source = incoming.source.clone();
                merged.insert(incoming.name.clone(), filled);
            }
        }
    }
    merged.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_fixed_and_little_endian() {
        let frame = pack_announce("hearth", PROTO_PORT, 7).unwrap();
        assert_eq!(&frame[0..4], &MAGIC);
        assert_eq!(frame[4], PROTO_VERSION);
        assert_eq!(&frame[5..7], &PROTO_PORT.to_le_bytes());
        assert_eq!(&frame[7..11], &7u32.to_le_bytes());
        assert_eq!(frame[11] as usize, "hearth".len());
        assert_eq!(&frame[HEADER_FIXED..], b"hearth");
        assert_eq!(frame.len(), HEADER_FIXED + "hearth".len());
    }

    #[test]
    fn round_trip() {
        let peer = parse_announce(&pack_announce("thunderbird", 8424, 0xDEAD_BEEF).unwrap());
        assert_eq!(peer, Some(Peer::new("thunderbird", 8424, 0xDEAD_BEEF)));
    }

    #[test]
    fn name_bounds() {
        assert!(parse_announce(&pack_announce(&"n".repeat(MAX_NAME), 1, 0).unwrap()).is_some());
        assert_eq!(pack_announce("", 1, 0), Err(MeshError::Name));
        assert_eq!(
            pack_announce(&"n".repeat(MAX_NAME + 1), 1, 0),
            Err(MeshError::Name)
        );
        assert_eq!(pack_announce("a\u{0}b", 1, 0), Err(MeshError::Name));
    }

    #[test]
    fn truncation_and_corruption_are_none() {
        let frame = pack_announce("hearth", PROTO_PORT, 1).unwrap();
        for cut in 0..frame.len() {
            assert!(parse_announce(&frame[..cut]).is_none(), "cut {cut}");
        }
        let mut wrong = frame.clone();
        wrong[0] = b'X';
        assert!(parse_announce(&wrong).is_none());
        let mut version = frame.clone();
        version[4] = 2;
        assert!(parse_announce(&version).is_none());
        let mut lying = frame.clone();
        lying[11] = 10;
        assert!(parse_announce(&lying).is_none());
    }

    #[test]
    fn merge_prefers_higher_seq_and_sorts() {
        let old = Peer::new("hearth", PROTO_PORT, 1);
        let new = Peer::new("hearth", PROTO_PORT, 2);
        assert_eq!(
            merge_peers(std::slice::from_ref(&old), &new),
            vec![new.clone()]
        );
        assert_eq!(
            merge_peers(std::slice::from_ref(&new), &old),
            vec![new.clone()]
        );

        let mut peers: Vec<Peer> = Vec::new();
        for name in ["zeta", "alpha", "hearth"] {
            peers = merge_peers(&peers, &Peer::new(name, 1, 1));
        }
        let names: Vec<&str> = peers.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "hearth", "zeta"]);
    }

    #[test]
    fn merge_fills_source_without_reordering() {
        let current = vec![Peer::new("hearth", 1, 1)];
        let mut heard = Peer::new("hearth", 1, 1);
        heard.source = "10.0.0.7".to_string();
        let merged = merge_peers(&current, &heard);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].source, "10.0.0.7");
        assert_eq!(current[0].source, "");
    }
}
