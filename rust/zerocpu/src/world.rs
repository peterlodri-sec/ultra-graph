//! The `UGMZ` world artifact and its synchronous integer update.
//!
//! Python writes the artifact (`ultragraph.zerocpu.emit`); this module parses it,
//! mutates it (so a compiled `.pqq` program can be replayed), and writes it back
//! byte-for-byte the way Python would.
//!
//! Layout v1 (little-endian, fixed width):
//!
//! ```text
//! header  32 B : magic "UGMZ", version u16, flags u16, n_nodes u32, n_edges u32,
//!                pos_dim u8, reserved 7 B, tick_count u64
//! node    24 B : state i8, pad 3, pos[3] i32, reserved u64
//! edge    24 B : src u32, dst u32, dt u64, weight i8, pad 7
//! ```

use std::fmt;

pub const MAGIC: &[u8; 4] = b"UGMZ";
pub const VERSION: u16 = 1;
pub const HEADER_SIZE: usize = 32;
pub const NODE_RECORD: usize = 24;
pub const EDGE_RECORD: usize = 24;

/// A parse/serialise failure, or an out-of-range graph edit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldError(pub String);

impl fmt::Display for WorldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for WorldError {}

type Result<T> = std::result::Result<T, WorldError>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(WorldError(msg.into()))
}

/// One directed, delay-carrying, ternary-weighted causal link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub src: u32,
    pub dst: u32,
    pub weight: i8,
    pub dt: u64,
}

/// A discrete spacetime world: nodes carry an `i8` state and a position, edges
/// carry a ternary weight and a time delay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct World {
    pub state: Vec<i8>,
    pub pos: Vec<[i32; 3]>,
    pub tick_count: u64,
    pub pos_dim: u8,
    /// Artifact order is preserved: `flip` rewrites the first match, `prune`
    /// removes all matches, exactly like `SpacetimeGraph`.
    pub edges: Vec<Edge>,
}

impl World {
    /// An empty world with the given position dimensionality (1..=3).
    pub fn new(pos_dim: u8) -> Result<Self> {
        if !(1..=3).contains(&pos_dim) {
            return err(format!("pos_dim {pos_dim} must be 1..=3"));
        }
        Ok(Self {
            state: Vec::new(),
            pos: Vec::new(),
            tick_count: 0,
            pos_dim,
            edges: Vec::new(),
        })
    }

    pub fn n_nodes(&self) -> usize {
        self.state.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Append a node; the state saturates into `i8` the way Python's
    /// `SpacetimeGraph.add_node` does.
    pub fn add_node(&mut self, state: i64, pos: [i32; 3]) -> usize {
        self.state.push(saturate_i8(state));
        self.pos.push(pos);
        self.state.len() - 1
    }

    pub fn add_edge(&mut self, src: u32, dst: u32, weight: i8, dt: u64) -> Result<Edge> {
        self.check_index(src)?;
        self.check_index(dst)?;
        if !(-1..=1).contains(&(weight as i32)) {
            return err(format!("causal weight must be ternary, got {weight}"));
        }
        let edge = Edge {
            src,
            dst,
            weight,
            dt,
        };
        self.edges.push(edge);
        Ok(edge)
    }

    fn check_index(&self, idx: u32) -> Result<()> {
        if idx as usize >= self.state.len() {
            return err(format!(
                "node index {idx} out of range (n={})",
                self.state.len()
            ));
        }
        Ok(())
    }

    /// Drop every `src -> dst` edge; returns how many were removed.
    pub fn remove_edge(&mut self, src: u32, dst: u32) -> usize {
        let before = self.edges.len();
        self.edges.retain(|e| !(e.src == src && e.dst == dst));
        before - self.edges.len()
    }

    /// Index of the first `src -> dst` edge, if any.
    pub fn find_edge(&self, src: u32, dst: u32) -> Option<usize> {
        self.edges.iter().position(|e| e.src == src && e.dst == dst)
    }

    pub fn set_edge_weight(&mut self, at: usize, weight: i8) -> Result<()> {
        if !(-1..=1).contains(&(weight as i32)) {
            return err(format!("causal weight must be ternary, got {weight}"));
        }
        self.edges[at].weight = weight;
        Ok(())
    }

    pub fn set_state(&mut self, idx: u32, value: i64) -> Result<()> {
        self.check_index(idx)?;
        self.state[idx as usize] = saturate_i8(value);
        Ok(())
    }

    /// One synchronous integer step: `state'[i] = clamp8(state[i] + sum_in w*state[src])`.
    ///
    /// The base state is read for the whole step before any write, which is what
    /// makes the update order-independent and matches `SpacetimeGraph.tick`.
    pub fn tick(&mut self) {
        let n = self.state.len();
        let mut delta = vec![0i64; n];
        let mut gather: Vec<i8> = Vec::new();
        let mut weights: Vec<i8> = Vec::new();
        for (dst, slot) in delta.iter_mut().enumerate() {
            gather.clear();
            weights.clear();
            for e in &self.edges {
                if e.dst as usize == dst {
                    gather.push(self.state[e.src as usize]);
                    weights.push(e.weight);
                }
            }
            if !gather.is_empty() {
                *slot = crate::dot_i8(&gather, &weights);
            }
        }
        for (state, step) in self.state.iter_mut().zip(delta.iter()) {
            *state = saturate_i8(i64::from(*state) + step);
        }
        self.tick_count += 1;
    }

    pub fn tick_n(&mut self, n: u64) {
        for _ in 0..n {
            self.tick();
        }
    }

    /// `state` as lowercase hex, one byte per node (`i8` reinterpreted as `u8`).
    pub fn state_hex(&self) -> String {
        let mut s = String::with_capacity(self.state.len() * 2);
        for b in &self.state {
            s.push_str(&format!("{:02x}", *b as u8));
        }
        s
    }

    /// Little-endian `i32` bytes per node, matching Python's `struct.pack("<iii")`.
    pub fn pos_hex(&self) -> String {
        let mut s = String::with_capacity(self.pos.len() * 24);
        for p in &self.pos {
            for c in p {
                for byte in c.to_le_bytes() {
                    s.push_str(&format!("{byte:02x}"));
                }
            }
        }
        s
    }

    // -- serialisation --------------------------------------------------------
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return err(format!("truncated header: {} bytes", bytes.len()));
        }
        if &bytes[0..4] != MAGIC {
            return err(format!("bad magic {:?}, want {:?}", &bytes[0..4], MAGIC));
        }
        let version = u16_at(bytes, 4);
        if version != VERSION {
            return err(format!("unsupported version {version}"));
        }
        let flags = u16_at(bytes, 6);
        if flags != 0 {
            return err(format!("unsupported flags {flags:#06x}"));
        }
        let n_nodes = u32_at(bytes, 8) as usize;
        let n_edges = u32_at(bytes, 12) as usize;
        let pos_dim = bytes[16];
        if !(1..=3).contains(&pos_dim) {
            return err(format!("pos_dim {pos_dim} must be 1..=3"));
        }
        let tick_count = u64_at(bytes, 24);
        let want = HEADER_SIZE + n_nodes * NODE_RECORD + n_edges * EDGE_RECORD;
        if bytes.len() != want {
            return err(format!("length {} != expected {want}", bytes.len()));
        }

        let mut state = Vec::with_capacity(n_nodes);
        let mut pos = Vec::with_capacity(n_nodes);
        let mut off = HEADER_SIZE;
        for _ in 0..n_nodes {
            state.push(bytes[off] as i8);
            pos.push([
                i32_at(bytes, off + 4),
                i32_at(bytes, off + 8),
                i32_at(bytes, off + 12),
            ]);
            off += NODE_RECORD;
        }

        let mut edges = Vec::with_capacity(n_edges);
        for _ in 0..n_edges {
            let src = u32_at(bytes, off);
            let dst = u32_at(bytes, off + 4);
            let dt = u64_at(bytes, off + 8);
            let weight = bytes[off + 16] as i8;
            if src as usize >= n_nodes || dst as usize >= n_nodes {
                return err(format!("edge references missing node: {src}->{dst}"));
            }
            if !(-1..=1).contains(&(weight as i32)) {
                return err(format!("causal weight must be ternary, got {weight}"));
            }
            edges.push(Edge {
                src,
                dst,
                weight,
                dt,
            });
            off += EDGE_RECORD;
        }

        Ok(Self {
            state,
            pos,
            tick_count,
            pos_dim,
            edges,
        })
    }

    /// Serialise to `UGMZ` v1 bytes.  `parse` of the result is `self`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            HEADER_SIZE + self.state.len() * NODE_RECORD + self.edges.len() * EDGE_RECORD,
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&(self.state.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.edges.len() as u32).to_le_bytes());
        out.push(self.pos_dim);
        out.extend_from_slice(&[0u8; 7]); // reserved
        out.extend_from_slice(&self.tick_count.to_le_bytes());
        for (i, s) in self.state.iter().enumerate() {
            out.push(*s as u8);
            out.extend_from_slice(&[0u8; 3]); // pad
            for c in self.pos[i] {
                out.extend_from_slice(&c.to_le_bytes());
            }
            out.extend_from_slice(&[0u8; 8]); // reserved
        }
        for e in &self.edges {
            out.extend_from_slice(&e.src.to_le_bytes());
            out.extend_from_slice(&e.dst.to_le_bytes());
            out.extend_from_slice(&e.dt.to_le_bytes());
            out.push(e.weight as u8);
            out.extend_from_slice(&[0u8; 7]); // pad
        }
        out
    }
}

fn saturate_i8(value: i64) -> i8 {
    value.clamp(i8::MIN as i64, i8::MAX as i64) as i8
}

fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn i32_at(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diamond() -> World {
        let mut w = World::new(3).expect("valid dim");
        for (i, s) in [1, 1, 1, 1].into_iter().enumerate() {
            w.add_node(s, [(i as i32) << 16, 0, 0]);
        }
        for (s, d) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            w.add_edge(s, d, 1, 1).expect("valid edge");
        }
        w
    }

    #[test]
    fn roundtrip_is_byte_identical() {
        let w = diamond();
        let bytes = w.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE + 4 * NODE_RECORD + 4 * EDGE_RECORD);
        assert_eq!(World::parse(&bytes).expect("parses"), w);
    }

    #[test]
    fn tick_matches_the_recorded_diamond_state() {
        let bytes = std::fs::read("tests/fixtures/vaked_diamond.ugmz").expect("artifact");
        let mut w = World::parse(&bytes).expect("parses");
        w.tick_n(2);
        let py = std::fs::read_to_string("tests/fixtures/vaked_diamond.expected.txt")
            .expect("expected file");
        let state = py
            .lines()
            .find_map(|l| l.strip_prefix("state="))
            .expect("state line")
            .trim();
        assert_eq!(w.state_hex(), state);
        assert_eq!(w.tick_count, 2);
    }

    #[test]
    fn edits_follow_python_semantics() {
        let mut w = diamond();
        assert_eq!(w.remove_edge(0, 1), 1);
        assert_eq!(w.remove_edge(0, 1), 0);
        w.add_edge(0, 1, -1, 4).expect("valid");
        let at = w.find_edge(0, 1).expect("present");
        assert_eq!(w.edges[at].weight, -1);
        w.set_edge_weight(at, 1).expect("valid");
        assert_eq!(w.edges[at].weight, 1);
        w.set_state(0, 999).expect("in range");
        assert_eq!(w.state[0], 127);
        w.set_state(0, -999).expect("in range");
        assert_eq!(w.state[0], -128);
        assert!(w.set_state(9, 0).is_err());
        assert!(w.add_edge(0, 9, 1, 0).is_err());
        assert!(w.add_edge(0, 1, 2, 0).is_err());
    }

    #[test]
    fn rejects_malformed_artifacts() {
        assert!(World::parse(b"UGM").is_err());
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(b"XXXX");
        header[4..6].copy_from_slice(&1u16.to_le_bytes());
        assert!(World::parse(&header).is_err());

        let mut good = World::new(3).expect("valid").to_bytes();
        good[16] = 9; // pos_dim
        assert!(World::parse(&good).is_err());
    }
}
