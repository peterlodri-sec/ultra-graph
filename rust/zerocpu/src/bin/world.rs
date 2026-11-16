//! `world` -- run a compiled `UGMZ` spacetime world without Python in the loop.
//!
//! ```text
//! world <artifact.ugmz> [ticks]
//! ```
//!
//! The artifact is written by `ultragraph.zerocpu.emit` (Python). This binary
//! parses it, ticks the world `ticks` times with the same synchronous integer
//! rule as `SpacetimeGraph.tick`, and prints the resulting state so the Python
//! side can assert byte equality.
//!
//! The per-node accumulation goes through [`zerocpu::dot_i8`], which dispatches
//! to the aarch64 NEON kernel. No floats, no hashing, no allocation per tick
//! beyond the scratch gather buffer.

use std::process::ExitCode;

const MAGIC: &[u8; 4] = b"UGMZ";
const VERSION: u16 = 1;
const HEADER_SIZE: usize = 32;
const NODE_RECORD: usize = 24;
const EDGE_RECORD: usize = 24;

struct World {
    state: Vec<i8>,
    pos: Vec<[i32; 3]>,
    tick_count: u64,
    n_edges: usize,
    pos_dim: u8,
    /// Per-node incoming fan-in, flattened: `incoming_start[i]..incoming_start[i+1]`.
    incoming_start: Vec<usize>,
    incoming_src: Vec<usize>,
    incoming_w: Vec<i8>,
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

impl World {
    fn parse(b: &[u8]) -> Result<Self, String> {
        if b.len() < HEADER_SIZE {
            return Err(format!("truncated header: {} bytes", b.len()));
        }
        if &b[0..4] != MAGIC {
            return Err(format!("bad magic {:?}, want {:?}", &b[0..4], MAGIC));
        }
        let version = u16_at(b, 4);
        if version != VERSION {
            return Err(format!("unsupported version {version}"));
        }
        let flags = u16_at(b, 6);
        if flags != 0 {
            return Err(format!("unsupported flags {flags:#06x}"));
        }
        let n_nodes = u32_at(b, 8) as usize;
        let n_edges = u32_at(b, 12) as usize;
        let pos_dim = b[16];
        if pos_dim > 3 {
            return Err(format!("pos_dim {pos_dim} exceeds 3"));
        }
        let tick_count = u64_at(b, 24);
        let want = HEADER_SIZE + n_nodes * NODE_RECORD + n_edges * EDGE_RECORD;
        if b.len() != want {
            return Err(format!("length {} != expected {want}", b.len()));
        }

        let mut state = Vec::with_capacity(n_nodes);
        let mut pos = Vec::with_capacity(n_nodes);
        let mut off = HEADER_SIZE;
        for _ in 0..n_nodes {
            state.push(b[off] as i8);
            pos.push([i32_at(b, off + 4), i32_at(b, off + 8), i32_at(b, off + 12)]);
            off += NODE_RECORD;
        }

        let mut srcs = Vec::with_capacity(n_edges);
        let mut dsts = Vec::with_capacity(n_edges);
        let mut ws = Vec::with_capacity(n_edges);
        for _ in 0..n_edges {
            let src = u32_at(b, off) as usize;
            let dst = u32_at(b, off + 4) as usize;
            let weight = b[off + 16] as i8;
            if src >= n_nodes || dst >= n_nodes {
                return Err(format!("edge references missing node: {src}->{dst}"));
            }
            if !(-1..=1).contains(&(weight as i32)) {
                return Err(format!("causal weight must be ternary, got {weight}"));
            }
            srcs.push(src);
            dsts.push(dst);
            ws.push(weight);
            off += EDGE_RECORD;
        }

        // Bucket by destination, preserving artifact order (order-independent anyway).
        let mut incoming_start = vec![0usize; n_nodes + 1];
        for &d in &dsts {
            incoming_start[d + 1] += 1;
        }
        for i in 1..incoming_start.len() {
            incoming_start[i] += incoming_start[i - 1];
        }
        let mut cursor = incoming_start.clone();
        let mut incoming_src = vec![0usize; n_edges];
        let mut incoming_w = vec![0i8; n_edges];
        for k in 0..n_edges {
            let d = dsts[k];
            let slot = cursor[d];
            incoming_src[slot] = srcs[k];
            incoming_w[slot] = ws[k];
            cursor[d] += 1;
        }

        Ok(Self {
            state,
            pos,
            tick_count,
            n_edges,
            pos_dim,
            incoming_start,
            incoming_src,
            incoming_w,
        })
    }

    /// One synchronous integer step: `state'[i] = clamp8(state[i] + sum_in w*state[src])`.
    fn tick(&mut self) {
        let n = self.state.len();
        let mut delta = vec![0i64; n];
        let mut gather = vec![0i8; n];
        let mut weights = vec![0i8; n];
        for dst in 0..n {
            let lo = self.incoming_start[dst];
            let hi = self.incoming_start[dst + 1];
            if lo == hi {
                continue;
            }
            let fan_in = hi - lo;
            gather.clear();
            weights.clear();
            for k in lo..hi {
                gather.push(self.state[self.incoming_src[k]]);
                weights.push(self.incoming_w[k]);
            }
            debug_assert_eq!(gather.len(), fan_in);
            delta[dst] = zerocpu::dot_i8(&gather, &weights);
        }
        for i in 0..n {
            self.state[i] = zerocpu::saturate_i8(i64::from(self.state[i]) + delta[i]);
        }
        self.tick_count += 1;
    }

    fn state_hex(&self) -> String {
        let mut s = String::with_capacity(self.state.len() * 2);
        for b in &self.state {
            s.push_str(&format!("{:02x}", *b as u8));
        }
        s
    }

    /// Little-endian i32 bytes, matching Python's `struct.pack("<iii")` -> `.hex()`.
    fn pos_hex(&self) -> String {
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
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!("usage: {} <artifact.ugmz> [ticks]", args[0]);
        return ExitCode::from(2);
    }
    let ticks: u64 = match args.get(2) {
        None => 0,
        Some(a) => match a.parse() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("bad tick count {a:?}: {e}");
                return ExitCode::from(2);
            }
        },
    };
    let bytes = match std::fs::read(&args[1]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {}: {e}", args[1]);
            return ExitCode::from(1);
        }
    };
    let mut world = match World::parse(&bytes) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("bad artifact: {e}");
            return ExitCode::from(1);
        }
    };
    let start_tick = world.tick_count;
    for _ in 0..ticks {
        world.tick();
    }
    println!(
        "ok magic=UGMZ version={} nodes={} edges={} pos_dim={} start_tick={} ticks={} tick={}",
        VERSION,
        world.state.len(),
        world.n_edges,
        world.pos_dim,
        start_tick,
        ticks,
        world.tick_count
    );
    println!("state={}", world.state_hex());
    println!("pos={}", world.pos_hex());
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_truncated_and_bad_magic() {
        assert!(World::parse(b"UGM").is_err());
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(b"XXXX");
        header[4..6].copy_from_slice(&1u16.to_le_bytes());
        assert!(World::parse(&header).is_err());
    }

    #[test]
    fn empty_world_header_roundtrips() {
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(MAGIC);
        header[4..6].copy_from_slice(&1u16.to_le_bytes());
        header[24..32].copy_from_slice(&7u64.to_le_bytes());
        let w = World::parse(&header).expect("empty world parses");
        assert_eq!(w.state.len(), 0);
        assert_eq!(w.tick_count, 7);
        assert_eq!(w.state_hex(), "");
    }
}
