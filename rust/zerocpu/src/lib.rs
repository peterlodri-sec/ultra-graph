//! zerocpu -- deterministic integer kernels for ultragraph, in Rust.
//!
//! The Python package (`ultragraph/zerocpu/`) is the reference implementation.
//! Everything here is checked against golden vectors produced by that Python
//! code (see `tests/parity.rs`), so the two cannot drift apart silently.
//!
//! Three concerns, matching the Python modules one-for-one:
//!
//! * [`fixed`] -- Q16.16 arithmetic with round-half-away-from-zero.
//! * [`ternary`] -- 5-trit-per-byte packing (`{-1,0,+1}` -> base 3) and dot
//!   products over ternary weights.
//! * [`neon`] -- the aarch64 accelerated dot product (scalar fallback elsewhere).
//! * [`lod`] -- quasi-regular partitioning: the level-of-detail compressor that
//!   keeps a large world model tickable.
//! * [`world`] -- the `UGMZ` artifact: parse, tick, mutate, write back.
//! * [`pqq`] -- compiled `.pqq` programs replayed against a [`world::World`].
//! * [`hope`] -- a text projected to integer heat, seeded into a world, ticked.
//! * [`trit`] -- the Q-delta ternary quantizer and A6P-15A bitplanes.
//! * [`router`] -- deterministic integer routing over prompt features.
//! * [`memnet`] -- sequence-ordered associative memory (merge-safe).
//! * [`mesh`] -- the LAN peer announce protocol.
//! * [`trit`] -- the A6P-15A quantizer and its bitplanes.
//! * [`router`] -- integer route selection.
//! * [`observer`] -- local clocks, guard chain, anchored memory.
//!
//! No floating point is used in any deterministic path.

pub mod fixed;
pub mod hope;
pub mod lod;
pub mod memnet;
pub mod mesh;
pub mod observer;
pub mod pqq;
pub mod router;
pub mod ternary;
pub mod trit;
pub mod world;

#[cfg(target_arch = "aarch64")]
pub mod neon;

pub use fixed::{
    from_int, from_ratio, fx_div, fx_dot, fx_mul, fx_sqrt, round_div, round_shift, saturate_i8,
    to_float, to_int, FRAC_BITS, ONE,
};
pub use lod::{coarsen, partition, seed_order, LodError, Partition};
pub use hope::{observer, resonate, signal, HopeError, Resonance, Signal};
pub use pqq::{PqError, Report};
pub use world::{Edge, World, WorldError};
pub use ternary::{dot_i8, dot_i8_scalar, dot_ternary, pack5, unpack5, TRITS_PER_BYTE};
