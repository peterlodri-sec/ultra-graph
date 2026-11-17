//! MemNet: sequence-ordered associative memory that merges without a clock.
//!
//! Every write carries a monotonic sequence number and [`MemNet::merge`] keeps
//! the higher rank.  The rank is `(seq, value)` -- sequence first, value as a
//! total tie-break -- which makes merge commutative, associative and
//! idempotent.  Keys are identified by their 32-bit FNV-1a id, so a collision
//! is treated as the same key.
//!
//! `ultragraph.zerocpu.memnet` is the reference; `tests/memnet_parity.rs` pins
//! the two together.

use std::collections::BTreeMap;
use std::fmt;

use crate::observer::{pade_exp_neg, ObserverError, RATE_ONE};

/// Address space a key id is folded into.
pub const MEMNET_SLOTS: u32 = 256;
/// Wire format version.
pub const MEMNET_VERSION: u32 = 1;
/// Fixed ring size for [`RingMemNet`] -- the whole memory fits in L1.
pub const MEMNET_RING: usize = 64;

const FNV_OFFSET: u32 = 0x811C_9DC5;
const FNV_PRIME: u32 = 0x0100_0193;

/// One association: which slot, which key, what value, written when.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemNetEntry {
    pub slot: u32,
    pub key_id: u32,
    pub value: i32,
    pub seq: u32,
}

impl MemNetEntry {
    /// Merge order: sequence dominates, value breaks ties totally.
    pub fn rank(&self) -> (u32, i32) {
        (self.seq, self.value)
    }

    /// `slot:keyid:value:seq`, the wire form of one entry.
    pub fn to_text(&self) -> String {
        format!("{}:{}:{}:{}", self.slot, self.key_id, self.value, self.seq)
    }
}

/// Everything that can go wrong here.
#[derive(Debug, PartialEq, Eq)]
pub enum MemNetError {
    /// The key was empty.
    EmptyKey,
    /// A value or sequence outside its wire range.
    Range(String),
    /// An entry was malformed.
    BadEntry(String),
    /// Two entries claimed the same key id.
    DuplicateKey(u32),
}

impl fmt::Display for MemNetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemNetError::EmptyKey => write!(f, "key must not be empty"),
            MemNetError::Range(what) => write!(f, "out of range: {what}"),
            MemNetError::BadEntry(chunk) => write!(f, "bad entry {chunk:?}"),
            MemNetError::DuplicateKey(id) => write!(f, "duplicate key id {id}"),
        }
    }
}

impl std::error::Error for MemNetError {}

/// 32-bit FNV-1a of the UTF-8 bytes of `key`.
pub fn key_id(key: &str) -> u32 {
    let mut digest = FNV_OFFSET;
    for byte in key.as_bytes() {
        digest = (digest ^ u32::from(*byte)).wrapping_mul(FNV_PRIME);
    }
    digest
}

/// A mergeable key -> `i32` store.  One entry per key id.
#[derive(Debug, Clone, Default)]
pub struct MemNet {
    entries: BTreeMap<u32, MemNetEntry>,
    seq: u32,
}

impl MemNet {
    /// An empty store.
    pub fn new() -> Self {
        MemNet::default()
    }

    /// Store `value` under `key` and return the entry that was written.
    pub fn put(&mut self, key: &str, value: i32) -> Result<MemNetEntry, MemNetError> {
        if key.is_empty() {
            return Err(MemNetError::EmptyKey);
        }
        let id = key_id(key);
        let entry = MemNetEntry {
            slot: id % MEMNET_SLOTS,
            key_id: id,
            value,
            seq: self.seq,
        };
        self.seq += 1;
        self.entries.insert(id, entry);
        Ok(entry)
    }

    /// The value stored under `key`, if any.
    pub fn get(&self, key: &str) -> Option<i32> {
        self.entries.get(&key_id(key)).map(|entry| entry.value)
    }

    /// Every entry, sorted by `(slot, key_id, seq)` -- never map order.
    pub fn entries(&self) -> Vec<MemNetEntry> {
        let mut out: Vec<MemNetEntry> = self.entries.values().copied().collect();
        out.sort_by_key(|entry| (entry.slot, entry.key_id, entry.seq));
        out
    }

    /// Fold `other` in; return how many entries changed.
    ///
    /// Idempotent and order-independent: merging the same peer twice is a
    /// no-op, and `a.merge(b)` leaves the same state as `b.merge(a)`.
    pub fn merge(&mut self, other: &MemNet) -> usize {
        let mut changed = 0usize;
        for incoming in other.entries.values() {
            let replace = match self.entries.get(&incoming.key_id) {
                None => true,
                Some(current) => incoming.rank() > current.rank(),
            };
            if replace {
                self.entries.insert(incoming.key_id, *incoming);
                changed += 1;
            }
        }
        if let Some(highest) = self.entries.values().map(|entry| entry.seq).max() {
            if self.seq <= highest {
                self.seq = highest + 1;
            }
        }
        changed
    }

    /// Wire form: `slot:keyid:value:seq` entries joined by commas.
    pub fn to_hex(&self) -> String {
        self.entries()
            .iter()
            .map(MemNetEntry::to_text)
            .collect::<Vec<String>>()
            .join(",")
    }

    /// Inverse of [`MemNet::to_hex`].
    pub fn from_hex(text: &str) -> Result<MemNet, MemNetError> {
        let mut net = MemNet::new();
        if text.is_empty() {
            return Ok(net);
        }
        for chunk in text.split(',') {
            let parts: Vec<&str> = chunk.split(':').collect();
            if parts.len() != 4 {
                return Err(MemNetError::BadEntry(chunk.to_string()));
            }
            let slot: i64 = parts[0]
                .parse()
                .map_err(|_| MemNetError::BadEntry(chunk.to_string()))?;
            let id: i64 = parts[1]
                .parse()
                .map_err(|_| MemNetError::BadEntry(chunk.to_string()))?;
            let value: i64 = parts[2]
                .parse()
                .map_err(|_| MemNetError::BadEntry(chunk.to_string()))?;
            let seq: i64 = parts[3]
                .parse()
                .map_err(|_| MemNetError::BadEntry(chunk.to_string()))?;
            if !(0..i64::from(MEMNET_SLOTS)).contains(&slot) {
                return Err(MemNetError::Range(chunk.to_string()));
            }
            if !(0..=i64::from(u32::MAX)).contains(&id) {
                return Err(MemNetError::Range(chunk.to_string()));
            }
            if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&value) {
                return Err(MemNetError::Range(chunk.to_string()));
            }
            if seq < 0 || seq > i64::from(u32::MAX) {
                return Err(MemNetError::Range(chunk.to_string()));
            }
            let id = id as u32;
            if net.entries.contains_key(&id) {
                return Err(MemNetError::DuplicateKey(id));
            }
            net.entries.insert(
                id,
                MemNetEntry {
                    slot: slot as u32,
                    key_id: id,
                    value: value as i32,
                    seq: seq as u32,
                },
            );
        }
        if let Some(highest) = net.entries.values().map(|entry| entry.seq).max() {
            net.seq = highest + 1;
        }
        Ok(net)
    }

    /// Number of entries held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when nothing is stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// True when `key` is stored.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(&key_id(key))
    }
}

/// One ring entry: a signed amplitude, when it was written, and its anchor bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingSlot {
    pub id: u32,
    pub amp_q16: i64,
    pub born_tick: i64,
    pub anchor: bool,
}

impl RingSlot {
    /// The sign of the amplitude, as -1, 0 or +1.
    pub fn trit(&self) -> i8 {
        crate::trit::trit_of(self.amp_q16)
    }
}

/// Fixed-capacity, allocation-free MemNet with protected anchors.
///
/// Where [`MemNet`] answers "who wrote last", this answers "what is left after
/// time": amplitudes decay by half-life through the closed-form Pade
/// exponential, anchors do not decay at all, and [`RingMemNet::consensus`]
/// collapses the weighted sum to a single trit.
///
/// Two behaviours are deliberately not copied from the sketch: a full ring of
/// anchors returns `None` instead of silently dropping the write while still
/// handing back an id, and placement scans from the cursor to the first free or
/// non-anchor slot, so eviction order is a function of the writes.
#[derive(Debug, Clone)]
pub struct RingMemNet {
    slots: [Option<RingSlot>; MEMNET_RING],
    /// Indices of the occupied slots, in the order they were first written.
    ///
    /// The ring is fixed-capacity and allocation-free, so this is a second
    /// fixed array rather than a `Vec`: reading a nearly empty ring walks a
    /// handful of entries instead of branching over 64 empty ones.
    occupied: [u16; MEMNET_RING],
    occupied_len: usize,
    cursor: usize,
    seq: u32,
}

impl Default for RingMemNet {
    fn default() -> Self {
        Self::new()
    }
}

impl RingMemNet {
    /// An empty ring.
    pub fn new() -> Self {
        Self {
            slots: [None; MEMNET_RING],
            occupied: [0; MEMNET_RING],
            occupied_len: 0,
            cursor: 0,
            seq: 0,
        }
    }

    /// Place a slot and return its id, or `None` when every slot is anchored.
    pub fn record(&mut self, amp_q16: i64, born_tick: i64, anchor: bool) -> Option<u32> {
        for offset in 0..MEMNET_RING {
            let index = (self.cursor + offset) % MEMNET_RING;
            let reusable = match &self.slots[index] {
                Some(existing) => !existing.anchor,
                None => true,
            };
            if reusable {
                if self.slots[index].is_none() {
                    self.occupied[self.occupied_len] = index as u16;
                    self.occupied_len += 1;
                }
                self.seq = self.seq.saturating_add(1);
                self.slots[index] = Some(RingSlot {
                    id: self.seq,
                    amp_q16,
                    born_tick,
                    anchor,
                });
                self.cursor = (index + 1) % MEMNET_RING;
                return Some(self.seq);
            }
        }
        None
    }

    /// The occupied slots, in ring order.
    pub fn slots(&self) -> impl Iterator<Item = &RingSlot> {
        self.slots.iter().flatten()
    }

    /// `E_v0[t]`: the weighted sum of every remembered amplitude at `now`.
    ///
    /// Anchors contribute their full amplitude forever; everything else decays
    /// as `e^-(elapsed / half_life)` and is then scaled by the observer's
    /// `rate_q16` -- the `dT/dt` term.  Every step is integer.
    pub fn variance(
        &self,
        now: i64,
        half_life_ticks: i64,
        rate_q16: i64,
    ) -> Result<i64, ObserverError> {
        if half_life_ticks <= 0 {
            return Err(ObserverError::BadHalfLife);
        }
        if rate_q16 < 0 {
            return Err(ObserverError::NegativeRate);
        }
        let mut total: i64 = 0;
        for &raw_index in &self.occupied[..self.occupied_len] {
            let slot = match &self.slots[raw_index as usize] {
                Some(slot) => slot,
                None => continue,
            };
            let amplitude = if slot.anchor {
                RATE_ONE
            } else {
                let elapsed = now - slot.born_tick;
                if elapsed <= 0 {
                    RATE_ONE
                } else {
                    pade_exp_neg(crate::observer::decay_x(elapsed, half_life_ticks))
                }
            };
            let term = (slot.amp_q16 * amplitude) >> 16;
            total = total.saturating_add((term * rate_q16) >> 16);
        }
        Ok(total)
    }

    /// The settled trit: +1, 0 or -1 from the current variance.
    pub fn consensus(
        &self,
        now: i64,
        half_life_ticks: i64,
        rate_q16: i64,
    ) -> Result<i8, ObserverError> {
        let total = self.variance(now, half_life_ticks, rate_q16)?;
        Ok(crate::trit::trit_of(total))
    }

    /// `E_v0[t]` with **no** rate scaling: the number every node agrees on.
    ///
    /// `variance` multiplies by an observer's `rate_q16` -- correct for a local
    /// reading, fatal for a verdict, because two frames with different rates
    /// would then disagree about the same band at the same world tick and no
    /// amount of message exchange could reconcile them.  Consensus therefore
    /// consumes world ticks only; a frame may relabel a tick, it may not change
    /// how much of it has passed.
    ///
    /// Consequence worth stating plainly: there is nothing to synchronise.
    /// Two nodes that exchange a world tick and a half-life compute the same
    /// integer on both sides, offline, with no clock authority anywhere.
    pub fn variance_world(
        &self,
        now_world: i64,
        half_life_ticks: i64,
    ) -> Result<i64, ObserverError> {
        self.variance(now_world, half_life_ticks, RATE_ONE)
    }

    /// The settled trit, +1 / 0 / -1, from `variance_world`.
    ///
    /// The sign of the weighted sum -- not "is any anchor alive".  An anchor
    /// that has been outvoted contributes its amplitude and still loses.
    pub fn consensus_world(
        &self,
        now_world: i64,
        half_life_ticks: i64,
    ) -> Result<i8, ObserverError> {
        Ok(crate::trit::trit_of(
            self.variance_world(now_world, half_life_ticks)?,
        ))
    }

    /// This observer's private reading: `variance` scaled by its own rate.
    ///
    /// Legitimate for display ("time runs fast here"), never for agreement.
    /// Kept a separate method so the rate cannot be passed by accident to a
    /// path that then calls itself consensus.
    pub fn local_variance(
        &self,
        frame: &crate::observer::ObserverFrame,
        now_world: i64,
        half_life_ticks: i64,
    ) -> Result<i64, ObserverError> {
        self.variance(now_world, half_life_ticks, frame.rate_q16)
    }

    /// Number of stored slots.
    pub fn len(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    /// `true` when nothing has been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_id_is_stable_fnv1a() {
        assert_eq!(key_id(""), 0x811C_9DC5);
        assert_eq!(key_id("a"), 0xE40C_292C);
        assert_ne!(key_id("fit"), key_id("fit "));
    }

    #[test]
    fn put_get_and_sequence() {
        let mut net = MemNet::new();
        let first = net.put("fit", 3).unwrap();
        assert_eq!(first.slot, key_id("fit") % MEMNET_SLOTS);
        assert_eq!(first.seq, 0);
        assert_eq!(net.get("fit"), Some(3));
        assert_eq!(net.get("missing"), None);
        assert_eq!(net.put("k", 1).unwrap().seq, 1);
    }

    #[test]
    fn merge_is_idempotent_and_takes_the_higher_rank() {
        let mut left = MemNet::new();
        let mut right = MemNet::new();
        left.put("a", 1).unwrap();
        right.put("shared", 10).unwrap();
        right.put("shared", 30).unwrap();
        assert_eq!(left.merge(&right), 1);
        assert_eq!(left.merge(&right), 0);
        assert_eq!(left.get("shared"), Some(30));
    }

    #[test]
    fn merge_is_order_independent() {
        let mut a = MemNet::new();
        let mut b = MemNet::new();
        a.put("k", 1).unwrap();
        b.put("k", 9).unwrap();
        let mut ab = MemNet::from_hex(&a.to_hex()).unwrap();
        let mut ba = MemNet::from_hex(&b.to_hex()).unwrap();
        ab.merge(&b);
        ba.merge(&a);
        assert_eq!(ab.to_hex(), ba.to_hex());
        assert_eq!(ab.get("k"), Some(9));
    }

    #[test]
    fn ring_anchors_outlive_every_half_life() {
        let mut ring = RingMemNet::new();
        assert_eq!(ring.record(3 * RATE_ONE, 0, true), Some(1));
        assert_eq!(ring.record(RATE_ONE, 0, false), Some(2));
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.variance(0, 10, RATE_ONE), Ok(4 * RATE_ONE));
        assert_eq!(ring.variance(1_000_000_000, 10, RATE_ONE), Ok(3 * RATE_ONE));
        assert_eq!(ring.consensus(1_000_000_000, 10, RATE_ONE), Ok(1));
        assert_eq!(
            ring.variance(0, 0, RATE_ONE),
            Err(crate::observer::ObserverError::BadHalfLife)
        );
    }

    #[test]
    fn ring_refuses_a_full_ring_of_anchors() {
        let mut ring = RingMemNet::new();
        for tick in 0..MEMNET_RING {
            assert!(ring.record(RATE_ONE, tick as i64, true).is_some());
        }
        assert_eq!(ring.record(RATE_ONE, 99, true), None);
        assert_eq!(ring.len(), MEMNET_RING);
        let mut mixed = RingMemNet::new();
        mixed.record(RATE_ONE, 0, true).unwrap();
        for tick in 1..MEMNET_RING {
            mixed.record(RATE_ONE, tick as i64, false).unwrap();
        }
        assert!(mixed.record(RATE_ONE, 999, false).is_some());
        assert_eq!(mixed.len(), MEMNET_RING);
        assert_eq!(mixed.slots().filter(|slot| slot.anchor).count(), 1);
    }

    #[test]
    fn hex_round_trip_and_junk() {
        let mut net = MemNet::new();
        net.put("fit", 0).unwrap();
        net.put("collapse", -1).unwrap();
        let text = net.to_hex();
        assert_eq!(MemNet::from_hex(&text).unwrap().to_hex(), text);
        assert_eq!(MemNet::from_hex("").unwrap().len(), 0);
        for junk in ["a:b", "999:0:0:0", "0:0:0:-1", "0:0:0:0,0:0:0:0"] {
            assert!(MemNet::from_hex(junk).is_err(), "{junk}");
        }
    }
}
