//! Observer frames, the Mahakala guard, and memories that outlive their clock.
//!
//! Rust twin of `ultragraph.zerocpu.observer`.  Same constants, same integer
//! recurrence, same refusal to sniff keywords: a frame is an explicit mapping
//! from world ticks to local time, and an anchor is something the caller
//! declares.  `local = (tick - origin) * rate >> 16` is exact, so two nodes can
//! print different "now"s for one world tick and still agree on the structure.
//!
//! `decay` is bounded to [`MAX_DECAY_STEPS`] multiplications, so no input can
//! spin it forever; `exp()` would drift between platforms, this cannot.

use std::collections::BTreeMap;
use std::fmt;

/// One world tick of proper time in Q16.16.
pub const RATE_ONE: i64 = 1 << 16;
/// `round(e ** -0.05 * 65536)`: per-tick variance multiplier.
pub const DECAY_Q16: i64 = 62339;
/// 0.1 in Q16.16; below this a memory is indistinguishable from noise.
pub const VARIANCE_FLOOR: i64 = 6554;
/// 100.0 in Q16.16; anchors never decay.
pub const ANCHOR_VARIANCE: i64 = 100 << 16;
/// Upper bound on decay iterations.
pub const MAX_DECAY_STEPS: i64 = 4096;
/// `round(1 / ln 2 * 65536)`: splits an exponent into a power of two and a
/// remainder in `[0, ln 2)`, so `e^-x = 2^-k * e^-r` stays integer.
pub const INV_LN2_Q16: i64 = 94548;
/// `round(ln 2 * 65536)`.
pub const LN2_Q16: i64 = 45426;
/// Past 40 nats `e^-x` is below one Q16.16 ulp; the closed form says zero.
pub const PADE_X_MAX: i64 = 40 << 16;

/// Default chain of names the guard defends.
pub const GUARDIANS: [&str; 3] = ["mother", "father", "son"];

/// Why an observer operation was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObserverError {
    /// A frame rate below zero has no meaning.
    NegativeRate,
    /// A memory span below zero has no meaning.
    NegativeSpan,
    /// A decay factor outside `[0, RATE_ONE]` would grow or explode.
    BadDecay,
    /// A half-life of zero ticks has no meaning.
    BadHalfLife,
    /// A frame with `rate_q16 == 0` reads the same local tick forever, so
    /// recovering a world tick from its reading is not possible.
    FrozenClock,
}

impl fmt::Display for ObserverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::NegativeRate => "rate_q16 must be non-negative",
            Self::NegativeSpan => "span must be non-negative",
            Self::BadDecay => "decay_q16 must be within [0, RATE_ONE]",
            Self::BadHalfLife => "half_life_ticks must be positive",
            Self::FrozenClock => "a frame with rate_q16 == 0 has no invertible clock",
        };
        f.write_str(text)
    }
}

impl std::error::Error for ObserverError {}

/// Apply `elapsed` ticks of geometric decay to a Q16.16 variance.
///
/// Integer-only and bounded: at most [`MAX_DECAY_STEPS`] multiplications run and
/// anything longer collapses to `floor`.  Non-positive `elapsed` returns the
/// input unchanged, so re-ticking at the same world tick is idempotent.
pub fn decay(variance_q16: i64, elapsed: i64, decay_q16: i64, floor: i64) -> i64 {
    if elapsed <= 0 || variance_q16 <= floor {
        return variance_q16;
    }
    if elapsed > MAX_DECAY_STEPS {
        return floor;
    }
    let mut value = variance_q16;
    for _ in 0..elapsed {
        value = (value * decay_q16) >> 16;
        if value <= floor {
            return floor;
        }
    }
    value
}

/// Convenience: decay with the module defaults.
pub fn decay_default(variance_q16: i64, elapsed: i64) -> i64 {
    decay(variance_q16, elapsed, DECAY_Q16, VARIANCE_FLOOR)
}

/// `e^-x` for Q16.16 `x`, integer-only, via a `[3/3]` Pade approximant.
///
/// `x` is split as `k * ln2 + r` with `r` in `[0, ln2)`, so
/// `e^-x = 2^-k * P(-r) / P(r)` with `P(t) = 120 + 60t + 12t^2 + t^3`.  No loop
/// and no `exp()`: `e^-1` is exactly `24109` on every machine.
pub fn pade_exp_neg(x_q16: i64) -> i64 {
    if x_q16 <= 0 {
        return RATE_ONE;
    }
    if x_q16 > PADE_X_MAX {
        return 0;
    }
    let k = (x_q16 * INV_LN2_Q16) >> 32;
    let r = x_q16 - k * LN2_Q16;
    let r2 = (r * r) >> 16;
    let r3 = (r2 * r) >> 16;
    let num = 120 * RATE_ONE - 60 * r + 12 * r2 - r3;
    let den = 120 * RATE_ONE + 60 * r + 12 * r2 + r3;
    if den <= 0 || k >= 64 {
        return 0;
    }
    ((num << 16) / den) >> k
}

/// Closed-form variance decay: `e^-(elapsed / half_life)`, Q16.16.
///
/// Unlike [`decay`] this is not a recurrence: any `elapsed` costs the same few
/// multiplications, so a decade-long gap cannot turn into a loop.
pub fn decay_pade(
    variance_q16: i64,
    elapsed: i64,
    half_life_ticks: i64,
) -> Result<i64, ObserverError> {
    if half_life_ticks <= 0 {
        return Err(ObserverError::BadHalfLife);
    }
    if elapsed <= 0 {
        return Ok(variance_q16);
    }
    // `elapsed << 16` wraps for astronomical elapsed values and would flip the
    // sign of the scale, reporting "barely decayed" for a band that has aged
    // out entirely. Widen, then clamp to the Padé domain's top: fully decayed.
    let x_q16 = ((elapsed as i128) << 16) / (half_life_ticks as i128);
    let x_q16 = x_q16.min(PADE_X_MAX as i128) as i64;
    let amplitude = pade_exp_neg(x_q16);
    Ok((variance_q16 * amplitude) >> 16)
}

/// A local clock: `local = (world_tick - origin_world) * rate >> 16`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObserverFrame {
    pub id: String,
    pub origin_world: i64,
    pub rate_q16: i64,
    pub local_tick: i64,
}

impl ObserverFrame {
    /// Build a frame; `rate_q16 = RATE_ONE` keeps time with the world.
    pub fn new(id: &str, origin_world: i64, rate_q16: i64) -> Result<Self, ObserverError> {
        if rate_q16 < 0 {
            return Err(ObserverError::NegativeRate);
        }
        Ok(Self {
            id: id.to_string(),
            origin_world,
            rate_q16,
            local_tick: 0,
        })
    }

    /// A frame synchronised to the world.
    pub fn world(id: &str) -> Self {
        Self {
            id: id.to_string(),
            origin_world: 0,
            rate_q16: RATE_ONE,
            local_tick: 0,
        }
    }

    /// This frame's reading at `world_tick` (floor division, exact).
    pub fn local_time(&self, world_tick: i64) -> i64 {
        ((world_tick - self.origin_world) * self.rate_q16) >> 16
    }

    /// The earliest world tick whose reading is `local_tick`.
    ///
    /// Precisely: the smallest world tick whose reading is at least
    /// `local_tick` (ceiling division, so it works for negative ticks too).
    /// For a reading that actually happened this is exact.  A fast clock steps
    /// by more than one local tick per world tick, so some local integers are
    /// never read at all; for those the answer reads *past* the request, which
    /// is detectable: `local_time(world_of(l)) != l` means `l` was never a
    /// reading.
    ///
    /// That lossiness is why the mesh carries **world ticks, not readings**.
    /// A peer needs the tick and its own frame; asking it to invert somebody
    /// else's rounded reading is how two nodes drift apart with no one to
    /// blame.  A frozen frame has no earliest tick at all, so this refuses
    /// rather than inventing one.
    pub fn world_of(&self, local_tick: i64) -> Result<i64, ObserverError> {
        if self.rate_q16 <= 0 {
            return Err(ObserverError::FrozenClock);
        }
        // Ceiling division: the smallest tick that *reaches* the reading.
        let span = local_tick * RATE_ONE;
        let step = span + self.rate_q16 - 1;
        Ok(self.origin_world + step.div_euclid(self.rate_q16))
    }

    /// Read the clock and store it as the current local tick.
    pub fn observe(&mut self, world_tick: i64) -> i64 {
        self.local_tick = self.local_time(world_tick);
        self.local_tick
    }

    /// Move the local clock forward without consulting the world.
    pub fn advance(&mut self, ticks: i64) -> i64 {
        self.local_tick += ticks;
        self.local_tick
    }
}

/// Two frames' readings of one world tick, and the gap between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Relativity {
    pub local_a: i64,
    pub local_b: i64,
}

impl Relativity {
    /// `local_b - local_a`.
    pub fn delta(&self) -> i64 {
        self.local_b - self.local_a
    }
}

impl fmt::Display for Relativity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "local_a={} local_b={} delta={}",
            self.local_a,
            self.local_b,
            self.delta()
        )
    }
}

/// Both frames' readings of `world_tick`; equality is not required.
pub fn relativity(a: &ObserverFrame, b: &ObserverFrame, world_tick: i64) -> Relativity {
    Relativity {
        local_a: a.local_time(world_tick),
        local_b: b.local_time(world_tick),
    }
}

/// A guard over a chain of names -- fierce, and honest about what it saw.
#[derive(Debug, Clone)]
pub struct Mahakala {
    pub guardians: Vec<String>,
}

impl Default for Mahakala {
    fn default() -> Self {
        Self::new(&GUARDIANS)
    }
}

impl Mahakala {
    /// Build a guard over `guardians` (matched case-insensitively).
    pub fn new(guardians: &[&str]) -> Self {
        Self {
            guardians: guardians.iter().map(|name| name.to_lowercase()).collect(),
        }
    }

    /// The guardians present in `text`, in chain order, de-duplicated.
    ///
    /// Matching is whole-token and ASCII-alphanumeric, exactly like the Python
    /// reference: `"seasonal"` does not summon the `"son"` guard.
    pub fn raise_guard(&self, text: &str) -> Vec<String> {
        let tokens: Vec<String> = text
            .to_lowercase()
            .split(|ch: char| !ch.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
            .map(str::to_string)
            .collect();
        let mut out: Vec<String> = Vec::new();
        for name in &self.guardians {
            if tokens.iter().any(|token| token == name) && !out.contains(name) {
                out.push(name.clone());
            }
        }
        out
    }

    /// `true` when any guardian is present.
    pub fn protects(&self, text: &str) -> bool {
        !self.raise_guard(text).is_empty()
    }
}

/// One remembered thing: birth tick, peak weight, anchor bit, write order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    pub key: String,
    pub born_tick: i64,
    pub variance0_q16: i64,
    pub anchor: bool,
    pub seq: u64,
}

impl Slot {
    /// Weight at `world_tick`: anchors never move, others decay from birth.
    pub fn variance_at(&self, world_tick: i64, decay_q16: i64) -> i64 {
        if self.anchor {
            return ANCHOR_VARIANCE;
        }
        decay(
            self.variance0_q16,
            world_tick - self.born_tick,
            decay_q16,
            VARIANCE_FLOOR,
        )
    }
}

/// Memory as a witness keeps it: what was said, when, and what must not fade.
///
/// Weight is recomputed from `(born_tick, world_tick)` rather than mutated in
/// place, so `tick` is idempotent and replaying a timeline cannot drift.
#[derive(Debug, Clone)]
pub struct Witness {
    pub decay_q16: i64,
    slots: BTreeMap<String, Slot>,
    seq: u64,
}

impl Default for Witness {
    fn default() -> Self {
        Self::new(DECAY_Q16)
    }
}

impl Witness {
    /// Build a memory with a custom per-tick decay factor.
    pub fn new(decay_q16: i64) -> Self {
        assert!(
            (0..=RATE_ONE).contains(&decay_q16),
            "decay_q16 must be within [0, RATE_ONE]"
        );
        Self {
            decay_q16,
            slots: BTreeMap::new(),
            seq: 0,
        }
    }

    /// Store or replace `key`; `span` is the caller's weight estimate.
    pub fn remember(
        &mut self,
        key: &str,
        world_tick: i64,
        span: i64,
        anchor: bool,
    ) -> Result<&Slot, ObserverError> {
        if span < 0 {
            return Err(ObserverError::NegativeSpan);
        }
        let capped = (span.saturating_mul(VARIANCE_FLOOR)).min(10 * RATE_ONE);
        let weight = if anchor { ANCHOR_VARIANCE } else { capped };
        self.seq += 1;
        self.slots.insert(
            key.to_string(),
            Slot {
                key: key.to_string(),
                born_tick: world_tick,
                variance0_q16: weight,
                anchor,
                seq: self.seq,
            },
        );
        Ok(self.slots.get(key).expect("just inserted"))
    }

    /// How many live (non-anchor) slots are still above the floor.
    pub fn tick(&self, world_tick: i64) -> usize {
        self.slots
            .values()
            .filter(|slot| {
                slot.anchor || slot.variance_at(world_tick, self.decay_q16) > VARIANCE_FLOOR
            })
            .count()
    }

    /// Weight of `key` at `world_tick`; `None` if never stored.
    pub fn variance(&self, key: &str, world_tick: i64) -> Option<i64> {
        self.slots
            .get(key)
            .map(|slot| slot.variance_at(world_tick, self.decay_q16))
    }

    /// Every declared monument, sorted so two implementations agree.
    pub fn anchors(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .slots
            .iter()
            .filter(|(_, slot)| slot.anchor)
            .map(|(key, _)| key.clone())
            .collect();
        keys.sort();
        keys
    }

    /// The heaviest `limit` slots at `world_tick`, ties broken by recency.
    pub fn strongest(&self, world_tick: i64, limit: usize) -> Vec<(String, i64)> {
        let mut ranked: Vec<(i64, u64, String)> = self
            .slots
            .iter()
            .map(|(key, slot)| {
                (
                    slot.variance_at(world_tick, self.decay_q16),
                    slot.seq,
                    key.clone(),
                )
            })
            .collect();
        ranked.sort_by(|a, b| b.cmp(a));
        ranked
            .into_iter()
            .take(limit)
            .map(|(weight, _seq, key)| (key, weight))
            .collect()
    }

    /// Number of stored slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// `true` when nothing has been stored.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_q16() {
        assert_eq!(RATE_ONE, 65536);
        assert_eq!(VARIANCE_FLOOR, 6554);
        assert_eq!(ANCHOR_VARIANCE, 6_553_600);
        // A range check rather than two comparisons: clippy reads a chain of
        // constant comparisons as an assertion with a constant value, which is
        // true and useless -- the point is to pin the constant, not to prove it.
        let low = 95 * RATE_ONE / 100;
        let high = 96 * RATE_ONE / 100;
        assert!((low..high).contains(&DECAY_Q16));
    }

    #[test]
    fn frames_label_one_tick_differently() {
        let mut slow = ObserverFrame::new("hearth", 4, RATE_ONE).unwrap();
        let fast = ObserverFrame::world("thunderbird");
        assert_eq!(slow.local_time(4), 0);
        assert_eq!(fast.local_time(4), 4);
        assert_eq!(slow.observe(10), 6);
        assert_eq!(slow.advance(2), 8);
        let report = relativity(&slow, &fast, 10);
        assert_eq!((report.local_a, report.local_b, report.delta()), (6, 10, 4));
        assert_eq!(report.to_string(), "local_a=6 local_b=10 delta=4");
    }

    #[test]
    fn frozen_and_slow_frames_are_exact() {
        let frozen = ObserverFrame::new("frozen", 0, 0).unwrap();
        assert_eq!(frozen.local_time(1000), 0);
        let half = ObserverFrame::new("half", 0, RATE_ONE / 2).unwrap();
        assert_eq!(half.local_time(1), 0);
        assert_eq!(half.local_time(3), 1);
        assert_eq!(half.local_time(-1), -1);
        assert_eq!(
            ObserverFrame::new("bad", 0, -1).unwrap_err(),
            ObserverError::NegativeRate
        );
    }

    #[test]
    fn decay_is_bounded_and_idempotent() {
        let value = 10 * RATE_ONE;
        let once = decay_default(value, 3);
        assert_eq!(decay_default(value, 3), once);
        assert!(once < value);
        assert_eq!(decay_default(value, 0), value);
        assert_eq!(decay_default(value, -5), value);
        assert_eq!(decay_default(value, MAX_DECAY_STEPS), VARIANCE_FLOOR);
        assert_eq!(decay_default(value, MAX_DECAY_STEPS + 1), VARIANCE_FLOOR);
        assert_eq!(decay_default(VARIANCE_FLOOR, 10), VARIANCE_FLOOR);
    }

    #[test]
    fn guard_matches_whole_tokens_in_chain_order() {
        let guard = Mahakala::default();
        assert_eq!(guard.raise_guard("Mother, father, son."), GUARDIANS);
        assert!(guard.raise_guard("seasonal reasoning").is_empty());
        assert!(!guard.protects("seasonal reasoning"));
        assert!(guard.protects("my SON"));
        let custom = Mahakala::new(&["son", "mother"]);
        assert_eq!(
            custom.raise_guard("Mother and son and son"),
            ["son", "mother"]
        );
    }

    #[test]
    fn anchors_outlive_every_clock() {
        let mut memory = Witness::default();
        memory.remember("plain", 0, 40, false).unwrap();
        let peak = memory.variance("plain", 0).unwrap();
        assert_eq!(peak, 40 * VARIANCE_FLOOR);
        assert!(memory.variance("plain", 5).unwrap() < peak);
        memory.remember("monument", 0, 1, true).unwrap();
        assert_eq!(
            memory.variance("monument", 1_000_000).unwrap(),
            ANCHOR_VARIANCE
        );
        assert_eq!(memory.anchors(), ["monument"]);
        assert_eq!(memory.tick(0), 2);
        assert_eq!(memory.tick(MAX_DECAY_STEPS), 1);
        assert_eq!(memory.len(), 2);
        assert!(memory.variance("gone", 0).is_none());
        assert_eq!(
            memory.remember("x", 0, -1, false).unwrap_err(),
            ObserverError::NegativeSpan
        );
        assert_eq!(
            memory.strongest(0, 2),
            vec![
                ("monument".to_string(), ANCHOR_VARIANCE),
                ("plain".to_string(), 40 * VARIANCE_FLOOR)
            ]
        );
    }
}
