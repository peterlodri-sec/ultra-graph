//! Hope cascade: a text becomes integer energy, then the world does the rest.
//!
//! There is no separate cascade engine here, and that is the point: a signal is
//! folded into per-node ternary states and propagation is [`World::tick_n`] --
//! the same synchronous integer rule the rest of the crate uses.  A resonance is
//! therefore inspectable: seed, tick, read the state.
//!
//! Conventions, stated plainly because they are conventions:
//!
//! * **bucket** -- character `i` with codepoint `o` lands in lane
//!   `(o * 31 + i) % n`; the position salt makes a permutation a different signal.
//! * **polarity** -- even codepoints push `+`, odd pull `-`, each by
//!   [`SIGNAL_STEP`] of heat.
//! * **centering** -- a lane's trit is the sign of its heat minus the per-lane
//!   mean (floor division via [`i64::div_euclid`]), so what resonates is the lanes
//!   above the text's own average.  Without it every lane of a long text drifts
//!   to `+1` and carries no shape.
//!
//! No floats, no keyword table, no narration: the artifact is a state.

use std::fmt;

use crate::world::World;

/// One unit of heat per character.  `1 << 8` keeps 256 characters inside the
/// int8 state range before saturation starts to matter.
pub const SIGNAL_STEP: i64 = 1 << 8;

/// Mixed into the lane index so position, not just content, shapes the signal.
pub const SALT: i64 = 31;

/// A malformed projection (empty lane count, or a signal that does not fit).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HopeError(pub String);

impl fmt::Display for HopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HopeError {}

/// A text projected onto `n` lanes: signed heat plus its ternary sign.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signal {
    pub heat: Vec<i64>,
    pub trits: Vec<i8>,
}

impl Signal {
    pub fn len(&self) -> usize {
        self.heat.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heat.is_empty()
    }

    pub fn total_heat(&self) -> i64 {
        self.heat.iter().sum()
    }

    /// Sum of trits: how lopsided the signal is, in `[-n, +n]`.
    pub fn balance(&self) -> i64 {
        self.trits.iter().map(|t| *t as i64).sum()
    }
}

/// The trace of a cascade: what went in, and every state it passed through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resonance {
    pub signal: Signal,
    pub rounds: u64,
    pub trace: Vec<String>,
    pub collapse: i8,
}

impl Resonance {
    pub fn settled(&self) -> &str {
        self.trace.last().map(String::as_str).unwrap_or("")
    }
}

/// Project `text` onto `n` lanes.  Deterministic; no floats, no keywords.
pub fn signal(text: &str, n: usize) -> Result<Signal, HopeError> {
    if n < 1 {
        return Err(HopeError(format!("n must be >= 1, got {n}")));
    }
    let mut heat = vec![0i64; n];
    for (i, ch) in text.chars().enumerate() {
        let o = ch as i64;
        let lane = ((o * SALT + i as i64) % n as i64) as usize;
        heat[lane] += if o & 1 == 0 { SIGNAL_STEP } else { -SIGNAL_STEP };
    }
    let mean = heat.iter().sum::<i64>().div_euclid(n as i64);
    let trits = heat
        .iter()
        .map(|h| {
            let d = *h - mean;
            if d == 0 {
                0
            } else if d > 0 {
                1
            } else {
                -1
            }
        })
        .collect();
    Ok(Signal { heat, trits })
}

/// Write a signal's ternary sign into the world.  Lane count must match exactly.
pub fn seed(world: &mut World, sig: &Signal) -> Result<(), HopeError> {
    if sig.len() != world.n_nodes() {
        return Err(HopeError(format!(
            "signal has {} lanes, world has {} nodes",
            sig.len(),
            world.n_nodes()
        )));
    }
    for (i, trit) in sig.trits.iter().enumerate() {
        let idx = u32::try_from(i).map_err(|_| HopeError(format!("lane {i} does not fit in u32")))?;
        world
            .set_state(idx, *trit as i64)
            .map_err(|e| HopeError(e.0))?;
    }
    Ok(())
}

/// Seed the world with `text`, tick `rounds` times, and record the trace.
pub fn resonate(world: &mut World, text: &str, rounds: u64) -> Result<Resonance, HopeError> {
    let sig = signal(text, world.n_nodes())?;
    seed(world, &sig)?;
    let mut trace = vec![world.state_hex()];
    for _ in 0..rounds {
        world.tick_n(1);
        trace.push(world.state_hex());
    }
    let collapse = observer(world);
    Ok(Resonance {
        signal: sig,
        rounds,
        trace,
        collapse,
    })
}

/// The read side: collapse a world state to one of `{-1, 0, +1}`.
pub fn observer(world: &World) -> i8 {
    let total: i64 = world
        .state
        .iter()
        .map(|s| if *s > 0 { 1 } else if *s < 0 { -1 } else { 0 })
        .sum();
    if total == 0 {
        0
    } else if total > 0 {
        1
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diamond() -> World {
        let bytes = std::fs::read("tests/fixtures/vaked_diamond.ugmz").expect("artifact");
        World::parse(&bytes).expect("parses")
    }

    #[test]
    fn signal_is_deterministic_and_carries_shape() {
        let text = "Hope is all around. Difference is everywhere.";
        let a = signal(text, 8).expect("ok");
        let b = signal(text, 8).expect("ok");
        assert_eq!(a, b);
        assert_eq!(a.len(), 8);
        assert!(a.trits.iter().all(|t| (-1..=1).contains(t)));
        // floor centering leaves the remainder in the lanes, so the trits are
        // not forced to sum to zero -- but the signal must have *shape*.
        assert!(a.balance().abs() <= a.len() as i64);
        assert!(a.trits.contains(&1) && a.trits.contains(&-1), "both polarities");
        // position matters: a permutation is a different signal
        let rev: String = text.chars().rev().collect();
        assert_ne!(signal(&rev, 8).expect("ok").heat, a.heat);
        assert!(signal(text, 0).is_err());
    }

    #[test]
    fn resonate_seeds_ticks_and_collapses() {
        let mut w = diamond();
        let r = resonate(&mut w, "Hope is all around.", 2).expect("ok");
        assert_eq!(r.rounds, 2);
        assert_eq!(r.trace.len(), 3);
        assert_eq!(r.settled(), w.state_hex());
        assert_eq!(r.collapse, observer(&w));
        assert_eq!(r.signal.len(), w.n_nodes());

        let mut mismatched = World::new(3).expect("ok");
        for i in 0..2 {
            mismatched.add_node(1, [i, 0, 0]);
        }
        let sig = signal("ab", 5).expect("ok");
        assert!(seed(&mut mismatched, &sig).is_err(), "lane/node mismatch");
    }

    #[test]
    fn matches_the_committed_fixture() {
        let text = std::fs::read_to_string("tests/fixtures/hope_signal.txt").expect("text");
        let want = std::fs::read_to_string("tests/fixtures/hope.expected.txt").expect("expected");
        let field = |key: &str| -> String {
            want.lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("missing {key}"))
                .trim()
                .to_string()
        };

        let mut w = diamond();
        let r = resonate(&mut w, &text, 2).expect("ok");

        let heat_hex: String = r
            .signal
            .heat
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .map(|b| format!("{b:02x}"))
            .collect();
        let trits: String = r.signal.trits.iter().map(|t| format!("{t}")).collect();
        assert_eq!(heat_hex, field("heat"));
        assert_eq!(trits, field("trits"));
        assert_eq!(r.trace.join(","), field("trace"));
        assert_eq!(r.collapse.to_string(), field("collapse"));
    }
}
