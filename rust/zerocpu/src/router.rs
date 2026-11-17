//! Deterministic integer routing: the SmartTree fanout.
//!
//! Same prompt, same route, on every machine: a fixed sequence of integer
//! comparisons over [`Features`], with no floats, no regex and no iteration
//! order dependence.  `ultragraph.zerocpu.router` is the reference;
//! `tests/router_parity.rs` pins the two together.

use std::fmt;

/// Local ternary kernel.  Cheapest, fully deterministic.
pub const ROUTE_CAVEMAN: u8 = 0;
/// Batched kernel pass, still integer-only.
pub const ROUTE_BATCH: u8 = 1;
/// Hand off to a model: too big or too sharp for the kernel.
pub const ROUTE_MODEL: u8 = 2;
/// An explicit `@`/`/` token named the handler.
pub const ROUTE_PLUGIN: u8 = 3;
/// Nothing to do.
pub const ROUTE_DENY: u8 = 4;

/// Above this many characters the kernel cannot hold the context.
pub const ROUTE_MAX_CHARS_HEAVY: usize = 2000;
/// At or below this many characters the Caveman kernel handles it alone.
pub const ROUTE_MAX_CHARS_BATCH: usize = 160;
/// Trition budget for one Caveman pass (reserved).
pub const ROUTE_MAX_TRITS: usize = 512;
/// This many `!`/`?` in a long prompt wants a real reasoner.
pub const ROUTE_MAX_MARKS: usize = 3;

/// Mixed into each character before the position weight is applied.
pub const HEAT_SALT: u32 = 31;
/// The heat word is 16 bits wide.
pub const HEAT_MASK: u32 = 0xFFFF;

/// Integer features of a prompt.  Field order mirrors `FEATURE_KEYS` in Python.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Features {
    pub chars: usize,
    pub words: usize,
    pub lines: usize,
    pub digits: usize,
    pub plugins: usize,
    pub marks: usize,
    pub heat: u16,
}

/// Everything that can go wrong here.
#[derive(Debug, PartialEq, Eq)]
pub enum RouterError {
    /// The route code is not one of the five names.
    UnknownRoute(u8),
}

impl fmt::Display for RouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RouterError::UnknownRoute(code) => write!(f, "unknown route code {code}"),
        }
    }
}

impl std::error::Error for RouterError {}

/// Integer features of `prompt`.
///
/// `heat` is a position-weighted sum so that `"ab"` and `"ba"` hash apart.
pub fn features(prompt: &str) -> Features {
    let mut chars = 0usize;
    let mut digits = 0usize;
    let mut marks = 0usize;
    let mut lines = 0usize;
    let mut heat: u32 = 0;
    for (i, ch) in prompt.chars().enumerate() {
        chars += 1;
        if ch == '\n' {
            lines += 1;
        }
        if ch.is_ascii_digit() {
            digits += 1;
        }
        if ch == '!' || ch == '?' {
            marks += 1;
        }
        let salted = (ch as u32).wrapping_mul(HEAT_SALT).wrapping_add(i as u32);
        heat = heat.wrapping_add(salted.wrapping_mul(i as u32 + 1) & HEAT_MASK);
    }
    let words = prompt.split_whitespace().count();
    let plugins = prompt
        .split_whitespace()
        .filter(|token| token.starts_with('@') || token.starts_with('/'))
        .count();
    Features {
        chars,
        words,
        lines: if chars == 0 { 0 } else { lines + 1 },
        digits,
        plugins,
        marks,
        heat: (heat & HEAT_MASK) as u16,
    }
}

/// Pick a route.  The order of these tests is the specification.
pub fn route(feats: &Features) -> u8 {
    if feats.chars == 0 {
        return ROUTE_DENY;
    }
    if feats.plugins > 0 {
        return ROUTE_PLUGIN;
    }
    if feats.chars >= ROUTE_MAX_CHARS_HEAVY {
        return ROUTE_MODEL;
    }
    if feats.marks >= ROUTE_MAX_MARKS && feats.chars > ROUTE_MAX_CHARS_BATCH {
        return ROUTE_MODEL;
    }
    if feats.chars <= ROUTE_MAX_CHARS_BATCH {
        return ROUTE_CAVEMAN;
    }
    ROUTE_BATCH
}

/// Human name for a route code.
pub fn route_name(code: u8) -> Result<&'static str, RouterError> {
    match code {
        ROUTE_CAVEMAN => Ok("ROUTE_CAVEMAN"),
        ROUTE_BATCH => Ok("ROUTE_BATCH"),
        ROUTE_MODEL => Ok("ROUTE_MODEL"),
        ROUTE_PLUGIN => Ok("ROUTE_PLUGIN"),
        ROUTE_DENY => Ok("ROUTE_DENY"),
        other => Err(RouterError::UnknownRoute(other)),
    }
}

/// The reason chain that [`route`] walks, ending in the decision.
pub fn explain(feats: &Features) -> Vec<String> {
    if feats.chars == 0 {
        return vec!["chars=0 -> ROUTE_DENY (nothing to route)".to_string()];
    }
    let mut lines = vec![format!(
        "chars={} words={} lines={} heat=0x{:04x}",
        feats.chars, feats.words, feats.lines, feats.heat
    )];
    if feats.plugins > 0 {
        lines.push(format!(
            "plugins={} > 0 -> ROUTE_PLUGIN (explicit delegation wins)",
            feats.plugins
        ));
        return lines;
    }
    if feats.chars >= ROUTE_MAX_CHARS_HEAVY {
        lines.push(format!(
            "chars={} >= ROUTE_MAX_CHARS_HEAVY={} -> ROUTE_MODEL",
            feats.chars, ROUTE_MAX_CHARS_HEAVY
        ));
        return lines;
    }
    if feats.marks >= ROUTE_MAX_MARKS && feats.chars > ROUTE_MAX_CHARS_BATCH {
        lines.push(format!(
            "marks={} >= ROUTE_MAX_MARKS={} and chars={} > ROUTE_MAX_CHARS_BATCH={} -> ROUTE_MODEL",
            feats.marks, ROUTE_MAX_MARKS, feats.chars, ROUTE_MAX_CHARS_BATCH
        ));
        return lines;
    }
    if feats.chars <= ROUTE_MAX_CHARS_BATCH {
        lines.push(format!(
            "chars={} <= ROUTE_MAX_CHARS_BATCH={} -> ROUTE_CAVEMAN",
            feats.chars, ROUTE_MAX_CHARS_BATCH
        ));
        return lines;
    }
    lines.push(format!(
        "ROUTE_MAX_CHARS_BATCH < chars={} < ROUTE_MAX_CHARS_HEAVY={} -> ROUTE_BATCH",
        feats.chars, ROUTE_MAX_CHARS_HEAVY
    ));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feats(prompt: &str) -> Features {
        features(prompt)
    }

    #[test]
    fn empty_prompt_is_denied() {
        let f = feats("");
        assert_eq!((f.chars, f.words, f.lines, f.heat), (0, 0, 0, 0));
        assert_eq!(route(&f), ROUTE_DENY);
    }

    #[test]
    fn counters_match_python_semantics() {
        let f = feats("abc 12 @tool /plugin why?!\nsecond");
        assert_eq!(f.chars, 33);
        assert_eq!(f.words, 6);
        assert_eq!(f.lines, 2);
        assert_eq!(f.digits, 2);
        assert_eq!(f.plugins, 2);
        assert_eq!(f.marks, 2);
    }

    #[test]
    fn heat_is_position_weighted() {
        assert_ne!(feats("ab").heat, feats("ba").heat);
        assert_eq!(feats("same").heat, feats("same").heat);
    }

    #[test]
    fn thresholds() {
        assert_eq!(
            route(&feats(&"a".repeat(ROUTE_MAX_CHARS_BATCH))),
            ROUTE_CAVEMAN
        );
        assert_eq!(
            route(&feats(&"a".repeat(ROUTE_MAX_CHARS_BATCH + 1))),
            ROUTE_BATCH
        );
        assert_eq!(
            route(&feats(&"a".repeat(ROUTE_MAX_CHARS_HEAVY))),
            ROUTE_MODEL
        );
        assert_eq!(route(&feats("@plugin hi")), ROUTE_PLUGIN);
        assert_eq!(
            route(&feats(&format!(
                "{}?!?!",
                "b".repeat(ROUTE_MAX_CHARS_BATCH + 10)
            ))),
            ROUTE_MODEL
        );
    }

    #[test]
    fn explain_ends_on_the_decision() {
        let long = "a".repeat(2500);
        for prompt in ["", "hi", "@x y", "why?!?!", long.as_str()] {
            let f = feats(prompt);
            let reasons = explain(&f);
            let name = route_name(route(&f)).unwrap();
            assert!(reasons.last().unwrap().contains(name), "{prompt}");
        }
        assert!(route_name(99).is_err());
    }
}
