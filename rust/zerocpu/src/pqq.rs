//! `.pqq` programs: a compiled, RNG-free mutation script for a [`World`].
//!
//! The Python driver (`ultragraph.zerocpu.pqq`) settles every `q P/Q` draw once,
//! at compile time, and writes the outcome.  What arrives here is therefore a
//! plain list of unconditional ops -- replay needs no PRNG, so it cannot diverge
//! from the Python that produced the file.
//!
//! ```text
//! pqq 1
//! # resolved: 4 applied, 2 skipped of 6
//! # seed 7
//! step 2
//! edge 1 3 1
//! flip 0 1
//! set 2 -9
//! ```
//!
//! Ops: `edge SRC DST W` (`W` in `{-1,0,1}`), `flip SRC DST`, `prune SRC DST`,
//! `set NODE VALUE`, `step N`.  Anything carrying a `q` ratio is rejected: an
//! unresolved program is a compile bug, not something to guess about.

use std::fmt;

use crate::world::World;

pub const PQQ_HEADER: &str = "pqq 1";

/// A malformed program, or an op the world cannot accept.  `line` is 1-based.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PqError {
    pub line: usize,
    pub message: String,
}

impl PqError {
    fn at(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for PqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            f.write_str(&self.message)
        } else {
            write!(f, "line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for PqError {}

/// One unconditional world edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Edge { src: u32, dst: u32, weight: i8 },
    Flip { src: u32, dst: u32 },
    Prune { src: u32, dst: u32 },
    Set { node: u32, value: i64 },
    Step { n: u64 },
}

/// What a replay did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Report {
    pub applied: u32,
    pub ticks: u64,
}

fn parse_index(token: &str, line: usize, what: &str) -> Result<u32, PqError> {
    let v: i64 = token
        .parse()
        .map_err(|_| PqError::at(line, format!("{what} must be an integer, got {token:?}")))?;
    if v < 0 {
        return Err(PqError::at(line, format!("{what} must be >= 0, got {v}")));
    }
    u32::try_from(v).map_err(|_| PqError::at(line, format!("{what} {v} does not fit in u32")))
}

fn parse_op(line: usize, kind: &str, args: &[&str]) -> Result<Op, PqError> {
    match kind {
        "edge" => {
            if args.len() != 3 {
                return Err(PqError::at(line, "edge needs SRC DST WEIGHT"));
            }
            let weight: i64 = args[2].parse().map_err(|_| {
                PqError::at(
                    line,
                    format!("weight must be an integer, got {:?}", args[2]),
                )
            })?;
            if !(-1..=1).contains(&weight) {
                return Err(PqError::at(
                    line,
                    format!("weight must be -1, 0 or 1, got {weight}"),
                ));
            }
            Ok(Op::Edge {
                src: parse_index(args[0], line, "src index")?,
                dst: parse_index(args[1], line, "dst index")?,
                weight: weight as i8,
            })
        }
        "flip" | "prune" => {
            if args.len() != 2 {
                return Err(PqError::at(line, format!("{kind} needs SRC DST")));
            }
            let src = parse_index(args[0], line, "src index")?;
            let dst = parse_index(args[1], line, "dst index")?;
            Ok(if kind == "flip" {
                Op::Flip { src, dst }
            } else {
                Op::Prune { src, dst }
            })
        }
        "set" => {
            if args.len() != 2 {
                return Err(PqError::at(line, "set needs NODE VALUE"));
            }
            let value: i64 = args[1].parse().map_err(|_| {
                PqError::at(line, format!("value must be an integer, got {:?}", args[1]))
            })?;
            Ok(Op::Set {
                node: parse_index(args[0], line, "node index")?,
                value,
            })
        }
        "step" => {
            if args.len() != 1 {
                return Err(PqError::at(line, "step needs N"));
            }
            let n: i64 = args[0].parse().map_err(|_| {
                PqError::at(
                    line,
                    format!("step count must be an integer, got {:?}", args[0]),
                )
            })?;
            if n < 0 {
                return Err(PqError::at(
                    line,
                    format!("step count must be >= 0, got {n}"),
                ));
            }
            Ok(Op::Step { n: n as u64 })
        }
        "wire" => Err(PqError::at(
            line,
            "wire is an UltraGraph op; a world has no named trees",
        )),
        other => Err(PqError::at(
            line,
            format!("unknown op {other:?}; expected edge, flip, prune, set or step"),
        )),
    }
}

/// Parse a compiled `.pqq` program.  `#` starts a comment; blank lines are ignored.
pub fn parse(text: &str) -> Result<Vec<Op>, PqError> {
    let mut ops = Vec::new();
    let mut seen_header = false;
    for (i, raw) in text.lines().enumerate() {
        let lineno = i + 1;
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if !seen_header {
            if toks.join(" ") != PQQ_HEADER {
                return Err(PqError::at(
                    lineno,
                    format!("expected header {PQQ_HEADER:?}, got {line:?}"),
                ));
            }
            seen_header = true;
            continue;
        }
        if toks.contains(&"q") {
            return Err(PqError::at(
                lineno,
                "unresolved draw: a .pqq must be compiled with pqq.resolve() first",
            ));
        }
        ops.push(parse_op(lineno, toks[0], &toks[1..])?);
    }
    if !seen_header {
        return Err(PqError::at(
            0,
            format!("empty program: expected {PQQ_HEADER:?} header"),
        ));
    }
    Ok(ops)
}

/// Apply one op; returns how many ticks it performed.
fn apply_one(world: &mut World, line: usize, op: Op) -> Result<u64, PqError> {
    match op {
        Op::Edge { src, dst, weight } => {
            world
                .add_edge(src, dst, weight, 1)
                .map_err(|e| PqError::at(line, e.0))?;
            Ok(0)
        }
        Op::Flip { src, dst } => {
            let at = world
                .find_edge(src, dst)
                .ok_or_else(|| PqError::at(line, format!("flip: no edge {src} -> {dst}")))?;
            let flipped = -world.edges[at].weight;
            world
                .set_edge_weight(at, flipped)
                .map_err(|e| PqError::at(line, e.0))?;
            Ok(0)
        }
        Op::Prune { src, dst } => {
            if world.remove_edge(src, dst) == 0 {
                return Err(PqError::at(line, format!("prune: no edge {src} -> {dst}")));
            }
            Ok(0)
        }
        Op::Set { node, value } => {
            world
                .set_state(node, value)
                .map_err(|e| PqError::at(line, e.0))?;
            Ok(0)
        }
        Op::Step { n } => {
            world.tick_n(n);
            Ok(n)
        }
    }
}

/// Replay `text` against `world`.  Fails atomically in the sense that matters:
/// the op that fails reports its line, and nothing after it is applied.
pub fn run(world: &mut World, text: &str) -> Result<Report, PqError> {
    let mut report = Report::default();
    let mut seen_header = false;
    for (i, raw) in text.lines().enumerate() {
        let lineno = i + 1;
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if !seen_header {
            if line != PQQ_HEADER {
                return Err(PqError::at(
                    lineno,
                    format!("expected header {PQQ_HEADER:?}, got {line:?}"),
                ));
            }
            seen_header = true;
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.contains(&"q") {
            return Err(PqError::at(
                lineno,
                "unresolved draw: a .pqq must be compiled with pqq.resolve() first",
            ));
        }
        let op = parse_op(lineno, toks[0], &toks[1..])?;
        let line_ticks = apply_one(world, lineno, op)?;
        report.ticks += line_ticks;
        report.applied += 1;
    }
    if !seen_header {
        return Err(PqError::at(
            0,
            format!("empty program: expected {PQQ_HEADER:?} header"),
        ));
    }
    Ok(report)
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
    fn replays_the_committed_program() {
        let text = std::fs::read_to_string("tests/fixtures/replay.pqq").expect("program");
        let base = std::fs::read("tests/fixtures/vaked_diamond.ugmz").expect("artifact");
        let mut world = World::parse(&base).expect("parses");
        let report = run(&mut world, &text).expect("replays");
        assert_eq!(report.applied, 4);
        assert_eq!(report.ticks, 3);

        let want = std::fs::read_to_string("tests/fixtures/replay.expected.txt").expect("expected");
        let field = |key: &str| -> &str {
            want.lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("missing {key}"))
                .trim()
        };
        assert_eq!(world.state_hex(), field("state"));
        assert_eq!(world.pos_hex(), field("pos"));
        assert_eq!(world.tick_count.to_string(), field("tick"));
        assert_eq!(world.n_edges().to_string(), field("edges"));
        // The written-back artifact must survive a round trip.
        let bytes = world.to_bytes();
        assert_eq!(World::parse(&bytes).expect("reparses"), world);
    }

    #[test]
    fn rejects_unresolved_and_malformed_programs() {
        let mut w = diamond();
        assert!(run(&mut w, "step 1").is_err(), "missing header");
        assert!(
            run(&mut w, "pqq 1\nset 1 2 q 1/2").is_err(),
            "unresolved draw"
        );
        assert!(
            run(&mut w, "pqq 1\nwire a b").is_err(),
            "wire is not a world op"
        );
        assert!(run(&mut w, "pqq 1\nedge 0 99 1").is_err(), "missing node");
        assert!(
            run(&mut w, "pqq 1\nedge 0 1 2").is_err(),
            "non-ternary weight"
        );
        assert!(run(&mut w, "pqq 1\nflip 0 2").is_err(), "no such edge");
        assert!(run(&mut w, "pqq 1\nprune 0 2").is_err(), "no such edge");
        assert!(run(&mut w, "pqq 1\nstep -1").is_err(), "negative step");
        assert!(run(&mut w, "pqq 1\nnope 1 2").is_err(), "unknown op");
        assert!(run(&mut w, "pqq 2\nstep 1").is_err(), "wrong version");
    }

    #[test]
    fn parse_and_run_agree_on_op_count() {
        let text = "pqq 1\n# c\nstep 1\n\nedge 0 1 -1 # tail\n";
        let ops = parse(text).expect("parses");
        assert_eq!(ops.len(), 2);
        let mut w = diamond();
        assert_eq!(run(&mut w, text).expect("runs").applied, 2);
    }
}
