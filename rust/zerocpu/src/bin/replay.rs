//! `replay` -- apply a compiled `.pqq` program to a `UGMZ` world.
//!
//! ```text
//! replay <artifact.ugmz> <program.pqq> [out.ugmz]
//! ```
//!
//! The program is RNG-free by construction (the Python driver resolved every
//! `q P/Q` draw at compile time), so this is a literal transcription of the ops.
//! With `out.ugmz` the mutated world is written back in the same v1 layout.

use std::process::ExitCode;

use zerocpu::pqq;
use zerocpu::world::World;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        eprintln!("usage: {} <artifact.ugmz> <program.pqq> [out.ugmz]", args[0]);
        return ExitCode::from(2);
    }
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
    let program = match std::fs::read_to_string(&args[2]) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cannot read {}: {e}", args[2]);
            return ExitCode::from(1);
        }
    };
    let start_tick = world.tick_count;
    let report = match pqq::run(&mut world, &program) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("bad program: {e}");
            return ExitCode::from(1);
        }
    };
    println!(
        "ok applied={} ticks={} nodes={} edges={} start_tick={} tick={}",
        report.applied,
        report.ticks,
        world.n_nodes(),
        world.n_edges(),
        start_tick,
        world.tick_count
    );
    println!("state={}", world.state_hex());
    println!("pos={}", world.pos_hex());
    if let Some(out) = args.get(3) {
        if let Err(e) = std::fs::write(out, world.to_bytes()) {
            eprintln!("cannot write {out}: {e}");
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}
