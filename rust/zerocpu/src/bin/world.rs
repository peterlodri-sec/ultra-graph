//! `world` -- tick a compiled `UGMZ` spacetime world without Python in the loop.
//!
//! ```text
//! world <artifact.ugmz> [ticks]
//! ```
//!
//! The artifact is written by `ultragraph.zerocpu.emit` (Python). This binary
//! parses it, ticks the world with the same synchronous integer rule as
//! `SpacetimeGraph.tick`, and prints the resulting state so the Python side can
//! assert byte equality. The per-node accumulation goes through
//! [`zerocpu::dot_i8`], which dispatches to the aarch64 NEON kernel.

use std::process::ExitCode;

use zerocpu::world::{World, VERSION};

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
    world.tick_n(ticks);
    println!(
        "ok magic=UGMZ version={} nodes={} edges={} pos_dim={} start_tick={} ticks={} tick={}",
        VERSION,
        world.n_nodes(),
        world.n_edges(),
        world.pos_dim,
        start_tick,
        ticks,
        world.tick_count
    );
    println!("state={}", world.state_hex());
    println!("pos={}", world.pos_hex());
    ExitCode::SUCCESS
}
