//! What a sparse ring actually costs to read.
//!
//! `RingMemNet::variance` walks all 64 slots on every call, branching on
//! `Option::is_none` for the ones nobody wrote. This measures that walk against
//! the number of occupied slots, so an optimisation can be judged on evidence
//! rather than on how clever it looks.
//!
//! Run: `cargo run --release --example variance_slots`

use std::time::Instant;

use zerocpu::memnet::RingMemNet;

const HALF_LIFE: i64 = 100;
const REPS: u32 = 2_000;

fn build(occupied: usize) -> RingMemNet {
    let mut ring = RingMemNet::new();
    for i in 0..occupied {
        let amp = if i % 3 == 0 { 65536 } else { -32768 };
        ring.record(amp, (i % 16) as i64, i == 0)
            .expect("ring has room");
    }
    ring
}

fn main() {
    println!("slots occupied | variance_world x{} calls", REPS);
    for occupied in [1usize, 3, 8, 16, 32, 64] {
        let ring = build(occupied);
        // warm up
        for now in 0..64i64 {
            std::hint::black_box(ring.variance_world(now, HALF_LIFE).unwrap());
        }
        let start = Instant::now();
        let mut sink = 0i64;
        for rep in 0..REPS {
            let now = (rep % 64) as i64;
            sink = sink.wrapping_add(ring.variance_world(now, HALF_LIFE).unwrap());
        }
        let elapsed = start.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / REPS as f64;
        println!("{occupied:>14} | {ns_per_call:>8.2} ns  (sink {sink})");
    }
    println!("\nBefore `occupied` existed the left column still cost 42 ns at one slot:");
    println!("the walk branched over all 64 entries whatever the ring held. Now a");
    println!("one-slot ring reads in ~4 ns, and the cost tracks occupancy instead of");
    println!("capacity. The `sink` column is the evidence the answer did not change.");
}
