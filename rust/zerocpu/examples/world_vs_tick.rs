//! Why the ring is closed-form and carries no bitmasks.
//!
//! A proposed `active_mask`/`anchor_mask` optimises a per-tick decay loop: skip
//! the slots that cannot change. This measures whether that loop should exist
//! at all, by doing both and comparing the answer as well as the clock.
//!
//! Run: `cargo run --release --example world_vs_tick`

use std::time::Instant;

use zerocpu::memnet::RingMemNet;
use zerocpu::observer::pade_exp_neg;

const HALF_LIFE: i64 = 100;
const REPS: u32 = 200;

fn build() -> RingMemNet {
    let mut ring = RingMemNet::new();
    for i in 0..64i64 {
        let amp = if i % 3 == 0 { 65536 } else { -32768 };
        ring.record(amp, i % 16, i == 0).expect("ring has room");
    }
    ring
}

/// One tick's worth of decay as a fixed-point multiply: `exp(-1/half_life)`.
fn per_tick_factor() -> i64 {
    pade_exp_neg((1 << 16) / HALF_LIFE)
}

fn main() {
    let ring = build();
    let factor = per_tick_factor();
    println!(
        "ring slots: {}  per-tick factor: {factor} (Q16)",
        ring.len()
    );

    for now in [400i64, 4_000, 40_000] {
        // --- closed form: one evaluation at `now` -----------------------
        let start = Instant::now();
        let mut closed = 0i64;
        for _ in 0..REPS {
            closed = ring.variance_world(now, HALF_LIFE).unwrap();
        }
        let closed_time = start.elapsed();

        // --- per-tick: mutate a copy of the band once per tick ----------
        let start = Instant::now();
        let mut ticked = 0i64;
        for _ in 0..REPS {
            let mut amps: Vec<i64> = ring.slots().map(|s| s.amp_q16).collect();
            let anchor = ring.slots().map(|s| s.anchor).collect::<Vec<_>>();
            let born: Vec<i64> = ring.slots().map(|s| s.born_tick).collect();
            for tick in 0..=now {
                for (i, amp) in amps.iter_mut().enumerate() {
                    if anchor[i] || tick <= born[i] {
                        continue;
                    }
                    *amp = (*amp * factor) >> 16;
                }
            }
            ticked = amps.iter().sum();
        }
        let ticked_time = start.elapsed();

        let ratio = ticked_time.as_nanos() as f64 / closed_time.as_nanos().max(1) as f64;
        let drift = ticked - closed;
        let ppm = if closed == 0 {
            0
        } else {
            drift * 1_000_000 / closed
        };
        println!(
            "now={now:>6}  closed={:>9} ns  per-tick={:>9} ns  ({ratio:>6.1}x)  \
             closed={closed:>10}  ticked={ticked:>10}  drift={drift:>8} ({ppm} ppm)",
            closed_time.as_nanos() / REPS as u128,
            ticked_time.as_nanos() / REPS as u128,
        );
    }
    println!(
        "\nA mask can remove work from the loop; it cannot make the loop shorter \
         than not having it -- and it cannot remove the drift the loop accumulates."
    );
}
