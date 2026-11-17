//! What the wide divide costs, and where the fast path earns it back.
//!
//! `decay_x` computes `x = elapsed * 65536 / half_life`, clamped to the Pade
//! domain. Three spellings, timed against each other:
//!
//!   wide    `((elapsed as i128) << 16) / half_life` -- exact, but `__divti3`
//!   hybrid  the shipped version: i64 multiply + i64 divide, wide only on
//!           overflow, plus a reject for bands older than the domain
//!   naive   `elapsed.wrapping_shl(16) / half_life` -- a lower bound that is
//!           *wrong* for astronomical elapsed, kept only to show the floor
//!
//! Run: `cargo run --release --example decay_x_cost`

use std::time::Instant;

use zerocpu::observer::{decay_x, PADE_X_MAX};

#[inline(never)]
fn wide(elapsed: i64, half_life: i64) -> i64 {
    (((elapsed as i128) << 16) / (half_life as i128)).min(PADE_X_MAX as i128) as i64
}

#[inline(never)]
fn hybrid(elapsed: i64, half_life: i64) -> i64 {
    decay_x(elapsed, half_life)
}

#[inline(never)]
fn naive(elapsed: i64, half_life: i64) -> i64 {
    (elapsed.wrapping_shl(16) / half_life).min(PADE_X_MAX)
}

fn main() {
    let cases: [(&str, i64, i64); 6] = [
        ("fresh band", 1, 100),
        ("mid band", 4_000, 100),
        ("aged out", 9_000_000_000_000, 10),
        ("i64::MAX elapsed", i64::MAX, 1),
        ("i64::MAX elapsed, i64::MAX half", i64::MAX, i64::MAX),
        ("huge half-life", 1_000, i64::MAX / 2),
    ];

    for (name, elapsed, half) in cases {
        let want = wide(elapsed, half);
        let got = hybrid(elapsed, half);
        println!(
            "{name:<32} wide={want:>10} hybrid={got:>10} {}",
            if want == got { "ok" } else { "MISMATCH" }
        );
        assert_eq!(want, got, "{name}");
    }
    println!("\nThe naive floor is only shown for timing; it is wrong on rows 3-5.");

    for (name, elapsed, half) in cases {
        const REPS: u32 = 20_000;
        let mut sinks = [0i64; 3];
        let mut times = [0u128; 3];
        for (slot, f) in [
            (0usize, wide as fn(i64, i64) -> i64),
            (1, hybrid as fn(i64, i64) -> i64),
            (2, naive as fn(i64, i64) -> i64),
        ] {
            let start = Instant::now();
            for _ in 0..REPS {
                sinks[slot] = sinks[slot].wrapping_add(f(elapsed, half));
            }
            times[slot] = start.elapsed().as_nanos();
        }
        println!(
            "{name:<32} wide={:>6} ns  hybrid={:>6} ns  naive={:>6} ns   (hybrid/wide {:.2}x)",
            times[0] / REPS as u128,
            times[1] / REPS as u128,
            times[2] / REPS as u128,
            times[1] as f64 / times[0].max(1) as f64,
        );
        println!("{sinks:?}");
    }
}
