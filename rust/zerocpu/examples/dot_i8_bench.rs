//! Throughput of the int8 dot product, by length.
//!
//! `dot_i8` is the kernel behind every `SpacetimeGraph.tick` and every
//! `Router` pass. Three spellings:
//!
//!   scalar   the reference loop
//!   neon     the shipped aarch64 kernel (vmull + vpadalq, 16 bytes per step)
//!   dotprod  aarch64 `vdotq_s32` -- four int8 multiply-accumulates per
//!            instruction, when the CPU has ARMv8.2 DotProd (every Apple
//!            Silicon, every Cortex-A55 or newer)
//!
//! Run: `cargo run --release --example dot_i8_bench`

use zerocpu::ternary::dot_i8_scalar;

fn main() {
    let n = 4096usize;
    let a: Vec<i8> = (0..n).map(|k| ((k * 37) % 255) as i8).collect();
    let b: Vec<i8> = (0..n).map(|k| ((k * 91 + 7) % 255) as i8).collect();

    const REPS: u32 = 2_000;
    let mut sink = 0i64;

    for len in [1usize, 8, 16, 64, 256, 4096] {
        let (x, y) = (&a[..len], &b[..len]);

        let start = std::time::Instant::now();
        for _ in 0..REPS {
            sink = sink.wrapping_add(dot_i8_scalar(x, y));
        }
        let scalar = start.elapsed().as_nanos() as f64 / REPS as f64;

        let start = std::time::Instant::now();
        for _ in 0..REPS {
            sink = sink.wrapping_add(zerocpu::ternary::dot_i8(x, y));
        }
        let shipped = start.elapsed().as_nanos() as f64 / REPS as f64;

        let neon = {
            let start = std::time::Instant::now();
            for _ in 0..REPS {
                sink = sink.wrapping_add(neon_kernel(x, y));
            }
            start.elapsed().as_nanos() as f64 / REPS as f64
        };
        println!(
            "len={len:>5}  scalar={scalar:>8.2} ns  neon={neon:>8.2} ns  \
             shipped={shipped:>8.2} ns   (shipped vs neon {:.2}x)  [sink {sink}]",
            scalar / shipped
        );
    }
    println!("\nARMv8.2 DotProd available here: {}", dotprod());
}

/// The hand-written NEON kernel, kept on the chart so the choice of dispatcher
/// can be judged rather than asserted.
fn neon_kernel(a: &[i8], b: &[i8]) -> i64 {
    zerocpu::neon::dot_i8_neon(a, b)
}

fn dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        return std::arch::is_aarch64_feature_detected!("dotprod");
    }
    #[allow(unreachable_code)]
    false
}
