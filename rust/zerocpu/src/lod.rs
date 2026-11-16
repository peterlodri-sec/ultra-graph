//! Quasi-regular partitioning -- the level-of-detail compressor, Rust side.
//!
//! Mirror of `ultragraph/zerocpu/lod.py`; the two must walk the identical tree
//! of candidate moves so their outputs can be compared byte for byte. Every
//! quantity is integer: counts in `i64`, densities in Q16.16 through the same
//! round-half-away-from-zero division the Python side uses.
//!
//! Two measures, deliberately different:
//!
//! * `concentration` -- `sum over block pairs of count^2 / (|a| * |b|)`, the
//!   objective the search increases. Large when edges live in few block pairs.
//! * `defect` -- the largest deviation of any block-pair density from the global
//!   density, i.e. how far the coarse blocks are from looking like uniform noise.
//!
//! Optimising structure raises the defect; that is not a bug, it is what
//! structure means. Reporting both stops the compressor from selling one as the
//! other, and neither number is a claim to have realised Szemeredi's lemma.

use crate::fixed::{round_div, ONE};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LodError {
    BadN(usize),
    BadK(usize, usize),
    EdgeOutOfRange(usize, usize, usize),
}

impl std::fmt::Display for LodError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LodError::BadN(n) => write!(f, "n must be positive, got {n}"),
            LodError::BadK(k, n) => write!(f, "need 1 <= k <= n, got k={k}, n={n}"),
            LodError::EdgeOutOfRange(u, v, n) => {
                write!(f, "edge {u}->{v} out of range (n={n})")
            }
        }
    }
}

impl std::error::Error for LodError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    pub blocks: usize,
    pub block_of: Vec<usize>,
    pub sizes: Vec<i64>,
    pub densities: Vec<i64>,
    pub global_density: i64,
    pub defect: i64,
    pub concentration: i64,
    pub iterations: u32,
}

/// `count^2 / (size_a * size_b)` in Q16.16, computed wide and rounded once.
fn pair_term(count: i64, size_a: i64, size_b: i64) -> i128 {
    let denom = (size_a as i128) * (size_b as i128);
    if denom == 0 {
        return 0;
    }
    let num = (count as i128) * (count as i128) * (ONE as i128);
    round_div_i128(num, denom)
}

/// Round half away from zero, arbitrary-width input (matches Python's `_round_div`).
fn round_div_i128(num: i128, den: i128) -> i128 {
    assert!(den != 0, "fixed-point division by zero");
    let (an, ad) = (num.abs(), den.abs());
    let mut q = an / ad;
    if (an % ad) * 2 >= ad {
        q += 1;
    }
    if (num < 0) == (den < 0) {
        q
    } else {
        -q
    }
}

fn density(count: i64, size_a: i64, size_b: i64) -> i64 {
    let denom = size_a * size_b;
    if denom == 0 {
        return 0;
    }
    round_div(count * ONE, denom)
}

fn global_density(n: usize, m: usize) -> i64 {
    if n == 0 {
        return 0;
    }
    round_div((m as i64) * ONE, (n as i64) * (n as i64))
}

fn concentration(counts: &[Vec<i64>], sizes: &[i64]) -> i128 {
    let k = counts.len();
    let mut sum = 0i128;
    for a in 0..k {
        for b in 0..k {
            sum += pair_term(counts[a][b], sizes[a], sizes[b]);
        }
    }
    sum
}

fn defect(counts: &[Vec<i64>], sizes: &[i64], global: i64) -> i64 {
    let k = counts.len();
    let mut worst = 0i64;
    for a in 0..k {
        for b in 0..k {
            let d = density(counts[a][b], sizes[a], sizes[b]);
            let dev = (d - global).abs();
            if dev > worst {
                worst = dev;
            }
        }
    }
    worst
}

fn counts_of(n: usize, edges: &[(usize, usize)], block_of: &[usize], k: usize) -> Vec<Vec<i64>> {
    let _ = n;
    let mut counts = vec![vec![0i64; k]; k];
    for &(u, v) in edges {
        counts[block_of[u]][block_of[v]] += 1;
    }
    counts
}

fn sizes_of(block_of: &[usize], k: usize) -> Vec<i64> {
    let mut sizes = vec![0i64; k];
    for &b in block_of {
        sizes[b] += 1;
    }
    sizes
}

/// Deterministic breadth-first order, smallest unvisited root first.
pub fn seed_order(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    for row in &mut adj {
        row.sort_unstable();
        row.dedup();
    }
    let mut order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    for root in 0..n {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut queue = std::collections::VecDeque::from([root]);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    queue.push_back(v);
                }
            }
        }
    }
    order
}

/// Cut a traversal order into `k` contiguous blocks of near-equal size.
fn balanced_order(n: usize, k: usize, order: &[usize]) -> Vec<usize> {
    let base = n / k;
    let rem = n % k;
    let mut block_of = vec![0usize; n];
    let mut i = 0usize;
    for b in 0..k {
        let size = base + usize::from(b < rem);
        for _ in 0..size {
            block_of[order[i]] = b;
            i += 1;
        }
    }
    block_of
}

/// Partition `n` vertices into `k` blocks that concentrate the edges.
///
/// `epsilon` stops early once the defect is at or below it (`0` = keep going
/// while structure improves). `max_iters` bounds the greedy passes.
pub fn partition(
    n: usize,
    edges: &[(usize, usize)],
    k: usize,
    epsilon: i64,
    max_iters: u32,
) -> Result<Partition, LodError> {
    if n == 0 {
        return Err(LodError::BadN(n));
    }
    if k == 0 || k > n {
        return Err(LodError::BadK(k, n));
    }
    for &(u, v) in edges {
        if u >= n || v >= n {
            return Err(LodError::EdgeOutOfRange(u, v, n));
        }
    }

    let block_of = balanced_order(n, k, &seed_order(n, edges));
    let mut counts = counts_of(n, edges, &block_of, k);
    let mut sizes = sizes_of(&block_of, k);
    let global = global_density(n, edges.len());
    let mut best = Partition {
        blocks: k,
        block_of,
        sizes: sizes.clone(),
        densities: (0..k)
            .flat_map(|a| (0..k).map(move |b| (a, b)))
            .map(|(a, b)| density(counts[a][b], sizes[a], sizes[b]))
            .collect(),
        global_density: global,
        defect: defect(&counts, &sizes, global),
        concentration: concentration(&counts, &sizes) as i64,
        iterations: 0,
    };
    if max_iters == 0 || best.defect <= epsilon {
        return Ok(best);
    }

    let mut current = best.block_of.clone();
    let capacity = (n.div_ceil(k) + 1) as i64;
    let mut out_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        out_adj[u].push(v);
        in_adj[v].push(u);
    }

    let apply = |counts: &mut Vec<Vec<i64>>,
                 sizes: &mut Vec<i64>,
                 current: &mut Vec<usize>,
                 u: usize,
                 src: usize,
                 dst: usize| {
        for &v in &out_adj[u] {
            counts[src][current[v]] -= 1;
            counts[dst][current[v]] += 1;
        }
        for &w in &in_adj[u] {
            counts[current[w]][src] -= 1;
            counts[current[w]][dst] += 1;
        }
        sizes[src] -= 1;
        sizes[dst] += 1;
        current[u] = dst;
    };

    for it in 1..=max_iters {
        let mut improved = false;
        for u in 0..n {
            let here = current[u];
            let mut target = here;
            let mut best_score = best.concentration;
            for b in 0..k {
                if b == here || sizes[b] >= capacity {
                    continue;
                }
                apply(&mut counts, &mut sizes, &mut current, u, here, b);
                let score = concentration(&counts, &sizes) as i64;
                apply(&mut counts, &mut sizes, &mut current, u, b, here);
                if score > best_score {
                    target = b;
                    best_score = score;
                }
            }
            if target == here {
                continue;
            }
            apply(&mut counts, &mut sizes, &mut current, u, here, target);
            improved = true;
            best = Partition {
                blocks: k,
                block_of: current.clone(),
                sizes: sizes.clone(),
                densities: (0..k)
                    .flat_map(|a| (0..k).map(move |b| (a, b)))
                    .map(|(a, b)| density(counts[a][b], sizes[a], sizes[b]))
                    .collect(),
                global_density: global,
                defect: defect(&counts, &sizes, global),
                concentration: best_score,
                iterations: it,
            };
            if best.defect <= epsilon {
                return Ok(best);
            }
        }
        if !improved {
            break;
        }
    }
    Ok(best)
}

/// Aggregate edges between blocks into weighted super-edges (sorted, deduped).
///
/// Aggregation is exact: the weights sum to the original edge count.
pub fn coarsen(n: usize, edges: &[(usize, usize)], block_of: &[usize]) -> Vec<(u32, u32, u32)> {
    assert_eq!(block_of.len(), n, "block_of length mismatch");
    let k = block_of.iter().copied().max().map_or(0, |b| b + 1);
    let mut agg = vec![0u32; k * k];
    for &(u, v) in edges {
        agg[block_of[u] * k + block_of[v]] += 1;
    }
    let mut out = Vec::new();
    for a in 0..k {
        for b in 0..k {
            let w = agg[a * k + b];
            if w > 0 {
                out.push((a as u32, b as u32, w));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clusters() -> (usize, Vec<(usize, usize)>) {
        let n = 12usize;
        let mut edges = Vec::new();
        for cl in [0..4usize, 4..8, 8..12] {
            for u in cl.clone() {
                for v in cl.clone() {
                    if u != v {
                        edges.push((u, v));
                    }
                }
            }
        }
        edges.extend([(0, 4), (4, 8), (8, 0)]);
        (n, edges)
    }

    #[test]
    fn rejects_bad_input() {
        assert!(matches!(partition(0, &[], 1, 0, 1), Err(LodError::BadN(0))));
        assert!(matches!(partition(4, &[], 0, 0, 1), Err(LodError::BadK(0, 4))));
        assert!(matches!(partition(4, &[], 5, 0, 1), Err(LodError::BadK(5, 4))));
        assert!(matches!(
            partition(4, &[(0, 9)], 2, 0, 1),
            Err(LodError::EdgeOutOfRange(0, 9, 4))
        ));
    }

    #[test]
    fn seed_order_is_a_permutation() {
        let (n, edges) = clusters();
        let order = seed_order(n, &edges);
        assert_eq!(order.len(), n);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn partition_keeps_sizes_balanced_and_recovers_a_cluster() {
        let (n, edges) = clusters();
        let p = partition(n, &edges, 3, 0, 8).expect("valid graph");
        assert_eq!(p.block_of.len(), n);
        assert_eq!(p.blocks, 3);
        let max = *p.sizes.iter().max().unwrap();
        let min = *p.sizes.iter().min().unwrap();
        assert!(max - min <= 1, "sizes {max} vs {min}");
        // vertices 0..4 form a near-clique; they must share one block
        let b0 = p.block_of[0];
        assert!((1..4).all(|v| p.block_of[v] == b0), "cluster split: {:?}", p.block_of);
        assert!(p.concentration > 0);
        assert_eq!(p.iterations, 1);
    }

    #[test]
    fn concentration_improves_over_the_seed() {
        let (n, edges) = clusters();
        let seed = balanced_order(n, 3, &seed_order(n, &edges));
        let seed_conc = concentration(&counts_of(n, &edges, &seed, 3), &sizes_of(&seed, 3));
        let p = partition(n, &edges, 3, 0, 8).expect("valid graph");
        assert!(p.concentration as i128 >= seed_conc, "{} < {seed_conc}", p.concentration);
    }

    #[test]
    fn coarsening_preserves_mass() {
        let (n, edges) = clusters();
        let p = partition(n, &edges, 3, 0, 8).expect("valid graph");
        let sup = coarsen(n, &edges, &p.block_of);
        let total: u32 = sup.iter().map(|&(_, _, w)| w).sum();
        assert_eq!(total as usize, edges.len());
        assert!(sup.len() <= edges.len());
    }

    #[test]
    fn epsilon_stops_before_any_move() {
        let (n, edges) = clusters();
        let p = partition(n, &edges, 3, ONE, 8).expect("valid graph");
        assert_eq!(p.iterations, 0);
    }

    #[test]
    fn single_block_is_trivial() {
        let (n, edges) = clusters();
        let p = partition(n, &edges, 1, 0, 8).expect("valid graph");
        assert_eq!(p.block_of, vec![0; n]);
        assert_eq!(p.sizes, vec![n as i64]);
        assert_eq!(p.defect, 0);
    }
}
