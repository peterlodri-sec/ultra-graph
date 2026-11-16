//! Cross-language parity for the LOD compressor.
//!
//! `tests/fixtures/lod_edges.txt`, `lod_params.txt` and `lod_expected.txt` are
//! produced by `ultragraph.zerocpu.lod` (Python); the Python test
//! `test_lod_fixture_is_stable` regenerates them and asserts byte equality, so a
//! stale fixture cannot make this test pass by accident.

use zerocpu::lod::{coarsen, partition};

fn read(name: &str) -> String {
    std::fs::read_to_string(format!("tests/fixtures/{name}")).expect("fixture present")
}

fn edges() -> Vec<(usize, usize)> {
    read("lod_edges.txt")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l.split_whitespace().map(|t| t.parse::<usize>().expect("vertex id"));
            (it.next().expect("src"), it.next().expect("dst"))
        })
        .collect()
}

fn params() -> (usize, usize, i64, u32) {
    let text = read("lod_params.txt");
    let get = |key: &str| -> String {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("missing {key} in lod_params.txt"))
            .trim()
            .to_string()
    };
    (
        get("n").parse().expect("n"),
        get("k").parse().expect("k"),
        get("epsilon").parse().expect("epsilon"),
        get("max_iters").parse().expect("max_iters"),
    )
}

fn expected() -> std::collections::HashMap<String, String> {
    read("lod_expected.txt")
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.trim().to_string()))
        .collect()
}

fn csv(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    text.split(',').map(|s| s.to_string()).collect()
}

#[test]
fn partition_matches_python_exactly() {
    let (n, k, epsilon, max_iters) = params();
    let e = edges();
    let want = expected();
    let p = partition(n, &e, k, epsilon, max_iters).expect("valid fixture graph");

    let block_of: Vec<String> = p.block_of.iter().map(|b| b.to_string()).collect();
    let sizes: Vec<String> = p.sizes.iter().map(|s| s.to_string()).collect();
    let densities: Vec<String> = p.densities.iter().map(|d| d.to_string()).collect();

    assert_eq!(block_of, csv(&want["block_of"]), "block_of");
    assert_eq!(sizes, csv(&want["sizes"]), "sizes");
    assert_eq!(densities, csv(&want["densities"]), "densities");
    assert_eq!(p.global_density.to_string(), want["global"], "global");
    assert_eq!(p.defect.to_string(), want["defect"], "defect");
    assert_eq!(p.concentration.to_string(), want["concentration"], "concentration");
    assert_eq!(p.iterations.to_string(), want["iterations"], "iterations");
}

#[test]
fn coarsening_matches_python_and_preserves_mass() {
    let (n, k, epsilon, max_iters) = params();
    let e = edges();
    let want = expected();
    let p = partition(n, &e, k, epsilon, max_iters).expect("valid fixture graph");
    let sup = coarsen(n, &e, &p.block_of);
    assert_eq!(sup.len().to_string(), want["super_edges"], "super-edge count");
    let total: u32 = sup.iter().map(|&(_, _, w)| w).sum();
    assert_eq!(total as usize, e.len(), "aggregation must preserve mass");
    assert!(sup.windows(2).all(|w| w[0] < w[1]), "super-edges sorted and deduped");
}
