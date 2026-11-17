//! Cross-language parity: the compiled `UGMZ` world must tick exactly like Python.
//!
//! `tests/fixtures/world.ugmz` and `tests/fixtures/world.expected.txt` are produced
//! by `ultragraph.zerocpu.emit` (Python). The Python test `test_ugmz_fixture_is_stable`
//! regenerates both and asserts they are byte-identical, so this test cannot pass
//! against a stale fixture: if either side's semantics drift, one of the two fails.

use std::process::Command;

const FIXTURE: &str = "tests/fixtures/world.ugmz";
const EXPECTED: &str = "tests/fixtures/world.expected.txt";

#[test]
fn world_binary_matches_python_golden() {
    let expected = std::fs::read_to_string(EXPECTED).expect("expected fixture present");
    let out = Command::new(env!("CARGO_BIN_EXE_world"))
        .args([FIXTURE, "3"])
        .output()
        .expect("world binary runs");
    assert!(
        out.status.success(),
        "world exited {:?}: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let got = String::from_utf8(out.stdout).expect("utf-8 output");
    assert_eq!(got.trim_end(), expected.trim_end());
}

#[test]
fn zero_ticks_reproduces_the_compiled_state() {
    let expected = std::fs::read_to_string(EXPECTED).expect("expected fixture present");
    let out = Command::new(env!("CARGO_BIN_EXE_world"))
        .args([FIXTURE, "0"])
        .output()
        .expect("world binary runs");
    assert!(out.status.success());
    let got = String::from_utf8(out.stdout).expect("utf-8 output");
    let mut lines = got.lines();
    let header = lines.next().unwrap_or_default();
    assert!(header.contains("start_tick=0 ticks=0 tick=0"), "{header}");
    // positions never change: the compiled pos line must equal the post-tick one
    let pos_expected = expected
        .lines()
        .find(|l| l.starts_with("pos="))
        .expect("pos in fixture");
    assert!(
        lines.any(|l| l == pos_expected),
        "positions must be tick-invariant"
    );
}

#[test]
fn rejects_garbage_artifact() {
    let dir = std::env::temp_dir().join("zerocpu-world-bad");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("bad.ugmz");
    std::fs::write(&path, b"UGMZnot-an-artifact").expect("write");
    let out = Command::new(env!("CARGO_BIN_EXE_world"))
        .arg(&path)
        .output()
        .expect("world binary runs");
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("bad artifact"));
}
