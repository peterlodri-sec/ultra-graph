# ULTRABACKYARDLOOP

One lap, end to end, with no loose ends between laps. Every lap leaves the tree
clean, pushed, and written down. A plausible branch may be ranked; only a
verified branch may be bound.

```
DO = reflect → improve → wire → polish → push → readme
```

This file is the readme step. `docs/lap-ledger.jsonl` is the machine-readable
half of the same record: one line per lap, with the commit and the gate.

---

## 1. Reflect

What is actually true, rather than what the last lap intended.

`ultra-graph` is a **deterministic, integer-only** substrate: every kernel has a
Python reference and a Rust twin, and the two are pinned to each other by golden
vectors generated from the Python side and replayed by the Rust tests. No
`f32`/`f64` exists on any deterministic path. That constraint is the product;
everything else is in service of it.

The load-bearing pieces, and what each one is for:

| module | what it decides | twin |
|---|---|---|
| `trit.py` | ternary quantisation, A6P-15A bitplanes, Kleene logic, SWAR packed dot | `ternary.rs`, `trit.rs` |
| `fixed.py` | Q16.16 fixed point: add, sub, mul, div, sqrt | `fixed.rs` |
| `observer.py` | frames, closed-form Padé decay, Mahakala guard, witness | `observer.rs` |
| `memnet.py` | 256-slot associative memory + 64-slot ring with anchors | `memnet.rs` |
| `mesh.py` / `meshd.py` | UGRD v1 announce frame, merge, loopback socket | `mesh.rs`, `bin/peers.rs` |
| `router.py` | five routes over position-weighted heat | `router.rs` |
| `hope.py` | text → integer heat → world ticks | `world.rs`, `bin/world.rs` |
| `spacetime.py` | causal graph, lightcones, simultaneity disagreement | `spacetime.rs` |
| `pqq.py` | compiled `.pqq` replay bundles | `pqq.rs`, `bin/replay.rs` |

The honest inventory of what is **not** here is as important as what is:

- `ultragraph/vakedc/` is vendored and broken without `VAKED_ROOT`.
- `dot_i8` (NEON) has no g8t capsule: the harness pins scalar signatures and
  compiles x86-64, so a slice kernel can never be submitted. The SWAR packed dot
  is the answer for the sizes that fit in a word.
- No capsule for `i32` gcd: the reference uses Rust's truncating `%`, so the
  sign of the result depends on the remainder sequence (`gcd(-6,-4)` is `-2`).
  A magnitude-based binary GCD cannot reproduce that, and faking the class would
  be worse than leaving the bounty.

## 2. Improve

A lap must move a number. Two standing measurements, both re-runnable:

```bash
cd rust/zerocpu
cargo run --release --example world_vs_tick     # closed form vs per-tick loop
cargo run --release --example variance_slots    # sparse ring read cost
```

**`world_vs_tick`** — why the ring is closed-form and carries no bitmasks. A
per-tick decay loop over a 64-slot ring, against one evaluation at `now`:

| at `now` | closed form | per-tick loop | ratio | drift |
|---:|---:|---:|---:|---:|
| 400 | 936 ns | 128 530 ns | 137× | −4.8% |
| 4 000 | 740 ns | 1 222 752 ns | 1 650× | −6.4% |
| 40 000 | 2 302 ns | 12 289 181 ns | 5 337× | −6.4% |

The drift is the loop rounding every tick. A mask can remove work *from* that
loop; it cannot make the loop shorter than not having it, and it cannot remove
the drift.

**`variance_slots`** — a read should cost what the ring holds, not what it can
hold. Lap 9 replaced the branch-over-64 scan with an index array:

| slots occupied | before | after | speedup |
|---:|---:|---:|---:|
| 1 | 42.40 ns | **4.19 ns** | 10.1× |
| 3 | 57.62 ns | **15.83 ns** | 3.6× |
| 8 | 105.29 ns | **47.19 ns** | 2.2× |
| 16 | 149.54 ns | **93.42 ns** | 1.6× |
| 32 | 247.96 ns | **184.94 ns** | 1.3× |
| 64 | 419.58 ns | **382.94 ns** | 1.1× |

Every `sink` value is identical before and after: the answer did not change,
only the amount of work spent producing it.

## 3. Wire

Every surface points at something. A module nobody imports is a defect, and a
function nobody calls in the demo is a claim nobody has checked.

The demo is the wiring test: `examples/caveman_ultra_demo.py` walks each layer
and prints the digest line, so a broken import or a changed invariant fails
visibly rather than silently. Current steps:

```
[1] trit      [2] fixed     [3] router    [4] memnet
[5] observer  [6] mesh      [7] memnet ring + Kleene + Padé
[8] world (rate-free agreement)          [9] swar packed ternary dot
```

## 4. Polish

```bash
python3 tests/run_all.py                        # no pytest required
ruff check ultragraph examples tests            # ruff format is never used
cargo fmt --all -- --check
cargo clippy -- -D warnings
cargo test                                      # CARGO_TARGET_DIR=/tmp/cargo-fin
```

ruff's **line length is 110** and its formatter is deliberately not run: the
diff churn would bury the actual change. `cargo fmt` is run.

## 5. Push

Three remotes, and they are not the same repository:

| alias | URL |
|---|---|
| `origin` | `github.com/8b-is/ultra-graph` |
| `codeberg` | `codeberg.org/8b-is/ultra-graph` |
| `mirror-github` | `github.com/peterlodri-sec/ultra-graph` |

```bash
git push-all     # all three
```

Staging never includes `assets/` or `.slim/`:

```bash
git add -A -- . ':!assets' ':!.slim'
```

Commit identity must be the account that owns the repository, or the push is
rejected: `267054113+peterlodri-sec@users.noreply.github.com`.

## 6. Readme

Update this file, append to `docs/lap-ledger.jsonl`, and touch `README.md` when
the lap changed a public surface. **A lap that stops before readme is a lap that
didn't happen.**

---

## Gate

A lap is verified only when all of these hold at the commit it claims:

1. `python3 tests/run_all.py` → `0 failed`
2. `ruff check ultragraph examples tests` → clean
3. `cargo test` → `0 failed`
4. `cargo fmt --all -- --check` → clean
5. `cargo clippy -- -D warnings` → clean
6. every new golden vector generated from the Python reference, never hand-typed
7. every measurement quoted in this file re-runnable from the command above

Item 6 is the one that catches the most: a hand-written fixture is a guess with
a green check next to it.

## Lap ledger

| lap | commit | version | what moved |
|---:|---|---|---|
| 1 | `0f5adb5` | 0.23.0 | hope projection: text to integer heat to world state |
| 2 | `01162b4` | 0.24.0 | trit quantizer, router, memnet, mesh |
| 3 | `bb9251d` | 0.25.0 | observer frames, Mahakala guard, anchored memory |
| 4 | `700557e` | 0.26.0 | Kleene trits, closed-form Padé decay, ring MemNet |
| 5 | `859bce4` | 0.27.0 | meshd: loopback listener, socket round trip in CI |
| 6 | `91296b3` | 0.28.0 | decay by world tick; rate-free consensus |
| 7 | `918ef67` | 0.29.0 | eight ternary lanes per word, no multiply |
| 8 | `4cc7ff6` | 0.29.0 | two cycle-golf crowns (binary gcd), one refused |
| 9 | `05d9e19` | 0.30.0 | skip empty slots when reading the ring (10.1× at one slot) |

---

*`<3,<3,<3+1` — from love, from within.*