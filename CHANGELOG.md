# Changelog

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[Semantic Versioning](https://semver.org/).

## [0.30.0] — 2026-11-17

### Changed
- `RingMemNet` keeps an `occupied` index array, so reading a nearly empty ring
  walks the slots that exist instead of branching over 64 that do not. The
  answer is unchanged -- every sink value in `examples/variance_slots.rs` is
  identical before and after -- only the work is:

| slots occupied | before | after | speedup |
|---:|---:|---:|---:|
| 1 | 42.40 ns | **4.19 ns** | 10.1× |
| 3 | 57.62 ns | **15.83 ns** | 3.6× |
| 8 | 105.29 ns | **47.19 ns** | 2.2× |
| 16 | 149.54 ns | **93.42 ns** | 1.6× |
| 32 | 247.96 ns | **184.94 ns** | 1.3× |
| 64 | 419.58 ns | **382.94 ns** | 1.1× |

  Fixed-capacity and allocation-free is preserved: a second `[u16; 64]` and a
  length, not a `Vec`.

### Added
- `docs/ULTRABACKYARDLOOP.md` and `docs/lap-ledger.jsonl` -- the DO loop
  (reflect → improve → wire → polish → push → readme) written down, with the
  gate a lap must pass and both standing measurements. Every earlier lap had
  skipped the readme step; this is that omission closed.

### Internal
- `cargo clippy --all-targets -- -D warnings` is clean for the first time:
  `manual_contains` in `pqq`, `needless_range_loop` in `trit`/`world`,
  `cloned_ref_to_slice_refs` in the `mesh` tests, `assertions_on_constants` in
  the `observer` constants test (now a range check, which is what it always
  meant), and `manual_is_multiple_of` in the mesh fixture reader.

## [0.29.0] — 2026-11-16

### Added
- `pack_ternary` / `unpack_ternary` / `ternary_dot_packed` — eight ternary lanes
  in one machine word, dotted without multiplying. `+1 * +1` and `-1 * -1` are
  `+1`, a mismatched sign is `-1`, a zero lane is `0`; the whole product is two
  AND masks per outcome and two popcounts. Portable (no NEON, no `x86`), exact,
  and the twin of `neon::dot_i8` for the sizes that fit in a word.
- `[packed]` fixture section and Rust parity, plus 7 Python tests. 26 trit
  tests now; 237 total.
- Demo step `[9]` prints a packed dot next to the naive sum so the two can be
  compared by eye.

### Fixed while writing it
- The obvious bit reading is wrong: `-1` is `0xFF`, whose lowest bit is set, so
  `a & LANE_ONE` reads every `-1` lane as a `+1` as well. The negative lanes
  have to be cleared before a lane counts as `+1`. Found by fuzzing the first
  version against the naive sum (11 376 of 20 000 cases wrong), then pinned by
  `test_packed_dot_handles_the_mixed_sign_case_that_breaks_naive_bits`.

### g8t
- `ternary_dot_packed` capsule published: 44 asm instructions, zero floats,
  zero unsafe, equivalence crown, **+400 FKN → 13 capsules / 6250 FKN**.

### g8t — cycle-golf bounty board
Two boss targets taken from the board, both modulo loops that pay for a real
divide on every round:

| capsule | was | now | asm | crown |
|---|---:|---:|---:|---|
| `gcd_binary_u32` | 288 cyc/op | **147** | 27 | yes (+400 FKN) |
| `gcd_binary_i64` | 677 cyc/op | **272** | 53 | yes (+400 FKN) |

Stein's binary GCD: strip the shared power of two, then subtract and discard
factors of two -- no divide, and the larger operand strictly decreases each
round, so it terminates. Both digests matched the board's equivalence classes
exactly (`eb3e1ec8f830…`, `cdce9b8cd667…`), and both were checked against the
naive loop on 200 000 random pairs plus the edge cases first: `gcd(0,0)=0` for
`u32`, `gcd(0,0)=1` and `saturating_neg` sign handling for `i64`.

Deliberately **not** attempted: the `i32` target. Its reference loop uses
Rust's truncating `%`, so the sign of the result depends on the sequence of
remainders (`gcd(-6,-4)` is `-2`, not `2`); a magnitude-based binary GCD cannot
reproduce that, and faking the equivalence class would be worse than leaving
the bounty. → 15 capsules / 7050 FKN.

## [0.28.0] — 2026-11-16

### Added
- `ObserverFrame.world_of(local_tick)` — the smallest world tick whose reading
  is at least `local_tick` (ceiling division, so negative ticks work). Exact
  for readings that actually happened; a reading that never happened is
  detectable, because it answers with a tick that reads past it. A frozen frame
  (`rate_q16 == 0`) refuses with `ObserverError::FrozenClock` rather than
  inventing a tick.
- `RingMemNet.variance_world` / `consensus_world` — the rate-free path. Decay
  consumes world ticks only, so two nodes at the same world tick compute the
  same integer offline, with no clock authority anywhere. `local_variance`
  keeps the rate-scaled reading under a name that cannot be mistaken for a
  verdict.
- `examples/world_vs_tick.rs` — closed form versus a per-tick decay loop over a
  full 64-slot ring: **137×** slower at `now=400`, **1650×** at 4000, **5337×**
  at 40000, and the loop drifts **-4.8% … -6.4%** against the closed form
  because it rounds every tick.
- `tests/test_sync.py` (12 tests) and `[world]` / `[world_of]` fixture sections
  with Rust parity, so the contract is pinned on both sides.

### Fixed
- `decay_pade` and `RingMemNet::variance` wrapped on astronomical `elapsed`:
  `elapsed << 16` flipped sign and reported "barely decayed" for a band that
  had aged out completely. Now widened to `i128`, clamped to the Padé domain,
  and pinned by two fixture rows. The Python side was already correct (big
  ints), which is why the fixtures had never caught it.

### Not copied from the source sketch
- The `active_mask` / `anchor_mask` bitmask. It optimises a per-tick loop, and
  the measured cost of having that loop at all is 137×–5337× plus 5–6% drift.
  A mask can remove work *from* the loop; it cannot make the loop shorter than
  not having it. The anchor skip it was reaching for is already a branch.
- `verdict()` returning `+1` whenever any anchor is active. Replaced by
  `consensus_world`, the sign of the weighted sum: an anchor that has been
  outvoted contributes its amplitude and still loses.

## [0.27.0] — 2026-11-16

### Added
- `ultragraph.zerocpu.meshd` — the socket path around `mesh.py`: `MeshListener`
  (bound loopback by default, ephemeral port by default), `announce()`,
  `poll()`, `listen()`, and `counts` for accepted vs rejected datagrams. The
  Python twin of `bin/peers.rs`; not re-exported from the package root, because
  a socket is runtime, not a deterministic primitive.
- `tests/mesh_socket.rs` — a real `sendto`/`recvfrom` round trip on
  `127.0.0.1:0`, so the announce frame is now pinned end-to-end in CI instead of
  only as bytes. Junk datagrams are asserted refused, not merged.
- `bin/peers.rs` gained optional destination (`announce`) and bind (`listen`)
  arguments, so a loopback run no longer means editing the source.
- `tests/test_mesh_socket.py` — six tests including a subprocess CLI run.

### Not copied from the source sketch
- An x86 `asm!` decay block: this tree targets aarch64, `shrd` is a *logical*
  shift, `i32` would break Q16.16 parity with Python, and `(v * 62339) >> 16`
  already compiles to two instructions — the asm is slower, unportable and
  unchecked. `decay_q16` stays a plain `i64` function.
- `assert!` on a zero denominator inside a `Fraction::new`: a panic in Rust is a
  process death, not a refusal, and an *empty* ledger has no ratio to report at
  all. This repo keeps the "no claim" case as a total function.

## [0.26.0] — 2026-11-16

### Added
- Kleene ternary logic in `zerocpu.trit` + Rust `zerocpu::trit`: `tnand(a, b) =
  -min(a, b)`, `tkand`, `tkor`, `tknot`, `trit_of`.  `1 NAND 1 = -1`, `-1 NAND x
  = +1`, and `0` propagates — an undecided operand can never be resolved into a
  definite answer.  Commutative, associative, branch-free, and validated:
  non-trits are rejected instead of silently coerced (`tknot(1.0)` is a
  `TypeError`, not `-1`).
- `pade_exp_neg(x_q16)` — `e^-x` via a `[3/3]` Padé approximant, integer-only:
  split `x = k*ln2 + r`, then `e^-x = 2^-k * P(-r)/P(r)` with `P(t) = 120 + 60t +
  12t^2 + t^3`.  No loop and no `exp()`, so `e^-1` is exactly `24109` on every
  machine; agreement with the analytic value is within `2e-5` (16-bit
  quantization, not a float error).
- `decay_pade(variance, elapsed, half_life)` — the same half-life as `decay()`
  but closed-form: any `elapsed` costs the same few multiplications, so a
  decade-long gap cannot become a 4096-step loop.  `half_life <= 0` is an error.
- `RingMemNet` (Python + Rust) — 64 slots, `MEMNET_RING`, `RingSlot` with a
  `.trit` property, `record(amp_q16, born_tick, anchor=)`, `variance(now,
  half_life, rate_q16)`, `consensus(...)`.  Two fixes over the sketch: a ring
  full of anchors returns `None` rather than silently dropping the write while
  still returning an id, and placement scans from the cursor to the first free
  or non-anchor slot so eviction order is a function of the writes.
- `examples/caveman_ultra_demo.py` gained step `[7]`: the guard, `e^-1`, and a
  ring whose variance collapses from `131072` to `196608` while the anchored
  memory holds the verdict at `+1`.
- Fixtures gained `[kleene]`, `[pade]`, `[pade_decay]`, `[ring]` and
  `[ring_expect]` sections; Python suite 212 tests, Rust 69 across 13 binaries.

### Not copied from the source sketch
- A separate `Trit` enum: a trit here is `i8`, and the Kleene ops were added to
  the existing module rather than beside it.
- `>> k.min(32)` (a shift by a signed value) and the 40-nat guard read as a
  weight: `PADE_X_MAX` means "below one ulp", and the shift amount is bounded by
  construction.
- `f32` anywhere in the decay path.

## [0.25.0] — 2026-11-16

### Added
- `ultragraph.zerocpu.observer` + Rust `zerocpu::observer` — observer frames:
  `local = (world_tick - origin_world) * rate_q16 >> 16`, exact and integer, so
  two nodes can print different "now"s for one world tick and still agree on the
  structure.  `relativity()` returns both readings and the delta.
- `decay()` — the variance half-life, integer-only: `v * 62339 >> 16` per tick,
  bounded to 4096 iterations, idempotent at a fixed tick.  `exp()` drifts
  between platforms; a rational recurrence cannot.
- `Mahakala` — a guard over a caller-supplied chain of names.  Whole-token,
  case-insensitive, in chain order, and it reports what it matched: `"seasonal"`
  does not summon `"son"`.
- `Witness` — memory ranked by weight, with `anchor=True` for what must not fade.
  Weight is recomputed from `(born_tick, world_tick)` instead of mutated in
  place, so re-ticking a timeline cannot accumulate drift.
- `examples/caveman_ultra_demo.py` gained step `[6]`: two frames, the guard, and
  an anchor outliving a plain memory (163850 → 22155 while the anchor stays
  6553600).
- `observer_vectors.txt` + `tests/observer_parity.rs`; Python suite 198 tests,
  Rust 66 across 13 binaries.  94 names exported from `ultragraph.zerocpu`.

### Notes
- No `ln()`-derived time dilation: a frame's rate is an explicit Q16.16 value the
  caller sets.  Deriving it from the input's depth was the sketch's idea; it makes
  the clock a function of the prompt, which is not a clock.
- No keyword sniffing for monuments.  `Witness.remember(..., anchor=True)` makes
  the caller declare meaning; a substring check would make `"Taj Mahal"` sacred
  and every translation of it profane.
- Ordering: with linear rates the order of two world ticks is frame independent —
  only labels differ.  Genuine disagreement needs spacelike separation, which is
  `SpacetimeGraph.lightcone`'s job.

## [0.24.0] — 2026-11-16

### Added
- `ultragraph.zerocpu.trit` + Rust `zerocpu::trit` — the `A6P-15A` quantizer: a
  Q16.16 magnitude is thresholded against a 15-step delta ladder, and the result
  is also emitted as 16 magnitude planes (`bitplanes` / `from_planes` /
  `plane_parity`) so a trit vector is a popcount away from NEON. Levels, deltas
  and planes are pinned by `trit_vectors.txt`.
- `ultragraph.zerocpu.router` — deterministic route selection from six integer
  counts plus **position-weighted** heat (`(ord*31 + i) * (i + 1)`), so `"ab"` and
  `"ba"` are different prompts. Five routes, `explain()` for the reasoning, and a
  `python3 -m ultragraph.zerocpu.router` JSON entry point.
- `ultragraph.zerocpu.memnet` — 256-slot associative memory keyed by FNV-1a and
  ranked by `(seq, value)`, never by clock: `merge` is commutative, associative
  and idempotent, which is what lets two nodes sync without asking who is right.
- `ultragraph.zerocpu.mesh` — `UGRD` v1 announce frame (`<4sBHIB`, 12-byte fixed
  part, 32-byte UTF-8 name), a parser that returns `None` instead of raising, and
  the peer merge rule. `bin/peers.rs` speaks it over UDP (stdlib only, not in CI).
- `examples/caveman_ultra_demo.py` — one command: route → quantize → world tick →
  memory merge/recall → mesh peers → digest.
- Parity fixtures `trit_vectors.txt`, `router_vectors.txt`, `memnet_vectors.txt`,
  `mesh_vectors.txt` generated from the Python modules, replayed by
  `tests/{trit,router,memnet,mesh}_parity.rs`. Python suite: 181 tests; Rust: 59.
- 82 names exported from `ultragraph.zerocpu`.

### Notes
- Heat is position-weighted on purpose: an order-blind hash made anagram prompts
  collide, and a router that cannot tell `"ab"` from `"ba"` is not a router.
- `memnet` ranks writes by sequence, not by timestamp: two nodes with unsynced
  clocks must still agree after a merge.
- Rust `router::features` counts `char`s, not bytes, to stay bit-identical with
  Python on non-ASCII prompts.

### Fixed
- `mesh::merge_peers` keeps the incoming peer when `seq >= existing.seq`, matching
  the Python rule exactly (tie goes to the announcer).

## [0.23.0] — 2026-11-16

### Added
- `ultragraph.zerocpu.hope` + Rust `zerocpu::hope` — a text is projected onto `n`
  lanes as signed Q16.16 heat (position-salted bucket), centered by the per-lane
  mean, and reduced to ternary trits. `resonate()` seeds the world, ticks, and
  returns the full state trace plus an `Observer` collapse; propagation is the
  ordinary `tick`, so nothing new can drift. Parity pinned by
  `hope_signal.txt` / `hope.expected.txt`.

### Notes
- No floats and no keyword tables: `"hope"` and `"Shawshank"` are inputs like any
  other. Centering is what gives a long text shape — uncentered, every lane drifts
  to `+1` and the projection carries no information.
- Deliberately not ported from the sketch: `f32` variance (breaks the
  integer-only contract), hard-coded `nodes.get_mut(&1)` state hacking (a special
  case for one word is not a kernel), and per-word narration in place of state.

## [0.22.0] — 2026-11-16

### Added
- `ultragraph.zerocpu.pqq` — `.pqq` compiled-program artifacts: `resolve()` settles every
  `q P/Q` draw once (compile time), `dump_pqq()`/`load_pqq()` round-trip the text, and
  `run_pqq()` replays it with no PRNG, no seed, and per-line error reporting.
  `write_bundle()` emits the `.ugmz` + `.pqq` pair from one prompt.
- Rust `zerocpu::world` — the `UGMZ` v1 parser/writer, `tick`/`tick_n`, and the same
  edit semantics as `SpacetimeGraph` (`add_edge`, `flip`, `prune`, `set`).
- Rust `zerocpu::pqq` — `parse`/`run` for compiled programs, rejecting unresolved draws.
- `replay` binary (`replay <world.ugmz> <prog.pqq> [out.ugmz]`); `world` binary now
  shares the library parser instead of carrying its own copy.
- Cross-language replay fixture (`replay.pqq` + `replay.expected.txt`): Rust replays the
  program to byte-identical state, tick count and edge count as Python.

### Published kernels
- g8t capsules `trit_at_u8` (`(u8,u32)->i8`, 5-cycle lane extract), `saturate_trit_i64`
  (`(i64)->i8`, 5 instructions) and `fx_sqrt_q16` (`(i64)->i64`, exact integer isqrt) —
  +1400 FKN, all equivalence crowns. `trit_at_u8` was rejected once for an unbounded
  index loop; capsules must terminate for any harness input.

### Notes
- `.pqq` deliberately carries outcomes, not a seed: reproducing CPython's MT19937 draw
  order in another language would be a drifting surface. Compile-time resolution keeps
  replay a literal transcription — the property worth guaranteeing.
- Apply-time failures now surface as `PqError` with the offending line, in both languages.

## [0.21.0] — 2026-11-15

Deterministic compute with a Rust twin: every number produced by this release is
identical on every platform, and reproducible from committed golden vectors.

### Added
- **`ultragraph.zerocpu`** — integer-only kernels with no float in any
  deterministic path. `fixed` (Q16.16, round-half-away-from-zero, symmetric about
  zero), `spacetime` (`SpacetimeGraph`: ternary causal edges with `dt` delay,
  `tick`/`untick` with history, light-cone queries, sha256 fingerprint), and
  `evolve` (a `p/q` prompt language — `edge`, `flip`, `prune`, `set`, `step`,
  `wire` — with exact-rational acceptance and a human-readable report).
- **`emit`** — the `UGMZ` artifact format (v1): 32-byte header, fixed-width
  24-byte node/edge records, little-endian, byte-identical across runs and
  across Python/Rust. Sibling of `.ugm`, not a replacement.
- **`vaked_bridge`** — a lowered Vaked workflow (`gen/workflow/wf.json`) becomes a
  spacetime world, so a `.vaked` source compiles straight to `.ugmz`
  (`lower_vaked`, `vaked_to_ugm`). Honours `VAKED_ROOT` and fails with a clear
  message rather than importing the broken vendored compiler.
- **`lod`** — quasi-regular partitioning for coarse-graining large worlds:
  calibrated clustering in Q16.16 that maximises structure (`concentration`) and
  *reports* the residual quasi-randomness (`defect`) instead of asserting
  regularity. `partition`, `coarsen`, `density_stats`.
- **Rust crate `rust/zerocpu`** (standalone, not shipped in the wheel): `fixed`,
  `ternary` (5 trits/byte, dot products over `i8` and ternary weights), `neon`
  (`vmull_s8`/`vpadalq_s16` path on aarch64, scalar elsewhere), `lod` mirroring
  the Python compressor, and a `world` binary that ticks a `.ugmz` artifact.
  Cross-language parity is enforced by committed fixtures, so the two
  implementations cannot drift silently.

### Notes
- Saturating at `int8` is not bijective, so the honest claim is *memory-based*
  reversibility (`tick(remember=True)` / `untick`), not unitary evolution.
- The vendored `ultragraph/vakedc/` remains unusable upstream (de-indented);
  `VAKED_ROOT` is the supported path to a working compiler.

## [0.20.0] — 2026-07-17

### Fixed
- **WASM `tanh`** — the tanh activation expression emitted a literal `{acc}` placeholder
  instead of the accumulator local, producing invalid WAT for any tanh-activated tree.
- **RP2040 codegen** — the generated forward pass reused one buffer
  (`tree_forward(buf, buf)`), so input/output aliasing corrupted every tree after the
  first; the buffer was also sized only from the first/last tree and overflowed on a
  wider intermediate tree. Now uses ping-pong buffers sized to the widest tree.
- **`Dropout`** — `p=1.0` divided by zero (`NaN`); now returns zeros.
- **File handles** — `io.load` / `load_params` and `GPT.load_deployed` / `load_model`
  now close the `np.load` `NpzFile` (previously leaked).
- **`ugm.load_ugm`** — validates the format version and guards the history-segment
  reshape against non-divisible lengths (previously a hard crash on malformed files).
- **`linker`** — dropped a stray `header=None` override.
- **Viz** — matplotlib figures are closed on error; generated WebGL guards division by
  zero (aspect ratio and vector normalize).

### Changed
- **`GPT.save` now embeds hyper-parameters**, so `GPT.load_model` reconstructs any
  architecture. It previously hardcoded `GPT(256, 128, 2, 2)` and only that shape
  loaded. **Legacy checkpoints saved without metadata now raise a clear error** instead
  of silently loading into the wrong shape — re-save them with `GPT.save`.
- **RP2040 codegen raises `NotImplementedError`** on residual/merge graphs instead of
  silently emitting wrong C (use the WASM backend for those topologies).

### Security
- **CLI** — report HTML now escapes model-generated text and names; `search` URL-encodes
  the query; `pull` rejects non-`http(s)` URL schemes.
- **Wiki client** — validates the language code before building the API URL; uses
  SHA-256 (not SHA-1) for cache-key hashing.

## [0.12.0] — 2026-07-11

### Added
- **Gradient accumulation** — `SGD` / `Adam` take `accum_steps=n`: accumulate grads
  over `n` `step()` calls, then apply one averaged update and auto-zero (train a large
  effective batch on a small memory budget). `accum_steps=1` is unchanged.
- **`GPT.generate_batch`** — batched decoding over `[B, T]` (equal-length) prompts on
  one shared KV-cache, with per-sequence sampling and per-sequence `stop`/EOS. Returns
  a list of `B` id lists.
- **`Mesh.generate`** — joint autoregressive decoding across the mixture: each expert
  keeps its own KV-cache; the router gate (re-evaluated as the sequence grows) mixes
  their next-token logits before sampling. Supports `top_k`/`top_p`/`repetition_penalty`/`stop`.
- **`examples/mesh_lm.py`** — a byte-level `Mesh` of GPT experts trained with gradient
  accumulation, plus joint `Mesh.generate` and batched generation.

## [0.11.0] — 2026-07-11

### Added
- **`model.Mesh`** — a graph of minds: a learned soft (or top-k) mixture of *full*
  models. Each expert maps token ids to logits; a small ternary router mixes them
  per sequence (`logits = Σ_e gate(ids)_e · expert_e(ids)`). `nn.MoE`'s routing
  lifted to whole networks — fully differentiable, router + experts train together.

### Fixed
- `Mesh` and `nn.MoE` now reject `top_k` outside `[1, n_experts]` (a non-positive
  `top_k` previously zeroed the whole mixture silently).

## [0.10.0] — 2026-07-11

### Added
- **Deployed inference from ternary bytes** — a dense `Tree` with no fp32 master now
  runs `forward` straight from its stored ternary weight bytes (`autograd.ternary_forward`,
  shared numerics with `ternary_linear` → byte-exact). `Tree.deployed` reports the mode.
- **`GPT.save_deployed` / `GPT.load_deployed`** — an inference-only checkpoint holding
  the bit-packed ternary weights (~1.6 bits/weight) plus the small fp32 pieces
  (embedding, norm gains, biases). **~10× smaller** than the fp32-master `save`
  (e.g. 3.4 MB → 334 KB on an 858k-param model) and byte-exact at inference.
- **`GPT.n_params()`** — trainable scalar count.

### Changed
- `Tree.parameters()` / `requantize()` no-op cleanly for a master-less (deployed) tree.

## [0.9.0] — 2026-07-11

### Added
- **`stop` tokens** — `GPT.generate(..., stop=id_or_ids)` halts as soon as a stop
  token is produced (emitted, then generation ends).
- **Repetition penalty** — `GPT.generate(..., repetition_penalty=…)`: the CTRL rule
  over the running history (`> 1` discourages already-seen tokens, `< 1` encourages).
- **`examples/gpt_lm.py`** — the whole stack end to end: `ByteTokenizer` → `GPT`
  (RoPE + KV-cache) → Adam → re-quantized ternary → streaming generation and a
  save/load round-trip, on a byte-level toy corpus.

## [0.8.0] — 2026-07-11

### Added
- **`GPT.save` / `GPT.load`** — persist and restore a model's fp32 parameters
  (byte-exact inference after re-quantize). Rebuild a `GPT` with the same
  hyper-parameters, then `.load(path)`.
- **Streaming generation** — `GPT.generate(..., stream=True)` returns a generator
  that yields each new token id as it is produced (drives token-by-token demos).
- **Nucleus sampling** — `GPT.generate(..., top_p=…)`: keep the smallest set of
  tokens whose cumulative probability reaches `top_p`, then renormalize. Composes
  with `top_k` and `temperature`.

## [0.7.0] — 2026-07-11

### Added
- **`nn.RoPE`** — rotary positional embeddings (no learned params). Encodes
  *relative* position in the attention dot-product and takes an `offset=` so it
  composes with a KV-cache; drop it into `MultiHeadAttention(..., rope=RoPE(...))`.
- **KV-cache** — `MultiHeadAttention(x, cache={'k':…,'v':…})` returns
  `(out, new_cache)` for incremental decoding. Because activation quant is
  per-token, a cached step is byte-for-byte the full-forward result at that
  position (see `tests/test_gpt.py`).
- **`autograd.cat`** — concatenate `Tensor`s along an axis; grad splits back to
  each input. The primitive behind RoPE's `rotate_half` and the KV-cache.
- **`model.GPT` / `model.TransformerBlock`** — a ternary GPT (token embedding +
  shared RoPE + N pre-norm blocks + ternary head) with `.generate(prompt, n_new,
  temperature=, top_k=, seed=)` autoregressive sampling over the KV-cache.

## [0.6.1] — 2026-07-11

### Changed
- Split `viz.py` (491 loc) into a `viz/` package — `viz/svg.py` (pure-SVG) +
  `viz/mpl.py` (optional matplotlib). Public API (`ultragraph.viz.*`) unchanged.
  Every module is now ≤ ~340 loc (readability/maintainability).

### Fixed
- Re-indented the vendored `vakedc` `parser.py`/`graph.py`/`emit.py` so they compile.
  Documented that upstream vakedc is de-indented **and lossy** (missing content), so
  `compile_vaked` still needs the original vakedc; `lower_graph` works standalone.

## [0.6.0] — 2026-07-11

### Added
- **`nn.Sequential`** — chain modules/callables; aggregates parameters, propagates
  `train()`/`eval()` (e.g. to Dropout), and re-quantizes ternary submodules.
- **Callable `Tree`** — `tree(x)` == `tree.forward(x)`, so trees compose in Sequential.
- **`io.save_params` / `io.load_params`** — persist the fp32 parameters of any module
  list (Attention, MoE, RMSNorm/LayerNorm, Embedding, Tree), closing the gap where
  graph save/load did not serialize those modules.
- **Optional vaked lowering** (`ultragraph.vaked`) — `lower_graph(nodes, edges)`
  lowers any labeled property graph into a sparse byte-graph `Tree`; `compile_vaked`
  bridges the vendored vakedc front-end (optional — requires an importable `vakedc`).

## [0.5.0] — 2026-07-11

### Added
- **Activations** — `Tensor.exp` / `tanh` / `sigmoid` autograd ops, plus
  `Tensor.gelu()` and `Tensor.silu()` (composed from primitives).
- **Cosine LR schedule** — `optim.CosineSchedule(opt, total_steps, warmup=…)`:
  linear warmup then cosine decay; call `.step()` once per optimizer step.

## [0.4.0] — 2026-07-11

### Added
- **Top-k routing for `nn.MoE`** — `MoE(..., top_k=k)` keeps only the top-k experts
  per token (gates renormalized); the full soft mixture stays the default.
- **`nn.Dropout`** — inverted dropout (rescales by `1/(1-p)`); passthrough when
  `training=False`.
- **Weight decay** — `weight_decay=` on `SGD` and `Adam` (L2).
- **Autograd** — `Tensor.sum(axis=, keepdims=)`.

## [0.3.0] — 2026-07-11

### Added
- **Mixture of experts** (`nn.MoE`) — a soft mixture of ternary-MLP experts with a
  learned router: `output = Σ_e softmax(router(x))_e · expert_e(x)`. Router and
  experts are dense ternary trees; fully differentiable.
- **Byte-level tokenizer** (`ByteTokenizer`) — lossless UTF-8 byte tokenization,
  vocab 256, no training or OOV. `examples/moe_lm.py` demos a byte-level MoE LM.
- **Autograd slicing** — `Tensor.__getitem__` (basic indexing) with a scatter
  backward, enabling per-expert gating.

## [0.2.1] — 2026-07-11

### Fixed
- `ultragraph.__version__` now derives from installed package metadata
  (`importlib.metadata`) instead of a hardcoded string, so it can no longer drift
  from the released version.

## [0.2.0] — 2026-07-11

### Added
- **Dense ternary bit-packing** (`pack.py`) — `pack_ternary` / `unpack_ternary`
  store 5 ternary values per byte (base-3), the true ~1.58-bit density (1.6
  bits/weight, a 5× shrink over one-byte-per-weight). `io.save(..., packed=True)`
  writes dense weights packed; `load` unpacks transparently (byte-exact).
- **Learned positional embeddings** (`nn.LearnedPositionalEmbedding`) — an fp32
  position table added to a sequence, with gradients flowing only to used
  positions.

## [0.1.0] — 2026-07-11

The byte-graph that is a 1-bit (ternary) LLM.

### Added
- **Core** (`core.py`) — `Node`/`Edge`/`Tree`/`UltraEdge`/`UltraGraph`, the byte
  buffers, the ad-hoc side store, and the dunder API (`node >> node` micro-edge,
  `tree >> tree` ultra-edge, `len`/`iter`/`in`, `_repr_svg_`).
- **Quantization** (`quant.py`) — ternary weights (absmean) + int8 activations
  (absmax), with the small-`eps` / finite guards.
- **Autograd** (`autograd.py`) — n-D reverse-mode tape: `+ - * / @`, `transpose`,
  `swapaxes`, `reshape`, `relu`, `sqrt`, `sum`, `mean(axis)`, `softmax`,
  `cross_entropy`; and `ternary_linear` with a straight-through estimator and
  per-token int8 activation quantization.
- **nn** (`nn.py`) — `linear_tree`, `mlp`, single-head `Attention`, batched
  `MultiHeadAttention` (causal), `RMSNorm`, `LayerNorm`.
- **Optimizers** (`optim.py`) — `SGD` and `Adam`, both with global-norm gradient
  clipping; re-quantize the ternary buffers after each step.
- **Visualization** (`viz.py`) — pure-SVG micro/macro/byte-heatmap views plus an
  optional matplotlib backend (`[viz]` extra).
- **I/O** (`io.py`) — byte-exact save/load.
- **Examples** — `char_lm.py`, `transformer_lm.py`, `mini_gpt.py` (batched
  multi-head transformer that memorizes a toy corpus), `render_viz.py`,
  `make_figures.py`.
- **Tests** — numeric-gradient checks for every op, an attention causality proof,
  save/load byte-exactness, and end-to-end training; a dependency-free runner.
- **Tooling & docs** — `ruff` (config in `pyproject.toml`), GitHub Actions CI
  (`ruff` + `pytest` on Python 3.11–3.13), `CONTRIBUTING.md`, `CHANGELOG.md`, and
  `docs/references.md` (an Erdős graph-theory reading list).

[0.12.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.12.0
[0.11.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.11.0
[0.10.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.10.0
[0.9.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.9.0
[0.8.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.8.0
[0.7.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.7.0
[0.6.1]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.6.1
[0.6.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.6.0
[0.5.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.5.0
[0.4.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.4.0
[0.3.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.3.0
[0.2.1]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.2.1
[0.2.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.2.0
[0.1.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.1.0
