"""Caveman Ultra: one end-to-end pass over the zerocpu primitives.

    python3 examples/caveman_ultra_demo.py "ultra backprop in ternary"

Route -> quantize -> A6P-15A bitplanes -> world tick -> memory -> mesh ->
observer frames.

Integer-only and deterministic: the same prompt produces the same transcript on
every run, on every machine.  Nothing here calls a model, a network socket, or
a float.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python3 examples/caveman_ultra_demo.py` from a checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultragraph.zerocpu import emit, hope
from ultragraph.zerocpu.memnet import MemNet, RingMemNet
from ultragraph.zerocpu.mesh import merge_peers, pack_announce, parse_announce
from ultragraph.zerocpu.observer import (
    RATE_ONE,
    Mahakala,
    ObserverFrame,
    Witness,
    pade_exp_neg,
    relativity,
)
from ultragraph.zerocpu.router import explain, features, route, route_name
from ultragraph.zerocpu.spacetime import SpacetimeGraph
from ultragraph.zerocpu.trit import (
    A6P_LEVELS,
    bitplanes,
    delta_for_level,
    plane_parity,
    quantize_many,
    tnand,
)

LANES = 8
PROTO_PORT = 8423


def _compact(trits: tuple[int, ...] | list[int]) -> str:
    return "".join("+" if t > 0 else ("-" if t < 0 else "0") for t in trits)


def main(argv: list[str]) -> int:
    prompt = argv[1] if len(argv) > 1 else "ultra backprop in ternary"
    print(f"prompt: {prompt!r}")

    feats = features(prompt)
    decision = route(feats)
    print(
        f"\n[1] fit      route={route_name(decision)}"
        f" chars={feats['chars']} words={feats['words']} heat=0x{feats['heat']:04x}"
    )
    for reason in explain(feats):
        print(f"             - {reason}")

    sig = hope.signal(prompt, LANES)
    seen = None
    for level in range(A6P_LEVELS):
        delta = delta_for_level(level)
        trits = quantize_many(sig.heat, delta)
        fired = sum(1 for t in trits if t)
        if seen is None and fired:
            seen = (level, delta, trits, fired)
    if seen is None:
        level, delta, trits, fired = A6P_LEVELS - 1, 1, quantize_many(sig.heat, 1), 0
    else:
        level, delta, trits, fired = seen
    planes = bitplanes(trits)
    parity = plane_parity(planes)
    print(f"\n[2] quantize heat={','.join(str(h) for h in sig.heat)}")
    print(f"             first level that fires: {level} delta={delta} lanes={fired}/{LANES}")
    print(f"             trits={_compact(trits)} planes={len(planes)} parity=0x{parity:04x}")

    world = SpacetimeGraph(name="caveman")
    for _ in range(LANES):
        world.add_node()
    for i in range(LANES):
        world.add_edge(i, (i + 1) % LANES)
    res = hope.resonate(world, prompt, rounds=3)
    print(
        f"\n[3] world    nodes={world.n_nodes} edges={world.n_edges}"
        f" ticks={res.rounds} settle={res.collapse:+d}"
    )
    print(f"             trace={' -> '.join(res.trace)}")
    raw = emit.to_bytes(world)
    print(f"             artifact={len(raw)}B state=0x{emit.state_hex(world)}")

    left = MemNet()
    right = MemNet()
    left.put("fit", decision)
    left.put("parity", parity)
    right.put("collapse", res.collapse)
    changed = left.merge(right)
    print(f"\n[4] memory   merged={changed} entries={len(left)}")
    print(f"             hex={left.to_hex()}")
    recalled = (
        left.get("fit") == decision
        and left.get("parity") == parity
        and left.get("collapse") == res.collapse
        and left.get("missing") is None
    )
    print(f"             recall fit/parity/collapse/missing ok={recalled}")

    peers = []
    frames = (
        pack_announce("hearth", PROTO_PORT, 1),
        pack_announce("thunderbird", PROTO_PORT, 1),
        pack_announce("hearth", PROTO_PORT, 2),
    )
    for frame in frames:
        peer = parse_announce(frame)
        if peer is not None:
            peers = merge_peers(peers, peer)
    names = ",".join(f"{p.name}:{p.port}@{p.seq}" for p in peers)
    print(f"\n[5] mesh     frame={frames[0].hex()} peers={names}")

    # [6] Two frames read the same tick differently; the guard names its chain;
    # a declared monument keeps its weight while a plain memory fades.
    hearth = ObserverFrame("hearth", rate_q16=1 << 15)
    thunderbird = ObserverFrame("thunderbird")
    report = relativity(hearth, thunderbird, world.tick_count)
    guard = Mahakala().raise_guard("Mother, father, son.")
    memory = Witness()
    memory.remember("the stream", 0, span=len(prompt))
    memory.remember("mumtaz mahal", 0, span=1, anchor=True)
    memory.remember("a whisper", 0, span=1)
    print(
        f"\n[6] frames   world={world.tick_count}"
        f" hearth={report.local_a} thunderbird={report.local_b}"
        f" delta={report.delta} guard={','.join(guard) or '-'}"
    )
    print(
        f"    witness  plain {memory.variance('the stream', 0)}"
        f" -> {memory.variance('the stream', 40)}"
        f" | anchor {memory.variance('mumtaz mahal', 40)}"
        f" | whisper {memory.variance('a whisper', 40)}"
        f" | live={memory.tick(40)} of {len(memory)}"
    )

    ring = RingMemNet()
    ring.record(3 * RATE_ONE, 0, anchor=True)
    ring.record(RATE_ONE, 0)
    ring.record(-2 * RATE_ONE, 4)
    print(
        f"\n[7] memnet   e^-1={pade_exp_neg(RATE_ONE)}"
        f" ring={len(ring)} var_now={ring.variance(0, 10, RATE_ONE)}"
        f" var_later={ring.variance(4000, 10, RATE_ONE)}"
        f" verdict={ring.consensus(4000, 10, RATE_ONE):+d}"
        f" guard={tnand(1, 1):+d}/{tnand(-1, 1):+d}"
    )

    fast = ObserverFrame("fast", rate_q16=RATE_ONE)
    slow = ObserverFrame("slow", rate_q16=(RATE_ONE * 3) // 4)
    print(
        f"\n[8] world    var_world={ring.variance_world(4000, 10)}"
        f" verdict={ring.consensus_world(4000, 10):+d}"
        f" local_fast={ring.local_variance(fast, 4000, 10)}"
        f" local_slow={ring.local_variance(slow, 4000, 10)}"
        f" (agreement is the rate-free column; the local ones differ on purpose)"
    )

    print(
        f"\ndigest route={route_name(decision)} trits={_compact(trits)}"
        f" parity=0x{parity:04x} state=0x{emit.state_hex(world)}"
        f" mem={len(left)} peers={len(peers)} delta={report.delta}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
