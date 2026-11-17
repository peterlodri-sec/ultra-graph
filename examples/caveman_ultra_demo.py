"""Caveman Ultra: one end-to-end pass over the zerocpu primitives.

    python3 examples/caveman_ultra_demo.py "ultra backprop in ternary"

Route -> quantize -> A6P-15A bitplanes -> world tick -> memory -> mesh.

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
from ultragraph.zerocpu.memnet import MemNet
from ultragraph.zerocpu.mesh import merge_peers, pack_announce, parse_announce
from ultragraph.zerocpu.router import explain, features, route, route_name
from ultragraph.zerocpu.spacetime import SpacetimeGraph
from ultragraph.zerocpu.trit import (
    A6P_LEVELS,
    bitplanes,
    delta_for_level,
    plane_parity,
    quantize_many,
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

    print(
        f"\ndigest route={route_name(decision)} trits={_compact(trits)}"
        f" parity=0x{parity:04x} state=0x{emit.state_hex(world)}"
        f" mem={len(left)} peers={len(peers)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
