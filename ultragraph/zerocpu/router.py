"""SmartTree fanout: a deterministic integer router.

Given a prompt, decide who should handle it: the Caveman integer kernel (cheap,
local), a batch pass (medium), a full model (expensive), an explicit plugin, or
nothing at all.  The decision is a fixed sequence of integer comparisons over
:func:`features` -- same prompt, same route, on every machine, in every process.

No floats, no regex, no randomness, no dictionary-iteration order dependence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping

#: Local ternary kernel.  Cheapest, fully deterministic.
ROUTE_CAVEMAN = 0
#: Batched kernel pass, still integer-only, more tokens of work.
ROUTE_BATCH = 1
#: Hand off to a model: the prompt is too big or too sharp for the kernel.
ROUTE_MODEL = 2
#: An explicit ``@``/``/`` token named the handler; do not second-guess it.
ROUTE_PLUGIN = 3
#: Nothing to do.
ROUTE_DENY = 4

#: Above this many characters the kernel cannot hold the context.
ROUTE_MAX_CHARS_HEAVY = 2000
#: At or below this many characters the Caveman kernel handles it alone.
ROUTE_MAX_CHARS_BATCH = 160
#: Trition budget for one Caveman pass (reserved; see ``zerocpu.trit``).
ROUTE_MAX_TRITS = 512
#: This many ``!``/``?`` in a long prompt means it wants a real reasoner.
ROUTE_MAX_MARKS = 3

#: Feature order is part of the contract: transcripts and fixtures rely on it.
FEATURE_KEYS = ("chars", "words", "lines", "digits", "plugins", "marks", "heat")

HEAT_SALT = 31
HEAT_MASK = 0xFFFF
PLUGIN_PREFIXES = ("@", "/")

_ROUTE_NAMES = {
    ROUTE_CAVEMAN: "ROUTE_CAVEMAN",
    ROUTE_BATCH: "ROUTE_BATCH",
    ROUTE_MODEL: "ROUTE_MODEL",
    ROUTE_PLUGIN: "ROUTE_PLUGIN",
    ROUTE_DENY: "ROUTE_DENY",
}


def features(prompt: str) -> dict[str, int]:
    """Integer features of ``prompt``, in :data:`FEATURE_KEYS` order.

    ``heat`` is a position-weighted sum over the characters, so ``"ab"`` and
    ``"ba"`` hash apart.  It is a summary, not the per-lane signal used by
    :mod:`ultragraph.zerocpu.hope`.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"features expects str, got {type(prompt).__name__}")
    chars = len(prompt)
    words = len(prompt.split())
    lines = prompt.count("\n") + 1 if chars else 0
    digits = 0
    marks = 0
    for ch in prompt:
        if "0" <= ch <= "9":
            digits += 1
        elif ch == "!" or ch == "?":
            marks += 1
    plugins = 0
    for token in prompt.split():
        if token.startswith(PLUGIN_PREFIXES):
            plugins += 1
    heat = 0
    for i, ch in enumerate(prompt):
        heat += ((ord(ch) * HEAT_SALT + i) * (i + 1)) & HEAT_MASK
    return {
        "chars": chars,
        "words": words,
        "lines": lines,
        "digits": digits,
        "plugins": plugins,
        "marks": marks,
        "heat": heat & HEAT_MASK,
    }


def route(feats: Mapping[str, int]) -> int:
    """Pick a route.  The order of these tests is the specification."""
    chars = int(feats.get("chars", 0))
    if chars <= 0:
        return ROUTE_DENY
    if int(feats.get("plugins", 0)) > 0:
        return ROUTE_PLUGIN
    if chars >= ROUTE_MAX_CHARS_HEAVY:
        return ROUTE_MODEL
    if int(feats.get("marks", 0)) >= ROUTE_MAX_MARKS and chars > ROUTE_MAX_CHARS_BATCH:
        return ROUTE_MODEL
    if chars <= ROUTE_MAX_CHARS_BATCH:
        return ROUTE_CAVEMAN
    return ROUTE_BATCH


def route_name(code: int) -> str:
    """Human name for a route code."""
    try:
        return _ROUTE_NAMES[code]
    except KeyError as exc:
        raise ValueError(f"unknown route code {code}") from exc


def explain(feats: Mapping[str, int]) -> list[str]:
    """The reason chain that :func:`route` walks, ending in the decision.

    Kept as an explicit mirror of :func:`route`; the tests assert the last
    entry always agrees with ``route_name(route(feats))``.
    """
    chars = int(feats.get("chars", 0))
    plugins = int(feats.get("plugins", 0))
    marks = int(feats.get("marks", 0))
    if chars <= 0:
        return ["chars=0 -> ROUTE_DENY (nothing to route)"]
    lines = [
        f"chars={chars} words={int(feats.get('words', 0))}"
        f" lines={int(feats.get('lines', 0))} heat=0x{int(feats.get('heat', 0)):04x}"
    ]
    if plugins > 0:
        lines.append(f"plugins={plugins} > 0 -> ROUTE_PLUGIN (explicit delegation wins)")
        return lines
    if chars >= ROUTE_MAX_CHARS_HEAVY:
        lines.append(
            f"chars={chars} >= ROUTE_MAX_CHARS_HEAVY={ROUTE_MAX_CHARS_HEAVY} -> ROUTE_MODEL"
        )
        return lines
    if marks >= ROUTE_MAX_MARKS and chars > ROUTE_MAX_CHARS_BATCH:
        lines.append(
            f"marks={marks} >= ROUTE_MAX_MARKS={ROUTE_MAX_MARKS} and"
            f" chars={chars} > ROUTE_MAX_CHARS_BATCH={ROUTE_MAX_CHARS_BATCH} -> ROUTE_MODEL"
        )
        return lines
    if chars <= ROUTE_MAX_CHARS_BATCH:
        lines.append(
            f"chars={chars} <= ROUTE_MAX_CHARS_BATCH={ROUTE_MAX_CHARS_BATCH} -> ROUTE_CAVEMAN"
        )
        return lines
    lines.append(
        f"ROUTE_MAX_CHARS_BATCH < chars={chars} < ROUTE_MAX_CHARS_HEAVY={ROUTE_MAX_CHARS_HEAVY}"
        " -> ROUTE_BATCH"
    )
    return lines


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ultragraph.zerocpu.router",
        description="Route a prompt through the deterministic SmartTree fanout.",
    )
    parser.add_argument("prompt", nargs="?", default="", help="prompt text")
    args = parser.parse_args(argv)
    feats = features(args.prompt)
    decision = route(feats)
    print(
        json.dumps(
            {
                "features": feats,
                "route": decision,
                "name": route_name(decision),
                "why": explain(feats),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
