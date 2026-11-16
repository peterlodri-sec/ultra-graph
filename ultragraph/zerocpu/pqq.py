"""Compiled p/q programs: ``(world, prompt, seed)`` -> text any interpreter can replay.

A ``.pqq`` artifact is the *resolved* form of a p/q prompt.  Every rational draw is
settled once, at compile time, and the outcome is written down; replay is therefore
pure interpretation -- no PRNG, no seed, no way for a second implementation to
diverge from the Python that produced the file.

Skipped ops stay in the file as comments, so the decision remains auditable::

    pqq 1
    # resolved: 4 applied, 2 skipped
    # seed 7
    step 3
    edge 0 1 +1
    set 2 -7
    # skip edge 1 2 -1 q 1/4: rejected (1/4)
    # skip prune 0 1 q 1/2: budget exhausted

Why resolve at compile time instead of carrying the seed into the artifact?  Because
reproducing CPython's ``random.Random(seed).randrange(q)`` in another language means
reimplementing MT19937 seeding bit-for-bit -- a large surface that can drift.  The
draw is a *compile-time* choice; the artifact records the choice.  Replay stays a
literal transcription of the ops, which is the property worth guaranteeing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .emit import emit_ugm
from .evolve import Op, PqError, Prompt, Report, apply_pq, parse_pq
from .spacetime import SpacetimeGraph

__all__ = [
    "PQQ_HEADER",
    "PQQ_VERSION",
    "Resolved",
    "dump_pqq",
    "load_pqq",
    "resolve",
    "run_pqq",
    "write_bundle",
]

PQQ_VERSION = 1
PQQ_HEADER = f"pqq {PQQ_VERSION}"


@dataclass(frozen=True)
class Resolved:
    """A prompt after its rational draws have been settled."""

    accepted: tuple[Op, ...] = ()
    skipped: tuple[tuple[Op, str], ...] = ()
    seed: int = 0

    def __len__(self) -> int:
        return len(self.accepted) + len(self.skipped)


def _plain(op: Op) -> Op:
    """The op with its rational suffix stripped: a compiled op is unconditional."""
    return Op(kind=op.kind, args=op.args, line=op.line)


def resolve(
    prompt: Prompt | str, seed: int = 0, budget: int | None = None
) -> Resolved:
    """Settle every ``q P/Q`` draw, consuming draws in the same order as :func:`apply_pq`.

    ``budget`` caps applied ops exactly as :func:`apply_pq` does, so
    ``resolve(p, s, b).accepted`` is the sequence that ``apply_pq(p, s, b)`` applies.
    """
    program = parse_pq(prompt) if isinstance(prompt, str) else prompt
    if not isinstance(program, Prompt):
        raise TypeError(f"resolve expects Prompt or str, got {type(program).__name__}")
    if budget is not None and budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    rng = random.Random(seed)
    accepted: list[Op] = []
    skipped: list[tuple[Op, str]] = []
    for op in program:
        if budget is not None and len(accepted) >= budget:
            skipped.append((op, "budget exhausted"))
            continue
        if not op.accepts(rng):
            skipped.append((op, f"rejected ({op.p}/{op.q})"))
            continue
        accepted.append(_plain(op))
    return Resolved(accepted=tuple(accepted), skipped=tuple(skipped), seed=seed)


def dump_pqq(
    resolved: Resolved,
    *,
    nodes: int | None = None,
    edges: int | None = None,
    tick: int | None = None,
) -> str:
    """Canonical ``.pqq`` text; ``load_pqq(dump_pqq(r))`` returns the same ops."""
    lines = [PQQ_HEADER]
    total = len(resolved)
    lines.append(
        f"# resolved: {len(resolved.accepted)} applied, {len(resolved.skipped)} skipped"
        f" of {total}"
    )
    lines.append(f"# seed {resolved.seed}")
    if nodes is not None or edges is not None or tick is not None:
        world = " ".join(
            f"{name} {value}"
            for name, value in (("nodes", nodes), ("edges", edges), ("tick", tick))
            if value is not None
        )
        lines.append(f"# world {world}".rstrip())
    lines.extend(op.text() for op in resolved.accepted)
    for op, why in resolved.skipped:
        lines.append(f"# skip {op.text()}: {why}")
    return "\n".join(lines) + "\n"


def load_pqq(text: str) -> Prompt:
    """Parse a ``.pqq`` program.  Requires the version header and no unresolved draws."""
    if not isinstance(text, str):
        raise TypeError(f"load_pqq expects str, got {type(text).__name__}")
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        stripped = raw.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped.split() != PQQ_HEADER.split():
            raise PqError(i + 1, f"expected header {PQQ_HEADER!r}, got {stripped!r}")
        lines[i] = ""  # blanked, not removed: keeps every later line number intact
        break
    else:
        raise PqError(0, f"empty program: expected {PQQ_HEADER!r} header")
    program = parse_pq("\n".join(lines))
    for op in program:
        if (op.p, op.q) != (1, 1):
            raise PqError(
                op.line,
                f"unresolved draw q {op.p}/{op.q}; compile with pqq.resolve() first",
            )
    return program


def run_pqq(graph: SpacetimeGraph, program: Prompt | str, *, budget: int | None = None) -> Report:
    """Replay a compiled program on a world.  Deterministic: no seed, no draws.

    Applied one op at a time so a program that does not fit the target world
    reports the offending line instead of a bare ``IndexError``.
    """
    ops = load_pqq(program) if isinstance(program, str) else program
    if not isinstance(ops, Prompt):
        raise TypeError(f"run_pqq expects Prompt or str, got {type(ops).__name__}")
    report = Report()
    for op in ops:
        if budget is not None and report.applied >= budget:
            report.skipped += 1
            report.log.append(f"line {op.line}: budget exhausted")
            continue
        try:
            one = apply_pq(graph, Prompt(ops=(op,)), seed=0)
        except PqError:
            raise
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            raise PqError(op.line, str(exc)) from exc
        report.applied += one.applied
        report.skipped += one.skipped
        report.log.extend(one.log)
    return report


def write_bundle(
    graph: SpacetimeGraph,
    base: str | Path,
    prompt: Prompt | str,
    *,
    seed: int = 0,
    budget: int | None = None,
) -> dict:
    """Write ``<base>.ugmz`` and ``<base>.pqq``, mutating ``graph`` with the program.

    The artifact is emitted *after* the program runs, so ``.ugmz`` plus ``.pqq``
    reproduces the same final world from either direction: replay the program, or
    load the emitted state.
    """
    # The header describes the world the program expects as input, so snapshot first.
    before = (graph.n_nodes, graph.n_edges, graph.tick_count)
    report = apply_pq(graph, prompt, seed=seed, budget=budget)
    resolved = resolve(prompt, seed=seed, budget=budget)
    path = Path(base)
    ugmz = path.with_suffix(".ugmz")
    pqq = path.with_suffix(".pqq")
    emit_ugm(graph, ugmz)
    pqq.write_text(
        dump_pqq(resolved, nodes=before[0], edges=before[1], tick=before[2])
    )
    return {
        "ugmz": str(ugmz),
        "pqq": str(pqq),
        "applied": report.applied,
        "skipped": report.skipped,
        "nodes": graph.n_nodes,
        "edges": graph.n_edges,
        "tick": graph.tick_count,
    }
