"""p/q prompts -> deterministic weight-programming.

A *p/q prompt* is a line-oriented mutation program: every line is one graph edit
carrying an exact rational acceptance weight.  ``q 1/2`` fires half of the
seeded draws, ``q 3/4`` three quarters, no suffix fires always.  The draw is
integer-only (``rng.randrange(q) < p``), so an evolution run is reproducible
from ``(source, seed)`` alone -- no float thresholds, no platform differences.

Example::

    # push the world forward, then program the wiring
    step 2
    edge 0 1 +1
    wire a b residual q 3/4
    flip 1 2 q 1/2
    set 3 -7
    prune 0 1 q 1/4

Ops and the targets that support them:

=========  =================  ==============================================
op         targets             effect
=========  =================  ==============================================
``edge``   Tree, Spacetime     add a ternary micro-edge / causal link
``flip``   Tree, Spacetime     negate an existing weight
``prune``  Tree, Spacetime     drop every ``src -> dst`` edge
``set``    Tree, Spacetime     write a node byte / world state
``step``   Spacetime           advance ``n`` deterministic ticks
``wire``   UltraGraph          add an ultra-edge between two named trees
=========  =================  ==============================================
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..core import Tree, UltraGraph
from .spacetime import SpacetimeGraph

__all__ = ["Op", "PqError", "Prompt", "Report", "apply_pq", "evolve", "parse_pq"]

_TERNARY = (-1, 0, 1)
_KINDS = ("EDGE", "FLIP", "PRUNE", "SET", "STEP", "WIRE")
_UE_KINDS = ("plain", "residual")


class PqError(ValueError):
    """A malformed p/q prompt, or an op the target cannot accept."""

    def __init__(self, line: int, message: str) -> None:
        self.line = line
        super().__init__(f"line {line}: {message}" if line else message)


@dataclass(frozen=True)
class Op:
    """One mutation with an exact rational acceptance weight ``p/q``."""

    kind: str
    args: tuple = ()
    p: int = 1
    q: int = 1
    line: int = field(default=0, compare=False)  # diagnostic, not identity

    def accepts(self, rng: random.Random) -> bool:
        """Deterministic acceptance test.  ``p >= q`` never consumes a draw."""
        if self.p >= self.q:
            return True
        if self.p <= 0:
            return False
        return rng.randrange(self.q) < self.p

    def text(self) -> str:
        body = " ".join(str(a) for a in self.args)
        head = self.kind.lower()
        base = f"{head} {body}".strip()
        return base if (self.p == 1 and self.q == 1) else f"{base} q {self.p}/{self.q}"

    def __str__(self) -> str:
        return self.text()


@dataclass(frozen=True)
class Prompt:
    """A parsed, immutable mutation program."""

    ops: tuple[Op, ...] = ()
    source: str = field(default="", compare=False)  # original text, not identity

    def text(self) -> str:
        """Canonical form; ``parse_pq(p.text()) == p`` holds."""
        return "\n".join(op.text() for op in self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self):
        return iter(self.ops)

    def __getitem__(self, i: int) -> Op:
        return self.ops[i]


@dataclass
class Report:
    """Outcome of one :func:`apply_pq` pass."""

    applied: int = 0
    skipped: int = 0
    log: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"applied": self.applied, "skipped": self.skipped, "log": list(self.log)}

    def __str__(self) -> str:
        return f"Report(applied={self.applied}, skipped={self.skipped})"


# -- parsing ------------------------------------------------------------------
def _ratio(token: str, line: int) -> tuple[int, int]:
    if "/" not in token:
        raise PqError(line, f"ratio must be P/Q, got {token!r}")
    num, _, den = token.partition("/")
    try:
        p, q = int(num), int(den)
    except ValueError as exc:
        raise PqError(line, f"ratio must be integers, got {token!r}") from exc
    if q < 1:
        raise PqError(line, f"ratio denominator must be >= 1, got {q}")
    if not 0 <= p <= q:
        raise PqError(line, f"ratio must satisfy 0 <= p <= q, got {p}/{q}")
    return p, q


def _int(token: str, line: int, what: str) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise PqError(line, f"{what} must be an integer, got {token!r}") from exc


def _operand(token: str, line: int, target_hint: str) -> int:
    idx = _int(token, line, f"{target_hint} index")
    if idx < 0:
        raise PqError(line, f"{target_hint} index must be >= 0, got {idx}")
    return idx


def _weight(token: str, line: int) -> int:
    w = _int(token, line, "weight")
    if w not in _TERNARY:
        raise PqError(line, f"weight must be one of {_TERNARY}, got {w}")
    return w


def _args(kind: str, toks: list[str], line: int) -> tuple:
    if kind == "EDGE":
        if len(toks) != 3:
            raise PqError(line, "edge needs SRC DST WEIGHT")
        return (_operand(toks[0], line, "src"), _operand(toks[1], line, "dst"), _weight(toks[2], line))
    if kind == "FLIP" or kind == "PRUNE":
        if len(toks) != 2:
            raise PqError(line, f"{kind.lower()} needs SRC DST")
        return (_operand(toks[0], line, "src"), _operand(toks[1], line, "dst"))
    if kind == "SET":
        if len(toks) != 2:
            raise PqError(line, "set needs NODE VALUE")
        return (_operand(toks[0], line, "node"), _int(toks[1], line, "value"))
    if kind == "STEP":
        if len(toks) != 1:
            raise PqError(line, "step needs N")
        n = _int(toks[0], line, "step count")
        if n < 0:
            raise PqError(line, f"step count must be >= 0, got {n}")
        return (n,)
    if kind == "WIRE":
        if len(toks) not in (2, 3):
            raise PqError(line, "wire needs SRC_TREE DST_TREE [plain|residual]")
        ue = toks[2].lower() if len(toks) == 3 else "plain"
        if ue not in _UE_KINDS:
            raise PqError(line, f"wire kind must be one of {_UE_KINDS}, got {ue!r}")
        return (toks[0], toks[1], ue)
    raise PqError(line, f"unknown op {kind.lower()!r}")  # pragma: no cover - defensive


def parse_pq(source: str) -> Prompt:
    """Parse a p/q prompt.  ``#`` starts a comment; blank lines are ignored."""
    if not isinstance(source, str):
        raise TypeError(f"parse_pq expects str, got {type(source).__name__}")
    ops: list[Op] = []
    for lineno, raw in enumerate(source.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        toks = line.split()
        p, q = 1, 1
        if "q" in toks:
            i = toks.index("q")
            if i + 1 >= len(toks):
                raise PqError(lineno, "q needs a P/Q ratio")
            p, q = _ratio(toks[i + 1], lineno)
            toks = toks[:i] + toks[i + 2 :]
        kind = toks[0].upper()
        if kind not in _KINDS:
            raise PqError(lineno, f"unknown op {toks[0]!r}; expected one of {_KINDS}")
        ops.append(Op(kind=kind, args=_args(kind, toks[1:], lineno), p=p, q=q, line=lineno))
    return Prompt(ops=tuple(ops), source=source)


# -- execution ----------------------------------------------------------------
class _Adapter:
    """Binds a p/q op to one graph type."""

    def apply(self, op: Op) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class _SpacetimeAdapter(_Adapter):
    def __init__(self, graph: SpacetimeGraph) -> None:
        self.graph = graph

    def apply(self, op: Op) -> str:
        g, k = self.graph, op.kind
        if k == "EDGE":
            src, dst, w = op.args
            g.add_edge(src, dst, w)
            return f"edge {src} -> {dst} w={w:+d}"
        if k == "FLIP":
            src, dst = op.args
            edge = g.find_edge(src, dst)
            if edge is None:
                raise PqError(op.line, f"flip: no edge {src} -> {dst}")
            g.set_edge_weight(src, dst, -edge.weight)
            return f"flip {src} -> {dst} w={-edge.weight:+d}"
        if k == "PRUNE":
            src, dst = op.args
            removed = g.remove_edge(src, dst)
            if removed == 0:
                raise PqError(op.line, f"prune: no edge {src} -> {dst}")
            return f"prune {src} -> {dst} ({removed} edge(s))"
        if k == "SET":
            node, value = op.args
            g.set_state(node, value)
            return f"set node {node} = {g.state()[node]}"
        if k == "STEP":
            g.tick(op.args[0])
            return f"step {op.args[0]} (t={g.tick_count})"
        raise PqError(op.line, f"{k.lower()} is not a SpacetimeGraph op; use edge/flip/prune/set/step")


class _TreeAdapter(_Adapter):
    """Node indices and micro-edges.  Weight bytes live in private lists."""

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def _index(self, idx: int, op: Op, what: str) -> int:
        if not 0 <= idx < len(self.tree):
            raise PqError(op.line, f"{what} index {idx} out of range (n={len(self.tree)})")
        return idx

    def apply(self, op: Op) -> str:
        t, k = self.tree, op.kind
        if k == "EDGE":
            src, dst, w = op.args
            self._index(src, op, "src")
            self._index(dst, op, "dst")
            t.add_edge(src, dst, w)
            return f"edge {src} -> {dst} w={w:+d}"
        if k == "FLIP":
            src, dst = op.args
            for i, (s, d) in enumerate(zip(t._esrc, t._edst)):
                if s == src and d == dst:
                    t._eval[i] = -t._eval[i]
                    return f"flip {src} -> {dst} w={t._eval[i]:+d}"
            raise PqError(op.line, f"flip: no edge {src} -> {dst}")
        if k == "PRUNE":
            src, dst = op.args
            before = len(t._esrc)
            keep = [i for i, (s, d) in enumerate(zip(t._esrc, t._edst)) if not (s == src and d == dst)]
            t._esrc = [t._esrc[i] for i in keep]
            t._edst = [t._edst[i] for i in keep]
            t._eval = [t._eval[i] for i in keep]
            removed = before - len(t._esrc)
            if removed == 0:
                raise PqError(op.line, f"prune: no edge {src} -> {dst}")
            return f"prune {src} -> {dst} ({removed} edge(s))"
        if k == "SET":
            node, value = op.args
            self._index(node, op, "node")
            t[node] = value  # __setitem__ clips into int8
            return f"set node {node} = {int(t.nodes[node])}"
        if k == "STEP":
            raise PqError(op.line, "step is a SpacetimeGraph op; a Tree has no clock")
        raise PqError(op.line, f"{k.lower()} is not a Tree op; use an UltraGraph to wire")


class _UltraAdapter(_Adapter):
    def __init__(self, ug: UltraGraph) -> None:
        self.ug = ug

    def apply(self, op: Op) -> str:
        if op.kind != "WIRE":
            raise PqError(op.line, f"{op.kind.lower()} is not an UltraGraph op; wire only")
        src_name, dst_name, kind = op.args
        src = next((t for t in self.ug.trees if t.name == src_name), None)
        dst = next((t for t in self.ug.trees if t.name == dst_name), None)
        if src is None:
            raise PqError(op.line, f"wire: no tree named {src_name!r}")
        if dst is None:
            raise PqError(op.line, f"wire: no tree named {dst_name!r}")
        self.ug.wire(src, dst, kind)
        return f"wire {src_name} -> {dst_name} ({kind})"


def _adapter_for(target) -> _Adapter:
    if isinstance(target, SpacetimeGraph):
        return _SpacetimeAdapter(target)
    if isinstance(target, UltraGraph):
        return _UltraAdapter(target)
    if isinstance(target, Tree):
        return _TreeAdapter(target)
    raise TypeError(
        f"apply_pq target must be Tree, UltraGraph, or SpacetimeGraph, got {type(target).__name__}"
    )


def apply_pq(target, prompt: Prompt | str, seed: int = 0, budget: int | None = None) -> Report:
    """Apply a p/q prompt to ``target`` and return a :class:`Report`.

    ``budget`` caps how many ops may land (a crude energy budget for evolution).
    Rejected draws and budget-skipped ops are counted, never silently dropped.
    """
    program = parse_pq(prompt) if isinstance(prompt, str) else prompt
    if not isinstance(program, Prompt):
        raise TypeError(f"apply_pq expects Prompt or str, got {type(program).__name__}")
    if budget is not None and budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    adapter = _adapter_for(target)
    rng = random.Random(seed)
    report = Report()
    for op in program:
        if budget is not None and report.applied >= budget:
            report.skipped += 1
            report.log.append(f"line {op.line}: budget exhausted")
            continue
        if not op.accepts(rng):
            report.skipped += 1
            report.log.append(f"line {op.line}: {op.kind.lower()} rejected ({op.p}/{op.q})")
            continue
        report.log.append(f"line {op.line}: {adapter.apply(op)}")
        report.applied += 1
    return report


def evolve(target, prompt: Prompt | str, seed: int = 0, rounds: int = 1, budget: int | None = None):
    """Run the same program ``rounds`` times, one deterministic report per round.

    Round ``r`` uses ``seed + r``, so the whole evolution is a pure function of
    ``(target, source, seed, rounds)``.
    """
    if rounds < 0:
        raise ValueError(f"rounds must be >= 0, got {rounds}")
    program = parse_pq(prompt) if isinstance(prompt, str) else prompt
    return [apply_pq(target, program, seed=seed + r, budget=budget) for r in range(rounds)]
