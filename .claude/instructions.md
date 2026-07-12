# ultragraph — Claude instructions

Start by reading `AGENTS.md` at the repo root for the full project reference (architecture, commands, conventions, gotchas, testing patterns, persistence).

## Key facts

- **Python >= 3.11**, only runtime dep is `numpy>=2.0`. Optional extras: `viz` (matplotlib), `wiki` (pymediawiki), `mcp` (mcp).
- **Package manager**: `uv`. Commands: `uv sync --extra viz`, `uv run pytest`, `uv run ruff check ultragraph tests examples assets`.
- **No type checker** (no mypy/pyright). Ruff lint rules: `F`, `E7`, `E9`, `I`, line-length 110.
- **`from __future__ import annotations`** in every source file.
- Custom autograd (`ultragraph.autograd.Tensor`), not PyTorch.
- Only numpy at module level; optional deps imported lazily.
- `vakedc/` is vendored but not importable (de-indented upstream loss). Excluded from CI and PyPI.

## BMad agents

Three agents live under `skills/` — activate by describing the task:
- **ByteSmith** (`skills/ultragraph-dev/`) — writes autograd ops, wires trees, trains models, explains internals
- **CorpusCrafter** (`skills/corpus-trainer/`) — fetches/enriches corpora, trains/deploys byte-level GPTs for any language
- **GraphViz** (`skills/viz-doc/`) — renders SVG/PNG visualizations and generates model card reports
