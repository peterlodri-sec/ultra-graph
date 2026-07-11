# vendored: vakedc

Vendored from [`peterlodri-sec/vaked-base`](https://github.com/peterlodri-sec/vaked-base)
— the `vakedc/` directory at commit `5d37e57`, on 2026-07-11. The Vaked-C compiler:
lexer → parser → resolve → check → lower → emit (+ graph, lsp, tracing, mlir, passes).

## ⚠️ upstream is stored de-indented (kompress artifact)

At this commit most `vakedc` `.py` files are committed **de-indented** in vaked-base
(block bodies at column 0), so they do not compile as-is:

- **Re-indented + validated locally** (compile + import OK): `resolve.py`, `tracing.py`
- **Still de-indented** — need the upstream original / decompressor:
  `check.py`, `emit.py`, `graph.py`, `lower.py`, `lsp.py`, `parser.py`
- **Fine upstream already**: `__init__.py`, `__main__.py`, `lexer.py`

The correct fix is to hydrate `vakedc` from its original source (or the kompress
decompressor) **upstream in vaked-base**, then re-sync this vendor. This directory
is a mirror, not imported by ultragraph, and excluded from CI.
