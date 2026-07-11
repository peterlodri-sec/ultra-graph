# vendored: vakedc

Vendored from [`peterlodri-sec/vaked-base`](https://github.com/peterlodri-sec/vaked-base)
— the `vakedc/` directory at commit `5d37e57`, on 2026-07-11.

## ⚠️ upstream source is de-indented AND lossy

At this commit the vakedc `.py` files are committed **de-indented** (block bodies at
column 0). Worse, re-indentation reveals the source is also **lossy**: e.g. in
`parser.py`, `_skip_nl_inline` has no body between it and the next `def _decl`, so
the only compilable arrangement nests `_decl` inside it and `self._decl()` then
raises `AttributeError`. Missing content cannot be recovered by re-indentation.

Status of the files:
- **Re-indented, compile** (whitespace-only reconstruction): `resolve.py`,
  `tracing.py`, `graph.py`, `emit.py`, `parser.py`. Of these, `parser.py` **compiles
  but is not functional** due to the lossy gap above.
- **Still de-indented**: `check.py` (1700 loc), `lower.py` (1972 loc), `lsp.py`.
- **Fine upstream already**: `__init__.py`, `__main__.py`, `lexer.py`.

**Consequence:** `ultragraph.vaked.compile_vaked()` cannot run against this vendored
copy (`import vakedc` fails on the de-indented `check.py`; and the parser is lossy
anyway). `compile_vaked` therefore raises a clean `ImportError`. The
`ultragraph.vaked.lower_graph()` lowering pass is fully functional and tested on its
own. To enable `compile_vaked`, hydrate `vakedc` from its **original source**
upstream in vaked-base (the de-indentation/kompress dropped content), then re-sync.

Not imported by ultragraph; excluded from CI.
