# vendored: vakedc

Vendored from [`peterlodri-sec/vaked-base`](https://github.com/peterlodri-sec/vaked-base) — the `vakedc/` directory at commit `5d37e57` (`5d37e5709478e83de611fc2fcd4ab7915d377370`), on 2026-07-11.

The Vaked-C compiler: lexer → parser → resolve → check → lower → emit (plus graph, lsp, tracing). Upstream vaked-base is the source of truth — re-sync from there rather than editing this copy.

> ⚠️ At this upstream commit, `resolve.py` and `tracing.py` are stored
> **de-indented** (a kompress-ultra artifact — bodies at column 0) and will not
> compile as-is. Fix upstream in `vaked-base` and re-sync; the other modules
> compile cleanly. This directory is a faithful mirror, not imported by
> ultragraph, and excluded from CI (ruff/pytest run only on
> `ultragraph tests examples assets`).
