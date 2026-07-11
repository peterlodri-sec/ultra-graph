# Contributing to ultragraph

Thanks for poking at a byte-graph. It's a small, pure-Python (+ numpy) library, so
the loop is short.

## Setup

Uses [uv](https://github.com/astral-sh/uv) and Python ≥ 3.11.

```sh
uv sync --extra viz     # core + dev (pytest, ruff) + matplotlib for viz tests
```

## The loop

```sh
just test        # pytest (or: uv run pytest)
just test-fast   # dependency-free runner (stdlib + numpy only)
uv run ruff check ultragraph tests examples assets   # lint
just demo        # end-to-end ternary mini-GPT
just viz         # render example graphs
```

CI (`.github/workflows/ci.yml`) runs `ruff check` + `pytest` on Python 3.11–3.13.
Both must be green before merge.

## Conventions

- **Correctness first.** New autograd ops need a numeric-gradient test
  (`tests/test_autograd.py` has the pattern: analytic backward vs central finite
  differences). Quantized paths are checked against their unquantized surrogate
  (that's what STE claims).
- **The byte contract holds.** One node = one byte (`int8` activation), one edge =
  one byte (ternary weight). Anything bigger than a byte lives in the ad-hoc side
  store, not the hot buffers.
- **Keep files focused.** Match the surrounding style: `ruff` config in
  `pyproject.toml` (`select = F, E7, E9, I`); imports sorted.
- **numpy is the only runtime dependency.** `matplotlib` is optional (`[viz]`
  extra); the SVG backend stays stdlib-only.

## Pull requests

Small, focused PRs. Include tests. Run `ruff check` + `pytest` locally first. Say
what you changed and why in the description.
