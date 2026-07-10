# ultragraph task runner (https://github.com/casey/just)
# `just` with no args lists recipes.

default:
    @just --list

# Create/sync the uv-managed environment (Python >=3.14, numpy, pytest).
install:
    uv sync

# Run the full test suite via pytest.
test:
    uv run pytest

# Run tests with no third-party deps (stdlib + numpy only).
test-fast:
    uv run python tests/run_all.py

# Run the end-to-end char-level ternary LM demo.
demo:
    uv run python examples/char_lm.py

# Render example SVG visualizations to ./out.
viz:
    uv run python examples/render_viz.py

# Byte-compile every module to catch syntax errors fast.
check:
    uv run python -m compileall ultragraph tests examples
