"""Dependency-free test runner (no pytest required): `python tests/run_all.py`."""
from __future__ import annotations

import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = [
    "test_quant",
    "test_autograd",
    "test_core",
    "test_io",
    "test_pack",
    "test_viz",
    "test_attention",
    "test_advanced",
    "test_e2e",
]


def main() -> int:
    passed = failed = 0
    for modname in MODULES:
        mod = importlib.import_module(modname)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"PASS {modname}.{name}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL {modname}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
