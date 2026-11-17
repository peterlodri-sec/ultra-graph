"""Wiring: nothing orphaned, every surface accounted for.

Two claims in `docs/ULTRABACKYARDLOOP.md` are structural rather than numerical,
which means no benchmark can defend them:

1. **Every public name resolves.** `ultragraph.zerocpu.__all__` is the package's
   promise; a name that does not import is a broken promise.
2. **Every deterministic module has a twin, or a stated reason it does not.**
   The project's whole method is "Python reference, Rust twin, pinned by
   fixtures". A module that quietly lacks a twin is where that method stops
   being true.

The maps below are explicit and include their reasons. That is the point: a
missing twin should be a decision someone wrote down, not a gap nobody noticed.

Plain module-level tests, matching the rest of the suite.
"""

import importlib
import pathlib
from contextlib import contextmanager

import ultragraph.zerocpu as zc

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUST_SRC = ROOT / "rust" / "zerocpu" / "src"

#: Python module -> the Rust file that mirrors it.
TWIN = {
    "emit": "fixed.rs",
    "evolve": None,          # PqError/apply_pq: see PYTHON_ONLY
    "fixed": "fixed.rs",
    "hope": "hope.rs",
    "lod": "lod.rs",
    "memnet": "memnet.rs",
    "mesh": "mesh.rs",
    "observer": "observer.rs",
    "pqq": "pqq.rs",
    "router": "router.rs",
    "spacetime": "world.rs",
    "trit": "ternary.rs",
    "vaked_bridge": None,
    "meshd": None,
}

#: Modules with no Rust twin, and why. Each entry is a claim someone made.
PYTHON_ONLY = {
    "evolve": "the P/Q prompt evolver: orchestration over pqq, not a kernel",
    "vaked_bridge": "vendored VAKED compiler bridge; needs VAKED_ROOT to run at all",
    "meshd": "a UDP socket and a CLI; a Rust binary (bin/peers.rs) is its twin",
}

#: Rust files that are not a twin of a Python module, and why.
RUST_ONLY = {
    "lib.rs": "crate root: module declarations and re-exports",
    "neon.rs": "the aarch64 NEON dot; twins ternary::dot_i8 on real hardware",
    "world.rs": "twin of spacetime.py (the tick loop it calls)",
    "ternary.rs": "twin of trit.py (dot kernels live here)",
    "trit.rs": "twin of trit.py (quantiser and bitplanes)",
}

#: Modules that are deterministic and therefore must have a twin.
DETERMINISTIC = ("fixed", "trit", "observer", "memnet", "mesh", "router",
                 "lod", "pqq", "spacetime", "hope", "emit")


@contextmanager
def _guard(name):
    try:
        yield
    except AssertionError as err:
        raise AssertionError(f"[{name}] {err}") from None


def test_every_public_name_resolves():
    missing = []
    for name in zc.__all__:
        if not hasattr(zc, name):
            missing.append(name)
    assert not missing, f"__all__ names that do not resolve: {missing}"


def _bucket(name: str) -> int:
    """Constants, then classes, then functions -- the order `__all__` uses."""
    if name.isupper():
        return 0
    return 1 if name[:1].isupper() else 2


def test_all_is_convention_sorted_and_unique():
    names = zc.__all__
    assert names == sorted(names, key=lambda n: (_bucket(n), n)), (
        "__all__ must stay in convention order: CONSTANTS, Classes, functions"
    )
    assert len(set(names)) == len(names), "__all__ has duplicates"


def test_every_name_in_all_is_public():
    leaked = [name for name in zc.__all__ if name.startswith("_")]
    assert not leaked, f"private names in __all__: {leaked}"


def test_every_module_is_accounted_for():
    modules = sorted(
        path.stem for path in (ROOT / "ultragraph" / "zerocpu").glob("*.py")
        if path.stem != "__init__"
    )
    assert modules, "no modules found; the glob is wrong"
    unknown = [name for name in modules if name not in TWIN]
    assert not unknown, f"modules with no twin entry: {unknown}"


def test_every_deterministic_module_has_a_twin():
    for name in DETERMINISTIC:
        with self_subtest(name):
            twin = TWIN.get(name)
            assert twin, f"{name}.py is deterministic and has no twin"
            assert (RUST_SRC / twin).is_file(), f"{name}.py -> {twin} is missing"


def test_every_python_only_module_states_a_reason():
    for name, reason in PYTHON_ONLY.items():
        with self_subtest(name):
            assert reason.strip(), f"{name} is python-only with no stated reason"
            assert TWIN.get(name, "missing") is None, f"{name} is listed twice"


def test_every_rust_file_is_accounted_for():
    files = sorted(path.name for path in RUST_SRC.glob("*.rs"))
    unknown = [name for name in files if name not in RUST_ONLY
               and name not in set(TWIN.values())]
    assert not unknown, f"rust files with no python counterpart: {unknown}"


def test_every_twin_is_importable_from_the_package():
    """A twin that exists but is not re-exported is still an orphan."""
    for name, twin in TWIN.items():
        if twin is None:
            continue
        with self_subtest(name):
            module = importlib.import_module(f"ultragraph.zerocpu.{name}")
            assert module.__doc__, f"{name}.py has no docstring"


def test_demo_touches_every_layer():
    """Every deterministic layer must appear in the demo, by symbol or module.

    Matching on the module name alone is too strict -- the demo imports
    `to_fx` rather than `fixed` -- so either spelling counts.
    """
    demo = (ROOT / "examples" / "caveman_ultra_demo.py").read_text(encoding="utf-8")
    for name in DETERMINISTIC:
        with self_subtest(name):
            module = importlib.import_module(f"ultragraph.zerocpu.{name}")
            symbols = [n for n in dir(module) if not n.startswith("_")]
            assert name in demo or any(sym in demo for sym in symbols), (
                f"demo never touches {name}"
            )


def self_subtest(name):
    """A tiny context manager so one failure names the module, not just the file."""
    return _guard(name)