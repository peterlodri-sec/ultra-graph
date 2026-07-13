"""Click-style CLI command decorators for ultra-graph.

Register commands with ``@ug.command()`` — no argparse boilerplate. Commands
are automatically wired into the ``ug`` CLI entry point.

Usage::

    from ultragraph.command import command, run

    @command("train")
    def train(model: str = "tiny", steps: int = 1000, lr: float = 0.001):
        \"\"\"Train a GPT model.\"\"\"
        ...

    @command("generate")
    def generate(prompt: str, n_new: int = 64):
        \"\"\"Generate text from a checkpoint.\"\"\"
        ...

    if __name__ == "__main__":
        run()

Every parameter with a default becomes an optional CLI flag; parameters
without defaults become positional arguments. Type annotations control
argument parsing (int, float, str, bool).
"""
from __future__ import annotations

import argparse
import inspect
import sys
import typing
from functools import wraps
from typing import Any, Callable, get_args, get_origin, get_type_hints

_registry: dict[str, dict] = {}

_TYPE_MAP = {"str": str, "int": int, "float": float, "bool": bool}


def _resolve_type(typ: Any) -> type:
    """Resolve a string annotation to a type if possible."""
    if isinstance(typ, str):
        return _TYPE_MAP.get(typ, str)
    return typ


def _parse_annotation(raw_typ: Any) -> tuple[type, bool]:
    """Parse a type annotation into (base_type, is_optional).

    ``Optional[int]`` → (int, True) meaning nargs="?"
    ``str`` → (str, False)
    """
    origin = get_origin(raw_typ)
    if origin is typing.Union or origin is getattr(typing, "Optional", None):
        args = get_args(raw_typ)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _resolve_type(non_none[0]), True
    return _resolve_type(raw_typ), False


def command(name: str | None = None, *, help: str = ""):
    """Decorator to register a function as a CLI sub-command.

    Args:
        name: Command name (defaults to the function name).
        help: One-line description (defaults to the function's docstring).

    The decorated function's type hints become CLI arguments:
    - Parameters without defaults → positional arguments
    - Parameters with defaults → optional flags (``--param VALUE``)
    - ``bool`` parameters → flags (``--param`` / ``--no-param``)
    """
    def decorator(fn: Callable) -> Callable:
        cmd_name = name or fn.__name__.replace("_", "-")
        sig = inspect.signature(fn)
        cmd_help = help or (fn.__doc__ or "").strip().split("\n")[0]
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}
        argspec = []
        for pname, param in sig.parameters.items():
            has_default = param.default is not inspect.Parameter.empty
            raw_typ = hints.get(pname, param.annotation if param.annotation is not inspect.Parameter.empty else str)
            base_type, is_optional = _parse_annotation(raw_typ)
            argspec.append({
                "name": pname,
                "type": base_type,
                "required": not has_default,
                "default": param.default if has_default else None,
                "optional": is_optional,
            })
        _registry[cmd_name] = {
            "fn": fn,
            "help": cmd_help,
            "args": argspec,
        }

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def run(argv: list[str] | None = None, *, prog: str = "ug"):
    """Parse argv and dispatch to the registered command.

    If no command is given, prints available commands.
    """
    parser = argparse.ArgumentParser(prog=prog, description="ultragraph CLI")
    subs = parser.add_subparsers(dest="command", title="commands")

    for cmd_name, info in _registry.items():
        sub = subs.add_parser(cmd_name, help=info["help"])
        for arg in info["args"]:
            name = arg["name"].replace("_", "-")
            kwargs: dict[str, Any] = {}
            if arg["type"] is bool:
                kwargs["action"] = argparse.BooleanOptionalAction
                kwargs["default"] = arg.get("default", False) if arg.get("default") is not None else None
                if arg.get("required"):
                    kwargs["required"] = True
                sub.add_argument(f"--{name}", **kwargs)
            else:
                typ = arg["type"]
                if typ is str:
                    typ = None  # argparse default
                kwargs["type"] = typ
                if arg.get("optional"):
                    kwargs["nargs"] = "?"
                    kwargs["default"] = arg.get("default")
                if arg.get("required"):
                    if arg.get("optional"):
                        kwargs["nargs"] = "?"
                        kwargs["default"] = arg.get("default")
                    sub.add_argument(name, **kwargs)
                else:
                    kwargs["default"] = arg.get("default")
                    sub.add_argument(f"--{name}", **kwargs)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return

    info = _registry[args.command]
    kwargs = {k: v for k, v in vars(args).items() if k != "command" and v is not None}
    # Remove the default for required args that weren't provided
    for arg in info["args"]:
        if arg["required"] and arg["name"] not in kwargs:
            print(f"✗ Missing required argument: {arg['name']}", file=sys.stderr)
            raise SystemExit(1)

    return info["fn"](**kwargs)
