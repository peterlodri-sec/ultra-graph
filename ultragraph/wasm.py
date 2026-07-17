""".ugm → WebAssembly (WAT generator).

Generates a standalone WebAssembly Text Format module from a .ugm file.
The WASM module embeds model weights as memory data and exports::

    memory  — linear memory (1+ pages, 64 KiB each)
    run     — (input_offset: i32, output_offset: i32) → void

Matrix multiplication is implemented as nested WASM loops. Supported
activations: none, relu, sigmoid (fast approximation), tanh (fast).

Usage::

    from ultragraph.wasm import generate_wat

    module = load_ugm("model.ugm")
    wat = generate_wat(module)              # → .wat text

    # compile to .wasm (requires wat2wasm from wabt)
    from ultragraph.wasm import compile_wat, save_wasm
    save_wasm(module, "model.wasm")         # auto-compile via wat2wasm
"""


import subprocess
from pathlib import Path

from .ugm import (
    ACT_IDENTITY,
    ACT_NONE,
    ACT_RELU,
    ACT_SIGMOID,
    ACT_TANH,
    UE_PLAIN,
    UE_RESIDUAL,
    UGMFile,
    UGMTree,
    UGMUltraEdge,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_ACT_NAME = {
    ACT_NONE: "none",
    ACT_IDENTITY: "none",
    ACT_RELU: "relu",
    ACT_SIGMOID: "sigmoid",
    ACT_TANH: "tanh",
}


# ---------------------------------------------------------------------------
# WAT string encoding
# ---------------------------------------------------------------------------

def _wat_str(data: bytes) -> str:
    """Encode bytes as a WAT string literal."""
    chars = ['"']
    for b in data:
        if b == 0x00:
            chars.append("\\00")
        elif b == 0x09:
            chars.append("\\t")
        elif b == 0x0A:
            chars.append("\\n")
        elif b == 0x0D:
            chars.append("\\r")
        elif b == 0x22:
            chars.append('\\"')
        elif b == 0x5C:
            chars.append("\\\\")
        elif 0x20 <= b < 0x7F:
            chars.append(chr(b))
        else:
            chars.append(f"\\{b:02x}")
    chars.append('"')
    return "".join(chars)


# ---------------------------------------------------------------------------
# memory layout
# ---------------------------------------------------------------------------

def _compute_layout(module: UGMFile) -> dict:
    """Compute byte offsets for weights and scratch buffers in WASM memory.

    Layout::

        [bias_0][wq_0][bias_1][wq_1]...[scratch_0][scratch_1]...

    Returns dict with offset/size arrays and total sizes.
    """
    bias_off: list[int] = []
    bias_size: list[int] = []
    wq_off: list[int] = []
    wq_size: list[int] = []
    scratch_off: list[int] = []

    off = 0
    for t in module.trees:
        bsz = t.out_dim * 4 if t.kind == 0 else 0
        bias_off.append(off)
        bias_size.append(bsz)
        off += bsz

        wsz = t.out_dim * t.in_dim if t.kind == 0 else 0
        wq_off.append(off)
        wq_size.append(wsz)
        off += wsz

    for t in module.trees:
        scratch_off.append(off)
        off += t.out_dim * 4

    return dict(
        bias_off=bias_off,
        bias_size=bias_size,
        wq_off=wq_off,
        wq_size=wq_size,
        scratch_off=scratch_off,
        data_end=off - sum(t.out_dim * 4 for t in module.trees),
        scratch_end=off,
    )


# ---------------------------------------------------------------------------
# activation expression
# ---------------------------------------------------------------------------

def _act_expr(act: int, acc: str = "$acc") -> str:
    """Return WAT expression for activation in *acc*."""
    name = _ACT_NAME.get(act, "none")
    if name == "relu":
        return f"(f32.max (local.get {acc}) (f32.const 0))"
    if name == "sigmoid":
        return (
            "(f32.add\n"
            "          (f32.const 0.5)\n"
            "          (f32.div\n"
            f"            (local.get {acc})\n"
            "            (f32.mul\n"
            "              (f32.const 2)\n"
            "              (f32.add\n"
            "                (f32.const 1)\n"
            f"                (f32.abs (local.get {acc}))\n"
            "              )\n"
            "            )\n"
            "          )\n"
            "        )"
        )
    if name == "tanh":
        # fast tanh: 2x / (1 + |2x|)
        return (
            "(f32.div\n"
            f"          (f32.mul (local.get {acc}) (f32.const 2))\n"
            "          (f32.add\n"
            "            (f32.const 1)\n"
            f"            (f32.abs (f32.mul (local.get {acc}) (f32.const 2)))\n"
            "          )\n"
            "        )"
        )
    return f"(local.get {acc})"


# ---------------------------------------------------------------------------
# topological order
# ---------------------------------------------------------------------------

def _resolve_order(module: UGMFile) -> list[int]:
    """Topological order matching UGMFile.run()."""
    incoming: dict[int, list[UGMUltraEdge]] = {i: [] for i in range(len(module.trees))}
    for ue in module.ultra_edges:
        incoming[ue.dst_idx].append(ue)
    remaining = set(range(len(module.trees)))
    order: list[int] = []
    outputs: set[int] = set()
    while remaining:
        progressed = False
        for ti in list(remaining):
            if all(e.src_idx in outputs for e in incoming[ti]):
                order.append(ti)
                outputs.add(ti)
                remaining.remove(ti)
                progressed = True
        if not progressed:
            msg = f"cycle at remaining trees {remaining}"
            raise ValueError(msg)
    return order


# ---------------------------------------------------------------------------
# tree forward function (WAT)
# ---------------------------------------------------------------------------

def _tree_wat(idx: int, tree: UGMTree, layout: dict) -> str:
    """Emit a WAT function ``$tree_N`` for one tree's forward pass.

    Signature: (func $tree_N (param $in i32) (param $out i32))
    """
    in_dim, out_dim = tree.in_dim, tree.out_dim
    bo = layout["bias_off"][idx]
    wo = layout["wq_off"][idx]
    act_code = _act_expr(tree.act)

    return f"""  ;; tree[{idx}]: {tree.name}  {in_dim}→{out_dim}  act={_ACT_NAME.get(tree.act,'none')}
  (func $tree_{idx} (param $in i32) (param $out i32)
    (local $j i32) (local $k i32) (local $acc f32)
    (local.set $j (i32.const 0))
    (loop $jloop_{idx}
      (local.set $acc (f32.load (i32.add (i32.const {bo}) (i32.mul (local.get $j) (i32.const 4)))))
      (local.set $k (i32.const 0))
      (loop $kloop_{idx}
        (local.set $acc
          (f32.add
            (local.get $acc)
            (f32.mul
              (f32.convert_i32_s (i32.load8_s (i32.add (i32.const {wo}) (i32.add (i32.mul (local.get $j) (i32.const {in_dim})) (local.get $k)))))
              (f32.load (i32.add (local.get $in) (i32.mul (local.get $k) (i32.const 4))))
            )
          )
        )
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (br_if $kloop_{idx} (i32.lt_s (local.get $k) (i32.const {in_dim})))
      )
      (local.set $acc {act_code})
      (f32.store (i32.add (local.get $out) (i32.mul (local.get $j) (i32.const 4))) (local.get $acc))
      (local.set $j (i32.add (local.get $j) (i32.const 1)))
      (br_if $jloop_{idx} (i32.lt_s (local.get $j) (i32.const {out_dim})))
    )
  )"""


# ---------------------------------------------------------------------------
# memory copy loop (WAT)
# ---------------------------------------------------------------------------

def _copy_loop_wat(dst_expr: str, src_off: int, sz: int, label: str) -> list[str]:
    """Emit a loop that copies *sz* bytes from *src_off* to *dst_expr*."""
    lines = [
        "    (local.set $i (i32.const 0))",
        f"    (block $b{label}",
        f"      (loop $l{label}",
        f"        (f32.store (i32.add {dst_expr} (local.get $i))",
        f"          (f32.load (i32.add (i32.const {src_off}) (local.get $i))))",
        "        (local.set $i (i32.add (local.get $i) (i32.const 4)))",
        f"        (br_if $l{label} (i32.lt_s (local.get $i) (i32.const {sz})))",
        "      )",
        "    )",
    ]
    return lines


def _add_loop_wat(dst_expr: str, src_off: int, sz: int, label: str) -> list[str]:
    """Emit a loop that adds memory at *src_off* into *dst_expr*."""
    lines = [
        "    (local.set $i (i32.const 0))",
        f"    (block $b{label}",
        f"      (loop $l{label}",
        f"        (f32.store (i32.add {dst_expr} (local.get $i))",
        "          (f32.add",
        f"            (f32.load (i32.add {dst_expr} (local.get $i)))",
        f"            (f32.load (i32.add (i32.const {src_off}) (local.get $i)))))",
        "        (local.set $i (i32.add (local.get $i) (i32.const 4)))",
        f"        (br_if $l{label} (i32.lt_s (local.get $i) (i32.const {sz})))",
        "      )",
        "    )",
    ]
    return lines


def _sum_loop_wat(src_offs: list[int], dst_off: int, sz: int, label: str) -> list[str]:
    """Emit a loop that sums several memory regions into *dst_off*.

    Generates::

        for i in range(0, sz, 4):
            dst[i] = sum(src_offs[k][i] for k in range(N))
    """
    total_expr = None
    for s_off in src_offs:
        val = f"(f32.load (i32.add (i32.const {s_off}) (local.get $i)))"
        if total_expr is None:
            total_expr = val
        else:
            total_expr = f"(f32.add {total_expr} {val})"
    return [
        "    (local.set $i (i32.const 0))",
        f"    (block $b{label}",
        f"      (loop $l{label}",
        "        (f32.store",
        f"          (i32.add (i32.const {dst_off}) (local.get $i))",
        f"          {total_expr}",
        "        )",
        "        (local.set $i (i32.add (local.get $i) (i32.const 4)))",
        f"        (br_if $l{label} (i32.lt_s (local.get $i) (i32.const {sz})))",
        "      )",
        "    )",
    ]


# ---------------------------------------------------------------------------
# generate_wat  — main entry point
# ---------------------------------------------------------------------------

def generate_wat(module: UGMFile, name: str = "model") -> str:
    """Generate WebAssembly Text format (.wat) from a .ugm module.

    The generated module exports::

        memory  — linear memory with embedded weights
        run     — (input_offset: i32, output_offset: i32) → void

    Compile to .wasm with ``wat2wasm`` from the wabt toolkit::

        wat2wasm model.wat -o model.wasm
    """
    layout = _compute_layout(module)
    order = _resolve_order(module)

    incoming: dict[int, list[UGMUltraEdge]] = {i: [] for i in range(len(module.trees))}
    for ue in module.ultra_edges:
        incoming[ue.dst_idx].append(ue)

    # ---- data segments ----
    data_segs: list[str] = []
    for i, t in enumerate(module.trees):
        if t.kind == 0:
            if t.bias is not None:
                data_segs.append(
                    f"  (data (i32.const {layout['bias_off'][i]}) {_wat_str(t.bias.tobytes())})"
                )
            if t.wq is not None:
                data_segs.append(
                    f"  (data (i32.const {layout['wq_off'][i]}) {_wat_str(t.wq.tobytes())})"
                )

    # ---- tree functions ----
    tree_funcs = [_tree_wat(i, t, layout) for i, t in enumerate(module.trees)]

    # ---- run function ----
    sink_idx = module.sink_idx
    scratch_off = layout["scratch_off"]
    run_parts: list[str] = []
    copy_counter = [0]

    for ti in order:
        edges = incoming[ti]
        is_sink = ti == sink_idx
        plain_srcs = [e.src_idx for e in edges if e.kind == UE_PLAIN]
        residual_srcs = [e.src_idx for e in edges if e.kind == UE_RESIDUAL]
        out_dim = module.trees[ti].out_dim
        out_sz = out_dim * 4

        if not plain_srcs:
            # external input → tree → scratch (or output)
            out_expr = "(local.get $output)" if is_sink else f"(i32.const {scratch_off[ti]})"
            run_parts.append(f"    ;; tree[{ti}]: {module.trees[ti].name} (input=external)")
            run_parts.append(f"    (call $tree_{ti} (local.get $input) {out_expr})")
            if residual_srcs:
                for rs in residual_srcs:
                    run_parts.append(f"    ;; residual from tree[{rs}]")
                    run_parts.extend(
                        _add_loop_wat(out_expr, scratch_off[rs], out_sz, f"r{ti}_{copy_counter[0]}")
                    )
                    copy_counter[0] += 1
            continue

        # Has plain sources
        src0_off = scratch_off[plain_srcs[0]]
        target_off = scratch_off[ti]

        if len(plain_srcs) == 1:
            run_parts.append(f"    ;; tree[{ti}]: {module.trees[ti].name} (input=tree[{plain_srcs[0]}])")
            run_parts.append(f"    (call $tree_{ti} (i32.const {src0_off}) (i32.const {target_off}))")
        else:
            # Sum sources into target_off
            run_parts.append(f"    ;; tree[{ti}]: sum of [{','.join(str(s) for s in plain_srcs)}]")
            run_parts.extend(
                _sum_loop_wat([scratch_off[s] for s in plain_srcs], target_off, out_sz, f"s{ti}")
            )
            run_parts.append(f"    (call $tree_{ti} (i32.const {target_off}) (i32.const {target_off}))")

        if residual_srcs:
            for rs in residual_srcs:
                run_parts.append(f"    ;; residual from tree[{rs}]")
                run_parts.extend(
                    _add_loop_wat(f"(i32.const {target_off})", scratch_off[rs], out_sz, f"r{ti}_{copy_counter[0]}")
                )
                copy_counter[0] += 1

        if is_sink:
            run_parts.append(f"    ;; tree[{ti}] is sink → copy to output")
            run_parts.extend(
                _copy_loop_wat("(local.get $output)", target_off, out_sz, f"o{ti}")
            )

    total_pages = max(1, (layout["scratch_end"] + 65535) // 65536)
    needs_local_i = any(
        len([e for e in module.ultra_edges if e.dst_idx == ti and e.kind == UE_PLAIN]) > 1
        or len([e for e in module.ultra_edges if e.kind == UE_RESIDUAL]) > 0
        for ti in range(len(module.trees))
    ) or any(
        len([e for e in module.ultra_edges if e.dst_idx == ti]) > 0
        for ti in [sink_idx]
    )

    lines = [
        '(module',
        f'  ;; {name} — compiled from .ugm',
        f'  ;; {len(module.trees)} trees, {len(module.ultra_edges)} ultra-edges',
        f'  (memory (export "memory") {total_pages})',
    ]
    lines.extend(data_segs)
    lines.append("")
    lines.extend(tree_funcs)
    lines.append("")
    
    local_decl = " (local $i i32)" if needs_local_i else ""
    run_lines = "\n".join(run_parts)
    lines.append(
        f'  (func (export "run") (param $input i32) (param $output i32){local_decl}'
    )
    lines.append(run_lines)
    lines.append("  )")
    lines.append(")\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# save_wasm  — convenience: write .wat + try wat2wasm
# ---------------------------------------------------------------------------

def save_wasm(
    module: UGMFile,
    path: str | Path,
    name: str = "model",
    *,
    wat2wasm: str | None = None,
) -> None:
    """Generate a .wat file (and optionally compile to .wasm).

    If *wat2wasm* is provided, it is used as the path to ``wat2wasm``.
    Otherwise, the function searches ``$PATH`` for ``wat2wasm`` first and
    falls back to only writing the .wat file.
    """
    path = Path(path)
    wat = generate_wat(module, name)
    wat_path = path.with_suffix(".wat") if path.suffix == ".wasm" else Path(str(path) + ".wat")
    wat_path.write_text(wat, encoding="utf-8")

    wasm_target = path if path.suffix == ".wasm" else path.with_suffix(".wasm")

    if wat2wasm is not None:
        cmd = [wat2wasm, str(wat_path), "-o", str(wasm_target)]
    else:
        try:
            subprocess.run(["wat2wasm", "--version"], capture_output=True, check=True)
            cmd = ["wat2wasm", str(wat_path), "-o", str(wasm_target)]
        except (FileNotFoundError, subprocess.CalledProcessError):
            cmd = None

    if cmd:
        subprocess.run(cmd, check=True)
        print(f"✓ {wasm_target} ({wasm_target.stat().st_size} bytes)")
    else:
        print(f"✓ {wat_path}  (install wabt for .wasm: brew install wabt)")


def compile_wat(wat_path: str | Path, wasm_path: str | Path | None = None) -> None:
    """Compile a .wat file to .wasm using ``wat2wasm``."""
    wat_path = Path(wat_path)
    if not wat_path.exists():
        raise FileNotFoundError(f"{wat_path} not found")
    wasm_path = Path(wasm_path) if wasm_path else wat_path.with_suffix(".wasm")
    subprocess.run(["wat2wasm", str(wat_path), "-o", str(wasm_path)], check=True)
    print(f"✓ {wasm_path}")
