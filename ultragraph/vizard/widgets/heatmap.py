"""Interactive byte heatmap widget — self-contained HTML.

Renders a 1D or 2D int8 array as an interactive grid of coloured cells.
Hover shows byte value. Zoom reveals individual cells.
"""

from __future__ import annotations

import json
import math

import numpy as np

from ultragraph.viz.svg import _byte_rgb, _hexrgb

_HEATMAP_JS_TEMPLATE = r"""(() => {
  const DATA = __DATA__;
  const W = __W__;
  const H = __H__;
  const CELL = __CELL__;
  const GAP = __GAP__;
  const PAD = __PAD__;
  const TITLE = __TITLE__;

  const canvas = document.getElementById('hm-canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = W;
  canvas.height = H;

  let zoom = 1, ox = PAD, oy = PAD;

  function draw() {
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, W, H);
    ctx.save();
    ctx.translate(ox, oy);
    ctx.scale(zoom, zoom);

    for (let r = 0; r < DATA.rows; r++) {
      for (let c = 0; c < DATA.cols; c++) {
        const v = DATA.grid[r * DATA.cols + c];
        const x = c * (CELL + GAP);
        const y = r * (CELL + GAP);
        ctx.fillStyle = DATA.colors[r * DATA.cols + c];
        ctx.fillRect(x, y, CELL, CELL);
      }
    }
    ctx.restore();

    if (TITLE) {
      ctx.fillStyle = '#222';
      ctx.font = '12px sans-serif';
      ctx.fillText(TITLE, PAD, 16);
    }
  }

  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const dz = e.deltaY < 0 ? 1.1 : 0.9;
    zoom *= dz;
    zoom = Math.max(0.2, Math.min(5, zoom));
    draw();
  });

  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left - ox) / zoom;
    const my = (e.clientY - rect.top - oy) / zoom;
    const c = Math.floor(mx / (CELL + GAP));
    const r = Math.floor(my / (CELL + GAP));
    const tip = document.getElementById('hm-tooltip');
    if (r >= 0 && r < DATA.rows && c >= 0 && c < DATA.cols) {
      const v = DATA.grid[r * DATA.cols + c];
      tip.style.display = 'block';
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top = (e.clientY - 8) + 'px';
      tip.textContent = `[${r},${c}] = ${v}`;
    } else {
      tip.style.display = 'none';
    }
  });

  draw();
})();
"""


def heatmap_widget(arr, cols=None, cell=12, gap=1, pad=8, title="") -> str:
    """Render a 1D or 2D int8 array as an interactive heatmap widget.

    Returns a complete HTML string.
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        n = int(a.shape[0])
        if cols is None:
            cols = max(1, int(math.ceil(math.sqrt(n)))) if n else 1
        cols = max(1, int(cols))
        rows = int(math.ceil(n / cols)) if n else 0
        flat = a.astype(int)
        grid = []
        colors = []
        for i in range(rows * cols):
            if i < n:
                grid.append(int(flat[i]))
                colors.append(_hexrgb(_byte_rgb(int(flat[i]))))
            else:
                grid.append(0)
                colors.append("#f0f0f0")
    elif a.ndim == 2:
        rows, cols = int(a.shape[0]), int(a.shape[1])
        flat = a.astype(int).flatten()
        grid = [int(v) for v in flat]
        colors = [_hexrgb(_byte_rgb(int(v))) for v in flat]
    else:
        raise ValueError("heatmap_widget expects a 1-D or 2-D array")

    content_w = cols * (cell + gap) - gap if cols else 0
    content_h = rows * (cell + gap) - gap if rows else 0
    w = max(200, int(content_w * 1.5) + 2 * pad)
    h = max(150, int(content_h * 1.5) + 2 * pad)

    data = {
        "rows": rows,
        "cols": cols,
        "grid": grid,
        "colors": colors,
    }

    js = (
        _HEATMAP_JS_TEMPLATE.replace("__DATA__", json.dumps(data))
        .replace("__W__", str(w))
        .replace("__H__", str(h))
        .replace("__CELL__", str(cell))
        .replace("__GAP__", str(gap))
        .replace("__PAD__", str(pad))
        .replace("__TITLE__", json.dumps(title))
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>byte heatmap</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ display:flex; justify-content:center; align-items:center; min-height:100vh;
         background:#f0f2f5; font-family:-apple-system,BlinkMacSystemFont,sans-serif; }}
  #hm-container {{ position:relative; border-radius:8px; overflow:hidden;
                  box-shadow:0 2px 16px rgba(0,0,0,0.12); background:#fff; }}
  #hm-canvas {{ display:block; }}
  #hm-tooltip {{ display:none; position:fixed; background:rgba(0,0,0,0.85); color:#fff;
                padding:4px 10px; border-radius:4px; font-size:12px; pointer-events:none;
                z-index:10; white-space:nowrap; }}
</style>
</head>
<body>
<div id="hm-container">
  <canvas id="hm-canvas"></canvas>
  <div id="hm-tooltip"></div>
</div>
<script>
{js}
</script>
</body>
</html>"""


def heatmap_to_file(arr, path: str, **kwargs):
    """Render a heatmap widget to an HTML file."""
    with open(path, "w") as f:
        f.write(heatmap_widget(arr, **kwargs))
