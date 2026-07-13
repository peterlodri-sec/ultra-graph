"""HTML5 Canvas renderer — self-contained interactive visualisation.

Produces a complete, standalone HTML document with embedded CSS and
JavaScript. No external dependencies, no CDN, no build step. The output
works offline in any modern browser.

Each HTML document includes:
- A <canvas> element for the scene
- Embedded JavaScript for interactivity (pan, zoom, hover)
- All data inlined — no server required
"""

from __future__ import annotations

import json

from ultragraph.vizard.core.scene import Scene

_SCENE_JS_TEMPLATE = r"""(() => {
  const DATA = __DATA__;
  const W = __WIDTH__;
  const H = __HEIGHT__;
  const BG = __BG__;
  const TITLE = __TITLE__;

  const canvas = document.getElementById('ug-canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = W;
  canvas.height = H;

  let cam = { x: 0, y: 0, zoom: 1 };
  let hovered = null;
  let tooltip = document.getElementById('ug-tooltip');

  function project(n) {
    const sx = (n.x + cam.x) * cam.zoom + W / 2;
    const sy = (n.y + cam.y) * cam.zoom + H / 3;
    return [sx, sy];
  }

  function draw() {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);
    ctx.save();

    // edges
    DATA.edges.forEach(e => {
      const src = DATA.nodes.find(n => n.id === e.src);
      const dst = DATA.nodes.find(n => n.id === e.dst);
      if (!src || !dst) return;
      const [x1, y1] = project(src);
      const [x2, y2] = project(dst);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = `rgba(${e.color.join(',')},${e.opacity})`;
      ctx.lineWidth = e.width * cam.zoom;
      if (e.kind === 'dashed') ctx.setLineDash([4, 4]);
      else ctx.setLineDash([]);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // nodes
    DATA.nodes.forEach(n => {
      const [sx, sy] = project(n);
      const r = n.radius * cam.zoom;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${n.color.join(',')},${n.opacity})`;
      ctx.fill();
      ctx.strokeStyle = '#444';
      ctx.lineWidth = Math.max(0.5, 1 * cam.zoom);
      ctx.stroke();
      if (n.label) {
        ctx.fillStyle = n.color[0] * 0.299 + n.color[1] * 0.587 + n.color[2] * 0.114 > 140 ? '#111' : '#fff';
        ctx.font = `${Math.max(8, 10 * cam.zoom)}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(n.label, sx, sy);
      }
    });

    // labels
    DATA.labels.forEach(l => {
      const [sx, sy] = project(l);
      ctx.fillStyle = `rgb(${l.color.join(',')})`;
      ctx.font = `${l.bold ? 'bold ' : ''}${Math.max(8, l.font_size * cam.zoom)}px sans-serif`;
      ctx.textAlign = l.anchor;
      ctx.textBaseline = 'top';
      ctx.fillText(l.text, sx, sy);
    });

    ctx.restore();

    // title
    if (TITLE) {
      ctx.fillStyle = '#222';
      ctx.font = '13px sans-serif';
      ctx.textAlign = 'start';
      ctx.fillText(TITLE, 12, 18);
    }
  }

  function hitTest(mx, my) {
    for (let i = DATA.nodes.length - 1; i >= 0; i--) {
      const n = DATA.nodes[i];
      const [sx, sy] = project(n);
      const r = n.radius * cam.zoom;
      if (Math.hypot(mx - sx, my - sy) < r) return n;
    }
    return null;
  }

  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const h = hitTest(mx, my);
    if (h !== hovered) {
      hovered = h;
      canvas.style.cursor = h ? 'pointer' : 'grab';
      if (h) {
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 12) + 'px';
        tooltip.style.top = (e.clientY - 8) + 'px';
        tooltip.textContent = h.label || h.id;
      } else {
        tooltip.style.display = 'none';
      }
    }
    if (hovered && tooltip.style.display === 'block') {
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top = (e.clientY - 8) + 'px';
    }
  });

  let dragging = false, lastX, lastY;
  canvas.addEventListener('mousedown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.style.cursor = 'grabbing'; });
  canvas.addEventListener('mouseup', () => { dragging = false; canvas.style.cursor = hovered ? 'pointer' : 'grab'; });
  canvas.addEventListener('mouseleave', () => { dragging = false; });
  canvas.addEventListener('mousemove', e => {
    if (!dragging) return;
    cam.x += (e.clientX - lastX) / cam.zoom;
    cam.y += (e.clientY - lastY) / cam.zoom;
    lastX = e.clientX;
    lastY = e.clientY;
    draw();
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const dz = e.deltaY < 0 ? 1.1 : 0.9;
    cam.zoom *= dz;
    cam.zoom = Math.max(0.1, Math.min(10, cam.zoom));
    draw();
  });

  draw();
})();
"""


def _color_tuple(c: tuple[int, int, int]) -> list[int]:
    return [int(max(0, min(255, v))) for v in c]


def _serialize_scene(scene: Scene) -> dict:
    return {
        "nodes": [
            {
                "id": n.id,
                "x": n.x - scene.width / 2,
                "y": n.y - scene.height / 3,
                "radius": n.radius,
                "label": n.label,
                "color": _color_tuple(n.color),
                "opacity": n.opacity,
            }
            for n in scene.nodes
        ],
        "edges": [
            {
                "src": e.src_id,
                "dst": e.dst_id,
                "kind": e.kind,
                "color": _color_tuple(e.color),
                "width": e.width,
                "opacity": e.opacity,
            }
            for e in scene.edges
        ],
        "labels": [
            {
                "id": ll.id,
                "x": ll.x - scene.width / 2,
                "y": ll.y - scene.height / 3,
                "text": ll.text,
                "font_size": ll.font_size,
                "color": _color_tuple(ll.color),
                "bold": ll.bold,
                "anchor": ll.anchor,
            }
            for ll in scene.labels
        ],
    }


def render_html(scene: Scene, title: str = "") -> str:
    """Render a Scene to a self-contained interactive HTML document.

    The output is a complete HTML file with embedded CSS and JavaScript.
    Drop it in a browser — no server, no dependencies.
    """
    data = _serialize_scene(scene)
    js = (
        _SCENE_JS_TEMPLATE.replace("__DATA__", json.dumps(data))
        .replace("__WIDTH__", str(scene.width))
        .replace("__HEIGHT__", str(scene.height))
        .replace("__BG__", json.dumps(f"rgb({scene.background[0]},{scene.background[1]},{scene.background[2]})"))
        .replace("__TITLE__", json.dumps(title or scene.title))
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape_html(title or scene.title or "ultragraph viz")}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ display:flex; justify-content:center; align-items:center; min-height:100vh;
         background:#f0f2f5; font-family:-apple-system,BlinkMacSystemFont,sans-serif; }}
  #ug-container {{ position:relative; border-radius:8px; overflow:hidden;
                  box-shadow:0 2px 16px rgba(0,0,0,0.12); background:#fff; }}
  #ug-canvas {{ display:block; }}
  #ug-tooltip {{ display:none; position:fixed; background:rgba(0,0,0,0.85); color:#fff;
                padding:4px 10px; border-radius:4px; font-size:12px; pointer-events:none;
                z-index:10; white-space:nowrap; }}
  #ug-legend {{ position:absolute; bottom:12px; right:16px; font-size:10px; color:#888; }}
</style>
</head>
<body>
<div id="ug-container">
  <canvas id="ug-canvas"></canvas>
  <div id="ug-tooltip"></div>
  <div id="ug-legend">drag to pan &bull; scroll to zoom</div>
</div>
<script>
{js}
</script>
</body>
</html>"""


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_to_file(scene: Scene, path: str, title: str = ""):
    """Render a scene to an HTML file."""
    with open(path, "w") as f:
        f.write(render_html(scene, title=title))
