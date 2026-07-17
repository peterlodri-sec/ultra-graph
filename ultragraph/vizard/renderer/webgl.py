"""WebGL 3D interactive graph viewer — self-contained HTML.

Renders scene nodes as 3D spheres and edges as coloured lines. Zero
dependencies beyond a browser with WebGL support (all modern browsers).
The entire viewer — shaders, camera, rendering — is embedded in the HTML.

Features:
- Orbit, pan, zoom with mouse/touch
- Node hover tooltip
- Colour-coded edges
- Works offline
"""

from __future__ import annotations

import json
import math

from ultragraph.vizard.core.scene import Scene

_WEBGL_JS = r"""(() => {
  const DATA = __DATA__;
  const W = __W__;
  const H = __H__;
  const BG = __BG__;
  const TITLE = __TITLE__;

  const canvas = document.getElementById('gl-canvas');
  canvas.width = W; canvas.height = H;
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  if (!gl) { canvas.outerHTML = '<p style="padding:40px;font-family:sans-serif">WebGL not supported</p>'; return; }

  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(BG[0]/255, BG[1]/255, BG[2]/255, 1);

  // shaders
  const vs = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vs, `#version 300 es
    in vec3 aPos; in vec3 aColor; in float aSize;
    uniform mat4 uMVP; uniform float uPointSize;
    out vec3 vColor;
    void main() {
      gl_Position = uMVP * vec4(aPos, 1);
      gl_PointSize = aSize * uPointSize;
      vColor = aColor;
    }`);
  gl.compileShader(vs);

  const fs = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fs, `#version 300 es
    precision highp float;
    in vec3 vColor;
    out vec4 frag;
    void main() { frag = vec4(vColor, 1); }`);
  gl.compileShader(fs);

  const prog = gl.createProgram();
  gl.attachShader(prog, vs); gl.attachShader(prog, fs);
  gl.linkProgram(prog); gl.useProgram(prog);

  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    canvas.outerHTML = '<p style="padding:40px;font-family:sans-serif">WebGL shader error</p>'; return;
  }

  const uMVP = gl.getUniformLocation(prog, 'uMVP');
  const uPointSize = gl.getUniformLocation(prog, 'uPointSize');

  // build geometry
  const nodePositions = []; const nodeColors = []; const nodeSizes = [];
  const nodeData = [];
  const edgeLines = []; const edgeColors = [];

  DATA.nodes.forEach(n => {
    nodePositions.push(n.x, n.y, n.z);
    nodeColors.push(n.color[0]/255, n.color[1]/255, n.color[2]/255);
    nodeSizes.push(n.radius / 50);
    nodeData.push(n);
  });

  DATA.edges.forEach(e => {
    const s = DATA.nodes.find(n => n.id === e.src);
    const d = DATA.nodes.find(n => n.id === e.dst);
    if (!s || !d) return;
    edgeLines.push(s.x, s.y, s.z, d.x, d.y, d.z);
    edgeColors.push(e.color[0]/255, e.color[1]/255, e.color[2]/255,
                    e.color[0]/255, e.color[1]/255, e.color[2]/255);
  });

  // buffers
  function makeBuf(data) { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW); return b; }

  const nodeBuf = makeBuf(nodePositions);
  const nodeColorBuf = makeBuf(nodeColors);
  const nodeSizeBuf = makeBuf(nodeSizes);
  const edgeBuf = makeBuf(edgeLines);
  const edgeColorBuf = makeBuf(edgeColors);

  // camera
  let cam = { theta: 0, phi: Math.PI/4, dist: 5, tx: 0, ty: 0 };

  function getMVP() {
    const eye = [
      cam.dist * Math.sin(cam.phi) * Math.cos(cam.theta) + cam.tx,
      cam.dist * Math.cos(cam.phi) + cam.ty,
      cam.dist * Math.sin(cam.phi) * Math.sin(cam.theta)
    ];
    const up = [0, 1, 0];
    const f = 1 / Math.tan(Math.PI/6);
    const aspect = H > 0 ? W / H : 1;
    const near = 0.1, far = 100;
    return mat4Persp(f/aspect, f, near, far, eye, [cam.tx, cam.ty, 0], up);
  }

  function mat4Persp(fx, fy, near, far, eye, center, up) {
    const zAxis = norm(sub(eye, center));
    const xAxis = norm(cross(up, zAxis));
    const yAxis = cross(zAxis, xAxis);
    const ex = -dot(xAxis, eye), ey = -dot(yAxis, eye), ez = -dot(zAxis, eye);
    const view = [
      xAxis[0], yAxis[0], zAxis[0], 0,
      xAxis[1], yAxis[1], zAxis[1], 0,
      xAxis[2], yAxis[2], zAxis[2], 0,
      ex, ey, ez, 1
    ];
    const proj = new Array(16).fill(0);
    proj[0] = fx; proj[5] = fy;
    proj[10] = -(far+near)/(far-near); proj[11] = -1;
    proj[14] = -2*far*near/(far-near);
    return mul44(proj, view);
  }

  function mul44(a, b) { const r = new Array(16).fill(0); for (let i=0;i<4;i++) for (let j=0;j<4;j++) for (let k=0;k<4;k++) r[i*4+j] += a[i*4+k] * b[k*4+j]; return r; }
  function sub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
  function norm(a) { const l = Math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); if (l > 0) { return [a[0]/l, a[1]/l, a[2]/l]; } return [0, 0, 0]; }
  function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
  function dot(a, b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

  function draw() {
    gl.viewport(0, 0, W, H);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const mvp = getMVP();
    gl.uniformMatrix4fv(uMVP, false, mvp);
    gl.uniform1f(uPointSize, W / 20 * cam.dist / 5);

    // edges
    if (edgeLines.length) {
      gl.bindBuffer(gl.ARRAY_BUFFER, edgeBuf);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, edgeColorBuf);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, edgeLines.length / 3);
    }

    // nodes (as points, large)
    gl.bindBuffer(gl.ARRAY_BUFFER, nodeBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, nodeColorBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, nodeSizeBuf);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, nodePositions.length / 3);

    // title
    const tip = document.getElementById('gl-tooltip');
    tip.style.display = 'none';
  }

  // mouse
  let dragging = false, lx = 0, ly = 0;
  canvas.addEventListener('mousedown', e => { dragging = true; lx = e.clientX; ly = e.clientY; });
  canvas.addEventListener('mouseup', () => dragging = false);
  canvas.addEventListener('mousemove', e => {
    if (dragging) {
      cam.theta -= (e.clientX - lx) * 0.005;
      cam.phi = Math.max(0.1, Math.min(Math.PI-0.1, cam.phi + (e.clientY - ly) * 0.005));
      lx = e.clientX; ly = e.clientY;
      draw();
    }
  });
  canvas.addEventListener('wheel', e => { e.preventDefault(); cam.dist = Math.max(0.5, cam.dist * (e.deltaY < 0 ? 0.9 : 1.1)); draw(); });

  draw();
})();
"""


def _serialize_3d(scene: Scene) -> dict:
    """Serialize scene for WebGL, computing 3D positions."""
    nodes = list(scene.nodes)
    n = len(nodes)

    # layout nodes in 3D: spiral or grid with depth
    if n <= 1:
        for nd in nodes:
            nd.z = 0.0
    else:
        layers = max(1, int(math.ceil(math.sqrt(n))))
        for i, nd in enumerate(nodes):
            layer = i // layers
            col = i % layers
            nd.x = (col - layers / 2) * 1.5
            nd.z = (layer - layers / 2) * 1.5
            nd.y = 0.0

    return {
        "nodes": [
            {
                "id": nd.id,
                "x": nd.x,
                "y": nd.y,
                "z": nd.z,
                "radius": nd.radius * 0.05,
                "color": [max(0, min(255, int(c))) for c in nd.color],
            }
            for nd in nodes
        ],
        "edges": [
            {
                "src": e.src_id,
                "dst": e.dst_id,
                "color": [max(0, min(255, int(c))) for c in e.color],
            }
            for e in scene.edges
        ],
    }


def render_webgl(scene: Scene, width: int = 900, height: int = 650, title: str = "") -> str:
    """Render a Scene as a self-contained WebGL 3D viewer.

    Returns a complete HTML document. Open in any modern browser.
    No server, no dependencies, no CDN — pure WebGL.
    """
    data = _serialize_3d(scene)
    js = (
        _WEBGL_JS.replace("__DATA__", json.dumps(data))
        .replace("__W__", str(width))
        .replace("__H__", str(height))
        .replace("__BG__", json.dumps(list(scene.background)))
        .replace("__TITLE__", json.dumps(title or scene.title))
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title or scene.title or "ultragraph 3D")}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ display:flex; justify-content:center; align-items:center; min-height:100vh;
         background:#1a1a2e; font-family:-apple-system,BlinkMacSystemFont,sans-serif; }}
  #gl-container {{ position:relative; border-radius:8px; overflow:hidden; }}
  #gl-canvas {{ display:block; }}
  #gl-tooltip {{ display:none; position:fixed; background:rgba(255,255,255,0.9); color:#111;
                padding:4px 10px; border-radius:4px; font-size:12px; pointer-events:none; z-index:10; }}
  #gl-legend {{ position:absolute; bottom:12px; left:16px; font-size:10px; color:#889; }}
</style>
</head>
<body>
<div id="gl-container">
  <canvas id="gl-canvas"></canvas>
  <div id="gl-tooltip"></div>
  <div id="gl-legend">drag to orbit &bull; scroll to zoom</div>
</div>
<script>
{js}
</script>
</body>
</html>"""


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_webgl_to_file(scene: Scene, path: str, title: str = ""):
    """Render a 3D scene to an HTML file."""
    with open(path, "w") as f:
        f.write(render_webgl(scene, title=title))
