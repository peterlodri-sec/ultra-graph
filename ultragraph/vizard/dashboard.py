"""Live training dashboard — built-in HTTP server with real-time charts.

``dashboard.serve(port=8765)`` starts a lightweight HTTP server that serves
an HTML page with live-updating charts: loss curve, gradient norms, weight
histograms. Training metrics are pushed via a thread-safe callback.

Uses only Python stdlib — no flask, no fastapi, no websockets. Charts are
rendered with inline HTML5 Canvas JavaScript.

Usage:
    from ultragraph.vizard.dashboard import Dashboard

    dash = Dashboard(port=8765)
    dash.start()

    for step in range(steps):
        loss = train_step(...)
        dash.push("loss", loss)
        dash.push("grad_norm", compute_grad_norm(model))
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ultragraph training dashboard</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0f0f1a; color:#dde; font-family:-apple-system,BlinkMacSystemFont,sans-serif;
         display:flex; flex-direction:column; align-items:center; padding:24px; min-height:100vh; }
  h1 { font-size:20px; margin-bottom:8px; color:#aad; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:16px;
          width:100%; max-width:1100px; }
  .card { background:#1a1a2e; border-radius:10px; padding:16px; }
  .card h2 { font-size:13px; color:#99b; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px; }
  canvas { width:100%; height:180px; display:block; }
  .step { text-align:right; font-size:11px; color:#556; margin-top:4px; }
</style>
</head>
<body>
<h1>&#9878; ultragraph training dashboard</h1>
<div class="grid">
  <div class="card"><h2>Loss</h2><canvas id="c-loss"></canvas></div>
  <div class="card"><h2>Gradient Norm</h2><canvas id="c-grad"></canvas></div>
  <div class="card"><h2>Learning Rate</h2><canvas id="c-lr"></canvas></div>
  <div class="card"><h2>Weight Histogram</h2><canvas id="c-whist"></canvas></div>
</div>
<div class="step" id="step-display">step 0</div>
<script>
(() => {
  const charts = {
    loss:   { canvas: document.getElementById('c-loss'),  data: [], color: '#e05555', maxPts: 200 },
    grad:   { canvas: document.getElementById('c-grad'),  data: [], color: '#55a0e0', maxPts: 200 },
    lr:     { canvas: document.getElementById('c-lr'),    data: [], color: '#e0c055', maxPts: 200 },
    whist:  { canvas: document.getElementById('c-whist'), data: [], color: '#55e0a0', maxPts: 50, isHist: true },
  };

  function drawChart(chart) {
    const c = chart.canvas;
    const ctx = c.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    c.width = c.offsetWidth * dpr;
    c.height = c.offsetHeight * dpr;
    ctx.scale(dpr, dpr);
    const w = c.offsetWidth, h = c.offsetHeight;

    ctx.fillStyle = '#111128';
    ctx.fillRect(0, 0, w, h);

    if (!chart.data.length) return;

    if (chart.isHist) {
      const bins = chart.data[chart.data.length - 1];
      if (!bins || !bins.length) return;
      const max = Math.max(...bins, 1);
      const bw = w / bins.length;
      bins.forEach((v, i) => {
        const bh = (v / max) * (h - 20);
        ctx.fillStyle = chart.color;
        ctx.fillRect(i * bw, h - bh - 12, bw - 1, bh);
      });
    } else {
      const pts = chart.data.slice(-chart.maxPts);
      if (pts.length < 2) return;
      const min = Math.min(...pts), max = Math.max(...pts);
      const range = max - min || 1;
      ctx.strokeStyle = chart.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      pts.forEach((v, i) => {
        const x = (i / (pts.length - 1)) * (w - 20) + 10;
        const y = h - 16 - ((v - min) / range) * (h - 32);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = '#889'; ctx.font = '9px monospace';
      ctx.fillText(min.toFixed(3), 4, h - 4);
      ctx.fillText(max.toFixed(3), w - 60, 14);
    }
  }

  function drawAll() { Object.values(charts).forEach(drawChart); }
  drawAll();

  const es = new EventSource('/stream');
  es.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.key === 'loss')  charts.loss.data.push(d.value);
    if (d.key === 'grad')  charts.grad.data.push(d.value);
    if (d.key === 'lr')    charts.lr.data.push(d.value);
    if (d.key === 'whist') charts.whist.data.push(d.value);
    if (d.key === 'step')  document.getElementById('step-display').textContent = 'step ' + d.value;
    drawAll();
  };
})();
</script>
</body>
</html>"""


class Dashboard:
    """Live training dashboard with built-in HTTP server.

    Thread-safe. Push metrics from the training loop; the dashboard
    streams them to any connected browser via Server-Sent Events.

    Example:
        dash = Dashboard(port=8765)
        dash.start()
        for i in range(1000):
            loss = train_step()
            dash.push("loss", loss)
            dash.push("step", i)
    """

    def __init__(self, port: int = 8765):
        self.port = port
        self._metrics: dict[str, list] = {}
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None
        self._clients: list[_SSEClient] = []
        self._thread: threading.Thread | None = None

    def start(self):
        """Start the HTTP server in a background thread."""
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # silent

            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(_DASHBOARD_HTML.encode())
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    client = _SSEClient(self.wfile)
                    dashboard._add_client(client)
                    try:
                        while True:
                            time.sleep(0.1)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        dashboard._remove_client(client)
                else:
                    self.send_response(404)
                    self.end_headers()

        self._server = HTTPServer(("0.0.0.0", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server = None

    def push(self, key: str, value):
        """Push a metric value to all connected clients.

        Args:
            key: Metric name (e.g. "loss", "grad", "lr", "whist", "step")
            value: Numeric value, or list of ints for histogram bins
        """
        with self._lock:
            msg = json.dumps({"key": key, "value": value}) + "\n"
            dead = []
            for c in self._clients:
                try:
                    c.send(msg)
                except Exception:
                    dead.append(c)
            for c in dead:
                self._clients.remove(c)

    def _add_client(self, client: "_SSEClient"):
        with self._lock:
            self._clients.append(client)
            # replay existing metrics
            for key, values in self._metrics.items():
                if values:
                    try:
                        client.send(json.dumps({"key": key, "value": values[-1]}) + "\n")
                    except Exception:
                        pass

    def _remove_client(self, client: "_SSEClient"):
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class _SSEClient:
    def __init__(self, wfile):
        self._wfile = wfile

    def send(self, data: str):
        self._wfile.write(f"data: {data}\n\n".encode())
        self._wfile.flush()
