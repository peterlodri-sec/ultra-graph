"""Animation engine for ultra-graph visualisation.

Produces keyframe-based animations that render as self-contained HTML
documents with embedded JavaScript animation loops. No video codec required.
For GIF/MP4 export, use the optional ``pillow`` dependency.

The animation system works by:
1. Capturing scene states at keyframe positions
2. Interpolating between keyframes in JavaScript
3. Rendering frames in a requestAnimationFrame loop
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from ultragraph.vizard.core.scene import Camera, Scene
from ultragraph.vizard.renderer.html5 import render_html


@dataclass
class Keyframe:
    """A scene state at a specific time (seconds)."""

    time: float
    camera: Camera
    label: str = ""


@dataclass
class Animation:
    """A sequence of keyframes defining a camera animation over a scene."""

    scene: Scene
    keyframes: list[Keyframe] = field(default_factory=list)
    duration: float = 5.0
    loop: bool = True

    def add_keyframe(self, camera: Camera, time: float, label: str = ""):
        self.keyframes.append(Keyframe(time=time, camera=camera, label=label))
        self.keyframes.sort(key=lambda k: k.time)
        self.duration = max(self.duration, time)


def animate_forward_pass(ug, duration: float = 5.0, frames: int = 3) -> str:
    """Create an animated flythrough of an ultra-graph topology.

    The animation orbits the camera around the graph, starting wide and
    zooming into the center. Returns a self-contained HTML document.

    Args:
        ug: An UltraGraph instance
        duration: Total animation duration in seconds
        frames: Number of keyframes (evenly spaced)

    Returns:
        Complete HTML string with embedded animation
    """
    from ultragraph.vizard.core.scene import scene_from_ultragraph

    scene = scene_from_ultragraph(ug)
    scene.width = 900
    scene.height = 600

    cx = scene.width / 2
    cy = scene.height / 2

    anim = Animation(scene=scene, duration=duration, loop=True)

    for i in range(frames):
        t = i * duration / max(1, frames - 1)
        angle = t / duration * math.pi * 2
        radius = max(50.0, 200.0 - t * 30)
        cam = Camera(
            x=cx + math.cos(angle) * radius * 0.3,
            y=cy + math.sin(angle) * radius * 0.15,
            z=100.0 + radius,
            rotation_y=angle * 0.5,
            zoom=1.0 + t / duration * 0.5,
        )
        anim.add_keyframe(cam, t, f"frame {i}")

    return _render_animation_html(anim)


def _render_animation_html(anim: Animation) -> str:
    """Render an Animation to a self-contained HTML document with JS playback."""
    keyframe_data = [
        {
            "t": kf.time,
            "cx": kf.camera.x,
            "cy": kf.camera.y,
            "cz": kf.camera.z,
            "rx": kf.camera.rotation_x,
            "ry": kf.camera.rotation_y,
            "rz": kf.camera.rotation_z,
            "zoom": kf.camera.zoom,
        }
        for kf in anim.keyframes
    ]

    # Use the renderer for the base scene, then inject animation script
    base_html = render_html(anim.scene, title=anim.scene.title)
    loop_str = "true" if anim.loop else "false"

    anim_inject = f"""<script>
{{
  const keyframes = {json.dumps(keyframe_data)};
  const duration = {anim.duration};
  const loop = {loop_str};
  let startTime = null;

  function lerp(a, b, t) {{ return a + (b - a) * t; }}

  function getCam(time) {{
    if (keyframes.length === 0) return null;
    if (keyframes.length === 1) return keyframes[0];
    const t = loop ? time % duration : Math.min(time, duration);
    let lo = keyframes[0];
    for (let i = 1; i < keyframes.length; i++) {{
      if (keyframes[i].t >= t) {{
        const frac = (t - lo.t) / (keyframes[i].t - lo.t + 0.0001);
        return {{
          x: lerp(lo.cx, keyframes[i].cx, frac),
          y: lerp(lo.cy, keyframes[i].cy, frac),
          zoom: lerp(lo.zoom, keyframes[i].zoom, frac),
        }};
      }}
      lo = keyframes[i];
    }}
    return keyframes[keyframes.length - 1];
  }}

  function tick(ts) {{
    if (!startTime) startTime = ts;
    const elapsed = (ts - startTime) / 1000;
    const cam = getCam(elapsed);
    if (cam && typeof window._ugCam !== 'undefined') {{
      window._ugCam.x = cam.x;
      window._ugCam.y = cam.y;
      window._ugCam.zoom = cam.zoom;
    }}
    requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}}
</script>
</body>"""

    return base_html.replace("</body>", anim_inject)
