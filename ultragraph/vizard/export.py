"""Frame-based export: GIF, MP4, PNG sequences from Scenes and Animations.

All exports are optional — dependencies (pillow, imageio) are imported lazily.
GIF export works with just pillow. MP4 requires imageio + ffmpeg.

Frame rendering uses PIL ImageDraw to paint the scene directly — no browser,
no canvas, no JavaScript. Frames are pure Python raster images.
"""

from __future__ import annotations

from ultragraph.vizard.core.scene import Camera, Scene


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = (max(0, min(255, int(round(v)))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _project_point(x: float, y: float, cam: Camera, scene: Scene) -> tuple[float, float]:
    sx = (x + cam.x) * cam.zoom + scene.width / 2
    sy = (y + cam.y) * cam.zoom + scene.height / 3
    return sx, sy


def _render_frame_pil(scene: Scene, camera: Camera | None = None):
    """Render a single scene frame to a PIL Image."""
    from PIL import Image, ImageDraw, ImageFont

    cam = camera or scene.camera
    img = Image.new("RGB", (scene.width, scene.height), scene.background)
    draw = ImageDraw.Draw(img)

    def load_font(size: int):
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except OSError:
            try:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
            except OSError:
                return ImageFont.load_default()

    # edges
    for e in scene.edges:
        src = next((n for n in scene.nodes if n.id == e.src_id), None)
        dst = next((n for n in scene.nodes if n.id == e.dst_id), None)
        if src is None or dst is None:
            continue
        x1, y1 = _project_point(src.x, src.y, cam, scene)
        x2, y2 = _project_point(dst.x, dst.y, cam, scene)
        color = tuple(max(0, min(255, int(c))) for c in e.color)
        width = max(1, int(e.width * cam.zoom))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

    # nodes
    for n in scene.nodes:
        sx, sy = _project_point(n.x, n.y, cam, scene)
        r = max(1, int(n.radius * cam.zoom))
        color = tuple(max(0, min(255, int(c))) for c in n.color)
        draw.ellipse(
            [(sx - r, sy - r), (sx + r, sy + r)],
            fill=color,
            outline=(80, 80, 80),
            width=max(1, int(cam.zoom)),
        )
        if n.label:
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = (255, 255, 255) if lum < 140 else (30, 30, 30)
            font = load_font(max(8, int(10 * cam.zoom)))
            bbox = draw.textbbox((0, 0), n.label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((sx - tw / 2, sy - th / 2 - 1), n.label, fill=text_color, font=font)

    # labels
    for ll in scene.labels:
        sx, sy = _project_point(ll.x, ll.y, cam, scene)
        color = tuple(max(0, min(255, int(c))) for c in ll.color)
        font = load_font(max(8, int(ll.font_size * cam.zoom)))
        draw.text((sx, sy), ll.text, fill=color, font=font)

    # title
    if scene.title:
        draw.text((10, 6), scene.title, fill=(40, 40, 40))

    return img


def export_gif(animation, path: str, fps: int = 24, duration: float | None = None):
    """Export an Animation as an animated GIF.

    Requires: ``pip install pillow``

    Args:
        animation: An Animation object with keyframes and scene
        path: Output file path (e.g. 'forward.gif')
        fps: Frames per second
        duration: Override total duration (seconds). Default: animation.duration
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError("GIF export requires pillow: pip install pillow")

    dur = duration or animation.duration
    total_frames = max(1, int(dur * fps))
    frames = []

    for fi in range(total_frames):
        t = fi / fps
        cam = _interpolate_camera(animation, t)
        frame_img = _render_frame_pil(animation.scene, cam)
        frames.append(frame_img.convert("P", palette=PILImage.ADAPTIVE, colors=256))

    if frames:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0 if animation.loop else 1,
        )


def export_mp4(animation, path: str, fps: int = 24, duration: float | None = None):
    """Export an Animation as an MP4 video.

    Requires: ``pip install imageio imageio-ffmpeg``

    Args:
        animation: An Animation object with keyframes and scene
        path: Output file path (e.g. 'forward.mp4')
        fps: Frames per second
        duration: Override total duration (seconds)
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("MP4 export requires imageio: pip install imageio imageio-ffmpeg")

    dur = duration or animation.duration
    total_frames = max(1, int(dur * fps))
    frames = []

    for fi in range(total_frames):
        t = fi / fps
        cam = _interpolate_camera(animation, t)
        frame_img = _render_frame_pil(animation.scene, cam)
        frames.append(frame_img)

    with imageio.get_writer(path, fps=fps, format="FFMPEG") as writer:
        for frame in frames:
            writer.append_data(frame)


def export_frame(scene: Scene, path: str, camera: Camera | None = None):
    """Export a single scene frame as a PNG.

    Requires: ``pip install pillow``
    """
    img = _render_frame_pil(scene, camera)
    img.save(path)


def _interpolate_camera(animation, t: float) -> Camera | None:
    """Interpolate camera between keyframes at time t."""
    kfs = animation.keyframes
    if not kfs:
        return animation.scene.camera
    if len(kfs) == 1:
        return kfs[0].camera

    duration = animation.duration
    loop_t = (t % duration) if animation.loop else min(t, duration)

    lo = kfs[0]
    for kf in kfs[1:]:
        if kf.time >= loop_t:
            frac = (loop_t - lo.time) / (kf.time - lo.time + 1e-6)
            frac = max(0.0, min(1.0, frac))
            return Camera(
                x=lo.camera.x + (kf.camera.x - lo.camera.x) * frac,
                y=lo.camera.y + (kf.camera.y - lo.camera.y) * frac,
                z=lo.camera.z + (kf.camera.z - lo.camera.z) * frac,
                zoom=lo.camera.zoom + (kf.camera.zoom - lo.camera.zoom) * frac,
                rotation_x=lo.camera.rotation_x + (kf.camera.rotation_x - lo.camera.rotation_x) * frac,
                rotation_y=lo.camera.rotation_y + (kf.camera.rotation_y - lo.camera.rotation_y) * frac,
                rotation_z=lo.camera.rotation_z + (kf.camera.rotation_z - lo.camera.rotation_z) * frac,
            )
        lo = kf

    return kfs[-1].camera
