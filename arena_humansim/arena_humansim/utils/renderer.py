from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from arena_humansim_msgs.msg import AgentFrame

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

from arena_humansim.utils.bag_io import AGENT_FRAME_TYPE, AGENT_META_TYPE, AGENT_STATES_TYPE, policy_names

_log = logging.getLogger("arena_humansim_render")


def _setup_logging(log_file: Path | None) -> None:
    fmt = logging.Formatter("[%(levelname)s] [%(name)s]: %(message)s")
    _log.setLevel(logging.INFO)
    _log.handlers.clear()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    _log.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        _log.addHandler(fh)


@dataclass
class Frame:
    t_ns: int
    agents: list[tuple[int, float, float, float, float, float, str, int]]


@dataclass
class Geometry:
    walls: list[tuple[tuple[float, float], tuple[float, float]]]
    obstacles: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]]
    world_objects: list[tuple[str, str, float, float]]


def _read_bag(bag_dir: Path) -> tuple[Geometry, list[Frame]]:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    type_map = {name: get_message(type_name) for name, type_name in topic_types.items()}

    geometry = Geometry(walls=[], obstacles=[], world_objects=[])
    frames: list[Frame] = []
    flat: list[tuple[int, AgentFrame]] = []
    metas: list[tuple[int, list[str]]] = []

    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        msg_type = type_map.get(topic)
        if msg_type is None:
            continue
        msg = deserialize_message(raw, msg_type)

        if topic.endswith("world_geometry"):
            geometry.walls = [((s.x, s.y), (e.x, e.y)) for s, e in zip(msg.wall_starts, msg.wall_ends)]
            geometry.obstacles = [((o.pose.x, o.pose.y, o.pose.theta), (o.bb_x_min, o.bb_x_max, o.bb_y_min, o.bb_y_max)) for o in msg.obstacles]
            geometry.world_objects = [(o.object_id, o.type, o.pose.x, o.pose.y) for o in msg.world_objects]
        elif topic_types[topic] == AGENT_META_TYPE:
            metas.append((t_ns, list(msg.policies)))
        elif topic.endswith("agent_states") and topic_types[topic] == AGENT_FRAME_TYPE:
            flat.append((t_ns, msg))
        elif topic.endswith("agent_states") and topic_types[topic] == AGENT_STATES_TYPE:
            agents = [(int(a.agent_id), a.pose.x, a.pose.y, a.pose.theta, a.velocity.x, a.velocity.y, a.policy or "", int(a.kind)) for a in msg.agents]
            frames.append(Frame(t_ns=t_ns, agents=agents))

    if flat:
        frames.extend(_flat_frames(flat, metas))
    return geometry, frames


def _flat_frames(flat: list[tuple[int, AgentFrame]], metas: list[tuple[int, list[str]]]) -> list[Frame]:
    frame_t = np.array([t for t, _ in flat], dtype=np.int64)
    counts = np.array([len(m.agent_id) for _, m in flat], dtype=np.int64)
    pidx = np.concatenate([np.asarray(m.policy_idx, dtype=np.int64) for _, m in flat])
    names = policy_names(np.array([t for t, _ in metas], dtype=np.int64), [p for _, p in metas], frame_t, counts, pidx).tolist()
    frames = []
    start = 0
    for (t_ns, m), n in zip(flat, counts.tolist(), strict=True):
        agents = list(zip(m.agent_id.tolist(), m.x.tolist(), m.y.tolist(), m.theta.tolist(), m.vx.tolist(), m.vy.tolist(), names[start : start + n], m.kind.tolist(), strict=True))
        frames.append(Frame(t_ns=t_ns, agents=agents))
        start += n
    return frames


_POLICY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def _policy_color(name: str, cache: dict[str, str]) -> str:
    if name not in cache:
        cache[name] = _POLICY_COLORS[len(cache) % len(_POLICY_COLORS)]
    return cache[name]


def _teardrop_template(n_arc: int = 48) -> np.ndarray:
    # Canonical teardrop pointing +x: sharp tip at (1, 0), body is a 3/4 arc of
    # radius 1/sqrt(2) centered at origin, corresponding to a rounded square
    # with border-radius: 0 50% 50% 50% (sharp corner aligned with heading).
    r_body = 1.0 / np.sqrt(2.0)
    theta = np.linspace(np.pi / 4.0, 7.0 * np.pi / 4.0, n_arc)
    arc = np.column_stack([r_body * np.cos(theta), r_body * np.sin(theta)])
    return np.vstack([[1.0, 0.0], arc])


_AGENT_TEMPLATE = _teardrop_template()


def _compute_bounds(geometry: Geometry, frames: list[Frame]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for (sx, sy), (ex, ey) in geometry.walls:
        xs.extend([sx, ex])
        ys.extend([sy, ey])
    for (px, py, _), (x_min, x_max, y_min, y_max) in geometry.obstacles:
        xs.extend([px + x_min, px + x_max])
        ys.extend([py + y_min, py + y_max])
    for _, _, x, y in geometry.world_objects:
        xs.append(x)
        ys.append(y)
    for frame in frames:
        for _, x, y, *_ in frame.agents:
            xs.append(x)
            ys.append(y)
    if not xs:
        return -10.0, 10.0, -10.0, 10.0
    pad = 1.0
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def render(bag_dir: Path, output: Path, fmt: str, fps: int, agent_radius: float = 0.35, dpi: int = 120) -> None:
    _log.info(f"reading bag {bag_dir}")
    geometry, frames = _read_bag(bag_dir)
    if not frames:
        raise RuntimeError(f"no /agent_states frames found in {bag_dir}")
    _log.info(f"read {len(frames)} frames, {len(geometry.walls)} walls, {len(geometry.obstacles)} obstacles")

    x_min, x_max, y_min, y_max = _compute_bounds(geometry, frames)

    data_w = max(x_max - x_min, 1e-6)
    data_h = max(y_max - y_min, 1e-6)
    max_fig = 10.0
    min_fig = 2.0
    if data_w >= data_h:
        fig_w = max_fig
        fig_h = max(min_fig, max_fig * data_h / data_w)
    else:
        fig_h = max_fig
        fig_w = max(min_fig, max_fig * data_w / data_h)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    for (sx, sy), (ex, ey) in geometry.walls:
        ax.plot([sx, ex], [sy, ey], color="black", linewidth=2, zorder=2)

    for (px, py, theta), (bx_min, bx_max, by_min, by_max) in geometry.obstacles:
        w = bx_max - bx_min
        h = by_max - by_min
        rect = patches.Rectangle(
            (bx_min, by_min),
            w,
            h,
            linewidth=1,
            edgecolor="dimgray",
            facecolor="lightgray",
            alpha=0.6,
            zorder=1,
        )
        from matplotlib.transforms import Affine2D

        rect.set_transform(Affine2D().rotate(theta).translate(px, py) + ax.transData)
        ax.add_patch(rect)

    for _, wo_type, x, y in geometry.world_objects:
        ax.plot(x, y, marker="s", markersize=10, color="goldenrod", zorder=3)
        ax.annotate(wo_type, (x, y), fontsize=7, ha="center", va="bottom")

    all_ids = sorted({aid for f in frames for aid, *_ in f.agents})
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}
    n_ids = len(all_ids)

    policy_cache: dict[str, str] = {}
    rgba_hidden = np.array([0.0, 0.0, 0.0, 0.0])

    face_colors = np.tile(rgba_hidden, (n_ids, 1))
    edge_colors = np.tile([0.0, 0.0, 0.0, 1.0], (n_ids, 1))
    line_widths = np.full(n_ids, 0.5)

    scaled_template = _AGENT_TEMPLATE * agent_radius
    hidden_verts = np.zeros_like(scaled_template)

    agent_polys = PolyCollection(
        [scaled_template.copy() for _ in range(n_ids)],
        facecolors=face_colors,
        edgecolors=edge_colors,
        linewidths=line_widths,
        zorder=4,
        animated=True,
    )
    ax.add_collection(agent_polys)

    robot_face_rgba = np.array([0.839, 0.153, 0.157, 1.0])  # #d62728
    robot_edge_rgba = np.array([0.0, 0.0, 0.0, 1.0])

    qx = np.zeros(n_ids)
    qy = np.zeros(n_ids)
    qu = np.zeros(n_ids)
    qv = np.zeros(n_ids)
    q_colors = np.tile(rgba_hidden, (n_ids, 1))
    quiver = ax.quiver(qx, qy, qu, qv, color=q_colors, scale_units="xy", scale=1.0, angles="xy", width=0.004, zorder=5)
    quiver.set_animated(True)

    title = ax.set_title("")
    title.set_animated(True)

    progress_every = max(1, len(frames) // 20)

    def _rgba(color_hex: str) -> tuple[float, float, float, float]:
        from matplotlib.colors import to_rgba

        return to_rgba(color_hex)

    def update(frame_idx: int):
        if frame_idx % progress_every == 0 or frame_idx == len(frames) - 1:
            _log.info(f"  frame {frame_idx + 1}/{len(frames)}")
        frame = frames[frame_idx]

        face_colors[:] = rgba_hidden
        edge_colors[:] = [0.0, 0.0, 0.0, 0.0]
        q_colors[:] = rgba_hidden
        qu[:] = 0.0
        qv[:] = 0.0
        verts_list = [hidden_verts] * n_ids

        for aid, x, y, theta, vx, vy, policy, kind in frame.agents:
            i = id_to_idx[aid]
            if kind == 1:
                face_colors[i] = robot_face_rgba
                edge_colors[i] = robot_edge_rgba
                line_widths[i] = 2.0
                q_colors[i] = robot_face_rgba
            else:
                col = _rgba(_policy_color(policy, policy_cache))
                face_colors[i] = col
                edge_colors[i] = [0.0, 0.0, 0.0, 1.0]
                line_widths[i] = 0.5
                q_colors[i] = col

            speed_sq = vx * vx + vy * vy
            heading = np.arctan2(vy, vx) if speed_sq > 1e-6 else theta
            c, s = np.cos(heading), np.sin(heading)
            rot = np.array([[c, -s], [s, c]])
            verts_list[i] = scaled_template @ rot.T + np.array([x, y])

            qx[i] = x
            qy[i] = y
            qu[i] = vx * 0.3
            qv[i] = vy * 0.3

        agent_polys.set_verts(verts_list)
        agent_polys.set_facecolor(face_colors.tolist())
        agent_polys.set_edgecolor(edge_colors.tolist())
        agent_polys.set_linewidth(line_widths.tolist())
        quiver.set_offsets(np.column_stack([qx, qy]))
        quiver.set_UVC(qu, qv)
        quiver.set_facecolor(q_colors)  # pyright: ignore[reportArgumentType]

        title.set_text(f"t = {frame.t_ns / 1e9:.2f}s   agents = {len(frame.agents)}")
        return [agent_polys, quiver, title]

    output.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    have_ffmpeg = bool(shutil.which("ffmpeg"))
    if fmt == "mp4" and not have_ffmpeg:
        _log.warning("ffmpeg not found; falling back to gif (install with `sudo apt install ffmpeg` for faster mp4)")
        fmt = "gif"
        output = output.with_suffix(".gif")

    _log.info(f"encoding {len(frames)} frames to {output}")

    dynamic = [agent_polys, quiver, title]
    if fmt == "mp4":
        _save_via_ffmpeg(fig, update, len(frames), fps, output, dynamic_artists=dynamic)
    elif fmt == "gif" and have_ffmpeg:
        _save_via_ffmpeg(fig, update, len(frames), fps, output, gif=True, dynamic_artists=dynamic)
    else:
        anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
        anim.save(str(output), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def _save_via_ffmpeg(
    fig: plt.Figure,
    update_fn: Callable[[int], object],
    n_frames: int,
    fps: int,
    output: Path,
    gif: bool = False,
    dynamic_artists: list | None = None,
) -> None:
    import subprocess

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    fig.canvas.draw()
    w, h = canvas.get_width_height()
    bg = canvas.copy_from_bbox(fig.bbox)

    if gif:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vf",
            "split[a][b];[a]palettegen[p];[b][p]paletteuse",
            str(output),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            str(output),
        ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for i in range(n_frames):
            update_fn(i)
            canvas.restore_region(bg)
            if dynamic_artists:
                for art in dynamic_artists:
                    fig.draw_artist(art)
            else:
                canvas.draw()
            canvas.blit(fig.bbox)
            proc.stdin.write(bytes(canvas.buffer_rgba()))
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited {rc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="arena_humansim scenario renderer")
    parser.add_argument("bag_dir", type=Path, help="path to rosbag2 directory (mcap)")
    parser.add_argument("--output", type=Path, required=False, help="output file (default: <bag_dir>/../scenario.<fmt>)")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--agent-radius", type=float, default=0.35, help="fallback radius for agents whose policy/state lacks it")
    parser.add_argument("--dpi", type=int, default=120, help="figure dpi (lower = faster, less detail)")
    parser.add_argument("--log-file", type=Path, default=None, help="also write log to this file")
    args = parser.parse_args(argv)

    _setup_logging(args.log_file)

    output = args.output
    if output is None:
        output = args.bag_dir.parent / f"scenario.{args.format}"

    render(args.bag_dir, output, args.format, args.fps, args.agent_radius, args.dpi)
    _log.info(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
