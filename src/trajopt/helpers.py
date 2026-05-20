from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco


def compute_gif_playback_settings(
    base_playback_dt: float,
    *,
    playback_speed: float = 8.0,
    frame_stride: int = 1,
) -> tuple[float, int, str, float]:
    """Shared GIF playback settings to keep replay and persistence consistent."""
    base_playback_dt = float(base_playback_dt)
    frame_stride = max(1, int(frame_stride))
    effective_playback_dt = base_playback_dt
    if playback_speed > 0:
        effective_playback_dt = base_playback_dt / playback_speed
        min_gif_dt = 0.01
        if effective_playback_dt < min_gif_dt:
            auto_stride = int(np.ceil(min_gif_dt / effective_playback_dt))
            frame_stride = max(frame_stride, auto_stride)
            effective_playback_dt = frame_stride * base_playback_dt / playback_speed
    fps = 1.0 / effective_playback_dt if effective_playback_dt > 0 else 0.0
    playback_suffix = f"_stride_{frame_stride}_fps_{fps:.2f}_dur_{effective_playback_dt:.4f}"
    return effective_playback_dt, frame_stride, playback_suffix, fps


def build_stride_frame_indices(num_steps: int, frame_stride: int) -> list[int]:
    """Return frame indices for the stride while ensuring the final index is included."""
    num_steps = int(num_steps)
    if num_steps <= 0:
        return []
    stride = max(1, int(frame_stride))
    indices = list(range(0, num_steps, stride))
    if indices[-1] != num_steps - 1:
        indices.append(num_steps - 1)
    return indices


def _capture_mj_data_state(mj_data: mujoco.MjData) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "userdata"):
        if not hasattr(mj_data, name):
            continue
        value = getattr(mj_data, name)
        try:
            arr = np.array(value)
        except Exception:
            continue
        if arr.size == 0:
            continue
        state[name] = arr
    state["time"] = np.array(float(mj_data.time))
    return state


def save_mj_data_state(path: str | Path, mj_data: mujoco.MjData) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **_capture_mj_data_state(mj_data))


def load_mj_data_state(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    print(f"loading mjdata from {path}")
    data = np.load(path)
    return {key: data[key] for key in data.files}


def apply_mj_data_state(
    mj_data: mujoco.MjData, state: dict[str, np.ndarray] | None
) -> None:
    if not state:
        return
    if "time" in state:
        mj_data.time = float(np.asarray(state["time"]))
    for name in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "userdata"):
        if name not in state:
            continue
        target = getattr(mj_data, name, None)
        if target is None:
            continue
        value = np.asarray(state[name])
        if target.shape != value.shape:
            continue
        target[:] = value
