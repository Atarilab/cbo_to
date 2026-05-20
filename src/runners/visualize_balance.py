"""Quick viewer to inspect the balance mocap reference."""

import time

import utils.mujoco_env as mujoco_env  # sets MUJOCO_GL early for headless runs
import mujoco

# The viewer module is a separate extension in newer MuJoCo releases.
try:
    from mujoco import viewer
except ImportError as err:  # pragma: no cover - convenience for runtime failure
    raise SystemExit(
        "MuJoCo viewer module is not available. Install mujoco with viewer support."
    ) from err

from tasks.humanoid import Humanoid


def main() -> None:
    # Load the task with the balance reference and starting at frame 0.
    task = Humanoid(
        reference_filename="DefaultDatasets/mocap/UnitreeG1/balance.npz",
        start_frame=0,
    )

    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Start at the configured initial pose.
    mj_data.qpos[:] = task.start_config
    mujoco.mj_forward(mj_model, mj_data)

    # Play through the reference frames in a loop.
    dt = 1.0 / task.reference_fps
    frame = 0
    n_frames = int(task.n_frame)

    with viewer.launch_passive(mj_model, mj_data) as v:
        last = time.perf_counter()
        while v.is_running():
            now = time.perf_counter()
            if now - last >= dt:
                frame = (frame + 1) % n_frames
                mj_data.qpos[:] = task.ref_config[frame]
                mujoco.mj_forward(mj_model, mj_data)
                last = now

            v.sync()


if __name__ == "__main__":
    main()
