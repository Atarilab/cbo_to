from __future__ import annotations

import argparse
import numpy as np
from huggingface_hub import hf_hub_download
import utils.mujoco_env as mujoco_env  # sets MUJOCO_GL early for headless runs
import mujoco
import imageio.v2 as imageio


def load_reference_qpos(filename: str) -> np.ndarray:
    npz = np.load(
        hf_hub_download(
            repo_id="robfiras/loco-mujoco-datasets",
            filename=filename,
            repo_type="dataset",
        )
    )
    return npz["qpos"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect reference base orientation.")
    parser.add_argument(
        "--filename",
        default="DefaultDatasets/mocap/UnitreeG1/balance.npz",
        help="Reference dataset file path within the repo.",
    )
    parser.add_argument(
        "--model-xml",
        default="models/g1/scene_23dof.xml",
        help="MuJoCo XML model path for rendering.",
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index to inspect.")
    parser.add_argument("--out", type=str, default=None, help="Output image path (PNG).")
    parser.add_argument("--width", type=int, default=800, help="Render width.")
    parser.add_argument("--height", type=int, default=600, help="Render height.")
    args = parser.parse_args()

    qpos = load_reference_qpos(args.filename)
    frame = max(min(args.frame, qpos.shape[0] - 1), 0)
    base_quat = qpos[frame][3:7]
    print(f"frame {frame} base quat: {base_quat}")

    model = mujoco.MjModel.from_xml_path(args.model_xml)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[frame]
    mujoco.mj_forward(model, data)
    renderer = None
    try:
        renderer = mujoco.Renderer(model, width=args.width, height=args.height)
        renderer.update_scene(data)
        image = renderer.render()
        out_path = args.out or f"reference_frame_{frame:05d}.png"
        imageio.imwrite(out_path, image)
        print(f"wrote {out_path}")
    except Exception as exc:
        print(
            "Warning: skipping render (MuJoCo OpenGL context not available). "
            f"Set MUJOCO_GL=egl or MUJOCO_GL=osmesa. Underlying error: {exc}"
        )
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
