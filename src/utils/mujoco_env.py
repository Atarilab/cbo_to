import os
import sys
import warnings


def configure_mujoco_gl() -> None:
    """
    Ensure MUJOCO_GL is set before importing mujoco in headless environments.

    This avoids mujoco.Renderer failing with:
    "an OpenGL platform library has not been loaded into this process".
    """
    if os.environ.get("MUJOCO_GL"):
        return
    if not sys.platform.startswith("linux"):
        return

    # Prefer GLFW on systems with a working display (needed for mujoco.viewer).
    # Fall back to EGL for headless/container runs or broken DISPLAY setups.
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        try:
            import glfw  # type: ignore

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if glfw.init():
                    glfw.terminate()
                    return
        except Exception:
            pass

    # Users can override with MUJOCO_GL=osmesa (CPU) or MUJOCO_GL=glfw (onscreen), etc.
    os.environ["MUJOCO_GL"] = "egl"


configure_mujoco_gl()
