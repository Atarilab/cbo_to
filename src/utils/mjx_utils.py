import mujoco  # MuJoCo core bindings and MJX access.
from hydrax.task_base import Task  # Task base type for shared helpers.


def apply_mjx_settings(task: Task, mjx_integration_time: float) -> None:  # Apply MJX integration config.
    task.dt = mjx_integration_time  # Set task-level integration step.
    task.mj_model.opt.timestep = task.dt  # Update MuJoCo model timestep.
    task.mj_model.opt.iterations = 1  # Set solver iterations for consistency.
    task.mj_model.opt.ls_iterations = 6  # Set line-search iterations for stability.
    task.mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]  # Mirror solver soft-constraint settings.
    task.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART  # Disable warmstart for determinism.
    task.model = mujoco.mjx.put_model(task.mj_model)  # Rebuild MJX model from MuJoCo model.
    task.model = task.model.replace(  # Replace MJX model options with desired settings.
        opt=task.model.opt.replace(  # Update the MJX opt dataclass.
            timestep=mjx_integration_time,  # Use the same integration timestep.
            iterations=1,  # Match solver iterations.
            ls_iterations=6,  # Match line-search iterations.
        )  # End opt.replace call.
    )  # End model.replace call.
