import os
import argparse
import numpy as np
import utils.mujoco_env as mujoco_env  # sets MUJOCO_GL early for headless runs
import mujoco
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from runners.compare_algorithms import build_algorithms

from tasks.double_cartpole import DoubleCartPoleUnconstrained
from tasks.cart_pole import CartPole
from tasks.humanoid import Humanoid
from trajopt.optimizer import TrajectoryOptimizer
from utils.params_loader import patch_init_params_from_disk
from utils.config_loader import load_compare_config
from utils.base_weight_scheduler import BaseOrientationWeightScheduler
from utils.terminal_cost_controller import (
    TerminalCostController,
    TerminalCostScheduler,
    TerminalCostWeights,
)


VISUALIZE_BEST = False

parser = argparse.ArgumentParser(
    description="Compare control algorithms with optional overrides.")
parser.add_argument(
    "--debug",
    action="store_true",
    help="Override config with small max_iter/num_samples for quick debug.",
)
parser.add_argument(
    "--base-ori-weight-start",
    type=float,
    default=None,
    help="Starting weight for base orientation cost (Humanoid task only).",
)
parser.add_argument(
    "--base-ori-weight-end",
    type=float,
    default=None,
    help="Ending weight for base orientation cost (Humanoid task only).",
)
parser.add_argument(
    "--load-particles",
    action="store_true",
    help="Initialize CBOx particles from disk instead of sampling.",
)
parser.add_argument(
    "--load-particles-dir",
    type=str,
    default=None,
    help="Path to an experiment dir or particles dir; defaults to latest under tmp/experiments.",
)
parser.add_argument(
    "--load-particles-algorithm",
    type=str,
    default=None,
    help="Algorithm subdir under compare_<timestamp>/ (e.g. CBOx).",
)
parser.add_argument(
    "--load-particles-iteration",
    type=int,
    default=None,
    help="Iteration number to load (e.g. 500); defaults to latest.",
)
parser.add_argument(
    "--fill-random-particles",
    action="store_true",
    help="When loading particles, fill missing samples with random particles.",
)
parser.add_argument(
    "--reset-temperature",
    action="store_true",
    help="When loading particles, ignore persisted temperature and use config value.",
)
parser.add_argument(
    "--se3",
    action="store_true",
    help="Use SE(3) twist for base position/orientation cost terms.",
)
parser.add_argument(
    "--ada-weight",
    action="store_true",
    help="Enable terminal cost weight scheduling.",
)
parser.add_argument(
    "--polar",
    action="store_true",
    help="Use polarized CBOx variant.",
)
parser.add_argument(
    "--polar-weight",
    type=float,
    default=None,
    help="Override polar kernel regularization loss weight from config.",
)
parser.add_argument(
    "--polar-auto-weight",
    action="store_true",
    help="Override polar kernel regularization loss weight to 1/ratio from pcbo.cal_consensus_polar_regularization.",
)
parser.add_argument(
    "--task",
    choices=["humanoid", "d-cartpole", "cartpole"],
    default=None,
    help="Task to run (humanoid, d-cartpole, cartpole); overrides config task.",
)
parser.add_argument(
    "--algo",
    type=str,
    default=None,
    help="Comma-separated algorithm list to override config (e.g. CBOx,MPPI).",
)
parser.add_argument(
    "--max-iter",
    type=int,
    default=None,
    help="Override max iterations from config.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=None,
    help="Override temperature from config.",
)
parser.add_argument(
    "--cbo-decay",
    type=float,
    default=None,
    help="Override cbo_decay from config.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for algorithm initialization (ensures deterministic init params).",
)
parser.add_argument(
    "config",
    nargs="?",
    default="config/compare_config.json",
    help="Path to compare config JSON (default: config/compare_config.json).",
)
parser.add_argument(
    "--config",
    dest="config_flag",
    type=str,
    default=None,
    help="Path to compare config JSON (overrides positional config).",
)
args = parser.parse_args()

config_path = Path(args.config_flag or args.config)
config = load_compare_config(config_path)
available_tasks = ("humanoid", "humanoid-standup", "d-cartpole", "cartpole")
task_name = args.task or config.task or "cartpole"
if task_name not in available_tasks:
    parser.error(
        f"Unknown task '{task_name}'. Available tasks: {', '.join(available_tasks)}"
    )
max_iter = config.max_iter
horizon = config.horizon
num_knots = config.num_knots
noise_level = config.noise_level
num_samples = config.num_samples
max_eval = config.max_eval
algorithms_names = list(config.algorithms)
temperature = config.temperature
if args.temperature is not None:
    temperature = args.temperature
mjx_integration_time = config.mjx_integration_time
spline_type = config.spline_type
cbo_decay = config.cbo_decay
if args.cbo_decay is not None:
    cbo_decay = args.cbo_decay
cbo_dt = config.cbo_dt
cbo_temp_decay = config.cbo_temp_decay
cem_sigma_min = config.cem_sigma_min
flag_multiply_cost = config.flag_multiply_cost
flag_multiply_knot_weights = config.flag_multiply_knot_weights
flag_anistropic = config.flag_anistropic
cbo_limit_temp = config.cbo_limit_temp
persist_particles_every = config.persist_particles_every
persist_particles_latest = config.persist_particles_latest
per_particle_consensus_every = config.per_particle_consensus_every
per_particle_consensus_kappa = config.per_particle_consensus_kappa
polar_kernel_reg_loss_weight = config.polar_kernel_reg_loss_weight
if args.polar_weight is not None:
    polar_kernel_reg_loss_weight = args.polar_weight
polar_auto_weight = config.polar_auto_weight
if args.polar_auto_weight:
    polar_auto_weight = True
sves_num_populations = config.sves_num_populations
sves_std_init = config.sves_std_init
sves_std_min = config.sves_std_min
sves_std_max = config.sves_std_max
sves_kernel_std = config.sves_kernel_std
sves_alpha = config.sves_alpha
persist_particle_gif_count = config.persist_particle_gif_count
mean_knots_cost_every = config.mean_knots_cost_every
gc_every = config.gc_every
jax_clear_every = config.jax_clear_every
tracemalloc_every = config.tracemalloc_every
load_particles_from_disk = config.load_particles_from_disk
load_particles_dir = config.load_particles_dir
load_particles_algorithm = config.load_particles_algorithm
load_particles_iteration = config.load_particles_iteration
base_ori_weight_start = config.base_ori_weight_start
base_ori_weight_end = config.base_ori_weight_end
base_weight_target_base_ori_cost = config.base_weight_target_base_ori_cost
base_weight_k = config.base_weight_k
base_weight_min = config.base_weight_min
base_weight_max = config.base_weight_max
base_weight_update_every = config.base_weight_update_every
base_weight_mode = config.base_weight_mode
base_weight_use_base_pos_cost = config.base_weight_use_base_pos_cost
base_weight_base_pos_target = config.base_weight_base_pos_target
base_weight_pre_activation_weight = config.base_weight_pre_activation_weight
stand_base_pos_cost_tol = config.stand_base_pos_cost_tol
stand_base_ori_cost_tol = config.stand_base_ori_cost_tol
stand_min_base_height = config.stand_min_base_height
terminal_fall_penalty = config.terminal_fall_penalty
stand_max_base_lin_vel = config.stand_max_base_lin_vel
stand_max_base_ang_vel = config.stand_max_base_ang_vel
stand_support_margin = config.stand_support_margin
stand_joint_limit_violation = config.stand_joint_limit_violation
terminal_full_config_cost_weight = config.terminal_full_config_cost_weight
terminal_cost_weight_mode = config.terminal_cost_weight_mode
terminal_cost_k = config.terminal_cost_k
terminal_cost_update_every = config.terminal_cost_update_every
terminal_cost_min_weight = config.terminal_cost_min_weight
terminal_cost_max_weight = config.terminal_cost_max_weight
terminal_cost_base_pos_weight = config.terminal_cost_base_pos_weight
terminal_cost_base_ori_weight = config.terminal_cost_base_ori_weight
terminal_cost_foot_pos_weight = config.terminal_cost_foot_pos_weight
terminal_cost_foot_ori_weight = config.terminal_cost_foot_ori_weight
terminal_cost_support_weight = config.terminal_cost_support_weight
terminal_cost_target_base_pos_cost = config.terminal_cost_target_base_pos_cost
terminal_cost_target_base_ori_cost = config.terminal_cost_target_base_ori_cost
terminal_cost_target_foot_pos_cost = config.terminal_cost_target_foot_pos_cost
terminal_cost_target_foot_ori_cost = config.terminal_cost_target_foot_ori_cost
terminal_cost_base_height_weight = config.terminal_cost_base_height_weight
terminal_cost_target_base_height_cost = config.terminal_cost_target_base_height_cost
terminal_cost_use_height_factor = config.terminal_cost_use_height_factor
terminal_cost_target_support_cost = config.terminal_cost_target_support_cost
use_se3_twist = config.use_se3_twist

if args.debug:
    max_iter = min(max_iter, 100)
    num_samples = 10
    persist_particles_every = 2
    persist_particles_latest = 1
    mean_knots_cost_every = 1

if mean_knots_cost_every is not None:
    os.environ["MEAN_KNOTS_COST_EVERY"] = str(mean_knots_cost_every)
if gc_every is not None:
    os.environ["GC_EVERY"] = str(gc_every)
if jax_clear_every is not None:
    os.environ["JAX_CLEAR_EVERY"] = str(jax_clear_every)
if tracemalloc_every is not None:
    os.environ["TRACEMALLOC_EVERY"] = str(tracemalloc_every)
if args.load_particles:
    load_particles_from_disk = True
reset_temperature_on_load = args.reset_temperature
fill_random_particles_on_load = args.fill_random_particles
if args.load_particles_dir is not None:
    load_particles_dir = args.load_particles_dir
if args.load_particles_algorithm is not None:
    load_particles_algorithm = args.load_particles_algorithm
if args.load_particles_iteration is not None:
    load_particles_iteration = args.load_particles_iteration
if not args.ada_weight:
    terminal_cost_update_every = 0
if args.se3:
    use_se3_twist = True
if args.algo:
    algorithms_names = [item.strip() for item in args.algo.split(",") if item.strip()]
if args.polar:
    if "CBOxPolarized" in algorithms_names:
        algorithms_names = [
            "CBOxPolarized",
            *[name for name in algorithms_names if name != "CBOxPolarized"],
        ]
    else:
        algorithms_names = ["CBOxPolarized", *algorithms_names]
if args.max_iter is not None:
    max_iter = args.max_iter
seed = args.seed
algorithm_settings = config.settings_for_algorithms(algorithms_names)
if args.debug:
    for settings in algorithm_settings:
        settings["num_samples"] = num_samples
        settings["persist_particles_every"] = persist_particles_every
        settings["persist_particles_latest"] = persist_particles_latest
if args.temperature is not None:
    for settings in algorithm_settings:
        settings["temperature"] = temperature
if args.cbo_decay is not None:
    for settings in algorithm_settings:
        settings["cbo_decay"] = cbo_decay
        settings["decay"] = cbo_decay
if args.polar_weight is not None:
    for settings in algorithm_settings:
        settings["polar_kernel_reg_loss_weight"] = polar_kernel_reg_loss_weight
if args.polar_auto_weight:
    for settings in algorithm_settings:
        settings["polar_auto_weight"] = polar_auto_weight


def _create_compare_run_dir() -> Path:
    base_dir = Path("tmp/experiments")
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"compare_{timestamp}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = base_dir / f"compare_{timestamp}_{suffix:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate

def _append_seed(run_dir: Path, seed_value: int) -> None:
    seed_path = run_dir / "seed.txt"
    with seed_path.open("a", encoding="utf-8") as f:
        f.write(f"{seed_value}\n")

if task_name == "humanoid":
    base_weight_scheduler = None
    if base_weight_target_base_ori_cost is not None:
        base_weight_scheduler = BaseOrientationWeightScheduler(
            target_base_ori_cost=base_weight_target_base_ori_cost,
            k=base_weight_k,
            min_weight=base_weight_min,
            max_weight=base_weight_max,
            update_every=base_weight_update_every,
            mode=base_weight_mode or "additive",
            use_base_pos_cost=bool(base_weight_use_base_pos_cost)
            if base_weight_use_base_pos_cost is not None
            else True,
            base_pos_target=base_weight_base_pos_target,
            pre_activation_weight=base_weight_pre_activation_weight,
        )
    terminal_cost_controller = TerminalCostController(
        weights=TerminalCostWeights(
            base_pos_weight=terminal_cost_base_pos_weight,
            base_ori_weight=terminal_cost_base_ori_weight,
            foot_pos_weight=terminal_cost_foot_pos_weight,
            foot_ori_weight=terminal_cost_foot_ori_weight,
            base_height_weight=terminal_cost_base_height_weight,
            support_weight=terminal_cost_support_weight,
        ),
        scheduler=TerminalCostScheduler(
            k=terminal_cost_k,
            update_every=terminal_cost_update_every,
            mode=terminal_cost_weight_mode,
            min_weight=terminal_cost_min_weight,
            max_weight=terminal_cost_max_weight,
            target_base_pos_cost=terminal_cost_target_base_pos_cost,
            target_base_ori_cost=terminal_cost_target_base_ori_cost,
            target_foot_pos_cost=terminal_cost_target_foot_pos_cost,
            target_foot_ori_cost=terminal_cost_target_foot_ori_cost,
            target_base_height_cost=terminal_cost_target_base_height_cost,
            use_height_factor=bool(terminal_cost_use_height_factor)
            if terminal_cost_use_height_factor is not None
            else True,
            target_support_cost=terminal_cost_target_support_cost,
        ),
    )
    humanoid_kwargs = dict(
        reference_filename="DefaultDatasets/mocap/UnitreeG1/balance.npz",
        model_xml_path="models/g1/scene_23dof.xml",
        base_ori_weight_start=base_ori_weight_start,
        base_ori_weight_end=base_ori_weight_end,
        base_weight_scheduler=base_weight_scheduler,
        terminal_cost_controller=terminal_cost_controller,
        terminal_full_config_cost_weight=terminal_full_config_cost_weight,
        use_se3_twist=bool(use_se3_twist) if use_se3_twist is not None else False,
        stand_base_pos_cost_tol=stand_base_pos_cost_tol,
        stand_base_ori_cost_tol=stand_base_ori_cost_tol,
        stand_min_base_height=stand_min_base_height,
        terminal_fall_penalty=terminal_fall_penalty,
        stand_max_base_lin_vel=stand_max_base_lin_vel,
        stand_max_base_ang_vel=stand_max_base_ang_vel,
        stand_support_margin=stand_support_margin,
        stand_joint_limit_violation=stand_joint_limit_violation,
        start_frame=0,
        terminal_base_se3_weight=1.0,
        flag_zero_running_cost=True,
    )
    if args.base_ori_weight_start is not None:
        humanoid_kwargs["base_ori_weight_start"] = args.base_ori_weight_start
    if args.base_ori_weight_end is not None:
        humanoid_kwargs["base_ori_weight_end"] = args.base_ori_weight_end
    task = Humanoid(**humanoid_kwargs)

    from utils.mjx_utils import apply_mjx_settings

    apply_mjx_settings(task, mjx_integration_time)
elif task_name == "humanoid-standup":
    from hydrax.tasks.humanoid_standup import HumanoidStandup
    task = HumanoidStandup()
elif task_name == "d-cartpole":
    task = DoubleCartPoleUnconstrained()
elif task_name == "cartpole":
    task = CartPole()
else:
    raise ValueError(f"{task_name} not implemented!")

# Define the task (cost and dynamics)
mj_model = task.mj_model          # Model used by the simulator when visualizing results
mj_data = mujoco.MjData(mj_model) # Data for both simulator and optimizer



mj_model = task.mj_model  # Model used by the simulator when visualizing results

mj_data = mujoco.MjData(mj_model) # Data for both simulator and optimizer

if task_name == "humanoid":
    mj_data.qpos[:] = task.start_config


algorithms = build_algorithms(
    algorithms_names,
    task=task,
    args=args,
    num_samples=num_samples,
    noise_level=noise_level,
    temperature=temperature,
    horizon=horizon,
    num_knots=num_knots,
    spline_type=spline_type,
    cbo_decay=cbo_decay,
    cbo_dt=cbo_dt,
    flag_multiply_cost=flag_multiply_cost,
    flag_multiply_knot_weights=flag_multiply_knot_weights,
    flag_anistropic=flag_anistropic,
    cbo_limit_temp=cbo_limit_temp,
    cbo_temp_decay=cbo_temp_decay,
    cem_sigma_min=cem_sigma_min,
    persist_particles_every=persist_particles_every,
    persist_particles_latest=persist_particles_latest,
    per_particle_consensus_every=per_particle_consensus_every,
    per_particle_consensus_kappa=per_particle_consensus_kappa,
    polar_kernel_reg_loss_weight=polar_kernel_reg_loss_weight,
    polar_auto_weight=polar_auto_weight,
    persist_particle_gif_count=persist_particle_gif_count,
    load_particles_from_disk=load_particles_from_disk,
    reset_temperature_on_load=reset_temperature_on_load,
    fill_random_particles_on_load=fill_random_particles_on_load,
    load_particles_dir=load_particles_dir,
    load_particles_algorithm=load_particles_algorithm,
    load_particles_iteration=load_particles_iteration,
    sves_num_populations=sves_num_populations,
    sves_std_init=sves_std_init,
    sves_std_min=sves_std_min,
    sves_std_max=sves_std_max,
    sves_kernel_std=sves_kernel_std,
    sves_alpha=sves_alpha,
    algorithm_settings=algorithm_settings,
)



cost_list = []
last_cost = []
last_controls_list = []
compare_run_dir = _create_compare_run_dir()
_append_seed(compare_run_dir, seed)
for i in range(len(algorithms)):
    print("\n starting" + algorithms_names[i])
    to = TrajectoryOptimizer(
        algorithms_names[i],
        algorithms[i],
        mj_model,
        mj_data,
        max_eval=max_eval,
        config_path=config_path,
    )
    alg_run_dir = compare_run_dir / algorithms_names[i]
    alg_run_dir.mkdir(parents=True, exist_ok=False)
    to.run_dir = alg_run_dir
    if args.load_particles:
        if task_name == "humanoid":
            task_slug = "humanoid"
        elif task_name == "humanoid-standup":
            task_slug = "humanoid_standup"
        elif task_name == "d-cartpole":
            task_slug = "double_cartpole"
        else:
            task_slug = "cartpole"
        patch_init_params_from_disk(
            algorithms[i],
            base_dir=load_particles_dir,
            algorithm_name=load_particles_algorithm or algorithms_names[i],
            task_slug=task_slug,
            iteration=load_particles_iteration,
        )
    if task_name == "humanoid":
        initial_knots = np.array([task.start_config[7:] for _ in range(num_knots)])
    else:
        initial_knots = None
    cost_per_iter, controls = to.optimize(
        max_iter,
        seed=seed,
        initial_knots=initial_knots,
    )
    np.save(compare_run_dir / f"{algorithms_names[i]}_cost_per_iter.npy", cost_per_iter)
    cost_list.append(cost_per_iter)
    last_cost.append(cost_per_iter[-1])
    print(algorithms_names[i] + ": cost =  " + str(cost_per_iter[-1]) + "\n")
    last_controls_list.append(controls[-1])
    gif_path = compare_run_dir / f"{task.__class__.__name__}_{algorithms_names[i]}_simulation.gif"
    to.visuals.savegif(controls[-1], extra_fname=algorithms_names[i], out_path=gif_path)


fig = plt.figure()
for i in range(len(algorithms)):
    plt.plot(np.array(cost_list[i]), label=algorithms_names[i])
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig(compare_run_dir / "compare_costs.pdf")
# to.plot_solution(cost_per_iter, controls[-1])
# to.visuals.visualize_solution(controls[-1])

best_idx = np.argmin(last_cost)
print("Best algorithm is " + algorithms_names[best_idx])


if VISUALIZE_BEST:
    to.visuals.visualize_solution(last_controls_list[best_idx])
