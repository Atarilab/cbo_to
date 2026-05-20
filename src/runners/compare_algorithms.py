from __future__ import annotations

from pathlib import Path
from typing import Iterable

# from hydrax.algs import CEM, MPPI, PredictiveSampling, Evosax
from hydrax.algs import CEM, MPPI, PredictiveSampling

from evosax.algorithms.distribution_based import CMA_ES
from evosax.algorithms.distribution_based.sv.sv_cma_es import SV_CMA_ES
from evosax.algorithms.population_based.diffusion_evolution import DiffusionEvolution

from algs.CBO_depreciated import CustomCBO
from algs.mppi_lr import MPPI_lr
from algs.evosax import Evosax
from algs.cbo_x import CBOx

try:
    from pcbo.cbox_polarized import CBOxPolarized
except ImportError:
    CBOxPolarized = None


def _direct_run_error(argv: list[str]) -> int:
    config = argv[1] if len(argv) > 1 else "<config.json>"
    seeds = argv[2].rstrip(",") if len(argv) > 2 else "<seed1,seed2,...>"
    print(
        "runners.compare_algorithms defines controller factories; it does not run "
        "experiments.\n\n"
        "Use one of these instead:\n"
        f"  scripts/run_compare_seeds.sh {config} {seeds}\n"
        f"  uv run python -m runners.compare {config} --seed <seed>",
        file=sys.stderr,
    )
    return 2


def build_algorithms(
    algorithms_names: Iterable[str],
    *,
    task,
    args,
    num_samples: int,
    noise_level: float,
    temperature: float,
    horizon: float,
    num_knots: int,
    spline_type: str,
    cbo_decay: float,
    cbo_dt: float,
    flag_multiply_cost: bool,
    flag_multiply_knot_weights: bool,
    flag_anistropic: bool,
    cbo_limit_temp: float,
    cbo_temp_decay: float,
    cem_sigma_min: float,
    persist_particles_every: int | None,
    persist_particles_latest: int | None,
    per_particle_consensus_every: int,
    per_particle_consensus_kappa: float,
    polar_kernel_reg_loss_weight: float,
    polar_auto_weight: bool,
    persist_particle_gif_count: int,
    load_particles_from_disk: bool,
    reset_temperature_on_load: bool,
    fill_random_particles_on_load: bool,
    load_particles_dir: str | None,
    load_particles_algorithm: str | None,
    load_particles_iteration: int | None,
    sves_num_populations: int = 1,
    sves_std_init: float | None = None,
    sves_std_min: float | None = None,
    sves_std_max: float | None = None,
    sves_kernel_std: float | None = None,
    sves_alpha: float | None = None,
    algorithm_settings: Iterable[dict] | None = None,
) -> list:
    algorithms = []
    settings_list = list(algorithm_settings or [])
    known_algorithms = {
        "MPPI",
        "PS",
        "CEM",
        "CMA-ES",
        "SV-CMA-ES",
        "DiffusionEvolution",
        "CBO",
        "CBOx",
        "CBOxPolarized",
        "MPPI_lr",
    }
    for idx, name in enumerate(algorithms_names):
        if name not in known_algorithms:
            raise ValueError(f"Unknown algorithm '{name}'. Available: {sorted(known_algorithms)}")
        settings = settings_list[idx] if idx < len(settings_list) else {}

        def setting(*keys, default=None):
            for key in keys:
                if key in settings:
                    return settings[key]
            return default

        alg_num_samples = int(setting("num_samples", default=num_samples))
        alg_noise_level = setting("noise_level", default=noise_level)
        alg_temperature = setting("temperature", default=temperature)
        alg_persist_particles_every = setting(
            "persist_particles_every", default=persist_particles_every
        )
        alg_persist_particles_latest = setting(
            "persist_particles_latest", default=persist_particles_latest
        )
        if name == "MPPI":
            alg = MPPI(
                task,
                num_samples=alg_num_samples,
                noise_level=alg_noise_level,
                temperature=alg_temperature,
                plan_horizon=horizon,
                num_knots=num_knots,
                spline_type=spline_type,
            )
        elif name == "PS":
            alg = PredictiveSampling(
                task,
                num_samples=alg_num_samples,
                noise_level=alg_noise_level,
                plan_horizon=horizon,
                spline_type="zero",
                num_knots=num_knots,
            )
        elif name == "CEM":
            alg = CEM(
                task,
                num_samples=alg_num_samples,
                num_elites=1,
                sigma_start=alg_noise_level,
                sigma_min=setting("sigma_min", "cem_sigma_min", default=cem_sigma_min),
                plan_horizon=horizon,
                spline_type=spline_type,
                num_knots=num_knots,
            )
        elif name == "CMA-ES":
            alg = Evosax(
                task=task,
                optimizer=CMA_ES,
                num_samples=alg_num_samples,
                plan_horizon=horizon,
                spline_type=spline_type,
                num_knots=num_knots,
            )
            alg.es_params = alg.es_params.replace(
                std_init=setting("std_init", default=alg_noise_level)
            )
            if alg_persist_particles_every is not None:
                alg.persist_particles_every = alg_persist_particles_every
            if alg_persist_particles_latest is not None:
                alg.persist_particles_latest = alg_persist_particles_latest
        elif name == "SV-CMA-ES":
            alg_sves_num_populations = int(
                setting("num_populations", "sves_num_populations", default=sves_num_populations)
            )
            if alg_sves_num_populations != 1:
                raise ValueError(
                    "SV-CMA-ES currently supports sves_num_populations=1 in this "
                    "wrapper; multiple SVES populations need Evosax rollout shape support."
                )
            alg = Evosax(
                task=task,
                optimizer=SV_CMA_ES,
                num_samples=alg_num_samples,
                plan_horizon=horizon,
                spline_type=spline_type,
                num_knots=num_knots,
                num_populations=alg_sves_num_populations,
            )
            sves_param_overrides = {
                "std_init": setting(
                    "std_init",
                    "sves_std_init",
                    default=alg_noise_level if sves_std_init is None else sves_std_init,
                ),
                "std_min": setting("std_min", "sves_std_min", default=sves_std_min),
                "std_max": setting("std_max", "sves_std_max", default=sves_std_max),
                "kernel_std": setting(
                    "kernel_std", "sves_kernel_std", default=sves_kernel_std
                ),
                "alpha": setting("alpha", "sves_alpha", default=sves_alpha),
            }
            alg.es_params = alg.es_params.replace(
                **{
                    name: value
                    for name, value in sves_param_overrides.items()
                    if value is not None and hasattr(alg.es_params, name)
                }
            )
            if alg_persist_particles_every is not None:
                alg.persist_particles_every = alg_persist_particles_every
            if alg_persist_particles_latest is not None:
                alg.persist_particles_latest = alg_persist_particles_latest
        elif name == "DiffusionEvolution":
            alg = Evosax(
                task=task,
                optimizer=DiffusionEvolution,
                num_samples=alg_num_samples,
                plan_horizon=horizon,
                spline_type=spline_type,
                num_knots=num_knots,
            )
            if hasattr(alg.es_params, "std_init"):
                alg.es_params = alg.es_params.replace(std_init=alg_noise_level)
            elif hasattr(alg.es_params, "sigma_init"):
                alg.es_params = alg.es_params.replace(sigma_init=alg_noise_level)
            if alg_persist_particles_every is not None:
                alg.persist_particles_every = alg_persist_particles_every
            if alg_persist_particles_latest is not None:
                alg.persist_particles_latest = alg_persist_particles_latest
        elif name == "CBO":
            alg = CustomCBO(
                task,
                num_samples=alg_num_samples,
                noise_level=alg_noise_level,
                temperature=alg_temperature,
                plan_horizon=horizon,
                num_knots=num_knots,
                spline_type=spline_type,
            )
        elif name in {"CBOx", "CBOxPolarized"}:
            if name == "CBOxPolarized" and CBOxPolarized is None:
                raise ImportError("CBOxPolarized requires the 'pcbo' package which is not installed.")
            controller_cls = CBOxPolarized if name == "CBOxPolarized" else CBOx
            alg = controller_cls(
                task,
                num_samples=alg_num_samples,
                noise_level=alg_noise_level,
                temperature=alg_temperature,
                plan_horizon=horizon,
                num_knots=num_knots,
                spline_type=spline_type,
                lambda_=setting("decay", "cbo_decay", default=cbo_decay),
                time_step=setting("dt", "cbo_dt", default=cbo_dt),
                flag_multiply_cost=setting(
                    "multiply_cost", "flag_multiply_cost", default=flag_multiply_cost
                ),
                flag_multiply_knot_weights=setting(
                    "multiply_knot_weights",
                    "flag_multiply_knot_weights",
                    default=flag_multiply_knot_weights,
                ),
                flag_anistropic=setting(
                    "anisotropic", "flag_anistropic", default=flag_anistropic
                ),
                limit_temp=setting("limit_temp", "cbo_limit_temp", default=cbo_limit_temp),
            )
            alg.temp_decay = setting("temp_decay", "cbo_temp_decay", default=cbo_temp_decay)
            if alg_persist_particles_every is not None:
                alg.persist_particles_every = alg_persist_particles_every
            if alg_persist_particles_latest is not None:
                alg.persist_particles_latest = alg_persist_particles_latest
            alg.per_particle_consensus_every = setting(
                "per_particle_consensus_every", default=per_particle_consensus_every
            )
            alg.per_particle_consensus_kappa = setting(
                "per_particle_consensus_kappa", default=per_particle_consensus_kappa
            )
            alg.polar_kernel_reg_loss_weight = setting(
                "polar_kernel_reg_loss_weight", default=polar_kernel_reg_loss_weight
            )
            alg.polar_auto_weight = setting("polar_auto_weight", default=polar_auto_weight)
            alg.persist_particle_gif_count = setting(
                "persist_particle_gif_count", default=persist_particle_gif_count
            )
            if args.debug:
                alg.persist_every = 1
            if load_particles_from_disk:
                alg.load_particles_from_disk = True
                alg.reset_temperature_on_load = reset_temperature_on_load
                alg.fill_random_particles_on_load = fill_random_particles_on_load
                if load_particles_dir is not None:
                    alg.load_particles_dir = Path(load_particles_dir)
                if load_particles_algorithm is not None:
                    alg.load_particles_algorithm = load_particles_algorithm
                else:
                    alg.load_particles_algorithm = name
                alg.load_particles_iteration = load_particles_iteration
        elif name == "MPPI_lr":
            alg = MPPI_lr(
                task,
                num_samples=alg_num_samples,
                noise_level=alg_noise_level,
                temperature=alg_temperature,
                plan_horizon=horizon,
                num_knots=num_knots,
                spline_type=spline_type,
                learning_rate=setting("learning_rate", default=0.001),
            )
        else:
            raise RuntimeError("ALGORITHM NOT DEFINED")

        algorithms.append(alg)
    return algorithms


if __name__ == "__main__":
    import sys

    raise SystemExit(_direct_run_error(sys.argv))
