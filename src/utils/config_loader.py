from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class AlgorithmConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: str | Dict[str, Any]) -> "AlgorithmConfig":
        if isinstance(raw, str):
            return cls(name=raw)
        if not isinstance(raw, dict):
            raise TypeError(f"Algorithm entries must be strings or objects, got {type(raw)!r}")
        if "name" not in raw:
            raise ValueError(f"Algorithm entry missing required 'name': {raw!r}")
        raw_params = raw.get("params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            raise TypeError(
                f"Algorithm params for {raw['name']!r} must be an object, got {type(raw_params)!r}"
            )
        params = dict(raw_params)
        for key, value in raw.items():
            if key not in {"name", "params"}:
                params[key] = value
        return cls(name=str(raw["name"]), params=_flatten_algorithm_params(params))


_TASK_PARAM_DEFAULTS: Dict[str, Any] = {
    "base_weight_target_base_ori_cost": None,
    "base_weight_k": None,
    "base_weight_min": None,
    "base_weight_max": None,
    "base_weight_update_every": None,
    "base_weight_mode": None,
    "base_weight_use_base_pos_cost": None,
    "base_weight_base_pos_target": None,
    "base_weight_pre_activation_weight": None,
    "base_ori_weight_start": 1.0,
    "base_ori_weight_end": 1.0,
    "stand_base_pos_cost_tol": None,
    "stand_base_ori_cost_tol": None,
    "stand_min_base_height": None,
    "terminal_fall_penalty": None,
    "stand_max_base_lin_vel": None,
    "stand_max_base_ang_vel": None,
    "stand_support_margin": None,
    "stand_joint_limit_violation": None,
    "terminal_cost_weight_mode": None,
    "terminal_cost_k": None,
    "terminal_cost_update_every": None,
    "terminal_cost_min_weight": None,
    "terminal_cost_max_weight": None,
    "terminal_cost_base_pos_weight": None,
    "terminal_cost_base_ori_weight": None,
    "terminal_cost_foot_pos_weight": None,
    "terminal_cost_foot_ori_weight": None,
    "terminal_cost_support_weight": None,
    "terminal_cost_target_base_pos_cost": None,
    "terminal_cost_target_base_ori_cost": None,
    "terminal_cost_target_foot_pos_cost": None,
    "terminal_cost_target_foot_ori_cost": None,
    "terminal_cost_base_height_weight": None,
    "terminal_cost_target_base_height_cost": None,
    "terminal_cost_use_height_factor": None,
    "terminal_cost_target_support_cost": None,
    "terminal_full_config_cost_weight": None,
    "use_se3_twist": None,
}

_ALGORITHM_DEFAULTS: Dict[str, Any] = {
    "num_samples": 1000,
    "noise_level": 0.3,
    "temperature": 1e-4,
    "cbo_decay": 1.0,
    "cbo_dt": 0.001,
    "cbo_temp_decay": 1.001,
    "cem_sigma_min": 0.01,
    "flag_multiply_cost": False,
    "flag_multiply_knot_weights": False,
    "flag_anistropic": False,
    "cbo_limit_temp": 1e-26,
    "persist_particles_every": None,
    "persist_particles_latest": None,
    "persist_particle_gif_count": 1,
    "per_particle_consensus_every": 1,
    "per_particle_consensus_kappa": 1.0,
    "polar_kernel_reg_loss_weight": 0.0,
    "polar_auto_weight": False,
    "sves_num_populations": 1,
    "sves_std_init": None,
    "sves_std_min": None,
    "sves_std_max": None,
    "sves_kernel_std": None,
    "sves_alpha": None,
}

_LOGGING_DEFAULTS: Dict[str, Any] = {
    "mean_knots_cost_every": None,
    "gc_every": None,
    "jax_clear_every": None,
    "tracemalloc_every": None,
}

_LOAD_PARTICLES_DEFAULTS: Dict[str, Any] = {
    "load_particles_from_disk": False,
    "load_particles_dir": None,
    "load_particles_algorithm": None,
    "load_particles_iteration": None,
}

_TASK_CORE_KEYS = {
    "name",
    "task",
    "horizon",
    "num_knots",
    "mjx_integration_time",
    "spline_type",
}

_CBO_ALIASES = {
    "decay": "cbo_decay",
    "dt": "cbo_dt",
    "temp_decay": "cbo_temp_decay",
    "limit_temp": "cbo_limit_temp",
    "multiply_cost": "flag_multiply_cost",
    "multiply_knot_weights": "flag_multiply_knot_weights",
    "anisotropic": "flag_anistropic",
}

_CEM_ALIASES = {
    "sigma_min": "cem_sigma_min",
}

_SVES_ALIASES = {
    "num_populations": "sves_num_populations",
    "std_init": "sves_std_init",
    "std_min": "sves_std_min",
    "std_max": "sves_std_max",
    "kernel_std": "sves_kernel_std",
    "alpha": "sves_alpha",
}


@dataclass
class CompareConfig:
    """Structured compare config with legacy flat-attribute compatibility."""

    run: Dict[str, Any]
    task_config: Dict[str, Any]
    algorithm_defaults: Dict[str, Any]
    algorithm_configs: list[AlgorithmConfig]
    logging: Dict[str, Any]
    load_particles: Dict[str, Any]
    _legacy: Dict[str, Any]

    @property
    def algorithms(self) -> list[str]:
        return [algorithm.name for algorithm in self.algorithm_configs]

    def __getattr__(self, name: str) -> Any:
        if name in self._legacy:
            return self._legacy[name]
        raise AttributeError(name)

    def settings_for_algorithms(self, algorithm_names: Iterable[str]) -> list[Dict[str, Any]]:
        settings_by_name = {
            algorithm.name: {**self.algorithm_defaults, **algorithm.params}
            for algorithm in self.algorithm_configs
        }
        return [
            dict(settings_by_name.get(name, self.algorithm_defaults))
            for name in algorithm_names
        ]

    def settings_for_algorithm(self, algorithm_name: str | None) -> Dict[str, Any]:
        if algorithm_name is None:
            return dict(self.algorithm_defaults)
        return self.settings_for_algorithms([algorithm_name])[0]


def _load_json_config(path: Path) -> Dict[str, Any]:
    resolved_path = path.expanduser()
    if not resolved_path.is_absolute():
        repo_relative = Path(__file__).resolve().parents[2] / resolved_path
        if repo_relative.exists():
            resolved_path = repo_relative
    with resolved_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_known(source: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def _flatten_algorithm_group(
    target: Dict[str, Any],
    group: Dict[str, Any],
    aliases: Dict[str, str],
) -> None:
    for key, value in group.items():
        target[aliases.get(key, key)] = value


def _flatten_algorithm_defaults(source: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {**_ALGORITHM_DEFAULTS}
    for key, value in source.items():
        if key not in {"cbo", "cem", "sves"}:
            defaults[key] = value
    if isinstance(source.get("cbo"), dict):
        _flatten_algorithm_group(defaults, source["cbo"], _CBO_ALIASES)
    if isinstance(source.get("cem"), dict):
        _flatten_algorithm_group(defaults, source["cem"], _CEM_ALIASES)
    if isinstance(source.get("sves"), dict):
        _flatten_algorithm_group(defaults, source["sves"], _SVES_ALIASES)
    return defaults


def _flatten_algorithm_params(source: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key, value in source.items():
        if key not in {"cbo", "cem", "sves"}:
            params[key] = value
    if isinstance(source.get("cbo"), dict):
        _flatten_algorithm_group(params, source["cbo"], _CBO_ALIASES)
    if isinstance(source.get("cem"), dict):
        _flatten_algorithm_group(params, source["cem"], _CEM_ALIASES)
    if isinstance(source.get("sves"), dict):
        _flatten_algorithm_group(params, source["sves"], _SVES_ALIASES)
    return params


def _legacy_from_parts(
    *,
    run: Dict[str, Any],
    task_config: Dict[str, Any],
    algorithm_defaults: Dict[str, Any],
    logging: Dict[str, Any],
    load_particles: Dict[str, Any],
) -> Dict[str, Any]:
    legacy = {
        **_TASK_PARAM_DEFAULTS,
        **_ALGORITHM_DEFAULTS,
        **_LOGGING_DEFAULTS,
        **_LOAD_PARTICLES_DEFAULTS,
    }
    legacy.update(algorithm_defaults)
    legacy.update(logging)
    legacy.update(load_particles)
    legacy["max_iter"] = run["max_iter"]
    legacy["max_eval"] = run["max_eval"]
    legacy["task"] = task_config.get("name")
    legacy["horizon"] = task_config["horizon"]
    legacy["num_knots"] = task_config["num_knots"]
    legacy["mjx_integration_time"] = task_config["mjx_integration_time"]
    legacy["spline_type"] = task_config["spline_type"]
    legacy.update(task_config.get("params", {}))
    return legacy


def _from_flat_config(config_dict: Dict[str, Any]) -> CompareConfig:
    legacy = {
        **_TASK_PARAM_DEFAULTS,
        **_ALGORITHM_DEFAULTS,
        **_LOGGING_DEFAULTS,
        **_LOAD_PARTICLES_DEFAULTS,
        **config_dict,
    }
    run = {
        "max_iter": legacy["max_iter"],
        "max_eval": legacy["max_eval"],
    }
    task_params = _copy_known(legacy, _TASK_PARAM_DEFAULTS.keys())
    task_config = {
        "name": legacy.get("task"),
        "horizon": legacy["horizon"],
        "num_knots": legacy["num_knots"],
        "mjx_integration_time": legacy["mjx_integration_time"],
        "spline_type": legacy["spline_type"],
        "params": task_params,
    }
    algorithm_defaults = _copy_known(legacy, _ALGORITHM_DEFAULTS.keys())
    logging = _copy_known(legacy, _LOGGING_DEFAULTS.keys())
    load_particles = _copy_known(legacy, _LOAD_PARTICLES_DEFAULTS.keys())
    algorithm_configs = [
        AlgorithmConfig.from_raw(raw) for raw in legacy.get("algorithms", [])
    ]
    return CompareConfig(
        run=run,
        task_config=task_config,
        algorithm_defaults=algorithm_defaults,
        algorithm_configs=algorithm_configs,
        logging=logging,
        load_particles=load_particles,
        _legacy=legacy,
    )


def _from_nested_config(config_dict: Dict[str, Any]) -> CompareConfig:
    run = dict(config_dict.get("run", {}))
    task_raw = dict(config_dict.get("task", {}))
    if "max_iter" not in run:
        raise ValueError("Nested config requires run.max_iter")
    run.setdefault("max_eval", 32)

    task_params = dict(task_raw.pop("params", {}))
    for key in list(task_raw.keys()):
        if key in _TASK_PARAM_DEFAULTS:
            task_params[key] = task_raw.pop(key)

    task_config = {
        "name": task_raw.pop("name", task_raw.pop("task", None)),
        "horizon": task_raw.pop("horizon"),
        "num_knots": task_raw.pop("num_knots"),
        "mjx_integration_time": task_raw.pop("mjx_integration_time", 0.005),
        "spline_type": task_raw.pop("spline_type", "zero"),
        "params": task_params,
    }
    for key, value in task_raw.items():
        if key not in _TASK_CORE_KEYS:
            task_config["params"][key] = value

    algorithm_defaults = _flatten_algorithm_defaults(
        dict(config_dict.get("algorithm_defaults", {}))
    )
    logging = {**_LOGGING_DEFAULTS, **dict(config_dict.get("logging", {}))}
    load_particles = {
        **_LOAD_PARTICLES_DEFAULTS,
        **dict(config_dict.get("load_particles", {})),
    }
    algorithm_configs = [
        AlgorithmConfig.from_raw(raw) for raw in config_dict.get("algorithms", [])
    ]
    legacy = _legacy_from_parts(
        run=run,
        task_config=task_config,
        algorithm_defaults=algorithm_defaults,
        logging=logging,
        load_particles=load_particles,
    )
    return CompareConfig(
        run=run,
        task_config=task_config,
        algorithm_defaults=algorithm_defaults,
        algorithm_configs=algorithm_configs,
        logging=logging,
        load_particles=load_particles,
        _legacy=legacy,
    )


def load_compare_config(
    path: Path = Path("config/compare_config.json"),
) -> CompareConfig:
    """Load the compare configuration from JSON."""
    config_dict = _load_json_config(path)
    if any(key in config_dict for key in ("run", "algorithm_defaults")):
        return _from_nested_config(config_dict)
    return _from_flat_config(config_dict)
