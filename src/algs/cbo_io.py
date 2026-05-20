from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx

from runners.particle_replay import ParticleSource, resolve_mjdata_path
from trajopt.helpers import apply_mj_data_state, load_mj_data_state


class CBOxIO:
    def __init__(self, controller) -> None:
        self.controller = controller
        self._particles_dir = Path("tmp/particles")

    def resolve_particles_dir(self) -> Path:
        base_dir = self.controller.load_particles_dir
        if base_dir is None:
            base_dir = Path("tmp/experiments")
            if not base_dir.exists():
                raise FileNotFoundError(
                    "No experiments found: tmp/experiments does not exist"
                )
        else:
            base_dir = Path(base_dir)
            if not base_dir.exists():
                raise FileNotFoundError(f"Particles base dir not found: {base_dir}")

        def pick_latest_with_particles(experiments_dir: Path) -> Path:
            candidates = sorted([p for p in experiments_dir.iterdir() if p.is_dir()])
            if not candidates:
                raise FileNotFoundError(f"No experiments found under {experiments_dir}")
            for experiment_path in reversed(candidates):
                particles_dir = self.resolve_particles_dir_from_experiment(
                    experiment_path
                )
                if particles_dir is not None:
                    return particles_dir
            raise FileNotFoundError(
                f"No particles dirs found under {experiments_dir}"
            )

        if (base_dir / "particles_latest.npy").exists() or list(
            base_dir.glob("particles_iter_*.npy")
        ):
            return base_dir
        if (base_dir / "particles").exists():
            return base_dir / "particles"
        if base_dir.name.startswith("compare_") or base_dir.parent.name == "experiments":
            particles_dir = self.resolve_particles_dir_from_experiment(base_dir)
            if particles_dir is None:
                raise FileNotFoundError(f"No particles dirs found under {base_dir}")
            return particles_dir
        if base_dir == Path("tmp/experiments"):
            return pick_latest_with_particles(base_dir)

        raise FileNotFoundError(f"Particles dir not found under {base_dir}")

    def resolve_particles_dir_from_experiment(
        self, experiment_path: Path
    ) -> Path | None:
        if experiment_path.name.startswith("compare_"):
            if self.controller.load_particles_algorithm is not None:
                candidate = experiment_path / self.controller.load_particles_algorithm
                if (candidate / "particles").exists():
                    return candidate / "particles"
                return None
            subdirs = sorted([p for p in experiment_path.iterdir() if p.is_dir()])
            with_particles = [p for p in subdirs if (p / "particles").exists()]
            if with_particles:
                return with_particles[-1] / "particles"
            return None
        if (experiment_path / "particles").exists():
            return experiment_path / "particles"
        return None

    def load_state_mjx_data(self) -> jax.Array | None:
        try:
            particles_dir = self.resolve_particles_dir()
        except FileNotFoundError:
            return None
        source = ParticleSource(
            experiment_dir=particles_dir.parent, particles_dir=particles_dir
        )
        mjdata_path = resolve_mjdata_path(
            source,
            which="iter"
            if self.controller.loaded_particles_iteration is not None
            else "latest",
            iteration=self.controller.loaded_particles_iteration,
        )
        if mjdata_path is None or not mjdata_path.exists():
            return None
        mj_model = getattr(self.controller.task, "mj_model", None)
        if mj_model is None:
            return None
        mj_data = mujoco.MjData(mj_model)
        state = load_mj_data_state(mjdata_path)
        apply_mj_data_state(mj_data, state)
        mjx_data = mjx.put_data(mj_model, mj_data)
        mjx_data = mjx_data.replace(
            mocap_pos=mj_data.mocap_pos,
            mocap_quat=mj_data.mocap_quat,
            time=mj_data.time,
            qpos=mj_data.qpos,
            qvel=mj_data.qvel,
        )
        return mjx_data

    def resolve_particles_file(self) -> Path:
        particles_dir = self.resolve_particles_dir()
        if self.controller.load_particles_iteration is not None:
            return (
                particles_dir
                / f"particles_iter_{self.controller.load_particles_iteration:05d}.npy"
            )

        latest_path = particles_dir / "particles_latest.npy"
        if latest_path.exists():
            return latest_path

        candidates = sorted(particles_dir.glob("particles_iter_*.npy"))
        if candidates:
            return candidates[-1]
        raise FileNotFoundError(f"No particles files found under {particles_dir}")

    def resolve_loaded_iteration(
        self, file_path: Path, particles_dir: Path
    ) -> int | None:
        if self.controller.load_particles_iteration is not None:
            return self.controller.load_particles_iteration
        name = file_path.stem
        if name.startswith("particles_iter_"):
            suffix = name.replace("particles_iter_", "")
            if suffix.isdigit():
                return int(suffix)
        latest_iter_path = particles_dir / "particles_latest_iter.txt"
        if latest_iter_path.exists():
            try:
                return int(latest_iter_path.read_text(encoding="utf-8").strip())
            except ValueError:
                return None
        return None

    def resolve_temperature_file(self) -> Path | None:
        particles_dir = self.resolve_particles_dir()
        if self.controller.load_particles_iteration is not None:
            candidate = (
                particles_dir
                / f"temperature_iter_{self.controller.load_particles_iteration:05d}.npy"
            )
            return candidate if candidate.exists() else None
        latest_path = particles_dir / "temperature_latest.npy"
        return latest_path if latest_path.exists() else None

    def load_particles_from_disk(
        self,
        tk: jax.Array | None = None,
        rng: jax.Array | None = None,
        mean: jax.Array | None = None,
    ) -> tuple[jax.Array | None, float | None, jax.Array | None]:
        if not getattr(self.controller, "load_particles_from_disk", False):
            return None, None, rng
        file_path = self.resolve_particles_file()
        if not file_path.exists():
            raise FileNotFoundError(f"Particles file not found: {file_path}")
        particles_np = np.load(file_path)
        if particles_np.ndim != 3:
            raise ValueError(
                f"Expected particles with shape (N,K,nu), got {particles_np.shape}"
            )
        if particles_np.shape[0] != self.controller.num_samples:
            if (
                particles_np.shape[0] < self.controller.num_samples
                and getattr(self.controller, "fill_random_particles_on_load", False)
            ):
                missing = self.controller.num_samples - particles_np.shape[0]
                if rng is None:
                    rng = jax.random.PRNGKey(0)
                rng, sample_rng = jax.random.split(rng)
                if mean is None:
                    mean = jnp.zeros(
                        (self.controller.num_knots, self.controller.task.model.nu)
                    )
                noise = jax.random.normal(
                    sample_rng,
                    (
                        missing,
                        self.controller.num_knots,
                        self.controller.task.model.nu,
                    ),
                )
                filler = mean + noise * self.controller.noise_level
                particles_np = np.concatenate(
                    [particles_np, np.asarray(filler)], axis=0
                )
            elif particles_np.shape[0] > self.controller.num_samples:
                particles_np = particles_np[: self.controller.num_samples]
            else:
                raise ValueError(
                    f"Loaded {particles_np.shape[0]} particles, expected {self.controller.num_samples}"
                )
        if (
            particles_np.shape[1] != self.controller.num_knots
            or particles_np.shape[2] != self.controller.task.model.nu
        ):
            raise ValueError(
                "Loaded particles shape "
                f"{particles_np.shape} does not match (num_samples,num_knots,nu)"
            )
        print(f"Loaded particles from {file_path}")
        self.controller.loaded_particles_iteration = self.resolve_loaded_iteration(
            file_path, self.resolve_particles_dir()
        )
        particles = jnp.asarray(particles_np)
        loaded_temperature = None
        try:
            temp_path = self.resolve_temperature_file()
        except FileNotFoundError:
            temp_path = None
        if temp_path is not None and temp_path.exists():
            loaded_temperature = float(np.asarray(np.load(temp_path)))
        state_mjx_data = self.load_state_mjx_data()
        if state_mjx_data is None:
            state_mjx_data = getattr(self.controller, "_init_state_mjx_data", None)
        self.controller.stats.log_loaded_particles_stats(
            particles, state_mjx_data, tk
        )
        return particles, loaded_temperature, rng

    def persist_particles(
        self,
        particles: jax.Array,
        costs: np.ndarray,
        iteration: int,
        temperature: float | jax.Array | None = None,
    ) -> tuple[bool, bool]:
        """Persist particles to disk (call from non-jitted Python code)."""
        persist_every = getattr(self.controller, "persist_particles_every", 500)
        persist_latest = getattr(self.controller, "persist_particles_latest", None)
        if isinstance(persist_latest, bool):
            persist_latest = 1 if persist_latest else None
        if persist_latest is not None and persist_latest <= 0:
            persist_latest = None

        should_persist_latest = (
            persist_latest is not None and iteration % int(persist_latest) == 0
        )
        should_persist_every = (
            persist_every is not None and iteration % persist_every == 0
        )
        if not (should_persist_latest or should_persist_every):
            return False, False

        self._particles_dir.mkdir(parents=True, exist_ok=True)
        particles_np = np.asarray(jax.device_get(particles))
        costs_np = np.asarray(costs)
        temp_np = None
        if temperature is not None:
            temp_np = np.asarray(temperature).astype(float)

        if should_persist_latest:
            np.save(self._particles_dir / "particles_latest.npy", particles_np)
            np.save(self._particles_dir / "costs_latest.npy", costs_np)
            if temp_np is not None:
                np.save(self._particles_dir / "temperature_latest.npy", temp_np)
            (self._particles_dir / "particles_latest_iter.txt").write_text(
                f"{iteration}\n", encoding="utf-8"
            )

        if should_persist_every:
            file_path = (
                self._particles_dir / f"particles_iter_{iteration:05d}.npy"
            )
            np.save(file_path, particles_np)
            costs_path = self._particles_dir / f"costs_iter_{iteration:05d}.npy"
            np.save(costs_path, costs_np)
            if temp_np is not None:
                temp_path = (
                    self._particles_dir
                    / f"temperature_iter_{iteration:05d}.npy"
                )
                np.save(temp_path, temp_np)
        return should_persist_latest, should_persist_every
