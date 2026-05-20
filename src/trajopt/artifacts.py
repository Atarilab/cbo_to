from __future__ import annotations

import json
import os
import shutil
import subprocess
import traceback
import tracemalloc
from pathlib import Path

import numpy as np

from trajopt.metadata import infer_algorithm_slug, infer_task_slug, write_run_metadata


class RunArtifacts:
    def __init__(self, *, controller_name: str, controller, config_path: Path | None) -> None:
        self.controller_name = controller_name
        self.controller = controller
        self.config_path = config_path
        self.run_dir: Path | None = None
        self.stats_log_path = Path("tmp/stats/consensus_stats.csv")
        self.loss_log_path = Path("tmp/stats/loss_curve.csv")
        self.cost_log_path = Path("tmp/stats/costs.csv")
        self._cost_log_filename: str | None = None

    def set_run_dir(self, run_dir: Path | None) -> None:
        self.run_dir = run_dir
        self._cost_log_filename = None
        if run_dir is None:
            return
        self.stats_log_path = run_dir / "stats" / "consensus_stats.csv"
        self.loss_log_path = run_dir / "stats" / "loss_curve.csv"
        self.cost_log_path = self.get_cost_log_path() or self.cost_log_path

    def get_last_output_path(self) -> Path | None:
        if self.run_dir is None:
            return None
        return self.run_dir / "stats" / "last_output.txt"

    def get_memory_log_path(self) -> Path | None:
        if self.run_dir is None:
            return None
        return self.run_dir / "stats" / "memory_usage.csv"

    def get_cost_log_path(self) -> Path | None:
        if self.run_dir is None:
            return None
        if self._cost_log_filename is None:
            num_samples, temperature = self._resolve_cost_log_params()
            algo_name = self._format_log_filename_value(self.controller_name)
            self._cost_log_filename = f"{algo_name}_{num_samples}_{temperature}.csv"
        return self.run_dir / "stats" / self._cost_log_filename

    def get_tracemalloc_log_path(self) -> Path | None:
        if self.run_dir is None:
            return None
        return self.run_dir / "stats" / "tracemalloc_top.txt"

    def log_tracemalloc(self, iteration: int) -> None:
        log_path = self.get_tracemalloc_log_path()
        if log_path is None:
            return
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")[:10]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"iteration {iteration}\n")
            for stat in top_stats:
                f.write(f"{stat}\n")
            f.write("\n")

    def log_memory_usage(self, iteration: int) -> None:
        log_path = self.get_memory_log_path()
        if log_path is None:
            return
        proc_kb = self._read_proc_status_kb()
        rss_mb = proc_kb.get("VmRSS")
        hwm_mb = proc_kb.get("VmHWM")
        if rss_mb is not None:
            rss_mb = int(rss_mb / 1024)
        if hwm_mb is not None:
            hwm_mb = int(hwm_mb / 1024)
        gpu_used_mb, gpu_total_mb = self._query_gpu_memory()

        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        with log_path.open("a", encoding="utf-8") as f:
            if write_header:
                f.write("iteration,cpu_rss_mb,cpu_hwm_mb,gpu_used_mb,gpu_total_mb\n")
            f.write(
                f"{iteration},{rss_mb},{hwm_mb},{gpu_used_mb},{gpu_total_mb}\n"
            )

    def dump_run_metadata(self) -> None:
        if self.run_dir is None:
            return
        if self.config_path is not None and self.config_path.exists():
            dest = self.run_dir / self.config_path.name
            if not dest.exists():
                shutil.copy2(self.config_path, dest)
        commit_path = self.run_dir / "commit.txt"
        if not commit_path.exists():
            commit_hash = self._read_git_commit()
            branch = self._read_git_branch()
            if commit_hash is not None:
                lines = [commit_hash]
                if branch:
                    lines.append(f"branch: {branch}")
                commit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        task_slug = infer_task_slug(getattr(self.controller, "task", None))
        config_name = self.config_path.name if self.config_path is not None else None
        algorithm_slug = infer_algorithm_slug(self.controller, self.controller_name)
        write_run_metadata(
            self.run_dir,
            controller_name=self.controller_name,
            task_slug=task_slug,
            config_name=config_name,
            algorithm_slug=algorithm_slug,
        )

    def mark_status_started(self) -> None:
        self._write_status("started")

    def mark_status_completed(self) -> None:
        self._write_status("completed")

    def mark_status_failed(self, exc: Exception) -> None:
        if self.run_dir is None:
            return
        status_dir = self.run_dir / "stats"
        status_dir.mkdir(parents=True, exist_ok=True)
        (status_dir / "status_failed.txt").write_text(
            f"failed: {type(exc).__name__}\n", encoding="utf-8"
        )
        (status_dir / "error_traceback.txt").write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )

    def _write_status(self, status: str) -> None:
        if self.run_dir is None:
            return
        status_dir = self.run_dir / "stats"
        status_dir.mkdir(parents=True, exist_ok=True)
        (status_dir / f"status_{status}.txt").write_text(
            f"{status}\n", encoding="utf-8"
        )

    def _read_proc_status_kb(self) -> dict[str, int]:
        metrics: dict[str, int] = {}
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                metrics[parts[0].rstrip(":")] = int(parts[1])
                            except ValueError:
                                continue
        except OSError:
            return metrics
        return metrics

    def _query_gpu_memory(self) -> tuple[int | None, int | None]:
        gpu_index = int(os.getenv("GPU_MONITOR_INDEX", "0"))
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_index}",
                ],
                text=True,
            ).strip()
        except Exception:
            return None, None
        if not out:
            return None, None
        parts = out.splitlines()[0].split(",")
        if len(parts) < 2:
            return None, None
        try:
            used_mb = int(parts[0].strip())
            total_mb = int(parts[1].strip())
        except ValueError:
            return None, None
        return used_mb, total_mb

    def _resolve_cost_log_params(self) -> tuple[str, str]:
        num_samples = getattr(self.controller, "num_samples", None)
        temperature = None
        if hasattr(self.controller, "temperature"):
            temperature = getattr(self.controller, "temperature")
        elif hasattr(self.controller, "initial_temperature"):
            temperature = getattr(self.controller, "initial_temperature")

        config_data = None
        if (num_samples is None or temperature is None) and self.config_path is not None:
            try:
                with self.config_path.open("r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                config_data = None
        if num_samples is None and config_data is not None:
            num_samples = config_data.get("num_samples")
        if temperature is None and config_data is not None:
            temperature = config_data.get("temperature")

        return self._format_log_filename_value(num_samples), self._format_log_filename_value(
            temperature
        )

    def _format_log_filename_value(self, value: object) -> str:
        if value is None:
            return "unknown"
        if hasattr(value, "shape") and not isinstance(value, (str, bytes)):
            value = float(np.asarray(value))
        return str(value).replace(" ", "").replace("/", "_")

    def _read_git_commit(self) -> str | None:
        repo_root = Path(__file__).resolve().parents[2]
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None
        return out or None

    def _read_git_branch(self) -> str | None:
        repo_root = Path(__file__).resolve().parents[2]
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None
        return out or None
