# Consensus based trajectory optimization

Consensus-based optimization (CBO) is a particle-based, derivative-free
optimization method that drives a population of candidate solutions toward a
cost-weighted consensus point. In this repository, CBO
is used as sampling-based controllers for MuJoCo trajectory optimization tasks,
with reproduction scripts for our RSS 2026 paper experiments.

The source tree uses a `src/` layout:

- `src/algs/`: CBO/CMA/MPPI algorithm wrappers.
- `src/runners/`: experiment and aggregation entrypoints.
- `src/tasks/`: MuJoCo task definitions.
- `src/plots/`: plotting entrypoints.
- `config/`: JSON experiment configurations for tasks, algorithms, logging,
  and reproduction runs.
- `models/`: MuJoCo XML assets used by the trajectory optimization tasks.
- `pyproject.toml`: Python project metadata and dependency specification used
  by `uv`.
- `scripts_reproduce_rss26/`: RSS 2026 reproduction scripts.

## Quick Checklist

- [ ] Install dependencies:

```bash
uv sync
```

- [ ] Verify the Python entrypoint resolves:

```bash
scripts/run_uv_debug.sh --module runners.compare --help
```

- [ ] Run the fast RSS smoke test:

```bash
scripts_reproduce_rss26/reproduce_double_cartpole.sh --debug --seeds 0
```

- [ ] Run full RSS reproductions when hardware is available:

```bash
scripts_reproduce_rss26/reproduce_double_cartpole.sh
scripts_reproduce_rss26/reproduce_humanoid.sh
```

- [ ] Find aggregated PDFs under `tmp/agg_*`.

## Detailed Reproduction Docs

RSS 2026 CBO experiments:

```text
scripts_reproduce_rss26/README.md
```

## Notes

Most full experiments are GPU-memory intensive. Start with the double-cartpole
debug smoke test to verify imports, aggregation, and plotting before launching
long runs.

## Citation

```bibtex
@misc{sun2026consensusbasedoptimizationcboglobal,
      title={Consensus-based optimization (CBO): Towards Global Optimality in Robotics},
      author={Xudong Sun and Armand Jordana and Massimo Fornasier and Jalal Etesami and Majid Khadiv},
      year={2026},
      eprint={2602.06868},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.06868},
}
```
