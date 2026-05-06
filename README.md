# myoarm-lambda-ep

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19948021.svg)](https://doi.org/10.5281/zenodo.19948021)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code accompanying the bioRxiv pre-print:

> Kobayashi, J. (2026). *Decoupling smoothness, accuracy, and kinematic
> invariance in biological reach: an ablation study of an equilibrium-point
> controller in a 34-muscle arm model.* bioRxiv. doi:
> [10.64898/2026.05.01.722167](https://doi.org/10.64898/2026.05.01.722167)

The repository contains a biologically motivated controller for the MyoSuite
`myoArmReachRandom-v0` environment (20-DoF, 34 Hill-type muscles), implementing
Feldman's λ-equilibrium-point hypothesis with a minimum-jerk virtual trajectory,
a 200 ms visuomotor correction, and γ-compatible spinal reflexes. The
manuscript (`paper/manuscript.md` / `paper/manuscript.pdf`) and reproduction
scripts (`scripts/experiment_myo_p15_*.py` and `scripts/figures/`) are the
primary artefacts; the rest of the tree exists to support them.

## Paper reproduction (Phase 1-6, MyoSuite myoArm)

The published-paper-relevant code is the MyoSuite branch of the project (Phase 1-6, 2026-04). It is **independent** of the Franka simulation tree below.

```bash
# 0. Environment (dependencies are declared in pyproject.toml)
python -m venv .venv && .venv/bin/pip install -e .
# Tested with: MyoSuite 2.12.1, MuJoCo 3.6.0, Gymnasium 1.2.3, Python 3.11, Linux 6.8

# 1. Reproduce headline results (n=50 across 6 conditions; ~6 min on a single CPU)
.venv/bin/python scripts/experiment_myo_p15_f16_n50.py
#   → results/experiment_myo_p15/f16_n50.json

# 2. Reproduce factorial ablation (n=20, 8 conditions; ~3 min)
.venv/bin/python scripts/experiment_myo_p15_f13_ablation.py
#   → results/experiment_myo_p15/f13_ablation.json

# 3. Reproduce no-cerebellum PD baseline control (n=50; ~30 s)
.venv/bin/python scripts/experiment_myo_p15_f17_pd_nocereb.py
#   → results/experiment_myo_p15/f17_pd_nocereb.json

# 4. Regenerate paper figures (Fig 1-5)
for f in scripts/figures/fig*_*.py; do .venv/bin/python "$f"; done
#   → figures/fig{1,2,3,4,5}.{pdf,png}

# 5. Build the manuscript PDF
bash paper/build.sh
#   → paper/manuscript.pdf  (uses xelatex; requires DejaVu Serif, Latin Modern Math)
```

Key paper artifacts:

- Controller source: [`src/myoarm/myo_controller.py`](src/myoarm/myo_controller.py) (λ-EP + virtual trajectory + visuomotor + reflexes + cerebellar branch)
- MyoSuite seed-reproducibility patch: [`src/myoarm/env_utils.py`](src/myoarm/env_utils.py) — `deterministic_reset`
- Trained CfC checkpoints: [`results/myo_cfc_data*/cfc_model.pt`](results/) (released with the manuscript)

### Minimal reproducer for the MyoSuite seed bug

In the MyoSuite versions tested (2.12.x with MuJoCo 3.6.x, Gymnasium
1.2.x), `env.reset(seed=N)` does not deterministically reproduce
the same target. The following snippet demonstrates the issue and
the fix:

```python
import gymnasium as gym
import myosuite  # noqa: F401
import numpy as np
from myoarm.env_utils import deterministic_reset  # the fix

env = gym.make("myoArmReachRandom-v0")

# Without the fix: same seed returns different targets across calls
env.reset(seed=0); t1 = np.array(env.unwrapped.obs_dict["reach_err"])
env.reset(seed=0); t2 = np.array(env.unwrapped.obs_dict["reach_err"])
print("native env.reset(seed=0) targets equal?", np.allclose(t1, t2))   # → False

# With the fix: identical targets
deterministic_reset(env, 0); t3 = np.array(env.unwrapped.obs_dict["reach_err"])
deterministic_reset(env, 0); t4 = np.array(env.unwrapped.obs_dict["reach_err"])
print("deterministic_reset targets equal?", np.allclose(t3, t4))         # → True
```

We encourage users to run this snippet on their own MyoSuite version
before relying on per-seed reproducibility.

## Repository layout

- [`src/myoarm/myo_controller.py`](src/myoarm/myo_controller.py) — λ-EP + virtual trajectory + visuomotor + reflexes + cerebellar branch
- [`src/myoarm/env_utils.py`](src/myoarm/env_utils.py) — `deterministic_reset` patch for MyoSuite seed reproducibility
- [`src/myoarm/{exp_utils,trajectory_planner}.py`](src/myoarm/) — kinematics metrics, statistics, minimum-jerk virtual trajectory generator
- [`src/methodB/cfc_forward_model.py`](src/methodB/cfc_forward_model.py) — CfC forward model (negative-result cerebellar branch)
- [`src/methodF/{delay_buffer,inferior_olive_analog}.py`](src/methodF/) — biological delay primitives used by the controller
- [`scripts/experiment_myo_p15_*.py`](scripts/) — ablation experiments F3–F17
- [`scripts/figures/`](scripts/figures/) — figure-generation pipeline (Fig 1–5)
- [`paper/`](paper/) — manuscript Markdown / xelatex source / built PDF / submission checklist
- [`figures/`](figures/) — final figure PDFs and PNGs (Springer 183 mm width)
- [`results/experiment_myo_p15/`](results/) — per-seed metric tables (JSON) for every condition

## Citation

If you use this code, please cite the bioRxiv pre-print and the Zenodo
software record:

```bibtex
@article{Kobayashi2026myoArmLambdaEP,
  author  = {Kobayashi, Jun},
  title   = {Decoupling smoothness, accuracy, and kinematic invariance in
             biological reach: an ablation study of an equilibrium-point
             controller in a 34-muscle arm model},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.05.01.722167},
  url     = {https://www.biorxiv.org/cgi/content/short/2026.05.01.722167v1},
  note    = {Pre-print},
}

@software{Kobayashi2026myoArmLambdaEPSoftware,
  author  = {Kobayashi, Jun},
  title   = {{myoarm-lambda-ep}: λ-EP controller for the MyoSuite
             {myoArmReachRandom-v0} environment},
  year    = {2026},
  doi     = {10.5281/zenodo.19948021},
  url     = {https://github.com/jkoba0512/myoarm-lambda-ep},
  version = {v1.0.0-bioRxiv},
}
```

The bioRxiv pre-print was posted on 2026-05-06 (DOI
[10.64898/2026.05.01.722167](https://doi.org/10.64898/2026.05.01.722167)).
The Zenodo software record corresponds to release `v1.0.0-bioRxiv`
(DOI [10.5281/zenodo.19948021](https://doi.org/10.5281/zenodo.19948021)).

## License

Released under the MIT License — see [LICENSE](LICENSE).
