# neuro-arm-control

Neuro-inspired robot arm control research code built around liquid neural network ideas, classical control, and biologically motivated modules.

This repository currently serves as the simulation-phase codebase for a neuro-inspired robot arm control project. The working code is centered on a MuJoCo Franka Panda torque-control simulation with a neural controller that combines:

- PD feedback
- CfC cerebellum-style compensation
- Izhikevich reflex arc
- Matsuoka CPG
- LIF proprioceptive feedback

OpenManipulator-X and SO-101 are no longer target platforms for this project. The current plan is:

- simulation experiments on Franka Panda
- future real-robot experiments on a new robot arm to be developed for this project

## Current Status

The current codebase is organized around the Franka Panda simulation and four main experiments:

- `2A`: static-hold ablation
- `2B`: disturbance rejection
- `2C`: cyclic motion with proprioceptive feedback
- `2D`: integrated evaluation with `tau_sys` visualization

An additional analysis script evaluates Cartesian error and the `tau_sys` statistics derived from the CfC compensator.

The latest project log and final result summary live in Obsidian:

- `/home/jkoba/Documents/Brain/Logs/2026-04-24-lnn-arm-control-franka-neural-controller.md`

## Repository Layout

- [`src/common/franka_env.py`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/src/common/franka_env.py:1): MuJoCo Franka Panda torque-control environment wrapper
- [`src/common/franka_neural_controller.py`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/src/common/franka_neural_controller.py:1): integrated neural controller
- [`src/methodB/cfc_compensator.py`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/src/methodB/cfc_compensator.py:1): CfC cerebellum-style compensation and `tau_sys` logging
- [`src/methodD/cpg.py`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/src/methodD/cpg.py:1): Matsuoka oscillator
- [`src/methodE/lif_proprioceptor.py`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/src/methodE/lif_proprioceptor.py:1): LIF proprioceptor
- [`src/methodE/izhikevich_reflex.py`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/src/methodE/izhikevich_reflex.py:1): reflex arc
- [`scripts/experiment_franka_2a.py`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/scripts/experiment_franka_2a.py:1): ablation and cerebellum training
- [`scripts/experiment_franka_2b.py`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/scripts/experiment_franka_2b.py:1): disturbance evaluation
- [`scripts/experiment_franka_2c.py`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/scripts/experiment_franka_2c.py:1): cyclic motion with load
- [`scripts/experiment_franka_2d.py`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/scripts/experiment_franka_2d.py:1): integrated evaluation
- [`scripts/analyze_tau_sys_cartesian.py`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/scripts/analyze_tau_sys_cartesian.py:1): Cartesian error and `tau_sys` statistics
- [`plan/research-plan-20260423.md`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/plan/research-plan-20260423.md:1): original broader research plan
- [`plan/implementation-gap-20260424.md`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/plan/implementation-gap-20260424.md:1): plan vs implementation gap analysis

## Environment

- Python `>=3.11`
- Dependencies are declared in [`pyproject.toml`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/pyproject.toml:1)
- A local virtual environment exists at `.venv/`

Run scripts with the project environment, for example:

```bash
.venv/bin/python scripts/experiment_franka_2a.py
.venv/bin/python scripts/experiment_franka_2b.py
.venv/bin/python scripts/experiment_franka_2c.py
.venv/bin/python scripts/experiment_franka_2d.py
.venv/bin/python scripts/analyze_tau_sys_cartesian.py
```

## Main Findings So Far

These are the current headline results reflected in the latest log and saved result files:

- `2A`: adding the cerebellum improves static-hold error substantially over PD alone
- `2B`: the full controller reduces peak disturbance error under severe disturbance
- `2C`: LIF proprioceptive feedback improves post-load endpoint accuracy in cyclic motion
- `2D`: the full controller improves no-disturbance MAE and some disturbance metrics over PD alone
- `tau_sys`: the latest quantitative analysis suggests a small but significant decrease after disturbance, which goes against the initial "faster adaptation" hypothesis

## Scope Clarification

Implemented now:

- Franka Panda MuJoCo torque-control environment
- Neuro-inspired integrated controller
- Reproducible experiment scripts and saved result artifacts
- `tau_sys` extraction from CfC internals

Not in current project scope:

- ILNN / invertible IK-ID learning
- OpenManipulator-X experiments
- SO-101 experiments

Planned later, but not implemented here yet:

- Real robot execution stack for the future in-house arm

See [`plan/implementation-gap-20260424.md`](/home/jkoba/SynologyDrive/00-Research/neuro-arm-control/plan/implementation-gap-20260424.md:1) for the detailed comparison.
