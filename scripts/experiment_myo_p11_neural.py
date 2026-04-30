"""
experiment_myo_p11_neural.py — Phase 1-1: MyoArmController（神経制御）評価。

Phase 1-0 の endpoint PD ベースラインに神経制御コンポーネントを加えた
MyoArmController を評価し、ベースラインとの性能比較を行う。

主な追加コンポーネント:
  - CfC 前向きモデル（小脳、事前学習済み）
  - InferiorOliveAnalog（IO 散発発火 ~1 Hz）
  - MyoIaIbReflexArc（Ia/Ib 反射、筋長速度・筋力ベース）
  - ReciprocalInhibition（相反抑制、拮抗筋ペアに基づく）
  - DelayBuffer（感覚遅延 20 ms、小脳ループ遅延 30 ms）

比較:
  - Phase 1-0 baseline: EndpointPID (solve 10/10, min_err=11.6 mm)
  - Phase 1-1 neural:  MyoArmController (このスクリプト)

出力:
  results/experiment_myo_p11/neural_summary.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import myosuite  # noqa
import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myoarm.myo_controller import MyoArmController, MyoArmConfig

RESULTS_DIR = ROOT / "results" / "experiment_myo_p11"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CFC_MODEL_PATH = ROOT / "results" / "myo_cfc_data" / "cfc_model.pt"


# ---------------------------------------------------------------------------

def run_episode(
    env:     gym.Env,
    ctrl:    MyoArmController,
    max_steps: int = 600,
    seed:    int   = 0,
) -> dict:
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    m, d = uw.mj_model, uw.mj_data

    ctrl.reset()
    ctrl.initialize(m, d)  # site_id 取得 + 初期ヤコビアン計算

    tip_errors:  list[float] = []
    solved_steps: list[int]  = []

    for step in range(max_steps):
        od = uw.obs_dict
        q          = np.array(od["qpos"])
        dq         = np.array(od["qvel"])
        reach_err  = np.array(od["reach_err"])
        tip_pos    = np.array(od["tip_pos"])
        muscle_vel   = d.actuator_velocity.copy()
        muscle_force = d.actuator_force.copy()

        tip_errors.append(float(np.linalg.norm(reach_err)))

        a_total, _ = ctrl.step(
            q=q, dq=dq,
            reach_err=reach_err, tip_pos=tip_pos,
            muscle_vel=muscle_vel, muscle_force=muscle_force,
            m=m, d=d,
        )

        obs, reward, terminated, truncated, info = env.step(a_total)

        # 小脳更新 (env.step() 後)
        od2    = uw.obs_dict
        q_next = np.array(od2["qpos"])
        ctrl.update_cerebellum(q_next, m, d)

        if info.get("solved", False):
            solved_steps.append(step)

        if terminated or truncated:
            break

    tip_final = tip_errors[-1] if tip_errors else float("nan")
    return {
        "seed":               seed,
        "n_steps":            step + 1,
        "tip_err_initial_mm": tip_errors[0] * 1000 if tip_errors else float("nan"),
        "tip_err_final_mm":   tip_final * 1000,
        "tip_err_mean_mm":    float(np.mean(tip_errors)) * 1000,
        "tip_err_min_mm":     float(np.min(tip_errors)) * 1000,
        "solved":             len(solved_steps) > 0,
        "first_solved_step":  int(solved_steps[0]) if solved_steps else None,
    }


# ---------------------------------------------------------------------------

def main() -> None:
    n_seeds   = 10
    max_steps = 600

    print("=== Phase 1-1: MyoArmController（神経制御コンポーネント統合）===")
    print(f"env: myoArmReachFixed-v0  seeds: {n_seeds}  max_steps: {max_steps}")

    env = gym.make("myoArmReachFixed-v0")
    uw  = env.unwrapped
    env.reset(seed=0)
    m = uw.mj_model
    muscle_names = [m.actuator(i).name for i in range(m.nu)]
    print(f"n_muscles: {m.nu}  n_joints: {m.nv}")
    print(f"CfC model: {CFC_MODEL_PATH}")

    cfg = MyoArmConfig(
        Kp_ee=80.0, Kd_ee=15.0, Ki_ee=2.0, act_bias=0.15,
        K_cereb=0.2,
        io_mode="sparse", io_firing_rate_hz=1.0,
    )
    ctrl = MyoArmController(config=cfg, muscle_names=muscle_names)
    if CFC_MODEL_PATH.exists():
        ctrl.load_cfc(CFC_MODEL_PATH)
        print("  CfC 前向きモデル: 学習済みをロード")
    else:
        print("  CfC 前向きモデル: 未学習（ランダム初期値）")

    results = []
    t0 = time.time()
    for seed in range(n_seeds):
        r = run_episode(env, ctrl, max_steps=max_steps, seed=seed)
        results.append(r)
        status = "✓ solved" if r["solved"] else "  —"
        print(
            f"  seed {seed:2d}: "
            f"init={r['tip_err_initial_mm']:6.1f} mm  "
            f"final={r['tip_err_final_mm']:6.1f} mm  "
            f"min={r['tip_err_min_mm']:6.1f} mm  "
            f"{status}"
        )
    env.close()
    elapsed = time.time() - t0

    final_errs = [r["tip_err_final_mm"] for r in results]
    min_errs   = [r["tip_err_min_mm"]   for r in results]
    n_solved   = sum(r["solved"] for r in results)

    summary = {
        "phase": "1-1",
        "env": "myoArmReachFixed-v0",
        "controller": "MyoArmController (CfC+IO+Ia/Ib+RI+DelayBuffer)",
        "cfc_pretrained": CFC_MODEL_PATH.exists(),
        "n_seeds": n_seeds,
        "max_steps": max_steps,
        "n_muscles": int(m.nu),
        "n_joints": int(m.nv),
        "aggregate": {
            "tip_err_final_mm_mean": float(np.mean(final_errs)),
            "tip_err_final_mm_std":  float(np.std(final_errs)),
            "tip_err_min_mm_mean":   float(np.mean(min_errs)),
            "n_solved":   n_solved,
            "solve_rate": n_solved / n_seeds,
        },
        "elapsed_s": round(elapsed, 1),
        "baseline_comparison": {
            "p10_tip_err_final_mm": 109.2,
            "p10_tip_err_min_mm":    11.6,
            "p10_solve_rate":        1.0,
        },
        "per_seed": results,
    }

    out_path = RESULTS_DIR / "neural_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== Summary ===")
    print(f"  final err : {summary['aggregate']['tip_err_final_mm_mean']:.1f} ± "
          f"{summary['aggregate']['tip_err_final_mm_std']:.1f} mm  "
          f"(baseline: 109.2 mm)")
    print(f"  min err   : {summary['aggregate']['tip_err_min_mm_mean']:.1f} mm  "
          f"(baseline: 11.6 mm)")
    print(f"  solve rate: {n_solved}/{n_seeds}  (baseline: 10/10)")
    print(f"  elapsed   : {elapsed:.1f} s")
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
