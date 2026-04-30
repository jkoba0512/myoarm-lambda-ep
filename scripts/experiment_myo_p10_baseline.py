"""
Phase 1-0: MyoSuite baseline — endpoint PD control on myoArmReachFixed-v0.

目的:
  - MyoSuite 環境の動作確認
  - エンドポイント PD → 筋活性化疑似逆行列による到達運動ベースラインの確立
  - Phase 1-1（神経制御移植）の比較ベースラインを記録

制御則（3 ステージ）:
  1. エンドポイント PD: F_ee = Kp * reach_err + Kd * (-tip_vel)
  2. ヤコビアン転置法: τ_pd = J_ee^T @ F_ee
  3. 重力補償付き筋活性化:
       τ_desired = τ_pd + τ_grav
       act = clip(J_act^+ @ τ_desired + act_bias, 0, 1)
  where:
    J_ee  : IFtip の位置ヤコビアン (3 x nv)
    J_act : 筋活性化→関節トルク ヤコビアン (nv x nu)  [数値計算]
    act_bias: 重力支持のための一定背景活性化

注意: この制御器は解剖学的意味を持たない工学的ベースライン。
      Phase 1-1 では神経制御コンポーネントに置き換える。

出力:
  results/experiment_myo_p10/baseline_summary.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import myosuite  # noqa: F401 — registers envs
import gymnasium as gym
import mujoco
import numpy as np

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS_DIR = ROOT / "results" / "experiment_myo_p10"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SITE_NAME = "IFtip"


# ---------------------------------------------------------------------------
# Jacobian utilities
# ---------------------------------------------------------------------------

def compute_actuator_jacobian(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    """
    筋活性化→関節トルク ヤコビアン J_act を数値微分で計算する。

    Returns
    -------
    J : (nv, nu) — J[:, i] は筋 i を Δact 増加させたときの qfrc_actuator 変化量
    """
    mujoco.mj_forward(m, d)
    qfrc0 = d.qfrc_actuator.copy()
    act0  = d.act.copy()
    delta = 0.05

    J = np.zeros((m.nv, m.nu))
    for i in range(m.nu):
        d.act[:] = act0
        d.act[i] += delta
        mujoco.mj_forward(m, d)
        J[:, i] = (d.qfrc_actuator - qfrc0) / delta

    d.act[:] = act0
    mujoco.mj_forward(m, d)
    return J


def get_site_id(m: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)


def endpoint_jacobian(m: mujoco.MjModel, d: mujoco.MjData, site_id: int) -> np.ndarray:
    """IFtip の位置ヤコビアン (3 x nv)"""
    Jp = np.zeros((3, m.nv))
    Jr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, Jp, Jr, site_id)
    return Jp


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class EndpointPDController:
    """
    エンドポイント PD + 筋活性化疑似逆行列制御器。

    Parameters
    ----------
    Kp, Kd  : 手先 PD ゲイン
    act_bias : 背景活性化（重力支持用）
    warmup_steps : J_act 計算前の安定化ステップ数
    """

    def __init__(
        self,
        n_joints:       int,
        n_muscles:      int,
        Kp:             float = 50.0,
        Kd:             float = 5.0,
        Ki:             float = 0.0,
        act_bias:       float = 0.15,
        warmup_steps:   int   = 20,
        jacobian_period: int  = 50,
    ) -> None:
        self.n_joints        = n_joints
        self.n_muscles       = n_muscles
        self.Kp              = Kp
        self.Kd              = Kd
        self.Ki              = Ki
        self.act_bias        = act_bias
        self.warmup_steps    = warmup_steps
        self.jacobian_period = jacobian_period

        self._J_act_pinv:  np.ndarray | None = None
        self._prev_tip:    np.ndarray | None = None
        self._integral:    np.ndarray        = np.zeros(3)
        self._step_count   = 0

    def reset(self) -> None:
        self._J_act_pinv  = None
        self._prev_tip    = None
        self._integral[:] = 0.0
        self._step_count  = 0

    def _refresh_jacobian(
        self, m: mujoco.MjModel, d: mujoco.MjData
    ) -> None:
        if (
            self._J_act_pinv is None
            or self._step_count % self.jacobian_period == 0
        ):
            J_act = compute_actuator_jacobian(m, d)
            self._J_act_pinv = np.linalg.pinv(J_act)

    def act(
        self,
        reach_err: np.ndarray,
        tip_pos:   np.ndarray,
        m:         mujoco.MjModel,
        d:         mujoco.MjData,
        site_id:   int,
    ) -> np.ndarray:
        self._step_count += 1

        # ウォームアップ中は一定の背景活性化のみ出力
        if self._step_count <= self.warmup_steps:
            self._prev_tip = tip_pos.copy()
            return np.full(self.n_muscles, self.act_bias, dtype=np.float32)

        self._refresh_jacobian(m, d)

        # 手先速度の近似 (1 ステップ差分)
        if self._prev_tip is None:
            tip_vel = np.zeros(3)
        else:
            tip_vel = tip_pos - self._prev_tip
        self._prev_tip = tip_pos.copy()

        # 積分項（windup 防止: ±2.0 にクリップ）
        self._integral = np.clip(self._integral + reach_err, -2.0, 2.0)

        # エンドポイント力 (世界座標系)
        F_ee = (
            self.Kp * reach_err
            - self.Kd * tip_vel
            + self.Ki * self._integral
        )  # (3,)

        # ヤコビアン転置で関節トルクに変換
        J_ee = endpoint_jacobian(m, d, site_id)           # (3, nv)
        tau_pd   = J_ee.T @ F_ee                           # (nv,)
        tau_bias = d.qfrc_bias.copy()                      # 重力+コリオリ補償
        tau_desired = tau_pd + tau_bias                    # (nv,)

        # 疑似逆行列で筋活性化に変換
        act_desired = self._J_act_pinv @ tau_desired       # (nu,)
        act_final   = np.clip(act_desired + self.act_bias, 0.0, 1.0)
        return act_final.astype(np.float32)


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(
    env:     gym.Env,
    ctrl:    EndpointPDController,
    site_id: int,
    max_steps: int = 600,
    seed:    int   = 0,
) -> dict:
    obs, _ = env.reset(seed=seed)
    ctrl.reset()
    uw = env.unwrapped
    m, d = uw.mj_model, uw.mj_data

    tip_errors: list[float] = []
    solved_steps: list[int] = []

    for step in range(max_steps):
        od = uw.obs_dict
        reach_err = np.array(od["reach_err"])
        tip_pos   = np.array(od["tip_pos"])
        err_norm  = float(np.linalg.norm(reach_err))
        tip_errors.append(err_norm)

        action = ctrl.act(reach_err, tip_pos, m, d, site_id)
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("solved", False):
            solved_steps.append(step)

        if terminated or truncated:
            break

    tip_final = tip_errors[-1] if tip_errors else float("nan")
    return {
        "seed": seed,
        "n_steps": step + 1,
        "tip_err_initial_mm": tip_errors[0] * 1000 if tip_errors else float("nan"),
        "tip_err_final_mm":   tip_final * 1000,
        "tip_err_mean_mm":    float(np.mean(tip_errors)) * 1000,
        "tip_err_min_mm":     float(np.min(tip_errors)) * 1000,
        "solved": len(solved_steps) > 0,
        "first_solved_step": int(solved_steps[0]) if solved_steps else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_seeds   = 10
    max_steps = 600

    print("=== Phase 1-0: MyoSuite endpoint-PD + muscle pseudo-inverse baseline ===")
    print(f"env: myoArmReachFixed-v0  seeds: {n_seeds}  max_steps: {max_steps}")

    env = gym.make("myoArmReachFixed-v0")
    uw  = env.unwrapped

    # 環境を 1 step 進めてモデルにアクセスできる状態にする
    env.reset(seed=0)
    m  = uw.mj_model
    site_id = get_site_id(m, SITE_NAME)
    print(f"n_muscles: {m.nu}  n_joints: {m.nv}  tip_site: {SITE_NAME} (id={site_id})")

    ctrl = EndpointPDController(
        n_joints=m.nv, n_muscles=m.nu,
        Kp=80.0, Kd=15.0, Ki=2.0, act_bias=0.15,
        warmup_steps=20, jacobian_period=50,
    )

    results = []
    t0 = time.time()
    for seed in range(n_seeds):
        r = run_episode(env, ctrl, site_id, max_steps=max_steps, seed=seed)
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
        "phase": "1-0",
        "env": "myoArmReachFixed-v0",
        "controller": "EndpointPID + muscle pseudo-inverse (Kp=80, Kd=15, Ki=2, bias=0.15, J_period=50)",
        "n_seeds": n_seeds,
        "max_steps": max_steps,
        "n_muscles": int(m.nu),
        "n_joints":  int(m.nv),
        "aggregate": {
            "tip_err_final_mm_mean": float(np.mean(final_errs)),
            "tip_err_final_mm_std":  float(np.std(final_errs)),
            "tip_err_min_mm_mean":   float(np.mean(min_errs)),
            "n_solved":   n_solved,
            "solve_rate": n_solved / n_seeds,
        },
        "elapsed_s": round(elapsed, 1),
        "per_seed": results,
    }

    out_path = RESULTS_DIR / "baseline_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== Summary ===")
    print(f"  final err : {summary['aggregate']['tip_err_final_mm_mean']:.1f} ± "
          f"{summary['aggregate']['tip_err_final_mm_std']:.1f} mm")
    print(f"  min err   : {summary['aggregate']['tip_err_min_mm_mean']:.1f} mm  (最接近)")
    print(f"  solve rate: {n_solved}/{n_seeds}")
    print(f"  elapsed   : {elapsed:.1f} s")
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
