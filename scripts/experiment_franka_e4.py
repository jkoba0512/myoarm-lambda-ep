"""
実験 E4: 運動皮質アナログ（Motor Cortex Analog）の評価

目的:
  MotorCortexAnalog が CPG 変調・co-contraction 上位制御により
  E3（VirtualCocontraction + IaIbReflexArc）を超えるかを検証する。

Sub-experiment 1: E4-hold  (D0/D1/D5 × 4条件 × 11 seed)
  比較条件:
    E4-PD   : PD のみ（ベースライン）
    E4-E3   : CC + Ia/Ib 反射弧（E3 ベスト再現）
    E4-MCA  : MCA のみ（CC なし、反射弧なし）
    E4-full : MCA + CC + Ia/Ib 反射弧

Sub-experiment 2: E4-switch (D0 のみ × 11 seed)
  タスク切り替えシナリオ:
    0-3s: hold  (CPG off)
    3-6s: oscillate (MCA が CPG amplitude = 0.3)
    6-9s: hold  (CPG off)

  条件:
    E4-PD-switch  : MCA なし（CPG 常時 off）
    E4-MCA-switch : MCA あり（タスク切り替え）

評価指標:
  E4-hold   : mae_post_mrad（外乱後 MAE）← E3 と直接比較可能
  E4-switch : hold_mae / oscillate_mae / transition_jerk

出力:
  results/experiment_franka_e4/{sweep_name}/seed<N>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.franka_env import FrankaEnv, N_JOINTS
from common.franka_neural_controller import FrankaNeuralController

RESULTS_DIR = ROOT / "results" / "experiment_franka_e4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

Q_OFFSET   = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])
CPG_PARAMS = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.0)

# E4-hold: E3 と同じ外乱条件
SIM_DURATION_HOLD = 6.0
DIST_T            = 3.0
DIST_JOINT        = 1
DIST_TORQUE       = 60.0
DIST_STEPS        = 20

# E4-switch: タスク切り替え評価
SIM_DURATION_SWITCH  = 9.0
SWITCH_HOLD1_END     = 3.0
SWITCH_OSC_END       = 6.0
SWITCH_HOLD2_END     = 9.0
TRANSITION_WINDOW    = 0.5   # 切り替え前後のジャーク計算ウィンドウ [s]

# Phase D 条件（E3 と同じ）
D_CONDITIONS = {
    "D0": {},
    "D1": {"kp_scale": 0.5},
    "D5": {"torque_saturation": 30.0},
}

KP_DEFAULT = np.array([50., 50., 50., 50., 10., 10., 10.])
KD_DEFAULT = np.array([7.,  7.,  7.,  7.,  1.5, 1.5, 1.5])


# ──────────────────────────────────────────────────────────────────
# コントローラ生成
# ──────────────────────────────────────────────────────────────────

def make_ctrl(
    kp_scale:    float       = 1.0,
    torque_sat:  float | None = None,
    use_cc:      bool        = False,
    use_ia_ib:   bool        = False,
    use_mca:     bool        = False,
    env_seed:    int         = 9999,
) -> tuple[FrankaNeuralController, FrankaEnv]:
    kp = KP_DEFAULT * kp_scale
    env_kw: dict = {}
    if torque_sat is not None:
        env_kw["torque_saturation"] = torque_sat

    ctrl_env = FrankaEnv(
        rng=np.random.default_rng(env_seed),
        obs_noise_std=0.002,
        **env_kw,
    )

    ctrl = FrankaNeuralController(
        dt=ctrl_env.dt,
        q_range=ctrl_env.ctrl_range,
        cpg_params=CPG_PARAMS,
        kp=kp, kd=KD_DEFAULT.copy(),
        use_proprioceptor=False,
        use_reflex=use_ia_ib,
        use_ia_ib_reflex=use_ia_ib,
        use_cerebellum=False,
        use_cocontraction=use_cc,
        use_motor_cortex=use_mca,
        cpg_alpha_fb=0.0,
    )
    return ctrl, ctrl_env


# ──────────────────────────────────────────────────────────────────
# E4-hold: 外乱回復実験（E3 と同一設定）
# ──────────────────────────────────────────────────────────────────

def run_hold_episode(
    ctrl: FrankaNeuralController,
    env:  FrankaEnv,
    do_disturb:    bool  = True,
    dist_torque:   float = DIST_TORQUE,
) -> dict:
    q, dq = env.reset(q0=Q_OFFSET.copy())
    ctrl.reset()
    t_log, q_log = [], []
    dist_applied = False

    while env.time < SIM_DURATION_HOLD:
        t = env.time
        q, dq = env.get_state()

        if do_disturb and not dist_applied and t >= DIST_T:
            td = np.zeros(N_JOINTS)
            td[DIST_JOINT] = dist_torque
            env.apply_disturbance(td, duration_steps=DIST_STEPS)
            dist_applied = True
            continue

        tau, info = ctrl.step(q, dq, Q_OFFSET)
        env.step(tau)
        t_log.append(t)
        q_log.append(q.copy())

    ta  = np.array(t_log)
    qa  = np.array(q_log)
    err = np.abs(qa - Q_OFFSET)

    result: dict = {"mae": float(err.mean())}
    if do_disturb:
        mask      = ta > DIST_T
        post_err  = err[mask, DIST_JOINT] if mask.any() else np.array([])
        rec_i     = np.where(post_err < 0.1)[0]
        result["peak_err"]      = float(post_err.max()) if len(post_err) else float("nan")
        result["recovery_time"] = float(ta[mask][rec_i[0]] - DIST_T) if len(rec_i) else None
        result["mae_post"]      = float(post_err.mean()) if len(post_err) else float("nan")
    return result


# ──────────────────────────────────────────────────────────────────
# E4-switch: タスク切り替え実験
# ──────────────────────────────────────────────────────────────────

def _compute_jerk(q_log: np.ndarray, t_log: np.ndarray, t_center: float, dt: float) -> float:
    """切り替え時点 t_center 前後 TRANSITION_WINDOW 秒の jerk ∫(d³q/dt³)² dt を返す。"""
    mask = (np.array(t_log) >= t_center - TRANSITION_WINDOW) & \
           (np.array(t_log) <= t_center + TRANSITION_WINDOW)
    q_seg = q_log[mask]   # (T, n)
    if len(q_seg) < 4:
        return float("nan")
    # 3 階差分で jerk を近似: d³q ≈ diff³ / dt³
    d3 = np.diff(q_seg, n=3, axis=0)
    jerk = d3 / (dt ** 3)
    return float(np.sum(jerk ** 2) * dt)


def run_switch_episode(
    ctrl:     FrankaNeuralController,
    env:      FrankaEnv,
    use_mca:  bool = False,
) -> dict:
    """
    タスク切り替えシナリオ:
      0-3s : hold
      3-6s : oscillate
      6-9s : hold
    """
    q, dq = env.reset(q0=Q_OFFSET.copy())
    ctrl.reset()
    if use_mca:
        ctrl.set_task_mode("hold")

    t_log = []
    q_log = []
    phase_switched_osc  = False
    phase_switched_hold = False

    while env.time < SIM_DURATION_SWITCH:
        t = env.time
        q, dq = env.get_state()

        # タスク切り替え（MCA あり）
        if use_mca:
            if not phase_switched_osc and t >= SWITCH_HOLD1_END:
                ctrl.set_task_mode("oscillate")
                phase_switched_osc = True
            elif not phase_switched_hold and t >= SWITCH_OSC_END:
                ctrl.set_task_mode("hold")
                phase_switched_hold = True

        tau, info = ctrl.step(q, dq, Q_OFFSET)
        env.step(tau)
        t_log.append(t)
        q_log.append(q.copy())

    ta = np.array(t_log)
    qa = np.array(q_log)
    err = np.abs(qa - Q_OFFSET)  # (T, n)

    dt = env.dt

    # 保持フェーズ MAE: 0-3s および 6-9s
    hold_mask = (ta < SWITCH_HOLD1_END) | (ta >= SWITCH_OSC_END)
    hold_mae  = float(err[hold_mask].mean()) if hold_mask.any() else float("nan")

    # 振動フェーズ MAE: 3-6s（Q_OFFSET からの偏差平均）
    osc_mask = (ta >= SWITCH_HOLD1_END) & (ta < SWITCH_OSC_END)
    osc_mae  = float(err[osc_mask].mean()) if osc_mask.any() else float("nan")

    # 切り替え時の jerk: t=3s, t=6s
    jerk_3 = _compute_jerk(qa, t_log, SWITCH_HOLD1_END, dt)
    jerk_6 = _compute_jerk(qa, t_log, SWITCH_OSC_END,   dt)
    transition_jerk = (
        (jerk_3 if not (jerk_3 != jerk_3) else 0.0)
        + (jerk_6 if not (jerk_6 != jerk_6) else 0.0)
    )
    if jerk_3 != jerk_3 and jerk_6 != jerk_6:
        transition_jerk = float("nan")

    return {
        "hold_mae":         hold_mae,
        "oscillate_mae":    osc_mae,
        "jerk_3s":          jerk_3,
        "jerk_6s":          jerk_6,
        "transition_jerk":  transition_jerk,
        "mae_overall":      float(err.mean()),
    }


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--sweep-name", type=str, default="default")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    seed   = args.seed
    base   = RESULTS_DIR if args.sweep_name == "default" else RESULTS_DIR / args.sweep_name
    outdir = base / f"seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    print(f"seed={seed}  out={outdir}")

    # ── Sub-experiment 1: E4-hold ──────────────────────────────────
    hold_conds = [
        ("E4-PD",   dict(use_cc=False, use_ia_ib=False, use_mca=False)),
        ("E4-E3",   dict(use_cc=True,  use_ia_ib=True,  use_mca=False)),
        ("E4-MCA",  dict(use_cc=False, use_ia_ib=False, use_mca=True)),
        ("E4-full", dict(use_cc=True,  use_ia_ib=True,  use_mca=True)),
    ]

    hold_results: dict = {}

    for d_idx, (d_name, d_kw) in enumerate(D_CONDITIONS.items()):
        kp_scale   = d_kw.get("kp_scale", 1.0)
        torque_sat = d_kw.get("torque_saturation", None)

        hold_results[d_name] = {}

        d_rng = np.random.default_rng(seed * 100 + d_idx)
        actual_dist_torque = float(
            np.clip(DIST_TORQUE * (1.0 + d_rng.normal(0.0, 0.25)), 20.0, 120.0)
        )
        d_env_seed = seed * 100 + d_idx

        for cname, ckw in hold_conds:
            ctrl, env = make_ctrl(
                kp_scale=kp_scale, torque_sat=torque_sat,
                env_seed=d_env_seed, **ckw,
            )
            res = run_hold_episode(ctrl, env, do_disturb=True,
                                   dist_torque=actual_dist_torque)
            hold_results[d_name][cname] = res

            rt   = res.get("recovery_time")
            rt_s = f"{rt:.3f}s" if rt else "未回復"
            print(
                f"  [hold/{d_name}] {cname:10s}  "
                f"MAE={res['mae']*1000:.1f} mrad  "
                f"Peak={res.get('peak_err', float('nan')):.4f} rad  "
                f"RT={rt_s}"
            )

    # ── Sub-experiment 2: E4-switch ────────────────────────────────
    switch_conds = [
        ("E4-PD-switch",  dict(use_cc=False, use_ia_ib=False, use_mca=False)),
        ("E4-MCA-switch", dict(use_cc=False, use_ia_ib=False, use_mca=True)),
    ]

    switch_results: dict = {}
    sw_env_seed = seed * 100 + len(D_CONDITIONS)

    for cname, ckw in switch_conds:
        ctrl, env = make_ctrl(
            kp_scale=1.0, torque_sat=None,
            env_seed=sw_env_seed, **ckw,
        )
        use_mca = ckw.get("use_mca", False)
        res = run_switch_episode(ctrl, env, use_mca=use_mca)
        switch_results[cname] = res
        print(
            f"  [switch]    {cname:16s}  "
            f"hold_MAE={res['hold_mae']*1000:.1f} mrad  "
            f"osc_MAE={res['oscillate_mae']*1000:.1f} mrad  "
            f"jerk={res['transition_jerk']:.4f}"
        )

    # ── JSON 保存 ──────────────────────────────────────────────────
    metrics_out: dict = {
        "experiment":   "e4",
        "seed":         seed,
        "sweep_name":   args.sweep_name,
        "hold_results": {
            d: {
                c: {
                    "mae_mrad":        float(r["mae"] * 1000),
                    "peak_err_rad":    float(r.get("peak_err", float("nan"))),
                    "recovery_time_s": r.get("recovery_time"),
                    "mae_post_mrad":   float(
                        (r.get("mae_post") or float("nan")) * 1000
                        if r.get("mae_post") is not None else float("nan")
                    ),
                }
                for c, r in cond_res.items()
            }
            for d, cond_res in hold_results.items()
        },
        "switch_results": {
            c: {
                "hold_mae_mrad":      float(r["hold_mae"] * 1000),
                "oscillate_mae_mrad": float(r["oscillate_mae"] * 1000),
                "transition_jerk":    float(r["transition_jerk"]),
                "jerk_3s":            float(r["jerk_3s"]),
                "jerk_6s":            float(r["jerk_6s"]),
            }
            for c, r in switch_results.items()
        },
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    print(f"\nJSON 保存: {outdir / 'metrics.json'}")

    # ── プロット ────────────────────────────────────────────────────
    hold_cond_names = ["E4-PD", "E4-E3", "E4-MCA", "E4-full"]
    colors          = ["tab:gray", "tab:blue", "tab:orange", "tab:purple"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Experiment E4: Motor Cortex Analog  [seed={seed}]", fontsize=12)

    for col, d_name in enumerate(D_CONDITIONS):
        for row, metric in enumerate(["mae", "peak_err"]):
            ax    = axes[row][col]
            vals  = [hold_results[d_name].get(c, {}).get(metric, float("nan"))
                     for c in hold_cond_names]
            scale = 1000.0 if metric == "mae" else 1.0
            vals  = [v * scale if v == v else float("nan") for v in vals]
            unit  = "mrad" if metric == "mae" else "rad"
            label = f"Static MAE [{unit}]" if metric == "mae" else f"Peak error [{unit}]"

            bars = ax.bar(range(len(hold_cond_names)), vals, color=colors, alpha=0.8)
            ax.set_xticks(range(len(hold_cond_names)))
            ax.set_xticklabels(hold_cond_names, fontsize=7, rotation=15, ha="right")
            ax.set_ylabel(label)
            ax.set_title(f"{d_name}: {label}")
            valid = [v for v in vals if v == v]
            vmax  = max(valid) if valid else 1.0
            for bar, v in zip(bars, vals):
                if v == v:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + vmax * 0.02,
                            f"{v:.1f}", ha="center", va="bottom", fontsize=7)
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = outdir / "plot_franka_e4.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"プロット保存: {plot_path}")

    # ── サマリ表示 ─────────────────────────────────────────────────
    print("\n=== E4 hold 結果サマリ ===")
    for d_name in D_CONDITIONS:
        print(f"\n  {d_name}:")
        for cname in hold_cond_names:
            r    = hold_results[d_name][cname]
            rt   = r.get("recovery_time")
            rt_s = f"{rt:.3f}s" if rt else "未回復"
            print(
                f"    {cname:10s}  MAE={r['mae']*1000:7.1f} mrad  "
                f"Peak={r.get('peak_err', float('nan')):.4f} rad  RT={rt_s}"
            )

    print("\n=== E4 switch 結果サマリ ===")
    for cname in ["E4-PD-switch", "E4-MCA-switch"]:
        r = switch_results[cname]
        print(
            f"  {cname:18s}  hold={r['hold_mae']*1000:.1f} mrad  "
            f"osc={r['oscillate_mae']*1000:.1f} mrad  "
            f"jerk={r['transition_jerk']:.4f}"
        )


if __name__ == "__main__":
    main()
