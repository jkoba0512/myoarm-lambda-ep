"""
実験 EF: 完全3層階層（Full Hierarchy）の統合評価

目的:
  E1（CfC Forward Model 小脳）+ E2（VirtualCocontraction）+
  E3（IaIbReflexArc）+ E4（MotorCortexAnalog）を全統合した
  「ヒト神経運動制御3層階層」の最終性能を検証する。

比較条件（D0/D1/D5 × 4条件 × 11 seed）:
  EF-PD     : PD のみ（ベースライン）
  EF-E4     : E4-full（MCA+CC+Ia/Ib、小脳なし）← 現時点の最良
  EF-cereb  : CfC Forward Model のみ（E1-fwd、他モジュールなし）
  EF-full   : CfC Forward Model + MCA + CC + Ia/Ib（完全階層）

CfC Forward Model は results/experiment_franka_e1/seed{N}/cfc_forward.pt
から事前学習済みモデルをロードする（再訓練不要）。

評価指標（E5 と統一）:
  - mae_post_mrad   : 外乱後 MAE [mrad]
  - energy_post_J   : 外乱後エネルギー ∫τ²dt [Nm²·s]
  - jerk_post       : 外乱後 jerk ∫(d³q/dt³)²dt [rad²/s⁵]
  - cc_ratio        : 同時収縮率
  - recovery_time_s : 外乱回復時間 [s]
  - peak_err_rad    : 外乱後ピーク誤差 [rad]
  - pred_err_mrad   : 小脳前向き予測誤差 [mrad]（EF-cereb / EF-full のみ）

成功判定:
  EF-full の mae_post が EF-E4 より低い（D0 または D1）

出力:
  results/experiment_franka_ef/{sweep_name}/seed<N>/metrics.json
  results/franka_ef_summary.json
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

try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

RESULTS_DIR = ROOT / "results" / "experiment_franka_ef"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

E1_DIR = ROOT / "results" / "experiment_franka_e1"

Q_OFFSET   = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])
CPG_PARAMS = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.0)

SIM_DURATION = 6.0
DIST_T       = 3.0
DIST_JOINT   = 1
DIST_TORQUE  = 60.0
DIST_STEPS   = 20

D_CONDITIONS = {
    "D0": {},
    "D1": {"kp_scale": 0.5},
    "D5": {"torque_saturation": 30.0},
}

KP_DEFAULT = np.array([50., 50., 50., 50., 10., 10., 10.])
KD_DEFAULT = np.array([7.,  7.,  7.,  7.,  1.5, 1.5, 1.5])


def run_episode(
    ctrl, env,
    do_disturb:   bool  = True,
    dist_torque:  float = DIST_TORQUE,
    use_cereb:    bool  = False,
) -> dict:
    q, dq = env.reset(q0=Q_OFFSET.copy())
    ctrl.reset()
    dt = env.dt

    t_log, q_log, tau_log, tau_cc_log, pred_err_log = [], [], [], [], []
    dist_applied = False

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()

        if do_disturb and not dist_applied and t >= DIST_T:
            td = np.zeros(N_JOINTS)
            td[DIST_JOINT] = dist_torque
            env.apply_disturbance(td, duration_steps=DIST_STEPS)
            dist_applied = True
            continue

        tau, info = ctrl.step(q, dq, Q_OFFSET)

        # 小脳あり条件: 観測値でモデルを更新
        if use_cereb:
            q_next, _ = env.get_state()
            ctrl.update_cerebellum(q_next)

        env.step(tau)

        t_log.append(t)
        q_log.append(q.copy())
        tau_log.append(tau.copy())
        tau_cc_log.append(
            info["tau_cc"].copy() if "tau_cc" in info and info["tau_cc"] is not None
            else np.zeros(N_JOINTS)
        )
        # 予測誤差（CfC Forward Model が有効な場合のみ）
        if "q_hat" in info and info["q_hat"] is not None:
            pred_err_log.append(float(np.mean(np.abs(info["q_hat"] - q) * 1000)))

    ta      = np.array(t_log)
    qa      = np.array(q_log)
    taua    = np.array(tau_log)
    tau_cc_a = np.array(tau_cc_log)
    err     = np.abs(qa - Q_OFFSET)

    result: dict = {"mae": float(err.mean())}

    if do_disturb:
        mask     = ta > DIST_T
        post_err = err[mask, DIST_JOINT] if mask.any() else np.array([])
        rec_i    = np.where(post_err < 0.1)[0]
        result["peak_err"]      = float(post_err.max()) if len(post_err) else float("nan")
        result["recovery_time"] = float(ta[mask][rec_i[0]] - DIST_T) if len(rec_i) else None
        result["mae_post"]      = float(post_err.mean()) if len(post_err) else float("nan")

    # エネルギー
    result["energy_J"] = float(np.sum(taua ** 2) * dt)
    if do_disturb and mask.any():
        result["energy_post_J"] = float(np.sum(taua[mask] ** 2) * dt)
    else:
        result["energy_post_J"] = float("nan")

    # Jerk
    if len(qa) >= 4:
        d3q = np.diff(qa, n=3, axis=0) / (dt ** 3)
        result["jerk"] = float(np.sum(d3q ** 2) * dt)
        if do_disturb and mask.any():
            post_idx = np.where(ta > DIST_T)[0]
            if len(post_idx) >= 4:
                d3_post = np.diff(qa[post_idx], n=3, axis=0) / (dt ** 3)
                result["jerk_post"] = float(np.sum(d3_post ** 2) * dt)
            else:
                result["jerk_post"] = float("nan")
        else:
            result["jerk_post"] = float("nan")
    else:
        result["jerk"] = result["jerk_post"] = float("nan")

    # CC 率
    mean_tau_cc  = float(np.mean(tau_cc_a))
    mean_abs_tau = float(np.mean(np.abs(taua)))
    result["cc_ratio"] = mean_tau_cc / (mean_abs_tau + mean_tau_cc + 1e-8)

    # 予測誤差
    result["pred_err_mrad"] = float(np.mean(pred_err_log)) if pred_err_log else float("nan")

    return result


def make_ctrl(
    kp_scale:    float        = 1.0,
    torque_sat:  float | None = None,
    use_cc:      bool         = False,
    use_ia_ib:   bool         = False,
    use_mca:     bool         = False,
    use_cereb:   bool         = False,
    env_seed:    int          = 9999,
    fwd_model_path: str | None = None,
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
        use_cerebellum=use_cereb,
        use_forward_model=use_cereb,
        use_cocontraction=use_cc,
        use_motor_cortex=use_mca,
        cpg_alpha_fb=0.0,
        cfc_hidden_units=64,
        device=DEVICE,
    )

    if use_cereb and fwd_model_path is not None:
        ctrl.load_cerebellum(fwd_model_path)

    return ctrl, ctrl_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--sweep-name", type=str, default="default")
    return p.parse_args()


def main():
    args   = parse_args()
    seed   = args.seed
    base   = RESULTS_DIR if args.sweep_name == "default" else RESULTS_DIR / args.sweep_name
    outdir = base / f"seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    print(f"seed={seed}  out={outdir}  device={DEVICE}")

    # E1 の事前学習済み CfC Forward Model をロード
    fwd_path = E1_DIR / f"seed{seed}" / "cfc_forward.pt"
    if not fwd_path.exists():
        print(f"WARNING: {fwd_path} が見つかりません。EF-cereb / EF-full は PD として動作します。")
        fwd_path_str = None
    else:
        fwd_path_str = str(fwd_path)
        print(f"CfC Forward Model ロード: {fwd_path}")

    conds = [
        ("EF-PD",    dict(use_cc=False, use_ia_ib=False, use_mca=False, use_cereb=False)),
        ("EF-E4",    dict(use_cc=True,  use_ia_ib=True,  use_mca=True,  use_cereb=False)),
        ("EF-cereb", dict(use_cc=False, use_ia_ib=False, use_mca=False, use_cereb=True)),
        ("EF-full",  dict(use_cc=True,  use_ia_ib=True,  use_mca=True,  use_cereb=True)),
    ]

    results: dict = {}

    for d_idx, (d_name, d_kw) in enumerate(D_CONDITIONS.items()):
        kp_scale   = d_kw.get("kp_scale", 1.0)
        torque_sat = d_kw.get("torque_saturation", None)
        results[d_name] = {}

        d_rng = np.random.default_rng(seed * 100 + d_idx)
        actual_dist_torque = float(np.clip(
            DIST_TORQUE * (1.0 + d_rng.normal(0.0, 0.25)), 20.0, 120.0,
        ))
        d_env_seed = seed * 100 + d_idx

        for cname, ckw in conds:
            use_c = ckw["use_cereb"]
            ctrl, env = make_ctrl(
                kp_scale=kp_scale, torque_sat=torque_sat,
                env_seed=d_env_seed,
                fwd_model_path=fwd_path_str if use_c else None,
                **ckw,
            )
            res = run_episode(
                ctrl, env,
                do_disturb=True, dist_torque=actual_dist_torque,
                use_cereb=use_c,
            )
            results[d_name][cname] = res

            rt  = res.get("recovery_time")
            rt_s = f"{rt:.3f}s" if rt else "未回復"
            pe   = res.get("pred_err_mrad", float("nan"))
            pe_s = f"  PredErr={pe:.1f}" if not np.isnan(pe) else ""
            print(
                f"  [{d_name}] {cname:10s}  "
                f"MAE_post={res.get('mae_post', float('nan'))*1000:7.1f} mrad  "
                f"E_post={res.get('energy_post_J', float('nan')):8.1f}  "
                f"cc={res['cc_ratio']:.3f}  RT={rt_s}{pe_s}"
            )

    # ── JSON 保存 ────────────────────────────────────────────────────
    metrics_out = {
        "experiment": "ef",
        "seed": seed,
        "sweep_name": args.sweep_name,
        "fwd_model_loaded": fwd_path_str is not None,
        "results": {
            d: {
                c: {
                    "mae_mrad":        float(r["mae"] * 1000),
                    "mae_post_mrad":   float(r.get("mae_post", float("nan")) or float("nan")) * 1000,
                    "peak_err_rad":    float(r.get("peak_err", float("nan"))),
                    "recovery_time_s": r.get("recovery_time"),
                    "energy_J":        float(r["energy_J"]),
                    "energy_post_J":   float(r.get("energy_post_J", float("nan"))),
                    "jerk":            float(r.get("jerk", float("nan"))),
                    "jerk_post":       float(r.get("jerk_post", float("nan"))),
                    "cc_ratio":        float(r["cc_ratio"]),
                    "pred_err_mrad":   float(r.get("pred_err_mrad", float("nan"))),
                }
                for c, r in cond_res.items()
            }
            for d, cond_res in results.items()
        },
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    # ── プロット ─────────────────────────────────────────────────────
    cond_names = ["EF-PD", "EF-E4", "EF-cereb", "EF-full"]
    colors     = ["tab:gray", "tab:purple", "tab:orange", "tab:green"]
    metrics_plot = [
        ("mae_post",    "MAE_post [mrad]",    1000.0),
        ("energy_post_J", "Energy_post [Nm²·s]", 1.0),
        ("cc_ratio",    "CC ratio",           1.0),
        ("pred_err_mrad", "Pred error [mrad]", 1.0),
    ]

    fig, axes = plt.subplots(len(metrics_plot), len(D_CONDITIONS), figsize=(16, 12))
    fig.suptitle(f"Experiment EF: Full 3-Layer Hierarchy  [seed={seed}]", fontsize=12)

    for col, d_name in enumerate(D_CONDITIONS):
        for row, (metric_key, ylabel, scale) in enumerate(metrics_plot):
            ax = axes[row][col]
            vals = [results[d_name].get(c, {}).get(metric_key, float("nan")) for c in cond_names]
            vals = [v * scale if not np.isnan(v) else float("nan") for v in vals]

            bars = ax.bar(range(len(cond_names)), vals, color=colors, alpha=0.8)
            ax.set_xticks(range(len(cond_names)))
            ax.set_xticklabels(cond_names, fontsize=7, rotation=15, ha="right")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f"{d_name}: {ylabel}", fontsize=8)
            valid = [v for v in vals if not np.isnan(v)]
            vmax  = max(valid) if valid else 1.0
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + vmax * 0.02,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=6,
                    )
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = outdir / "plot_franka_ef.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"\nプロット保存: {plot_path}")

    print("\n=== EF 結果サマリ ===")
    for d_name in D_CONDITIONS:
        print(f"\n  {d_name}:")
        for cname in cond_names:
            r = results[d_name][cname]
            rt  = r.get("recovery_time")
            rt_s = f"{rt:.3f}s" if rt else "未回復"
            print(
                f"    {cname:10s}  "
                f"MAE_post={r.get('mae_post', float('nan'))*1000:7.1f} mrad  "
                f"E_post={r.get('energy_post_J', float('nan')):8.1f}  "
                f"cc={r['cc_ratio']:.3f}  RT={rt_s}"
            )


if __name__ == "__main__":
    main()
