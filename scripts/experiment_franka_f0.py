"""
実験 F0: F0-abstract vs F0-faithful 比較（静止保持タスク）

目的:
  - F0-abstract (EF-full): 旧 FrankaNeuralController をベースライン
  - F0-faithful-nodelay:  AnatomicalController (遅延=0) で実装整合性を確認
  - F0-faithful:          AnatomicalController (正常遅延 20/30 ms) を評価

比較条件:
  D0  : 標準（フル PD ゲイン、外乱あり 60 Nm）
  D1  : 低ゲイン（kp * 0.5、外乱あり）
  D5  : トルク飽和（上限 30 Nm、外乱あり）

呼び出し順の違い:
  F0-abstract  : ctrl.step → ctrl.update_cerebellum(q_current) → env.step  [旧バグ動作]
  F0-faithful  : ctrl.step → env.step → ctrl.update_cerebellum(q_actual)   [正しい順序]

出力:
  results/experiment_franka_f0/
    f0_abstract/    {d0,d1,d5}/seed{0..9}/metrics.json
    f0_faithful/    {nodelay_d0,nodelay_d1,d0,d1,d5}/seed{0..9}/metrics.json

完了条件 (F0):
  1. f0_abstract と f0_faithful_nodelay の MAE_post が同等（差 < 10%）
  2. f0_faithful_nodelay の全ステップで prop_delay_steps=0, cereb_delay_steps=0 を確認
  3. 全条件で共通 JSON キーが揃っている
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch as _torch

from common.franka_env import FrankaEnv, N_JOINTS
from common.franka_neural_controller import FrankaNeuralController
from methodF import AnatomicalController, AnatomicalConfig

# ── 定数 ─────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if _torch.cuda.is_available() else "cpu"
SIM_DURATION = 6.0
DIST_T       = 3.0
DIST_STEPS   = 20
DIST_JOINT   = 1
DIST_TORQUE  = 60.0

Q_OFFSET = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])
KP_DEFAULT = np.array([50., 50., 50., 50., 10., 10., 10.])
KD_DEFAULT = np.array([ 7.,  7.,  7.,  7.,  1.5, 1.5, 1.5])

CPG_PARAMS = dict(tau=0.3, tau_r=0.6, beta=2.5, w=2.0, amplitude=0.0)

D_CONDITIONS: dict[str, dict] = {
    "D0": {},
    "D1": {"kp_scale": 0.5},
    "D5": {"torque_saturation": 30.0},
}

E1_DIR     = ROOT / "results" / "experiment_franka_e1"
RESULTS_DIR = ROOT / "results" / "experiment_franka_f0"

SEEDS_DEFAULT = list(range(10))


# ── メトリクス計算 ─────────────────────────────────────────────────────────

def _jerk_rms(qa: np.ndarray, dt: float) -> float:
    """Jerk RMS = sqrt(mean((d³q/dt³)²)) [rad/s³] (全関節平均)"""
    if len(qa) < 4:
        return float("nan")
    d3q = np.diff(qa, n=3, axis=0) / (dt ** 3)
    return float(np.sqrt(np.mean(d3q ** 2)))


def _oscillation_freq(err_1d: np.ndarray, dt: float) -> float:
    """外乱後誤差の支配周波数 [Hz] を FFT で推定する。"""
    if len(err_1d) < 8:
        return float("nan")
    n = len(err_1d)
    fft = np.abs(np.fft.rfft(err_1d - err_1d.mean()))
    freqs = np.fft.rfftfreq(n, d=dt)
    if len(freqs) < 2:
        return float("nan")
    peak_idx = int(np.argmax(fft[1:])) + 1
    return float(freqs[peak_idx])


def compute_metrics(
    t_arr:  np.ndarray,
    q_arr:  np.ndarray,
    tau_arr: np.ndarray,
    dt:     float,
    io_stats: dict | None = None,
) -> dict:
    """共通メトリクスを計算して dict で返す。"""
    err_arr  = np.abs(q_arr - Q_OFFSET)
    mask_pre  = t_arr < DIST_T
    mask_post = t_arr > DIST_T

    mae_pre  = float(err_arr[mask_pre].mean())  if mask_pre.any()  else float("nan")
    mae_post = float(err_arr[mask_post].mean()) if mask_post.any() else float("nan")

    # ピーク誤差・回復時間
    peak_err = float("nan")
    recovery_time = None
    if mask_post.any():
        post_err_j  = err_arr[mask_post, DIST_JOINT]
        post_t      = t_arr[mask_post]
        peak_err    = float(post_err_j.max())
        rec_idx     = np.where(post_err_j < 0.1)[0]
        if len(rec_idx):
            recovery_time = float(post_t[rec_idx[0]] - DIST_T)

    # オーバーシュート比（post ピーク / pre 定常誤差）
    pre_steady = float(err_arr[mask_pre, DIST_JOINT].mean()) if mask_pre.any() else float("nan")
    overshoot_ratio = (peak_err / (pre_steady + 1e-9)) if not np.isnan(peak_err) else float("nan")

    # 振動周波数
    if mask_post.any():
        osc_freq = _oscillation_freq(err_arr[mask_post, DIST_JOINT], dt)
    else:
        osc_freq = float("nan")

    # エネルギー・ジャーク
    energy     = float(np.sum(tau_arr ** 2) * dt)
    energy_post = float(np.sum(tau_arr[mask_post] ** 2) * dt) if mask_post.any() else float("nan")
    jerk_rms   = _jerk_rms(q_arr, dt)

    m: dict = {
        "MAE_pre":          round(mae_pre * 1000, 4),
        "MAE_post":         round(mae_post * 1000, 4),
        "peak_err_rad":     round(peak_err, 6),
        "recovery_time_s":  recovery_time,
        "overshoot_ratio":  round(float(overshoot_ratio), 4),
        "oscillation_freq": round(osc_freq, 4),
        "energy_J":         round(energy, 4),
        "energy_post_J":    round(energy_post, 4),
        "jerk_rms":         round(jerk_rms, 6),
    }

    if io_stats:
        m["io_fire_count"]         = io_stats.get("io_fire_count", 0)
        m["io_fire_rate_hz"]       = round(io_stats.get("io_fire_rate_hz", 0.0), 4)
        m["io_fire_interval_mean"] = round(io_stats.get("io_fire_interval_mean", 0.0), 4)
    else:
        m["io_fire_count"]         = None
        m["io_fire_rate_hz"]       = None
        m["io_fire_interval_mean"] = None

    return m


# ── エピソード実行 ──────────────────────────────────────────────────────────

def run_abstract(
    ctrl: FrankaNeuralController,
    env:  FrankaEnv,
    seed: int,
    dist_torque: float,
    use_cereb:   bool,
) -> dict:
    """
    F0-abstract (EF-full) のエピソードを実行する。
    既存スクリプトの動作に合わせ、update_cerebellum を env.step 前に呼ぶ。
    """
    rng = np.random.default_rng(seed)
    actual_torque = float(np.clip(dist_torque * (1.0 + rng.normal(0.0, 0.25)), 20.0, 120.0))

    q, dq = env.reset(q0=Q_OFFSET.copy())
    ctrl.reset()
    dt = env.dt

    t_log, q_log, tau_log = [], [], []
    dist_applied = False

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()

        if not dist_applied and t >= DIST_T:
            td = np.zeros(N_JOINTS)
            td[DIST_JOINT] = actual_torque
            env.apply_disturbance(td, duration_steps=DIST_STEPS)
            dist_applied = True
            continue

        tau, _ = ctrl.step(q, dq, Q_OFFSET)

        # 旧バグ動作: env.step 前に update_cerebellum (q_current)
        if use_cereb:
            ctrl.update_cerebellum(q)

        env.step(tau)
        t_log.append(t)
        q_log.append(q.copy())
        tau_log.append(tau.copy())

    return compute_metrics(
        np.array(t_log), np.array(q_log), np.array(tau_log), dt,
    )


def run_faithful(
    ctrl: AnatomicalController,
    env:  FrankaEnv,
    seed: int,
    dist_torque: float,
) -> dict:
    """
    F0-faithful (AnatomicalController) のエピソードを実行する。
    正しい呼び出し順: ctrl.step → env.step → ctrl.update_cerebellum(q_actual)
    """
    rng = np.random.default_rng(seed)
    actual_torque = float(np.clip(dist_torque * (1.0 + rng.normal(0.0, 0.25)), 20.0, 120.0))

    q, dq = env.reset(q0=Q_OFFSET.copy())
    ctrl.reset()
    dt = env.dt

    t_log, q_log, tau_log = [], [], []
    dist_applied = False

    while env.time < SIM_DURATION:
        t = env.time
        q, dq = env.get_state()

        if not dist_applied and t >= DIST_T:
            td = np.zeros(N_JOINTS)
            td[DIST_JOINT] = actual_torque
            env.apply_disturbance(td, duration_steps=DIST_STEPS)
            dist_applied = True
            continue

        tau, _ = ctrl.step(q, dq, Q_OFFSET)
        env.step(tau)
        q_actual, _ = env.get_state()
        ctrl.update_cerebellum(q_actual)

        t_log.append(t)
        q_log.append(q.copy())
        tau_log.append(tau.copy())

    io_stats = ctrl.get_io_stats()
    return compute_metrics(
        np.array(t_log), np.array(q_log), np.array(tau_log), dt,
        io_stats=io_stats,
    )


# ── コントローラ生成 ────────────────────────────────────────────────────────

def make_abstract(
    kp_scale:       float        = 1.0,
    torque_sat:     float | None = None,
    env_seed:       int          = 9999,
    fwd_model_path: str  | None  = None,
) -> tuple[FrankaNeuralController, FrankaEnv]:
    kp = KP_DEFAULT * kp_scale
    env_kw: dict = {}
    if torque_sat is not None:
        env_kw["torque_saturation"] = torque_sat

    env = FrankaEnv(rng=np.random.default_rng(env_seed), obs_noise_std=0.002, **env_kw)
    ctrl = FrankaNeuralController(
        dt=env.dt, q_range=env.ctrl_range,
        cpg_params=CPG_PARAMS,
        kp=kp, kd=KD_DEFAULT.copy(),
        use_proprioceptor=False,
        use_reflex=True,
        use_ia_ib_reflex=True,
        use_cerebellum=True,
        use_forward_model=True,
        use_cocontraction=True,
        use_motor_cortex=True,
        cpg_alpha_fb=0.0,
        cfc_hidden_units=64,
        device=DEVICE,
    )
    if fwd_model_path:
        ctrl.load_cerebellum(fwd_model_path)
    return ctrl, env


def make_faithful(
    kp_scale:         float        = 1.0,
    torque_sat:       float | None = None,
    env_seed:         int          = 9999,
    prop_delay_steps: int          = 10,
    cereb_delay_steps: int         = 15,
    cfc_path:         str  | None  = None,
    ctrl_seed:        int          = 0,
) -> tuple[AnatomicalController, FrankaEnv]:
    kp = KP_DEFAULT * kp_scale
    env_kw: dict = {}
    if torque_sat is not None:
        env_kw["torque_saturation"] = torque_sat

    env = FrankaEnv(rng=np.random.default_rng(env_seed), obs_noise_std=0.002, **env_kw)

    cfg = AnatomicalConfig(
        prop_delay_steps=prop_delay_steps,
        cereb_delay_steps=cereb_delay_steps,
        kp=kp,
        kd=KD_DEFAULT.copy(),
        io_mode="sparse",
        io_firing_rate_hz=1.0,
        io_gain=5.0,
        inverse_model_loc="m1",
        efcopy_enabled=True,
        cfc_hidden_units=64,
        cfc_device=DEVICE,
    )
    ctrl = AnatomicalController(cfg, seed=ctrl_seed)
    if cfc_path:
        ctrl.load_cfc(cfc_path)
    return ctrl, env


# ── seed 単位のメイン処理 ────────────────────────────────────────────────────

def run_seed(seed: int, args: argparse.Namespace) -> None:
    fwd_path = E1_DIR / f"seed{seed}" / "cfc_forward.pt"
    if not fwd_path.exists():
        print(f"  WARNING: {fwd_path} が見つかりません。CfC なしで実行します。")
        fwd_path_str = None
    else:
        fwd_path_str = str(fwd_path)

    print(f"\nseed={seed}  CfC={'loaded' if fwd_path_str else 'NONE'}")

    for d_name, d_kw in D_CONDITIONS.items():
        kp_scale   = d_kw.get("kp_scale", 1.0)
        torque_sat = d_kw.get("torque_saturation", None)
        env_seed   = seed * 100 + list(D_CONDITIONS).index(d_name)

        # ── F0-abstract ───────────────────────────────────────────────
        out_dir = RESULTS_DIR / "f0_abstract" / d_name.lower() / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ctrl_abs, env_abs = make_abstract(
            kp_scale=kp_scale, torque_sat=torque_sat,
            env_seed=env_seed, fwd_model_path=fwd_path_str,
        )
        m_abs = run_abstract(ctrl_abs, env_abs, seed, DIST_TORQUE, use_cereb=True)
        m_abs.update({
            "condition": f"f0_abstract_{d_name.lower()}",
            "controller": "f0_abstract",
            "prop_delay_steps": 0,
            "cereb_delay_steps": 0,
            "io_mode": None,
            "seed": seed,
        })
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(m_abs, f, indent=2, ensure_ascii=False)

        rt_abs = "n/a" if m_abs["recovery_time_s"] is None else f"{m_abs['recovery_time_s']:.3f}s"
        print(f"  [{d_name}] abstract   MAE_post={m_abs['MAE_post']:7.2f} mrad  RT={rt_abs}")

        # ── F0-faithful-nodelay ───────────────────────────────────────
        out_dir = RESULTS_DIR / "f0_faithful" / f"nodelay_{d_name.lower()}" / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ctrl_nd, env_nd = make_faithful(
            kp_scale=kp_scale, torque_sat=torque_sat,
            env_seed=env_seed,
            prop_delay_steps=0, cereb_delay_steps=0,
            cfc_path=fwd_path_str, ctrl_seed=seed,
        )
        m_nd = run_faithful(ctrl_nd, env_nd, seed, DIST_TORQUE)
        m_nd.update({
            "condition": f"f0_faithful_nodelay_{d_name.lower()}",
            "controller": "f0_faithful_nodelay",
            "prop_delay_steps": 0,
            "cereb_delay_steps": 0,
            "io_mode": "sparse",
            "seed": seed,
        })
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(m_nd, f, indent=2, ensure_ascii=False)

        rt_nd = "n/a" if m_nd["recovery_time_s"] is None else f"{m_nd['recovery_time_s']:.3f}s"
        print(f"  [{d_name}] nodelay    MAE_post={m_nd['MAE_post']:7.2f} mrad  RT={rt_nd}")

        # ── F0-faithful（正常遅延）───────────────────────────────────
        out_dir = RESULTS_DIR / "f0_faithful" / d_name.lower() / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ctrl_f, env_f = make_faithful(
            kp_scale=kp_scale, torque_sat=torque_sat,
            env_seed=env_seed,
            prop_delay_steps=10, cereb_delay_steps=15,
            cfc_path=fwd_path_str, ctrl_seed=seed,
        )
        m_f = run_faithful(ctrl_f, env_f, seed, DIST_TORQUE)
        m_f.update({
            "condition": f"f0_faithful_{d_name.lower()}",
            "controller": "f0_faithful",
            "prop_delay_steps": 10,
            "cereb_delay_steps": 15,
            "io_mode": "sparse",
            "seed": seed,
        })
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(m_f, f, indent=2, ensure_ascii=False)

        rt_f = "n/a" if m_f["recovery_time_s"] is None else f"{m_f['recovery_time_s']:.3f}s"
        print(f"  [{d_name}] faithful   MAE_post={m_f['MAE_post']:7.2f} mrad  RT={rt_f}")


# ── F0 完了条件チェック ───────────────────────────────────────────────────────

def check_completion(seeds: list[int]) -> None:
    """全完了条件を確認して結果を表示する。"""
    print("\n" + "=" * 60)
    print("F0 完了条件チェック")
    print("=" * 60)

    for d_name in D_CONDITIONS:
        abs_maes, nd_maes, f_maes = [], [], []
        nd_delays_ok = True

        for seed in seeds:
            for variant, maes in [
                ("f0_abstract",          abs_maes),
                ("f0_faithful/nodelay_", nd_maes),
                ("f0_faithful",          f_maes),
            ]:
                if "nodelay" in variant:
                    p = RESULTS_DIR / "f0_faithful" / f"nodelay_{d_name.lower()}" / f"seed{seed}" / "metrics.json"
                else:
                    p = RESULTS_DIR / variant / d_name.lower() / f"seed{seed}" / "metrics.json"
                if p.exists():
                    with open(p) as fh:
                        d = json.load(fh)
                    v = d.get("MAE_post")
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        maes.append(v)
                    # 遅延ステップ確認
                    if "nodelay" in variant:
                        if d.get("prop_delay_steps") != 0 or d.get("cereb_delay_steps") != 0:
                            nd_delays_ok = False

        if abs_maes and nd_maes:
            ratio = abs(np.mean(abs_maes) - np.mean(nd_maes)) / (np.mean(abs_maes) + 1e-9)
            ok1 = ratio < 0.10
            print(
                f"[{d_name}] Abstract={np.mean(abs_maes):.2f} vs nodelay={np.mean(nd_maes):.2f} "
                f"mrad  diff={ratio*100:.1f}%  {'OK' if ok1 else 'NG'}"
            )
        if nd_maes:
            print(f"[{d_name}] nodelay遅延ステップ=0: {'OK' if nd_delays_ok else 'NG'}")
        if f_maes:
            print(f"[{d_name}] faithful MAE_post = {np.mean(f_maes):.2f} ± {np.std(f_maes):.2f} mrad")


# ── argparse ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="F0: F0-abstract vs F0-faithful 比較実験"
    )
    p.add_argument("--seed",  type=int, default=None,
                   help="単一シード（省略時は --seeds で指定）")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT,
                   help="実行するシードのリスト（デフォルト: 0〜9）")
    p.add_argument("--check-only", action="store_true",
                   help="実験を実行せず完了条件チェックのみ")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    seeds = [args.seed] if args.seed is not None else args.seeds

    if args.check_only:
        check_completion(seeds)
        return

    print(f"F0 実験開始  seeds={seeds}  device={DEVICE}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        run_seed(seed, args)

    check_completion(seeds)
    print("\n完了")


if __name__ == "__main__":
    main()
