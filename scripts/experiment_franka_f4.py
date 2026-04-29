"""
実験 F4: 登上線維疎性（仮説 H4 検証）

仮説 H4:
  下オリーブ核の散発発火（〜1 Hz）は、毎ステップ発火（continuous）よりも
  学習効率が高い（ノイズが少なく有意な誤差のみ学習する）。

  [注意] F4 は学習効率の比較なのでオンライン学習（online_lr > 0）が必要。
  静的な事前訓練済みモデルだけでは学習曲線が現れない。
  online_lr=0.0005 で online adaptation を有効にして実験する。

実験設計:
  io_mode と io_firing_rate_hz のみを変化させる。
  prop_delay_steps=10, cereb_delay_steps=15 で固定。
  エピソード長は 60 秒（オンライン学習曲線を観察するため）。

事前指定評価指標 (pre-specified, H4):
  - 学習曲線: エピソードを 6 s ブロック × 10 で記録し MAE_block[i] をプロット
  - io_fire_count / io_fire_rate_hz: 実際の発火頻度
  - final_MAE_post: 最終ブロックの MAE_post（収束後の性能）

条件:
  f4_continuous  : io_mode="continuous" （毎ステップ発火）
  f4_10hz        : io_mode="sparse",  firing_rate_hz=10.0
  f4_1hz         : io_mode="sparse",  firing_rate_hz=1.0  （生体正常）
  f4_error_gated : io_mode="error_gated" （誤差閾値超時のみ）

出力:
  results/experiment_franka_f4/
    {f4_continuous,f4_10hz,f4_1hz,f4_error_gated}/seed{0..9}/metrics.json
    f4_summary.json
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
from methodF import AnatomicalController, AnatomicalConfig
from methodB.cfc_forward_model import CfCForwardModel

DEVICE       = "cuda" if _torch.cuda.is_available() else "cpu"
# F4 はエピソード長 60 s（学習曲線観察用）
SIM_DURATION  = 60.0
BLOCK_DURATION = 6.0   # ブロック分割単位
N_BLOCKS       = int(SIM_DURATION / BLOCK_DURATION)

# F4 では外乱なし（定常追従で学習曲線を観察）
DIST_T       = float("inf")

Q_OFFSET   = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.5, 0.0])
KP_DEFAULT = np.array([50., 50., 50., 50., 10., 10., 10.])
KD_DEFAULT = np.array([ 7.,  7.,  7.,  7.,  1.5, 1.5, 1.5])

PROP_DELAY_FIXED  = 10
CEREB_DELAY_FIXED = 15
ONLINE_LR         = 0.0005  # オンライン適応学習率

CONDITIONS = {
    "f4_continuous":  {"io_mode": "continuous", "io_firing_rate_hz": 500.0},
    "f4_10hz":        {"io_mode": "sparse",     "io_firing_rate_hz": 10.0},
    "f4_1hz":         {"io_mode": "sparse",     "io_firing_rate_hz": 1.0},
    "f4_error_gated": {"io_mode": "error_gated", "io_firing_rate_hz": 1.0},
}

E1_DIR      = ROOT / "results" / "experiment_franka_e1"
RESULTS_DIR = ROOT / "results" / "experiment_franka_f4"
SEEDS_DEFAULT = list(range(10))


def run_faithful_blocks(ctrl, env, seed) -> dict:
    """60 秒エピソードを 6 秒ブロックに分割して MAE を記録する。"""
    q, dq = env.reset(q0=Q_OFFSET.copy())
    ctrl.reset()
    dt = env.dt

    block_maes: list[float] = []
    block_errs: list[list[float]] = []
    current_block: list[float] = []
    next_block_t  = BLOCK_DURATION

    while env.time < SIM_DURATION:
        t  = env.time
        q, dq = env.get_state()

        # ブロック区切り
        if t >= next_block_t:
            if current_block:
                block_maes.append(float(np.mean(current_block)))
                block_errs.append(current_block[:])
                current_block = []
            next_block_t += BLOCK_DURATION

        tau, _ = ctrl.step(q, dq, Q_OFFSET)
        env.step(tau)
        q_actual, _ = env.get_state()
        ctrl.update_cerebellum(q_actual)

        err = float(np.mean(np.abs(q - Q_OFFSET)) * 1000)
        current_block.append(err)

    if current_block:
        block_maes.append(float(np.mean(current_block)))

    io_stats = ctrl.get_io_stats()
    return {
        "block_maes_mrad":     block_maes,
        "final_MAE_post":      block_maes[-1] if block_maes else float("nan"),
        "initial_MAE":         block_maes[0]  if block_maes else float("nan"),
        "MAE_improvement":     ((block_maes[0] - block_maes[-1]) / (block_maes[0] + 1e-9)
                                if len(block_maes) >= 2 else float("nan")),
        "io_fire_count":       io_stats.get("io_fire_count", 0),
        "io_fire_rate_hz":     round(io_stats.get("io_fire_rate_hz", 0.0), 4),
        "io_fire_interval_mean": round(io_stats.get("io_fire_interval_mean", 0.0), 4),
    }


def make_ctrl(io_mode, io_firing_rate_hz, env_seed, cfc_path, ctrl_seed):
    env = FrankaEnv(rng=np.random.default_rng(env_seed), obs_noise_std=0.002)
    cfg = AnatomicalConfig(
        prop_delay_steps=PROP_DELAY_FIXED,
        cereb_delay_steps=CEREB_DELAY_FIXED,
        kp=KP_DEFAULT.copy(), kd=KD_DEFAULT.copy(),
        io_mode=io_mode,
        io_firing_rate_hz=io_firing_rate_hz,
        io_gain=5.0,
        inverse_model_loc="m1",
        efcopy_enabled=True,
        cfc_hidden_units=64,
        cfc_device=DEVICE,
    )
    ctrl = AnatomicalController(cfg, seed=ctrl_seed)
    if cfc_path:
        # オンライン学習を有効化してから CfC をロード
        ctrl._cerebellum.cfc.online_lr = ONLINE_LR
        ctrl.load_cfc(cfc_path)
    return ctrl, env


def run_seed(seed: int) -> None:
    fwd_path = E1_DIR / f"seed{seed}" / "cfc_forward.pt"
    fwd_path_str = str(fwd_path) if fwd_path.exists() else None
    print(f"\nseed={seed}")

    for cond_name, cond_kw in CONDITIONS.items():
        out_dir = RESULTS_DIR / cond_name / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ctrl, env = make_ctrl(
            io_mode=cond_kw["io_mode"],
            io_firing_rate_hz=cond_kw["io_firing_rate_hz"],
            env_seed=seed * 100,
            cfc_path=fwd_path_str,
            ctrl_seed=seed,
        )
        m = run_faithful_blocks(ctrl, env, seed)
        m.update({
            "condition":         cond_name,
            "io_mode":           cond_kw["io_mode"],
            "io_firing_rate_hz": cond_kw["io_firing_rate_hz"],
            "online_lr":         ONLINE_LR,
            "seed":              seed,
        })

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)

        impr = m["MAE_improvement"] * 100
        print(
            f"  {cond_name:18s}  initial={m['initial_MAE']:6.2f}  "
            f"final={m['final_MAE_post']:6.2f} mrad  "
            f"impr={impr:+.1f}%  io={m['io_fire_rate_hz']:.2f}Hz"
        )


def build_summary(seeds: list[int]) -> dict:
    summary: dict = {
        "experiment": "f4",
        "hypothesis": "H4: 散発IO発火 → 高い学習効率",
        "seeds": seeds,
        "online_lr": ONLINE_LR,
        "conditions": {},
    }
    for cond_name in CONDITIONS:
        finals, imprs, io_rates = [], [], []
        all_block_maes: list[list[float]] = []
        for seed in seeds:
            p = RESULTS_DIR / cond_name / f"seed{seed}" / "metrics.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            finals.append(d.get("final_MAE_post", float("nan")))
            imprs.append(d.get("MAE_improvement", float("nan")))
            io_rates.append(d.get("io_fire_rate_hz", float("nan")))
            if "block_maes_mrad" in d:
                all_block_maes.append(d["block_maes_mrad"])

        mean_curve = None
        if all_block_maes:
            min_len = min(len(b) for b in all_block_maes)
            arr = np.array([b[:min_len] for b in all_block_maes])
            mean_curve = [round(float(v), 3) for v in arr.mean(axis=0)]

        summary["conditions"][cond_name] = {
            "final_MAE_post_mean":   round(float(np.nanmean(finals)), 3) if finals else None,
            "MAE_improvement_mean":  round(float(np.nanmean(imprs)), 4) if imprs else None,
            "io_fire_rate_hz_mean":  round(float(np.nanmean(io_rates)), 4) if io_rates else None,
            "learning_curve_mean":   mean_curve,
            "n_seeds":               len(finals),
        }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F4: 登上線維疎性実験 (仮説 H4)")
    p.add_argument("--seed",  type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--check-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    seeds = [args.seed] if args.seed is not None else args.seeds

    if not args.check_only:
        print(f"F4 実験開始  seeds={seeds}  SIM={SIM_DURATION}s  device={DEVICE}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            run_seed(seed)

    summary = build_summary(seeds)
    out = RESULTS_DIR / "f4_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("F4 集計結果")
    print("=" * 70)
    print(f"{'condition':>20}  {'final_MAE':>10}  {'improvement':>12}  {'io_rate':>8}")
    for cname, data in summary["conditions"].items():
        mae_s  = f"{data['final_MAE_post_mean']:.2f}" if data["final_MAE_post_mean"] else "N/A"
        impr_s = f"{data['MAE_improvement_mean']*100:+.1f}%" if data["MAE_improvement_mean"] else "N/A"
        io_s   = f"{data['io_fire_rate_hz_mean']:.2f}Hz" if data["io_fire_rate_hz_mean"] else "N/A"
        print(f"  {cname:>18}  {mae_s:>10}  {impr_s:>12}  {io_s:>8}")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
