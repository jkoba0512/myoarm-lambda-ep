"""
論文用の主結果・補助解析・claim対応表を生成する。

入力:
  results/franka_master_summary.json

出力:
  results/franka_main_results.json
  results/franka_secondary_results.json
  results/franka_claim_map.json

使い方:
  .venv/bin/python scripts/export_paper_results.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[1]
RESULTS_DIR = ROOT / "results"
MASTER_PATH = RESULTS_DIR / "franka_master_summary.json"
MAIN_PATH = RESULTS_DIR / "franka_main_results.json"
SECONDARY_PATH = RESULTS_DIR / "franka_secondary_results.json"
CLAIM_MAP_PATH = RESULTS_DIR / "franka_claim_map.json"


def load_master() -> dict[str, Any]:
    with open(MASTER_PATH) as f:
        return json.load(f)


def export_main_results(master: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": {
            "platform": "Franka Panda MuJoCo torque-control simulation",
            "status": "primary results",
            "note": "Main metrics only. tau_sys-related analysis is intentionally excluded.",
        },
        "seeds_used": master.get("seeds_used", []),
        "main_metrics": {
            "2a_static_holding": {
                "metric": "static_mae_mrad",
                "conditions": {
                    "PD": master["2a"]["PD"]["static_mae_mrad"],
                    "PD+CfC": master["2a"]["PD+CfC"]["static_mae_mrad"],
                    "Full": master["2a"]["Full"]["static_mae_mrad"],
                },
                "effect_size": master["2a"]["effect_size"],
            },
            "2b_disturbance": {
                "metrics": ["peak_err_rad", "recovery_time_s"],
                "levels": {
                    level: {
                        "PD+CfC": {
                            "peak_err_rad": master["2b"][level]["PD+CfC"]["peak_err_rad"],
                            "recovery_time_s": master["2b"][level]["PD+CfC"]["recovery_time_s"],
                        },
                        "Full": {
                            "peak_err_rad": master["2b"][level]["Full"]["peak_err_rad"],
                            "recovery_time_s": master["2b"][level]["Full"]["recovery_time_s"],
                        },
                        "effect_size": master["2b"][level]["effect_size"],
                    }
                    for level in ["light_30Nm", "medium_60Nm", "heavy_87Nm"]
                },
            },
            "2c_cyclic_motion": {
                "metrics": ["ep_err_post_mrad", "mae_post_mrad"],
                "conditions": {
                    "CPG+CfC": {
                        "ep_err_post_mrad": master["2c"]["CPG+CfC"]["ep_err_post_mrad"],
                        "mae_post_mrad": master["2c"]["CPG+CfC"]["mae_post_mrad"],
                    },
                    "CPG+CfC+LIF_FB": {
                        "ep_err_post_mrad": master["2c"]["CPG+CfC+LIF_FB"]["ep_err_post_mrad"],
                        "mae_post_mrad": master["2c"]["CPG+CfC+LIF_FB"]["mae_post_mrad"],
                    },
                },
                "effect_size": master["2c"]["effect_size"],
            },
            "2d_integrated": {
                "metrics": ["static_mae_mrad", "peak_err_rad", "recovery_time_s"],
                "conditions": {
                    cond: {
                        "static_mae_mrad": master["2d"][cond]["static_mae_mrad"],
                        "peak_err_rad": master["2d"][cond]["peak_err_rad"],
                        "recovery_time_s": master["2d"][cond]["recovery_time_s"],
                    }
                    for cond in ["PD", "PD+CfC", "Full"]
                },
                "effect_size": master["2d"]["effect_size"],
            },
        },
    }


def export_secondary_results(master: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": {
            "status": "secondary_or_exploratory_results",
            "note": "Secondary metrics and internal-dynamics-facing quantities. Not for headline claims.",
        },
        "seeds_used": master.get("seeds_used", []),
        "secondary_metrics": {
            "2c_reference_alignment": {
                "metric": "ep_err_pre_mrad",
                "conditions": {
                    "CPG+CfC": master["2c"]["CPG+CfC"].get("ep_err_pre_mrad"),
                    "CPG+CfC+LIF_FB": master["2c"]["CPG+CfC+LIF_FB"].get("ep_err_pre_mrad"),
                },
            },
            "notes": [
                "tau_sys should be read from analyze_tau_sys_cartesian outputs and related figures.",
                "Cartesian error belongs to auxiliary analysis, not the main result table.",
                "Reflex torque and LIF firing rate are diagnostic plots rather than main comparison metrics.",
            ],
            "analysis_sources": {
                "master_summary": str(MASTER_PATH.relative_to(ROOT)),
                "cartesian_tau_analysis_dir": "results/analyze_tau_sys_cartesian",
                "summary_figures_dir": "results/summary_figures",
            },
        },
    }


def export_claim_map() -> dict[str, Any]:
    return {
        "paper_claims": [
            {
                "claim_id": "C1",
                "claim": "The integrated neuro-inspired controller improves static holding over a PD baseline in Franka Panda simulation.",
                "supported_by": [
                    "results/franka_main_results.json -> main_metrics.2a_static_holding",
                    "results/summary_figures/fig1_ablation_mae.png",
                    "results/summary_figures/fig4_contribution_matrix.png",
                ],
            },
            {
                "claim_id": "C2",
                "claim": "CfC is the dominant contributing module, while reflex and LIF feedback are auxiliary contributors.",
                "supported_by": [
                    "results/franka_contribution_table.json",
                    "results/franka_main_results.json -> main_metrics.2a_static_holding",
                    "results/franka_main_results.json -> main_metrics.2b_disturbance",
                    "results/franka_main_results.json -> main_metrics.2c_cyclic_motion",
                    "results/summary_figures/fig4_contribution_matrix.png",
                ],
            },
            {
                "claim_id": "C3",
                "claim": "The controller shows task-dependent benefits across static holding, disturbance response, and cyclic motion in Franka Panda simulation.",
                "supported_by": [
                    "results/franka_main_results.json",
                    "results/summary_figures/fig1_ablation_mae.png",
                    "results/summary_figures/fig2_disturbance.png",
                    "results/summary_figures/fig3_cyclic.png",
                ],
            },
        ],
        "secondary_claims": [
            {
                "claim_id": "S1",
                "claim": "tau_sys-related quantities are exploratory internal-dynamics observations rather than headline performance claims.",
                "supported_by": [
                    "results/franka_secondary_results.json",
                    "results/analyze_tau_sys_cartesian",
                ],
            }
        ],
    }


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    master = load_master()
    main_results = export_main_results(master)
    secondary_results = export_secondary_results(master)
    claim_map = export_claim_map()

    dump_json(MAIN_PATH, main_results)
    dump_json(SECONDARY_PATH, secondary_results)
    dump_json(CLAIM_MAP_PATH, claim_map)

    print(f"保存: {MAIN_PATH}")
    print(f"保存: {SECONDARY_PATH}")
    print(f"保存: {CLAIM_MAP_PATH}")


if __name__ == "__main__":
    main()
