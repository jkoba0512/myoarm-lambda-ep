"""
ReciprocalInhibition — Ia 抑制性介在ニューロンによる相反抑制。

解剖学的根拠:
  Ia 求心性線維が脊髄に入ると、拮抗筋の運動ニューロンへ
  Ia 抑制性介在ニューロンを介して抑制を送る (Eccles et al. 1956)。
  これにより屈筋収縮時に伸筋が抑制され、スムーズな協調運動が実現される。
  筋骨格制御でこれを省くと屈伸筋が同時に最大収縮し関節が破綻する。

実装:
  定義された (主動筋, 拮抗筋) ペアについて、
  主動筋の活性化 a_ag が threshold を超えた分に比例した抑制信号を
  拮抗筋活性化から減算する。

    Δa_ant[j] = -K_ri * max(a_ag[i] - threshold, 0)

  これを主動筋活性化 a から差し引くのではなく、
  呼び出し元の MyoArmController が a_total に適用する。
"""

from __future__ import annotations

import numpy as np


# myoArm 34 筋の拮抗筋ペア定義
# 形式: [(主動筋インデックス or 名前, 拮抗筋インデックス or 名前), ...]
# 名前ベースで定義し、__init__ 時にインデックスに変換する。
_ANTAGONIST_PAIRS_NAMES: list[tuple[str, str]] = [
    # 肘屈曲 ↔ 肘伸展
    ("BIClong",  "TRIlong"),
    ("BIClong",  "TRIlat"),
    ("BIClong",  "TRImed"),
    ("BICshort", "TRIlong"),
    ("BICshort", "TRIlat"),
    ("BICshort", "TRImed"),
    ("BRA",      "TRIlong"),
    ("BRA",      "TRIlat"),
    ("BRD",      "ANC"),
    # 肩外転 (DELT) ↔ 肩内転 (PECM, LAT, CORB)
    ("DELT1",    "PECM1"),
    ("DELT1",    "PECM2"),
    ("DELT2",    "LAT1"),
    ("DELT2",    "LAT2"),
    ("DELT3",    "LAT3"),
    ("DELT1",    "CORB"),
    # 肩外旋 ↔ 肩内旋
    ("INFSP",    "SUBSC"),
    ("TMIN",     "SUBSC"),
    ("SUPSP",    "SUBSC"),
    # 手首屈曲 ↔ 手首伸展
    ("FCR",      "ECRL"),
    ("FCR",      "ECRB"),
    ("FCR",      "ECU"),
    ("FCU",      "ECRL"),
    ("FCU",      "ECRB"),
    ("FCU",      "ECU"),
    ("PL",       "ECRL"),
    ("PL",       "ECRB"),
    # 回内 ↔ 回外
    ("PT",       "SUP"),
    ("PQ",       "SUP"),
]


class ReciprocalInhibition:
    """
    相反抑制（Ia 抑制性介在ニューロン）。

    Parameters
    ----------
    muscle_names : list[str]
        アクチュエータ名のリスト（順序が筋インデックスに対応）
    K_ri : float
        抑制ゲイン（主動筋の過活性化分に対する拮抗筋抑制量のスケール）
    threshold : float
        抑制が始まる活性化の閾値
    """

    def __init__(
        self,
        muscle_names: list[str],
        K_ri:      float = 0.5,
        threshold: float = 0.3,
    ) -> None:
        self.K_ri      = K_ri
        self.threshold = threshold
        self._n        = len(muscle_names)

        # (主動筋 idx, 拮抗筋 idx) ペアをインデックスで保持
        name_to_idx = {name: i for i, name in enumerate(muscle_names)}
        self._pairs: list[tuple[int, int]] = []
        for ag_name, ant_name in _ANTAGONIST_PAIRS_NAMES:
            if ag_name in name_to_idx and ant_name in name_to_idx:
                self._pairs.append((name_to_idx[ag_name], name_to_idx[ant_name]))
                # 逆方向も登録（対称的な相反抑制）
                self._pairs.append((name_to_idx[ant_name], name_to_idx[ag_name]))

    # ------------------------------------------------------------------

    def inhibit(self, act: np.ndarray) -> np.ndarray:
        """
        現在の筋活性化 `act` に対して相反抑制量（負のデルタ）を返す。

        Parameters
        ----------
        act : (n_muscles,)  現在の筋活性化 [0, 1]

        Returns
        -------
        delta_inhibit : (n_muscles,)  各筋への抑制量（≤ 0）
        """
        delta = np.zeros(self._n)
        for ag_idx, ant_idx in self._pairs:
            excess = max(act[ag_idx] - self.threshold, 0.0)
            delta[ant_idx] -= self.K_ri * excess
        return delta

    def reset(self) -> None:
        pass  # stateless
