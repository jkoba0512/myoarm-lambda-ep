"""
env_utils.py — myoArmReach 系 env の決定論的 reset ヘルパ。

問題:
  myosuite/envs/myo/myobase/reach_v0.py の `reset(seed=N)` は、
  generate_target_pose() が super().reset(**kwargs) より前に呼ばれている。
  さらに myosuite の BaseV0.reset は seed を robot.reset へ渡すだけで
  `env.np_random` を再シードしない。
  結果: env.reset(seed=N) を呼んでも np_random 状態は前回の続きから進み、
  「同じ seed」が毎回違うターゲットを返す (=実験の seed 再現性が崩壊)。

修正:
  reset の前に env.unwrapped.np_random を seed_envs(N) で手動再シードしてから
  reset を呼ぶ。これで generate_target_pose が決定論的に同じ uniform を引く。

検証:
  trial を 3 回繰り返して seed=[0,1,2,0,7,0] を投げると、3 trial × 6 seed の
  全 18 reset で「同じ seed → 同じターゲット」を確認 (2026-04-30)。
"""

from __future__ import annotations

import gymnasium as gym
from myosuite.utils import seed_envs


def deterministic_reset(env: gym.Env, seed: int):
    """seed を決定論的に効かせる reset (myoArmReachRandom-v0 等の seed バグ対策)。

    Parameters
    ----------
    env  : gymnasium.Env (myoArmReach 系)
    seed : int

    Returns
    -------
    (obs, info) — gym.Env.reset の戻り値と同形式

    Notes
    -----
    - 通常の env.reset(seed=N) は使わないこと (seed が効かない)
    - env を作り直しても seed パラメータは効かないので、本ヘルパが必須
    """
    np_rand, _ = seed_envs(seed)
    env.unwrapped.np_random = np_rand
    return env.reset()
