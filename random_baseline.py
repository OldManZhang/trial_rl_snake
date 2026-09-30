"""
随机策略基线 —— 这是【地板线】

运行：uv run python random_baseline.py [局数]

为什么必须先跑这个：

    没有它，后面 PPO 跑出来的任何数字你都无法判断好坏。
    📎 LunarLander 那次，「随机策略 -193.6」这个数就是这么来的 ——
       后来测试拿到 -784.5 时，才能立刻说"这比随机还差得多，肯定坏了"。

⚠️ 分数单位：每局必然以 -1 结束（撞死或饿死），所以

        评估分 = 吃到的豆子数 - 1

    所以这个脚本【只报豆子数】，不报 reward —— 免得每次都要换算。

这个文件还提供 run_episodes() / print_report()，heuristic.py 直接复用，
保证两者的口径【完全一致】才能比。
"""

import sys
from collections import Counter

import numpy as np

from snake_env import SnakeEnv

REASON_CN = {"wall": "撞墙", "self": "咬自己", "starve": "饿死", "win": "通关"}


# ============================================================
# 评测：policy(env, obs) -> action
# ============================================================
def run_episodes(policy, episodes=50, seed=0, env=None):
    own = env is None
    if own:
        env = SnakeEnv()

    beans, steps, reasons = [], [], Counter()
    for i in range(episodes):
        obs, info = env.reset(seed=seed + i)
        while True:
            a = policy(env, obs)
            obs, r, te, tr, info = env.step(a)
            if te or tr:
                break
        beans.append(info["food_eaten"])
        steps.append(info["steps"])
        reasons[info["end_reason"]] += 1

    if own:
        env.close()

    b, s = np.array(beans, dtype=float), np.array(steps, dtype=float)
    return {
        "episodes": episodes,
        "beans_mean": b.mean(), "beans_med": float(np.median(b)),
        "beans_max": b.max(), "beans_min": b.min(), "beans_std": b.std(),
        "steps_mean": s.mean(),
        "steps_per_bean": (s.sum() / b.sum()) if b.sum() > 0 else float("nan"),
        "reasons": reasons,
        "beans": beans,
    }


def print_report(name, st, highlight=None):
    """highlight: 想特别标出来的一行（比如标定 MAX_HUNGER 用的每豆步数）"""
    print(f"  {name:<14} {st['episodes']:>3} 局 │ "
          f"平均豆子 {st['beans_mean']:5.2f} │ 中位 {st['beans_med']:4.0f} │ "
          f"最好 {st['beans_max']:3.0f} │ 最差 {st['beans_min']:3.0f}")
    print(f"  {'':<14}       │ 平均步数 {st['steps_mean']:6.1f} │ "
          f"每豆步数 {st['steps_per_bean']:6.2f} │ "
          f"死因 " + "  ".join(f"{REASON_CN.get(k, k)} {v}" for k, v in st["reasons"].most_common()))
    if highlight:
        print(f"  {'':<14}       │ {highlight}")


def random_policy(env, obs):
    return env.action_space.sample()


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("=" * 92)
    print(f"随机策略基线 · {N} 局")
    print("=" * 92)
    print()
    print(f"  {'':<14}       │ ⚠️ 分数 = 豆子数 - 1（每局必然以 -1 结束）")
    print()
    print(f"  {'策略':<14} {'局数':>3}    │ {'':<10}   │ {'':<8} │ {'':<8} │")

    st = run_episodes(random_policy, episodes=N, seed=0)
    print_report("随机策略", st)

    print()
    print("=" * 92)
    print("怎么读这个数")
    print("=" * 92)
    print(f"""
  平均豆子 {st['beans_mean']:.2f}  —— 基本吃不到东西，这就是"地板线"。

  平均步数 {st['steps_mean']:.1f}：
      {'远小于 MAX_HUNGER=100 → 随机策略【先撞墙】，饥饿计时从不参与' if st['steps_mean'] < 100 else '接近 MAX_HUNGER → 定时器在起作用'}

  死因分布：{' '.join(f'{REASON_CN.get(k,k)} {v}' for k,v in st['reasons'].most_common())}

  📌 这才是 PPO 要打败的第一条线。跑完 heuristic.py 会得到第二条（目标线）。
""")
