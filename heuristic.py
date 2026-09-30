"""
手写规则 AI —— 这是【目标线】

运行：uv run python heuristic.py [局数]

为什么需要它：

    跑完 PPO 你会得到一个数字，但**没有对照，你无法判断它是好是坏**。
    📎 LunarLander 那次就是拿环境自带的 heur_lander 当对照 ——
       结果手写的 1.828 比训练出来的 PPO 的 1.196 还差，这个对比本身就很有信息量。

⚠️ 它【作弊】：能看全盘（蛇身、食物、网格），不是公平对手。
    这是故意的 —— 它要当【上界参照】，不是当"另一个策略"。
    如果限制它只能用和网络一样的 12 维特征，它就退化成"另一个 PPO"，
    参考价值反而更低。

⚠️ 它复用 env 的 `_simulate()` / `_reachable()`，而不是自己重写碰撞逻辑。
    否则测出来的是「手写规则和 env 规则哪个对」，不是「规则 AI 有多强」。

============================================================
两档，一好一坏：

  朴素贪心      三个方向里挑安全的，选离豆最近的
                → 只吃豆，不规划空间
                → 预期：吃 8~15 个后【自陷身亡】
                → 它的意义：证明「只看 1 格视野」的天花板有多低

  贪心 + 空间   在安全的基础上，优先选「走完还剩足够空间」的
                （用 flood fill，和状态里的 [9:12] 是同一个东西）
                → 预期：吃 20~40 个
                → 它的意义：① 证明 8×8 这个环境是【可解的】
                            ② 给出 PPO 的靶子
                            ③ 给出 MAX_HUNGER 的标定值
============================================================
"""

import sys

import numpy as np

from snake_env import SnakeEnv, EMPTY
from random_baseline import run_episodes, print_report

# 贪心+空间那一档的门槛：走完之后剩下的可达空间至少要这么多
#   太松 = 退化成朴素贪心；太紧 = 正常的好棋也被否掉
SPACE_MARGIN = 2


def _candidates(env):
    """三个动作里所有【不会立刻死】的，附带 (到食物距离, 走完后的可达空间)"""
    head = env.snake[0]
    out = []
    for a in range(3):
        nh, will_eat, danger = env._simulate(a)
        if danger != EMPTY:
            continue
        dist = (abs(nh[0] - env.food[0]) + abs(nh[1] - env.food[1])) if env.food else 0
        occ = set(env.snake) if will_eat else set(env.snake[:-1])
        occ.discard(nh)
        space = env._reachable(nh, occ)
        out.append({"a": a, "dist": dist, "space": space})
    return out


def make_greedy(use_space, seed=12345):
    """use_space=False → 朴素贪心；True → 贪心 + 空间"""
    rng = np.random.default_rng(seed)

    def policy(env, obs):
        cands = _candidates(env)
        if not cands:
            return 0                      # 三个方向都会死，随便走吧

        if use_space:
            # 优先保住空间：走完至少还要剩「当前长度 + 余量」格
            roomy = [c for c in cands if c["space"] >= len(env.snake) + SPACE_MARGIN]
            if roomy:
                cands = roomy

        # 在候选里选离豆最近的；平局就选空间大的；再平局随机（避免固定路线死循环）
        best = min(c["dist"] for c in cands)
        cands = [c for c in cands if c["dist"] == best]
        best_space = max(c["space"] for c in cands)
        cands = [c for c in cands if c["space"] == best_space]
        return int(cands[int(rng.integers(len(cands)))]["a"])

    return policy


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("=" * 92)
    print(f"手写规则 AI · {N} 局（⚠️ 能看全盘，是【上界参照】不是公平对手）")
    print("=" * 92)
    print()
    print(f"  {'':<14}       │ ⚠️ 分数 = 豆子数 - 1")
    print()

    env = SnakeEnv()
    st_rand = run_episodes(lambda e, o: e.action_space.sample(), N, 0, env=env)
    print_report("随机策略", st_rand)
    print()

    st_naive = run_episodes(make_greedy(use_space=False), N, 0, env=env)
    print_report("朴素贪心", st_naive)
    print()

    st_space = run_episodes(make_greedy(use_space=True), N, 0, env=env)
    print_report("贪心+空间", st_space)
    env.close()

    print()
    print("=" * 92)
    print("结论")
    print("=" * 92)
    print(f"""
  ① 空间特征值多少：
        朴素贪心  平均豆子 {st_naive['beans_mean']:5.2f}
        贪心+空间 平均豆子 {st_space['beans_mean']:5.2f}
        → 差距 {st_space['beans_mean'] - st_naive['beans_mean']:+.2f} 个豆
        这就是状态里 [9:12] 那三维 flood fill 的动机。

  ② 环境是可解的：手写规则能吃到 {st_space['beans_mean']:.1f} 个，
        PPO 学不出来就【不是环境的问题】。

  ③ ⭐ MAX_HUNGER 的标定值：
        贪心+空间 平均每豆 {st_space['steps_per_bean']:.1f} 步
        → 建议 MAX_HUNGER ≈ 5~6 × {st_space['steps_per_bean']:.1f} = {5*st_space['steps_per_bean']:.0f} ~ {6*st_space['steps_per_bean']:.0f}
        （当前设的是 100）

  ④ PPO 的靶子：
        至少要打赢【朴素贪心】的 {st_naive['beans_mean']:.1f} 个豆，
        目标是逼近【贪心+空间】的 {st_space['beans_mean']:.1f} 个。
""")
