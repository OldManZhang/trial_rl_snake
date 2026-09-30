"""
端到端贪吃蛇 · 训练

================================================================
【和 ppo.py 什么关系】

    ppo.py 训的是【18 维手工特征】，这里训的是【8×8 原始网格】。

    算法部分【直接 import，一行没改】：

        from ppo import collect_trajectories, compute_gae_multi, ppo_update, eval_detail

    这次连【奖励】都没动 —— 和特征版完全同一套：
        吃到豆 +1 / 撞死 -1 / 饿死 -1 / 其他 0

    换掉的只有一样东西：网络看到的输入长什么样。

        features : (12,) 或 (15,) 或 (18,)   人用 BFS 算好的特征
        grid     : (4, 8, 8)                 原始棋盘，头永远朝上

    ⚠️ snake_env.py 一行没改 —— 表示层整个在 e2e_grid.py 里，
       游戏规则是冻结的（53 条自测仍然有效），对照才干净。

【要回答的问题】

    把 BFS 的结果拿走，只给原始棋盘，网络能不能自己算出来？

    特征版里值钱的那 6 维（flood fill / 2 步前瞻 / 蛇尾可达）
    全是人【迭代计算】出来的。BFS 的原料（整张棋盘）还在网格里，
    但 CNN 是前馈的、BFS 是迭代的 —— 想用卷积模拟 N 步迭代，就得堆 N 层。

【两条基线，两种问法】

    --arch mlp   主干和特征版【完全同架构】（2×128 Tanh），只换输入
                 → 回答「把特征换成原始网格，差多少」
    --arch cnn   加卷积
                 → 再回答「卷积能不能补回来」

【训练时最该盯的一行】

    「本批最好」 = 吃到的豆子数 - 1

    和 ppo.py 一样，看这一列比看均分有用 —— 稀疏奖励下，
    「一次都没吃到」和「吃到了但学不动」是两种完全不同的病。

【跑多久 · 以及一个 MPS 的反直觉结论】

    uv run python e2e.py --smoke                                  # 2 批，确认不炸
    uv run python e2e.py --arch mlp --out snake_e2e_mlp.pth       # 400 批，约 7 分钟
    uv run python e2e.py --arch cnn --out snake_e2e_cnn.pth       # 400 批，实测 3.3 小时

    ⚠️ 【MPS 在这件事上比 CPU 慢】—— 反直觉，但量出来的：

       arch   设备    每批     400 批推算
       mlp    cpu    0.98s       6.5 分     ← 用这个
       mlp    mps    7.80s      52.0 分
       cnn    cpu   25.05s     167.0 分
       cnn    mps   14.54s      96.9 分   ← 实测全程 3.3 小时

       原因：一批里有 1280 次【极小】的梯度更新（minibatch=128，输入才 8×8×4），
             每次都是独立的 kernel 启动。GPU 的启动开销吃掉了并行收益，
             而 CPU 的 BLAS 在这种小矩阵上反而更划算。

       顺带验了正确性：同一个种子同一批数据，CPU 和 MPS 的 loss
       相对差 0.00%（mlp）/ 0.27%（cnn）—— 两边算得一样，只是快慢不同。

       📌 教训：微基准（重复同一个前向）会骗人。
         真结论只能从【真实的训练内循环】里量。

    ⚠️ 默认 --solve 100（不可达 = 不早停），保证跑到预算用满，
       不然四条腿的训练量对不齐，没法比。

【怎么跟特征版比】

    18 维手工特征 + MLP   59.12 个豆（满分 61）   ← snake_both.pth
    (4,8,8) 网格  + MLP    ?
    (4,8,8) 网格  + CNN    ?

================================================================
"""

import argparse
import sys
import time

import numpy as np
import torch
import torch.optim as optim

from snake_env import SnakeEnv, W, MAX_HUNGER
from e2e_grid import GridObs, GridActorCritic, N_CH

# ⭐ 算法部分：直接复用，没有副本
import ppo
from ppo import (collect_trajectories, compute_gae_multi, ppo_update,
                 evaluate, eval_detail)


# ============================================================
# 建环境
# ============================================================
def make_env_kw():
    """内层 SnakeEnv 的配置。

    ⚠️ 三个特征开关【全部关掉】—— 不是为了消融，纯为了快。

       内层照常算完 12 维特征（含 flood fill / 2 步前瞻 / 蛇尾可达，
       一大把 BFS），然后被 GridObs 整个丢掉换成网格。
       全关掉之后 obs_dim=9，一次 BFS 都不跑，白省。

       反正 wrapper 只看 env.snake / env.food / env.direction / env.hunger 这几个属性，
       根本读不到那 9 个数。
    """
    return dict(use_space=False, use_deep=False, use_tail=False)


def build(arch):
    env = GridObs(SnakeEnv(**make_env_kw()))
    model = GridActorCritic(obs_shape=(N_CH, W, W), action_dim=3,
                            hidden_size=128, arch=arch).to(ppo.DEVICE)
    return env, model


# ============================================================
# 主训练循环
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "cnn"], default="mlp",
                    help="mlp = 和特征版同架构只换输入；cnn = 再加卷积")
    ap.add_argument("--smoke", action="store_true", help="只跑 2 批，确认不炸")
    ap.add_argument("--budget", type=int, default=400, help="跑多少批")
    ap.add_argument("--solve", type=float, default=100.0,
                    help="达标线（评估分 = 豆子-1）。默认 100 = 不可达 = 不早停")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu",
                    help="MPS 只在 CNN 上更快（约 1.7×），MLP 上反而慢 8×，见文件头")
    args = ap.parse_args()

    # ⚠️ 换设备要在【建模型之前】设好 —— ppo 里所有张量创建都读这个
    ppo.DEVICE = torch.device(args.device)

    SMOKE = args.smoke
    OUT = args.out or f"snake_e2e_{args.arch}.pth"

    SEED = 0
    NUM_TRAJECTORIES = 32
    BUDGET = 2 if SMOKE else args.budget
    MINIBATCH_SIZE = 128
    UPDATE_EPOCHS = 10
    GAMMA = 0.99               # 回合上限 ≈ MAX_HUNGER = 100，1/(1-γ)=100 刚好匹配
    LAM = 0.95
    LR = 3e-4
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    CRITIC_COEF = 0.5
    EVAL_EVERY = args.eval_every
    EVAL_EPISODES = 20
    SOLVE_SCORE = args.solve
    SOLVE_STREAK = 3

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env, model = build(args.arch)
    eval_env, _ = build(args.arch)

    ENV_KW = make_env_kw()
    n_par = model.num_params()

    print("=" * 104)
    print(f"端到端贪吃蛇 · PPO · arch={args.arch}" + ("  【烟测：只跑 2 批】" if SMOKE else ""))
    print("=" * 104)
    print(f"输入      : 原始网格 (4, {W}, {W}) = {N_CH * W * W} 个数，头永远朝上")
    print(f"            ch0 蛇身 / ch1 蛇头 / ch2 食物 / ch3 饥饿常数平面")
    print(f"网络      : {args.arch}，{n_par/1000:.1f}k 参数")
    print(f"每批 {NUM_TRAJECTORIES} 局 | minibatch={MINIBATCH_SIZE} epochs={UPDATE_EPOCHS} "
          f"lr={LR} gamma={GAMMA} clip={CLIP_EPSILON} entropy={ENTROPY_COEF} critic={CRITIC_COEF}")
    print(f"网格 {W}×{W} | 动作 3（左转/直行/右转）| MAX_HUNGER={MAX_HUNGER}")
    print(f"奖励      : 和特征版【完全一样】（吃到 +1 / 撞死 -1 / 饿死 -1 / 其他 0）")
    print(f"⚠️ 分数 = 豆子数 − 1（每局必然以 −1 结束）")
    print(f"对照线    : 特征版 18 维 MLP 终测 59.12 个豆（满分 {W*W-3}）")
    print(f"达标线    : 贪心评估 ≥ {SOLVE_SCORE:+.0f} 连续 {SOLVE_STREAK} 次（默认不可达 = 不早停）")
    print("=" * 104, flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_eval = -float("inf")
    best_state = None
    streak = 0
    total_steps = 0
    t0 = time.time()
    ever_ate = False

    print(f"{'批次':>5} | {'步数':>8} | {'秒':>6} | {'训练窗口':>9} | {'本批最好':>8} | "
          f"{'贪心评估':>9} | {'历史最好':>9} | {'actor':>7} {'critic':>8} {'熵':>6} | {'clip':>6}")
    print("-" * 104)

    for it in range(BUDGET):
        tb = time.time()
        (states, actions, rewards, old_log_probs,
         values, traj_lengths, traj_scores) = collect_trajectories(
            env, model, NUM_TRAJECTORIES)

        returns, advantages = compute_gae_multi(rewards, values, traj_lengths, GAMMA, LAM)

        _, clip_rate, a_loss, c_loss, ent = ppo_update(
            model, optimizer, states, actions, old_log_probs, returns, advantages,
            clip_epsilon=CLIP_EPSILON, epochs=UPDATE_EPOCHS,
            minibatch_size=MINIBATCH_SIZE, entropy_coef=ENTROPY_COEF, critic_coef=CRITIC_COEF)

        avg = float(np.mean(traj_scores))
        best = float(np.max(traj_scores))
        total_steps += len(states)
        bsec = time.time() - tb

        if best > -1.0 and not ever_ate:
            ever_ate = True
            print(f"\n  🍎 第 {it+1} 批：【第一次吃到豆子】 本批最好 {best:+.1f}"
                  f"（= {int(best)+1} 个豆）\n", flush=True)

        if (it + 1) % EVAL_EVERY == 0 or SMOKE:
            results = evaluate(model, eval_env, EVAL_EPISODES, seed=12345)
            ev = float(np.mean(results))

            if ev > best_eval:
                best_eval = ev
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            streak = streak + 1 if ev >= SOLVE_SCORE else 0

            print(f"{it+1:>5} | {total_steps:>8} | {bsec:>6.1f} | {avg:>9.2f} | {best:>8.1f} | "
                  f"{ev:>9.2f} | {best_eval:>9.2f} | {a_loss:>7.3f} {c_loss:>8.3f} {ent:>6.3f} | "
                  f"{clip_rate*100:>5.1f}%", flush=True)

            if streak >= SOLVE_STREAK:
                print(f"\n🎉 连续 {streak} 次贪心评估 ≥ {SOLVE_SCORE:+.0f}，第 {it+1} 批早停")
                break

    env.close()
    eval_env.close()

    if SMOKE:
        print(f"\n✅ 烟测通过（loss 有限、有值、无 NaN）| 每批约 {(time.time()-t0)/BUDGET:.1f} 秒")
        sys.exit(0)

    if best_state is not None:
        model.load_state_dict(best_state)
        # ⚠️ 把【表示方式】和权重存一起 —— 否则 watch.py 不知道该建网格环境还是特征环境。
        #    (4,8,8) 和 (18,) 光看 state_dict 的键名分不出来。
        torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()},
                    "obs_mode": "grid",
                    "arch": args.arch,
                    "obs_shape": (N_CH, W, W),
                    "env_kw": ENV_KW,
                    "eval_score": best_eval}, OUT)
        print(f"\n✅ 已保存 {OUT}（评估最好 {best_eval:+.2f}，{args.arch}，网格输入）")

    # ---------------- 终测 ----------------
    print("\n" + "=" * 104)
    print("终测：50 局，贪心，固定种子")
    print("=" * 104)
    test_env, _ = build(args.arch)
    beans, steps, reasons = eval_detail(model, test_env, episodes=50, seed=777)
    test_env.close()

    REASON_CN = {"wall": "撞墙", "self": "咬自己", "starve": "饿死", "win": "通关"}
    print(f"  平均豆子 {beans.mean():6.2f} │ 中位 {np.median(beans):4.0f} │ "
          f"最好 {beans.max():3.0f} │ 最差 {beans.min():3.0f}")
    print(f"  平均步数 {steps.mean():6.1f} │ 每豆步数 {steps.sum()/max(beans.sum(),1):6.2f}")
    print(f"  死因 " + "  ".join(f"{REASON_CN.get(k,k)} {v}" for k, v in reasons.most_common()))
    print()
    print("  ┌─ 三条线放一起 ──────────────────────────────────────────────")
    print(f"  │ 随机策略                    {0.16:6.2f} 个豆")
    print(f"  │ 朴素贪心                    {17.84:6.2f} 个豆")
    print(f"  │ 贪心+空间（手写上界）        {24.04:6.2f} 个豆")
    print(f"  │ 18 维手工特征 + MLP         {59.12:6.2f} 个豆   ← 要追的就是这条")
    print(f"  │ 端到端 {args.arch:<3} 网格输入       {beans.mean():6.2f} 个豆   "
          f"{'✅ 追上了' if beans.mean() > 59.12 else f'⚠️ 差 {59.12-beans.mean():.2f}'}")
    print("  └───────────────────────────────────────────────────────────")
    print(f"\n训练用时 {time.time()-t0:.1f}s | 总步数 {total_steps}")
