"""
贪吃蛇 · 训练（全仓库唯一的训练循环）

================================================================
【为什么只有一个 train.py】

    以前是两个：ppo.py 的 __main__（143 行）和 e2e.py 的 __main__（155 行）。
    两份的常量【逐个相同】—— 32 局/批、400 批、minibatch 128、epochs 10、
    γ=0.99、λ=0.95、lr=3e-4、clip=0.2、entropy=0.01、critic=0.5、
    eval every 10、eval 20 局、streak 3。

    加一个模型就要再抄一份，而且抄完两份会开始漂：改了一边的评估口径，
    另一边的分数就悄悄不可比了。

    现在：换模型只换 --model，训练循环一个字不改。

【跑法】

    # ① 先验特征版（16 秒的快速版，用来确认整条链路通）
    uv run python train.py --model feature --out snake_best.pth

    # ② 先验特征版，18 维（文章里 59.12 那版，约 39 分钟）
    uv run python train.py --model feature --deep --tail --out snake_both.pth

    # ③ 端到端 CNN（文章里 50.82 那版，CPU 约 3.3 小时）
    uv run python train.py --model cnn --arch cnn --out snake_e2e_cnn.pth

    # ④ 对照组：同一个网格输入，主干和特征版一样（回答"是没特征不行，还是没卷积不行"）
    uv run python train.py --model cnn --arch mlp --out snake_e2e_mlp.pth

    uv run python train.py --model cnn --smoke          # 2 批，确认不炸

【特征版的三个开关 = 消融实验】

    不给参数        12 维（9 基础 + flood fill）
    --no-space       9 维（只有基础）
    --deep           +3 维 2 步前瞻
    --tail           +3 维 蛇尾可达
    三个都给         18 维  ← 最强

【MPS 在这件事上比 CPU 慢（反直觉，但量出来的）】

       arch   设备    每批     400 批推算
       mlp    cpu    0.98s       6.5 分     ← 用这个
       mlp    mps    7.80s      52.0 分
       cnn    cpu   25.05s     167.0 分
       cnn    mps   14.54s      96.9 分   ← 实测全程 3.3 小时

       原因：一批里有 1280 次【极小】的梯度更新（minibatch=128，输入才 8×8×4），
             每次都是独立的 kernel 启动。GPU 的启动开销吃掉了并行收益，
             而 CPU 的 BLAS 在这种小矩阵上反而更划算。

       📌 教训：微基准（重复同一个前向）会骗人。
         真结论只能从【真实的训练内循环】里量。

【训练时最该盯的一行】

    「本批最好」 = max(traj_scores) = 吃到的豆子数 - 1

    - 200 批过去还是 -1.0 → 一次都没吃到过 → 这是【探索问题】，别乱调学习率
    - 本批最好 > -1 但均分不涨 → 【信用分配问题】(+1 离前面 50 步太远) → 该上塑形了
    - 均分涨了但贪心评估不涨 → 【两把尺子】(insights #14)

【对照线（由 heuristic.py / random_baseline.py 量出）】

    随机策略       0.16 个豆
    朴素贪心      17.84 个豆
    贪心+空间     24.04 个豆
    18 维特征版   59.12 个豆（满分 61）   ← 目标

    ⚠️ 默认 --solve 100（不可达 = 不早停），保证跑到预算用满，
       不然几条腿的训练量对不齐，没法比。

================================================================
"""

import argparse
import sys
import time

import numpy as np
import torch
import torch.optim as optim

import algo
import models

REASON_CN = {"wall": "撞墙", "self": "咬自己", "starve": "饿死", "win": "通关"}

# 已量出来的几条参照线（终测 50 局，贪心，固定种子）
REFERENCE = [
    ("随机策略", 0.16, "random_baseline.py"),
    ("朴素贪心", 17.84, "heuristic.py"),
    ("贪心+空间（手写上界）", 24.04, "heuristic.py"),
    ("18 维先验特征 + MLP", 59.12, "snake_both.pth"),
]


def build_cfg(args):
    """命令行 → 注册表认识的 cfg。"""
    if args.model == "feature":
        env_kw = dict(use_space=not args.no_space, use_deep=args.deep, use_tail=args.tail)
        return {"model": "feature", "env_kw": env_kw}
    return {"model": "cnn", "arch": args.arch, "env_kw": {}}


def main():
    ap = argparse.ArgumentParser(
        description="贪吃蛇 PPO 训练。换模型只换 --model，训练循环一个字不改。")
    ap.add_argument("--model", choices=sorted(models.REGISTRY), default="feature",
                    help="用哪个模型（models/ 里的注册名）")
    ap.add_argument("--arch", choices=["mlp", "cnn"], default="cnn",
                    help="只在 --model cnn 时有效：mlp = 和特征版同架构的对照组")
    # ---- 特征版的消融开关 ----
    ap.add_argument("--deep", action="store_true", help="feature：加 [2 步前瞻] 那 3 维")
    ap.add_argument("--tail", action="store_true", help="feature：加 [蛇尾可达] 那 3 维")
    ap.add_argument("--no-space", action="store_true", help="feature：去掉 flood fill 那 3 维")
    # ---- 训练 ----
    ap.add_argument("--smoke", action="store_true", help="只跑 2 批，确认不炸")
    ap.add_argument("--budget", type=int, default=400, help="跑多少批")
    ap.add_argument("--solve", type=float, default=100.0,
                    help="达标线（评估分 = 豆子-1）。默认 100 = 不可达 = 不早停")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu",
                    help="MPS 只在 CNN 上更快（约 1.7×），MLP 上反而慢 8×，见文件头")
    args = ap.parse_args()

    # ⚠️ 换设备要在【建模型之前】—— 模型参数是按当时的设备建的
    algo.set_device(args.device)

    cfg = build_cfg(args)
    OUT = args.out or (f"snake_e2e_{args.arch}.pth" if args.model == "cnn" else "snake_best.pth")

    SMOKE = args.smoke
    SEED = 0
    NUM_TRAJECTORIES = 32      # 蛇的回合短（随机策略才 13 步），10 局不够一批
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
    SOLVE_SCORE = args.solve    # ⚠️ 设低了会在【上升期】就被早停掐断
    SOLVE_STREAK = 3

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env, model = models.make(cfg)
    eval_env, _ = models.make(cfg)
    cfg["obs_dim"] = getattr(env, "obs_dim", None)

    print("=" * 104)
    print(f"贪吃蛇 · PPO · {cfg['model']}" + ("  【烟测：只跑 2 批】" if SMOKE else ""))
    print("=" * 104)
    print(f"模型      : {models.describe(cfg)}")
    _shape = tuple(env.observation_space.shape)
    print(f"输入      : {_shape} ，{int(np.prod(_shape))} 个数")
    print(f"网络      : {model.num_params()/1000:.1f}k 参数")
    print(f"设备      : {algo.DEVICE}")
    print(f"每批 {NUM_TRAJECTORIES} 局 | minibatch={MINIBATCH_SIZE} epochs={UPDATE_EPOCHS} "
          f"lr={LR} gamma={GAMMA} clip={CLIP_EPSILON} entropy={ENTROPY_COEF} critic={CRITIC_COEF}")
    print(f"奖励      : 吃到 +1 / 撞死 -1 / 饿死 -1 / 其他 0（所有模型都一样）")
    print(f"⚠️ 分数 = 豆子数 − 1（每局必然以 −1 结束）")
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
         values, traj_lengths, traj_scores) = algo.collect_trajectories(
            env, model, NUM_TRAJECTORIES)

        returns, advantages = algo.compute_gae_multi(rewards, values, traj_lengths, GAMMA, LAM)

        _, clip_rate, a_loss, c_loss, ent = algo.ppo_update(
            model, optimizer, states, actions, old_log_probs, returns, advantages,
            clip_epsilon=CLIP_EPSILON, epochs=UPDATE_EPOCHS,
            minibatch_size=MINIBATCH_SIZE, entropy_coef=ENTROPY_COEF, critic_coef=CRITIC_COEF)

        avg = float(np.mean(traj_scores))
        best = float(np.max(traj_scores))          # ⭐ 全文件最重要的一列
        total_steps += len(states)
        bsec = time.time() - tb

        if best > -1.0 and not ever_ate:
            ever_ate = True
            print(f"\n  🍎 第 {it+1} 批：【第一次吃到豆子】 本批最好 {best:+.1f}"
                  f"（= {int(best)+1} 个豆）\n", flush=True)

        if (it + 1) % EVAL_EVERY == 0 or SMOKE:
            results = algo.evaluate(model, eval_env, EVAL_EPISODES, seed=12345)
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
        # ⚠️ 把【表示方式】和权重存一起 —— 否则看的时候不知道该建哪种环境。
        #    (18,) 和 (4,8,8) 光看 state_dict 的键名分不出来。
        #    写什么由模型自己定（models/xxx.py 的 save_meta）—— 这里不硬编码。
        torch.save({**models.save_meta(cfg),
                    "state_dict": {k: v.cpu() for k, v in best_state.items()},
                    "eval_score": best_eval,
                    "train": {"budget": BUDGET, "batches": it + 1,
                              "steps": total_steps, "seconds": round(time.time() - t0, 1),
                              "device": str(algo.DEVICE)}}, OUT)
        print(f"\n✅ 已保存 {OUT}（评估最好 {best_eval:+.2f}）")

    # ---------------- 终测 ----------------
    print("\n" + "=" * 104)
    print("终测：50 局，贪心，固定种子")
    print("=" * 104)
    test_env, _ = models.make(cfg)
    beans, steps, reasons = algo.eval_detail(model, test_env, episodes=50, seed=777)
    test_env.close()

    print(f"  平均豆子 {beans.mean():6.2f} │ 中位 {np.median(beans):4.0f} │ "
          f"最好 {beans.max():3.0f} │ 最差 {beans.min():3.0f} │ "
          f"满分率 {(beans >= test_env.W * test_env.W - 3).mean()*100:.0f}%")
    print(f"  平均步数 {steps.mean():6.1f} │ 每豆步数 {steps.sum()/max(beans.sum(),1):6.2f}")
    print(f"  死因 " + "  ".join(f"{REASON_CN.get(k,k)} {v}" for k, v in reasons.most_common()))
    print()
    print("  ┌─ 和已有的几条线放一起 ──────────────────────────────────────")
    for label, val, src in REFERENCE:
        print(f"  │ {label:<22} {val:6.2f} 个豆   ({src})")
    win = beans.mean() > REFERENCE[1][1]
    print(f"  │ {'本次 ' + models.describe(cfg)[:20]:<22} {beans.mean():6.2f} 个豆   "
          f"{'✅ 打赢了朴素贪心' if win else '⚠️ 还没打赢朴素贪心'}")
    print("  └───────────────────────────────────────────────────────────")
    print(f"\n训练用时 {time.time()-t0:.1f}s | 总步数 {total_steps}")


if __name__ == "__main__":
    main()
