"""
贪吃蛇 · PPO

================================================================
【和前面三个例子的关系】

    `compute_gae` / `collect_trajectories` / `ppo_update` ——【一个字都没改】。
    换掉的只有 env。

    这就是「奖励是唯一能操控的旋钮」的另一半含义：
    算法部分是通用机器，你写的只是环境和奖励。

【从哪抄的，为什么】

    骨架抄 `lunarlander/ppo.py`，但【故意避开了 acrobot 版的 ret_scale】：

        acrobot:   critic_loss = MSE(...) / (returns.std()**2 + 1e-8)
        lunarlander: critic_loss = MSE(...)                    ← 用这版

    acrobot 那个「critic 尺度配平」是为 -500 量级的回报设计的。
    蛇的回报是 O(1)（吃到 N 个豆 → 回报 = N-1，std ≈ 0.3），
    除以 std² 等于把 critic loss 【放大 10 倍】；

    更糟的是开局：一批全是「10 步撞墙，-1」，returns.std() ≈ 0
    → 除以 1e-8 → critic 梯度爆炸。
    而它和 actor 共用主干 → 通过公共参数把 actor 也带崩（insights #19 / #26）。

    所以这里用 lunarlander 的 SepNet（actor / critic 各有主干）+ 不缩放。
    分开主干之后，谁大谁小都淹不到对方。

【训练时最该盯的一行】

    「本批最好」 = max(traj_scores) = 吃到的豆子数 - 1

    - 200 批过去还是 -1.0 → 一次都没吃到过 → 这是【探索问题】，别乱调学习率
    - 本批最好 > -1 但均分不涨 → 【信用分配问题】(+1 离前面 50 步太远) → 该上塑形了
    - 均分涨了但贪心评估不涨 → 【两把尺子】(insights #14)

【对照线（由 heuristic.py 量出）】

    随机策略      0.16 个豆      （评估分 -0.84）
    朴素贪心     17.84 个豆      （评估分 16.84）   ← SOLVE 线定在这附近
    贪心+空间    24.04 个豆      （评估分 23.04）

================================================================
"""

import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from snake_env import SnakeEnv, W, MAX_HUNGER


# ============================================================
# 设备 —— 所有张量创建都走 _T()，换设备只改 DEVICE 这一行
#
#   特征版的 MLP 太小，CPU 就够了；端到端的 CNN 在 CPU 上
#   一轮 400 批要 2~3 小时，换 MPS 是唯一现实的选择。
#   （e2e.py 里会 `import ppo; ppo.DEVICE = torch.device("mps")`）
# ============================================================
DEVICE = torch.device("cpu")


def _T(x, dtype=torch.float32):
    """建张量的唯一入口。⚠️ 用 as_tensor 而不是 FloatTensor ——
    FloatTensor 硬编码建在 CPU 上，模型搬到 MPS 之后会设备不匹配。"""
    return torch.as_tensor(np.asarray(x), dtype=dtype).to(DEVICE)


# ============================================================
# 网络：actor / critic 【各有主干】（SepNet）
# ============================================================
class PPOActorCritic(nn.Module):
    def __init__(self, state_dim=12, action_dim=3, hidden_size=128):
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.critic_body = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return (torch.softmax(self.actor(self.actor_body(x)), dim=-1),
                self.critic(self.critic_body(x)))

    def dist_and_value(self, x):
        probs, value = self.forward(x)
        return Categorical(probs), value


# ============================================================
# 采样（环境无关，逐字来自 lunarlander）
# ============================================================
def collect_one_trajectory(env, model):
    states, actions, rewards, old_log_probs, values = [], [], [], [], []

    state, _ = env.reset()
    while True:
        with torch.no_grad():
            dist, value = model.dist_and_value(_T(state).unsqueeze(0))
            action = dist.sample()

        states.append(state)
        actions.append(action.item())
        old_log_probs.append(dist.log_prob(action).item())
        values.append(value.squeeze().item())

        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)

        if terminated or truncated:
            break

    return states, actions, rewards, old_log_probs, values


def collect_trajectories(env, model, num_trajectories=32):
    all_states, all_actions, all_rewards, all_old_log_probs, all_values = [], [], [], [], []
    traj_lengths, traj_scores = [], []

    for _ in range(num_trajectories):
        s, a, r, olp, v = collect_one_trajectory(env, model)
        traj_lengths.append(len(r))
        traj_scores.append(sum(r))
        all_states += s
        all_actions += a
        all_rewards += r
        all_old_log_probs += olp
        all_values += v

    return (all_states, all_actions, all_rewards,
            all_old_log_probs, all_values, traj_lengths, traj_scores)


# ============================================================
# GAE（纯数组，环境无关）
# ============================================================
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    n = len(rewards)
    values_next = values[1:] + [0.0]
    deltas = [rewards[t] + gamma * values_next[t] - values[t] for t in range(n)]

    advantages, A = [], 0.0
    for t in reversed(range(n)):
        A = deltas[t] + gamma * lam * A
        advantages.insert(0, A)

    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return returns, advantages


def compute_gae_multi(rewards, values, traj_lengths, gamma=0.99, lam=0.95):
    all_returns, all_advantages = [], []
    start = 0
    for L in traj_lengths:
        ret_seg, adv_seg = compute_gae(rewards[start:start + L], values[start:start + L], gamma, lam)
        all_returns += ret_seg
        all_advantages += adv_seg
        start += L

    returns = _T(all_returns)
    advantages = _T(all_advantages)
    # ⚠️ 整批标准化一次，切 minibatch 前定死
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


# ============================================================
# PPO 更新（环境无关）
# ============================================================
def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages,
               clip_epsilon=0.2, epochs=10, minibatch_size=128,
               entropy_coef=0.01, critic_coef=0.5):
    n = len(states)
    states_t = _T(np.array(states))
    actions_t = _T(actions, torch.long)
    old_lp_t = _T(old_log_probs)

    clip_hits, total_seen, grad_steps = 0, 0, 0
    last_actor = last_critic = last_entropy = 0.0

    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch_size):
            mb = perm[start:start + minibatch_size]

            probs, values = model(states_t[mb])
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions_t[mb])
            ratio = torch.exp(new_log_probs - old_lp_t[mb])

            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages[mb]
            actor_loss = -torch.min(surr1, surr2).mean()

            entropy = dist.entropy().mean()
            critic_loss = nn.MSELoss()(values.squeeze(-1), returns[mb])

            loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                clip_hits += ((ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)).sum().item()
                total_seen += len(mb)
                last_actor = actor_loss.item()
                last_critic = critic_loss.item()
                last_entropy = entropy.item()
            grad_steps += 1

    return grad_steps, clip_hits / max(total_seen, 1), last_actor, last_critic, last_entropy


# ============================================================
# 评估：⭐ 贪心 + 固定种子（这才是能信的分数）
# ============================================================
def evaluate(model, env, episodes=20, seed=12345):
    results = []
    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        total = 0.0
        while True:
            with torch.no_grad():
                probs, _ = model(_T(state).unsqueeze(0))
                action = probs.argmax(dim=-1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        results.append(total)
    return results


def eval_detail(model, env, episodes=50, seed=777):
    """终测：豆子数 + 死因直方图（evaluate 只给 reward，这里要更细的）"""
    from collections import Counter
    beans, reasons, steps = [], Counter(), []
    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        while True:
            with torch.no_grad():
                probs, _ = model(_T(state).unsqueeze(0))
                action = probs.argmax(dim=-1).item()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        beans.append(info["food_eaten"])
        steps.append(info["steps"])
        reasons[info["end_reason"]] += 1
    return np.array(beans), np.array(steps), reasons


# ============================================================
# 给 watch.py / record.py 用的共用工具
#   （放在这里而不是各写一份：两个脚本本来就要 import 本文件的
#     PPOActorCritic，白捡一个共用模块，不用再多一个文件）
# ============================================================
DEFAULT_CKPT = "snake_both.pth"      # 目前最好的一版：18 维，终测 59.12 个豆


def load_for_view(path=None):
    """读 checkpoint → (env, model)。找不到文件直接退出并给出下一步。

    支持两种表示，看 checkpoint 里记的 obs_mode：

        features  9/12/15/18 维手工特征  →  SnakeEnv + PPOActorCritic
        grid      (4, 8, 8) 原始网格     →  GridObs(SnakeEnv) + GridActorCritic

    ⚠️ 必须按【模型自带的配置】建环境 —— 用错了不会报错，只会【静默玩得很烂】。
       特征版的 15 维还有歧义：+2步前瞻 和 +蛇尾可达 的特征顺序不同，
       光看维度猜不出来。网格版更没法猜（(4,8,8) 和 (18,) 是两种完全不同的输入）。
       所以配置必须跟权重存一起。
    """
    path = path or DEFAULT_CKPT
    try:
        ckpt = torch.load(path, weights_only=True)
    except FileNotFoundError:
        sys.exit(f"❌ 找不到 {path}\n   先在这个目录下跑：uv run python ppo.py")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:      # 新格式：带配置
        state_dict = ckpt["state_dict"]
        env_kw = ckpt.get("env_kw", {})
        obs_mode = ckpt.get("obs_mode", "features")
        arch = ckpt.get("arch", "mlp")
    else:                                                     # 旧格式：裸 state_dict
        state_dict, env_kw, obs_mode, arch = ckpt, {}, "features", "mlp"
        print("⚠️ 旧格式 checkpoint（没记录环境配置），按 12 维默认环境跑")

    if obs_mode == "grid":
        # 懒加载：只有看网格模型时才需要这个模块，也顺便避开 import 环
        #   （依赖方向是 ppo → e2e_grid → snake_env，没有环）
        from e2e_grid import GridObs, GridActorCritic, N_CH
        env = GridObs(SnakeEnv(**env_kw))
        model = GridActorCritic(obs_shape=(N_CH, env.W, env.W), action_dim=3,
                                hidden_size=128, arch=arch)
    else:
        env = SnakeEnv(**env_kw)
        model = PPOActorCritic(state_dim=env.obs_dim, action_dim=3, hidden_size=128)

    # ⚠️ 权重可能是用别的设备训的（e2e.py 支持换设备），
    #    看的时候一律搬回 CPU —— 看是交互式的，慢一点无所谓，别搞出设备不匹配。
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()
    return env, model


# ============================================================
# 主训练循环
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只跑 2 批，确认不炸")
    ap.add_argument("--budget", type=int, default=400, help="跑多少批")
    ap.add_argument("--solve", type=float, default=12.0, help="达标线（评估分 = 豆子-1）")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--out", default="snake_best.pth")
    # ---- 状态特征开关（做消融用）----
    ap.add_argument("--deep", action="store_true", help="加 [2 步前瞻] 那 3 维")
    ap.add_argument("--tail", action="store_true", help="加 [蛇尾可达] 那 3 维")
    ap.add_argument("--no-space", action="store_true", help="去掉 flood fill 那 3 维")
    args = ap.parse_args()

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
                               #    本次实测：线设 12，但批 30 已经跑到 23.4 了
                               #    要训到收敛就传 --solve 100（不可达 = 不早停）
    SOLVE_STREAK = 3

    torch.manual_seed(SEED)

    ENV_KW = dict(use_space=not args.no_space, use_deep=args.deep, use_tail=args.tail)
    env = SnakeEnv(**ENV_KW)
    env.reset(seed=SEED)
    eval_env = SnakeEnv(**ENV_KW)

    print("=" * 104)
    print("贪吃蛇 · PPO" + ("  【烟测：只跑 2 批】" if SMOKE else ""))
    print("=" * 104)
    print(f"每批 {NUM_TRAJECTORIES} 局 | minibatch={MINIBATCH_SIZE} epochs={UPDATE_EPOCHS} "
          f"lr={LR} gamma={GAMMA} clip={CLIP_EPSILON} entropy={ENTROPY_COEF} critic={CRITIC_COEF}")
    print(f"网格 {W}×{W} | 状态 {env.obs_dim} 维 "
          f"(flood fill={int(not args.no_space)} 2步前瞻={int(args.deep)} 蛇尾可达={int(args.tail)}) "
          f"| 动作 3（左转/直行/右转）| MAX_HUNGER={MAX_HUNGER}")
    print(f"⚠️ 分数 = 豆子数 − 1（每局必然以 −1 结束）")
    print(f"对照线：随机 0.16 豆 | 朴素贪心 17.8 豆 | 贪心+空间 24.0 豆")
    print(f"达标线：贪心评估 ≥ {SOLVE_SCORE:+.0f} 连续 {SOLVE_STREAK} 次（≈ 13 个豆）")
    print("=" * 104, flush=True)

    model = PPOActorCritic(state_dim=env.obs_dim, action_dim=3, hidden_size=128)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_eval = -float("inf")
    best_state = None
    streak = 0
    total_steps = 0
    t0 = time.time()
    ever_ate = False

    print(f"{'批次':>5} | {'步数':>7} | {'训练窗口':>9} | {'本批最好':>8} | {'贪心评估':>9} | "
          f"{'历史最好':>9} | {'actor':>7} {'critic':>8} {'熵':>6} | {'clip':>6} | 达标")
    print("-" * 104)

    for it in range(BUDGET):
        (states, actions, rewards, old_log_probs,
         values, traj_lengths, traj_scores) = collect_trajectories(
            env, model, NUM_TRAJECTORIES)

        returns, advantages = compute_gae_multi(rewards, values, traj_lengths, GAMMA, LAM)

        _, clip_rate, a_loss, c_loss, ent = ppo_update(
            model, optimizer, states, actions, old_log_probs, returns, advantages,
            clip_epsilon=CLIP_EPSILON, epochs=UPDATE_EPOCHS,
            minibatch_size=MINIBATCH_SIZE, entropy_coef=ENTROPY_COEF, critic_coef=CRITIC_COEF)

        avg = float(np.mean(traj_scores))
        best = float(np.max(traj_scores))          # ⭐ 全文件最重要的一列
        total_steps += len(states)

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

            print(f"{it+1:>5} | {total_steps:>7} | {avg:>9.2f} | {best:>8.1f} | {ev:>9.2f} | "
                  f"{best_eval:>9.2f} | {a_loss:>7.3f} {c_loss:>8.3f} {ent:>6.3f} | "
                  f"{clip_rate*100:>5.1f}% | {streak}/{SOLVE_STREAK}", flush=True)

            if streak >= SOLVE_STREAK:
                print(f"\n🎉 连续 {streak} 次贪心评估 ≥ {SOLVE_SCORE:+.0f}，第 {it+1} 批早停")
                break

    env.close()
    eval_env.close()

    if SMOKE:
        print("\n✅ 烟测通过（loss 有限、本批最好有值、无 NaN）")
        sys.exit(0)

    if best_state is not None:
        model.load_state_dict(best_state)
        # ⚠️ 把【环境配置】和权重存在一起 —— 否则 watch.py 不知道该建几维的环境。
        #    15 维是有歧义的（+2步前瞻 还是 +蛇尾可达，两者的特征顺序不同），
        #    光看维度猜不出来，必须显式记下来。
        torch.save({"state_dict": best_state, "env_kw": ENV_KW, "obs_dim": env.obs_dim,
                    "eval_score": best_eval}, args.out)
        print(f"\n✅ 已保存 {args.out}（评估最好 {best_eval:+.2f}，{env.obs_dim} 维）")

    # ---------------- 终测 ----------------
    print("\n" + "=" * 104)
    print("终测：50 局，贪心，固定种子")
    print("=" * 104)
    test_env = SnakeEnv(**ENV_KW)
    beans, steps, reasons = eval_detail(model, test_env, episodes=50, seed=777)
    test_env.close()

    REASON_CN = {"wall": "撞墙", "self": "咬自己", "starve": "饿死", "win": "通关"}
    print(f"  平均豆子 {beans.mean():6.2f} │ 中位 {np.median(beans):4.0f} │ "
          f"最好 {beans.max():3.0f} │ 最差 {beans.min():3.0f}")
    print(f"  平均步数 {steps.mean():6.1f} │ 每豆步数 {steps.sum()/max(beans.sum(),1):6.2f}")
    print(f"  死因 " + "  ".join(f"{REASON_CN.get(k,k)} {v}" for k, v in reasons.most_common()))
    print()
    print("  ┌─ 三条线放一起 ──────────────────────────────────────────────")
    print(f"  │ 随机策略        {0.16:6.2f} 个豆")
    print(f"  │ 朴素贪心        {17.84:6.2f} 个豆")
    print(f"  │ PPO（本次）     {beans.mean():6.2f} 个豆   {'✅ 打赢了' if beans.mean() > 17.84 else '⚠️ 还没打赢'}")
    print(f"  │ 贪心+空间       {24.04:6.2f} 个豆   （上界参照）")
    print("  └───────────────────────────────────────────────────────────")
    print(f"\n训练用时 {time.time()-t0:.1f}s | 总步数 {total_steps}")
