"""
贪吃蛇 · PPO 算法

================================================================
【这个文件里【一条蛇都没有】】

    采样 / GAE / PPO 更新 / 评估 —— 四样东西只依赖两个契约：

        env     gymnasium 的 reset() / step() 契约
        model   两条： model(x) → (probs, value)
                       model.dist_and_value(x) → (Categorical, value)

    换环境能用，换网络能用。这就是「算法是通用机器，你写的只是环境和奖励」
    那一句的落地 —— 只不过这次连"你写的"都被拆成了三个文件：

        snake_env.py        游戏规则
        models/feature.py   一种表示 + 一种网络
        models/cnn.py       另一种表示 + 另一种网络
        algo.py             ← 你（几乎）不用动这个

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
    （SepNet 是【网络】的选择，所以它在 models/*.py 里，不在这儿。）

【训练时最该盯的一行】

    「本批最好」 = max(traj_scores) = 吃到的豆子数 - 1

    - 200 批过去还是 -1.0 → 一次都没吃到过 → 这是【探索问题】，别乱调学习率
    - 本批最好 > -1 但均分不涨 → 【信用分配问题】(+1 离前面 50 步太远) → 该上塑形了
    - 均分涨了但贪心评估不涨 → 【两把尺子】(insights #14)

【对照线（由 heuristic.py 量出）】

    随机策略      0.16 个豆      （评估分 -0.84）
    朴素贪心     17.84 个豆      （评估分 16.84）
    贪心+空间    24.04 个豆      （评估分 23.04）

================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ============================================================
# 设备 —— 所有张量创建都走 _T()，换设备只调 set_device()
#
#   特征版的 MLP 太小，CPU 就够了；端到端的 CNN 在 CPU 上
#   一轮 400 批要 2~3 小时，换 MPS 是唯一现实的选择。
#
# ⚠️ 以前是 `import ppo; ppo.DEVICE = torch.device("mps")` 这种猴补丁 ——
#    改完忘了哪一行生效过，排查起来很痛苦。现在收成一个函数。
# ============================================================
DEVICE = torch.device("cpu")


def set_device(name):
    """'cpu' / 'mps'。必须在【建模型之前】调 —— 模型参数是按当时的设备建的。"""
    global DEVICE
    DEVICE = torch.device(name)
    return DEVICE


def _T(x, dtype=torch.float32):
    """建张量的唯一入口。⚠️ 用 as_tensor 而不是 FloatTensor ——
    FloatTensor 硬编码建在 CPU 上，模型搬到 MPS 之后会设备不匹配。"""
    return torch.as_tensor(np.asarray(x), dtype=dtype).to(DEVICE)


# ============================================================
# 采样（环境无关）
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
# 自测
# ============================================================
if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium import spaces

    ok = 0

    def check(name, cond, extra=""):
        global ok
        assert cond, f"❌ {name}  {extra}"
        ok += 1
        print(f"  ✅ {name}{('  ' + extra) if extra else ''}")

    print("=" * 78)
    print("algo 自测 · 算法层（用一个假环境，跟蛇无关）")
    print("=" * 78)

    # ---- 用【最笨的玩具环境】测算法层 ----
    # ⭐ 这是这个文件唯一正确的测法：如果 algo 真的与环境无关，
    #    那它就该能在一个跟蛇毫无关系的小环境上跑起来。
    class Toy(gym.Env):
        """往右走 N 步就赢。状态 2 维，动作 3 个 —— 故意和蛇不一样。"""

        def __init__(self):
            self.observation_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
            self.action_space = spaces.Discrete(3)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return np.array([0.0, 1.0], dtype=np.float32), {}

        def step(self, a):
            self.t += 1
            good = (a == 1)
            r = 1.0 if good else -0.5
            done = self.t >= 8
            return np.array([self.t / 8, 1.0], dtype=np.float32), r, done, False, {
                "food_eaten": self.t if good else 0, "steps": self.t, "end_reason": "wall"}

    class ToyNet(nn.Module):
        def __init__(self, state_dim=2, action_dim=3, hidden=16):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh())
            self.actor = nn.Linear(hidden, action_dim)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.body(x)
            return torch.softmax(self.actor(h), dim=-1), self.critic(h)

        def dist_and_value(self, x):
            probs, value = self.forward(x)
            return Categorical(probs), value

    print("\n[1] 采样")
    env, net = Toy(), ToyNet()
    s, a, r, olp, v = collect_one_trajectory(env, net)
    check("单局五元组长度一致", len(s) == len(a) == len(r) == len(olp) == len(v), f"{len(r)} 步")
    check("动作都在 {0,1,2}", all(x in (0, 1, 2) for x in a))
    check("log_prob 是有限数", all(np.isfinite(x) for x in olp))

    out = collect_trajectories(env, net, 4)
    check("批量采样返回 7 样东西", len(out) == 7)
    check("traj_lengths 求和 == 总步数", sum(out[5]) == len(out[2]), f"{sum(out[5])}")

    print("\n[2] GAE")
    rets, advs = compute_gae([1.0, 1.0, -1.0], [0.5, 0.5, 0.5])
    # 手算：delta = r + γV' - V；末步 V'=0
    g = 0.99
    d2 = -1.0 + g * 0.0 - 0.5
    d1 = 1.0 + g * 0.5 - 0.5
    d0 = 1.0 + g * 0.5 - 0.5
    A2 = d2
    A1 = d1 + g * 0.95 * A2
    A0 = d0 + g * 0.95 * A1
    check("advantage 与手算一致", np.allclose(advs, [A0, A1, A2], atol=1e-6), f"{[round(x,4) for x in advs]}")
    check("returns 就是折扣回报", np.allclose(rets, [1 + g * 1 + g * g * -1, 1 + g * -1, -1], atol=1e-6))

    rets_m, advs_m = compute_gae_multi([1.0, 1.0, -1.0, 1.0], [0.5] * 4, [3, 1])
    check("分轨迹后 advantage 仍是零均值单位方差（整批标准化）",
          abs(float(advs_m.mean())) < 1e-5 and abs(float(advs_m.std()) - 1.0) < 1e-4)

    print("\n[3] PPO 更新")
    # 用【刚采的这批】算 GAE —— 长度天然对齐
    rets_b, advs_b = compute_gae_multi(out[2], out[4], out[5])
    check("GAE 输出长度 == 总步数", len(rets_b) == len(out[2]), f"{len(rets_b)}")
    opt = optim.Adam(net.parameters(), lr=3e-4)
    before = [p.detach().clone() for p in net.parameters()]
    steps, clip, al, cl, ent = ppo_update(net, opt, out[0], out[1], out[3], rets_b, advs_b,
                                          epochs=2, minibatch_size=8)
    after = list(net.parameters())
    moved = sum(1 for b, a2 in zip(before, after) if not torch.equal(b, a2))
    check("梯度步数 > 0", steps > 0, f"{steps} 步")
    check("参数确实被更新了", moved == len(before), f"{moved}/{len(before)} 组")
    check("三个 loss 都是有限数", all(np.isfinite(x) for x in (al, cl, ent)),
          f"actor={al:.3f} critic={cl:.3f} 熵={ent:.3f}")
    check("clip 比例在 [0,1]", 0.0 <= clip <= 1.0, f"{clip*100:.1f}%")

    print("\n[4] 评估（贪心 + 固定种子）")
    r1 = evaluate(net, env, episodes=5, seed=99)
    r2 = evaluate(net, env, episodes=5, seed=99)
    check("同一模型同一组种子 → 结果完全可复现", r1 == r2, f"{[round(x,2) for x in r1]}")
    beans, stp, reasons = eval_detail(net, env, episodes=3, seed=7)
    check("eval_detail 拿到 3 局的豆数", len(beans) == 3, f"{beans}")
    check("死因统计有内容", len(reasons) > 0, str(dict(reasons)))

    print("\n" + "=" * 78)
    print(f"✅ 全部 {ok} 条断言通过")
    print("   ⭐ 注意：全程【一条蛇都没有】—— 算法层确实与环境无关")
    print("=" * 78)
