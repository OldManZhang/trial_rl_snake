"""
模型 ① feature · 先验特征 + MLP

================================================================
【这个文件是"一个模型"的完整定义】

    它回答三个问题（注册表只问这三个）：

        NAME              我叫什么                    → "feature"
        make(cfg)         怎么建 env + net
        describe(cfg)     我是干什么的（打印用）
        save_meta(cfg)    存 checkpoint 时写什么

    ⚠️ 加第 5 个模型 = 新建一个这样的文件 + 在 models/__init__.py 加一行。
       不需要动 snake_env.py，不需要动 algo.py。

【表示：18 维 = 9 基础 + 9 先验】

    前面 9 维【直接沿用游戏给的】—— 这个 wrapper 是【扩展型】的：

        observation(obs)  →  np.concatenate([obs, 自己算的 9 维])

    后面 9 维是三组"人用 BFS 算好了喂进去"的东西：

        [9:12]   flood fill   走这个动作之后还剩多少可达空间（1 步前瞻）
        [12:15]  2 步前瞻     走完再往前看一步，最差还剩多少
        [15:18]  蛇尾可达     走这个动作之后，头能不能追到自己的尾巴

    ⚠️ 这三组的下标和动作 0/1/2 【一一对齐】。

【为什么 [9:12] 必须有】

    1 格视野只能避免「立刻死」，避免不了「走进死胡同」。
    8×8 上一条长度 20+ 的蛇，死亡绝大多数是【自陷】而不是撞墙 ——
    而 [0:3] 在「左边是空地」和「左边那条通道被身体堵死」时输出【完全一样】。

    实测（heuristic.py）：朴素贪心 17.84 → 贪心+空间 24.04，差 6.2 个豆。
    这 6.2 就是 flood fill 值多少钱。

【观察者模式：两种 wrapper 的对照】

    FeatObs   【扩展】  内层的 9 维照单全收，在后面接上自己算的
    GridObs   【替换】  内层的 9 维直接扔掉，从 env.snake 重画成 (4,8,8)

    两种都是合法写法。想加第三种（帧堆叠 / 位置编码 / 更大棋盘），
    照抄这个文件的结构就行。

================================================================
"""

import os
import sys

# ⚠️ 直接 `uv run python models/feature.py` 跑自测时，Python 只把【脚本所在目录】
#    （也就是 models/）加进 sys.path，所以下面的 `import snake_env` 会 ModuleNotFoundError。
#    往上一级补一条路 —— 这样 `python models/feature.py` 和 `python -m models.feature`
#    两种跑法都能用。（从 train.py / watch.py 里 import 时用不到这一行。）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium import spaces

from snake_env import SnakeEnv, W, MAX_HUNGER, EMPTY, UP, RIGHT, DOWN, LEFT

NAME = "feature"

# 给人看的（watch.py 不用，网页 demo 用）——
# ⚠️ 放在这里，是为了让"这个模型叫什么"只有【一个】答案。
#    以前这句话在 demo_server.py 的 SIDES 里、颜色在 demo.html 的 ACCENT 里，
#    加模型要改两处，漏一处不报错（只是网页少一行、或者两个模型撞色）。
LABEL = "先验规则 + MLP"
SUB = "6 个基础维度 + 12 个先验规则"
COLOR = "#96f0af"

# SnakeEnv 的构造参数 vs FeatObs 的构造参数 —— 得分开传
_ENV_KEYS = ("grid", "max_hunger", "render_mode", "seed")
_OBS_KEYS = ("use_space", "use_deep", "use_tail")


# ============================================================
# 三组先验特征 —— 全是 BFS，全是"人算好了喂进去的"
#
# ⚠️ 它们住在【模型】里，不在游戏里。因为它们不是"规则"，
#    是"这个模型选择怎么看这个局面"——换模型就不该带着走。
# ============================================================
def can_reach_tail(env, body):
    """蛇头能不能走到蛇尾？（在给定局面的空格图里 BFS）

    ⭐ 这是 Snake 的经典安全判据：
       只要能追到自己的尾巴，就可以一直跟着尾巴走 —— 【永远不会被围死】。
       追不到，说明自己把自己圈进了一个封闭区域。
    """
    from collections import deque
    head, tail = body[0], body[-1]
    # 尾巴下一步会移开，所以它是【可以进】的
    occ = set(body) - {tail}
    seen = {head}
    q = deque([head])
    while q:
        r, c = q.popleft()
        for dr, dc in (UP, RIGHT, DOWN, LEFT):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.W and 0 <= nc < env.W):
                continue
            if (nr, nc) == tail:
                return True
            if (nr, nc) in seen or (nr, nc) in occ:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return False


def space_2step(env, action):
    """走 action 之后，【下一步】最差还能剩多少可达空间。

    1 步前瞻只能看到"这一步之后"，而自陷是十几步的过程。
    这里往前多看一步：走完 a 之后，对手（其实是自己）从 3 个后续里
    挑最差的那个 —— 得到的数就是"这个选择的安全垫有多厚"。

    ⚠️ 复用 env.simulate_from / env.reachable，不自己重写碰撞判定。
    """
    nxt = env.apply(action)
    if nxt is None:
        return 0
    body1, dir1, food1 = nxt
    worst = None
    for b in range(3):
        nh, eat, danger = env.simulate_from(body1, dir1, food1, b)
        if danger != EMPTY:
            continue                      # 这一后续会死，不算"被逼"的选项
        occ = set(body1) if eat else set(body1[:-1])
        occ.discard(nh)
        sp = env.reachable(nh, occ)
        worst = sp if worst is None else min(worst, sp)
    return 0 if worst is None else worst      # 3 个后续全死 → 0


# ============================================================
# 表示层：9 基础 → 18 维
# ============================================================
class FeatObs(gym.ObservationWrapper):
    """SnakeEnv（基础 9 维）→ 18 维手工特征。

    三个开关就是消融实验的那三档：

        9  →  12（+flood fill）→  15（+2步前瞻）→  18（+蛇尾可达）

    ⚠️ 这是【扩展型】wrapper：内层那 9 维照单全收。
       不去重算它 —— 重算就是第二份实现，一定会漂。
    """

    def __init__(self, env, use_space=True, use_deep=False, use_tail=False):
        super().__init__(env)
        self.use_space = use_space          # [9:12]     flood fill（1 步前瞻）
        self.use_deep = use_deep            # 接下来 3 维  2 步前瞻的最差空间
        self.use_tail = use_tail            # 接下来 3 维  走完能否到达蛇尾

        self.extra = 3 * (int(use_space) + int(use_deep) + int(use_tail))
        self.obs_dim = 9 + self.extra
        # ⚠️ 必须覆盖 obs_dim / observation_space —— 不覆盖会【穿透】到内层的 9，
        #    把 check_env、watch.py 的打印、网络的第一层维度全骗过去。
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

    def __getattr__(self, name):
        # ⚠️ gymnasium 的 Wrapper 只转发标准 gym API，【不转发自定义属性】。
        #    watch.py 要读 env.W / env.snake / env.direction / env.max_hunger，
        #    少了这个就会 AttributeError。
        #    用 self.__dict__ 取 env 是为了避免 __getattr__ 递归。
        env = self.__dict__.get("env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)

    def observation(self, obs):
        return np.concatenate([obs, self._extra()]).astype(np.float32)

    def current_obs(self):
        """当前局面的完整观测（内层的 9 维 + 自己那几维）。

        ⚠️ 给【分析代码】用的（probe_cnn.py 拿它当真值）——
           它们要的是"这个局面下的 18 维是多少"，
           而不是"下一次 step 之后会返回什么"。
        """
        return self.observation(self.env._get_obs())

    def _extra(self):
        """后面那 9 维（或 6 / 3 / 0 维）。下标一律和动作 0/1/2 对齐。"""
        env, size = self.env, self.env.W
        out = []

        # flood fill：走这个动作之后还剩多少可达空间（1 步前瞻）
        if self.use_space:
            for a in range(3):
                new_head, will_eat, danger = env.simulate(a)
                if danger != EMPTY:
                    out.append(0.0)
                    continue
                occ = set(env.snake) if will_eat else set(env.snake[:-1])
                occ.discard(new_head)
                out.append(env.reachable(new_head, occ) / (size * size))

        # 2 步前瞻：走 a 之后，下一步【最差】还剩多少空间
        if self.use_deep:
            for a in range(3):
                out.append(space_2step(env, a) / (size * size))

        # 走 a 之后，蛇头能不能到达蛇尾
        if self.use_tail:
            for a in range(3):
                nxt = env.apply(a)
                out.append(0.0 if nxt is None else float(can_reach_tail(env, nxt[0])))

        return np.asarray(out, dtype=np.float32)


# ============================================================
# 网络
# ============================================================
class PPOActorCritic(nn.Module):
    """2×128 Tanh，actor / critic 【各有主干】（SepNet）。

    ⚠️ 为什么分开主干：共用主干时 critic 的梯度会通过公共参数把 actor 带崩
       （insights #19 / #26）。分开之后谁大谁小都淹不到对方。

    ⚠️ 契约（algo.py 只要求这两条）：
           model(x)                → (probs, value)
           model.dist_and_value(x) → (Categorical, value)
    """

    def __init__(self, state_dim=18, action_dim=3, hidden_size=128):
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

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 注册表要的四样
# ============================================================
def make(cfg):
    """建 (env, model)。

    ⚠️ env_kw 里混着两种参数：SnakeEnv 的和 FeatObs 的，得分开传。
       老 checkpoint 存的是 {'use_space':.., 'use_deep':.., 'use_tail':..}——
       那时候它们确实是 SnakeEnv 的参数，现在归 FeatObs 管。
       这里自动分流，所以 4 个老 checkpoint 都不用重训。
    """
    kw = dict(cfg.get("env_kw", {}))
    env_kw = {k: v for k, v in kw.items() if k in _ENV_KEYS}
    obs_kw = {k: v for k, v in kw.items() if k in _OBS_KEYS}

    env = FeatObs(SnakeEnv(**env_kw), **obs_kw)
    net = PPOActorCritic(state_dim=env.obs_dim, action_dim=3, hidden_size=128)
    return env, net


def describe(cfg, env=None):
    """watch.py / demo_server 用的一句话简介。"""
    kw = dict(cfg.get("env_kw", {}))
    on = lambda k: bool(kw.get(k, k == "use_space"))     # noqa: E731
    flags = f"flood fill={int(on('use_space'))} 2步前瞻={int(on('use_deep'))} " \
            f"蛇尾可达={int(on('use_tail'))}"
    dim = cfg.get("obs_dim") or (9 + 3 * (on('use_space') + on('use_deep') + on('use_tail')))
    return f"先验特征 {dim} 维（{flags}）"


def save_meta(cfg):
    """存 checkpoint 时写进去的东西。

    ⚠️ 必须把【表示方式】和权重存一起 —— 否则看的时候不知道该建几维环境。
       15 维是有歧义的（+2步前瞻 还是 +蛇尾可达，两者特征顺序不同），
       光看维度猜不出来。
    """
    return {"model": NAME,
            "env_kw": {k: v for k, v in cfg.get("env_kw", {}).items() if k in _OBS_KEYS},
            "obs_dim": cfg.get("obs_dim")}


def ui(cfg):
    """网页要的展示信息（名字 / 副标题 / 颜色）。"""
    return {"key": NAME, "name": LABEL, "subtitle": SUB, "color": COLOR}


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    ok = 0

    def check(name, cond, extra=""):
        global ok
        assert cond, f"❌ {name}  {extra}"
        ok += 1
        print(f"  ✅ {name}{('  ' + extra) if extra else ''}")

    print("=" * 78)
    print("models/feature 自测 · 先验特征表示 + MLP")
    print("=" * 78)

    # ---- 1. 三个开关拼出 9 / 12 / 15 / 18 ----
    print("\n[1] 消融的四个档位")
    for flags, want in [((False, False, False), 9), ((True, False, False), 12),
                        ((True, True, False), 15), ((True, False, True), 15),
                        ((True, True, True), 18), ((False, True, True), 15)]:
        e = FeatObs(SnakeEnv(), use_space=flags[0], use_deep=flags[1], use_tail=flags[2])
        o, _ = e.reset(seed=0)
        check(f"space={int(flags[0])} deep={int(flags[1])} tail={int(flags[2])} → {want} 维",
              e.obs_dim == want and o.shape == (want,), f"{e.obs_dim}/{o.shape}")

    # ---- 2. 契约 / 属性穿透 ----
    print("\n[2] gymnasium 契约 + 属性穿透")
    env = FeatObs(SnakeEnv(), use_space=True, use_deep=True, use_tail=True)
    o, info = env.reset(seed=0)
    check("shape == (18,)", o.shape == (18,), str(o.shape))
    check("dtype == float32", o.dtype == np.float32, str(o.dtype))
    check("obs_dim 被覆盖成 18（不是内层的 9）", env.obs_dim == 18, str(env.obs_dim))
    check("observation_space.shape == (18,)", env.observation_space.shape == (18,))
    check_env(FeatObs(SnakeEnv()), skip_render_check=True)
    check("check_env 通过（Wrapper 的 warning 是预期的）", True)
    check("env.W 穿透", env.W == W)
    check("env.snake 穿透", env.snake == [(4, 4), (4, 3), (4, 2)], str(env.snake))
    check("env.direction 穿透", env.direction == RIGHT, str(env.direction))
    check("env.max_hunger 穿透", env.max_hunger == MAX_HUNGER)
    check("env.render() 穿透", env.render().shape == (W * 40, W * 40, 3))
    check("⭐ env.simulate() 穿透（新模型靠它，不用重写规则）", env.simulate(1)[1] in (True, False))

    # ---- 3. ⭐ 扩展型：前 9 维必须和内层【一模一样】----
    print("\n[3] ⭐ 前 9 维 == 游戏给的基础 9 维（扩展，不是重算）")
    raw = SnakeEnv()
    raw.reset(seed=9)
    raw.snake = [(4, 4), (4, 3), (4, 2)]
    raw.direction = RIGHT
    raw.food = (2, 6)
    base = raw._get_obs()
    w = FeatObs(SnakeEnv(), use_space=True, use_deep=True, use_tail=True)
    w.reset(seed=9)
    w.env.snake, w.env.direction, w.env.food = raw.snake, raw.direction, raw.food
    full = w._get_obs() if hasattr(w, "_get_obs") else w.observation(w.env._get_obs())
    check("前 9 维逐位相同", np.array_equal(base, full[:9]),
          f"{base[:4]} vs {full[:4]}")
    check("后 9 维确实是自己加的", not np.array_equal(full[:9], full[9:]))

    # ---- 4. ⭐ 危险码 ⇔ 真死（状态不许撒谎）----
    print("\n[4] ⭐ 危险码 ⇔ step() 真的会死（300 局，每步对 3 个动作都查）")
    bad = 0
    for _ in range(300):
        env.reset()
        for _ in range(200):
            o = env.observation(env.env._get_obs())
            for a in range(3):
                if (o[a] > 0) != env.will_die(a):
                    bad += 1
            _, _, te, _, _ = env.step(env.action_space.sample())
            if te:
                break
    check("300 局全程无矛盾", bad == 0, f"矛盾 {bad} 次")

    # ---- 5. 蛇尾可达的语义 ----
    print("\n[5] 蛇尾可达 的语义")
    e = FeatObs(SnakeEnv(), use_tail=True, use_space=False)
    e.reset(seed=2)
    e.env.snake = [(4, 4), (5, 4), (5, 3), (4, 3)]       # 2×2 的环
    e.env.direction = LEFT
    body1 = e.env.apply(1)[0]
    check("2×2 环里，走一步后能追到尾巴", can_reach_tail(e.env, body1))

    # 蛇头被自己【完全包死】
    e.env.snake = [(4, 4), (4, 3), (5, 3), (5, 4), (5, 5), (4, 5), (3, 5), (3, 4), (3, 3)]
    e.env.direction = UP
    check("头的四个邻居全是身体", all(
        e.env.simulate_from(e.env.snake, UP, e.env.food, a)[2] != EMPTY for a in range(3)))
    check("被包死时追不到尾巴", not can_reach_tail(e.env, e.env.snake))

    # ---- 6. 2 步前瞻的语义 ----
    print("\n[6] 2 步前瞻 的语义")
    e2 = FeatObs(SnakeEnv(), use_deep=True, use_space=False)
    e2.reset(seed=3)
    e2.env.snake = [(0, 3), (0, 2), (0, 1)]
    e2.env.direction = UP
    check("必死的动作 → 2 步前瞻空间为 0", space_2step(e2.env, 1) == 0)

    # 2 步前瞻一般 ≤ 1 步前瞻 —— 但【不是恒成立】：
    #   两次前瞻的占位集合不同（第 2 步时尾巴又移开了一格），
    #   所以"再看一步"偶尔能看到【更多】空间。
    #   这里只断言"绝大多数情况更保守"，不假装它是定理。
    e2.reset(seed=3)
    worse, total = 0, 0
    for _ in range(400):
        for a in range(3):
            nh, eat, danger = e2.env.simulate(a)
            if danger != EMPTY:
                continue
            occ = set(e2.env.snake) if eat else set(e2.env.snake[:-1])
            occ.discard(nh)
            sp1 = e2.env.reachable(nh, occ)
            sp2 = space_2step(e2.env, a)
            total += 1
            if sp2 > sp1:
                worse += 1
            assert 0 <= sp2 <= e2.env.W * e2.env.W
        a = e2.action_space.sample()
        _, _, te, _, _ = e2.step(a)
        if te:
            e2.reset()
    check("2 步前瞻绝大多数更保守（不是恒成立）",
          worse / total < 0.02, f"{worse}/{total} = {worse/total*100:.2f}% 例外")

    # ---- 7. 全程不越界 ----
    print("\n[7] 跑满三种组合，观测都在 [-1,1]")
    for flags in [(True, False, False), (True, True, False),
                  (True, False, True), (True, True, True)]:
        ee = FeatObs(SnakeEnv(), use_space=flags[0], use_deep=flags[1], use_tail=flags[2])
        lo, hi = 1e9, -1e9
        o, _ = ee.reset()
        for _ in range(1500):
            o, _, te, _, _ = ee.step(ee.action_space.sample())
            lo, hi = min(lo, float(o.min())), max(hi, float(o.max()))
            if te:
                ee.reset()
        check(f"space/deep={int(flags[1])} tail={int(flags[2])} 观测仍在 [-1,1]",
              lo >= -1.0 and hi <= 1.0, f"[{lo:.3f}, {hi:.3f}]")

    # ---- 8. 网络 ----
    print("\n[8] 网络")
    net = PPOActorCritic(state_dim=18)
    x = torch.randn(8, 18)
    probs, v = net(x)
    check(f"批量输出 probs{tuple(probs.shape)} / value{tuple(v.shape)}",
          probs.shape == (8, 3) and v.shape == (8, 1))
    check("probs 行和为 1", torch.allclose(probs.sum(-1), torch.ones(8), atol=1e-5))
    (probs.sum() + v.sum()).backward()
    gnorm = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    check("反向有梯度", gnorm > 0, f"总梯度 {gnorm:.2f}")
    print(f"     参数量 {net.num_params()/1000:.1f}k")

    # ---- 9. 注册表四件套 ----
    print("\n[9] 注册表接口（make / describe / save_meta）")
    cfg = {"env_kw": {"use_space": True, "use_deep": True, "use_tail": True}, "obs_dim": 18}
    env2, net2 = make(cfg)
    check("make() 建出 18 维环境", env2.obs_dim == 18, str(env2.obs_dim))
    check("make() 建出对应输入维度的网络",
          net2.actor_body[0].in_features == 18, str(net2.actor_body[0].in_features))
    check("describe() 给出人话", "18 维" in describe(cfg), describe(cfg))
    check("save_meta() 记下了模型名", save_meta(cfg)["model"] == "feature")
    # ⭐ 老 checkpoint 的 env_kw 里混着 SnakeEnv 参数，make 要能自动分流
    env3, _ = make({"env_kw": {"use_space": True, "max_hunger": 50}})
    check("⭐ env_kw 里的 grid/max_hunger 自动归 SnakeEnv",
          env3.max_hunger == 50 and env3.obs_dim == 12, f"{env3.max_hunger}/{env3.obs_dim}")

    print("\n" + "=" * 78)
    print(f"✅ 全部 {ok} 条断言通过")
    print("=" * 78)
