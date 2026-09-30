"""
端到端贪吃蛇 · 表示层

================================================================
【这一层要干什么】

    把 SnakeEnv 吐的【9~18 维手工特征】换成【8×8 的原始网格】，
    让网络自己从格子里学特征。

    要回答的问题是：

        把 BFS 的结果拿走，只给原始棋盘，网络能不能自己算出来？

    flood fill / 2 步前瞻 / 蛇尾可达 —— 这三组值钱的维度
    全是人用 BFS 算好了喂进去的。这里把它们全拿掉，只留原料。

【为什么用 wrapper，而不是改 snake_env.py】

    snake_env.py 已经过 53 条自测，是【冻结】的。
    在外面套一层 ObservationWrapper 换掉观测，
    游戏规则就保证没变 —— 对照才干净（同棋盘、同奖励、同算法，只换输入）。

    ⚠️ 但 gymnasium 的 Wrapper 【不转发自定义属性】。
       不补 __getattr__ 的话，watch.py 读 env.W / env.snake / env.direction
       会直接 AttributeError。

【四个通道】

    ch0  蛇身    1 = 这一格是蛇（头尾都算）
    ch1  蛇头    1 = 这一格是蛇头
    ch2  食物    1 = 这一格是豆子
    ch3  饥饿    常数平面，64 格全填 hunger / MAX_HUNGER

    ⚠️ ch3 是【特意补的】，不是可选项。
       网格本身没有时间概念 —— 不加这层，网络不知道自己快饿死了，
       实验就变成了「缺了饥饿信息会怎样」，而不是「缺了特征工程会怎样」。
       （Atari DQN 喂 lives / score 常数平面也是这个道理。）

【旋转：头永远朝上】

    不转的话，网络得自己学会「不管朝哪边，前方都是我眼睛正对的那一格」。
    转成规范朝向之后，卷积核只要学【一套】模式，样本效率高得多。

    ⚠️ 只旋转，不平移 —— 蛇头的位置必须保留。
       贴着墙和贴着中间完全是两回事。

【网格里有什么、没有什么】

    有：身体 / 头 / 朝向 / 食物在哪、饥饿计时
    没有：食物多远、蛇多长、头尾多远          ← 要自己算
          【还剩多少空格】【会不会被逼死】【追不追得到尾巴】
                                              ← 要自己跑 BFS

    输入从 18 个数变成 256 个数（大 14 倍），能直接用的信息反而少了三类。

================================================================
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium import spaces

from snake_env import SnakeEnv, W, UP, RIGHT, DOWN, LEFT

# 通道编号
N_CH = 4
CH_BODY, CH_HEAD, CH_FOOD, CH_HUNGER = 0, 1, 2, 3


# ============================================================
# ⭐ 唯一的旋转实现 —— obs、自测、注释里的例子全调它
#    别处再写一份的话会【静默转错方向】，而且不报错
# ============================================================
# 一个方向转几次（每次顺时针 90°）才能变成「朝上」
TURNS_TO_UP = {UP: 0, LEFT: 1, DOWN: 2, RIGHT: 3}


def canonical(r, c, direction, size):
    """把格子 (r, c) 转到【蛇头朝上】的规范坐标系里。

    顺时针 90°： (r, c) → (c, N-1-r)   ，重复 TURNS_TO_UP[direction] 次。

    验算（N=8，头 (4,4) 朝右 → 要转 3 次）：
        头   (4,4) → (4,3) → (3,3) → (3,4)
        前方 (4,5) → (5,3) → (3,2) → (2,4)
        头在 (3,4)、前方在 (2,4) —— 正上方 ✅
    """
    n = size - 1
    for _ in range(TURNS_TO_UP[direction]):
        r, c = c, n - r
    return r, c


class GridObs(gym.ObservationWrapper):
    """SnakeEnv → (4, W, W) 原始网格，蛇头永远朝上。"""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_CH, env.W, env.W), dtype=np.float32)
        # ⚠️ 必须覆盖 obs_dim —— 不覆盖会【穿透】到内层的 12/18，把打印骗过去
        self.obs_dim = self.observation_space.shape

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
        return self._grid()

    def _grid(self):
        e, size, d = self.env, self.env.W, self.env.direction
        g = np.zeros((N_CH, size, size), dtype=np.float32)

        for (r, c) in e.snake:
            rr, cc = canonical(r, c, d, size)
            g[CH_BODY, rr, cc] = 1.0

        hr, hc = canonical(*e.snake[0], d, size)
        g[CH_HEAD, hr, hc] = 1.0

        if e.food is not None:
            fr, fc = canonical(*e.food, d, size)
            g[CH_FOOD, fr, fc] = 1.0

        g[CH_HUNGER, :, :] = min(1.0, e.hunger / e.max_hunger)
        return g


# ============================================================
# 网络
# ============================================================
class GridActorCritic(nn.Module):
    """吃 (C, W, W) 网格，吐动作概率和 value。

    arch="mlp"  ⭐ 把网格直接拉平。主干和【特征版完全同架构】（2×128 Tanh），
                   只换输入 —— 所以这是「特征 vs 原始网格」最干净的控制变量。
    arch="cnn"  两层 3×3 卷积 + 1×1 降通道，再看卷积值多少。

    ⚠️ 两条主干分开（SepNet），理由见 ppo.py 开头：共用主干时
       critic 的梯度会通过公共参数把 actor 带崩（insights #19 / #26）。
    """

    def __init__(self, obs_shape=(N_CH, W, W), action_dim=3, hidden_size=128, arch="mlp"):
        super().__init__()
        self.arch = arch
        self.obs_shape = tuple(obs_shape)
        self.actor_body = self._make_body(hidden_size, arch)
        self.critic_body = self._make_body(hidden_size, arch)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def _make_body(self, hidden_size, arch):
        c, h, w = self.obs_shape
        if arch == "mlp":
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(c * h * w, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, hidden_size), nn.Tanh())
        if arch == "cnn":
            return nn.Sequential(
                nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                # ⚠️ 1×1 卷积先把通道从 32 降到 8 —— 不然 flatten 之后
                #    32*8*8=2048 → 128 那层一个人就 26 万参数，白胖
                nn.Conv2d(32, 8, 1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8 * h * w, hidden_size), nn.Tanh())
        raise ValueError(f"arch 只能是 mlp / cnn，收到 {arch}")

    def forward(self, x):
        # ⚠️ 入口统一成 (B, C, H, W)：外面可能传 (1,C,H,W)（单帧）或 (B,C,H,W)（一批）
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return (torch.softmax(self.actor(self.actor_body(x)), dim=-1),
                self.critic(self.critic_body(x)))

    def dist_and_value(self, x):
        probs, value = self.forward(x)
        return Categorical(probs), value

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


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
    print("e2e_grid 自测 · 原始网格表示")
    print("=" * 78)

    # ---- 1. 形状 / dtype / 范围 / gymnasium 契约 ----
    print("\n[1] 观测规格 + gymnasium check_env")
    env = GridObs(SnakeEnv())
    o, info = env.reset(seed=0)
    check("shape == (4, 8, 8)", o.shape == (4, 8, 8), str(o.shape))
    check("dtype == float32", o.dtype == np.float32, str(o.dtype))
    check("obs_dim 被覆盖成 (4,8,8)（不是内层的 12）", env.obs_dim == (4, 8, 8), str(env.obs_dim))
    check("observation_space.shape == (4,8,8)", env.observation_space.shape == (4, 8, 8))
    # check_env 会对着 observation_space 严格校验 shape+dtype+范围
    check_env(GridObs(SnakeEnv()), skip_render_check=True)
    check("check_env 通过（Wrapper 的 warning 是预期的）", True)

    # 属性穿透 —— watch.py 全靠这个
    check("env.W 穿透", env.W == W)
    check("env.snake 穿透", env.snake == [(4, 4), (4, 3), (4, 2)], str(env.snake))
    check("env.direction 穿透", env.direction == RIGHT, str(env.direction))
    check("env.max_hunger 穿透", env.max_hunger == 100)
    check("env.render() 穿透", env.render().shape == (W * 40, W * 40, 3))

    # ---- 2. ⭐ 旋转正确性：脖子永远在头的正下方 ----
    print("\n[2] ⭐ 旋转正确性（头永远朝上）—— 搞错了不会报错，只会学得很烂")

    def consistent_snake(head, d, n=4):
        """造一条【和朝向自洽】的蛇：脖子必须在上一步头所在的位置，即 head − d。

        ⚠️ 不能随便设 direction 就完事 —— 身体和朝向对不上的话是个【不可能局面】，
           测出来的东西没有意义。（这条最开始就写错了，被断言当场抓住。）
        """
        return [(head[0] - d[0] * i, head[1] - d[1] * i) for i in range(n)]

    DIRS = [(UP, "上"), (RIGHT, "右"), (DOWN, "下"), (LEFT, "左")]

    for d, name in DIRS:
        e = GridObs(SnakeEnv())
        e.reset(seed=1)
        e.env.direction = d
        e.env.snake = consistent_snake((4, 4), d)
        # 蛇头转过去之后，脖子必须正好在【下一行】—— 也就是蛇在往上走
        h = canonical(*e.env.snake[0], d, W)
        nk = canonical(*e.env.snake[1], d, W)
        check(f"朝{name}时，脖子在头的正下方",
              nk == (h[0] + 1, h[1]), f"头{h} 脖子{nk}")

    # 再走一遍【完整链路】（从 obs 里读数，不是直接调 canonical）——
    # 万一 observation() 忘了用 canonical，上面那条查不出来
    for d, name in DIRS:
        e = GridObs(SnakeEnv())
        e.reset(seed=2)
        e.env.direction = d
        # (4,4) 是唯一一个【四个朝向都装得下 4 节】的位置：
        # 身体朝 head−d 方向铺 3 格，从中心出发四边都够
        e.env.snake = consistent_snake((4, 4), d)
        g = e._grid()
        hr, hc = np.argwhere(g[CH_HEAD] > 0)[0]
        check(f"朝{name}：obs 里头({hr},{hc})的正下方是身体", g[CH_BODY, hr + 1, hc] == 1)

    # ---- 3. ⭐ 动作映射在规范坐标系里固定 ----
    print("\n[3] ⭐ 三个动作在规范坐标系里永远是【固定】的三个偏移")
    # 规范系里：动作0(左转)→左，动作1(直行)→上，动作2(右转)→右
    EXPECT = {0: (0, -1), 1: (-1, 0), 2: (0, 1)}
    bad = []
    for d in (UP, RIGHT, DOWN, LEFT):
        for _ in range(40):
            e = GridObs(SnakeEnv())
            e.reset(seed=None)
            if len(e.env.snake) < 3:
                continue
            e.env.direction = d
            h = canonical(*e.env.snake[0], d, W)
            for a in range(3):
                nh, eat, danger = e.env._simulate(a)
                if danger != 0:          # 这一步会死，没有落点
                    continue
                nc = canonical(*nh, d, W)
                got = (nc[0] - h[0], nc[1] - h[1])
                if got != EXPECT[a]:
                    bad.append((d, a, got, EXPECT[a]))
            a = e.env.action_space.sample()
            e.env.step(a)
    check("4 个朝向 × 3 个动作 × 40 局，偏移全部对得上", not bad,
          f"错的 {len(bad)} 个：{bad[:3]}" if bad else "")

    # ---- 4. 旋转是等距变换（不改变曼哈顿距离）----
    print("\n[4] 旋转是等距变换")
    rng = np.random.default_rng(0)
    worst = 0
    for _ in range(3000):
        (r1, c1), (r2, c2) = rng.integers(0, W, 4).reshape(2, 2)
        d = [UP, RIGHT, DOWN, LEFT][int(rng.integers(4))]
        a1 = canonical(int(r1), int(c1), d, W)
        a2 = canonical(int(r2), int(c2), d, W)
        before = abs(r1 - r2) + abs(c1 - c2)
        after = abs(a1[0] - a2[0]) + abs(a1[1] - a2[1])
        worst = max(worst, abs(int(before) - int(after)))
    check("3000 组随机点对，曼哈顿距离完全不变", worst == 0, f"最大偏差 {worst}")

    # 旋转必须是棋盘上的双射（转完还在界内）
    out = [(r, c) for r in range(W) for c in range(W)
           for d in (UP, RIGHT, DOWN, LEFT)
           if not (0 <= canonical(r, c, d, W)[0] < W and 0 <= canonical(r, c, d, W)[1] < W)]
    check("转完不会跑到棋盘外", not out, f"越界 {len(out)} 个")

    all_cells = [canonical(r, c, RIGHT, W) for r in range(W) for c in range(W)]
    check("旋转是双射（64 个格子转完还是 64 个不重复的）", len(set(all_cells)) == W * W)

    # ---- 5. 饥饿平面 ----
    print("\n[5] ch3 饥饿常数平面")
    e = GridObs(SnakeEnv())
    e.reset(seed=3)
    e.env.hunger = 42
    g = e._grid()
    check("整片都等于 hunger/max_hunger", np.allclose(g[CH_HUNGER], 0.42), f"{g[CH_HUNGER][0,0]:.4f}")
    # 朝向变了饥饿平面不能跟着变（它和棋盘几何无关）
    planes = []
    for d in (UP, RIGHT, DOWN, LEFT):
        e.env.direction = d
        planes.append(e._grid()[CH_HUNGER])
    check("饥饿平面和朝向无关（4 个朝向完全一样）",
          all(np.array_equal(planes[0], p) for p in planes[1:]))
    e.env.hunger = 0
    check("hunger=0 → 全 0", np.allclose(e._grid()[CH_HUNGER], 0.0))
    e.env.hunger = e.env.max_hunger + 50
    check("hunger 越界也夹在 [0,1]", e._grid()[CH_HUNGER].max() == 1.0)

    # ---- 6. 通道不串味 ----
    print("\n[6] 四个通道各管各的")
    e = GridObs(SnakeEnv())
    e.reset(seed=4)
    e.env.snake = [(4, 4), (4, 3), (4, 2)]
    e.env.direction = RIGHT
    e.env.food = (0, 7)
    g = e._grid()
    check("ch0 蛇身格数 == 蛇长", int(g[CH_BODY].sum()) == 3, f"{int(g[CH_BODY].sum())}")
    check("ch1 蛇头恰好 1 格", int(g[CH_HEAD].sum()) == 1)
    check("ch2 食物恰好 1 格", int(g[CH_FOOD].sum()) == 1)
    check("头格在 ch0 和 ch1 里都是 1", bool((g[CH_BODY] * g[CH_HEAD]).sum() == 1))
    check("食物格在 ch0（蛇身）里是 0", bool((g[CH_BODY] * g[CH_FOOD]).sum() == 0))
    check("食物格在 ch1（蛇头）里是 0", bool((g[CH_HEAD] * g[CH_FOOD]).sum() == 0))
    check("ch0 ∩ ch2 == 空（身体不会长在豆子上）", bool((g[CH_BODY] * g[CH_FOOD]).sum() == 0))
    # 关键：旋转之后食物和头的【相对位置】必须保持
    h = np.argwhere(g[CH_HEAD] > 0)[0]
    f = np.argwhere(g[CH_FOOD] > 0)[0]
    h0 = canonical(*e.env.snake[0], RIGHT, W)
    f0 = canonical(*e.env.food, RIGHT, W)
    check("食物在 obs 里的位置 == 旋转后的真实位置",
          tuple(h) == h0 and tuple(f) == f0, f"头{tuple(h)}vs{h0} 豆{tuple(f)}vs{f0}")

    # ---- 7. 随机跑 300 局，全程合法 ----
    print("\n[7] 随机跑 300 局")
    e = GridObs(SnakeEnv())
    lo, hi, shapes, bad_head = 1e9, -1e9, set(), 0
    for _ in range(300):
        o, _ = e.reset()
        while True:
            shapes.add(o.shape)
            lo, hi = min(lo, float(o.min())), max(hi, float(o.max()))
            if o[CH_HEAD].sum() != 1:
                bad_head += 1
            o, r, te, tr, info = e.step(e.action_space.sample())
            if te or tr:
                break
    check("全程 shape 都是 (4,8,8)", shapes == {(4, 8, 8)}, str(shapes))
    check("全程取值都在 [0,1]", lo >= 0.0 and hi <= 1.0, f"[{lo:.3f}, {hi:.3f}]")
    check("每一帧都恰好有一个蛇头", bad_head == 0, f"异常 {bad_head} 帧")

    # ---- 8. 网络能前向、能反向 ----
    print("\n[8] 两个网络")
    for arch in ("mlp", "cnn"):
        net = GridActorCritic(obs_shape=(N_CH, W, W), arch=arch)
        x = torch.randn(8, N_CH, W, W)
        probs, v = net(x)
        check(f"{arch}: 批量输出 probs{tuple(probs.shape)} / value{tuple(v.shape)}",
              probs.shape == (8, 3) and v.shape == (8, 1))
        single = net(torch.randn(N_CH, W, W))[0]
        check(f"{arch}: 单帧 (C,H,W) 也能吃", single.shape == (1, 3), str(tuple(single.shape)))
        check(f"{arch}: probs 行和为 1", torch.allclose(probs.sum(-1), torch.ones(8), atol=1e-5))
        loss = probs.sum() + v.sum()
        loss.backward()
        gnorm = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
        check(f"{arch}: 反向有梯度", gnorm > 0, f"总梯度 {gnorm:.2f}")
        print(f"     {arch} 参数量 {net.num_params()/1000:.1f}k")

    # ---- 9. 网格版和特征版看到的是【同一个局面】----
    print("\n[9] 网格版 vs 特征版：同一局面，两种表示")
    raw = SnakeEnv()
    raw.reset(seed=9)
    raw.snake = [(4, 4), (4, 3), (4, 2)]
    raw.direction = RIGHT
    raw.food = (2, 6)
    feat = raw._get_obs()          # 特征版看到的东西
    grid = GridObs(SnakeEnv())
    grid.reset(seed=9)
    grid.env.snake, grid.env.direction, grid.env.food = raw.snake, raw.direction, raw.food
    g = grid._grid()
    check("同一局面下，特征版是 1 维 12 个数", feat.shape == (12,))
    check("同一局面下，网格版是 (4,8,8) 共 256 个数", g.shape == (4, 8, 8))
    # 特征版的 [9:12]（flood fill）在网格版里【没有任何对应物】
    check("网格版里【没有】flood fill / 2步前瞻 / 蛇尾可达 的任何通道",
          N_CH == 4, "只有 蛇身/蛇头/食物/饥饿 四层")

    print("\n" + "=" * 78)
    print(f"✅ 全部 {ok} 条断言通过")
    print("=" * 78)
