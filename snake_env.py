"""
贪吃蛇 · 自写 Gymnasium 环境

================================================================
【为什么要自己写一个】

    前三个例子（LunarLander / Pendulum / Acrobot）的【状态和奖励都是官方给的】，
    你只需要调超参。

    这个例子里，两样都得自己设计 —— 而这两样恰恰是最容易翻车的地方。
    （insights.md #25：奖励是唯一能操控的旋钮，但前三个例子里那旋钮一直是别人在拧。）

【状态：12 维】

    0-2    左 / 直 / 右 三格是什么      0=空  0.5=墙  1=身体   （除 2 归一化）
    3      食物相对蛇头的【前后】      -1/0/1
    4      食物相对蛇头的【左右】      -1/0/1
    5      到食物的曼哈顿距离 / (2(W-1))
    6      饥饿计数 / MAX_HUNGER              （夹到 1.0，别越界）
    7      当前长度 / (W*W)
    8      蛇头到蛇尾的曼哈顿距离 / (2(W-1))
    9-11   走 左/直/右 之后的【可达空格数】/ (W*W)      ← flood fill

    ⚠️ [3][4] 是【蛇头坐标系】（随朝向旋转），不是屏幕的上下左右。
    ⚠️ [9:12] 的下标和动作 0/1/2 【一一对齐】。

    [9:12] 为什么必须有：1 格视野只能避免「立刻死」，避免不了「走进死胡同」。
    8×8 上一条长度 20+ 的蛇，死亡绝大多数是【自陷】而不是撞墙 ——
    而 [0:3] 在「左边是空地」和「左边那条通道被身体堵死」时输出【完全一样】。

【奖励：只奖励吃豆】

    吃到豆          +1.0
    撞墙 / 咬自己    -1.0   terminated
    饥饿超时         -1.0   terminated
    其他每步          0.0

    每局必然以 -1 结束，所以：  return = 吃到的豆子数 - 1

    ⚠️ 绝对不要加「每步 -0.01」这种密集惩罚 —— 那会让【主动送死】变成最优解：
           绕 100 步饿死：  -1(饿死) - 1.00(100步)  = -2.0
           10 步撞死：      -1(撞死) - 0.10         = -1.1   ← 更划算
       （这就是 insights.md #25 里「油费 ×100 → 直接摔」的翻版。）

【truncated 恒为 False】

    撞墙 / 咬自己 / 饿死 都是【任务规则】→ 全部 terminated=True。
    没有额外的步数上限 —— 饥饿计时已经封住了回合长度。
    所以仓库里那份 GAE 的「结尾补 0」在这里【恰好完全正确】，不用改一行。

【自写环境特有的坑（gymnasium 本来帮你免掉的）】

    ⚠️ 1. reset 必须接 seed，且第一行 super().reset(seed=seed)
          —— 否则固定种子的贪心评估变成随机 20 局，SOLVE 判据完全失效
    ⚠️ 2. 随机数一律用 self.np_random，别自建 default_rng
    ⚠️ 3. obs 每次必须是【新数组】——
          collect_one_trajectory 是 states.append(state) 存引用，
          复用缓冲区的话整批数据会变成最后一帧
    ⚠️ 4. 【尾巴腾格】非进食时尾巴会移开，所以「头进入尾格」是【合法】的。
          obs 和 step 必须共用同一个 _simulate()，否则状态撒谎、value 永远学不会
    ⚠️ 5. 左右旋转只写一份实现，否则动作映射和食物编码会静默镜像 180°

================================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ============================================================
# 参数
# ============================================================
W = 8                 # 网格边长（8×8 = 64 格）
INIT_LEN = 3          # 初始蛇长
MAX_HUNGER = 100      # 太久没吃到豆就饿死
                      #   标定公式：≈ 5~6 × 手写 AI 的「平均每豆步数」
                      #   太大 → 绕圈不死；太小 → 随机策略撞不到第一个豆
CELL = 40             # 渲染时每格多少像素

# 方向：(dr, dc)，行号向下增大
UP, RIGHT, DOWN, LEFT = (-1, 0), (0, 1), (1, 0), (0, -1)

EMPTY, WALL, BODY = 0, 1, 2       # 危险格编码

# 颜色（render 用）
C_BG      = (22, 26, 34)
C_GRID    = (38, 44, 56)
C_HEAD    = (150, 240, 175)
C_FOOD    = (240, 110, 110)

# ⚠️ 一整条蛇涂同一个颜色的话，看的时候【分不出头尾，也看不出身体往哪边走】。
#    所以拆成三件事：
#      1. 头：最亮 + 画两只眼睛（眼睛朝哪，蛇就往哪边走）
#      2. 身体：沿长度从亮到暗【渐变】—— 颜色由浅入深的方向就是「头 → 尾」
#      3. 尾巴：单独一个暖橙色，一眼就能从身体里挑出来
C_BODY_NEAR = (104, 204, 138)   # 紧挨着头的那一节
C_BODY_FAR  = (40, 104, 92)     # 快接到尾巴那一节（暗青）
C_TAIL      = (236, 172, 74)    # 尾巴：暖橙
C_EYE       = (14, 18, 24)


# ============================================================
# 中文字体（watch.py / record.py 画界面文字用）
#
# ⚠️ 这里踩过一个坑：
#    `pygame.font.match_font("PingFang SC")` 在 macOS 上【返回 None】——
#    PingFang 压根不在 pygame 能枚举出来的字体表里。
#    于是静默退回 SysFont(None) → FreeSans → 中文全渲染成【豆腐块 □□□】，
#    而且不报任何错。所以：先按【文件路径】直接找，找不到再按名字找。
#
# ⚠️ 另一条：中文字体里【没有 emoji 字形】，标题里别放 🎉 之类的东西 ——
#    同样是豆腐块。要用符号就用 ★ ← → 这种字体里真有的。
# ============================================================
CJK_FONTS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "pingfangsc", "hiraginosansgb", "stheitimedium",
    "microsoftyahei", "simhei", "notosanscjksc",
]


def find_cjk_font():
    """能渲染中文的字体路径；一个都找不到返回 None。"""
    import os
    import pygame
    for c in CJK_FONTS:
        if c.startswith("/"):
            if os.path.exists(c):
                return c
        else:
            p = pygame.font.match_font(c)
            if p:
                return p
    return None


def pick_font(size, bold=False):
    """按字号取一个中文字体。⚠️ 调用前必须先 pygame.init()。"""
    import pygame
    path = find_cjk_font()
    f = pygame.font.Font(path, size) if path else pygame.font.SysFont(None, size)
    f.set_bold(bold)
    return f


# ============================================================
# ⭐ 唯一的旋转实现 —— 只写这一份，别处一律调它
#    屏幕坐标（行号向下增大）下：
#       turn_right(UP)=RIGHT  turn_right(RIGHT)=DOWN  ...
#       turn_left (RIGHT)=UP  ← 自测里会断言这一条
# ============================================================
def turn_right(d):
    dr, dc = d
    return (dc, -dr)


def turn_left(d):
    dr, dc = d
    return (-dc, dr)


def action_to_dir(d, action):
    """动作 0=左转 / 1=直行 / 2=右转 → 新朝向"""
    if action == 0:
        return turn_left(d)
    if action == 1:
        return d
    if action == 2:
        return turn_right(d)
    raise ValueError(f"动作只能是 0/1/2，收到 {action}")


class SnakeEnv(gym.Env):
    """8×8 贪吃蛇，相对动作，特征状态 12 维。"""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self, grid=W, max_hunger=MAX_HUNGER, use_space=True,
                 use_deep=False, use_tail=False, render_mode=None, seed=None):
        super().__init__()
        self.W = grid
        self.max_hunger = max_hunger
        self.use_space = use_space          # [9:12]  flood fill（1 步前瞻）
        self.use_deep = use_deep            # [d:d+3] 2 步前瞻的最差空间
        self.use_tail = use_tail            # [t:t+3] 走完能否到达蛇尾
        self.render_mode = render_mode

        self.obs_dim = 9 + (3 if use_space else 0) + \
                       (3 if use_deep else 0) + (3 if use_tail else 0)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)      # 0=左转 1=直行 2=右转

        self._rng = np.random.default_rng(seed) if seed is not None else None

        self.snake = []       # [(row, col), ...]  头在 [0]
        self.direction = RIGHT
        self.food = (0, 0)
        self.hunger = 0
        self.steps = 0
        self.food_eaten = 0
        self.end_reason = None
        self._screen = None

    # --------------------------------------------------------
    # 随机数：优先用 gymnasium 的 np_random（可复现），没播种时退回自建
    # --------------------------------------------------------
    def _rand(self):
        if self._rng is not None:
            return self._rng
        return self.np_random

    # --------------------------------------------------------
    # reset
    # --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        # ⚠️ 这一行不能省：gymnasium 的 _np_random 只由 super().reset() 创建
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = None      # 交给 self.np_random

        # 固定起始：头 (4,4) 朝右，身体在左边 —— 可复现，方差小
        r0, c0 = self.W // 2, self.W // 2
        self.snake = [(r0, c0), (r0, c0 - 1), (r0, c0 - 2)]
        self.direction = RIGHT
        self.hunger = 0
        self.steps = 0
        self.food_eaten = 0
        self.end_reason = None
        self._place_food()

        return self._get_obs(), self._info()

    def _place_food(self):
        """从【空格集合】里采样。不能 while food in snake —— 快满盘时会死循环。"""
        occupied = set(self.snake)
        empty = [(r, c) for r in range(self.W) for c in range(self.W)
                 if (r, c) not in occupied]
        if not empty:
            self.food = None          # 满盘，没地方放
            return
        idx = int(self._rand().integers(len(empty)))
        self.food = empty[idx]

    def _info(self):
        return {
            "food_eaten": self.food_eaten,
            "steps": self.steps,
            "hunger": self.hunger,
            "length": len(self.snake),
            "end_reason": self.end_reason,
        }

    # --------------------------------------------------------
    # ⭐ 唯一的规则实现 —— obs 和 step 都调它
    #    返回 (新蛇头, 是否吃到豆, 危险码)   危险码 0=安全 1=墙 2=身体
    # --------------------------------------------------------
    def _simulate_from(self, body, direction, food, action):
        """⭐ 唯一的规则实现。step / obs / flood fill / 2步前瞻 / heuristic 全调它。

        在【任意给定局面】下走一步，返回 (新蛇头, 是否吃到豆, 危险码)。
        抽成"传局面进来"而不是读 self，是为了 2 步前瞻能复用同一套判定 ——
        否则第 2 步又会变成"另一份实现"，和第 2 个坑一样。
        """
        d = action_to_dir(direction, int(action))
        hr, hc = body[0]
        nr, nc = hr + d[0], hc + d[1]

        if not (0 <= nr < self.W and 0 <= nc < self.W):
            return (nr, nc), False, WALL

        will_eat = (food is not None and (nr, nc) == food)

        # ⚠️ 尾巴腾格：不进食时尾巴会移开，所以尾格是【可以进】的
        occ = set(body) if will_eat else set(body[:-1])
        if (nr, nc) in occ:
            return (nr, nc), will_eat, BODY

        return (nr, nc), will_eat, EMPTY

    def _simulate(self, action):
        return self._simulate_from(self.snake, self.direction, self.food, action)

    def _apply(self, action):
        """走完 action 之后的局面：(新蛇身, 新朝向, 新食物位置或 None)。死了返回 None。"""
        nh, eat, danger = self._simulate(action)
        if danger != EMPTY:
            return None
        body = [nh] + list(self.snake)
        if not eat:
            body = body[:-1]
        return body, action_to_dir(self.direction, int(action)), (None if eat else self.food)

    def will_die(self, action):
        """走这个动作会不会立刻死。自测里用它对着 obs[0:3] 逐格核对。"""
        return self._simulate(action)[2] != EMPTY

    # --------------------------------------------------------
    # step
    # --------------------------------------------------------
    def step(self, action):
        self.steps += 1
        self.hunger += 1

        new_head, will_eat, danger = self._simulate(action)

        if danger != EMPTY:
            self.end_reason = "wall" if danger == WALL else "self"
            self.direction = action_to_dir(self.direction, int(action))
            return self._get_obs(), -1.0, True, False, self._info()

        # 走一步
        self.direction = action_to_dir(self.direction, int(action))
        self.snake.insert(0, new_head)
        if will_eat:
            self.food_eaten += 1
            self.hunger = 0
            if len(self.snake) == self.W * self.W:
                self.end_reason = "win"
                return self._get_obs(), 1.0, True, False, self._info()
            self._place_food()
            return self._get_obs(), 1.0, False, False, self._info()

        self.snake.pop()          # 没吃到就掉尾

        if self.hunger > self.max_hunger:
            self.end_reason = "starve"
            return self._get_obs(), -1.0, True, False, self._info()

        return self._get_obs(), 0.0, False, False, self._info()

    # --------------------------------------------------------
    # 观测
    # --------------------------------------------------------
    def _reachable(self, start, occupied):
        """从 start 出发能走到的空格数（BFS）。occupied 是【移动后】的占位。"""
        from collections import deque
        seen = {start}
        q = deque([start])
        n = 0
        while q:
            r, c = q.popleft()
            for dr, dc in (UP, RIGHT, DOWN, LEFT):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.W and 0 <= nc < self.W):
                    continue
                if (nr, nc) in seen or (nr, nc) in occupied:
                    continue
                seen.add((nr, nc))
                n += 1
                q.append((nr, nc))
        return n

    def _can_reach_tail(self, body):
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
                if not (0 <= nr < self.W and 0 <= nc < self.W):
                    continue
                if (nr, nc) == tail:
                    return True
                if (nr, nc) in seen or (nr, nc) in occ:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        return False

    def _space_2step(self, action):
        """走 action 之后，【下一步】最差还能剩多少可达空间。

        1 步前瞻只能看到"这一步之后"，而自陷是十几步的过程。
        这里往前多看一步：走完 a 之后，对手（其实是自己）从 3 个后续里
        挑最差的那个 —— 得到的数就是"这个选择的安全垫有多厚"。
        """
        nxt = self._apply(action)
        if nxt is None:
            return 0
        body1, dir1, food1 = nxt
        worst = None
        for b in range(3):
            nh, eat, danger = self._simulate_from(body1, dir1, food1, b)
            if danger != EMPTY:
                continue                      # 这一后续会死，不算"被逼"的选项
            occ = set(body1) if eat else set(body1[:-1])
            occ.discard(nh)
            sp = self._reachable(nh, occ)
            worst = sp if worst is None else min(worst, sp)
        return 0 if worst is None else worst      # 3 个后续全死 → 0

    def _get_obs(self):
        # ⚠️ 每次必须新建数组 —— 存引用的话整批数据会变成最后一帧
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        head = self.snake[0]

        # [0:3] 三个动作的危险码
        for a in range(3):
            _, _, danger = self._simulate(a)
            obs[a] = danger / 2.0            # 0 / 0.5 / 1

        # [3][4] 食物在【蛇头坐标系】的方位
        if self.food is not None:
            dr, dc = self.food[0] - head[0], self.food[1] - head[1]
            d = self.direction
            rd = turn_right(d)               # 蛇头的"右"
            fwd = dr * d[0] + dc * d[1]      # 投影到朝向前方
            side = dr * rd[0] + dc * rd[1]   # 投影到朝向右侧
            obs[3] = float(np.sign(fwd))
            obs[4] = float(np.sign(side))
            # [5] 曼哈顿距离
            obs[5] = (abs(dr) + abs(dc)) / (2 * (self.W - 1))

        # [6] 饥饿（夹到 1.0，否则终止那一步会越界）
        obs[6] = min(1.0, self.hunger / self.max_hunger)

        # [7] 长度
        obs[7] = len(self.snake) / (self.W * self.W)

        # [8] 头到尾的距离
        tail = self.snake[-1]
        obs[8] = (abs(head[0] - tail[0]) + abs(head[1] - tail[1])) / (2 * (self.W - 1))

        # ---- 三组"走这一步会怎样"的特征，下标都和动作 0/1/2 对齐 ----
        off = 9

        # flood fill：走这个动作之后还剩多少可达空间（1 步前瞻）
        if self.use_space:
            for a in range(3):
                new_head, will_eat, danger = self._simulate(a)
                if danger != EMPTY:
                    obs[off + a] = 0.0
                    continue
                occ = set(self.snake) if will_eat else set(self.snake[:-1])
                occ.discard(new_head)
                obs[off + a] = self._reachable(new_head, occ) / (self.W * self.W)
            off += 3

        # 2 步前瞻：走 a 之后，下一步【最差】还剩多少空间
        #   1 步前瞻看不到"盘死"的过程，这个往前多看一步
        if self.use_deep:
            for a in range(3):
                obs[off + a] = self._space_2step(a) / (self.W * self.W)
            off += 3

        # 走 a 之后，蛇头能不能到达蛇尾
        #   能追到尾巴 = 可以一直跟着尾巴走 = 永远不会被围死
        if self.use_tail:
            for a in range(3):
                nxt = self._apply(a)
                obs[off + a] = 0.0 if nxt is None else float(self._can_reach_tail(nxt[0]))
            off += 3

        return obs

    # --------------------------------------------------------
    # 渲染（返回 HWC uint8，和 gymnasium 的 rgb_array 一致）
    # --------------------------------------------------------
    def _seg_style(self, i, n):
        """第 i 节身体（0 = 头）怎么画 → (颜色, 内缩像素)。

        渐变 + 递减的内缩，两样一起用，「哪头是头」就不需要猜了。
        最前和最后一节特判：头要亮，尾要另一个色。
        """
        if i == 0:
            return C_HEAD, 2
        if i == n - 1:
            return C_TAIL, 6                       # 尾巴：暖橙 + 明显小一圈
        t = (i - 1) / max(n - 3, 1)                # 在第 1 ~ n-2 节之间线性铺开
        col = tuple(int(round(a + (b - a) * t))
                    for a, b in zip(C_BODY_NEAR, C_BODY_FAR))
        return col, 2 + int(3 * t)

    def render(self):
        import pygame
        h = w = self.W * CELL
        if self._screen is None:
            self._screen = pygame.Surface((w, h))
        surf = self._screen
        surf.fill(C_BG)

        for i in range(self.W + 1):
            pygame.draw.line(surf, C_GRID, (0, i * CELL), (w, i * CELL))
            pygame.draw.line(surf, C_GRID, (i * CELL, 0), (i * CELL, h))

        if self.food is not None:
            fr, fc = self.food
            pygame.draw.circle(surf, C_FOOD,
                               (fc * CELL + CELL // 2, fr * CELL + CELL // 2),
                               CELL // 2 - 7)

        n = len(self.snake)
        for i, (r, c) in enumerate(self.snake):
            col, inset = self._seg_style(i, n)
            rect = pygame.Rect(c * CELL + inset, r * CELL + inset,
                               CELL - 2 * inset, CELL - 2 * inset)
            pygame.draw.rect(surf, col, rect, border_radius=6)

        # 眼睛：朝【前方】偏移，再朝【蛇头的右侧】左右分开
        #   眼睛是整张图上唯一能读出「它下一步要去哪」的东西
        hr, hc = self.snake[0]
        d = self.direction
        rd = turn_right(d)                            # 蛇头的「右边」
        cx, cy = hc * CELL + CELL // 2, hr * CELL + CELL // 2
        fx, fy = cx + d[1] * 8, cy + d[0] * 8         # 视线方向前移 8px
        for s in (1, -1):
            ex, ey = fx + rd[1] * s * 7, fy + rd[0] * s * 7
            pygame.draw.circle(surf, C_EYE, (int(ex), int(ey)), 3)

        arr = pygame.surfarray.array3d(surf)          # (w, h, 3)
        return np.ascontiguousarray(np.transpose(arr, (1, 0, 2)))   # → (h, w, 3)

    def close(self):
        self._screen = None


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

    print("=" * 76)
    print("snake_env 自测")
    print("=" * 76)

    # ---- 1. gymnasium 官方检查（免费帮你查 reset/step 契约、dtype、seed）----
    print("\n[1] gymnasium check_env")
    check_env(SnakeEnv(), skip_render_check=True)      # 失败会直接抛异常
    check("check_env 通过（reset/step 契约、obs dtype/shape、seed 可复现性）", True)

    env = SnakeEnv()

    # ---- 2. 观测的形状 / dtype / 范围 ----
    print("\n[2] 观测规格")
    obs, info = env.reset(seed=0)
    check("shape == (12,)", obs.shape == (12,), str(obs.shape))
    check("dtype == float32", obs.dtype == np.float32, str(obs.dtype))
    for flags, want in [((True, False, False), 12), ((True, True, False), 15),
                        ((True, False, True), 15), ((True, True, True), 18),
                        ((False, False, False), 9),
                        ((False, True, True), 15)]:
        e = SnakeEnv(use_space=flags[0], use_deep=flags[1], use_tail=flags[2])
        o, _ = e.reset(seed=0)
        check(f"obs_dim 组合 space={int(flags[0])} deep={int(flags[1])} tail={int(flags[2])} → {want}",
              e.obs_dim == want and o.shape == (want,), f"{e.obs_dim}/{o.shape}")
    lo, hi = 1e9, -1e9
    for _ in range(2000):
        a = env.action_space.sample()
        obs, r, te, tr, info = env.step(a)
        lo, hi = min(lo, obs.min()), max(hi, obs.max())
        if te:
            env.reset()
    check("所有观测值落在 [-1, 1]", lo >= -1.0 and hi <= 1.0, f"[{lo:.3f}, {hi:.3f}]")
    check("truncated 恒为 False", tr is False)

    # ---- 3. 左右旋转：不镜像 ----
    print("\n[3] 旋转方向（最容易静默镜像的地方）")
    check("RIGHT 左转 == UP", turn_left(RIGHT) == UP, f"{turn_left(RIGHT)}")
    check("RIGHT 右转 == DOWN", turn_right(RIGHT) == DOWN, f"{turn_right(RIGHT)}")
    d = RIGHT
    for _ in range(4):
        d = turn_left(d)
    check("连按 4 次左回到原点", d == RIGHT)
    d = RIGHT
    for _ in range(4):
        d = turn_right(d)
    check("连按 4 次右回到原点", d == RIGHT)

    # ---- 4. ⭐ 食物方位编码 == 动作映射（同时锁死三套坐标系）----
    print("\n[4] 食物方位 ⟷ 动作映射（同一个坐标系）")
    env.reset(seed=1)
    head = env.snake[0]
    # 把食物放到「动作 0（左转）会进入的那一格」
    tgt = (head[0] + turn_left(env.direction)[0], head[1] + turn_left(env.direction)[1])
    env.food = tgt
    o = env._get_obs()
    check("食物在动作0目标格 → obs[3](前后)==0", o[3] == 0, f"{o[3]}")
    check("食物在动作0目标格 → obs[4](左右)==-1 即左边", o[4] == -1, f"{o[4]}")
    # 动作 2（右转）那一格
    tgt = (head[0] + turn_right(env.direction)[0], head[1] + turn_right(env.direction)[1])
    env.food = tgt
    o = env._get_obs()
    check("食物在动作2目标格 → obs[4](左右)==+1 即右边", o[4] == 1, f"{o[4]}")

    # ---- 5. ⭐ 危险码 ⇔ 真死（状态不许撒谎）----
    print("\n[5] 危险码 ⇔ step() 真的会死（跑 300 局随机，每步对 3 个动作都查）")
    bad = 0
    for ep in range(300):
        env.reset()
        for _ in range(200):
            o = env._get_obs()
            for a in range(3):
                pred = o[a] > 0                       # 状态说会死
                actual = env.will_die(a)              # 规则说会死
                if pred != actual:
                    bad += 1
            a = env.action_space.sample()
            _, _, te, _, _ = env.step(a)
            if te:
                break
    check("300 局全程无矛盾", bad == 0, f"矛盾 {bad} 次")

    # ---- 6. 尾巴腾格：头进入尾格必须活着 ----
    print("\n[6] 尾巴腾格规则")
    env.reset(seed=2)
    # 造一个 2×2 的方框：头在 (4,4)，尾巴在 (4,3)，向左走会进尾格
    env.snake = [(4, 4), (5, 4), (5, 3), (4, 3)]
    env.direction = LEFT
    check("左走的目标格正是尾格", env.snake[-1] == (4, 3))
    check("_simulate 判为安全", env._simulate(1)[2] == EMPTY)
    _, r, te, _, _ = env.step(1)
    check("走进去确实活着", not te, f"reward={r}")
    # 反过来：吃到豆那一步尾巴不动，进尾格就该死
    env.reset(seed=2)
    env.snake = [(4, 4), (5, 4), (5, 3), (4, 3)]
    env.direction = LEFT
    env.food = (4, 3)          # 尾格上放豆 → 尾巴不动 → 撞自己
    check("尾格上有豆时，进尾格判为撞身体", env._simulate(1)[2] == BODY)

    # ---- 7. 吃豆 / 撞墙 / 咬自己 / 饿死 ----
    print("\n[7] 四条结束路径")
    env.reset(seed=3)
    env.food = (env.snake[0][0], env.snake[0][1] + 1)
    L0 = len(env.snake)
    _, r, te, _, info = env.step(1)
    check("吃豆 → reward=+1", r == 1.0, f"{r}")
    check("吃豆 → 蛇变长 1", len(env.snake) == L0 + 1, f"{L0}→{len(env.snake)}")
    check("吃豆 → 不结束", not te)
    check("吃豆 → 饥饿清零", info["hunger"] == 0)

    env.reset(seed=3)
    env.snake = [(0, 3), (0, 2), (0, 1)]
    env.direction = UP
    _, r, te, _, info = env.step(1)
    check("撞墙 → reward=-1, terminated", r == -1.0 and te)
    check("撞墙 → end_reason == 'wall'", info["end_reason"] == "wall")

    env.reset(seed=3)
    env.snake = [(4, 4), (5, 4), (5, 3), (4, 3), (3, 3)]
    env.direction = LEFT
    env.food = (0, 0)
    _, r, te, _, info = env.step(1)
    check("咬到自己 → reward=-1, terminated", r == -1.0 and te)
    check("咬到自己 → end_reason == 'self'", info["end_reason"] == "self")

    env.reset(seed=3)
    env.food = (0, 7)
    hit = None
    for i in range(1, MAX_HUNGER + 5):
        # 原地绕圈：一直左转，蛇只有 3 节，不会咬到自己
        _, r, te, _, info = env.step(0)
        if te:
            hit = i
            break
    check(f"饥饿恰好在第 {MAX_HUNGER+1} 步触发", hit == MAX_HUNGER + 1, f"实际 {hit}")
    check("饿死 → reward=-1", r == -1.0, f"{r}")
    check("饿死 → end_reason == 'starve'", info["end_reason"] == "starve")

    # ---- 8. 食物永远不在蛇身上 ----
    print("\n[8] 食物生成")
    on_snake = 0
    for ep in range(400):
        env.reset()
        for _ in range(150):
            if env.food in env.snake:
                on_snake += 1
            _, _, te, _, _ = env.step(env.action_space.sample())
            if te:
                break
    check("400 局食物从未落在蛇身上", on_snake == 0, f"越界 {on_snake} 次")

    # ---- 9. 满盘不崩 ----
    print("\n[9] 边界情况")
    env.reset(seed=4)
    env.snake = [(r, c) for r in range(W) for c in range(W)][:63]   # 留 1 格
    env._place_food()
    check("63/64 格时食物放在剩下那一格", env.food == (7, 7), f"{env.food}")

    env.snake = [(r, c) for r in range(W) for c in range(W)]        # 满盘
    env.food = None
    for _ in range(3):                       # 没有空格 → 不能抛异常
        env._place_food()
    check("满盘时 _place_food 不抛异常", env.food is None)

    # ---- 10. 可复现性 ----
    print("\n[10] 随机种子")
    e1, e2, e3 = SnakeEnv(), SnakeEnv(), SnakeEnv()
    r1 = [e1.reset(seed=7)[0].copy()]
    for _ in range(200):
        r1.append(e1.step(1)[0].copy())
        if e1.end_reason:
            e1.reset(seed=7)
    r2 = [e2.reset(seed=7)[0].copy()]
    for _ in range(200):
        r2.append(e2.step(1)[0].copy())
        if e2.end_reason:
            e2.reset(seed=7)
    check("reset(seed=7) 跑 200 步两次完全一致", all(np.array_equal(a, b) for a, b in zip(r1, r2)))

    eA, eB = SnakeEnv(), SnakeEnv()
    eA.reset(seed=7); fA = eA.food
    eB.reset(seed=8); fB = eB.food
    check("seed=7 与 seed=8 食物不同", fA != fB, f"{fA} vs {fB}")

    e4 = SnakeEnv()
    foods = set()
    for _ in range(30):
        e4.reset()
        foods.add(e4.food)
    check("未播种的连续 reset 得到不同食物（训练时不会每局重演）", len(foods) > 10, f"30 局里的不同食物 {len(foods)} 种")

    # ---- 11. obs 是新对象 ----
    print("\n[11] obs 必须是新数组（这个坑会静默毁掉整批数据）")
    e5 = SnakeEnv()
    o_first = e5.reset(seed=5)[0]
    for _ in range(3):
        o_last = e5.step(1)[0]
    check("首帧与末帧不共享内存", not np.shares_memory(o_first, o_last))
    check("首帧与末帧内容不同", not np.array_equal(o_first, o_last))

    # ---- 12. 新增的三组特征：语义对不对 ----
    print("\n[12] 2 步前瞻 / 蛇尾可达 的语义自测")

    e7 = SnakeEnv(use_deep=True, use_tail=True)
    check("obs_dim == 18", e7.obs_dim == 18, str(e7.obs_dim))

    # 蛇尾可达：头紧贴着尾巴（一个 2×2 的环）→ 应该能追到
    e7.reset(seed=2)
    e7.snake = [(4, 4), (5, 4), (5, 3), (4, 3)]
    e7.direction = LEFT
    _, _, danger = e7._simulate_from(e7.snake, e7.direction, e7.food, 1)
    body1, _, _ = e7._apply(1)
    check("2×2 环里，走一步后能追到尾巴", e7._can_reach_tail(body1))

    # 蛇头被自己【完全包死】→ 追不到尾巴
    #   螺旋：(4,4) 的上下左右 (3,4)(5,4)(4,3)(4,5) 全是自己的身体
    e7.reset(seed=2)
    e7.snake = [(4, 4), (4, 3), (5, 3), (5, 4), (5, 5),
                (4, 5), (3, 5), (3, 4), (3, 3)]
    e7.direction = UP
    check("头的四个邻居全是身体", all(
        e7._simulate_from(e7.snake, e7.direction, e7.food, a)[2] == BODY for a in range(3)))
    check("被包死时追不到尾巴", not e7._can_reach_tail(e7.snake))
    check("被包死时 _reachable == 0", e7._reachable(e7.snake[0], set(e7.snake)) == 0)

    # 2 步前瞻：走一步会撞死 → 0
    e7.reset(seed=3)
    e7.snake = [(0, 3), (0, 2), (0, 1)]
    e7.direction = UP
    check("必死的动作 → 2 步前瞻空间为 0", e7._space_2step(1) == 0)

    # 2 步前瞻一般 ≤ 1 步前瞻 —— 但【不是恒成立】：
    #   两次前瞻的占位集合不同（第 2 步时尾巴又移开了一格），
    #   所以"再看一步"偶尔能看到【更多】空间。
    #   这里只断言"绝大多数情况更保守"，不假装它是定理。
    e7.reset(seed=3)
    worse, total = 0, 0
    for _ in range(400):
        for a in range(3):
            nh, eat, danger = e7._simulate(a)
            if danger != EMPTY:
                continue
            occ = set(e7.snake) if eat else set(e7.snake[:-1])
            occ.discard(nh)
            sp1 = e7._reachable(nh, occ)
            sp2 = e7._space_2step(a)
            total += 1
            if sp2 > sp1:
                worse += 1
            assert 0 <= sp2 <= e7.W * e7.W
        a = e7.action_space.sample()
        _, _, te, _, _ = e7.step(a)
        if te:
            e7.reset()
    check("2 步前瞻绝大多数更保守（不是恒成立）",
          worse / total < 0.02, f"{worse}/{total} = {worse/total*100:.2f}% 例外")

    # 新特征不能越界
    for flags in [(True, True, False), (True, False, True), (True, True, True)]:
        e8 = SnakeEnv(use_deep=flags[1], use_tail=flags[2])
        lo2, hi2 = 1e9, -1e9
        o, _ = e8.reset()
        for _ in range(1500):
            o, _, te, _, _ = e8.step(e8.action_space.sample())
            lo2, hi2 = min(lo2, o.min()), max(hi2, o.max())
            if te:
                e8.reset()
        check(f"space/deep={int(flags[1])} tail={int(flags[2])} 观测仍在 [-1,1]",
              lo2 >= -1.0 and hi2 <= 1.0, f"[{lo2:.3f}, {hi2:.3f}]")

    # ---- 13. 渲染 ----
    print("\n[13] 渲染")
    e6 = SnakeEnv()
    e6.reset(seed=6)
    frame = e6.render()
    check("render() 返回 (H, W, 3) uint8", frame.shape == (W * CELL, W * CELL, 3) and frame.dtype == np.uint8,
          str(frame.shape))

    print("\n" + "=" * 76)
    print(f"✅ 全部 {ok} 条断言通过")
    print("=" * 76)
