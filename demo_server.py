"""
贪吃蛇 · 双边对照 Demo（本地网页）

跑起来：

    uv run python demo_server.py
    → 浏览器打开  http://127.0.0.1:8770

--------------------------------------------------------------------
⚠️ 这个文件【不重新实现任何东西】。

网页只负责【画】。所有计算都在这里，用的就是仓库里那套真东西：

    snake_env.py              真环境（连 BFS 特征都是它自己算的）
    snake_both.pth            真模型：18 维先验规则 + MLP
    snake_e2e_cnn.pth         真模型：纯 CNN

一行都没有重写。所以画面上看到的，就是终测 59.12 和 50.82 的那两个模型在玩，
不是"照着它们仿的一个网页版"。

两边从【同一个 seed】开局 —— 同一盘棋、同一个豆、同一个朝向，
之后各自发展（吃豆时机一不同，后面就岔开了，这本来就是要看的）。
--------------------------------------------------------------------

只依赖标准库 + 项目已有的包，不需要额外装东西。
"""

import json
import os
import random
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from snake_env import W
from ppo import load_for_view

HOST, PORT = "127.0.0.1", 8770
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 两个空闲阈值 ----
#
# 为什么不是 30 秒：线上实测过一次 **19.9 秒** 的响应 —— 那不是冷启动慢
# （冷启动本身只要 0.3 秒），是进程退出之后，内核把 torch 那几百 MB 的 .so
# 从内存里挤出去了，下一个访客要等云盘把它们重新读回来。
# 所以频繁退出反而更慢。
#
# 现在：不玩 10 分钟先停（省 CPU），再等 20 分钟才退（省内存）。
# 留 30 分钟是为了让 page cache 尽量还热着 —— 这期间来的访客是毫秒级响应。
IDLE_PAUSE = float(os.environ.get("SN_IDLE_PAUSE", 600.0))    # 10 分钟：暂停游戏

# 空闲这么久就【整个进程退出】，把内存也还回去（下次访问由 systemd 的
# .socket 单元再拉起来）。只在 socket 激活模式下生效，本地跑不受影响。
#
# ⚠️ 为什么不直接用 systemd 的 TimeoutIdleSec：那是个 systemd 指令，
#    不同版本行为/名字不一定一样（这台是 systemd 259，文档都查不全）。
#    自己退出还多一个好处 —— journal 里能看到"为什么没了"。
IDLE_EXIT = float(os.environ.get("SN_IDLE_EXIT", 1800.0))     # 30 分钟：退出进程

# main() 里如果发现自己是 systemd 拉起来的，会置 True
SOCKET_ACTIVATED = False

REASON_CN = {"wall": "撞墙", "self": "咬到自己", "starve": "饿死", "win": "填满整盘！"}

# 左 / 右。想换成 ③ 纯 MLP 做三方对照，这里加一行就行。
SIDES = [
    ("feat", "先验规则 + MLP", "6 个基础维度 + 12 个先验规则", "snake_both.pth"),
    ("cnn", "纯 CNN", "(4, 8, 8) 原始网格", "snake_e2e_cnn.pth"),
]


# ============================================================
# 一边：一个环境 + 一个模型 + 累计战绩
# ============================================================
class Side:
    def __init__(self, key, name, subtitle, ckpt):
        self.key, self.name, self.subtitle = key, name, subtitle
        self.ckpt = ckpt

        self.env, self.model = load_for_view(ckpt)
        self.model.eval()

        # GridObs 是套在 SnakeEnv 外面的一层 wrapper，画图要读的是【里面】那个真环境。
        # 特征版没有 wrapper，getattr 拿回自己。
        self.inner = getattr(self.env, "env", self.env)

        self.games = 0          # 打了几局
        self.history = []       # 每局的豆数
        self.reasons = Counter()
        self.home = 0           # 打满次数

        self.reset(seed=0)

    # --------------------------------------------------------
    def reset(self, seed):
        self.obs, _ = self.env.reset(seed=seed)
        self.seed = seed
        self.done = False
        self.death = None
        self.info = {"food_eaten": 0, "steps": 0, "hunger": 0, "length": 3, "end_reason": None}

    @torch.no_grad()
    def step(self):
        """走一步。贪心 —— 取概率最大的动作，和终测用的口径一致。"""
        if self.done:
            return
        x = torch.as_tensor(np.asarray(self.obs), dtype=torch.float32).unsqueeze(0)
        probs, _ = self.model(x)          # forward 已经 softmax 过了
        self.obs, _, term, trunc, info = self.env.step(int(probs.argmax(-1)))
        self.info = info

        if term or trunc:
            self.done = True
            self.death = REASON_CN.get(info["end_reason"], str(info["end_reason"]))

    def record(self):
        """把这一盘的成绩记进累计。

        ⚠️ 不在这里判「该不该记」—— 由 Arena 统一决定（见 _new_board_locked）。
           理由：如果死的时候就各自记分，中途按「新棋局」时，
           死得早的那边已经记了、还活着的那边那盘被丢掉，
           两边的局数就对不上，均分也就没法比了。
        """
        reason = self.info["end_reason"]
        self.games += 1
        self.history.append(self.info["food_eaten"])
        self.reasons[reason] += 1
        if reason == "win":
            self.home += 1

    # --------------------------------------------------------
    def snapshot(self):
        """给网页的那一小撮数据 —— 只发画图要用的，不发 obs 本身。"""
        i = self.inner
        n = len(self.history)

        # ⚠️ 满盘那一步，snake_env 判完 win 就 return 了，**没走 _place_food()** ——
        #    所以 i.food 还停在上一次的位置，而那格现在被蛇盖住了。
        #    那是环境该有的行为（局都结束了还放什么豆），但网页不该把它画出来。
        body = {(int(r), int(c)) for r, c in i.snake}
        food = None
        if i.food is not None and (int(i.food[0]), int(i.food[1])) not in body:
            food = [int(i.food[0]), int(i.food[1])]

        return {
            "key": self.key,
            "name": self.name,
            "subtitle": self.subtitle,
            "snake": [[int(r), int(c)] for r, c in i.snake],
            # ⚠️ direction 是元组 (dr, dc)（UP=(-1,0) / RIGHT=(0,1) …），不是编号。
            #    原样发过去，网页那边就不用再维护一张"编号↔方向"的表 —— 那种表最容易对错。
            "direction": [int(i.direction[0]), int(i.direction[1])],
            "food": food,
            "score": int(self.info["food_eaten"]),
            "steps": int(self.info["steps"]),
            "hunger": int(self.info["hunger"]),
            "max_hunger": int(i.max_hunger),
            "length": len(i.snake),
            "done": self.done,
            "death": self.death,
            "games": self.games,
            "avg": round(sum(self.history) / n, 1) if n else None,
            "best": max(self.history) if n else None,
            "home": self.home,
            "reasons": {REASON_CN.get(k, k): v for k, v in self.reasons.items()},
        }


# ============================================================
# 竞技场：两边同步推进
# ============================================================
class Arena:
    def __init__(self):
        self.sides = [Side(*s) for s in SIDES]
        self.speed = 15           # 每秒走几步
        self.running = True
        self.seed = 0
        self.round = 0
        self.lock = threading.Lock()

        self.last_seen = time.time()   # 最近一次有人访问
        self.auto_paused = False       # 是"没人看所以暂停"还是"用户手动暂停"

        self._new_board_locked()
        threading.Thread(target=self._loop, daemon=True).start()

    # --------------------------------------------------------
    def _new_board_locked(self):
        """开一盘新的。

        ⭐ 两层「一样」保证了均分可比：
           1. 两边吃同一个 seed —— 同一盘开局
           2. **只有两边都打完的那一盘才记分** —— 中途按「新棋局」时两边都不记，
              不会出现"死得早的先记了"这种一边多一边少
        """
        if all(s.done for s in self.sides):
            for s in self.sides:
                s.record()

        self.seed = random.randrange(1, 10 ** 9)
        self.round += 1
        for s in self.sides:
            s.reset(self.seed)

    def touch(self):
        """每个 HTTP 请求都调一次 —— 记下"有人在看"，并把自动暂停解除掉。

        ⚠️ 只解除【自动】暂停。用户自己按的暂停不能被一个轮询偷偷打开，
           不然他按了暂停、页面每 50ms 轮询一次，等于没按。
        """
        with self.lock:
            self.last_seen = time.time()
            if self.auto_paused:
                self.auto_paused = False
                # 两边都死了就别"恢复"了 —— 恢复也没得跑，只会把按钮闪一下
                if not all(s.done for s in self.sides):
                    self.running = True

    def new_board(self):
        """人工点「新棋盘」才走这儿。

        顺便把 running 打开 —— 两边打完时 _loop 会自动暂停（见那儿），
        这时点「新棋局」的意图显然是「再来一盘」，不该还要再点一次「继续」。
        """
        with self.lock:
            self._new_board_locked()
            self.running = True

    def toggle(self):
        with self.lock:
            self.running = not self.running
            return self.running

    def set_speed(self, v):
        with self.lock:
            self.speed = max(1, min(120, int(v)))
            return self.speed

    def step_once(self):
        """单步：暂停状态下，两边各走一格（看慢镜头用）。"""
        with self.lock:
            self.running = False
            for s in self.sides:
                s.step()

    # --------------------------------------------------------
    def _loop(self):
        """游戏主循环。

        ⚠️⚠️ 铁律：**绝对不能在 `with self.lock` 里面 sleep。**

        线上真炸过一次，记在这儿：原来 "游戏停着" 那一支写的是

            with self.lock:
                if not self.running:
                    time.sleep(0.05)      # ← 抱着锁睡
                    continue

        于是只要游戏停在"停着"的状态（两边都死了、等人工点「新棋盘」），
        这个循环就【几乎 100% 的时间占着锁】，每个 HTTP 请求都得排队等。
        请求一慢，浏览器就开更多连接，而 ThreadingHTTPServer 线程数【没有上限】——
        实测堆到 288 个线程、278 个 ESTABLISHED，请求 30 秒级超时，整台假死。

        为什么是老代码却这时才炸：以前两边死后 1.5 秒会自动开新盘，running 几乎
        总是 True，那条分支很少进。改成"不自动重开"之后才开始长时间停在那儿。

        改法：锁里只【算】下一步该歇多久，sleep 挪到锁外面。
        """
        while True:
            nap = 0.05
            with self.lock:
                # 空闲退出放最前面 —— 它必须在 `if not self.running` 之前。
                # IDLE_PAUSE 一定会先于 IDLE_EXIT 触发、把 running 置 False，
                # 而 running=False 之后每轮都会短路走开，退出检查就成了死代码。
                #
                # os._exit 而不是 sys.exit：这里不是主线程，sys.exit 退不出去。
                if SOCKET_ACTIVATED and time.time() - self.last_seen > IDLE_EXIT:
                    print(f"  空闲 {time.time() - self.last_seen:.0f}s，进程退出"
                          f"（下次访问由 systemd 的 snake.socket 再拉起来）", flush=True)
                    os._exit(0)

                if not self.running:
                    nap = 0.05                      # 停着 —— 轻量轮询等唤醒

                # 没人看就先停 —— 线上别让两个模型空转。
                elif time.time() - self.last_seen > IDLE_PAUSE:
                    self.running = False
                    self.auto_paused = True
                    nap = 0.05

                else:
                    for s in self.sides:
                        s.step()

                    # 两边都完了 —— 停在这儿，不自动开下一盘（等人工点「新棋盘」）。
                    # 在同一次迭代里就置 False：拖到下一轮的话，第二边刚死那一瞬间
                    # 网页会看到 running=True，下一帧才纠正，白闪一下。
                    if all(s.done for s in self.sides):
                        self.running = False
                        nap = 0.05
                    else:
                        nap = 1.0 / self.speed

            time.sleep(nap)

    # --------------------------------------------------------
    def health(self):
        """只读，不改任何状态。给监控和排查用。"""
        with self.lock:
            return {
                "running": self.running,
                "auto_paused": self.auto_paused,
                "idle_seconds": round(time.time() - self.last_seen, 1),
                "idle_pause_after": IDLE_PAUSE,
                "round": self.round,
                "all_done": all(s.done for s in self.sides),
            }

    def snapshot(self):
        with self.lock:
            return {
                "board": W,
                "seed": self.seed,
                "round": self.round,
                "speed": self.speed,
                "running": self.running,
                "sides": [s.snapshot() for s in self.sides],
            }


# ============================================================
# HTTP
# ============================================================
ARENA = None


class Server(ThreadingHTTPServer):
    """ThreadingHTTPServer 的两个默认值在公网上不够用，改掉。

    ⚠️ 背景：这台服务被浏览器以 20 次/秒轮询。默认 `request_queue_size = 5`
       意味着监听队列只排 5 个连接，一超就被拒 —— 而 Caddy 会重试，
       越重试越堵。抬到 64。
    """
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


class Handler(BaseHTTPRequestHandler):
    # ⚠️ 默认是 HTTP/1.0 —— 那意味着【每个请求都要新建一条 TCP 连接】。
    #    浏览器 20 次/秒轮询 = 20 条/秒，全走 Caddy 转进来，连接churn 极大。
    #    改成 1.1 之后 Caddy 能复用连接，实测连接数从几百降到个位数。
    #    前提：每个响应都必须带 Content-Length —— 下面的 _send 一直都有。
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                  # 关掉默认的逐条打印，太吵

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path.startswith("/state"):
            ARENA.touch()
            return self._json(ARENA.snapshot())
        if self.path.startswith("/health"):
            # ⚠️ 故意【不调 touch()】—— 这是个纯观察窗口。
            #    /state 一进来就 touch（=有人在看，快醒来），所以【没法用它观察
            #    "空闲暂停"到底有没有生效】—— 它自己会把自己叫醒。
            #    排查问题时看这个，它不改变任何状态。
            return self._json(ARENA.health())
        if self.path in ("/", "/index.html"):
            # 线上这一步其实轮不到 —— Caddy 会把 demo.html 当静态文件直接发，
            # 只有本地开发（和 handle_path 之外的情况）才走这儿。
            with open(os.path.join(HERE, "demo.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        self.send_error(404)

    def do_POST(self):
        ARENA.touch()
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}

        cmd = body.get("cmd")
        if cmd == "new":
            ARENA.new_board()
        elif cmd == "toggle":
            ARENA.toggle()
        elif cmd == "speed":
            ARENA.set_speed(body.get("value", 15))
        elif cmd == "step":
            ARENA.step_once()
        else:
            return self.send_error(400)

        self._json(ARENA.snapshot())


def systemd_socket():
    """如果是被 systemd 的 .socket 单元拉起来的，就把那个监听 fd 接过来。

    没有（本地直接跑）就返回 None，走自己 bind 的老路。两边的代码完全一样。

    原理：systemd 先建好监听 socket，第一个连接进来时才启动本进程，
    并把 fd 3 传下来。所以"服务没在跑"的时候，端口依然是有人接的 ——
    访客不会 connection refused，只是要多等几秒（等我们 import torch）。

    ⚠️ LISTEN_PID 必须等于自己的 pid。systemd 靠这个防止子进程
       误用父进程继承来的 fd（`sd_listen_fds` 的第一条规矩）。
    ⚠️ fd 号永远从 3 开始（0/1/2 是 stdin/out/err）。
    """
    if os.environ.get("LISTEN_PID") != str(os.getpid()):
        return None
    n = int(os.environ.get("LISTEN_FDS") or 0)
    if n < 1:
        return None
    import socket as _socket
    return _socket.fromfd(3, _socket.AF_INET, _socket.SOCK_STREAM)


def main():
    global ARENA

    # flush=True：输出被重定向到文件（比如 `> log`）时，stdout 是块缓冲的，
    #             不加这个的话 banner 要等缓冲区满才出现 —— 看着像卡住了。
    print("=" * 62, flush=True)
    print("  贪吃蛇 · 双边对照 Demo", flush=True)
    print("=" * 62, flush=True)
    for _, name, sub, ckpt in SIDES:
        print(f"  {name:<16} {sub:<22} {ckpt}", flush=True)
    print(flush=True)

    ARENA = Arena()

    global SOCKET_ACTIVATED

    sock = systemd_socket()

    # 测试用：SN_FAKE_SOCKET=1 假装是 systemd 拉起来的。
    # 用来在本机验「空闲退出」——不然要等 10 分钟，或者真去装个 systemd。
    if os.environ.get("SN_FAKE_SOCKET"):
        SOCKET_ACTIVATED = True
        srv = Server((HOST, PORT), Handler)
        print(f"  （SN_FAKE_SOCKET：假装是 socket 激活，本地起在 {HOST}:{PORT}）", flush=True)
        print("   → 用来验下面的空闲退出逻辑", flush=True)
    elif sock is not None:
        SOCKET_ACTIVATED = True
        # 接 systemd 的 fd。bind_and_activate=False 是因为轮不到我们 bind/listen ——
        # systemd 早就 listen 好了，我们只管 accept。
        srv = Server((HOST, PORT), Handler, bind_and_activate=False)
        srv.socket.close()                 # 上面那行顺手建了个没用的 socket，关掉
        srv.socket = sock
        srv.server_address = sock.getsockname()
        print(f"  （由 systemd socket 激活启动，fd 3 = {srv.server_address}）", flush=True)
        print("   → 线上由 Caddy 转进来", flush=True)
    else:
        srv = Server((HOST, PORT), Handler)
        print(f"  → 浏览器打开   http://{HOST}:{PORT}", flush=True)
    print("=" * 62, flush=True)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  停了。")


if __name__ == "__main__":
    main()
