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

REASON_CN = {"wall": "撞墙", "self": "咬到自己", "starve": "饿死", "win": "填满整盘！"}

# 左 / 右。想换成 ③ 纯 MLP 做三方对照，这里加一行就行。
SIDES = [
    ("feat", "先验规则 + MLP", "18 个数字（人算的）", "snake_both.pth"),
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
            reason = info["end_reason"]
            self.death = REASON_CN.get(reason, str(reason))
            self.games += 1
            self.history.append(info["food_eaten"])
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
        self._rest_until = None   # 两边都死了 → 停一会儿再开新局

        self._new_board_locked()
        threading.Thread(target=self._loop, daemon=True).start()

    # --------------------------------------------------------
    def _new_board_locked(self):
        """⭐ 两边吃同一个 seed —— 所以是同一盘开局。"""
        self.seed = random.randrange(1, 10 ** 9)
        self.round += 1
        for s in self.sides:
            s.reset(self.seed)

    def new_board(self):
        with self.lock:
            self._new_board_locked()
            self._rest_until = None

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
        while True:
            with self.lock:
                if not self.running:
                    time.sleep(0.05)
                    continue

                if all(s.done for s in self.sides):
                    # 都完了 —— 停 1.5 秒让人看清怎么死的，再开新棋局
                    if self._rest_until is None:
                        self._rest_until = time.time() + 1.5
                    elif time.time() >= self._rest_until:
                        self._new_board_locked()
                        self._rest_until = None
                else:
                    self._rest_until = None
                    for s in self.sides:
                        s.step()

                speed = self.speed

            time.sleep(1.0 / speed)

    # --------------------------------------------------------
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


class Handler(BaseHTTPRequestHandler):
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
            return self._json(ARENA.snapshot())
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "demo.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        self.send_error(404)

    def do_POST(self):
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

    print(f"  → 浏览器打开   http://{HOST}:{PORT}", flush=True)
    print("     （Ctrl+C 停）", flush=True)
    print("=" * 62, flush=True)

    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  停了。")


if __name__ == "__main__":
    main()
