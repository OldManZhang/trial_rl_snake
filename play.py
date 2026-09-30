"""
贪吃蛇 · 你自己用键盘玩

运行：uv run python play.py

按键（⚠️ 是【相对转向】，不是上下左右）：
    ← / A      左转
    ↑ / W      直行
    → / D      右转
    R          重开一局
    ESC        退出

⚠️ 打开窗口后【先点一下窗口】，键盘才生效。

为什么不用 gymnasium 自带的 play()：
    那个是给「绝对动作」设计的（按住方向键 = 那个方向）。
    蛇用的是相对动作，按住的语义完全不同，而且它要求环境已注册。
    所以这里照 lunarlander/play.py 自己写事件循环。

为什么 HUD 上一定要画【当前朝向】：
    相对动作下，人很容易迷失方向 —— 你以为在按"往右"，
    其实按的是"右转"，蛇转完之后再按还是"右转"，会原地打转。
    盯着那个箭头玩。
"""

import sys

import numpy as np
import pygame

from snake_env import SnakeEnv, CELL, turn_left, turn_right, UP, RIGHT, DOWN, LEFT

SCALE = 2                 # 窗口放大倍数
BAR_H = 46                # 顶部信息栏高度
FPS = 12

# 按键 → 动作
KEYMAP = {
    pygame.K_LEFT: 0, pygame.K_a: 0,      # 左转
    pygame.K_UP: 1, pygame.K_w: 1,        # 直行
    pygame.K_RIGHT: 2, pygame.K_d: 2,     # 右转
}

ARROW = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}

C_BG   = (22, 26, 34)
C_BAR  = (16, 19, 26)
C_FG   = (232, 236, 244)
C_DIM  = (140, 150, 168)
C_HIT  = (250, 120, 120)


def pick_font(size):
    for name in ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti", "Arial Unicode MS"]:
        p = pygame.font.match_font(name)
        if p:
            return pygame.font.Font(p, size)
    return pygame.font.SysFont(None, size)


env = SnakeEnv()
obs, info = env.reset()

pygame.init()
clock = pygame.time.Clock()
GW = env.W * CELL
screen = pygame.display.set_mode((GW * SCALE, (GW + BAR_H) * SCALE))
pygame.display.set_caption("Snake · 键盘玩")
f_big = pick_font(15)
f_hud = pick_font(17)

print("=" * 64)
print("贪吃蛇 · 键盘玩")
print("=" * 64)
print("  ← / A  左转       ↑ / W  直行       → / D  右转")
print("  R 重开            ESC 退出")
print("  ⚠️ 打开后先【点一下窗口】，键盘才生效")
print("=" * 64)
print()

running, paused = True, False
episode = 0
best = 0
last = None

while running:
    # ---------------- 事件 ----------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                obs, info = env.reset()
                episode += 1
                last = None
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key in KEYMAP and not paused:
                obs, r, te, tr, info = env.step(KEYMAP[event.key])
                if te:
                    episode += 1
                    best = max(best, info["food_eaten"])
                    reason = {"wall": "撞墙", "self": "咬到自己",
                              "starve": "饿死", "win": "通关！"}.get(info["end_reason"], "结束")
                    last = f"第 {episode} 局  吃到 {info['food_eaten']} 个豆  {info['steps']} 步  {reason}"
                    print("  " + last)
                    obs, info = env.reset()

    # ---------------- 画 ----------------
    frame = env.render()
    surf = pygame.transform.scale(
        pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (GW * SCALE, GW * SCALE))

    screen.fill(C_BG)
    screen.blit(surf, (0, BAR_H * SCALE))

    # 顶部信息栏
    bar = pygame.Surface((GW * SCALE, BAR_H * SCALE))
    bar.fill(C_BAR)
    d = env.direction
    txt = (f"朝向 {ARROW[d]}    长度 {len(env.snake)}    "
           f"豆 {info['food_eaten']}    饥饿 {info['hunger']}/{env.max_hunger}")
    bar.blit(f_hud.render(txt, True, C_FG), (10, 4))
    tip = f"← 左转   ↑ 直行   → 右转    R 重开   ESC 退出" + ("     [已暂停 空格继续]" if paused else "")
    bar.blit(f_big.render(tip, True, C_DIM), (10, 26))
    screen.blit(bar, (0, 0))

    if last:
        screen.blit(f_big.render(last, True, C_HIT), (10, (GW + BAR_H - 14) * SCALE))

    pygame.display.flip()
    clock.tick(FPS)

env.close()
pygame.quit()
print(f"\n退出。本局最好成绩：吃到 {best} 个豆")
