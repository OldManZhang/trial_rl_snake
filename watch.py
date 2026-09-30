"""
看训练好的 PPO 自己玩贪吃蛇

运行：uv run python watch.py [checkpoint]
      checkpoint 不填就用 ppo.DEFAULT_CKPT（snake_both.pth，18 维，终测 59 个豆）
      没有模型就先跑一遍：uv run python ppo.py

按键：
    ESC 或关窗口     退出
    空格             暂停 / 继续；【死了之后按它开下一局】
    Enter / → / N    死了之后直接开下一局
    A                自动连播开关（死后停 1.6 秒自动下一局）
    S                存一张截图（snake_shot_<时间戳>.png）
    ↑ / ↓            加快 / 放慢

⚠️ 开局和每局结束时都会【停住等你按键】—— 一局打完了先看清楚它死在哪，
   别让它立刻重开把结尾冲掉。不想一局一按就按 A 开自动连播。

界面上怎么读这条蛇：
    最亮 + 有眼睛的那节   = 头，眼睛指的方向就是要走的方向
    从亮绿渐变到暗青      = 身体，颜色【由浅入深】的方向就是「头 → 尾」
    橙色那一节            = 尾巴

看的时候留意：
    1. 它是不是【直奔豆子】—— 而不是随机游走
    2. 蛇长了之后，它怎么绕开自己 —— 这是这个游戏真正的难点
    3. 被自己堵住时会不会【提前转向】留出空间
       （这是状态里 [9:12] 那三维 flood fill 在起作用）

⚠️ 每条命都以 -1 结束（撞墙 / 咬自己 / 饿死），
   所以「吃到 N 个豆」对应「得分 N-1」。
"""

import sys
import time
from collections import Counter

import numpy as np
import pygame
import torch

from ppo import DEFAULT_CKPT, load_for_view
from snake_env import CELL, pick_font

CKPT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
SCALE = 2
PANEL = 236          # 右侧面板宽度
HINT_H = 34          # 棋盘下方按键提示条高度
PAD = 18             # 面板内边距
FPS = 12
AUTO_DELAY = 1.6     # 自动连播时死后停留多久

ARROW = {(-1, 0): "↑", (0, 1): "→", (1, 0): "↓", (0, -1): "←"}
REASON_CN = {"wall": "撞墙", "self": "咬到了自己", "starve": "饿死", "win": "通关！"}

# 面板配色
BG, PANEL_BG, LINE = (16, 19, 26), (22, 26, 34), (44, 51, 64)
FG, MUTED, ACCENT, WARN = (232, 236, 244), (128, 139, 158), (150, 240, 175), (236, 172, 74)

env, model = load_for_view(CKPT)
MAX_BEANS = env.W * env.W - 3        # 初始蛇长 3，最多再吃这么多豆就满盘

print("=" * 68)
print("贪吃蛇 · 看 PPO 自己玩")
print("=" * 68)
print(f"  模型: {CKPT}")
print(f"  环境: {env.obs_dim} 维 (flood fill={int(env.use_space)} "
      f"2步前瞻={int(env.use_deep)} 蛇尾可达={int(env.use_tail)})")
print("  ESC 退出 | 空格 暂停/下一局 | Enter 下一局 | A 自动连播 | S 截图 | ↑↓ 调速")
print("  ⚠️ 打开后先【点一下窗口】，键盘才生效")
print("=" * 68)
print()

pygame.init()
clock = pygame.time.Clock()

f_big, f_h1 = pick_font(54, True), pick_font(20, True)
f_body, f_small, f_hint = pick_font(16), pick_font(13), pick_font(14)
f_over = pick_font(32, True)

GW = env.W * CELL
BOARD = GW * SCALE
WIN_W, WIN_H = BOARD + PANEL, BOARD + HINT_H
X0 = BOARD + PAD                                  # 面板里所有文字的左边
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Snake · PPO")


def kv(label, value, y, vcolor=FG):
    """面板里的一行「左标签 / 右数值」，返回下一行的 y"""
    screen.blit(f_body.render(label, True, MUTED), (X0, y))
    s = f_body.render(str(value), True, vcolor)
    screen.blit(s, (BOARD + PANEL - PAD - s.get_width(), y))
    return y + 23


def sep(y):
    pygame.draw.line(screen, LINE, (X0, y + 5), (BOARD + PANEL - PAD, y + 5))
    return y + 17


def chart(y, values, last_n=12):
    """最近 N 局的豆子数柱状图"""
    vals = values[-last_n:]
    bar_w = (PANEL - 2 * PAD) // last_n
    h_max, top = 56, y + 24
    for i, v in enumerate(vals):
        h = max(2, int(h_max * v / MAX_BEANS))
        col = ACCENT if v >= MAX_BEANS * 0.8 else (WARN if v >= MAX_BEANS * 0.4 else (74, 122, 104))
        pygame.draw.rect(screen, col, (X0 + i * bar_w, top + h_max - h, bar_w - 3, h),
                         border_radius=2)
    # 满盘那条参考线
    pygame.draw.line(screen, LINE, (X0, top), (X0 + last_n * bar_w - 3, top))
    return top + h_max


obs, info = env.reset(seed=777)
episode = 0
scores, reasons = [], Counter()
running, paused, over, auto = True, False, False, False
over_at, last_tag, last_beans, last_steps = 0.0, "", 0, 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_a:
                auto = not auto
            elif event.key == pygame.K_s:          # 存一张截图，方便贴出来讨论
                pygame.image.save(screen, f"snake_shot_{int(time.time())}.png")
                print("  📷 已存截图")
            elif event.key == pygame.K_UP:
                FPS = min(60, FPS + 4)
            elif event.key == pygame.K_DOWN:
                FPS = max(2, FPS - 4)
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_RIGHT, pygame.K_n):
                if over:                       # 结束后：空格 = 下一局
                    obs, info = env.reset()
                    over = False
                else:                          # 游戏进行中：空格 = 暂停
                    paused = not paused

    # ---------- 推进一局 ----------
    if over:
        if auto and time.time() - over_at >= AUTO_DELAY:
            obs, info = env.reset()
            over = False
    elif not paused:
        with torch.no_grad():
            probs, _ = model(torch.FloatTensor(obs).unsqueeze(0))
            action = probs.argmax(dim=-1).item()

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            # ⚠️ 这里【故意不 reset】—— 先把死掉的那一幕留在屏幕上
            episode += 1
            last_beans, last_steps = info["food_eaten"], info["steps"]
            last_tag = REASON_CN.get(info["end_reason"], info["end_reason"])
            scores.append(last_beans)
            reasons[info["end_reason"]] += 1
            print(f"  第 {episode:>2} 局   {last_beans:2d} 个豆   "
                  f"{last_steps:3d} 步   {last_tag}")
            over, over_at = True, time.time()

    # ---------- 画：棋盘 ----------
    screen.fill(BG)
    frame = env.render()
    surf = pygame.transform.scale(
        pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (BOARD, BOARD))
    screen.blit(surf, (0, 0))

    # ---------- 画：结束横幅（盖在棋盘上）----------
    if over:
        veil = pygame.Surface((BOARD, BOARD), pygame.SRCALPHA)
        veil.fill((8, 10, 14, 168))
        screen.blit(veil, (0, 0))
        win = info["end_reason"] == "win"
        title = "通关！填满整盘" if win else last_tag
        t = f_over.render(title, True, ACCENT if win else (255, 140, 130))
        screen.blit(t, ((BOARD - t.get_width()) // 2, BOARD // 2 - 62))
        sub = f_h1.render(f"第 {episode} 局 · 吃到 {last_beans} 个豆 · {last_steps} 步",
                          True, FG)
        screen.blit(sub, ((BOARD - sub.get_width()) // 2, BOARD // 2 + 2))
        tip = f_body.render("按 空格 开下一局" + ("　（自动连播中…）" if auto else "　或按 A 自动连播"),
                            True, MUTED)
        screen.blit(tip, ((BOARD - tip.get_width()) // 2, BOARD // 2 + 42))

    # ---------- 画：右侧面板 ----------
    pygame.draw.rect(screen, PANEL_BG, (BOARD, 0, PANEL, WIN_H))
    pygame.draw.line(screen, LINE, (BOARD, 0), (BOARD, WIN_H))

    y = 18
    screen.blit(f_small.render("本局吃到", True, MUTED), (X0, y))
    y += 22
    screen.blit(f_big.render(str(info["food_eaten"]), True, ACCENT), (X0 - 4, y))
    y += 70
    screen.blit(f_body.render(f"个豆　满分 {MAX_BEANS}", True, MUTED), (X0, y))
    y += 30
    y = sep(y)

    y = kv("局数", episode, y)
    y = kv("长度", len(env.snake), y)
    y = kv("步数", info["steps"], y)
    y = kv("饥饿", f"{info['hunger']} / {env.max_hunger}",
           y, WARN if info["hunger"] > env.max_hunger * 0.7 else FG)
    y = kv("朝向", ARROW.get(env.direction, "?"), y)
    y = sep(y)

    y = kv("本局均分", f"{np.mean(scores):.1f}" if scores else "—", y)
    y = kv("历史最好", max(scores) if scores else "—", y, ACCENT)
    y = sep(y)

    screen.blit(f_body.render("死因累计", True, MUTED), (X0, y))
    y += 23
    for k in ("wall", "self", "starve", "win"):
        y = kv("　" + REASON_CN.get(k, k), reasons.get(k, 0), y,
               ACCENT if k == "win" and reasons.get(k) else FG)
    y = sep(y)

    screen.blit(f_body.render("最近 12 局", True, MUTED), (X0, y))
    chart(y, scores)

    # ---------- 画：棋盘下方提示条 ----------
    pygame.draw.rect(screen, BG, (0, BOARD, BOARD, HINT_H))
    pygame.draw.line(screen, LINE, (0, BOARD), (BOARD, BOARD))
    status = "[已结束·等按键]" if over else ("[暂停]" if paused else f"{FPS} FPS")
    if auto:
        status += "　自动连播"
    screen.blit(f_hint.render(
        f"ESC 退出　空格 暂停/下一局　A 自动连播　↑↓ 调速　　　{status}",
        True, MUTED), (12, BOARD + 9))

    pygame.display.flip()
    clock.tick(FPS)

env.close()
pygame.quit()
print(f"\n退出。{len(scores)} 局平均 {np.mean(scores):.1f} 个豆"
      if scores else "\n退出。")
