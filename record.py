"""
把训练好的模型跑一局，录成一张胶片图（不用开窗口）

运行：uv run python record.py [checkpoint] [局数上限]
      checkpoint 不填就用 ppo.DEFAULT_CKPT（snake_both.pth）
      没有模型就先跑一遍：uv run python ppo.py

输出：snake_filmstrip.png —— 一张图里 5×4 共 20 个关键时刻，
      从左上到右下是这一局的推进顺序，每帧标了吃到的豆子数。

默认会挑一局【打得最好的】来录（先扫若干局，选豆子最多的那局）。
想动起来看就用 watch.py，那个是开窗口实时播。
"""

import sys

import numpy as np
import pygame
import torch

from ppo import DEFAULT_CKPT, load_for_view
from snake_env import pick_font

CKPT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
SCAN = int(sys.argv[2]) if len(sys.argv) > 2 else 60     # 扫多少局挑最好的
COLS, ROWS = 5, 4
SCALE = 0.55
OUT = "snake_filmstrip.png"

env, model = load_for_view(CKPT)


def play(seed, keep_frames):
    """跑一局。keep_frames=True 时顺便存下每一帧。"""
    obs, info = env.reset(seed=seed)
    frames, beans = [], []
    while True:
        with torch.no_grad():
            probs, _ = model(torch.FloatTensor(obs).unsqueeze(0))
        obs, r, te, tr, info = env.step(probs.argmax(dim=-1).item())
        if keep_frames:
            frames.append(env.render())
            beans.append(info["food_eaten"])
        if te or tr:
            break
    return frames, beans, info


# ---------- 先扫一遍，挑打得最好的一局 ----------
best = (-1, None)
print("=" * 66)
print(f"先用 {SCAN} 局挑一局打得最好的…")
print("=" * 66)
for sd in range(SCAN):
    _, _, info = play(sd, keep_frames=False)
    if info["food_eaten"] > best[0]:
        best = (info["food_eaten"], sd)
    if info["food_eaten"] >= env.W * env.W - 3:      # 已经满分了，不用再找
        break

target_seed, target_beans = best[1], best[0]
print(f"  选中第 {target_seed} 局：吃到 {target_beans} 个豆")

frames, beans, info = play(target_seed, keep_frames=True)
env.close()

n = len(frames)
picked = [int(i * (n - 1) / (COLS * ROWS - 1)) for i in range(COLS * ROWS)]

# ---------- 拼图 ----------
pygame.init()
h, w = frames[0].shape[:2]
fw, fh = int(w * SCALE), int(h * SCALE)
HEADER = 34


f_title = pick_font(17)
f_tag = pick_font(13)

canvas = pygame.Surface((fw * COLS, HEADER + fh * ROWS))
canvas.fill((16, 18, 24))

# ⚠️ 标题里【不要放 emoji】—— 中文字体里没有 emoji 字形，会渲染成豆腐块 □
REASON_CN = {"wall": "撞墙", "self": "咬到了自己", "starve": "饿死", "win": "满盘"}
won = target_beans >= env.W * env.W - 3
ending = "★ 满分通关（蛇长填满 64 格）" if won else f"结局：{REASON_CN.get(info['end_reason'], info['end_reason'])}"
canvas.blit(f_title.render(
    f"贪吃蛇 · {CKPT}   ——   第 {target_seed} 局：吃到 {target_beans} 个豆，"
    f"{info['steps']} 步   {ending}",
    True, (255, 225, 130) if won else (232, 236, 244)), (10, 8))

for idx, fi in enumerate(picked):
    surf = pygame.transform.smoothscale(
        pygame.surfarray.make_surface(np.transpose(frames[fi], (1, 0, 2))), (fw, fh))
    r, c = divmod(idx, COLS)
    x, y = c * fw, HEADER + r * fh
    canvas.blit(surf, (x, y))
    pygame.draw.rect(canvas, (52, 58, 70), (x, y, fw, fh), 1)
    canvas.blit(f_tag.render(f"豆 {beans[fi]:2d}   步 {fi+1:3d}", True, (215, 240, 220)), (x + 4, y + 3))

pygame.image.save(canvas, OUT)
pygame.quit()
print(f"\n✅ 已保存 {OUT}（{fw*COLS}×{canvas.get_height()}）")
