"""
端到端 CNN 里到底装了什么知识？—— 线性探针 + 投影消融

================================================================
【要回答的问题】

    训练好的 CNN（(4,8,8) 原始网格 → 50.82 个豆，中位 61）
    它脑子里那 128 维，到底编码了什么？

    分三类去看：

      ① 人给过的量        flood fill / 2步前瞻 / 蛇尾可达 —— 它自己重新发现了吗？
      ② 基础量            危险码 / 食物方位 / 饥饿 / 长度 ——  sanity check
      ③ 人【没】给过的量   图距离 / 连通块 / 食物可达 —— 有没有长出【新知识】？

    第 ③ 类才是重点：如果 CNN 自己长出了人没想到要喂的量，
    那就是「涌现」的实证，而不只是「把人算好的抄了一遍」。

【两个实验，缺一不可】

    ⭐ 实验 A：线性探针
       冻结 CNN，收集 (128 维激活, 真值) 对，做岭回归，看 held-out R²。
       R² 高 = 信息【在里面】。

    ⚠️ 但 R² 高 ≠ CNN 在用这个信息。可能只是"恰好在那里，但没被用"。
       所以必须做第二个：

    ⭐ 实验 B：投影消融
       把探针方向从激活里扣掉（h' = h − (h·ŵ)ŵ），看贪心分数掉不掉。
       掉了 = 它【真的在用】这个方向。

       对照组：随机方向（同等处理）—— 用来量"扣掉任意一个方向"
       本身会造成多少损伤。只有显著超过随机对照，才算真信号。

【一个方法学上的坑】

    切训练/测试集必须【按局切】，不能按步切。
    同一局里相邻两步的激活高度相关，按步随机切会把测试集泄漏进训练集，
    R² 会被虚高。这里按打乱的局号切 80/20。

【顺带的对照组】

    同一批数据，用【原始网格 256 维】直接回归同一个目标。
    这条能回答「信息在不在输入里、是不是线性可读」——
    和 CNN 的 128 维一比，就知道 CNN 到底把表示变好了多少。

================================================================
"""

import sys
from collections import deque

import numpy as np
import torch

from snake_env import SnakeEnv, W, MAX_HUNGER, UP, RIGHT, DOWN, LEFT
from e2e_grid import GridObs, GridActorCritic, N_CH

CKPT = sys.argv[1] if len(sys.argv) > 1 else "snake_e2e_cnn.pth"
N_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 260
ABLATE_EPISODES = 50

DIRS = (UP, RIGHT, DOWN, LEFT)


# ============================================================
# ③ 人没给过的量：真正的图距离 / 连通块 / 可达性
# ============================================================
def graph_stats(env):
    """在【当前局面】上做 BFS，算出人从来没喂过的那几个量。

    ⚠️ 和 snake_env 里的 flood fill 不一样：
       flood fill 数的是「走一步之后还剩多少空格」，
       这里算的是「头到食物/尾巴的真实图距离」和「空间被切成几块」。

    尾巴会让位，所以尾格算可进（和 snake_env 的规则保持一致）。
    """
    free = {(r, c) for r in range(W) for c in range(W)} - set(env.snake)
    head, tail = env.snake[0], env.snake[-1]
    food = env.food
    free.add(tail)                      # 尾巴下一步会让开

    # 从蛇头 BFS
    dist = {head: 0}
    q = deque([head])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            n = (r + dr, c + dc)
            if n in free and n not in dist:
                dist[n] = dist[(r, c)] + 1
                q.append(n)

    d_food = dist.get(food, -1) if food is not None else -1      # 追不到 = -1
    d_tail = dist.get(tail, -1)
    comp = len(dist)                                             # 头能到达的格子数

    # 全盘连通块数（把所有空格分块）
    left = set(free)
    n_comp = 0
    while left:
        n_comp += 1
        s = left.pop()
        q = deque([s])
        while q:
            r, c = q.popleft()
            for dr, dc in DIRS:
                n = (r + dr, c + dc)
                if n in left:
                    left.discard(n)
                    q.append(n)

    n_free = len(free)
    return {
        "图距离 头→豆": d_food,
        "图距离 头→尾": d_tail,
        "连通块数": n_comp,
        "头区占空格比": comp / max(n_free, 1),
        "豆在头区里": 0.0 if (food is None or d_food < 0) else 1.0,
        "剩余空格数": n_free - 1,          # 减掉刚加回去的尾格
    }


# ============================================================
# 收集
# ============================================================
def collect(model, n_games):
    """跑若干局，每步记下 (CNN 激活, 原始网格, 特征真值, 新量真值, 局号)"""
    # ⚠️ 内层把三个特征开关全打开 —— 但【只用来当真值】，不喂给 CNN
    env = GridObs(SnakeEnv(use_space=True, use_deep=True, use_tail=True))
    acts, grids, feats, news, gids = [], [], [], [], []

    for g in range(n_games):
        obs, _ = env.reset(seed=1000 + g)
        while True:
            grid = env._grid()                     # CNN 看到的 (4,8,8)
            feat = env.env._get_obs()              # 18 维真值（含 BFS）
            with torch.no_grad():
                h = model.actor_body(torch.as_tensor(grid).unsqueeze(0))
            acts.append(h.squeeze(0).numpy())
            grids.append(grid.ravel())
            feats.append(feat)
            news.append(graph_stats(env.env))
            gids.append(g)

            with torch.no_grad():
                probs, _ = model(torch.as_tensor(grid).unsqueeze(0))
            obs, r, te, tr, info = env.step(int(probs.argmax(-1)))
            if te or tr:
                break
    env.close()
    return (np.array(acts), np.array(grids), np.array(feats), news, np.array(gids))


# ============================================================
# 岭回归 + held-out R²
# ============================================================
def _r2(pred, yte):
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    ym = ytr.mean()
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return _r2(B @ w + ym, yte)


def mlp_r2(Xtr, ytr, Xte, yte, epochs=400, hidden=96, seed=0):
    """非线性探针。

    ⚠️ 必须有这个：线性 R² 低【不能】推出「信息不在里面」——
       也可能只是线性读不出来。两者必须分开。

    ⚠️ 用训练集再切 10% 当验证集早停，【不能】拿测试集选 epoch —— 那等于偷看。
    """
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = torch.as_tensor(((Xtr - mu) / sd), dtype=torch.float32)
    B = torch.as_tensor(((Xte - mu) / sd), dtype=torch.float32)
    ym, ys = ytr.mean(), ytr.std() + 1e-8
    ya = torch.as_tensor((ytr - ym) / ys, dtype=torch.float32).unsqueeze(-1)
    yb = torch.as_tensor((yte - ym) / ys, dtype=torch.float32).unsqueeze(-1)

    n_val = max(1, len(A) // 10)
    Atr, Aval = A[:-n_val], A[-n_val:]
    ytr_, yval = ya[:-n_val], ya[-n_val:]

    net = torch.nn.Sequential(
        torch.nn.Linear(A.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best, best_state, patience = 1e9, None, 0
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(Atr), ytr_)
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = torch.nn.functional.mse_loss(net(Aval), yval).item()
        if v < best - 1e-5:
            best, best_state, patience = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience > 40:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = (net(B) * ys + ym).squeeze(-1).numpy()
    return _r2(pred, yte)


# ============================================================
# 消融：把某个方向从激活里投影掉
# ============================================================
def play_ablated(model, w_hat, episodes=ABLATE_EPISODES, seed=777):
    """贪心玩若干局，返回平均豆数。w_hat=None 表示不消融。"""
    env = GridObs(SnakeEnv(**dict(use_space=False, use_deep=False, use_tail=False)))
    beans = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        while True:
            with torch.no_grad():
                h = model.actor_body(torch.as_tensor(obs).unsqueeze(0))
                if w_hat is not None:
                    # h 和 w_hat 都是 (1,128)，投影 = 点积后沿该方向减掉
                    h = h - (h * w_hat).sum(-1, keepdim=True) * w_hat
                probs = torch.softmax(model.actor(h), dim=-1)
            obs, r, te, tr, info = env.step(int(probs.argmax(-1)))
            if te or tr:
                break
        beans.append(info["food_eaten"])
    env.close()
    return float(np.mean(beans))


# ============================================================
if __name__ == "__main__":
    print("=" * 94)
    print("CNN 里装了什么知识 · 线性探针 + 投影消融")
    print("=" * 94)

    env0 = GridObs(SnakeEnv(**dict(use_space=False, use_deep=False, use_tail=False)))
    model = GridActorCritic(obs_shape=(N_CH, W, W), action_dim=3, hidden_size=128, arch="cnn")
    try:
        ck = torch.load(CKPT, weights_only=True)
    except FileNotFoundError:
        sys.exit(f"❌ 找不到 {CKPT}")
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"  模型 {CKPT}（{ck.get('arch')}，评估最好 {ck.get('eval_score', float('nan')):+.2f}）")
    print()

    print(f"正在跑 {N_GAMES} 局收集激活…", flush=True)
    acts, grids, feats, news, gids = collect(model, N_GAMES)
    print(f"  {len(acts)} 步，激活 {acts.shape[1]} 维\n", flush=True)

    # ---- 定义三类探针目标 ----
    # ① 人给过的（[9:18]）
    P1 = [("flood fill 左", 9), ("flood fill 直", 10), ("flood fill 右", 11),
          ("2步前瞻 左", 12), ("2步前瞻 直", 13), ("2步前瞻 右", 14),
          ("蛇尾可达 左", 15), ("蛇尾可达 直", 16), ("蛇尾可达 右", 17)]
    # ② 基础量（[0:9]）
    P2 = [("危险码 左", 0), ("危险码 直", 1), ("危险码 右", 2),
          ("食物前后", 3), ("食物左右", 4), ("食物曼哈顿距离", 5),
          ("饥饿", 6), ("长度", 7), ("头尾曼哈顿距离", 8)]
    NEW = list(news[0].keys()) if news else []

    # ---- 按【局】切训练 / 测试 ----
    rng = np.random.default_rng(0)
    uniq = np.unique(gids)
    rng.shuffle(uniq)
    tr_g = set(uniq[: int(len(uniq) * 0.8)].tolist())
    tr = np.array([g in tr_g for g in gids])
    te = ~tr
    print(f"  按局切分：训练 {tr.sum()} 步 / 测试 {te.sum()} 步"
          f"（{len(tr_g)} 局 vs {len(uniq)-len(tr_g)} 局）\n")

    print("=" * 100)
    print("实验 A · 探针（held-out R²）—— 线性 / 非线性分开报")
    print("=" * 100)
    print(f"  {'目标':<18} {'类别':<11} {'线性':>7} {'非线性':>8} {'← CNN 128维':>12} "
          f"{'线性':>7} {'非线性':>8} {'← 原始网格 256维':>12}")
    print("  " + "-" * 88)

    rows = []
    for label, group, idx in ([(l, "① 人给过", i) for l, i in P1]
                              + [(l, "② 基础", i) for l, i in P2]
                              + [(l, "③ 人没给过", None) for l in NEW]):
        if idx is not None:
            y = feats[:, idx].astype(np.float64)
        else:
            y = np.array([n[label] for n in news], dtype=np.float64)
        r_cnn = ridge_r2(acts[tr], y[tr], acts[te], y[te])
        n_cnn = mlp_r2(acts[tr], y[tr], acts[te], y[te])
        r_raw = ridge_r2(grids[tr], y[tr], grids[te], y[te])
        n_raw = mlp_r2(grids[tr], y[tr], grids[te], y[te])
        rows.append((label, group, max(r_cnn, n_cnn), r_cnn, n_cnn, y, idx))
        print(f"  {label:<18} {group:<11} {r_cnn:>7.3f} {n_cnn:>8.3f} {'':>12} "
              f"{r_raw:>7.3f} {n_raw:>8.3f}", flush=True)

    # ---- 实验 B：投影消融 ----
    print()
    print("=" * 94)
    print("实验 B · 投影消融（把这一个方向从激活里扣掉，看贪心分数掉不掉）")
    print("=" * 94)
    base = play_ablated(model, None)
    print(f"  基线（不消融）              {base:6.2f} 个豆\n", flush=True)

    # 随机方向对照：跑 5 个，取均值和范围
    rnd = []
    for k in range(5):
        v = torch.randn(128, generator=torch.Generator().manual_seed(k))
        rnd.append(play_ablated(model, (v / v.norm()).unsqueeze(0)))
    print(f"  随机方向对照（5 次）        {np.mean(rnd):6.2f} 个豆"
          f"   （{min(rnd):.1f} ~ {max(rnd):.1f}）")
    print(f"  ⚠️ 只有【显著低于】这个区间，才算真信号\n", flush=True)

    # ⚠️ 消融【必须把 ③ 类也测了】。
    #    只按 R² 排序挑前几名的话，选出来的全是 ①② 类 ——
    #    那样就回答不了「有没有【新】知识在用」：③ 类 R² 低【不代表】没在用，
    #    也可能只是非线性可读。唯一能分辨的办法就是把它的方向也扣掉试试。
    top12 = [r for r in sorted(rows, key=lambda r: -r[3]) if r[1] != "③ 人没给过"][:5]
    new3 = [r for r in rows if r[1] == "③ 人没给过"]
    picks = top12 + new3
    print(f"  {'扣掉的方向':<20} {'类别':<11} {'线性R²':>8} {'非线性R²':>9} "
          f"{'消融后':>9} {'掉幅':>8}  判读")
    print("  " + "-" * 88)
    lo_null = min(rnd)
    for label, group, best, r_cnn, n_cnn, y, idx in picks:
        Xtr = (acts[tr] - acts[tr].mean(0)) / (acts[tr].std(0) + 1e-8)
        ym = y[tr].mean()
        w = np.linalg.solve(Xtr.T @ Xtr + 1.0 * np.eye(128), Xtr.T @ (y[tr] - ym))
        w_hat = torch.as_tensor(w / (np.linalg.norm(w) + 1e-12), dtype=torch.float32).unsqueeze(0)
        sc = play_ablated(model, w_hat)
        drop = base - sc
        verdict = "★ 在用" if sc < lo_null - 3 else ("· 疑似" if sc < lo_null else "无信号")
        print(f"  {label:<20} {group:<11} {r_cnn:>8.3f} {n_cnn:>9.3f} {sc:>9.2f} "
              f"{drop:>+8.2f}  {verdict}", flush=True)

    print()
    print("=" * 100)
    print("读法（⚠️ 三条都容易读错）")
    print("=" * 100)
    print("  · 线性 R² 低【不能】推出「信息不在」—— 要看非线性那一列。")
    print("      线性低 + 非线性高  = 信息在，但不是线性可读的")
    print("      线性低 + 非线性也低 = 信息大概率真的不在")
    print("  · 原始网格那一列【不是对照组】。'饥饿=1.0' 是因为网格里真有常数平面；")
    print("      但 'flood fill≈0.85' 不代表网格编码了 BFS —— 它只是和蛇长/头位置高度相关，")
    print("      而那两个本来就线性可读。【相关 ≠ 计算】")
    print("  · 探针 R² 高 ≠ CNN 在用。只有【消融掉幅显著超过随机对照】才算真在用。")
    print("  · ③ 类（人没给过）的 R² 才是「新知识」的证据。")
