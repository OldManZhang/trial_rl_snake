"""
模型注册表 —— 【全仓库唯一】认 checkpoint 的地方

================================================================
【为什么要有这个文件】

    以前这件事散在 4 个地方，每加一个模型都要挨个改，漏一个不报错：

        ① ppo.py::load_for_view    if obs_mode == "grid": ...   ← 住在"算法文件"里
        ② watch.py                  if len(shape) == 3: ...      打印说明
        ③ demo_server.py            SIDES = [...]                名字/副标题/ckpt
        ④ demo.html                 ACCENT = {...}               颜色

    现在收成一处：想加模型 → 写一个 models/xxx.py + 在下面 REGISTRY 加一行。

【一个模型文件要提供什么】

    NAME              字符串，注册名
    make(cfg)         → (env, net)      建环境和网络
    describe(cfg)     → str             一句话简介（watch.py / 网页用）
    save_meta(cfg)    → dict            存 checkpoint 时写什么

    cfg 就是一个 dict，至少含 {"env_kw": {...}}，可能还有 arch / obs_shape / obs_dim。

【checkpoint 格式的兼容】

    ⚠️ 已经训好的 4 个模型【必须继续能读】—— 59.12 那版是文章的核心数据，不能白训。
       所以 normalize() 要把三种历史格式都归一：

        新格式  {"model": "cnn", ...}
        旧格式  {"obs_mode": "grid", "arch": "cnn", ...}      → cnn
        更旧    {"env_kw": {...}, "obs_dim": 18}              → feature（当年只有这一种）
        最旧    裸 state_dict                                 → feature 12 维

================================================================
"""

import os
import sys

# ⚠️ 直接 `uv run python models/__init__.py` 跑自测时，Python 只把【脚本所在目录】
#    （也就是 models/）加进 sys.path —— 而下面要 import 的是【包】models，
#    得让 models/ 的上一级在路径上。补一条，两种跑法就都能用。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import feature, cnn

# ⭐ 加模型 = 加一行
REGISTRY = {
    feature.NAME: feature,
    cnn.NAME: cnn,
}

DEFAULT_CKPT = "snake_both.pth"      # 目前最好的一版：feature，18 维，终测 59.12 个豆

# 历史的 obs_mode 写法 → 现在的注册名
_ALIAS = {"features": "feature", "feat": "feature", "feature": "feature",
          "grid": "cnn", "cnn": "cnn"}

# 网格模型出现的【之前】，特征版是唯一的选择；它的默认环境长这样
_LEGACY_FEATURE_ENV = {"use_space": True, "use_deep": False, "use_tail": False}


def normalize(ckpt):
    """把任意历史格式的 checkpoint 归一成 cfg。"""
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
        # 最旧：裸 state_dict
        print("⚠️ 旧格式 checkpoint（没记录环境配置），按 feature 12 维默认环境跑")
        return {"model": "feature", "env_kw": dict(_LEGACY_FEATURE_ENV), "obs_dim": 12}

    cfg = dict(ckpt)
    cfg.pop("state_dict", None)

    # 新格式写了 model；旧格式写的是 obs_mode
    name = cfg.get("model") or _ALIAS.get(cfg.get("obs_mode", "features"))
    if name not in REGISTRY:
        sys.exit(f"❌ checkpoint 里的模型名 {name!r} 不认识\n"
                 f"   认识的只有：{', '.join(REGISTRY)}")
    cfg["model"] = name
    return cfg


def make(cfg):
    """按 cfg 建 (env, net)。注册表的核心一行。"""
    return REGISTRY[cfg["model"]].make(cfg)


def describe(cfg):
    return REGISTRY[cfg["model"]].describe(cfg)


def save_meta(cfg):
    """train.py 存 checkpoint 时调它，拿到该写进文件的元信息。"""
    return REGISTRY[cfg["model"]].save_meta(cfg)


def ui(cfg):
    """网页展示用：{key, name, subtitle, color}。见各模型文件里的 LABEL/SUB/COLOR。"""
    return REGISTRY[cfg["model"]].ui(cfg)


def load_for_view(path=None):
    """读 checkpoint → (env, model)。找不到文件直接退出并给出下一步。

    ⚠️ 必须按【模型自带的配置】建环境 —— 用错了不会报错，只会【静默玩得很烂】。
       feature 的 15 维还有歧义：+2步前瞻 和 +蛇尾可达 的特征顺序不同，
       光看维度猜不出来。cnn 更没法猜（(4,8,8) 和 (18,) 是两种完全不同的输入）。
       所以配置必须跟权重存一起。

    ⚠️ 权重可能是用别的设备训的，看的时候一律搬回 CPU ——
       看是交互式的，慢一点无所谓，别搞出设备不匹配。

    想看 cfg（比如要调 describe()）就用 load_all()，省一次读盘。
    """
    env, model, _ = load_all(path)
    return env, model


def load_all(path=None):
    """→ (env, model, cfg)。watch.py 要 cfg 才能问 describe()。"""
    path = path or DEFAULT_CKPT
    try:
        ckpt = torch.load(path, weights_only=True)
    except FileNotFoundError:
        sys.exit(f"❌ 找不到 {path}\n   先在这个目录下跑：uv run python train.py")

    cfg = normalize(ckpt)
    env, model = make(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to("cpu")
    model.eval()
    return env, model, cfg


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    ok = 0

    def check(name, cond, extra=""):
        global ok
        assert cond, f"❌ {name}  {extra}"
        ok += 1
        print(f"  ✅ {name}{('  ' + extra) if extra else ''}")

    print("=" * 78)
    print("models 自测 · 注册表 + 4 个已训好模型的兼容性")
    print("=" * 78)

    print("\n[1] 注册表")
    check("两个模型都注册了", set(REGISTRY) == {"feature", "cnn"}, str(sorted(REGISTRY)))
    for name, mod in REGISTRY.items():
        check(f"{name}: 四件套齐全",
              all(hasattr(mod, k) for k in ("NAME", "make", "describe", "save_meta")))
        check(f"{name}: NAME 和注册键一致", mod.NAME == name)

    print("\n[2] normalize —— 三种历史格式")
    check("新格式 {model: cnn}",
          normalize({"state_dict": {}, "model": "cnn"})["model"] == "cnn")
    check("旧格式 {obs_mode: grid} → cnn",
          normalize({"state_dict": {}, "obs_mode": "grid"})["model"] == "cnn")
    check("旧格式 {obs_mode: features} → feature",
          normalize({"state_dict": {}, "obs_mode": "features"})["model"] == "feature")
    check("更旧 {env_kw, obs_dim} → feature",
          normalize({"state_dict": {}, "env_kw": {}, "obs_dim": 18})["model"] == "feature")
    check("最旧 裸 state_dict → feature 12 维",
          normalize({"a": 1})["obs_dim"] == 12)

    print("\n[3] ⭐ 4 个已训好的模型全部能读（这是重构的验收线）")
    import os
    for f, want_name, note in [
            ("snake_both.pth", "feature", "18 维，终测 59.12"),
            ("snake_long.pth", "feature", "12 维，终测 34.84"),
            ("snake_e2e_mlp.pth", "cnn", "网格+MLP 对照"),
            ("snake_e2e_cnn.pth", "cnn", "网格+CNN，终测 50.82")]:
        if not os.path.exists(f):
            check(f"{f} 存在", False, "文件不在？")
            continue
        env, model = load_for_view(f)
        shape = env.observation_space.shape
        check(f"{f:<20} → {want_name:<8} 输入 {str(shape):<12} {note}",
              model.training is False)

    print("\n[4] 加载出来的模型能真的跑一局")
    import numpy as np
    env, model = load_for_view("snake_both.pth")
    o, _ = env.reset(seed=777)
    steps = 0
    while True:
        with torch.no_grad():
            probs, _ = model(torch.as_tensor(np.asarray(o), dtype=torch.float32).unsqueeze(0))
        o, r, te, tr, info = env.step(int(probs.argmax(-1)))
        steps += 1
        if te or tr:
            break
    check("snake_both 跑完一局", steps > 3, f"{info['food_eaten']} 个豆 / {steps} 步")

    print("\n[5] 三种表示在同一个局面下都能建出来")
    for name in REGISTRY:
        cfg = {"model": name}
        if name == "cnn":
            cfg["arch"] = "cnn"
        e, n = make(cfg)
        o, _ = e.reset(seed=0)
        check(f"{name}: obs {np.asarray(o).shape} → 网络能前向",
              n(torch.as_tensor(np.asarray(o), dtype=torch.float32).unsqueeze(0))[0].shape == (1, 3))

    print("\n" + "=" * 78)
    print(f"✅ 全部 {ok} 条断言通过")
    print("=" * 78)
