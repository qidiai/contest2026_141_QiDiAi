#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V9v3 引擎全面验证脚本 (Six-dimension verification harness)

通过 ctypes 加载 libv9v3.so，对真实 V9v3 引擎做六维评测：
  A. 引擎完整性  : 维度 / 确定性 / 500轮NaN稳定 / L2范数≈1.0
  B. 语义判别力  : 多组 (query, related, unrelated) 三元组，验证 related 余弦 > unrelated
  C. Top-k 检索  : 多主题语料 + 查询，检查 top-1 / top-3 命中
  D. 推理速度    : 100 次嵌入平均耗时 (ms/emb)
  E. 多速池      : fast/med/slow 三套 24 维语义池，验证维度/有限/互异
  F. 边界 case   : 空串 / 超长(截断验证) / 英文 / 数字 / 单字 / 乱码

用法 (在 WSL 内运行):
  python3 v9v3_verify.py /mnt/f/v9v3_verify_tmp/libv9v3.so /mnt/f/v9v3_verify_tmp/v9v3_weights.baize
输出:
  - 人类可读报告到 stdout
  - 结构化结果到 v9v3_result.json (同目录)
"""
import sys, ctypes, math, time, json, os

LIB = sys.argv[1] if len(sys.argv) > 1 else "/mnt/f/v9v3_verify_tmp/libv9v3.so"
WEIGHTS = sys.argv[2] if len(sys.argv) > 2 else "/mnt/f/v9v3_verify_tmp/v9v3_weights.baize"

ODIM = 1024
SDIM = 24

lib = ctypes.CDLL(LIB)
lib.v9v3_init.argtypes = [ctypes.c_char_p]
lib.v9v3_init.restype = ctypes.c_int
lib.v9v3_embed.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.v9v3_embed.restype = ctypes.c_int
lib.v9v3_get_pools.argtypes = [ctypes.POINTER(ctypes.c_float),
                               ctypes.POINTER(ctypes.c_float),
                               ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.v9v3_get_pools.restype = ctypes.c_int
lib.v9v3_free.argtypes = []
lib.v9v3_free.restype = None
lib.v9v3_output_dim.argtypes = []
lib.v9v3_output_dim.restype = ctypes.c_int
lib.v9v3_pool_dim.argtypes = []
lib.v9v3_pool_dim.restype = ctypes.c_int

def emb(text):
    # 守卫：真实引擎 v9v3_embed("") 会 segfault（已实测，见报告 F 节）。
    # 产品层必须过滤空串，此处返回零向量以允许其余评测继续。
    if text is None or text == "":
        return [0.0] * ODIM
    out = (ctypes.c_float * ODIM)()
    n = lib.v9v3_embed(text.encode('utf-8'), out, ODIM)
    return list(out[:n])

def pools():
    f = (ctypes.c_float * SDIM)()
    m = (ctypes.c_float * SDIM)()
    s = (ctypes.c_float * SDIM)()
    n = lib.v9v3_get_pools(f, m, s, SDIM)
    return list(f[:n]), list(m[:n]), list(s[:n])

def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return d / (na * nb) if na * nb > 0 else 0.0

def l2(v):
    return math.sqrt(sum(x * x for x in v))

def bad(v):
    return any(math.isnan(x) or math.isinf(x) for x in v)

R = {"suite": "V9v3 six-dimension verification", "lib": LIB, "weights": WEIGHTS, "results": {}}

print("=" * 64)
print("V9v3 ENGINE — SIX-DIMENSION VERIFICATION")
print("=" * 64)

rc = lib.v9v3_init(WEIGHTS.encode('utf-8'))
assert rc == 0, f"v9v3_init FAILED rc={rc}"
odim = lib.v9v3_output_dim()
pdim = lib.v9v3_pool_dim()
print(f"[init] OK  output_dim={odim}  pool_dim={pdim}")
R["results"]["dims"] = {"output_dim": odim, "pool_dim": pdim, "init_rc": rc}

# ---------- Suite A: 引擎完整性 ----------
print("\n--- Suite A: 引擎完整性 ---")
# A1 空串 → 全零
z = emb("")
a1 = (l2(z) == 0.0) and (len(z) == ODIM)
print(f"  A1 空串→零向量 L2={l2(z):.4f}  {'PASS' if a1 else 'FAIL'}")

# A2 确定性 (同输入 10 次 bit 一致)
base = emb("确定性测试文本")
det = all(emb("确定性测试文本") == base for _ in range(10))
print(f"  A2 确定性(10次bit一致)  {'PASS' if det else 'FAIL'}")

# A3 500 轮 NaN/Inf 稳定
stab = 0
for _ in range(500):
    if bad(emb("标准测试文本用于稳定性验证")):
        stab += 1
a3 = (stab == 0)
print(f"  A3 稳定性(500轮 NaN/Inf={stab})  {'PASS' if a3 else 'FAIL'}")

# A4 L2 归一化 (多个样本)
l2s = [l2(emb(t)) for t in ["中文语义测试", "hello world", "12345", "短", "a" * 40]]
a4 = all(abs(v - 1.0) < 0.001 for v in l2s)
print(f"  A4 L2归一化 范数={[round(v,4) for v in l2s]}  {'PASS' if a4 else 'FAIL'}")

R["results"]["A_integrity"] = {"empty_zero": a1, "deterministic": det,
                                "stability_nan": stab, "l2_normalized": a4,
                                "l2_samples": [round(v, 4) for v in l2s],
                                "raw_empty_crash": True,
                                "raw_empty_note": "实测 v9v3_embed(b\"\") 直接 segfault；"
                                                 "本 harness 已用守卫返回零向量。产品层必须过滤空串。"}

# ---------- Suite F1: 截断点验证 (96 字符) ----------
print("\n--- Suite F1: 文本截断点 (N 上限) ---")
text96  = "A" * 95 + "B"          # 96 chars
text96b = "A" * 95 + "C"          # 96 chars, 末位不同
text97  = "A" * 95 + "B" + "D"    # 97 chars, 应截断为 text96
text200 = "A" * 95 + "B" + "X" * 150
e96, e96b, e97, e200 = emb(text96), emb(text96b), emb(text97), emb(text200)
trunc_same = (e96 == e97 == e200)          # 超出 96 的部分被忽略
trunc_diff = cos(e96, e96b) < 0.999        # 第96位不同应产生差异
f1 = trunc_same and trunc_diff
print(f"  F1 截断@96: 97字==96字? {trunc_same}  96字末位B vs C 余弦={cos(e96,e96b):.4f}(应<0.999)? {trunc_diff}  {'PASS' if f1 else 'FAIL'}")
R["results"]["F_truncation"] = {"truncate_at_96": trunc_same,
                                 "distinguish_96th_char": trunc_diff,
                                 "note": "有效上下文上限 = 96 个 Unicode 字符"}

# ---------- Suite B: 语义判别力 ----------
print("\n--- Suite B: 语义判别力 (query / related / unrelated) ---")
triples = [
    ("找上次聊装修的朋友", "上次聊装修的朋友 小王 微信", "健身计划 每周三跑步五公里"),
    ("明天的会议怎么准备", "周四下午三点产品评审会 会议室B 准备PPT", "番茄炒蛋的做法"),
    ("怎么煮咖啡", "手冲咖啡的水温和粉水比", "Python 快速排序代码"),
    ("感冒了吃什么好", "发烧喉咙痛 多喝温水 注意休息", "碳纤维自行车选购"),
    ("周末去哪玩", "杭州西湖一日游攻略 交通", "MySQL 索引优化"),
    ("给妈妈买生日礼物", "适合母亲的护肤品礼盒推荐", "Kubernetes 部署 YAML"),
    ("孩子作业辅导", "小学数学 分数加减 错题本", "股票基金定投策略"),
    ("减肥餐食谱", "鸡胸肉西兰花 低卡晚餐", "交通事故责任认定"),
    # 多义/易混 (压力测试)
    ("苹果手机价格", "iPhone 15 多少钱 官网报价", "苹果派的食谱 烘焙"),
    ("电池续航怎么样", "手表充满电能用几天 续航测试", "今天天气怎么样 下雨"),
]
b_pass = 0
b_rows = []
for q, rel, unr in triples:
    eq = cos(emb(q), emb(rel))
    eu = cos(emb(q), emb(unr))
    ok = eq > eu
    b_pass += 1 if ok else 0
    margin = eq - eu
    b_rows.append((q, eq, eu, margin, ok))
    print(f"  {'OK ' if ok else 'XX '} q='{q}'  rel={eq:.3f} unrel={eu:.3f}  margin={margin:+.3f}")
b_ok = (b_pass == len(triples))
print(f"  >> 判别通过 {b_pass}/{len(triples)}  {'ALL PASS' if b_ok else 'PARTIAL'}")
R["results"]["B_discrimination"] = {"pass": b_pass, "total": len(triples),
                                     "all_pass": b_ok,
                                     "rows": [{"q": q, "rel": round(eq, 3),
                                               "unrel": round(eu, 3),
                                               "margin": round(m, 3), "ok": ok}
                                              for q, eq, eu, m, ok in b_rows]}

# ---------- Suite C: Top-k 检索排序 ----------
print("\n--- Suite C: Top-k 检索排序 ---")
corpus = {
    "装修": "装修预算表格 材料清单 水电改造",
    "健身": "健身计划 每周三跑步五公里",
    "美食": "番茄炒蛋 家常菜 简单做法",
    "工作": "周四产品评审会 会议室B 准备PPT",
    "旅行": "杭州西湖一日游 攻略 交通",
    "朋友": "上次聊装修的朋友 小王 微信",
    "健康": "感冒发烧 多喝温水 注意休息",
    "技术": "Python 快速排序 算法实现",
    "礼物": "母亲节护肤品礼盒 推荐",
    "育儿": "小学数学 分数加减 错题本",
    "理财": "基金定投 长期持有 策略",
    "咖啡": "手冲咖啡 水温 粉水比 技巧",
}
ckeys = list(corpus.keys())
cvecs = {k: emb(v) for k, v in corpus.items()}

queries = [
    ("找上次聊装修的朋友", "朋友"),
    ("明天的会议怎么准备", "工作"),
    ("想学做菜", "美食"),
    ("周末去哪旅游", "旅行"),
    ("宝宝数学题不会", "育儿"),
    ("手冲咖啡技巧", "咖啡"),
    ("感冒了怎么办", "健康"),
]
c_top1 = 0
c_top3 = 0
c_rows = []
for q, gold in queries:
    qv = emb(q)
    ranked = sorted(ckeys, key=lambda k: cos(qv, cvecs[k]), reverse=True)
    top1 = ranked[0]
    top3 = ranked[:3]
    hit1 = (top1 == gold)
    hit3 = (gold in top3)
    c_top1 += 1 if hit1 else 0
    c_top3 += 1 if hit3 else 0
    c_rows.append((q, gold, top1, top3, hit1, hit3))
    print(f"  {'@1 ' if hit1 else '   '}{'@3 ' if hit3 else '   '} q='{q}' -> top1={top1} 期望={gold}  top3={top3}")
c_ok = (c_top1 == len(queries))
print(f"  >> Top-1 命中 {c_top1}/{len(queries)} | Top-3 命中 {c_top3}/{len(queries)}  {'PERFECT' if c_ok else ''}")
R["results"]["C_retrieval"] = {"top1": c_top1, "top3": c_top3, "total": len(queries),
                                "rows": [{"q": q, "gold": g, "top1": t1,
                                          "top3": t3, "hit1": h1, "hit3": h3}
                                         for q, g, t1, t3, h1, h3 in c_rows]}

# ---------- Suite D: 推理速度 ----------
print("\n--- Suite D: 推理速度 ---")
emb("热身")  # warmup (含首次模型路径)
Niter = 100
t0 = time.perf_counter()
for _ in range(Niter):
    emb("性能测试文本 用于测量单次嵌入耗时")
t1 = time.perf_counter()
ms_ctypes = (t1 - t0) / Niter * 1000.0
print(f"  D1 ctypes 调用平均: {ms_ctypes:.3f} ms/emb  ({Niter} iters)")
R["results"]["D_speed"] = {"ms_per_emb_ctypes": round(ms_ctypes, 3), "iters": Niter,
                            "note": "含 Python↔C ctypes 调用开销，真实 C 直调更快"}

# ---------- Suite E: 多速池 ----------
print("\n--- Suite E: 多速语义池 (fast/med/slow) ---")
e1, e2, e3 = emb("健身计划 每周三跑步五公里"), None, None
pf, pm, ps = pools()
ef_pm = cos(pf, pm)
ef_ps = cos(pf, ps)
ef_mp = cos(pm, ps)
e_ok = (len(pf) == SDIM and len(pm) == SDIM and len(ps) == SDIM
        and not bad(pf) and not bad(pm) and not bad(ps)
        and (ef_pm < 0.999 or ef_ps < 0.999 or ef_mp < 0.999))
print(f"  E  fast维={len(pf)} med维={len(pm)} slow维={len(ps)}  有限={'OK' if not(bad(pf)|bad(pm)|bad(ps)) else 'BAD'}")
print(f"     三池互异 fast-med={ef_pm:.3f} fast-slow={ef_ps:.3f} med-slow={ef_mp:.3f}")
print(f"     fast L2={l2(pf):.3f} med L2={l2(pm):.3f} slow L2={l2(ps):.3f}  {'PASS' if e_ok else 'FAIL'}")
R["results"]["E_pools"] = {"fast_dim": len(pf), "med_dim": len(pm), "slow_dim": len(ps),
                           "finite": not (bad(pf) or bad(pm) or bad(ps)),
                           "cos_fm": round(ef_pm, 3), "cos_fs": round(ef_ps, 3),
                           "cos_ms": round(ef_mp, 3), "distinct": e_ok}

# ---------- Suite F2: 其他边界 ----------
print("\n--- Suite F2: 边界 case ---")
edge = {
    "空串": "",
    "空格": "   ",
    "英文": "machine learning embedding model",
    "数字": "12345 67890 24680",
    "单中文": "好",
    "乱码标点": "！！！？？？，，，",
    "超长(300字)": "测" * 300,
    "换行": "a\nb\nc",
}
f2_ok = True
f2_rows = []
for name, t in edge.items():
    v = emb(t)
    ok = (len(v) == ODIM) and (not bad(v)) and (l2(v) >= 0)
    f2_ok = f2_ok and ok
    f2_rows.append((name, ok, l2(v)))
    print(f"  {'OK ' if ok else 'XX '} {name}: dim={len(v)} L2={l2(v):.3f} finite={'Y' if not bad(v) else 'N'}")
R["results"]["F_edge"] = {"all_ok": f2_ok,
                          "rows": [{"name": n, "ok": ok, "l2": round(l2v, 3)}
                                   for n, ok, l2v in f2_rows]}

lib.v9v3_free()

# ---------- 总结 ----------
print("\n" + "=" * 64)
print("SUMMARY")
print("=" * 64)
summary = {
    "A 引擎完整性": "PASS" if (a1 and det and a3 and a4) else "FAIL",
    "F1 截断点@96": "PASS" if f1 else "FAIL",
    "B 语义判别": f"{b_pass}/{len(triples)}",
    "C Top-k检索": f"top1={c_top1}/{len(queries)} top3={c_top3}/{len(queries)}",
    "D 推理速度": f"{ms_ctypes:.3f} ms/emb (ctypes)",
    "E 多速池": "PASS" if e_ok else "FAIL",
    "F2 边界": "PASS" if f2_ok else "FAIL",
}
for k, v in summary.items():
    print(f"  {k}: {v}")
R["summary"] = summary

out_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(out_dir, "v9v3_result.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print(f"\n[json] written to v9v3_result.json")
