#!/usr/bin/env python3
"""llama.cpp 挖矿后的 Strata 速度 A/B 基准 v2 (2026-10-13)
用法: bench_strata.py [label] [repeats]
先 warmup, 然后每项测 repeats 次取最佳 (本对话自身推理占卡, 取最佳接近无干扰值)
环境变量: STRATA_KEY(必填,无默认值) / STRATA_URL / STRATA_MODEL 可覆盖"""
import json, os, sys, time, urllib.request

KEY = os.environ.get("STRATA_KEY", "")
if not KEY:
    sys.exit("STRATA_KEY env required (this repo ships no default key)")
URL = os.environ.get("STRATA_URL", "http://127.0.0.1:8081/v1/chat/completions")
MODEL = os.environ.get("STRATA_MODEL", "qwen3.8-flash-next-gsq-rco-iq3_s-256k")

LONG_PROMPT = (
    "请用中文写一篇关于分布式推理引擎 MoE 专家缓存设计的详细技术分析，"
    "内容包括：专家路由的 top-k 选择策略、显存与主机内存的分层管理、"
    "页面锁定内存的预取机制、PCIe 传输带宽的利用效率、以及双 GPU 层切分下的"
    "流水线窗口设计。要求结构清晰、论证严谨、给出具体的性能数据示例。"
    "另外请讨论一下在 Tesla V100 这类旧架构卡上做 INT8/INT4 量化的取舍。"
)

def chat(messages, max_tokens, tag, retries=2):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": False,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            dt = time.time() - t0
            usage = d.get("usage", {})
            ct = usage.get("completion_tokens", 0)
            pt = usage.get("prompt_tokens", 0)
            return dt, pt, ct, (ct / dt if dt > 0 else 0)
        except Exception as e:
            print(f"[{tag}] attempt {attempt+1} failed: {e}")
            time.sleep(10)
    return None, 0, 0, 0

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    print(f"=== bench {label} @ {time.strftime('%H:%M:%S')} ===")
    # warmup (触发专家缓存/对话缓存热 path)
    chat([{"role": "user", "content": "你好"}], 8, "warmup")
    best = {}
    for rep in range(reps):
        for name, fn in [
            ("decode-160", lambda: chat([{"role": "user", "content": LONG_PROMPT}], 160, f"decode-160#{rep}")),
            ("short-read", lambda: chat([{"role": "user", "content": "你好，请介绍一下你自己。"}], 8, f"short-read#{rep}")),
            ("long-gen-512", lambda: chat([{"role": "user", "content": LONG_PROMPT}], 512, f"long-gen-512#{rep}")),
        ]:
            dt, pt, ct, tps = fn()
            print(f"[{label}] {name}: prompt={pt} completion={ct} total={dt:.2f}s -> {tps:.1f} tok/s")
            if tps > best.get(name, (0,))[0]:
                best[name] = (tps, dt, pt, ct)
    print(f"=== BEST {label} ===")
    for k, (tps, dt, pt, ct) in best.items():
        print(f"  {k}: {tps:.1f} tok/s (prompt={pt} completion={ct} total={dt:.2f}s)")
