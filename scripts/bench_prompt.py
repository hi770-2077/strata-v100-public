#!/usr/bin/env python3
"""pf_wmma A/B 基准 (2026-10-13): prompt-iso (长 prompt + max_tokens=1) + decode/long-gen 回归
用法: bench_prompt.py [label] [reps]
每次请求加唯一前缀, 避开对话缓存 (强制全量 fresh read)
环境变量: STRATA_KEY(必填,无默认值) / STRATA_URL / STRATA_MODEL 可覆盖"""
import json, os, sys, time, urllib.request

KEY = os.environ.get("STRATA_KEY", "")
if not KEY:
    sys.exit("STRATA_KEY env required (this repo ships no default key)")
URL = os.environ.get("STRATA_URL", "http://127.0.0.1:8081/v1/chat/completions")
MODEL = os.environ.get("STRATA_MODEL", "sc117-flash-next-abliterated-iq3_s-256k-vision")

PARA = ("请用中文写一篇关于分布式推理引擎 MoE 专家缓存设计的详细技术分析，"
        "内容包括：专家路由的 top-k 选择策略、显存与主机内存的分层管理、"
        "页面锁定内存的预取机制、PCIe 传输带宽的利用效率、以及双 GPU 层切分下的"
        "流水线窗口设计。要求结构清晰、论证严谨、给出具体的性能数据示例。"
        "另外请讨论一下在 Tesla V100 这类旧架构卡上做 INT8/INT4 量化的取舍，"
        "包括张量核心对 FP16 的支持情况、整数运算单元的吞吐瓶颈、以及反量化开销。")

def chat(content, max_tokens, tag, retries=2):
    body = json.dumps({
        "model": MODEL, "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            dt = time.time() - t0
            usage = d.get("usage", {})
            return dt, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        except Exception as e:
            print(f"[{tag}] attempt {attempt+1} failed: {e}")
            time.sleep(10)
    return None, 0, 0

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"=== bench {label} @ {time.strftime('%H:%M:%S')} ===")
    chat("你好", 8, "warmup")
    rows = []
    for rep in range(reps):
        # 唯一前缀 + 24 段长文 ≈ 数千 token; max_tokens=1 → 墙钟≈prompt read
        content = f"请求编号 ab{int(time.time())}-{rep}。\n" + PARA * 24
        dt, pt, ct = chat(content, 1, f"prompt-iso#{rep}")
        if dt and pt:
            print(f"[{label}] prompt-iso#{rep}: prompt={pt} 1tok total={dt:.2f}s -> {pt/dt:.1f} tok/s")
            rows.append((pt / dt, dt, pt))
    # decode 回归 (短 prompt + 160 tok)
    for rep in range(2):
        dt, pt, ct = chat(PARA, 160, f"decode-160#{rep}")
        if dt and ct:
            print(f"[{label}] decode-160#{rep}: prompt={pt} completion={ct} total={dt:.2f}s -> {ct/dt:.1f} tok/s")
    # long-gen 回归
    dt, pt, ct = chat(PARA, 512, "long-gen-512")
    if dt and ct:
        print(f"[{label}] long-gen-512: prompt={pt} completion={ct} total={dt:.2f}s -> {ct/dt:.1f} tok/s")
    if rows:
        best = max(rows)
        allv = [f"{r[0]:.1f}" for r in rows]
        print(f"=== BEST {label} prompt-iso: {best[0]:.1f} tok/s (prompt={best[2]} total={best[1]:.2f}s) all={allv} ===")
