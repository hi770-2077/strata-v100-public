#!/usr/bin/env python3
"""short-read 路径判别测试: 两轮对话, 第二轮只发几个 fresh tokens
第一轮建立对话缓存, 第二轮测 fresh read 耗时 (short-read 开关的真正作用域)
环境变量: STRATA_KEY(必填,无默认值) / STRATA_URL / STRATA_MODEL 可覆盖"""
import json, os, sys, time, urllib.request

KEY = os.environ.get("STRATA_KEY", "")
if not KEY:
    sys.exit("STRATA_KEY env required (this repo ships no default key)")
URL = os.environ.get("STRATA_URL", "http://127.0.0.1:8081/v1/chat/completions")
MODEL = os.environ.get("STRATA_MODEL", "qwen3.8-flash-next-gsq-rco-iq3_s-256k")

def chat(messages, max_tokens, tag):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    u = d.get("usage", {})
    print(f"[{tag}] prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} total={dt:.2f}s")
    return dt

label = sys.argv[1] if len(sys.argv) > 1 else "run"
print(f"=== shortread-disc {label} @ {time.strftime('%H:%M:%S')} ===")
# 第一轮: 中等长度 prompt + 短生成 (建立 conversation cache)
chat([{"role": "user", "content": "请详细解释一下 MoE 模型里专家缓存的设计原理和显存管理策略，越详细越好。"}], 64, "turn1")
time.sleep(1)
# 第二轮: 只发 10 个汉字左右 -> fresh tokens 极少, 走 short-read 路径
for i in range(3):
    chat([
        {"role": "user", "content": "请详细解释一下 MoE 模型里专家缓存的设计原理和显存管理策略，越详细越好。"},
        {"role": "assistant", "content": "专家缓存的核心是把高频路由的专家权重常驻显存，低频专家留在主机内存按需换入。"},
        {"role": "user", "content": "那换入的开销怎么控制？"},
    ], 64, f"turn2-{i}")
    time.sleep(1)
