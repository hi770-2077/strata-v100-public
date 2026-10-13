#!/usr/bin/env python3
"""
bench-production.py — Benchmark a live Strata server and record raw results.

Rates come from the engine's own `timings` object in the response, not from
dividing token counts by wall time:

  prompt_per_second      prefill rate (tok/s)
  predicted_per_second   decode rate (tok/s)
  prompt_ms / predicted_ms
  draft_n / draft_n_accepted   MTP speculative sampling: drafts and accepted.
                              A high accept rate means MTP is paying off.

Why per-run prompt salting matters: the server caches prefixes, so re-sending
an identical prompt makes `prompt_tokens` collapse (e.g. 14876 -> 5) and the
reported prefill rate meaningless. Every run here gets a unique salt prefix.

Usage:
  export STRATA_API_KEY=...        # never hardcode the key
  ./bench-production.py --out prod-bench.json --rounds 3

Environment overrides:
  STRATA_ENDPOINT         default http://127.0.0.1:8080/v1/chat/completions
  STRATA_PROXY_ENDPOINT   default http://127.0.0.1:8081/v1/chat/completions
"""
import argparse, json, os, statistics, sys, time, urllib.request

ENGINE = os.environ.get("STRATA_ENDPOINT",
                        "http://127.0.0.1:8080/v1/chat/completions")
PROXY  = os.environ.get("STRATA_PROXY_ENDPOINT",
                        "http://127.0.0.1:8081/v1/chat/completions")
# Never hardcode a key. Export STRATA_API_KEY before running.
KEY    = os.environ.get("STRATA_API_KEY", "")


def health(port):
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=10) as r:
            return json.load(r)
    except Exception as e:
        return {"error": str(e)}


def gpu_state():
    try:
        out = os.popen(
            "nvidia-smi --query-gpu=name,memory.used,memory.total,"
            "utilization.gpu,temperature.gpu,power.draw "
            "--format=csv,noheader,nounits").read().strip()
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception as e:
        return [f"ERR {e}"]


def synth_prompt(n_tokens, salt=""):
    unit = ("以下是一段用于测速的合成文本。The quick brown fox jumps over the lazy dog. "
            "本段用于把 prefill 阶段的输入长度推到目标 token 数。0123456789 ")
    body = unit * max(1, n_tokens // 40)
    # salt 必不可少：服务端有前缀缓存，prompt 一致时第 2、3 轮会命中缓存，
    # prompt_tokens 掉到 5、prefill 数字作废（我第一次实测就踩了这个坑）。
    return f"[run-{salt}]\n" + body if salt else body


def one_run(url, prompt, max_tokens):
    body = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KEY}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    wall = time.perf_counter() - t0

    t = d.get("timings") or {}
    u = d.get("usage") or {}
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    return {
        "wall_s": round(wall, 3),
        "prompt_tokens": t.get("prompt_n") or u.get("prompt_tokens"),
        "output_tokens": t.get("predicted_n") or u.get("completion_tokens"),
        "prefill_tok_s": t.get("prompt_per_second"),
        "decode_tok_s": t.get("predicted_per_second"),
        "prefill_ms": t.get("prompt_ms"),
        "decode_ms": t.get("predicted_ms"),
        "cached_tokens": t.get("cache_n"),
        "mtp_draft_n": dn,
        "mtp_accepted": da,
        "mtp_accept_rate": (round(da / dn, 3) if dn else None),
    }


def series(url, prompt_fn, max_tokens, rounds, tag=""):
    """prompt_fn(i) 每轮返回不同 prompt，绕开前缀缓存。"""
    rows = []
    for i in range(rounds):
        try:
            r = one_run(url, prompt_fn(i), max_tokens)
            r["round"] = i + 1
            rows.append(r)
            print(f"  {tag} r{i+1}: prefill={r['prefill_tok_s']} "
                  f"decode={r['decode_tok_s']} tok/s | prompt={r['prompt_tokens']} "
                  f"out={r['output_tokens']} mtp={r['mtp_accept_rate']}")
        except Exception as e:
            print(f"  {tag} r{i+1}: FAILED {e}")
        time.sleep(1)
    return rows


def med(rows, key):
    v = [r[key] for r in rows if r.get(key)]
    return round(statistics.median(v), 2) if v else None


def summarize(rows):
    return {
        "prefill_tok_s_median": med(rows, "prefill_tok_s"),
        "decode_tok_s_median": med(rows, "decode_tok_s"),
        "prefill_tok_s_all": [r["prefill_tok_s"] for r in rows],
        "decode_tok_s_all": [r["decode_tok_s"] for r in rows],
        "mtp_accept_rate_median": med(rows, "mtp_accept_rate"),
        "prompt_tokens": rows[0]["prompt_tokens"] if rows else None,
        "output_tokens": rows[0]["output_tokens"] if rows else None,
        "wall_s_median": med(rows, "wall_s"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prod-bench.json")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    if not KEY:
        sys.exit("export STRATA_API_KEY=<your key> first "
                 "(do not hardcode it in this script or in a committed file)")

    rep = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "health_8080": health(8080),
        "health_8081": health(8081),
        "gpu_before": gpu_state(),
        "suites": {},
    }

    stamp = time.strftime("%H%M%S")
    suites = [
        ("decode_steady", "短 prompt + 512 输出：稳态 decode",
         lambda i: f"[run-{stamp}-d{i}] 用一句话解释什么是 KV cache。", 512),
        ("longgen_2k_1024", "2K prompt + 1024 输出：长回答场景",
         lambda i: synth_prompt(2000, f"{stamp}-g{i}"), 1024),
        ("prefill_13k", "12.9K prompt + 64 输出：prefill 主导",
         lambda i: synth_prompt(12900, f"{stamp}-p{i}"), 64),
        ("prefill_32k", "32K prompt + 64 输出：长上下文 prefill",
         lambda i: synth_prompt(32000, f"{stamp}-q{i}"), 64),
    ]
    for key, desc, pfn, mt in suites:
        print(f"== {key}: {desc} ==")
        rows = series(ENGINE, pfn, mt, a.rounds, tag=key)
        rep["suites"][key] = {"desc": desc, "summary": summarize(rows),
                              "runs": rows}

    # 代理前后对比（8081 前缀复用：同一 prompt 连打，第 2+ 轮命中缓存）
    print("== proxy_compare: 8080 直连 vs 8081 前缀复用代理 ==")
    salt = f"{stamp}-cmp"
    p = synth_prompt(12900, salt)
    d = {"desc": "同一 12.9K prompt 重复 3 次，8081 应命中前缀缓存",
         "engine_8080": series(ENGINE, lambda i: p, 64, a.rounds, "8080"),
         "proxy_8081": series(PROXY, lambda i: p, 64, a.rounds, "8081")}
    s0, s1 = summarize(d["engine_8080"]), summarize(d["proxy_8081"])
    d["summary_8080"] = s0
    d["summary_8081"] = s1
    if s0["prefill_tok_s_median"] and s1["prefill_tok_s_median"]:
        d["prefill_speedup"] = round(
            s1["prefill_tok_s_median"] / s0["prefill_tok_s_median"], 2)
    rep["suites"]["proxy_compare"] = d

    rep["gpu_after"] = gpu_state()
    with open(a.out, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("\nWROTE", a.out)


if __name__ == "__main__":
    main()
