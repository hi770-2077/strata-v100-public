# Swapping the engine: llama.cpp vs Strata on identical hardware and model

**Measured:** 2026-10-07, 2× Tesla V100-PCIE-32GB (sm_70), same GGUF file,
same machine, same day.

This is not a "which engine is better" page. It is a controlled experiment whose
most useful output was **not** the headline number — it was identifying where the
bottleneck actually was, which then explained several of our own tuning results.

## The headline

**Same model, same hardware, changing only the inference engine plus one tensor
placement: prefill went from 28.7 to 427.6 tok/s — a 14.9× difference.**

Decode (no speculative decoding) was 44.6 tok/s, against Strata's 64.8–71.5 with
`--spec 4` + MTP enabled. That decode gap is entirely explained by speculative
decoding being present on one side and absent on the other, not by engine
efficiency.

## Full measurements

### Strata (production baseline at the time)

| Metric | Value | Condition |
|---|---:|---|
| Cold prefill | **28.7 tok/s** (16 869 tokens / 588.7 s) | first API call of the session |
| TTFT | 0.36–0.64 s | after recovery |
| Decode | **64.8–71.5 tok/s** | `--spec 4` + `--mtp`, streaming |
| VRAM | 62.9 / 65.5 GB (96%) | `--kv-resident 32768` |

### llama.cpp (`3bb386e`, `--fused-moe` default on, compiled for sm_70)

| # | Configuration | PP512 | TG128 | Peak VRAM |
|---|---|---:|---:|---:|
| 1 | Default `-ngl 999` | 142.02 | 38.62 | 59 658 MiB |
| 2 | `--fit 1` auto | 325.85 | 17.84 | 56 606 MiB |
| 3 | embedding **and** output → CPU | 405.13 | 11.03 | 61 036 MiB |
| 4 | 3 + `--n-cpu-moe 0` | 392.81 | 10.72 | 61 036 MiB |
| **6** | **embedding → CPU only, output stays on GPU** | **410.50** | **44.16** | 61 158 MiB |
| **7** | **6 + `--n-cpu-moe 0`** | **424.77** | **44.58** | 61 158 MiB |
| 8 | baseline re-run (repeatability check) | 137.19 | 40.37 | 61 068 MiB |
| **9** | **6 + `-b 2048 -ub 2048`** | **427.56** | **44.53** | 61 158 MiB |

Run 8 is a repeated baseline: 137.19 / 40.37 against run 1's 142.02 / 38.62 —
within 3%, so the measurements are reproducible.

## The actual finding: where the embedding lives

The nine configurations were not a parameter sweep. They isolate one variable:
**the placement of the token embedding tensor.**

```
        embedding on GPU  →  27 GiB of VRAM occupied  →  experts evicted
                                                          → PP collapses to 142 tok/s

        embedding on CPU  →  all experts fit in VRAM    → PP 425+ tok/s (+199%)
```

And **output / lm_head must stay on GPU**:

| Output placement | TG128 |
|---|---:|
| On GPU (runs 6/7/9) | **44.2–44.6 tok/s** |
| On CPU (runs 3/4/5) | **10.7–11.0 tok/s** (−75%) |

The mechanism is direct: **decode runs the output head once per token**, so
putting it on CPU means a PCIe round trip per token. Prefill is batched and
barely touches the output head, but it looks up embeddings constantly — so
embedding on GPU is pure loss.

**The two tensors want opposite placements.** That is the whole finding.

### Why the embedding is so large here

| Component | Size | Share |
|---|---|---:|
| Routed experts | 53.32 GiB | 64.4% |
| **Token embedding** | **27.08 GiB** | **32.7%** |
| Attention | 1.11 GiB | 1.3% |
| Output / lm_head | 0.56 GiB | 0.7% |
| SSM state | 0.24 GiB | 0.3% |

A single tensor at 32.7% of the model is abnormally large — 248 320 vocabulary
entries × 2560 dimensions, plus `per_layer_token_embd`. Keeping it in VRAM is
**the most expensive mistake available** on a 2×32GB card.

## Reproducing

```bash
cd /path/to/llama.cpp
TMPDIR=/path/with/space ./build/bin/llama-bench \
  -m /path/to/Qwen3.8-Flash-Next-IQ3_S-merged.gguf \
  -ngl 999 \
  -ot 'token_embd\.weight=CPU' \
  -b 2048 -ub 2048 \
  -p 512 -n 128 -r 1 -w 0 -t 36
```

For serving, `llama-server` takes the same `-ot 'token_embd\.weight=CPU'`.

## Why this mattered for our own tuning

The HF write-up that first described this pattern was directionally right, and
our numbers confirm it:

> "Decode is bound by RAM access latency of the per-token expert gather, not by
> streaming bandwidth."
> "fused MoE kernels raise same-hardware PP by 15–160%."

We measured **PP +199%**, above the report's upper bound, because beyond the
fused MoE kernels the "move all experts into VRAM" step contributed a large
additional gain on its own.

It also explains a result that otherwise looks like a contradiction: our
`--expert-cache auto` ends up with **all 24 576 experts resident in VRAM** and a
99.8% hit rate. That is the same outcome the embedding move produces on the
llama.cpp side, reached by a different mechanism — and it is why every
"reduce expert cache misses" optimization measures as a no-op on our setup.

## What we did not do

**We did not switch engines.** Strata's MTP speculative decoding (98.6%
acceptance on structured output) is worth more to our workload than llama.cpp's
prefill advantage, and the model we serve carries MTP tensors in a Strata-specific
format that llama.cpp cannot read directly.

The point of this page is the *placement* finding, which is engine-independent:
**on a card that can barely hold the model, a 27 GiB embedding tensor is the
first thing to move, and the output head is the last.**

## Two environment rules learned the hard way

1. **`ulimit -v` is capped at 4 GB in some sandboxed shells** (soft and hard, and
   `ulimit -v unlimited` is refused). Any large-model load or benchmark must run
   in a real session where it is unlimited. Check the process's cgroup if a load
   fails for no apparent reason.
2. **A systemd-managed service with `Restart=always` cannot be stopped by
   killing the process.** Killing it just gets it relaunched. Point the config at
   a nonexistent binary so it exits on start, then restore and kill once.
