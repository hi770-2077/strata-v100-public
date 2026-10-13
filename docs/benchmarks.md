# Benchmarks — measured on 2× Tesla V100-32GB

**Measured:** 2026-10-13 23:00 CST
**Measured on:** the live production service (not a synthetic offline run) —
same process, same flags, same GPU state that real requests hit.
**Raw data:** [`benchmarks/prod-bench-2026-10-13.json`](benchmarks/prod-bench-2026-10-13.json)
**Reproduce:** `scripts/bench-production.py`

## Hardware and software under test

| | |
|---|---|
| GPU | 2× Tesla V100-PCIE-32GB (sm_70, Volta, 32 GB) |
| Interconnect | PCIe only — **no NVLink** on this board |
| Driver / CUDA | 580.178.04 / CUDA 12.8 |
| Model | `Qwen3.8-Flash-Next-...-IQ3_S` (MoE, 512 experts, top-10) |
| Engine | Strata v0.1.43 + PR #1771 patch, `STRATA_PF_WMMA=2` |
| Runtime env | `STRATA_SM70_TABLE=1 STRATA_SPLIT_OWN=auto` |
| Serving | layer-split 25, expert cache fully resident in VRAM |
| KV cache | k8v4, kv-resident 49152 |
| Context | 262144 max |
| VRAM at rest | 30652 / 32768 MiB (card 0), 30976 / 32768 MiB (card 1) |

**Read the numbers as a reference point for this exact setup, not as a V100
general benchmark.** We have no NVLink; a machine that does will differ.

## Headline numbers

All figures are the **median of 3 runs**, and every run used a *different*
prompt so that no run benefited from a cached prefix (see
[Measurement pitfalls](#measurement-pitfalls) — this is the single easiest
way to publish bogus speed numbers).

| Scenario | Prompt | Output | **Prefill** | **Decode** | MTP accept |
|---|---|---|---|---|---|
| Chat / steady state | 72 | 85–137 | 142 tok/s | **110.8 tok/s** | 0.88 |
| Long answer | 2 364 | 386–597 | **997.6 tok/s** | 95.7 tok/s | 0.91 |
| Prefill-heavy | 14 876 | 64 | **1 704 tok/s** | 92.4 tok/s | 0.93 |
| Long context | 36 864 | 64 | **1 817 tok/s** | 85.4 tok/s | 0.73 |

Prefill is **~16× faster than decode**. That asymmetry drives every tuning
decision below, and it explains which optimizations were worth doing at all.

## Reading the table

### Prefill scales well, decode does not

Prefill goes 998 → 1 704 → 1 817 tok/s as the prompt grows. The gain is
batching: a longer prompt turns more GEMM work into fewer, larger matmuls, and
V100's FP16 tensor cores get better utilized.

Decode sits in a 85–111 tok/s band regardless of context length. **Decode is
bandwidth-bound** — every generated token must read the weights back out of
VRAM. Once you are bandwidth-bound, arithmetic optimizations stop paying.

### Long context *lowers* MTP acceptance

MTP (multi-token prediction, speculative decoding) acceptance rate:

| Context | Accept rate |
|---|---|
| 72 tok | 0.88 |
| 2 364 tok | 0.91 |
| 14 876 tok | 0.93 |
| 36 864 tok | **0.73** |

Acceptance falls off sharply at 36K. Past some context length the draft model
diverges from the target model often enough that rejected drafts eat the
throughput gain. If you run near the context limit, **measure your own MTP
acceptance rather than assuming it stays above 0.9** — this is why we set
`reasoning_budget 16384` and a bounded MTP window in production.

### Thermal headroom was fine

GPU temperature 41 °C before / 56–57 °C after the run, power 43–53 W of a
250 W limit. No thermal throttling, so these numbers are not clipped by cooling.

## What the numbers imply

1. **Prefill-side work is where the wins are.** Our WMMA prefill patch is worth
   +4.1–4.3% on prefill (1 127 vs 1 081 tok/s in the A/B) — real, but small,
   because prefill was already tensor-core bound.
2. **Decode-side arithmetic work is mostly wasted.** Every "make the multiply
   cheaper" idea we tried on the decode path came out flat or negative. On a
   bandwidth-bound decode, this is the expected outcome. See
   [`flag-ledger.md`](flag-ledger.md) for the six optimizations that measured
   as no-ops or regressions.
3. **Expert-cache pressure does not exist on this setup.** Both cards sit at
   ~30.7 GB / 32.8 GB with every expert resident, so the entire family of
   "reduce expert cache misses" optimizations has, by construction, nothing to
   gain. This is a 2-card-result and does **not** generalize to 4 cards or to
   models with more experts.
4. **Prefix caching dominates any prefill micro-optimization.** Re-sending the
   same 12.9K prompt drops `prompt_tokens` from 14 876 to **5** and prefill cost
   from ~9.4 s to ~37 ms. That is a **254×** difference — two orders of
   magnitude larger than anything tuning flags gave us. If your workload
   repeats a long system prompt, cache it before you tune anything else.

## Measurement pitfalls

We got these wrong first; they are worth more than the numbers above.

**A cached prefix silently destroys your benchmark.** On our first run, rounds
2 and 3 reported `prompt_tokens = 5` and a "prefill rate" of 22 tok/s. The
engine was correct — it had cached the prefix from round 1 and had almost no
work to do. Reporting that as a prefill number would have looked like a 90%
regression. **Always vary the prompt between runs** (we prefix a per-run salt),
and sanity-check `prompt_tokens` in the response against what you sent.

**Do not divide tokens by wall time when the engine reports its own timings.**
Strata's response carries a `timings` object with `prompt_per_second` and
`predicted_per_second`, measured inside the engine around the actual compute.
Wall time includes HTTP, JSON serialization, and queueing. We use the
server-side numbers throughout.

**Throughput and latency are different questions.** `predicted_per_second`
tells you tokens per second while generating. It says nothing about how long
the first token took. If you care about time-to-first-token, prefill is your
number.

**Three rounds is a floor, not a ceiling.** Our short-prompt decode spread was
104.6–112.3 tok/s, about 7%. Report the median and the raw numbers so readers
can judge the spread themselves.

**Only compare like with like.** Changing quantization, context length, or
expert residency between runs changes the workload, not just the version.
Our A/B protocol is in [`how-we-measure.md`](how-we-measure.md).

## Reproducing

Run against a live Strata server. On our setup (2× V100, PCIe):

```bash
python3 scripts/bench-production.py \
    --out prod-bench.json --rounds 3
```

The script writes every raw run, not just the medians, so you can re-derive
whatever statistic you care about. It reads rates from the engine's `timings`
field rather than computing them from wall clock.

## Historical context

The figures above are the current production configuration. Earlier points in
the tuning history (baseline before the WMMA patch, and before MTP was enabled)
are documented in the private working repository — they are machine-specific
and change as upstream lands new kernels, so we pin the measurement date
rather than presenting them as a trend line.
