# The five calibers — why one V100 has five different "tok/s"

**The single most important measurement discipline in this repository.**

The same machine, the same model, the same parameters, measured within one week,
produced five decode numbers spanning **87 to 203 tok/s**. Every one of them is
real. None of them is comparable to another. This document defines each caliber,
says exactly how to reproduce it, and explains what drives the spread — so you
can tell which number you are actually looking at, in ours or anyone else's.

| # | Caliber | How it is measured | Decode | When to use it |
|---|---|---|---:|---|
| 1 | **Peak** | Stream-scrape: run with `--until 200`, sample continuously, freeze the moment the target is crossed. `reasoning_effort=none` | **203.1** | Marketing, and upper-bound analysis |
| 2 | **A/B benchmark median** | 5×512 tokens, `reasoning_effort=none`, median | **105.2** | Comparing two configurations |
| 3 | **Real-task median** | Last 12 real requests from production `/v1/metrics` | **159.2** | "What will I actually get" |
| 4 | **Chat, thinking on** | 72-token prompt, thinking enabled, 3-run median on the live service | **110.8** | Day-to-day user experience |
| 5 | **Full 256K context** | Generate after a 260K-token prompt | **87.4** | Long-document work |

## What drives the 2.3× spread

Three independent multipliers, each measured on this machine:

**1. The thinking switch — 16.2%.**
Same config, same prompts, 512 tokens:

| Caliber | tok/s |
|---|---:|
| No thinking, client-side wall clock | **93.74** |
| No thinking, engine-reported timings | 97.60 |
| Thinking on, client-side wall clock | 78.79 |
| Thinking on, engine-reported timings | **78.60** |

Engine-reported vs wall clock differ by only ~4%. **The thinking switch is worth
16.2%** — it is not a measurement artifact, and it is not small.

**2. Task type — another large step.**
A structured benchmark (code enumeration, sequences, SQL — questions with a
single correct answer) produces short, predictable output that the draft model
predicts well. Free-form chat produces output the draft model gets wrong. This is
why the A/B benchmark reads 105 and real agent tasks read 159 while chat reads
110: they are three different *workloads*, not three different *measurements of
the same workload*.

**3. Context depth — a further step.**
1K → 256K costs 18% (106.0 → 87.4). The draft model diverges more as context
grows; rejected drafts eat the speculative-decoding gain.

And the peak caliber sits on top of all three: no thinking + structured output +
short context + the luckiest sample in a continuous scrape.

## Why we put 110.8 in the README headline

Because it is the hardest number in the table to produce, and the only one that
describes what a user actually experiences.

203.1 requires: thinking disabled, a structured benchmark, a short context, and a
stream-scrape that keeps the best instant. 110.8 requires: thinking enabled, a
normal chat prompt, the live production service, and three rounds with a fresh
prompt each time. **If we led with 203, we would be selecting the most flattering
of five true numbers.** Publishing all five lets a reader pick the one that
matches their use case, and it costs us nothing.

This is the opposite of how most V100/Strata numbers circulate: a single
unlabeled figure with no caliber, no prompt, no thinking state, and no context
length. **A decode number without a caliber is not a number.**

## How to reproduce each caliber

| Caliber | Script | Notes |
|---|---|---|
| Peak | `photo_speed.py --until 200` | Stream-scrape; freezes on crossing target |
| A/B median | `decode_bench.py` | `reasoning_effort=none`, warmup 1×256 then 5×512 |
| Real-task median | `curl /v1/metrics` | Take the last 12 requests, not the best |
| Chat | `bench-production.py` | Thinking on, fresh prompt per round |
| Long context | `longctx_ctx.py` | Unique salt prefix per tier |

**Warm up first.** A cold first run measures ~55 tok/s and drags the median down
by five points. Standard procedure is warmup 1×256, then the measured runs.

**Median of at least 3, and report the spread.** Our short-prompt decode spread
was 104.6–112.3 tok/s (~7%) across rounds of the *same* configuration. A single
round tells you nothing below ~8% difference.

## Cross-caliber comparison is invalid

The most common way these numbers get misused:

- Someone reads our 110.8 and compares it to a single-card V100's 52 — that is a
  hardware comparison (2 cards vs 1), not a caliber comparison.
- Someone reads our 203.1 and compares it to the 4090's 110.8 — that is comparing
  our best instant against their steady state.
- Someone reads "256K costs 18%" and assumes it applies to prefill too. It does
  not; prefill at 256K falls off for thermal reasons, which is a different
  mechanism entirely.

Before comparing two decode numbers, confirm all four of these match:
**thinking on/off, task type, context depth, and instant-vs-median.**

## Related

- [`benchmarks.md`](benchmarks.md) — the production numbers behind caliber 4
- [`community-comparison.md`](community-comparison.md) — how our calibers line up
  against other people's published figures
- [`how-we-measure.md`](how-we-measure.md) — the A/B protocol behind caliber 2
