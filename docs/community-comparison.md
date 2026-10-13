# Community comparison — where this setup sits

**Compiled:** 2026-10-13
**Our numbers:** [`benchmarks.md`](benchmarks.md) (measured in-house, 2026-10-13 23:00)
**Rule for this file:** every community figure below links to its source. If we
could not trace a number to a public report, it is not here. Several numbers
that circulate online were excluded for exactly that reason.

**Confidence markers used below:**

| Marker | Meaning |
|---|---|
| (no marker) | First-party or fully traceable. Our own measurements, a linked benchmark repository, or a vendor release note. |
| † | Secondhand. The source is a write-up or aggregator; we did not open the original thread. |
| ‡ | Secondhand and the host blocks automated requests, so the link may not resolve outside a browser. |

We did not reproduce anyone else's test. Cross-machine comparison is therefore
always weaker than the A/B numbers in
[`flag-ledger.md`](flag-ledger.md), which are same-machine, single-variable.

## Short answer

**2× V100-32GB running Qwen3.8-Flash-Next IQ3_S is a strong result for this
hardware, and it is competitive with much newer cards.**

Our 110.8 tok/s decode is identical to what an independent third-party
repository reports for a **single RTX 4090** on the same model and quant
([dev.to fact-check](https://dev.to/ashraf_chowdury09/a-125b-model-at-100-toks-on-one-rtx-4090-heres-what-the-hn-hype-leaves-out-463n)).
That is the single most useful anchor here: a 2017 datacenter card matching a
2022 consumer flagship.

We do **not** lead the field. Newer or better-connected hardware beats us, and
one shortcoming is structural rather than fixable by tuning — see
[Where we fall short](#where-we-fall-short).

## Like-for-like: same model, same quant, Strata engine

| Configuration | Decode tok/s | Prefill tok/s | Source |
|---|---:|---:|---|
| **2× V100-32GB, PCIe, no NVLink (ours)** | **110.8** | **1 704** @ 14.9K | in-house, 2026-10-13 |
| 1× RTX 4090 24GB, 280 W cap | **110.8** | ~4 750 cold @ 32K | [dev.to / 4090 measurement repo](https://dev.to/ashraf_chowdury09/a-125b-model-at-100-toks-on-one-rtx-4090-heres-what-the-hn-hype-leaves-out-463n) |
| 1× RTX 3090 24GB | 60–65 | 1 200–1 300 | [llamaperf V100/3090 reports](https://llamaperf.com/gpu/v100-32gb) |
| 1× V100 32GB, IQ4_XS | 50–55 | 850–900 | [llamaperf V100/3090 reports](https://llamaperf.com/gpu/v100-32gb) |
| 1× V100 16GB, Q2_0, 1K ctx | 52.0 | 398.6 @ 1K | [Strata-V100 fork benchmark](https://github.com/jmnargi/Strata-V100) |
| 2× RTX 3090 64 GB, 262K ctx | 120+ | — | [vramcalculator release notes](https://vramcalculator.com/strata-qwen3-8-flash-next) |
| 1× RTX 5090 96 GB, Strata 0.1.42 | 179.8–221.1 | — | [vramcalculator release notes](https://vramcalculator.com/strata-qwen3-8-flash-next) |

Our decode ties a 4090 and sits ~8% below a pair of 3090s. Our prefill is
**~2.8× slower** than the 4090's reported cold prefill — that gap is discussed
under [Where we fall short](#where-we-fall-short).

## Direct two-card V100 comparison (same engine, same model family)

This is the closest apples-to-apples check available, because it is the same
engine and the same card type. From the
[Strata-V100 fork](https://github.com/jmnargi/Strata-V100) (upstream v0.1.39
integration, measured 2026-10-04, two cards, Qwen3.8-Flash-Next **Q2_0**, int8
KV, `--spec 8`):

| Target prompt | Their two-card prefill (tok/s) | Our two-card prefill (tok/s) |
|---:|---:|---:|
| 2K | 718.5 | — |
| 8K | 1 458.0 | — |
| 32K | 2 283.7 | — |
| 14.9K | — | **1 704** |

Read carefully: their 32K figure (2 283.7 tok/s) is **above ours at 14.9K**, and
their 2K figure (718.5) is **below ours at 14.9K**. The reason is context
depth — prefill throughput rises with prompt length on both machines because
longer prompts amortize launch overhead into larger GEMMs. At a *comparable*
input size we are in the same band, roughly 1.5–2.3K tok/s.

Two honest caveats on that row: their quant is Q2_0 (smaller, so it is
advantaged on prefill) and their engine is v0.1.39, while we run v0.1.43 plus
our [WMMA prefill patch](multigpu-wmma-bug.md). We did not reproduce their
test, so treat this as orientation, not a controlled A/B.

## Longer context on V100 — the honest comparison

Our 110.8 tok/s is a **short-context** figure (72-token prompt). Context depth
degrades decode on every configuration, so this table is the fair one:

| Configuration | Context | Decode tok/s | Source |
|---|---:|---:|---|
| Ours, 2× V100-32GB | 72 tok | **110.8** | in-house |
| Ours, 2× V100-32GB | 14.9K | 92.4 | in-house |
| Ours, 2× V100-32GB | 36.9K | 85.4 | in-house |
| Chinese community write-up, 2× V100-32GB | ~220K | 62–65 | [wqu.guru hands-on](https://wqu.guru/i/01M45VRY093PXJ86F72C9BQTN3) † |
| Chinese community write-up, 1× V100-32GB | ~220K | ~57 | [wqu.guru hands-on](https://wqu.guru/i/01M45VRY093PXJ86F72C9BQTN3) † |
| 1× V100 32GB, llama.cpp Q8_0, 131K ctx | 131K | 47.2 | [llamaperf](https://llamaperf.com/gpu/v100-32gb) |
| 4× V100 SXM2 + NVLink, PXA, MTP k=7 | 3K prompt | 159.6 | [llamaperf](https://llamaperf.com/gpu/v100-32gb) |

† This write-up blocks automated requests, so the link may not resolve in a
script or preview pane; it opens normally in a browser. We cite it because it is
the only 220K-context two-V100 figure we found. Treat it as a secondhand
data point and prefer our own measurements where they overlap.

Our 85.4 tok/s at 36.9K is within reach of the ~62–65 tok/s that a Chinese
community write-up reports for **220K** context on two V100s. That write-up
also records an improvement between Strata 0.1.38 and 0.1.39 (single-card
~57 tok/s, two-card 62–65 tok/s at ~220K input), so part of the remaining gap
is simply that they run a much deeper context than our 36.9K test.

## llama.cpp on V100 — a different engine, listed for scale

Different engine, and mostly a **different model**. Qwen3-30B-A3B is a 30B
model, not our 125B Flash-Next, so these numbers are context, not competition.

| Model | Quant | Engine | Test | Result | Source |
|---|---|---|---|---:|---|
| Qwen3-30B-A3B | Q5_K_M | llama.cpp | pp512 | 735.81 ± 13.51 | llama.cpp issue #19894, via [survey](https://markaicode.com/benchmarks/llamacpp-qwen-3-a100-80gb-throughput-benchmark) ‡ |
| Qwen3-30B-A3B | Q5_K_M | llama.cpp | tg128 | 59.02 ± 0.36 | llama.cpp issue #19894, via [survey](https://markaicode.com/benchmarks/llamacpp-qwen-3-a100-80gb-throughput-benchmark) ‡ |
| Qwen3.5-35B-A3B | Q4_0 | llama.cpp (ROCm 7900XTX) | tg128 | 77.82 ± 0.02 | HF ubergarm discussion, via [survey](https://markaicode.com/benchmarks/llamacpp-qwen-3-a100-80gb-throughput-benchmark) ‡ |
| Qwen3-30B-A3B | Q8_0 | llama.cpp (DGX Spark GB10) | tg32 @ 0 ctx | 45.67 ± 0.11 | llama.cpp discussion #16578, via [survey](https://markaicode.com/benchmarks/llamacpp-qwen-3-a100-80gb-throughput-benchmark) ‡ |
| Qwen3-30B-A3B | Q8_0 | llama.cpp (DGX Spark GB10) | tg32 @ 8K ctx | 38.09 ± 0.09 | llama.cpp discussion #16578, via [survey](https://markaicode.com/benchmarks/llamacpp-qwen-3-a100-80gb-throughput-benchmark) ‡ |

‡ These four rows reach us through a survey that aggregates public
`llama-bench` logs and attributes each row to a specific upstream issue or
discussion. The survey host blocks automated requests, so the link may not
resolve in a script. We have not opened the underlying llama.cpp issues
ourselves, so treat these as **secondhand** and verify at the source before
citing them anywhere load-bearing.

A useful cross-check from that survey: a V100 and an AMD 7900XTX land within
~30% of each other on token generation, while their *prompt processing*
numbers diverge by much more. That is the same compute-bound vs
bandwidth-bound split we measured — MoE decode is memory-bound, so card
generation matters far less than bandwidth.

## The fastest V100 decode number we found — and why it is not ours

People searching "V100 decode tok/s" will eventually hit **201–219 tok/s**. That
number belongs to a different project, **NInfer** (`geowatts/gw`), and it is
honest to say so plainly rather than quietly leave it out or, worse, quietly
bank it:

| | This repository | NInfer |
|---|---|---|
| Engine | Strata (upstream, patched) | Own C++/CUDA inference engine |
| Model | Qwen3.8-Flash-Next **MoE**, 48 layers / 512 experts | Qwen3.8 **27B dense** |
| Quantization | IQ3_S (GGUF) | Software-emulated **NVFP4** |
| Cards | 2× V100-32GB, PCIe, no NVLink | **1×** V100 |
| Headline decode | **110.8 tok/s** | **201–219 tok/s** |

Every one of those five variables differs. The 201.0 tok/s figure is measured on
a 172-token **verbatim-copy prompt**, which that project documents as a
"context-reproduction fast path" — a different workload from real generation, and
the reason it does not transfer to our numbers. Their maintainer log also shows
169–190 tok/s on longer, more realistic prompts.

We list it as the **ceiling for this hardware class**: it says a single V100 has
roughly twice our per-card headroom left on the table, reachable only by leaving
GGUF and Strata behind. It is a target, not a score. Anyone comparing our 110.8
against 219 without reading this section would be comparing different engines,
different models, different numeric formats and different workloads.

## Where we fall short

**Prefill, by a wide margin, and it is structural.** A Strata fork on an IBM
AC922 (4–6 V100 SXM2 **with NVLink**, POWER9) reports **7 357 tok/s prefill**
and ~113 tok/s decode — roughly 4× our prefill at comparable decode
([r/LocalLLaMA report, via aggregator](https://pulseaugur.com/cluster/279310-strata-fork-boosts-ibm-ac922-inference-to-7-350-tokens-sec) §).
§ Single-source, and the aggregator host blocks automated requests. We have not
verified it against the original thread. It is the most striking prefill number
we found for V100-class hardware, which is exactly why we would rather flag it
as unverified than present it as settled.
The decode is comparable to ours; the prefill is not, and the reason is
interconnect: **our two cards talk over PCIe with no NVLink**, so every
layer-split token crosses the bus. No amount of flag tuning fixes that. This
is the single strongest argument for an NVLink-equipped machine.

**MLP head offload.** One V100 16GB report shows prefill of 1 567 tok/s at an
8 192-token prompt with Q2_0 ([vramcalculator](https://vramcalculator.com/strata-qwen3-8-flash-next)),
which is close to our 1 704 at 14.9K despite half the VRAM. If that holds up, it
suggests our prefill is limited by cross-card transfer rather than by the cards
themselves. We have not reproduced it, so we record it as a lead, not a claim.

## Three ways these comparisons mislead

**Model identity first.** Most V100 reports online are for **Qwen3.8-27B**, a
dense 27B model, not our 125B-class MoE. Dense-27B figures cluster at 23–65
tok/s; ours is 110.8 on a different, larger MoE. Comparing them directly
double-counts the difference in the model, not the hardware.

**Context depth changes everything.** On one V100 32GB with llama.cpp and
Q4_K_M, decode runs 65.07 tok/s at 1K and 42.22 tok/s at 128K. Any table
comparing a batch-1 empty-context number against a deep-context number is
comparing two different workloads. Compare like with like, or not at all.

**MTP acceptance is workload-dependent.** Community measurements on this same
model report acceptance around **0.59 on prose, 0.88 on code, 0.93 on
structured output**
([dev.to](https://dev.to/ashraf_chowdury09/a-125b-model-at-100-toks-on-one-rtx-4090-heres-what-the-hn-hype-leaves-out-463n)).
Our 0.88–0.93 came from synthetic prompts that are closer to structured text.
On conversational prose we would land lower, and the widely circulated "100
tok/s" headline is itself a code-and-structured-output figure. Do not present
it as a chat benchmark.

## What we would publish next

To make this page stronger rather than just longer:

1. Re-run our four scenarios at 128K and 256K so our own curve reaches the
   depths the community actually reports at. This is the most valuable missing
   measurement.
2. Reproduce the [Strata-V100 fork](https://github.com/jmnargi/Strata-V100)
   two-card protocol (2K/8K/32K, Q2_0) on our hardware. That converts this
   page's "same band, not controlled" into a real A/B.
3. Benchmark a code-heavy prompt set and a prose prompt set separately, and
   report both acceptance rates. The community gap between 0.59 and 0.93 makes
   any single acceptance number nearly meaningless.
