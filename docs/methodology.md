# Methodology — seven loops, six self-refutations

**This page is worth more than any parameter value in this repository.**

On 2026-10-11 we ran seven consecutive optimization loops on the same machine.
**Six of the seven started from an assumption the data then destroyed.** That ratio
is not a joke about our process — it is the process. Every conclusion below was
reached by measuring, and several of them are the opposite of what any reasonable
person (including us, one loop earlier) would have predicted.

If you take one thing from this repository, take the table in §2. If you take two,
take the six-step protocol in §4.

---

## 1. What we were trying to answer

"Where is the remaining headroom on this machine?" — a question with an obvious
wrong answer ("the GPU is only at 72% utilization, so 28% is available") and no
obvious right one.

We split it into four candidate directions and ran each to a conclusion:

| Direction | Assumption at start | Verdict |
|---|---|---|
| Memory upgrade (64 → 128 GB) | Broad speedup | Prefill +8–10%, load −38%, **decode flat** |
| Clock locking | Throttling = insufficient cooling | Locking 1250 is *slower* than locking 1380 |
| Fan tuning | Requires opening the case | A single `echo` to sysfs fixed it |
| Concurrency / batching | Official docs claim +40% | 4 concurrent is **slower** than 1 request |
| Prefix caching | Already working | Reuse was **0%** because of how we wrote the request |
| Pinned shared prefix | "Just add a field" | 4.19× — but the *control group* was the real finding |

## 2. The seven loops and what each one destroyed

| Loop | Initial assumption | What the data actually said | Refuted? |
|---|---|---|---|
| 1 — memory upgrade | More RAM speeds everything up | Decode **completely flat**; experts were already 100% resident in VRAM, so the decode path never reads RAM | 🔴 premise wrong |
| 2 — clock locking | Thermal throttle means poor cooling | Locking 1250 (never throttles) is **slower** than locking 1380 (throttles briefly at peak temp) | 🔴 counter-intuitive |
| 3 — fan investigation | Will need to open the case and clean it | One `echo 200` to a sysfs PWM channel; the V100's turbine shares a PWM channel with the CPU fan | 🔴 premise wrong |
| 4 — fan sweep | Higher fan speed → more performance | Three speeds, identical performance within noise. **The fan buys safety margin, not speed** | 🟡 target wrong |
| 5 — cache audit | prompt-cache isn't doing anything | Reuse rate 100%, TTFT 12.4× — it was working, we had never measured it | 🔴 premise wrong |
| 6 — concurrency | Batching raises throughput 30–40% | 4 concurrent = 43.6 tok/s vs 64.7 single. Strictly serial queueing | 🔴 direction wrong |
| 7 — pinned prefix | It's just an extra field | **Control group reuse was 0%** — the request *shape* determines whether caching works at all | 🔴 cognition wrong |

**Seven loops, six refutations, one partial.** The pattern is consistent: the
assumption that fails is always the one that felt obvious.

### The two most expensive lessons in detail

**Loop 7 — "it's just a field."**
We assumed adding `"strata_prefix": {"messages": 1}` to a request body was a
small opt-in tweak. The measurement said otherwise:

| Group | Per request | Cached | Reuse | 5 questions |
|---|---:|---:|---:|---:|
| Control (document and question in one message) | 8.40 s | 0 | **0%** | 41.98 s |
| Experiment (document as message 0 + prefix field) | **2.00 s** | 9 190 | **100%** | **10.01 s** |

The finding is not "the field gives 4.19×." The finding is that **the control group
got 0% reuse** — because the question follows the document inside the same message,
every request has a different prefix, so the cache never matches. *Writing the
request the normal way makes the cache invisible.* The optimization is not the
field; it is the request shape. Every client we own was writing requests the way
that guarantees zero reuse.

**Loop 2 — "throttling means cooling is inadequate."**
SM clock had dropped from the locked 1380 MHz to 757 MHz during long-context runs,
and prefill fell 28%. The obvious cause is heat. Locking the clock *lower* (1250,
never throttles) should then be the fix. It is not:

| Lock | 128K prefill | decode | Min clock observed |
|---|---:|---:|---|
| **1380 (kept)** | **2153** | **92.6** | 1080 (brief) |
| 1250 | 2022 | 87.1 | 1252 (never drops) |

1380 spends most of its time at full clock and only dips briefly at peak
temperature; its average is still higher than a stable 1252. **The fix was the fan
PWM, not the clock lock.** And the fan sweep (Loop 4) then showed the fan buys
temperature margin, not performance — three fan speeds, identical throughput.

## 3. The 1+1>2 structure that fell out of this

The loops revealed that the four optimization families act on **different terms**
of the end-to-end equation, which makes them multiplicative rather than
substitutable:

```
end-to-end ≈ [first turn: N tokens cold prefill] + [K turns × (cached prefill ≈ 0 + decode)]
              └──── kernel work acts here ────┘        └──── reuse acts here ────┘
```

Kernel and frequency work lowers the cost of *each* computation. Prefix reuse
eliminates *the occurrence* of computations. One divides, the other subtracts —
mathematically independent, so they compound.

Measured: with every kernel optimization already applied, prefix reuse still
delivered TTFT **12.4×** (5.48 s → 0.44 s) at 100% hit rate. If the kernel work
had not been done first, the same reuse would only have been worth ~8.2×.

**Corollary:** neither dimension alone reaches the ceiling. Kernel work can never
remove the first-turn cost; reuse can never speed up a single generation.

## 4. The protocol (this is the reusable part)

Six steps, each of which exists because skipping it produced a wrong conclusion
at least once in this project:

| Step | Rule | The failure it prevents |
|---|---|---|
| **First principles** | Compute where the time actually goes before choosing what to optimize | Optimizing a term that is 2% of the total |
| **Chain reasoning** | Follow each optimization chain to its **break point**, and stop there | Pursuing a direction that is structurally blocked |
| **Harness control** | Every experiment carries its own rollback; injection is verified three ways before results are trusted | A killed A/B leaving production on the last test config |
| **Loop iteration** | Each round attempts to refute the previous round's assumption, not refine it | Six rounds of micro-tuning a wrong premise |
| **Adversarial review** | Delete fabricated or fatal content *before* integrating it | Acting on parameters that do not exist |
| **Cross-validation** | Tag every claim with an evidence grade | Documenting a plausible inference as a measurement |

### Evidence grades used throughout this repository

| Grade | Meaning | May be relied on? |
|---|---|---|
| 🟢 | Measured on this machine, protocol documented | Yes |
| 🔵 | Official documentation, not yet measured here | Only after confirming it applies |
| 🟡 | Documentation claims only | No — must be measured first |

### Three rules that generalize beyond this machine

1. **Ask what the engine is deciding on your behalf before tuning the numeric knobs.**
   `--layer-split auto`, `--expert-cache auto`, `--prefill auto` are *decisions*,
   not settings. Our single largest win (+12.8%) came from discovering `auto` had
   chosen a 22/26 split on two identical cards. Nobody had ever printed what it
   chose.
2. **Verify an upstream flag's premise holds for you before adopting its number.**
   The same flag measured +2.7 ms/token at 4 096 expert slots and **−43%** at
   24 576 slots on this machine. The upstream figure was correct; its premise was
   not ours.
3. **Kernel-time reduction ≠ end-to-end speedup.** Upstream themselves warn that
   a 27.9% kernel-time cut yielded only 1.3–3.0% end-to-end prefill. Our v0.1.40.3
   "upgrade" compiled the IQ3_S AVX2 optimization in (+10.7% object size) and
   changed nothing, because our expert hit rate is 99.8% and that code path is
   never taken. **Before adopting "optimized kernel X", ask whether your workload
   walks path X at all.**

## 5. What this cost, and what it bought

The seven loops took one working day. They produced exactly **one adopted
parameter** (`STRATA_SPLIT_OWN=auto`, prefill −20%) and **nine documented
"does not apply here" exclusions**.

That is a good trade. The nine exclusions are permanently valuable — they stop
anyone (including us, six months from now) from re-running the same dead ends.
And the one real gain was found by the loop that started from the most wrong
assumption of all.

**A tuning session that adopts nothing is not a wasted session. A tuning session
that adopts something without knowing why is.**

---

Related: [`calibers.md`](calibers.md) — the measurement discipline these loops
depended on · [`how-we-measure.md`](how-we-measure.md) — the A/B protocol ·
[`flag-ledger.md`](flag-ledger.md) — what the loops adopted and refuted
