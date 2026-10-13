# Client-side call patterns — the last 4× is in the request body

**Server-side tuning has a ceiling. The next layer of gain is in how you write
the request.** Measured on this machine: the same engine, the same parameters,
the same document — **0% vs 100% prefix reuse, 8.40 s vs 2.00 s per question** —
decided entirely by request shape.

This page is the part most easily skipped and most easily worth more than any
flag in [`flag-ledger.md`](flag-ledger.md).

## 1. The finding that reframes everything

One 4 413-token technical document, five different questions, two ways of
writing the request:

| Group | Per request | Cached | Reuse | 5 questions total |
|---|---:|---:|---:|---:|
| **Control** — document and question concatenated into one message | 8.40 s | 0 | **0%** | 41.98 s |
| **Experiment** — document as message 0, question as message 1, plus `strata_prefix` | **2.00 s** | 9 190 | **100%** | **10.01 s** |

**4.19× faster, 31.97 seconds saved.** But the number that matters is the control
group's **0%**.

The reason: when the document and the question share one message, the question
sits *after* the document in the prefix. Every question is different, so the
prefix never matches, so the cache never hits. `--prompt-cache 8` is enabled and
working — it simply has nothing to match against.

**Writing the request the obvious way makes the cache invisible.** This is not a
tuning problem; it is an API-usage problem, and it is free to fix.

## 2. The correct shapes

### Long document / long system prompt + consecutive questions

```python
body = {
  "model": "<your-model>",
  "messages": [
    {"role": "system", "content": 长文档},      # message 0: byte-identical prefix
    {"role": "user",   "content": "问题..."}     # then alternate user/assistant
  ],
  "strata_prefix": {"messages": 1}   # ← declares everything before message 1 is fixed
}
```

When the document and the question must share one message, use a character
boundary instead:

```python
"strata_prefix": {"message": 0, "chars": 210000}
```

Prerequisite: `--prompt-cache ≥ 3` (we run 8).

### Tool calls — append the result as a user message

Do not rebuild the message array each turn. Append the tool result as a `user`
message and let the engine roll the seen content into a checkpoint at end of
turn. Measured, with a 1.7k tool result inserted mid-conversation:

| Turn | prompt | cached | What happened |
|---|---:|---:|---|
| R1 first question | 4 120 | 4 115 | document prefix established |
| R2 + tool result 1.7k | 5 826 | 4 115 | **document 100% preserved**, new result not yet cached |
| R3 + trace 1.2k | 7 012 | 5 821 | **R2's tool result absorbed into the prefix** |
| R4 follow-up only | 7 045 | 7 007 | full history hit, 33 new tokens |

Real agent-loop TTFT stabilizes at 1.1–3.4 s, against 5–8 s for a cold 7k read.

### One-off requests (classification, probes)

```python
body = { "messages": [{"role": "user", "content": "..."}],
         "strata_checkpoint": False }
```

No session slot, no checkpoints. **Only for genuinely single-shot requests.**

## 3. Two traps, both measured

**`strata_checkpoint: false` and `strata_prefix` are mutually exclusive.**
A request carrying `false` saves no checkpoints at all, so the document prefix is
never persisted and every subsequent question re-reads the whole thing
(10.2 s × N). Adding this field to a multi-turn conversation silently destroys
the entire reuse benefit.

**The prefix must be byte-identical.** One changed punctuation mark in the system
message and the prefix no longer matches. Never put timestamps, random IDs, or
other dynamic content in the system message.

## 4. For clients you cannot modify: the proxy

WorkBuddy / Claude Code / OpenClaw and similar tools cannot change their request
bodies. A small proxy in front of the engine injects the prefix automatically:

```
client ──(8081)── strata_proxy.py ──(8080)── engine
                      │
                      └─ rule: messages ≥ 2 AND message 0 is system/user
                                 AND content ≥ 2000 chars AND request has no
                                 strata_prefix already
                              → inject "strata_prefix": {"messages": 1}
                              never inject strata_checkpoint (multi-turn must
                              keep its checkpoints)
                              SSE streamed chunk-by-chunk, everything else
                              forwarded untouched
```

Measured with a realistic 8 700-character manual plus tools, client unaware:
first question 4.66 s to build the cache → second question `cached=5477`
**0.37 s** → third 0.52 s.

Injection is observable — each request logs a line, so you can confirm it is
actually firing rather than assuming.

## 5. Output budget ≠ context window

A failure worth understanding because the error message points the wrong way:
"Max Token exceeded" does **not** mean the context window is full.

- **The bowl** = `--max-context 262144`: total input + output budget. 262K is the
  model's native trained length.
- **The cup** = a single request's `max_tokens`: if the client omits it, the
  engine uses whatever bowl space remains. If the client sends something absurd
  like 999999, the engine **rejects with HTTP 400** — which reads like a context
  problem but is not one.

The production fix, both now in our config:

| Setting | Value | Effect |
|---|---|---|
| `fit_max_tokens` | `true` | A `max_tokens` larger than the remaining context is **shrunk to fit** instead of 400 |
| `reasoning_budget_tokens` | `16384` | Hard cap on thinking, so thinking cannot consume the whole output budget and leave no room for the answer |

Verified: `max_tokens=999999` now returns HTTP 200 with a normal answer; long
generation shows no regression.

## 6. A related engine bug: pinned prefixes freeze the conversation cache

Found while validating the above. Worth recording because the symptom is
misleading and the fix is two numbers.

**Symptom:** the log fills with
`conversation cache: skip parking (snapshot N MiB exceeds available budget)` —
122 times.

**The counter-intuitive evidence:** the failing snapshot was **229 MiB** while
the budget was 8192 MiB and actual usage was 3 039 MiB. It fits. It is refused
anyway.

**Root cause chain** (verified layer by layer in the engine source):

1. The proxy injects `strata_prefix` for any request whose message 0 is ≥ 2000
   chars; the server translates that to `pin=N`.
2. The engine marks that checkpoint `pinned=true`.
3. `evict_oldest()` **only evicts non-pinned entries**.
4. When all slots are full and every entry is pinned, `make_room()` finds nothing
   it is allowed to evict and refuses to park.

So: **every conversation sharing a long system prompt is pinned, none can be
evicted, and the cache freezes permanently.** A continuing conversation survives
because `drop_superseded()` deletes its own previous copy; a *new* conversation
has no old copy to delete and can only rely on eviction — which is blocked.

| Parameter | Was | Now | Why |
|---|---|---|---|
| `--conversation-cache-slots` | 4 | **8** | The real constraint is slot count, not bytes |
| `--conversation-cache-mib` | 8192 | **24576** | Snapshot cost measured at ~26 KB/token; a full 256K session is ~6.4 GiB |

After the fix: `skip parking` 122 → **0**, three independent long-prefix sessions
all parked and restored, second round `1313 reused + 23 read` (98.3%免重算).

**Honest residual risk:** this raises the ceiling, it does not remove it. More
than 8 concurrent all-pinned conversations will freeze it again. A real fix needs
the engine to allow evicting pinned entries under pressure, or the proxy to stop
injecting prefixes — at the cost of the reuse gain. We chose the ceiling.

## 7. What to take away

Server-side flags gave us, in total, roughly +30% over a tuned baseline. The
request-shape change gave **4.19×** on the workload it applies to, and the
conversation-cache fix turned a frozen cache back into a working one.

**Tune the server first — then check how you are calling it. The last multiplier
is usually in the client.**
