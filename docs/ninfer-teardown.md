# NInfer 拆解：一个 219 tok/s 的解剖，以及我们拿走了什么

> 缘起：2026-10-14。社区项目 [NInfer](https://github.com/geoffwatts/ninfer-v100)（自写 C++/CUDA
> 推理引擎）在单张 V100 上跑出 219 tok/s，这个数字在我们周边传了一圈。
> 本文把它拆开：数字怎么来的、哪些机制能移植、哪些不能。
>
> **结论先行：219 是合成续写语料上的 fast-path 峰值，不是日常生成速度——
> NInfer 自己测的真实文本曲线是 43~93 tok/s。但它的窄窗投机与 N-gram 草稿思路
> 值得我们逐条验证，其中一项已经直接变成了我们本月的 A/B 测试。**

---

## 1. 先归属数字：三个"不同"

流传的 201~219 tok/s 与我们的数字之间隔着三个"不同"，任何一个都足以让对比失去意义：

| | NInfer 的 219 | 我们 |
|---|---|---|
| 模型 | Qwen3.8-27B **dense**（非 MoE），软件 NVFP4 4-bit ≈14 GB | Qwen3.8-Flash-Next **MoE**（48 层 / 512 专家 top-10），IQ3_S |
| 引擎 | 自写 C++/CUDA | Strata |
| 工况 | 172-token **逐字复现**提示词的 context-reproduction fast path | 日常生成（短 prompt / 长上下文 / Agent 任务） |

所以这个数字不进本仓库任何对照表——理由见 [`community-comparison.md`](community-comparison.md)
的 Model identity 一节。本页只做技术拆解：**把别人的峰值拆成机制，看哪些能长在我们自己身上。**

---

## 2. 219 的三层解构

### 第一层：头条数字是怎么跑出来的

- 模型：Qwen3.8-27B dense，软件模拟 NVFP4 4-bit，≈14 GB，**单卡** V100-32GB
- 投机：MTP K=1，接受率 **99.2%**
- CUDA Graphs：整个投机循环一次 launch
- 语料：**合成续写语料**——NInfer 自己在报告里标注
  *"synthetic-corpus ceiling, not a general-generation rate"*

### 第二层：同一张卡、同一个模型的真实文本

NInfer 用 37,758 token 的非重复语料测了自己的 varied-context 曲线：

| 上下文 | decode | 接受率 |
|---|---:|---:|
| 2K | 67.66 | 59.9% |
| 8K | 92.74 | — |
| 16K | 49.80 | — |
| 32K | 55.04 | — |
| 150K | 43.63 | — |

同一台机器的日常数字（我们的模型，四档上下文，详见 [`benchmarks.md`](benchmarks.md)）：
72 tok → 110.8 / 2.4K → 95.7 / 14.9K → 92.4 / 36.9K → 85.4 tok/s。
**跨模型比绝对速度只有参考意义，两条曲线的形状才是信息**：NInfer 的曲线在 8K 之后
崩塌（92.7 → 49.8），我们的在 37K 仍守住 85+——这与两者草稿来源的结构差异有关
（见 §3 机制 6），也提醒我们：任何投机解码的收益都随上下文衰减，长上下文场景要单独测。

### 第三层：context-lookup fast path 的真相

外面传的"219 是 memcpy 作弊"是错的。实际机制：

1. host 在 prompt 里找**最近的 16-token 后缀**；
2. 复用其后**最多 15 个 token** 作草稿；
3. 这些草稿**全部仍经 target 模型验证**（lossless，不是跳过计算）。

关键在于第 1 步的 eligibility test：**普通散文根本触发不了**。
只有"让模型逐字续写自己刚见过的内容"时才命中——合成语料正是这种工况的主场。

### 三个数字之间的关系

合成语料 = fast path 的主场（后缀必然命中，接受率 99.2%）；
真实文本 = fast path 基本熄火（接受率 59.9% 起，长上下文崩塌）。
**它自己两组数字之间差 2.4~5 倍——这就是 219 的全部真相。**

---

## 3. 六个机制逐条评估

| # | 机制 | NInfer 的做法 | 我们的现状 | 评级 |
|---|---|---|---|---|
| 1 | 草稿/验证宽度解耦 | `verify_k` 与 `proposal_k` 分开配，proposal extent 按「剩余输出预算 / 剩余上下文 / 配置窗」三者最小值自适应收缩 | Strata 早有对应物：`--spec` 只定 MTP 草稿数，验证宽度运行态扩到 `min(草稿数 + lookup_chain, 8)`（kVerifyMaxT）；`spec/draft_policy.hpp` 注释原话 "takes one only where it pays" | **已内置** |
| 2 | 窄窗赢 | K=1 → 219.0，K=7 → 181：接受率饱和后，验证成本主导 | 已测 spec 2 vs 4（2026-10-14 ABAB）：**输 2.0%**，接受率反升 4.9pp 但 decode 降 | **已测：不移植**（见 §5） |
| 3 | 一次 CUDA Graph launch | 整个投机循环一张图 | Strata 已有（session.cpp：每层两个 graph 在 router 处切开） | 打平，划掉 |
| 4 | Volta 专属投机验证内核 | `small_t_i8_volta` / `small_t_bf16_volta` | 已测（2026-10-14 晚）：verify 窗口 T≤8 与 decode 共享 dp4a 多列内核，实测边际列仅 6.6 µs/列、算力只占 dp4a 峰值 ~10%；Volta 无 int8 mma 硬件；SM70_TABLE 三件套已覆盖验证路径仅有的 Volta 专属受益点且生产已开 | **已测：无候选价值**（见 §5） |
| 5 | KV int8 + Hadamard | D256 Hadamard 变换（QuaRot 家族）压 INT8 KV 离群值，默认集成 | 已测 `STRATA_KV_ROT=1`（2026-10-14 ABAB 四轮）：**无差异**——k8v4 主路径结构性排除（layer.cpp:704），仅 MTP drafter 的 INT8 ring 受影响且翻不动接受决策 | **已测：无增益**（见 §5） |
| 6 | N-gram 草稿源 | context-lookup（上文 §2.3） | Strata 的 **suffix drafter 默认开启**：提示词后缀 N-gram 作草稿，官方 bench 记录 code 场景 **+6~11%**、普通散文持平 | **方向已采纳**，NInfer 版是其无损强化 |

关于 #6 的一个注脚：NInfer 的 context-lookup 与我们的 suffix drafter 是同一个思想
（从已见文本里捡草稿，不经模型），它的贡献在于证明了这条路在 lossless 前提下
能把接受率推到 99%——**前提是工况配合**。

---

## 4. 明确不跟

- **NVFP4 格式移植**：软件模拟的 4-bit 浮点，dense 模型的甜点；我们的 MoE + GGUF
  生态搬它代价大、收益不确定。
- **DFlash2**：NInfer 自己的对比数据里，同 shape 下 MTP 全胜——社区已经替我们证伪了。

---

## 5. 我们在测什么（三项均已测：窄窗输 2.0%，KV 旋转无差异，Volta 验证内核无候选价值）

机制 #2 直接变成了本月的 A/B：`--spec 4`（MTP 4 drafts + 验证宽度 8）
vs `--spec 2`（MTP 2 drafts + 验证宽度 6），ABAB 交替、日常口径
（`reasoning_effort=none` 流式测速 + 12 prompt 接受率探针，协议同
[`how-we-measure.md`](how-we-measure.md)）。

**实测结果（2026-10-14，日常口径，best-of-3 中值）：**

| 配置 | decode 中位 | 接受率 | bench |
|---|---:|---:|---:|
| `--spec 4`（现生产） | 103.60 tok/s | 85.7% | 106.6 |
| `--spec 2` | 101.50 tok/s | **90.6%** | 105.3 |

判定：**窄窗证伪，−2.0%**（ABAB 各 2–3 轮，复现差 <0.2%，结论决定性）。
接受率反升 4.9pp 是选择效应——草稿少了单条更易中，但每步吃进的 token 更少：
lookup 免费草稿两个配置都有，砍掉的正是 MTP 的深预测。
**教训：接受率是误导性指标，必须与 decode 同时报。**

经济账：NInfer 的 K=1 赢在 99.2% 接受率的合成 fast-path 语料（§2.3），
且它没有免费的 lookup 草稿分摊验证预算；我们的 suffix drafter 默认开启，
简单预测已被免费覆盖，再砍 MTP 砍的是深预测——同一个机制，两种草稿结构，结论相反。

**机制 #5 也已测完（2026-10-14 下午）：无增益。** `STRATA_KV_ROT=1` ABAB 四轮
（off on off on），日常 decode 103.70→103.83、接受率 85.8%→86.0%、bench 106.9→106.7，
全部噪声内；固定 prompt temp 0 的 canary 输出四轮**逐字节一致**。代码层面原因：
k8v4 主路径结构性排除旋转（layer.cpp:704），唯一理论通道是 MTP drafter 的 INT8 ring
（mtp.cpp:336-337），扰动翻不动接受决策；且 `c.kv_rot` 进 SessionConfig 指纹
（generate.cpp:7787），开启即使无增益也废掉会话缓存。详情见
[`flag-ledger.md`](flag-ledger.md) 证伪表。

**机制 #4 也已测完（2026-10-14 晚）：无候选价值。** 代码走查 + `iq_multi_parity --bench`
实测（数据见 [`iq_mmvq_bench.md`](iq_mmvq_bench.md)）：verify 对窗口 T 个 token 跑完整
目标模型前向，与 decode 共享同一批 dp4a 内核，且「权重解码一次」的多列路径已覆盖 T=2..8——
IQ3_S 8192×2560 单列 27.4 µs（权重 9.01 MB → 329 GB/s = 37% of 900 GB/s），8 列 73.5 µs，
**边际 6.6 µs/列 vs 旧逐列路径 20.8 µs/列**；生产 spec=4 的 T=4 验证窗已享受
46.5 vs 89.5 µs（1.92×）。ridge point ~35 MACs/byte vs 单列强度 2.3、8 列复用 18.6
→ 深内存瓶颈区，写算力向内核（无论 mma 还是别的）零收益；且 Volta 无 int8 `mma.sync`
（sm_75+ 才有），llama.cpp 自己的 IQ3_S mma 路径同样是 Turing 门控——「移植」的对象
在 V100 上不存在。

**剩下的队列**：iq_mmvq 访存优化（带宽利用率 37% → 现实目标 1.3–1.5×，需先做全局
GPU 时间分解再定优先级），见 [`flag-ledger.md`](flag-ledger.md) 待验证候选区。
旧候选「mma IQ3_S 移植」已证伪关闭（Volta 无 int8 mma 硬件，上游 IQ3_S mma 路径
Turing+ 门控；f16 mma 路线属从零写内核且无算力瓶颈可解）。

---

## 出处

- NInfer 主仓：`github.com/geoffwatts/ninfer-v100`（`docs/v100.md` 有完整 K 扫描表与
  varied-context 表，本文第二层数据出处）
- 顺带看过的 fork：andrewleech/ninfer-v100、tuxKOH/ninfer-V100X2、shensanshu/ninfer-master-shensanshu-kvmem
