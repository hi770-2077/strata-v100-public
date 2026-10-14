# 23 个调优开关的完整台账

> 硬件：2× Tesla V100-PCIE-32GB（GV100GL, sm_70），驱动 580.178.04，CUDA 12.8
> 模型：Qwen3.8-Flash-Next GSQ-RCO IQ3_S（48 层 / 512 专家 top-10 / GDN+QSA 混合 / PLE n-gram 层）
> 引擎：Strata v0.1.43 + sm_70 实验构建
> 全部数字来自三轮 ABAB 交替（同一二进制，只切环境变量），best-of-3 中值，引擎日志交叉核对。
> **测量协议见 [`how-we-measure.md`](how-we-measure.md) —— 换机器复现前请先读。**

---

## 采纳（13 项，生产在跑）

| 开关 | 收益 | 实测依据与机制 |
|---|---|---|
| `--pipeline-windows 2` | prefill **+16.6%**，decode 平 | 隐藏层切分 hand-off 的 stall。社区 #1447（2×3090：1798→2097 tok/s）；**只在有 stall 可隐藏的慢 GPU 链路有效**，快链路机器实测 −1%（噪声内）—— 我们 PCIe 无 NVLink 属慢链路，采纳。注意与 `--remote-expert-opt` 互斥（我们没开） |
| `--mtp-hnorm stream` | **峰值 +3.7%**（190.3 那一档的关键项） | Qwen4exp MTP 图里 `pre_fc_norm_hidden` 逐流归一化（默认 `pooled` 是四条流一起做一次 RMS）。接受率 +1.5pp。**注意：很多人以为 `--spec-min-p 0.5→0.8` 的增量是它的功劳，实测 0.70 在噪声内，真正的增量来自这一项** |
| `--host-core last` | **峰值 +6.5%**（191.4 首破 190 的来源） | host 线程挪到最后一个物理核。Windows 把 GPU 中断发给第一个逻辑核，自旋等待的 host 线程会被打断。中位只有 +1.0%（噪声边缘），但峰值显著 |
| `STRATA_SPLIT_OWN=auto` | prefill 12K 中值 **−20%**（8.36→6.65s，≈+25% 吞吐）；抖动 15.4%→7.6% | 每卡独立 prompt 缓冲。⚠️ **有代价：decode −6%**（89.4→83.7 tok/s），且输出与默认**不是** bitwise（官方声明 stable and coherent）。出处官方 MULTI_GPU.md（RDNA 上 prefill +16%、decode 不变）—— **我们 V100 上 prefill 收益更大、decode 却有代价**，同一个 flag 两种卡两种结果。**适用判断：长任务（读文档/Agent 多轮工具调用）强烈推荐开；纯短问答连发可不开** |
| `STRATA_SM70_TABLE=1` | decode **+5.7%** | v0.1.41 引入的 sm_70 专用解码三件套：expert mode 8 + `mmvq_il` Volta rows 表 + `gr_fast`，每window GPU 时间 −6.8%，输出 bitwise 不变。上游因无 V100 机器保持 opt-in。**不开只有 99.61 tok/s，落在噪声内** |
| `--layer-split 25` | decode **+4.7%**（vs 24）；长生成 **+8.5%** | 逐层扫描 23=99.43 / 24=102.88 / **25=107.27** / 26=107.01 tok/s。卡 0 还背 vision encoder，25/26 是平台期，取 25。**双卡 94% 时间同忙，切分不均** |
| `STRATA_PF_WMMA=2` | prompt **+4.1~4.3%**（1127/1127 vs 1081/1082） | down-only WMMA 混合，自研模式。全网首个 IQ3_S gate/up 的 WMMA 数据点。parity 8 格式全过 |
| `--mtp-q4 all` | decode 107.4→110.0（**+2.4%**）；长生成 105.6→107.5 | 投机解码草稿头 INT4 化。接受率 85.7~86.9% → **89.6%**（不降）；草稿头每步读取 212.9→146 MiB |
| `--kv k8v4` + `--kv-resident 49152` | 26.8K 上下文 KV 流式 **97.8%** 命中显存 | 显存总开关 |
| `--spec 4` / `--short-read 32` / `--prefill auto` | `prefill auto` 长 prompt 预填充 **+99%** | 甜点值。`--spec 4` 运行态自动扩到 8（验证宽度 = min(MTP 草稿数 + lookup_chain, 8)，引擎日志与 `/metrics` 双通道可证）；**spec 2 已证伪（−2.0%）、spec 1 直接拒启，下限 2** |
| `--expert-cache auto` | 专家 **100%** 显存常驻（11776/11776） | 双卡装得下所有专家 → "减少等miss"类优化理论增益为 0 |
| `fit_max_tokens` | 正确性护栏 | `max_tokens` 超剩余上下文时自动收缩（原来直接 400） |
| `reasoning_budget_tokens 16384` | 正确性护栏 | 思考硬上限 16K |

> ⚠️ **前两项（`--mtp-hnorm` / `--host-core`）的收益集中在峰值口径，中位口径分别只有
> +1.0% 左右（噪声边缘）。** 它们的价值是「把上限推高」，不是「把日常体验推高」。
> 报这两个数字时必须同时说明口径，否则读者会误以为日常也能拿到 +3.7%/+6.5%。
> 详见 [`calibers.md`](calibers.md)。

---

## 证伪（12 项，附失败原因）

| 开关 | 实测结果 | 失败原因 |
|---|---|---|
| `STRATA_PF_WMMA=1`（全 WMMA） | 净打平（~1084 vs 1081） | **IQ3_S gate/up 慢 20%**（Volta 上 MMQ 走 int8 `mma.sync`，吞吐 2× FP16，且q8_1 激活 1 字节/元素）；IQ4_NL down 快 20% → 取 down-only 混合（mode 2） |
| `--expert-cache-device1/2` | **引擎拒启** | 与 layer-split 架构**互斥**：要求一张不跑 stage 的空卡，4 卡以上才玩得起 |
| `STRATA_STAGE_PIN=1` | **解码损坏** | IQ3_S 4096 prompt 后输出崩坏。严禁 |
| `--short-read 0` | 5 个 fresh token 读取 129→274 ms | 小批次解码窗口下更快（判别法见 `scripts/bench_shortread.py`） |
| `--spec 8` | 长生成 105.6→99.5 tok/s | 窗口专家流量增速超过投机解码收益 |
| `--spec 2` | 日常 decode **−2.0%**（103.6→101.5 tok/s，ABAB 各 2–3 轮复现差 <0.2%）；bench −1.3%。**接受率反升 85.7%→90.6%** | 草稿少了单条更易中（选择效应），但每步吃进的 token 更少——lookup 免费草稿两个配置都有，砍掉的正是 MTP 的深预测。**接受率单独看会误导，必须与 decode 同时报**。NInfer 拆解见 [`ninfer-teardown.md`](ninfer-teardown.md) §5 |
| `--spec 1` | **引擎拒启** | native IQ pack 启动硬检查：`it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK`——「关掉投机看地板」在这个 setup 上不存在，spec 下限就是 2 |
| `--prefill auto:16384` | 仍解析为 8192 | 专家缓存 100% 占满，借不出缓冲区 |
| `--mtp-q4`（非 `all` 取值） | **−34%** | 与 `--mtp-q4 all` 是两回事，别混 |
| `--coupled-draft` | −24%~−43% | 旧版扫描证伪 |
| `--expert-cache-per-layer` | 有害 | 旧版扫描证伪 |
| `--spec 6` / `--mtp-max-t 8` / `--mtp-window 65536` / `--short-read 64` / `--prefill auto:32768` / `--batch` | 有害（−24%~−43%） | 旧版扫描证伪 |

**噪声内（双卡 99.7% 命中，无增益）**：`STRATA_ROUTE_RESIDENT` / `--aux-cpus auto` / `STRATA_ROUTE_TAIL_SKIP`

---

## 待验证的下一个候选

**① `STRATA_KV_ROT=1`：KV Hadamard 旋转（成本最低——只改一个环境变量）。**

背景：NInfer 把 D256 Hadamard 变换（QuaRot 家族）作为默认路径集成，用来压 INT8 KV
的离群值；Strata 有同款机制但是 opt-in（`generate.cpp` 2459-2461，INT8 K/V 过旋转），
我们没开。我们的 `--kv k8v4` 里 K 侧就是 INT8——**离群 token 的量化误差可能正在
吃掉长上下文的接受率**（36.9K 档已从 0.93 掉到 0.73，见 [`benchmarks.md`](benchmarks.md)）。
⚠️ 会改数值、输出非 bitwise，必须先跑 parity 再测速；口径按**长上下文**设计
（这条要的不是日常 decode 几个点，是长上下文接受率能否回抬）。

成本：改一个环境变量，ABAB 半小时出数。**下一个就测它。**
分析见 [`ninfer-teardown.md`](ninfer-teardown.md) 机制 #5。

**② 把 llama.cpp 的 mma IQ3_S 内核路径移植进 Strata。**

背景：llama.cpp（Unsloth fork b10995）已完整支持本架构 `qwen4exp`（含 PLE/indexer/GDN），
但它的 IQ3_S 在 Volta 上走 mma tensor-core，Strata 走的是转录自 llama.cpp **旧版 vec dot** 的
CUDA-core 路径。decode 阶段已是显存带宽瓶颈（专家 100% 常驻，每步要读整个模型），
所以这条收益主要在 **prefill / 多token 窗口**（那里是算力受限）。

优先级：中。要改 Strata 内核，且只对prefill 有效。

分析见 [`llama-cpp-mining-notes.md`](llama-cpp-mining-notes.md)。

---

## 我们的推荐生产配置（供对照）

```
STRATA_SM70_TABLE=1        # sm_70 解码三件套
--layer-split 25# 双卡 25/23
--pipeline-windows 2        # prefill +16.6%，decode 平
--spec 4                    # 运行态自动扩 8
--spec-min-p 0.8            # 原值即最优
--mtp-q4 all                # 草稿头 INT4
--expert-cache auto         # 专家 100% 常驻
--kv k8v4 --kv-resident 49152
--prefill auto              # 长 prompt +99%
--short-read 32
fit_max_tokens
reasoning_budget_tokens 16384
```

实测（2026-10-13 生产服务，四档场景每档 3 轮中位数）：

**decode ~110 tok/s**（短上下文）· **prefill 1 704 tok/s**（14.9K prompt）

四档完整数据、原始 JSON 与测量坑见 [`benchmarks.md`](benchmarks.md)。简表：

| 场景 | prefill | decode | MTP 接受率 |
|---|---|---|---|
| 72 tok prompt | 142 tok/s | 110.8 tok/s | 0.88 |
| 2.4K prompt | 997.6 tok/s | 95.7 tok/s | 0.91 |
| 14.9K prompt | 1 704 tok/s | 92.4 tok/s | 0.93 |
| 36.9K prompt | 1 817 tok/s | 85.4 tok/s | **0.73** |

**这张表解释了本台账的形状**：prefill 是 decode 的 16 倍，所以
「让乘法更快」的改动（WMMA、算子优化）能在 prefill 上拿到几个百分点，
却在带宽瓶颈的 decode 上全部归零 —— 这就是下面 12 条证伪项里大多数的共同原因。

> 注：这是我们这台机器的最优点，**不是通用最优解**。你的硬件拓扑、模型、上下文长度不同，
> 数字会不一样。`--layer-split` 和 `--mtp-q4` 这类与显存布局强相关的尤其要自己重扫。
> 参考 [`multi-gpu-and-interconnect.md`](multi-gpu-and-interconnect.md) 里的扫描区间。