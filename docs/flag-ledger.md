# 29 个调优开关的完整台账

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

## 证伪（16 项，附失败原因）

| 开关 | 实测结果 | 失败原因 |
|---|---|---|
| `STRATA_PF_WMMA=1`（全 WMMA） | 净打平（~1084 vs 1081） | **IQ3_S gate/up 慢 20%**；IQ4_NL down 快 20% → 取 down-only 混合（mode 2）。⚠️ 机制注释 2026-10-14 修正：~~「Volta 上 MMQ 走 int8 mma.sync」有误~~——**V100 没有 int8 mma.sync 硬件**（sm_75+ 才有），MMQ 路径是 CUDA core 上的 dp4a；全 WMMA 要把权重反量化成 f16 再上 tensor core，IQ3_S 的反量化（iq3s_grid 双查表 + 符号异或 + 每 4 值 scale）远比 IQ4_NL（单查表 + 乘 scale）贵，加上 f16 激活 2 字节/元素，转换开销吃掉了 f16 mma 的吞吐优势——IQ4_NL down 反量化便宜所以能 +20%。**数字是实测，机制以上述修正为准** |
| `--expert-cache-device1/2` | **引擎拒启** | 与 layer-split 架构**互斥**：要求一张不跑 stage 的空卡，4 卡以上才玩得起 |
| `STRATA_STAGE_PIN=1` | **解码损坏** | IQ3_S 4096 prompt 后输出崩坏。严禁 |
| `--short-read 0` | 5 个 fresh token 读取 129→274 ms | 小批次解码窗口下更快（判别法见 `scripts/bench_shortread.py`） |
| `--spec 8` | 长生成 105.6→99.5 tok/s | 窗口专家流量增速超过投机解码收益 |
| `--spec 2` | 日常 decode **−2.0%**（103.6→101.5 tok/s，ABAB 各 2–3 轮复现差 <0.2%）；bench −1.3%。**接受率反升 85.7%→90.6%** | 草稿少了单条更易中（选择效应），但每步吃进的 token 更少——lookup 免费草稿两个配置都有，砍掉的正是 MTP 的深预测。**接受率单独看会误导，必须与 decode 同时报**。NInfer 拆解见 [`ninfer-teardown.md`](ninfer-teardown.md) §5 |
| `--spec 1` | **引擎拒启** | native IQ pack 启动硬检查：`it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK`——「关掉投机看地板」在这个 setup 上不存在，spec 下限就是 2 |
| `STRATA_KV_ROT=1` | **无差异**（噪声内）：日常 decode 103.70→103.83（+0.1%），bench 106.9→106.7，接受率 85.8%→86.0%；**固定 prompt temp 0 输出 4 轮逐字节一致**（canary md5 全同，顺带证明引擎 temp 0 确定性） | k8v4 主路径**结构性排除**旋转（layer.cpp:704：`st.kv_rot = kv_q4 || (kv_int8 && g_kv_int8_rot)`，K8V4 只旋 V）；唯一理论通道是 MTP drafter 的 INT8 ring（mtp.cpp:336-337 创建 drafter 状态时把 globals 切纯 INT8），但 drafter 草稿本来就要过 target 模型验证，这点扰动翻不动任何接受决策。上游注释本就按包分化（NVFP4 首 token KL 0.0066→0.0051 改善，IQ2_XS 0.0022→0.0054 恶化）——IQ3_S/k8v4 属「无感」档。⚠️ **附带代价：`c.kv_rot` 进 SessionConfig 指纹（generate.cpp:7787），开启即使无增益也会废掉会话缓存**。NInfer 拆解见 [`ninfer-teardown.md`](ninfer-teardown.md) §5 |
| `--prefill auto:16384` | 仍解析为 8192 | 专家缓存 100% 占满，借不出缓冲区 |
| mma IQ3_S 内核移植（旧候选①） | **不移植**：sm_70 上无路可走 | 三行代码事实推翻了原候选的前提：① llama.cpp 的 IQ3_S mma 路径由 `TURING_MMA_AVAILABLE` 门控（common.cuh:288，sm_75+ 才有 int8 `mma.sync` m16n8k32），**Volta 根本没有 int8 mma 硬件**；② 上游自己在 sm_70 上给 IQ3_S 的就是 dp4a 路径（mmq.cuh:418 `MMQ_DP4A_TXS_Q8_0` + mmq-load-tiles.cuh:1359 的 `#else` 分支）——「移植」的对象在 V100 上不存在；③ Volta 只有 f16 mma 且只有 m8n8k4 一种形状（原始 HMMA，m16n8k16 要 sm_80+；mmf.cuh 的 Volta 分支 tile 32×8/8×4 正是以它搭建，服务 FP16 非量化路径），「IQ3_S 反量化成 f16 再走 mma」属于从零写内核而非移植，且 iq_multi_parity --bench 实测边际列算力只占 dp4a 峰值 ~10%（6.4 TOPS / ~31 T MAC/s）——没有算力瓶颈可解，反量化查表开销只会吃掉理论增益。数据见下条与 [`iq_mmvq_bench.md`](iq_mmvq_bench.md) |
| Volta MTP 验证内核（NInfer 机制 #4） | **无候选价值** | verify 窗口 T≤8 个 token 与 decode 共享 dp4a 内核，且已有多列权重复用路径。实测 IQ3_S 8192×2560：nc=1 27.4 µs（权重 9.01 MB → 329 GB/s = 37% of 900 GB/s），nc=8 73.5 µs，**边际 6.6 µs/列 vs 旧路径 20.8 µs/列（2.34×）**；生产 spec=4 的 T=4 验证窗已享受 46.5 vs 89.5 µs（1.92×）。ridge point ~35 MACs/byte vs 单列强度 2.3、8 列复用 18.6 → 深内存瓶颈区，写算力向内核（Volta 无 int8 mma，f16 mma 也帮不上权重流式部分）零收益。`STRATA_SM70_TABLE` 三件套（expert mode 8 / gr_fast / mmvq_il Volta rows 表）已覆盖验证路径仅有的 Volta 专属受益点且生产已开；il 表只放行 type 23/12/13/14/11，不覆盖 IQ3_S |
| `--mtp-q4`（非 `all` 取值） | **−34%** | 与 `--mtp-q4 all` 是两回事，别混 |
| `--coupled-draft` | −24%~−43% | 旧版扫描证伪 |
| `--expert-cache-per-layer` | 有害 | 旧版扫描证伪 |
| `--spec 6` / `--mtp-max-t 8` / `--mtp-window 65536` / `--short-read 64` / `--prefill auto:32768` / `--batch` | 有害（−24%~−43%） | 旧版扫描证伪 |
| `STRATA_HC_Q8=1`（HC 投影 Q8_0 化） | decode **−7.4%**（111.0 vs 119.9 tok/s，同二进制 ABABAB 只切 env）；质量门禁 8/8 逐字一致 | 「字节减半」假设失效：decode 不被 HC 投影流量约束（MoE 专家主导每步读取），且 q8 内核全局 partials 的单次读取比生产 staged BF16 更慢，省下的字节被算法低效吃掉还倒付。完整过程含一个 FP16 bias-15 假装成的「内核 bug」，见 [`hc-q8-2026-10-15.md`](hc-q8-2026-10-15.md) |

**噪声内（双卡 99.7% 命中，无增益）**：`STRATA_ROUTE_RESIDENT` / `--aux-cpus auto` / `STRATA_ROUTE_TAIL_SKIP`

---

## 待验证的下一个候选

**iq_mmvq 访存优化：把权重流式带宽利用率从 37% 往上做。**

背景（2026-10-14 `iq_multi_parity --bench` 实测意外发现，数据见 [`iq_mmvq_bench.md`](iq_mmvq_bench.md)）：
IQ3_S 8192×2560（= 9.01 MB 权重）单列 27.4 µs → **329 GB/s，只有 V100 HBM2 900 GB/s 的 37%**。
dp4a 算力只占 2.4%（21M MACs @ 峰值 0.67 µs / 实测 27.4 µs）——内核既没吃满带宽也没吃满算力，
是**内存并行度/占用率瓶颈**，典型可优化形态（每块仅 4 行 4.4 KB、每线程 ~34 B 加载）。
decode 每步要流式读全模型权重，该内核直接决定 decode 地板；verify 边际列同样受益。
同族对照：IQ3_XXS 351 / IQ2_XXS 275 / IQ2_S 210 / IQ1_M 180 GB/s——IQ3_S 已是族内最高，但离 70%+ 仍有距离。

方向：提高 in-flight 加载数（展开/向量化 LDG、占用率调优），不动算法语义（parity 已有多列新旧内核位级一致的基础）。
上限诚实评估：纯流式内核 70–80% peak 可达，本内核带反量化 + dp4a，现实目标 **1.3–1.5×**；
且全局 decode 还含 attention / MoE 路由 / 专家计算，内核快 1.3× ≠ decode 快 1.3×——先做全局 GPU 时间分解再定优先级。

优先级：低-中。原候选①（mma IQ3_S 移植）已证伪关闭，见上表。

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
却在带宽瓶颈的 decode 上全部归零 —— 这就是下面 16 条证伪项里大多数的共同原因。

> 注：这是我们这台机器的最优点，**不是通用最优解**。你的硬件拓扑、模型、上下文长度不同，
> 数字会不一样。`--layer-split` 和 `--mtp-q4` 这类与显存布局强相关的尤其要自己重扫。
> 参考 [`multi-gpu-and-interconnect.md`](multi-gpu-and-interconnect.md) 里的扫描区间。