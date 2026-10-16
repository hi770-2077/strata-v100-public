# 32 个调优开关的完整台账

> 硬件：2× Tesla V100-PCIE-32GB（GV100GL, sm_70），驱动 580.178.04，CUDA 12.8
> 模型：Qwen3.8-Flash-Next GSQ-RCO IQ3_S（48 层 / 512 专家 top-10 / GDN+QSA 混合 / PLE n-gram 层）
> 引擎：Strata v0.1.43 + sm_70 实验构建
> 全部数字来自三轮 ABAB 交替（同一二进制，只切环境变量），best-of-3 中值，引擎日志交叉核对。
> **测量协议见 [`how-we-measure.md`](how-we-measure.md) —— 换机器复现前请先读。**
>
> ⚠️ **2026-10-16 生产已换档**：模型切到 **Q4_K_M 档**（dense 投影/超连接/lm_head 全部 Q8_0）、引擎升到 **v0.1.44.1**。
> 本表 1–19 行的数字大都在 IQ3_S 档 + 0.1.43 上测得，**换档前有效**；换档后的新测行会标注档位。
> IQ3_S 时代的逐窗分解账已作废，新账（Q4_K_M 档 ms/窗 = 6.98 + 4.68×T，剖析器关）见
> [`window2-2c-window-account-q4k-2026-10-16.md`](window2-2c-window-account-q4k-2026-10-16.md)。

---

## 采纳（15 项，生产在跑）

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
| **共享专家旁路流 `STRATA_SH_STREAM`**（CUDA 默认已开，2026-10-16 实测确认保留） | 关掉会 **−1.73%（T=1）/ −2.25%（T=4）** | 共享专家跑到独立流上与主线重叠。**源码注释说 HIP 上跨流 event 依赖比共享专家本身还贵（同一二进制 62.9 vs 43.4 tok/s），我们在 PCIe 双卡上不是这种情况——别照搬别人的失败结论，测了才算。** 另一个意义：它跑在旁路流上，所以剖析器的时间戳完全罩不住它（`gpu_stamp` 只对 `grp==0`、`collect_profile` 还要求 `G==1`）。见 [`window2-2e-unnamed-row-cost-2026-10-16.md`](window2-2e-unnamed-row-cost-2026-10-16.md) §2 |
| **Q8_0 进 `native_mmvq_il`**（自研：`IlQ80` traits + 派发 + sm_70 行宽表，2026-10-16 上生产） | 生产默认工作点 **+0.8~1.2%**（108.30→109.60 / 111.80→113.10 / 114.20→115.50，确定性 4 轮）；**T=4 窗 +1.6~2.1%，T=1 窗仅 +0.2%** | Q8_0（type 8）**从来不在交错门里**，而 Q4_K_M 档的 dense 投影 / 超连接 / **lm_head（644 MiB）全是 Q8_0**——换档后最热的一批 GEMV 集体吃了慢路径。补内核后 `mmvq_il_parity` 7 形状 × T1-4 × 行宽 1/2/4 **0 FAIL（逐位一致）**。表按格决定（2560×640 这类小形状交错是**输**的，留 0）。⚠️ 又一条「隔离高估」：head T=4 隔离快 22.9%，端到端只有 +2%。完整数据见 [`window2-2d-q8-0-il-2026-10-16.md`](window2-2d-q8-0-il-2026-10-16.md) |

> ⚠️ **前两项（`--mtp-hnorm` / `--host-core`）的收益集中在峰值口径，中位口径分别只有
> +1.0% 左右（噪声边缘）。** 它们的价值是「把上限推高」，不是「把日常体验推高」。
> 报这两个数字时必须同时说明口径，否则读者会误以为日常也能拿到 +3.7%/+6.5%。
> 详见 [`calibers.md`](calibers.md)。

---

## 证伪（19 项，附失败原因）

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
| `STRATA_Q8_PACKED=1`（打包 Q8_0 解码布局，Aurora 生产默认 +2.3%） | 13 形状 × 4 列数 **全部 memcmp 逐位相同**，但按生产等效合并后 dense 总量 nc1 **13.617 vs 9.976 ms（+36.5%）**，nc4 +14.1%；attn_qkv 单列 40.2 → **62.7 µs（+56%）**、lm_head 单列 +81% | 无损 qs/d 平面拆分把一个线程 34 B 连续载入拆成两条不相邻的流；大 L2 / 大扇区的卡上两条流各自完全合并，**Volta 只有 6 MB L2 + 128 B 扇区，两条流互挤**，有效带宽掉到 400 GB/s 档。同一个 flag 两种卡两种结果（同 `STRATA_SPLIT_OWN` 的教训）。数据见 [`window2-2b-dense-q4k-2026-10-16.md`](window2-2b-dense-q4k-2026-10-16.md) §2 |
| `native_mmvq_colwarp_kernel`（列分 warp，`STRATA_MMVQ_COLWARP=1`，自研） | 寄存器假设**完全正确**（80 → **38/48 寄存器、零 spill**，占用率确实上去了），但 nc=4 **20.350 vs 14.083 ms/前向（慢 44%）** | 每 warp 独立把**自己那一列的整块权重**读一遍 ⇒ nc=4 = 4 份 DRAM 流量，而 attn_qkv 单块 26.56 MB ≫ 6 MB L2，**L2 完全兜不住**。占用率与访存局部性在这个内核里直接对立，不是可以同时要的两个好处。内核已整段移除。见 [`window2-2b-dense-q4k-2026-10-16.md`](window2-2b-dense-q4k-2026-10-16.md) §3 |
| `STRATA_NO_MMVQ_ROWS2=1`（dense Q8_0 强制每块一行；上游 gfx12/sm_86 已实测更快，sm_70 未测） | **隔离基准赢 8.5%（nc2 12.053→11.034 ms）且 52 个输出全部逐位相同**，但端到端 decode **−5%**（101.35/103.45 vs 基线 106.75/108.4；回滚后复测基线 109.5/108.3 排除重启效应） | 每块一行 ⇒ grid 从 n_out/2 变 n_out，**块数翻倍**。独占 bench 里免费，生产里 dense GEMV 与 MoE 专家计算、层切分跨 PCIe 传输**并发抢同一张卡**，翻倍的块数是实打实的调度与尾部成本。**本仓库迄今最干净的「隔离基准高估」证据**：逐位相同排除了「数值不同导致下游变慢」这一整类解释，13 个百分点的差只能来自执行环境。见 [`window2-2b-dense-q4k-2026-10-16.md`](window2-2b-dense-q4k-2026-10-16.md) §4 |

**噪声内（双卡 99.7% 命中，无增益）**：`STRATA_ROUTE_RESIDENT` / `--aux-cpus auto` / `STRATA_ROUTE_TAIL_SKIP`

---

## 待验证的下一个候选

**（2026-10-16 更新，同日两次完成）**「在 Q4_K_M 档重取逐窗账」已完成，新账见
[`window2-2c-window-account-q4k-2026-10-16.md`](window2-2c-window-account-q4k-2026-10-16.md)：
**ms/窗 = 6.98 + 4.68×T**（IQ3_S 档 7.79 + 4.49T：固定 −10%，每行 +4%，
每行净增与共享专家字节 +48% 的折算几乎严丝合缝——换档的隐性代价就是共享专家）。
每行头号单项仍是**专家 VRAM 读 1.34 ms（+12%）**，dense GEMV 每行腰斩（0.41→0.18，窗口②关闭判决维持），
HC 相位对字节减半完全无感（与 `STRATA_HC_Q8=1` 的 −7.4% 互相印证，**HC 量化路线两档判死**），
lm_head 变贵（+140%/行）是换档另一笔隐性代价。
**（2026-10-16 第三轮完成）**上述两个候选都已处理，结论一正一反，详见
[`window2-2d-q8-0-il-2026-10-16.md`](window2-2d-q8-0-il-2026-10-16.md)：

- **① 专家读的跨行共享 —— 「已经实现了」这部分成立，「82% 健康」这部分撤回。**
  `resident_plan_kernel` / `expert_source.cpp` 建 GpuPlanSink 时**已按 distinct expert 去重分组**
  （窗内不重复载同一份权重），组内复用在 `GRP_NC` 批次 ⇒ **「跨窗重复读」并不存在**，这条路仍是死胡同。
  但当天稍后发现那张相位表是**两个 stage 相加**、与墙钟**没有固定换算率**（1.59 @T=1 → 1.09 @T=4），
  由此推出的 **「1.34 ms/行 ⇒ 742 GB/s ≈ 峰值 82%」同样不可用** —— 专家读健康与否，
  **目前没有证据支持任何一边**。
  **⇒ 候选①从「关闭」改挂起：等权威的墙钟/DRAM 数据（nsys）。届时若带宽确实低，要解的是单次访问效率（占用率/展开/扇区），不是去重。**
- **② lm_head —— 已转化为上面的采纳项**（真病灶是 Q8_0 被挡在交错门外，不是「延迟/切分」；
  投机解码下每个被验位置都要全词表分布，切词表没有正确性空间）。已吃满。

**⇒ ⚠️ 新头号候选已经升级为「先把秤修好」——见
[`window2-2e-unnamed-row-cost-2026-10-16.md`](window2-2e-unnamed-row-cost-2026-10-16.md)：**

**相位表是「两个 stage 相加」，不能当墙钟读。** 实测换算率 **1.59 @T=1 → 1.09 @T=4**
（T=1 时两卡更多时间在互相等，相加把等待也算进去了）⇒ **没有固定换算率**
⇒ 2-2c 的每行**绝对值**（含「专家读 1.34 ms/行」及由此换算的「742 GB/s = 82% 峰值」）**不可用作决策依据**。
**表内比较仍然可用**：跨档比值（dense 腰斩）、相位排名、同档内对比（HC 对字节减半无感）。
（⚠️ 当晚曾提出另一个解释 —— `verify.cpp:877` `G = (split_ && T>=2) ? 2 : 1` 而 `collect_profile()`
要 `G==1`、`gpu_stamp` 只对 `grp==0` ⇒ 只收了半个窗口 —— **已加桩证伪：这套配置下 G 恒为 1，
连 T=3.94 的窗口也只有 group 0 打点，因为压根没有 group 1。** 补丁作为潜在 bug 修复保留在 fork 里。）

同一轮还量到一个确定结论：**共享专家旁路流 `STRATA_SH_STREAM`（CUDA 默认开）在我们这张卡上是
净赚的，`=0` 会让 T=1 −1.73%、T=4 −2.25%** —— 源码注释说 HIP 上跨流 event 更贵（62.9 vs 43.4 tok/s），
**我们在 PCIe 双卡上不是这种情况，别照搬。** 但它只值 0.086 + 0.124×T ms，解释不了任何大项。

**拿到权威数据的手段**：**nsys 抓单个验证窗**（逐内核墙钟 + DRAM 吞吐/字节，一次拿到每个相位的真实
占比和专家读的真实带宽，不需要改代码）。**在此之前，任何针对行成本的优化都是猜，候选①保持挂起。**

原候选「iq_mmvq 访存优化：把权重流式带宽利用率从 37% 往上做」**在 Q4_K_M 档上没有对应病灶**：
Q8_0 单列实测 695–785 GB/s = 峰值的 **81–91%**（IQ3_S 档才是 37%）。反量化越简单、内核越接近纯流式。
剩下的问题是**多列塌陷**（同一份 5.78 GiB，nc=4 比 nc=1 多花 4.12 ms / +41%，字节一个没多读），
它要的是跨列共享权重读取的结构性重写——而 2026-10-16 的三个候选（打包布局 / 列分 warp / 每块一行）
全部证伪，其中一个「逐位相同 + 隔离快 8.5%」端到端反而慢 5%。
**⇒ 重写之前先想清楚怎么让它在混跑时也不亏；完整数据与机制分析见
[`window2-2b-dense-q4k-2026-10-16.md`](window2-2b-dense-q4k-2026-10-16.md)。**

IQ3_S 时代留下的两条仍然成立：

**iq_mmvq 访存优化（IQ3_S 档语境）**

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