# Strata on Tesla V100（sm_70）— 实测调优笔记与补丁

**双 Tesla V100-32GB（Volta, sm_70，无 NVLink）跑 [Strata](https://github.com/Niko1221/Strata)
+ Qwen3.8-Flash-Next IQ3_S 的全部实测数字、补丁和翻车记录。**

## How fast is it? 我们这台机器现在能跑多快

### 先说一件事：同一个配置有五个完全不同的 decode 数字

这是本项目最重要的一条测量纪律。**下面五个数字全部来自同一台机器、同一个模型、
同一套参数，只是口径不同** —— 混着报任何一个，都会被人当成造假：

| # | 口径 | 怎么测的 | decode |
|---|---|---|---:|
| 1 | **峰值** | 流式抓拍（`--until 200` 边跑边采样，破目标即定格），关思考 | **203.1** |
| 2 | **A/B 基准中值** | 5×512 token，`reasoning_effort=none`，取中位数 | **105.2** |
| 3 | **真实 Agent 任务中位** | 生产 `/v1/metrics` 最近 12 个真实请求 | **159.2** |
| 4 | **日常对话（开思考）** | 72 token prompt，开思考，生产服务 3 轮中位 | **110.8** |
| 5 | **256K 满上下文** | 260K token prompt 后生成，关思考 | **87.4** |

差异从哪来，一句话说清：**思考开关差 16.2%**（关思考 93.74 vs 开思考 78.60，
同配置同 prompt 实测），**任务类型再差一截**（结构化基准 vs 自由问答），
**上下文深度再差一截**（1K vs 256K）。

> **我们选择把 110.8 放在最显眼的位置，不是因为它最好看，而是因为它最难跑出来。**
> 峰值 203 是在关思考 + 结构化基准 + 短上下文下的抓拍；日常你打开思考问它问题，
> 拿到的就是 110 这个量级。想复现 203，照下面的口径表做；想评估日常体验，看 110。

完整口径定义、每个数字的原始数据、以及**五个踩过的测量坑**见
[`docs/benchmarks.md`](docs/benchmarks.md)。

### 生产服务实测（2026-10-13，双 V100-32GB / PCIe 无 NVLink / IQ3_S）

每档 3 轮取中位数，**每轮换不同 prompt** 防止命中前缀缓存：

| 场景 | Prompt | 输出 | **prefill** | **decode** | MTP 接受率 |
|---|---|---|---|---|---|
| 日常对话 | 72 | 85–137 | 142 tok/s | **110.8 tok/s** | 0.88 |
| 长回答 | 2 364 | 386–597 | **997.6 tok/s** | 95.7 tok/s | 0.91 |
| 长提示词 | 14 876 | 64 | **1 704 tok/s** | 92.4 tok/s | 0.93 |
| 超长上下文 | 36 864 | 64 | **1 817 tok/s** | 85.4 tok/s | 0.73 |
| **同一提示词重发（命中前缀缓存）** | 14 876 | 64 | — | **37 ms 全程** | — |

### 长上下文全档位（2026-10-10，Strata v0.1.41 + SM70_TABLE）

1K → 256K 九档实测，每档唯一 salt 前缀防缓存命中：

| Prompt | Prefill | **Decode** | TTFT | MTP |
|---:|---:|---:|---:|---:|
| 1K | 463 | **106.0** | 2.3 s | 83% |
| 8K | 952 | **101.7** | 8.7 s | 83% |
| 32K | 1 742 | **97.6** | 18.8 s | 83% |
| 64K | 1 982 | **95.5** | 33.1 s | 83% |
| 100K | 1 993 | **93.3** | 51.4 s | 83% |
| 128K | 1 990 | **92.5** | 65.9 s | 83% |
| 160K | 1 940 | **91.0** | 84.5 s | 83% |
| 200K | 1 867 | **89.3** | 109.6 s | 83% |
| **256K 顶格** | 1 756 | **87.4** | 148.6 s | 83% |

**256K 只掉 18%**，MTP 接受率全程恒定 83%，无热降频（峰值 72°C）。
对照社区单卡 V100 同型测试（Q2_0，1K→256K）：52.0 → 36.8，**剩 71% 且 84°C 热降频**。
我们的曲线明显更平 —— 而且**真正的痛点是 TTFT 不是 decode**（256K 首字要等 149 秒）。

### 优化时间线：从 33.5 到 203，每一步都有实测日志

| 时间 | 改动 | decode | 说明 |
|---|---|---:|---|
| 起点 | 社区单卡基线 | 33.5 | 参照系 |
| 10-07 | 起点（双卡未调） | 69.87 | |
| 10-08 | `--layer-split 24` + `--lookup-chain 4` | 88.12 | 见下方「三个发现」 |
| 10-09 | 切 GSQ-RCO 量化档（社区优化版） | 91.69 | 关思考 8×512 中值 |
| 10-09 | `--mtp-hnorm stream` + `--host-core last` | **190.3 峰值** | 峰值口径，172 中位 |
| 10-10 | v0.1.41 + `STRATA_SM70_TABLE=1` | **203.1 峰值** | 同口径 +6.7%，首破 200 |
| 10-11 | 128G 内存 + 风扇 PWM 200 | prefill +8~53% | decode 持平（专家已 100% 驻显存）；掉速真凶是风扇，见 [`docs/memory-and-thermals.md`](docs/memory-and-thermals.md) |
| 10-13 | `--layer-split 25` + `--mtp-q4 all` | 107→110 | 4.7% + 2.4% 两轮叠加；同轮切 v0.1.43（A/B 持平，为 NaN guard），见 [`docs/engine-upgrade-notes.md`](docs/engine-upgrade-notes.md) |
| 10-16 | 引擎 v0.1.43 → **v0.1.44.1** | 中性（+0.7%，噪声内） | 315 个上游提交换来崩溃修复与启动诊断；**prefill 块仍是 8192**——层切分下借不出缓冲区，是结构性天花板，不是旋钮问题 |
| 10-16 | 量化档 IQ3_S → **Q4_K_M** | — | **换档 = 所有内核结论作废**：dense 投影全变 Q8_0，单列带宽从 37–67% 峰值跳到 **81–91%**，原本设想的「访存优化」病灶直接消失。三个 dense 候选全部证伪，见 [`docs/window2-2b-dense-q4k-2026-10-16.md`](docs/window2-2b-dense-q4k-2026-10-16.md) |
| 10-16 | 新档重取逐窗账（引擎自报） | — | **ms/窗 = 6.98 + 4.68×T**（IQ3_S 档 7.79 + 4.49T）：固定 −10%、每行 +4%。dense 每行腰斩、HC 相位对量化无感（HC 量化路线两档判死）。⚠️ **相位表的每行绝对值当晚已被判作废**，见下行，见 [`docs/window2-2c-window-account-q4k-2026-10-16.md`](docs/window2-2c-window-account-q4k-2026-10-16.md) |
| 10-16 | **发现剖面只对半个窗口有效**（秤是坏的） | — | `verify.cpp` 里 T≥2 的验证窗拆成 2 个 group，但 `collect_profile()` 要 `G==1`、`gpu_stamp` 只对 `grp==0` ⇒ 2-2c 的每行 ms **全部作废待重测**（连带撤回「专家读 82% 峰值、候选①关闭」）。同轮实测：共享专家旁路流是净赚 **+1.7~2.2%**（HIP 上的失败结论不适用我们的 PCIe 双卡）。**下一步是修秤（覆盖所有 group 或 nsys 抓单窗），不是继续优化。** 见 [`docs/window2-2e-unnamed-row-cost-2026-10-16.md`](docs/window2-2e-unnamed-row-cost-2026-10-16.md) | **+0.8~1.2%**（T=4 窗 +2.1%） | Q8_0 **从来不在交错门里**，而 Q4_K_M 档的 dense/超连接/**lm_head 全是 Q8_0** ⇒ 换档后最热的一批 GEMV 集体走慢路径。逐位一致（7 形状 × T1-4 × 行宽 1/2/4，0 FAIL）。逐窗账也印证：每行成本 4.68 → **4.50**（−3.8%）。当轮另一条否证（2-2c 把每行净增判给共享专家，站不住——它按窗口批量、与 T 无关）仍然成立；但「专家读 82% 峰值、健康」已随下行撤回。见 [`docs/window2-2d-q8-0-il-2026-10-16.md`](docs/window2-2d-q8-0-il-2026-10-16.md) |
| 10-16 | **nsys 逐内核账本（权威秤）** | — | 前三次全废的根因：驱动 ≥11.7 时 `--cuda-graph-trace` 默认 `graph`，**CUDA graph 内部内核一条都不收**，而验证窗正是 graph ⇒ 抓到的「解码」其实是加载段（24 token 与 200 token 结果一模一样）。加 `--cuda-graph-trace=node` 后：每多一行草稿 **4.50 ms**（生产）＝ 路由专家 1.32 + dense Q8_0 1.04 + 超连接 0.59 + 其余 0.36 ms。**候选①改判为「只针对门/上投影」**：下投影 IQ4_NL 已 775 GB/s = 峰值 **86%**（吃到顶），门/上 IQ2_S·IQ3_S 只有 **571–691 GB/s = 64–81%**，差 15–35%，预估 **2–4%**，病根是位拆解不是访存。解码期**无** PCIe 专家缺失。见 [`docs/window2-2f-nsys-kernel-budget-2026-10-16.md`](docs/window2-2f-nsys-kernel-budget-2026-10-16.md) |

**注意 10-11 那一行：加内存让 prefill 涨 8~10%、加载快 38%，decode 一点没动。**
原因不是白花钱 —— 专家早已 100% 驻留显存，decode 主路径根本不读内存。
「加内存 = 全面提升」是想当然，实测只解决「读不到」，不解决「算不快」。

### 这在社区里算什么水平？

**我们双 V100 的 110.8 tok/s decode（日常口径），与社区独立第三方实测的单张
RTX 4090（24GB，280W 功耗墙）完全相同** —— 2022 年的消费级旗舰卡，被 2017 年的
数据中心卡打平。同期对照：

| 对照 | decode tok/s |
|---|---|
| **我们 2× V100-32GB（2017 年卡，无 NVLink）** | **110.8**（日常口径） |
| 社区 1× RTX 4090 24GB（同模型同量化） | **110.8 —— 打平** |
| 社区 2× RTX 3090（262K 上下文） | ~120 |
| 社区 1× RTX 5090 96GB | 180–221 |
| 社区 4× V100 **带 NVLink**（IBM AC922 fork） | ~113 |
| 社区 1× V100 32GB（同模型，Strata 官方无调优） | 50–55 |

**显存更大的卡会更快。** 我们这套数字的前提是 2×32GB 走 PCIe 层切分；
`--layer-split 25` + 专家 100% 常驻已经把这台机器榨到接近上限，换成 NVLink 机器或
更大显存的卡，prefill 还能再上一个台阶（社区 4–6 卡 V100 带 NVLink 报到
**7 357 tok/s prompt**，是我们 1 817 的 4 倍）。但**在没有 NVLink 的机器上，
下面这套配置就是接近最优的解** —— 32 个开关我们全试过，采纳 13 个、证伪 19 个。

三个从数字里直接读出来的结论：

- **prefill 比 decode 快约 16 倍**（1 817 vs 110.8）。prefill 吃算力、decode 吃带宽，
  这个不对称决定了下面所有取舍 —— 也是为什么我们的算子优化在 decode 上全部归零。
- **36K 上下文时 MTP 接受率掉到 0.73**（短上下文 0.88–0.93）。上下文越长猜得越不准，
  收益被拒稿吃掉 —— 所以我们生产上把 MTP 窗口和 `reasoning_budget` 都设了上限。
- **前缀缓存的收益比任何 flag 调优大两个数量级**：重发同一段 12.9K 提示词，
  `prompt_tokens` 从 14 876 掉到 **5**，耗时从 ~9.4 s 掉到 ~37 ms（**254×**）。
  我们折腾的所有内核/开关加起来都远不到这个量级。
- **调用姿势比服务端参数更值钱**：同一份 4.4K 文档问 5 个问题，文档和问题拼在一条
  message 里 = 复用率 **0%**、8.40 s/问；拆成两条 message 加 `strata_prefix` 字段 =
  复用率 **100%**、2.00 s/问（**4.19×**）。**写法不对，再好的缓存机制也是零。**

完整环境配置、每一轮原始数据、以及**五个踩过的测量坑**（含"缓存前缀会悄悄毁掉你的跑分"
这个真实翻车），见 [`docs/benchmarks.md`](docs/benchmarks.md)；
每个社区数字都带出处链接，附**四个最容易造成误读的对比陷阱**（头一个就是口径错配），见
[`docs/community-comparison.md`](docs/community-comparison.md)。

**为什么同一个配置有五个速度数字、以及每个口径怎么测**，见
[`docs/calibers.md`](docs/calibers.md) —— 这是本项目最核心的一条测量纪律：
一个不带口径的 decode 数字不是数字。

**这些数字是怎么一个个试出来的**：29 个开关跑了 7 轮，其中 **8 次我们自己的初始假设
被自己的数据杀死**（"降频 = 散热不够" → 锁 1250 反而更慢；"并发一定更快" → 4 并发比
单请求慢；"内存大一定快" → decode 完全持平）。完整的自我推翻记录 + 六步可复用实验协议见
[`docs/methodology.md`](docs/methodology.md) ——
**一个什么都没采纳的调优 session 不是浪费的 session；一个不知道为什么采纳的才是。**

**我们的短板也写在这**：prefill 1 704 vs 带 NVLink 多卡 V100 的 7 357，差 4 倍。
原因是这台机器两张卡走 PCIe 无 NVLink，层切分时每个 token 都要跨总线。
**这是硬件约束，调优解决不了**，别按我们的 prefill 数字去推断 NVLink 机器。

> ⚠️ 这是**我们这台机器的最优点，不是通用最优解**。无 NVLink 是这套数字的前提之一；
> 有 NVLink 或换模型/换量化格式，结论都要重测。

## 四个最有价值的发现（都有补丁 / 有数据）

1. **上游多卡 bug：layer-split 下 WMMA 必失败**，第二个 prompt 请求就报错
   `prefill: the WMMA expert tiles could not launch`。根因是 `cudaFuncSetAttribute`
   按 GPU 设备生效，上游却用一个 `static bool` 记住"设过了"—— **上游基准机是单卡，
   所以这个 bug 从没暴露过**。补丁约 10 行，在 [`patches/`](patches/)，
   **可直接提 PR 给上游**。
2. **`--mtp-hnorm stream` + `--host-core last` = 峰值 +3.7% / +6.5%**，
   这是 190.3 和首破 200 的直接来源。前者把 MTP 草稿头的归一化从"四条流一起做一次"
   改成"逐流做"；后者把 host 线程挪到最后一个物理核避开 Windows 的 GPU 中断。
   ⚠️ **两项收益都集中在峰值口径**（中位只有 +1.0%，噪声边缘）—— 详见口径说明。
3. **IQ3_S 在 Volta 上走 WMMA tensor core 反而慢 20%** —— 社区首个实测数据点
   （上游 issue 区 313 条 volta/v100 讨论里没人测过这个组合），附机制解释。
4. **`STRATA_PF_WMMA=2` 混合模式 prompt +4.1~4.3%**（1081 → 1127 tok/s，
   三轮 ABAB 交替实测）—— gate/up 留 MMQ、只把 down 送上瓦片。

---

## 先读这一段（术语速查）

没接触过 Strata / llama.cpp 内核也能读懂后面的一切：

| 词 | 意思 |
|---|---|
| **sm_70** | Volta 架构的计算能力编号。V100 就是 sm_70；RTX 4090 是 sm_89。不同架构走不同的 GPU 内核。 |
| **IQ3_S / IQ4_NL** | 模型权重的量化格式（3bit/4bit）。量化=用更少的比特存权重，让模型塞进显存。 |
| **MoE / 专家** | Mixture-of-Experts：48 层里每层有 512 个"专家"小网络，每个 token 只激活 top-10 个。 |
| **MMQ** | 一种 GPU 内核实现方式（按量化格式整块矩阵乘法）。Strata 的默认路径。 |
| **WMMA** | 另一种内核路径，走 tensor core（GPU 的矩阵运算单元），理论更快。 |
| **prefill（预填充）** | 处理你输入的那一段提示词，一次性算完。**吃算力**。 |
| **decode（解码）** | 一个字一个字往外吐。**吃显存带宽**。 |
| **layer-split（层切分）** | 双卡跑一个模型：卡 A 算 1-25 层，卡 B 算 26-48 层，像流水线一样接力。 |
| **专家常驻（resident）** | 把 512 个专家全部放进显存，不够就淘汰。我们的配置下全部装得下。 |
| **tok/s** | 每秒生成多少个 token。**prompt tok/s** 和 **decode tok/s** 是两回事，别混。 |
| **MTP / 投机解码** | 让模型先猜几个 token 再验证，用"猜错的代价小"换吞吐。我们的场景接受率约 90%（36K 上下文时掉到 0.73）。 |

---

## 四个发现的细节展开

### 1. 多卡 layer-split 下 WMMA 必失败（上游 bug，已修）

`cudaFuncSetAttribute`（给内核申请大块共享内存的接口）是**按 GPU 设备**生效的。
上游代码用一个 `static bool attr` 静态变量记住"设过了"，于是：

- **单卡**：第一个调用设上属性，之后都正常 → **上游基准机是单卡，所以这个 bug 从没暴露过**
- **双卡层切分**：卡 0 首次调用设上属性；卡 1 的调用**从没设过** → 直接报
  `prefill: the WMMA expert tiles could not launch`，第一个 prompt 请求就失败

补丁：把 `static bool attr` 换成按设备索引的数组，约 10 行。
详见 [`patches/`](patches/) 与 [`docs/multigpu-wmma-bug.md`](docs/multigpu-wmma-bug.md)。

### 2. `--mtp-hnorm stream` + `--host-core last`：峰值口径下的两个大项

这两个是 190.3 / 首破 200 的直接来源，也是本地实测里**单项收益最大**的两个发现。

**`--mtp-hnorm stream`**（峰值 +3.7%，接受率 +1.5pp）
Qwen4exp 的 MTP 图里有个 `pre_fc_norm_hidden`，默认 `pooled` 是**四条流一起做一次
RMS 归一化**；改成 `stream` 是**逐流各做一次**。投机解码的草稿头对归一化方式敏感，
逐流之后草稿命中率上升。

> ⚠️ **一个常见的误归因**：`--spec-min-p` 从 0.5 调到 0.8 那一档的增量，
> 社区会以为是阈值调优的功劳。我们实测 **0.70 本身在噪声内**，
> 真正的增量来自 `--mtp-hnorm stream` 这一项。两个参数同时改，收益会被记错账。

**`--host-core last`**（峰值 +6.5%，中位仅 +1.0%）
把 host 线程绑到**最后一个物理核**。Windows 会把 GPU 中断发给第一个逻辑核，
在那自旋等待的 host 线程会被反复打断。

| 口径 | 收益 |
|---|---|
| 峰值 | **+6.5%**（191.4 是本机首次破 190） |
| 中位 | +1.0%（噪声边缘） |

**⚠️ 这两项的收益集中在峰值口径。** 报它们时必须同时说明口径，
否则读者会误以为日常体验也能拿到 +3.7% / +6.5% —— 实际日常只拿到 +1% 左右。
这正是 [`docs/calibers.md`](docs/calibers.md) 存在的原因。

### 3. IQ3_S 走 WMMA 反而慢 20%（社区首个数据点）

上游 issue 区有 313 条 volta/v100/mmq 讨论，但**没有任何人测过 V100 + IQ3_S 这个组合**。我们测了：

| 阶段 | MMQ（默认） | 全 WMMA | 结论 |
|---|---:|---:|---|
| gate/up 矩阵（IQ3_S） | 238-255 ms | 287-300 ms | **慢 20%** |
| down 矩阵（IQ4_NL） | 129-131 ms | 102-109 ms | 快 20% |

**机制**：Volta 的 MMQ 走 int8 `mma.sync`，吞吐是 FP16 的 2 倍，而量化数据每个元素只有 1 字节。
IQ3_S 的 256 宽 superblock 反量化本身就不便宜，扔到 tensor core 上反而是负担。

推论：**不能假定"上 tensor core 就更快"**——要看量化格式和矩阵形状。

### 4. 全 WMMA 净打平，混合模式才赚

两个方向的收益互相抵消，所以我们在 fork 里加了 `STRATA_PF_WMMA=2`：
**gate/up 保持 MMQ，只把 down 送上瓦片**。结果：

| 配置 | prompt tok/s |
|---|---:|
| `STRATA_PF_WMMA=0`（全 MMQ，默认） | 1081 / 1082 |
| `STRATA_PF_WMMA=1`（全 WMMA） | ~1084（净打平） |
| **`STRATA_PF_WMMA=2`（down-only）** | **1127 / 1127（+4.1~4.3%）** |

三轮 ABAB 交替、best-of-3 中值、引擎日志互证。decode 不受影响（这是预填充专用路径）。

---

## 32 个调优开关的完整台账

**采纳 13 个**（生产在跑）：

| 开关 | 收益 | 说明 |
|---|---|---|
| `--mtp-hnorm stream` | **峰值 +3.7%** | MTP 草稿头归一化逐流化，接受率 +1.5pp。见「发现 2」 |
| `--host-core last` | **峰值 +6.5%** | host 线程避开 Windows GPU 中断；中位仅 +1.0%（噪声边缘） |
| `STRATA_SM70_TABLE=1` | decode **+5.7%** | v0.1.41 引入的 sm_70 专用解码内核三件套。上游因为没有 V100 机器，保留为 opt-in |
| `--layer-split 25` | decode **+4.7%** | 25/48 层切分最优（扫描 23/24/25/26） |
| `--mtp-q4 all` | decode **+2.4%** | 投机解码草稿头 INT4 化，每步读取 212.9→146 MiB |
| `STRATA_PF_WMMA=2` | prompt **+4.2%** | 见上文 |
| `--kv k8v4` + `--kv-resident 49152` | 显存总开关 | 26.8K 上下文 KV 流式 97.8% 命中显存 |
| `--spec 4` / `--short-read 32` / `--prefill auto` | — | 甜点值；`prefill auto` 长 prompt 预填充 +99% |
| `fit_max_tokens` + `reasoning_budget_tokens 16384` | 正确性护栏 | 思考硬上限 16K |

**证伪 16 个**（含失败原因，省你重走）：

| 开关 | 结果 | 为什么 |
|---|---|---|
| `STRATA_PF_WMMA=1` | 净打平 | IQ3_S gate/up 慢 20%，IQ4_NL down 快 20%，互相抵消（机制：全 WMMA 的反量化开销吃掉 f16 mma 增益，IQ3_S 反量化比 IQ4_NL 贵） |
| `--spec 8` | 长生成 −5.8% | 窗口专家流量增速超过收益 |
| `--spec 2` | 日常 decode **−2.0%** | 接受率反升 4.9pp（选择效应）但每步吃进的 token 更少——**接受率单独看会误导，必须与 decode 同时报** |
| `STRATA_KV_ROT=1` | 无差异（噪声内） | k8v4 主路径结构性排除旋转，仅 MTP drafter 的 INT8 ring 受影响且翻不动接受决策；开启还废会话缓存 |
| mma IQ3_S 内核移植 | **不移植** | sm_70 无 int8 mma.sync 硬件（ptxas 编译矩阵 + 生产 SASS 零 IMMA 证实）；上游 IQ3_S mma 路径 Turing+ 门控，「移植对象」在 V100 上不存在 |
| Volta MTP 验证内核（int8） | **无候选价值** | verify 窗口 T≤8 已走 dp4a 多列内核（边际 6.6 µs/列 vs 旧 20.8，2.34×），算力只占峰值 ~10%——写算力向内核零收益 |
| `--short-read 0` | 5 token 读取 129→274 ms | 小批次解码窗口下更快 |
| `--prefill auto:16384` | 仍解析为 8192 | 专家缓存 100% 占满，借不出缓冲区 |
| `--expert-cache-device1/2` | 引擎拒启 | 与 layer-split 互斥（要求一张空卡，4 卡以上才玩得起） |
| `STRATA_STAGE_PIN=1` | **解码损坏** | IQ3_S 4096 prompt 后输出崩坏，严禁 |
| `STRATA_Q8_PACKED=1`（Aurora 生产默认 +2.3%） | 逐位全对，但 attn_qkv 单列 692 → **445 GB/s（慢 40%）** | 无损 qs/d 平面拆分把一次 34 B 连续载入拆成两条流；**Volta 只有 6 MB L2 + 128 B 扇区，两条流互挤**。同一个 flag 两种卡两种结果 |
| 列分 warp 内核（自研） | 寄存器 80 → **38 零 spill**（假设全对），但 nc=4 **慢 44%** | 每 warp 独立读自己那列的整块权重 ⇒ nc=4 = 4 份 DRAM 流量，单块 26.56 MB ≫ 6 MB L2。**占用率与访存局部性在这个内核里直接对立** |
| `STRATA_NO_MMVQ_ROWS2=1` | **隔离基准快 8.5%、52 个输出逐位相同，端到端却慢 5%** | 每块一行 ⇒ grid 块数翻倍；独占时免费，生产里与 MoE 计算、跨 PCIe 传输并发抢卡就要付钱。**「隔离基准会高估」迄今最干净的证据——逐位相同也免检不了** |

完整台账（含另 10 个有害项）见 [`docs/flag-ledger.md`](docs/flag-ledger.md)。

---

## 怎么自己复现

```bash
# 1. sm_70 构建配方（Ninja + gcc-14 + CUDA 12.8）
./scripts/build-sm70.sh

# 2. 正确性回归：8 种量化格式的 parity 全过
./build-cuda/pf_wmma_parity

# 3. A/B：只切环境变量，重启后对比
STRATA_KEY=xxx STRATA_URL=http://127.0.0.1:8080 python3 scripts/bench_prompt.py
```

三个必须踩对的坑（脚本注释里都写了）：CMake 默认拒 sm_70 / 静默编出 CPU 版 / nvcc 不认 gcc 16。

---

## 更多内容

- [`docs/benchmarks.md`](docs/benchmarks.md) — **生产环境实测速度**：四档场景的 prefill/decode/MTP 数据、原始 JSON、以及五个测量坑
- [`docs/calibers.md`](docs/calibers.md) — **口径分层方法论**：同一台机器为什么有五个不同的 tok/s（87~203），每个口径怎么测、什么时候用、为什么不能跨口径比
- [`docs/methodology.md`](docs/methodology.md) — **七个 Loop 六次推翻自己的初始假设**：一条优化假设怎么被数据杀死，以及六步可复用的实验协议
- [`docs/memory-and-thermals.md`](docs/memory-and-thermals.md) — **加内存/锁频/风扇三个"想当然"全部被推翻**：decode 为什么不涨、锁 1250 为什么反而慢、风扇自动模式为什么没生效（256K prefill +53% 的真解法）
- [`docs/llamacpp-engine-comparison.md`](docs/llamacpp-engine-comparison.md) — **同机同模型换引擎**：prefill 28.7→427.6 tok/s（14.9×），瓶颈真身是 embedding 的放置位置
- [`docs/engine-upgrade-notes.md`](docs/engine-upgrade-notes.md) — **v0.1.41→v0.1.43 升级实录**：307 个提交里只有三件与我们相关；A/B 持平为什么还升；一个会静默编出 CPU 版的坑
- [`docs/storage-and-io.md`](docs/storage-and-io.md) — **存储与 I/O 核查**：SATA 不是热路径；`--ple-io ram` 在内存紧张的 NVMe 机器上会「解码正常但客户端卡几十秒」；先用分析器量瓶颈
- [`docs/auditing-ai-tuning-plans.md`](docs/auditing-ai-tuning-plans.md) — **如何审计一份 AI 生成的调优方案**：14 个参数 11 个查无此物；「改专家数造四档模型」为什么物理不存在；相似匹配缓存为什么对 Agent 是系统性错误源
- [`docs/client-call-patterns.md`](docs/client-call-patterns.md) — **调用姿势 4.19×**：同一个缓存机制，写法不对复用率就是 0%；另附 2,048 token checkpoint 门槛实测
- [`docs/community-comparison.md`](docs/community-comparison.md) — **与社区数据的横向对比**：带出处链接、同一模型/引擎的对照表、四个容易误读的陷阱
- [`docs/ninfer-teardown.md`](docs/ninfer-teardown.md) — **拆解别人的 219 tok/s**：三层解构（合成语料峰值 vs 真实文本曲线 vs lossless fast path）、六个机制逐条评级——窄窗投机已实测证伪（−2.0%），KV Hadamard 旋转已实测无差异（k8v4 主路径结构性排除），Volta int8 验证内核已实测无候选价值（sm_70 无 int8 mma 硬件）
- [`docs/iq_mmvq_bench.md`](docs/iq_mmvq_bench.md) — **内核微观基准**：IQ3_S 多列内核 ncols 1→8 实测曲线（单列带宽 329 GB/s = 37% peak、边际列 6.6 vs 20.8 µs）+ mma.sync 硬件支持矩阵（sm_70 只有 m8n8k4 f16，int8 一票没有）+ 新候选「访存优化」的出发点
- [`docs/multigpu-wmma-bug.md`](docs/multigpu-wmma-bug.md) — 多卡 bug 的完整分析与补丁说明
- [`docs/flag-ledger.md`](docs/flag-ledger.md) — 32 个开关的完整台账与证伪原因
- [`docs/multi-gpu-and-interconnect.md`](docs/multi-gpu-and-interconnect.md) — 单卡/双卡/三卡/四卡 + NVLink 的建议矩阵
- [`docs/window2-2b-dense-q4k-2026-10-16.md`](docs/window2-2b-dense-q4k-2026-10-16.md) — **换量化档之后重测 dense GEMV**：Q8_0 单列 81–91% 峰值（IQ3_S 档的「37% 带宽」病灶直接消失）、三个候选全灭，含「逐位相同 + 隔离快 8.5% → 端到端慢 5%」这个反例的完整数据
- [`docs/window2-2c-window-account-q4k-2026-10-16.md`](docs/window2-2c-window-account-q4k-2026-10-16.md) — **新档逐窗账重新点名**：干净墙钟 ms/窗 = 6.98 + 4.68×T，dense 腰斩/HC 量化判死；⚠️ 相位表的每行绝对值已作废（见 2-2e）
- [`docs/window2-2e-unnamed-row-cost-2026-10-16.md`](docs/window2-2e-unnamed-row-cost-2026-10-16.md) — **发现秤是坏的**：T≥2 的验证窗拆成 2 个 group 而剖面只收集 group 0 ⇒ 2-2c 全部「ms/行」作废待重测；顺带实测共享专家旁路流净赚 +1.7~2.2%。含「三种比法只有一种对」的口径纠错表
- [`docs/window2-2f-nsys-kernel-budget-2026-10-16.md`](docs/window2-2f-nsys-kernel-budget-2026-10-16.md) — **权威秤：nsys 逐内核账本**。含三次抓废的完整坑表（`--cuda-graph-trace=node` 是关键）、每窗口/每行的相位预算、用 `grid.y` 逐个专家组量出的专家读带宽（下投影 86% 峰值 vs 门/上 64–81%），以及「解码期无 PCIe 专家缺失」「看不到跨行去重」两条附带结论
- [`docs/window2-2d-q8-0-il-2026-10-16.md`](docs/window2-2d-q8-0-il-2026-10-16.md) — **Q8_0 接回交错 MMVQ**（本轮唯一采纳项，+0.8~1.2%）+ 两条否证（「专家跨行共享」早已实现；2-2c 的「共享专家」对账被 `shared_expert_multi(n_tok)` 的批量语义否证），以及四轮端到端 A/B（含反序消漂移、nonce 方差作废）
- [`docs/how-we-measure.md`](docs/how-we-measure.md) — ABAB 协议与相位表读法（**先读这个再看数字**）

---

## 关于本仓库

- 这是 **fork 的实验与记录仓库**：补丁和调优逻辑在 `strata-v100-opt`（我们自持的 Strata fork）。
- 上游项目：[Niko1221/Strata](https://github.com/Niko1221/Strata)。我们不改 Strata 的定位，只补 V100 的实测与补丁。
- 欢迎提 issue / PR，尤其 bug 补丁部分——如果它对上游有价值，请直接联系。
- 硬件：2× Tesla V100-PCIE-32GB（GV100GL），驱动 580.178.04，CUDA 12.8。