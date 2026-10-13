# 多卡与互连优化建议（单卡 / 双卡 / 三四卡 / NVLink）

> 本文回答一个问题：**同样的 Qwen3.8-Flash-Next IQ3_S + Strata，换成别的卡数或互连拓扑，
> 哪些优化值得搬、哪些搬了也没用、哪些是坑。**
> 我们是双 V100-PCIe-32GB（无 NVLink），所以「双卡无 NVLink」一节最厚，
> 其余场景按证据强度给出建议，并明确标注每条结论的来源等级。

## 0. 证据等级约定（全文通用）

| 标记 | 含义 |
|---|---|
| 【实测】 | 我们这台双 V100-PCIe-32GB 生产机（Strata v0.1.43 + PR #1771 + `STRATA_PF_WMMA=2`）上的 A/B 实测 |
| 【社区】 | 上游 issue 区他人机器的实测（附 issue 编号，硬件/配置随原文，**不能直接外推到我们的驻留率/拓扑**） |
| 【推导】 | 依据上游文档行为/引擎报错/架构原理的推断，**未在本机型验证，用前先小范围试** |

> 挖矿报告的教训先摆在这：taxah92 的 V100-SXM2-16GB 流式机（专家仅 12-16% 驻留）
> 上 decode 三件套只有 **+1.4%**，而 christopherrobertbrooks-tech 的全驻留台式
> **+4.4~5.3%**——同一优化，驻留率不同收益差 3 倍以上。**社区数字必须匹配自己的
> 专家驻留率和互连拓扑再信。**

---

## 1. 先选对「第二卡模式」：三种模式互斥，选错引擎直接拒启

Strata 没有 tensor parallelism（上游 MULTI_GPU.md 原话：*"This is pipeline (layer)
parallelism, not tensor parallelism"*），多卡只有三种第二卡用法：

| 模式 | 卡间传什么 | 需要 NVLink / P2P？ | 与 layer-split | 适用场景 |
|---|---|---|---|---|
| **layer-split 层切分**（流水线） | 每个 verify 窗口一次 hand-off，几百 KB 走 pinned RAM | **否**，x1/x4 插槽都行 | —（它就是本人） | 2~4+ 卡的主力模式 |
| `--expert-cache-device1..3`（helper 专家缓存） | 专家激活输入 + 输出，pinned host 往返 | 否 | **互斥** | 4 卡以上（需要一张不跑 stage 的空卡） |
| `--peer-device N`（peer 专家缓存） | 激活 + 专家行走 P2P | **是**（NVLink 等 P2P 路径） | **互斥** | 有 NVLink 的双卡，且愿意放弃层切分 |

引擎对互斥是硬拒绝，不是警告——我们实测原文：

```
--expert-cache-remote with a layer split needs a GPU that runs no stage
```

也就是说：**双卡两台都在跑 layer-split 的 stage 时，helper cache 架构性堵死**
（部署方案 §12 结论）。4 卡 = 3 卡跑 stage + 1 张空卡才玩得起 helper 路线。
【实测】

---

## 2. 单卡场景

### 2.1 单卡直接可用（与卡数无关的通用件）

| 项 | 建议 | 证据 |
|---|---|---|
| `STRATA_SM70_TABLE=1` | **开**。sm_70 解码三件套（expert mode 8 + mmvq_il Volta rows 表 + gr_fast），−6.8% GPU/window，bitwise | 【实测】decode +5.7%；【社区】10+ 台 V100 复现（挖矿报告 §1.3） |
| `STRATA_PF_WMMA=2` | **开**（需自建，PR #1771 + 多卡修复）。WMMA 只改每卡本地的专家 GEMM 路径，与卡数无关；PR 自己的 down 相位数据（−24.9%）就来自他们的单卡基准机 | 【实测】prompt +4.1~4.3%；【社区】down 相位 −24.9%（PR #1771，5/5 轮） |
| `--spec 4` / `--spec-min-p 0.8` / `--short-read 32` / `--prefill auto` | 甜点值。这四条**思路**跨硬件通用，但数值不要照抄（部署方案 §13 原话：不要抄 V100 参数） | 【实测，双卡同口径扫描】 |
| 编译三坑 | CMake 默认拒 sm_70（要 `-DSTRATA_EXPERIMENTAL_SM60=ON`）、nvcc 不认 gcc 16（显式 `g++-14`）、可能静默编出 CPU 版（`nm -D … cudaMalloc` ≥ 6 验证） | 【实测】 |

### 2.2 单卡 32GB：驻留率下降，「减少等 miss」类优化重新有意义

我们双卡 64GB 合计驻留 11776/11776 专家（100%），所以「减少等 miss」类优化
对我们理论增益为 0——**这个结论不能外推到单卡**：

- **`STRATA_ROUTE_TAIL_SKIP`**（0.1.42 引入）：驻留 >90% 时自动关闭
  （V100 32GB 94.8% 驻留实测 0.1.42 慢 1-2%）；驻留率低的机器反而该开。
  【社区 #1884】
- **`--kv-resident` / 专家缓存调优**：单卡 32GB 装 IQ3_S 全部专家（约 25GB
  级权重 + KV + dense 副本）会很紧，驻留率可能落到 80-90% 区间，
  `--expert-cache auto` 按 profile 热度填，剩下走 CPU 池。【推导】
- **`STRATA_DISJOINT_ADAPT` / helper 类**：单卡无第二卡，不适用。

### 2.3 单卡 16GB：这个模型基本不可行

16GB 装不下 IQ3_S 的专家权重（我们选购时的结论：必须 32G 版）。
社区 justxiami 的 4×V100-**16GB** 能跑是因为 4 卡合计 64GB + NVLink 层切分。
【推导，与部署方案 §2 一致】

### 2.4 第一卡装得下就别切（opt-in）

`"split_skip_if_fits": true`（`--layer-split auto` 时）：第一卡独自装得下全部
profile 专家 + KV + draft + reserve 时就单卡跑。社区 R9700 32G + RX 9070 XT
实测：4K prompt 1776 vs 1244 tok/s（切分），decode ~60 vs ~51。
**我们双 V100 32G×2 单卡装不下，用不上；但 4×32G 或未来更大显存卡值得知道。**
【社区】

---

## 3. 双卡无 NVLink（我们的场景，重点）

### 3.1 为什么无 NVLink 也能跑满

层切分是**流水线并行**：一个 token 每窗口只跨卡一次（几百 KB，pinned RAM），
不是每层两次。PCIe 探测按每张卡自己的链路走（`--pcie-frac` 是所有卡共同的份额，
会跳过探测）。**我们两卡 94% 时间同忙，跨卡同步不是瓶颈。**
【实测，部署方案 §2】

### 3.2 layer-split 点数扫描：双卡最值钱的一个旋钮

**机理**：层切分后关键路径 = max(卡0 层耗时和, 卡1 层耗时和)。48 层同构，
24/24 看似最优，但**卡 0 还背着 vision encoder 等杂务，每层实际更慢**——
把一层挪给卡 1（23/25）才平衡。【实测】

| layer-split | decode 中值 | 相对 24 |
|---|---|---|
| 23 | 99.43 | −3% |
| 24（原生产） | 102.88 / 101.88 | 基线 |
| **25（现生产）** | **107.27 / 107.20 / 106.54** | **+4.7%** |
| 26 | 107.01 | 持平（平台期） |

连带收益：长生成 176→191 tok/s（+8.5%）、prefill 1030→1056（+2.5%）、
显存从卡 0 挪 2.3GB 到卡 1（卡1 余量 3GB vs 卡0 0.9GB）。
25≈26 说明平台期从 25 开始，取 25。【实测，部署方案 §14.8】

**非同构 / 不同带宽的机器**：社区 4×V100 基准显示**带宽比例切分 +12.9%**
（按每张卡的实际带宽占比分层，而不是对半）。我们是对称双 V100，23-26 已扫过，
25 就是最优——**对称机扫点数，非对称机先按带宽比例再微调。**【社区 #2048】

### 3.3 auto 的行为与局限：同构机别全交给 auto

`--layer-split auto`（默认）会尝试所有 placement，按「缓存能装下多少 profile
专家（热度加权）」择优。但有两个行为要注意：

1. **选卡顺序**（#1352）：auto 把**更快的卡（SM 数 × 最大时钟）放最后**——
   最后一级跑 output head + draft 层 + verify，一个 prompt chunk 要等它
   （社区实测 6K prompt：一个顺序 50s，另一个 15s）。同分数（同型号）保持
   你的顺序；手动 split 也保持。【社区 #1352 / 上游 MULTI_GPU.md】
2. **择优目标是显存装满，不是延迟均衡**：我们的扫描显示 auto 选的 24 不是
   延迟最优（25 才是）——**同构双卡建议手动扫 23-26 四点。**【实测】

### 3.4 双卡 opt-in 旋钮清单（按推荐度排序）

| 项 | 收益 | 证据 | 我们的状态 |
|---|---|---|---|
| `--pipeline-windows 2` | prefill +16.6%（社区 2×3090：1798→2097），decode 平 | 【社区 #1447】 | **生产在跑** |
| `STRATA_SPLIT_OWN=auto` | 每卡 prompt 缓冲区 ≤12% 显存时各用各的，避免借空缓存槽 | 【社区，上游 0.1.32】 | 生产在跑（dropin） |
| `STRATA_PREFILL_HELP=1` | 单 chunk prompt 让闲卡帮忙算专家行：1.1K +45.8% / 1.5K +23.6% / 2K +28.5% / 2.5K +27.4% / 3K +14.3%；≥3.3K 自动无收益 | 【社区，2×3090 无 P2P】 | 未开（我们 prompt 以长文档为主，且与 `STRATA_PF_FUSED`、`--peer-device` 不兼容） |
| `STRATA_SPLIT_MTP_BATCH=1` | draft K/V 批处理：25K +6.7% / 51K +4.3% / 104K +3.3% prefill；decode 中值 ~−3%（两次重启无法证实也无法排除） | 【社区，2×3080】 | 未开（decode 风险未排除，我们是 decode 敏感型） |
| `STRATA_STAGE_TRIM=1` | 每卡只加载自己层的 dense weights，省出的显存进专家缓存：2×MI50 专家 8819→10626，decode 39.2→41.7（+6.4%） | 【社区，2×MI50 16G】 | 未开（我们已 100% 驻留，无显存可省；输出会有数值差异，需重新验证） |

### 3.5 双卡的坑（都是我们或社区踩过的）

1. **多卡 WMMA 共享内存 bug（我们修的）**：`cudaFuncSetAttribute`（>48KB 动态
   共享内存 opt-in）用 `static bool` 只设一次——该属性**按设备/上下文**生效，
   layer-split 下第二张卡启动必失败（`prefill: the WMMA expert tiles could not
   launch`）。上游基准机是单卡所以从未暴露。**用 PR #1771 必须带我们的修复**
   （commit `eb38b43c`，生产二进制已含）。【实测，见 `PR1771_上游回报…md`】
2. **`STRATA_ADAPT_OVERLAP`**：层切分下自动关闭（0.1.43 的 overlap 回归
   不触及多卡 layer-split）。【上游文档 + 我们的确认】
3. **conversation-cache 的 parking 成本**：fresh 长 prompt（`reused_kv_bytes=0`）
   白付 2.6-3.2s parking；关掉后 81K prefill +10.5%；但多轮续话命中 68.7%，
   该开。**我们 parking 只要 150ms/12.9K tokens 且多轮复用实测有效，保持开启；
   若将来跑大批 fresh 长文档，记得 `--conversation-cache-mib 0`。**【社区 #2048 + 实测】
4. **`--vram-elastic` 与 layer-split 互斥**（引擎启动时自动关，单卡 only）。
   【上游 VRAM_ELASTIC.md】
5. **`--vram-reserve-later-mib`**：后卡的显示输出等杂务可以少留（驱动屏幕的卡
   才需要大头 reserve）。【上游文档】
6. **prompt 缓冲区的显存账**：切分下每卡各有自己的 prompt 缓冲区
   （默认 2048-token chunk 时约 1.5GB/卡；`--prefill 1024` 减半），
   第一卡的显式 `--expert-cache` 会被砍掉这部分。【上游文档】

---

## 4. 三卡 / 四卡场景

### 4.1 layer-split 扩展到 3-4 卡

- **语法**：`"layer_split": "24,36,42"`（48 层模型）= 卡0 跑 0-23、卡1 跑 24-35、
  卡2 跑 36-41、卡3 跑 42-47。auto 从 0.1.40 起支持四卡枚举。【上游文档】
- **每多一卡，每窗口多一次 round**：社区实测 RTX 2080 Ti 作为第三卡加入
  5080+3090，decode 从 84/110 掉到 68/90——**慢卡不要进 split，宁可不用。**
  【社区，上游 MULTI_GPU.md】
- **`--pipeline-windows 2` 在 4 stage 上收益更大**：4× RX 7900 XT 实测
  4K decode 59.4→67.6（+13.8%）、32K 51.8→65.6（+26.6%），greedy 输出与
  串行环逐字节相同（`STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0` 下）。
  **NVIDIA 上始终 opt-in**（AMD RDNA4 三 stage 以上才默认开）。【社区，上游 0.1.43】
- **`--batch-groups` 需要每 stage 独占卡**（部署方案 §13 同结论）。【推导】

### 4.2 helper cache 路线：4 卡起才玩得起

- **配置**：3 卡跑 stage + 1 张空卡，`--expert-cache-device1..3` 在空卡上建
  独立专家缓存；setup：`--gpu1-experts 5000 --gpu2-experts 5000 …`。
  CUDA0 仍管 dense/KV/MTP，prompt prefill 仍走 CUDA0，helper 只服务 decode。
  【上游 SECOND_GPU.md】
- **`--remote-expert-opt`**：helper 路径的优化，作者实测双 RTX 4090
  **+63% mixed / +132% code** decode、RTX 5090+4090 +28%/+63%。
  **三/四卡未经验证**，且改变浮点求和序（不保证 bitwise）。【社区，PR 作者数据】
- **`STRATA_DISJOINT_ADAPT=1`**：防止主卡自适应把 helper 已持有的专家再提升
  一份（实测约 1900-2000 重复槽），2× RTX 3080 上 decode 81.5→94.1 tok/s
  （+15.5%）。【社区 #854 背景 + PR 实测】
- **`--gpu-placement layer`**：每层固定分给一张 helper 卡，减少同层多卡切换。
  实验性。【上游文档】

### 4.3 混合架构（V100 + 新卡）

- **可以**：单二进制 `-DCMAKE_CUDA_ARCHITECTURES="70;86"`，社区 V100+A100
  混合层切分实测 OK（V100 stage −6.4% GPU/window，A100 stage 不受影响）。
  注意 auto 会把快卡放最后（§3.3）。【社区 Toastmaster9】
- **坑**：`70;60`（Volta+Pascal）编译失败（`nvcuda::` 未加 guard）——多架构
  构建先小范围验证。我们只编 sm_70，无此问题。【社区 aleesposito85】
- **每卡 ≥8GB 下限**：每卡持有一份 dense weights 副本 + 自己的 prompt 缓冲区。
  【上游文档】
- **<7.5 的老卡**：auto 不选；`--gpus` 指名强上走实验 CUDA 12 引擎
  （P40/P100 社区有跑，P100 还能当 helper 卡）。【上游 OLDER_GPUS.md + 社区 #1069/#1079】

---

## 5. 有 NVLink 的场景（先纠正预期，再谈用法）

### 5.1 NVLink 对 layer-split 基本没用——别为它买 SXM2

层切分的卡间流量是**每窗口几百 KB**，PCIe Gen3 x16 就喂饱了（我们 94% 同忙）。
社区 4×V100-16GB **NVLink** 层切分的 decode +5.6%（87.5→92.4）经归因
**几乎全部来自 expert mode 8，而不是 NVLink 本身**。我们选 PCIE 版
就是为了避开 SXM2 的 NVLink 主板成本——这个决策被社区数据侧面验证。
【实测 + 社区 justxiami】

### 5.2 NVLink 真正有用的地方：`--peer-device`

`--peer-device N` 把第二张卡变成一个**对等专家缓存**：装下 CUDA0 缓存装不下的
热度次一级专家，激活和专家行**走 P2P（NVLink）**传输，peer 自己算自己那份的
decode 窗口行、甚至 prompt 行（`--peer-prefill-rows`）。

要点：

- 与 `--layer-split`、`--expert-cache-device1..3` **互斥**（引擎原文拒绝，
  见 §1）。要 NVLink 路线就得放弃层切分，**先 A/B 再切生产**。【上游文档】
- 需要 `--expert-profile` + 已启用的专家缓存；启动参数
  `--peer-reserve-mib`（默认留 600MiB）/ `--peer-slots` / `--peer-adapt-swaps` /
  `--peer-prefill-rows`。【上游文档】
- 正确性：同一专家集跨两卡分布时，在 exactness gate 下输出与单卡逐字节一致；
  prompt 路径保持 MMQ（不兼容 fused int8 prompt 路径）。【上游文档】
- **我们没有 NVLink，没测过这条路线——以上全部是【推导】（来自上游文档），
  不是我们的实测。** 双卡有 NVLink 的机器如果 layer-split 已到平台期，
  可以拿它当第二个实验方向。

### 5.3 tensor-split：上游没有，别找

上游明确多卡是流水线（layer）并行而非张量并行。想要张量并行的收益
（NVLink 高带宽利用） currently 无路可走——这是引擎架构选择，不是配置问题。
【上游 MULTI_GPU.md】

---

## 6. 任何卡数都值得做的通用项

| 项 | 建议 | 证据 |
|---|---|---|
| **内存加满** | 我们 62→128GB：模型加载 50s→31s（−38%）、prefill +8~10%。性价比最高的一招，且与卡数无关 | 【实测】 |
| `STRATA_SM70_TABLE=1` | 所有 V100 场景通用（§2.1） | 【实测+社区】 |
| `STRATA_PF_WMMA=2` | 单卡/多卡通用（注意多卡必须带我们的 smem 修复） | 【实测】 |
| Linux 而非 Windows/WSL2 | WDDM 下专家 arena 只 pin 8GiB（更多映射进两个 GPU 上下文会被拒）；Linux 全 pin（0.1.31+）。`STRATA_ARENA_PIN_GIB=N` 可调 | 【上游文档 + 社区 #253】 |
| 验证协议 | 同二进制唯一变量、ABAB、restart 后 warmup；输出对齐用 `--pcie-frac 0 --adapt-every 0`；bit-identical 检查用 parity + 新旧二进制对比 | 【实测，README A/B 协议】 |
| 慢卡不进 split | 每多一卡每窗口多一次 round（§4.1） | 【社区】 |

---

## 7. 速查决策表

| 你的机器 | 推荐路线 |
|---|---|
| 1× V100 32G | 单卡：`STRATA_SM70_TABLE=1` + `STRATA_PF_WMMA=2` + §2.1 甜点值；驻留率掉下来再考虑 route-tail-skip 类 |
| 2× V100 PCIe 无 NVLink（= 我们） | layer-split **手动扫 23-26**（别只信 auto 的 24）+ `--pipeline-windows 2` + `STRATA_SPLIT_OWN=auto` + `STRATA_SM70_TABLE=1` + `STRATA_PF_WMMA=2` |
| 2× V100 SXM2 有 NVLink | layer-split 同样是最优路线（NVLink 用不上）；想试 `--peer-device` 必须先放弃 layer-split 并 A/B |
| 3× V100 | layer-split 三点位扫描 + `--pipeline-windows 2`；helper 需要第 4 张空卡，3 卡玩不了 |
| 4× V100 | 路线 A：layer-split 四阶段 + `--pipeline-windows 2`（收益最大）；路线 B：3 stage + 1 helper + `--remote-expert-opt` + `STRATA_DISJOINT_ADAPT=1`（实验性，三/四卡未验证） |
| V100 + 新卡混合 | 单二进制 `70;80`，layer-split，快卡放最后（或手动指定），`STRATA_SM70_TABLE=1` 只影响 V100 stage |
| 任何卡数 + 批量 fresh 长文档 | `--conversation-cache-mib 0`（省 parking）；多轮续话场景则必须开 |

---

## 附：证据出处索引

本文每条建议都标注了证据等级（【实测】/【社区】/【推导】），出处如下：

- 【实测】我们在双 V100 上的部署方案记录与扫描台账（含硬件配置、helper cache 互斥性验证、
  异硬件迁移注意事项、PR #1771 A/B 原始数据）。完整原始记录见本仓库私有版
  `strata-v100-opt` 的部署方案文档。
- 【社区】上游 issue 区与官方文档：
  - [Niko1221/Strata issue #1401](https://github.com/Niko1221/Strata/issues/1401) 评论区
    （10+ 台 V100 用户的配置讨论）、#2048（4×V100 基准）、#1884、#1352、#1447、#854；
  - 上游文档：`docs/MULTI_GPU.md`、`docs/SECOND_GPU.md`、`docs/NVIDIA_V100.md`、
    `docs/OLDER_GPUS.md`、`docs/VRAM_ELASTIC.md`；
  - 系统梳理见 [`llama-cpp-mining-notes.md`](llama-cpp-mining-notes.md)。
- 引用的社区机器（2× RTX 3090 / 2× RTX 3080 / 2× MI50 / 4× RX 7900 XT /
  4× V100 等）与我们的双 V100-PCIe 在**驻留率、带宽、架构**上均不同，
  **搬运任何数字前先按 §0 的教训核对适用边界**。
