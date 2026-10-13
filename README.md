# Strata on Tesla V100（sm_70）— 实测调优笔记与补丁

在**双 Tesla V100-32GB（Volta, sm_70）**上跑 [Strata](https://github.com/Niko1221/Strata)
推理引擎 + **Qwen3.8-Flash-Next IQ3_S**（48 层 / 512 专家 MoE）的完整实测记录。

这个仓库回答三个问题，每条都有可复算的数字：

1. **上游有个多卡 bug 会让第二张卡降速跑** —— 已修，补丁在 `patches/`，可直接提 PR。
2. **IQ3_S 在 Volta 上跑 WMMA tensor-core 反而慢 20%** —— 社区首个实测数据点，附机制解释。
3. **19 个调优开关哪些真有用、哪些有害** —— 采纳 9、证伪 10，每条带失败原因。

---

## 我们这台机器现在能跑多快

2026-10-13 实测**生产服务**（不是离线跑分），双 V100-32GB / PCIe 无 NVLink /
Qwen3.8-Flash-Next IQ3_S。每档 3 轮取中位数，**每轮换不同 prompt** 防止命中前缀缓存：

| 场景 | Prompt | 输出 | **prefill** | **decode** | MTP 接受率 |
|---|---|---|---|---|---|
| 日常对话 | 72 | 85–137 | 142 tok/s | **110.8 tok/s** | 0.88 |
| 长回答 | 2 364 | 386–597 | **997.6 tok/s** | 95.7 tok/s | 0.91 |
| 长提示词 | 14 876 | 64 | **1 704 tok/s** | 92.4 tok/s | 0.93 |
| 超长上下文 | 36 864 | 64 | **1 817 tok/s** | 85.4 tok/s | 0.73 |

三个从数字里直接读出来的结论：

- **prefill 比 decode 快约 16 倍**（1 817 vs 110.8）。prefill 吃算力、decode 吃带宽，
  这个不对称决定了下面所有取舍。
- **36K 上下文时 MTP 接受率掉到 0.73**（短上下文 0.88–0.93）。上下文越长猜得越不准，
  收益被拒稿吃掉 —— 所以我们生产上把 MTP 窗口和 `reasoning_budget` 都设了上限。
- **前缀缓存的收益比任何 flag 调优大两个数量级**：重发同一段 12.9K 提示词，
  `prompt_tokens` 从 14 876 掉到 **5**，耗时从 ~9.4 s 掉到 ~37 ms（**254×**）。
  我们折腾的所有内核/开关加起来都远不到这个量级。

完整环境配置、每一轮原始数据、以及**五个踩过的测量坑**（含"缓存前缀会悄悄毁掉你的跑分"
这个真实翻车），见 [`docs/benchmarks.md`](docs/benchmarks.md)。

### 这在社区里算什么水平？

**我们双 V100 的 110.8 tok/s decode，与社区独立第三方仓库实测的单张 RTX 4090
（24GB，280W 功耗墙）完全相同** —— 2022 年的消费级旗舰卡，被 2017 年的数据中心卡打平。
同期对照：

| 配置 | decode tok/s |
|---|---|
| **我们 2× V100-32GB** | **110.8** |
| 社区 1× RTX 4090 24GB（同模型同量化） | **110.8** |
| 社区 2× RTX 3090（262K 上下文） | ~120 |
| 社区 1× RTX 5090 96GB | 180–221 |
| 社区 1× V100 32GB（同模型，Strata） | 50–55 |
| 社区 4× V100 **带 NVLink**（IBM AC922 fork） | ~113 |

我们不是第一：5090 和多卡 NVLink 机器仍然更快。**短板在 prefill —— 我们 1 704，
带 NVLink 的 4–6 卡 V100 报到 7 357，差 4 倍**，原因是这台机器两张卡走 PCIe 无 NVLink，
层切分时每个 token 都要跨总线。这是硬件约束，调优解决不了。

每个数字都带出处链接，另附**三个最容易造成误读的对比陷阱**（同模型≠同工作量、
上下文深度会吃掉大部分差距、MTP 接受率强依赖任务类型），见
[`docs/community-comparison.md`](docs/community-comparison.md)。

> ⚠️ 这是**我们这台机器的最优点，不是通用最优解**。无 NVLink 是这套数字的前提之一；
> 有 NVLink 或换模型/换量化格式，结论都要重测。

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

## 三个最有价值的发现

### 1. 多卡 layer-split 下 WMMA 必失败（上游 bug，已修）

`cudaFuncSetAttribute`（给内核申请大块共享内存的接口）是**按 GPU 设备**生效的。
上游代码用一个 `static bool attr` 静态变量记住"设过了"，于是：

- **单卡**：第一个调用设上属性，之后都正常 → **上游基准机是单卡，所以这个 bug 从没暴露过**
- **双卡层切分**：卡 0 首次调用设上属性；卡 1 的调用**从没设过** → 直接报
  `prefill: the WMMA expert tiles could not launch`，第一个 prompt 请求就失败

补丁：把 `static bool attr` 换成按设备索引的数组，约 10 行。
详见 [`patches/`](patches/) 与 [`docs/multigpu-wmma-bug.md`](docs/multigpu-wmma-bug.md)。

### 2. IQ3_S 走 WMMA 反而慢 20%（社区首个数据点）

上游 issue 区有 313 条 volta/v100/mmq 讨论，但**没有任何人测过 V100 + IQ3_S 这个组合**。我们测了：

| 阶段 | MMQ（默认） | 全 WMMA | 结论 |
|---|---:|---:|---|
| gate/up 矩阵（IQ3_S） | 238-255 ms | 287-300 ms | **慢 20%** |
| down 矩阵（IQ4_NL） | 129-131 ms | 102-109 ms | 快 20% |

**机制**：Volta 的 MMQ 走 int8 `mma.sync`，吞吐是 FP16 的 2 倍，而量化数据每个元素只有 1 字节。
IQ3_S 的 256 宽 superblock 反量化本身就不便宜，扔到 tensor core 上反而是负担。

推论：**不能假定"上 tensor core 就更快"**——要看量化格式和矩阵形状。

### 3. 全 WMMA 净打平，混合模式才赚

两个方向的收益互相抵消，所以我们在 fork 里加了 `STRATA_PF_WMMA=2`：
**gate/up 保持 MMQ，只把 down 送上瓦片**。结果：

| 配置 | prompt tok/s |
|---|---:|
| `STRATA_PF_WMMA=0`（全 MMQ，默认） | 1081 / 1082 |
| `STRATA_PF_WMMA=1`（全 WMMA） | ~1084（净打平） |
| **`STRATA_PF_WMMA=2`（down-only）** | **1127 / 1127（+4.1~4.3%）** |

三轮 ABAB 交替、best-of-3 中值、引擎日志互证。decode 不受影响（这是预填充专用路径）。

---

## 19 个调优开关的完整台账

**采纳 9 个**（生产在跑）：

| 开关 | 收益 | 说明 |
|---|---|---|
| `STRATA_SM70_TABLE=1` | decode **+5.7%** | v0.1.41 引入的 sm_70 专用解码内核三件套。上游因为没有 V100 机器，保留为 opt-in |
| `--layer-split 25` | decode **+4.7%** | 25/48 层切分最优（扫描 23/24/25/26） |
| `--mtp-q4 all` | decode **+2.4%** | 投机解码草稿头 INT4 化，每步读取 212.9→146 MiB |
| `STRATA_PF_WMMA=2` | prompt **+4.2%** | 见上文 |
| `--kv k8v4` + `--kv-resident 49152` | 显存总开关 | 26.8K 上下文 KV 流式 97.8% 命中显存 |
| `--spec 4` / `--short-read 32` / `--prefill auto` | — | 甜点值；`prefill auto` 长 prompt 预填充 +99% |
| `fit_max_tokens` + `reasoning_budget_tokens 16384` | 正确性护栏 | 思考硬上限 16K |

**证伪 10 个**（含失败原因，省你重走）：

| 开关 | 结果 | 为什么 |
|---|---|---|
| `STRATA_PF_WMMA=1` | 净打平 | IQ3_S gate/up 慢 20%，IQ4_NL down 快 20%，互相抵消 |
| `--spec 8` | 长生成 −5.8% | 窗口专家流量增速超过收益 |
| `--short-read 0` | 5 token 读取 129→274 ms | 小批次解码窗口下更快 |
| `--prefill auto:16384` | 仍解析为 8192 | 专家缓存 100% 占满，借不出缓冲区 |
| `--expert-cache-device1/2` | 引擎拒启 | 与 layer-split 互斥（要求一张空卡，4 卡以上才玩得起） |
| `STRATA_STAGE_PIN=1` | **解码损坏** | IQ3_S 4096 prompt 后输出崩坏，严禁 |

完整台账（含另 4 个有害项）见 [`docs/flag-ledger.md`](docs/flag-ledger.md)。

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
- [`docs/community-comparison.md`](docs/community-comparison.md) — **与社区数据的横向对比**：带出处链接、同一模型/引擎的对照表、三个容易误读的陷阱
- [`docs/multigpu-wmma-bug.md`](docs/multigpu-wmma-bug.md) — 多卡 bug 的完整分析与补丁说明
- [`docs/flag-ledger.md`](docs/flag-ledger.md) — 19 个开关的完整台账与证伪原因
- [`docs/multi-gpu-and-interconnect.md`](docs/multi-gpu-and-interconnect.md) — 单卡/双卡/三卡/四卡 + NVLink 的建议矩阵
- [`docs/how-we-measure.md`](docs/how-we-measure.md) — ABAB 协议与相位表读法（**先读这个再看数字**）

---

## 关于本仓库

- 这是 **fork 的实验与记录仓库**：补丁和调优逻辑在 `strata-v100-opt`（我们自持的 Strata fork）。
- 上游项目：[Niko1221/Strata](https://github.com/Niko1221/Strata)。我们不改 Strata 的定位，只补 V100 的实测与补丁。
- 欢迎提 issue / PR，尤其 bug 补丁部分——如果它对上游有价值，请直接联系。
- 硬件：2× Tesla V100-PCIE-32GB（GV100GL），驱动 580.178.04，CUDA 12.8。