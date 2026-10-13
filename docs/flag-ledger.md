# 19 个调优开关的完整台账

> 硬件：2× Tesla V100-PCIE-32GB（GV100GL, sm_70），驱动 580.178.04，CUDA 12.8
> 模型：Qwen3.8-Flash-Next GSQ-RCO-abliterated IQ3_S（48 层 / 512 专家 top-10 / GDN+QSA 混合 / PLE n-gram 层）
> 引擎：Strata v0.1.43 + sm_70 实验构建
> 全部数字来自三轮 ABAB 交替（同一二进制，只切环境变量），best-of-3 中值，引擎日志交叉核对。
> **测量协议见 [`how-we-measure.md`](how-we-measure.md) —— 换机器复现前请先读。**

---

## 采纳（9 项，生产在跑）

| 开关 | 收益 | 实测依据与机制 |
|---|---|---|
| `STRATA_SM70_TABLE=1` | decode **+5.7%** | v0.1.41 引入的 sm_70 专用解码三件套：expert mode 8 + `mmvq_il` Volta rows 表 + `gr_fast`，每window GPU 时间 −6.8%，输出 bitwise 不变。上游因无 V100 机器保持 opt-in。**不开只有 99.61 tok/s，落在噪声内** |
| `--layer-split 25` | decode **+4.7%**（vs 24）；长生成 **+8.5%** | 逐层扫描 23=99.43 / 24=102.88 / **25=107.27** / 26=107.01 tok/s。卡 0 还背 vision encoder，25/26 是平台期，取 25。**双卡 94% 时间同忙，切分不均** |
| `STRATA_PF_WMMA=2` | prompt **+4.1~4.3%**（1127/1127 vs 1081/1082） | down-only WMMA 混合，自研模式。全网首个 IQ3_S gate/up 的 WMMA 数据点。parity 8 格式全过 |
| `--mtp-q4 all` | decode 107.4→110.0（**+2.4%**）；长生成 105.6→107.5 | 投机解码草稿头 INT4 化。接受率 85.7~86.9% → **89.6%**（不降）；草稿头每步读取 212.9→146 MiB |
| `--kv k8v4` + `--kv-resident 49152` | 26.8K 上下文 KV 流式 **97.8%** 命中显存 | 显存总开关 |
| `--spec 4` / `--short-read 32` / `--prefill auto` | `prefill auto` 长 prompt 预填充 **+99%** | 甜点值。`--spec 4` 运行态自动扩到 8 |
| `--expert-cache auto` | 专家 **100%** 显存常驻（11776/11776） | 双卡装得下所有专家 → "减少等miss"类优化理论增益为 0 |
| `fit_max_tokens` | 正确性护栏 | `max_tokens` 超剩余上下文时自动收缩（原来直接 400） |
| `reasoning_budget_tokens 16384` | 正确性护栏 | 思考硬上限 16K |

---

## 证伪（10 项，附失败原因）

| 开关 | 实测结果 | 失败原因 |
|---|---|---|
| `STRATA_PF_WMMA=1`（全 WMMA） | 净打平（~1084 vs 1081） | **IQ3_S gate/up 慢 20%**（Volta 上 MMQ 走 int8 `mma.sync`，吞吐 2× FP16，且q8_1 激活 1 字节/元素）；IQ4_NL down 快 20% → 取 down-only 混合（mode 2） |
| `--expert-cache-device1/2` | **引擎拒启** | 与 layer-split 架构**互斥**：要求一张不跑 stage 的空卡，4 卡以上才玩得起 |
| `STRATA_STAGE_PIN=1` | **解码损坏** | IQ3_S 4096 prompt 后输出崩坏。严禁 |
| `--short-read 0` | 5 个 fresh token 读取 129→274 ms | 小批次解码窗口下更快（判别法见 `scripts/bench_shortread.py`） |
| `--spec 8` | 长生成 105.6→99.5 tok/s | 窗口专家流量增速超过投机解码收益 |
| `--prefill auto:16384` | 仍解析为 8192 | 专家缓存 100% 占满，借不出缓冲区 |
| `--mtp-q4`（非 `all` 取值） | **−34%** | 与 `--mtp-q4 all` 是两回事，别混 |
| `--coupled-draft` | −24%~−43% | 旧版扫描证伪 |
| `--expert-cache-per-layer` | 有害 | 旧版扫描证伪 |
| `--spec 6` / `--mtp-max-t 8` / `--mtp-window 65536` / `--short-read 64` / `--prefill auto:32768` / `--batch` | 有害（−24%~−43%） | 旧版扫描证伪 |

**噪声内（双卡 99.7% 命中，无增益）**：`STRATA_ROUTE_RESIDENT` / `--aux-cpus auto` / `STRATA_ROUTE_TAIL_SKIP`

---

## 待验证的下一个候选

**把 llama.cpp 的 mma IQ3_S 内核路径移植进 Strata。**

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

实测：**decode ~110 tok/s · 长生成 ~107-111 tok/s · prompt 1127 tok/s**

> 注：这是我们这台机器的最优点，**不是通用最优解**。你的硬件拓扑、模型、上下文长度不同，
> 数字会不一样。`--layer-split` 和 `--mtp-q4` 这类与显存布局强相关的尤其要自己重扫。
> 参考 [`multi-gpu-and-interconnect.md`](multi-gpu-and-interconnect.md) 里的扫描区间。