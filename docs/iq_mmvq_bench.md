# iq_mmvq --bench 实测：IQ3_S 多列内核的时间曲线（2026-10-14）

> 硬件：2× Tesla V100-PCIE-32GB（sm_70），驱动 580.178.04，CUDA 12.8
> 二进制：`/home/shun/tmpwork/strata-143-src/build-cuda/iq_multi_parity`
> （源码与 fork `strata-1771` 的 `src/kernels/iq_multi_parity.cpp` 逐字节相同，已 diff 验证；
> 该树 CMAKE_CUDA_ARCHITECTURES=70 + STRATA_EXPERIMENTAL_SM60=ON）
> 运行环境：真机、GPU 0% 利用率（生产服务只占显存不占算力）、`STRATA_SM70_TABLE=1`（同生产）
> 原始日志：`/home/shun/tmpwork/iqbench.log`；复现：`./iq_multi_parity --bench`
> （`--bench` 会先跑完整 parity 校验再测时间；old = 逐列内核，new = 权重解码一次的多列内核）

## 为什么跑这个

flag-ledger 旧候选①假设「llama.cpp 的 IQ3_S 在 Volta 上走 mma tensor-core，Strata 走
CUDA-core dp4a，所以有移植空间」。代码走查推翻了这个前提（llama.cpp 的 IQ3_S mma 路径
由 `TURING_MMA_AVAILABLE` 门控，sm_75+ 才有 int8 `mma.sync`；Volta 连 int8 mma 硬件都没有）。
改用实测回答两个问题：

1. **机制 #4（Volta MTP 验证内核）**：verify 窗口 T≤8 个 token 跑完整目标模型前向，
   算力向内核有没有戏？—— 测 ncols 1→8 曲线，看时间随列数怎么长。
2. **iq_mmvq 本身离硬件极限多远**：单列时间换算带宽，看权重流式路径有没有余量。

## Parity 结果：0 failures

6 类型 × 2 形状 dense 全过（`bitwise equal to the old kernel`，IQ3_S ref rel ≤ 6.8e-08）；
8 组 grouped expert 全过（`bitwise equal to the old kernels`）；q8_1 量化器三条路径
237 finite blocks 逐位一致、3 个溢出块正确钳位。**新旧内核数值语义一致**——下面所有
「old/new」差异纯性能，无正确性风险。

## 主数据：IQ3_S dense 8192×2560（µs/call，200 次迭代）

权重规模：8192×2560 = 20.97M 值 × 110 B/256 值（3.4375 bpw）= **9.01 MB**

| ncols | old（逐列） | new（解码一次） | new/old |
|---|---|---|---|
| 1 | 27.0 | 27.4 | 1.01 |
| 2 | 47.6 | 34.1 | 0.72 |
| 3 | 68.4 | 40.9 | 0.60 |
| 4 | 89.5 | 46.5 | 0.52 |
| 5 | 110.8 | 53.2 | 0.48 |
| 6 | 131.2 | 59.1 | 0.45 |
| 7 | 151.7 | 67.1 | 0.44 |
| 8 | 172.3 | 73.5 | **0.43** |

- **old 路径严格线性**：27.0 + 20.75×(nc−1)，R² = 1.000 —— 每列重新解码一遍权重。
- **new 路径次线性**：27.4 + 6.59×(nc−1) —— 权重解码一次，边际列只付激活/输出/点积的钱。
- nc=8 时 new 是 old 的 **2.34×**。

## 派生指标

| 指标 | 数值 | 含义 |
|---|---|---|
| nc=1 带宽 | 9.01 MB / 27.4 µs = **329 GB/s** | V100 HBM2 900 GB/s 的 **37%** |
| 边际列算力 | 21.0M MACs / 6.59 µs = 3.19 T MAC/s | dp4a 峰值 ~31 T MAC/s 的 **~10%** |
| nc=1 算力占比 | 21M MACs @ 峰值 0.67 µs / 27.4 µs = **2.4%** | 单列既没吃满带宽也没吃满算力 |
| ridge point | 31 T MAC/s ÷ 900 GB/s ≈ **35 MACs/byte** | 算力/带宽平衡点 |
| 单列算术强度 | 1 MAC / 0.43 B ≈ **2.3 MACs/byte** | 深内存瓶颈区（差 15×） |
| 8 列复用强度 | 8 MAC / 0.43 B ≈ **18.6 MACs/byte** | 仍在内存瓶颈区（差 2×） |

**解读**：单列内核 37% 带宽 + 2.4% 算力 → 瓶颈是内存并行度/占用率，不是算力。
这直接判死了两件事：

1. **机制 #4 无候选价值**：verify 的 T≤8 列全落在 new 路径上，边际列 6.6 µs/列里算力只占 ~10%，
   Volta 又没有 int8 mma（f16 mma 也帮不上权重流式部分）——算力向内核零收益。
   且 spec=4 的 T=4 验证窗已经在享受 46.5 vs 89.5 µs（1.92×）。
2. **mma IQ3_S 移植不成立**：对象不存在（见 flag-ledger 证伪行）；V100 可选的只有
   「反量化到 f16 再 mma」，属从零写内核，且没有算力瓶颈可解。

## 全类型对照（dense 8192×2560，µs/call）

| 类型 | bpw | 权重 MB | nc=1 new | 带宽 GB/s | nc=8 old | nc=8 new |
|---|---|---|---|---|---|---|
| IQ2_XXS (16) | 2.0625 | 5.41 | 19.7 | 275 | 131.0 | 65.5 |
| IQ2_XS (17) | 2.3125 | 6.06 | 22.6 | 268 | 156.2 | 69.6 |
| IQ3_XXS (18) | 3.0625 | 8.03 | 22.9 | **351** | 142.3 | 69.0 |
| IQ3_S (21) | 3.4375 | 9.01 | 27.4 | 329 | 172.3 | 73.5 |
| IQ2_S (22) | 2.5625 | 6.73 | 25.5 | 264 | 164.4 | 71.8 |
| IQ1_M (29) | 1.75 | 4.59 | 20.9 | 220 | 137.5 | 67.1 |

（bpw 按 ggml-common.h 块结构核算：IQ2_XXS 66 B / IQ2_XS 74 B / IQ3_XXS 98 B /
IQ3_S 110 B / IQ2_S 82 B / IQ1_M 56 B，均每 256 值。IQ3_S 是同族带宽利用率最高的，
但离 70%+ 仍有距离——这就是 flag-ledger 新候选「iq_mmvq 访存优化」的出发点
（现实目标 1.3–1.5×，且需先做全局 GPU 时间分解）。
IQ3_XXS 351 GB/s 说明该内核家族在更小块上反而更高——小块的解码/访存 pattern 更友好，
可作为优化时的参照系。）

## Grouped expert 形状（16 组 × m 条目，2560×640，µs/call）

| 类型 | m=1 old/new | m=4 old/new | m=8 old/new |
|---|---|---|---|
| IQ2_XXS | 65.7/65.6 | 106.6/106.3 | 190.5/191.3 |
| IQ2_XS | 84.5/73.1 | 235.2/121.1 | 446.0/208.8 |
| IQ3_XXS | 71.9/72.0 | 116.3/116.8 | 214.3/214.6 |
| IQ3_S | 75.8/76.0 | 115.5/114.5 | 211.7/208.6 |
| IQ2_S | 67.7/67.7 | 108.6/108.4 | 197.0/197.4 |
| IQ1_M | 77.6/69.9 | 213.2/116.4 | 405.2/200.1 |

**值得注意**：IQ2_XS / IQ1_M 的 grouped 路径 new 内核有 1.8–2.1× 增益，但 **IQ3_S grouped
old ≈ new**（差异在噪声内）——IQ3_S 的 grouped expert 路径没有走多列权重复用
（或复用了但无增益）。若生产专家是 IQ3_S，MoE 部分的多列红利吃的不是这一份，
需另测 `native_grouped` / `s2_expert_grouped` parity 工具确认专家内核的真实曲线。

## 附录：mma.sync 硬件支持矩阵（CUDA 12.8 ptxas 编译裁决，2026-10-14）

为判定「Volta 能不能走 int8 mma 验证内核」，用 `nvcc -ptx` + `ptxas -arch=sm_XX`
对 6 个 mma 变体逐个编译（变体取自 vendored llama.cpp `mma.cuh` 真实指令串）：

| 指令 | sm_70（我们的 V100） | sm_75（Turing） | sm_80（Ampere） |
|---|---|---|---|
| `m8n8k4 f16→f32`（llama.cpp:1372，Volta 路径） | ✅ | ✅ | ✅ |
| `m16n8k8 f16→f32`（:1168） | ❌ 需 sm_75 | ✅ | ✅ |
| `m8n8k16 s8`（:929） | ❌ 需 sm_75 | ✅ | ✅ |
| `m16n8k32 s8`（:946） | ❌ 需 sm_80 | ❌ 需 sm_80 | ✅ |
| `m16n8k16 f16→f16`（:977） | ❌ 需 sm_80 | ❌ 需 sm_80 | ✅ |
| `m16n8k16 f16→f32`（:1163） | ❌ 需 sm_80 | ❌ 需 sm_80 | ✅ |

**结论：sm_70 上唯一可用的 mma.sync 是 m8n8k4 f16（Volta HMMA），int8 mma 一票没有。**
交叉验证：生产二进制（sm_70 构建）SASS 反汇编只有 `HMMA.884.F32.F32`（1920 处，
STEP0-3 四条一组），**零 IMMA**。这同时解释了 `STRATA_PF_WMMA` 收益为何温和——
K=4 的窄形状每个指令只推进 4 个 k，issue 开销占比高。
