# 窗口②-1：iq_mmvq 定向优化（2026-10-14）

## 结论先行

给 `native_gu_fused_kernel` 加了最小块数约束 `__launch_bounds__(256, 3)`：

- **MoE 专家调用（gu+down）加权 −3.33%**，gate/up 单独 −4.39%
- 受益的是 IQ2_S 那 20 层：IQ2_S/IQ4_NL 63.9 → 58.9 µs（−7.8%）、IQ2_S/Q2_0 55.7 → 50.8 µs（−8.8%）
- 其余 5 种组合逐位不变，两个 down 内核不受影响
- 7 个组合的 q8_1 输出 **FNV64 逐位 MATCH**
- tok=1/4/16 三档增益保持（溢出没有随循环放大）

**但要说清楚**：换算到端到端只有 **约 −0.7~1.2%**（专家分组 GEMV 占 stage 20%、MoE 侧合计 37%），
**低于路线图窗口② 的 +3–6% 目标**。这一项是"把显式硬伤修掉"，不是"把 341 GB/s 推到 60–70% 峰值"。

## 一、先纠两个测量方法错误

这两条如果没发现，后面所有优化都是在对着错的靶子打。

1. **`sudo -n env LD_LIBRARY_PATH=...` 把 `STRATA_SM70_TABLE` 从环境里剥掉了**。
   `exp_mode()` 靠它才返回 Volta 的 mode 8；剥掉后退回 `kExpModeDefault`，
   ncu 抓到的是 `native_gu_multi_kernel` + `swiglu_q8_1` + `native_down_multi_kernel`，
   **不是生产跑的 fused/LDS 内核**。修法：把变量一起透传进 sudo。
2. **之前只按 IQ3_S/Q2_0 一种组合剖**。真实模型是**逐层混合量化**，这个组合 48 层里只占 1 层。

## 二、生产真实形态（GGUF 实测）

n_embd 2560 / n_ff 640 / 512 专家 top-10 / 48 层。逐层 (gu, d) 分布：

| gu / d | 层数 | 单层 gu | 单层 down | 单层合计 |
|---|---|---|---|---|
| IQ2_S(22) / IQ4_NL(20) | 15 | 36.9 µs | 25.5 µs | 63.8 µs |
| IQ3_XXS(18) / IQ4_NL(20) | 14 | 32.3 µs | 25.9 µs | 58.6 µs |
| IQ3_S(21) / IQ4_NL(20) | 9 | 34.6 µs | 26.1 µs | 61.0 µs |
| IQ2_S(22) / Q2_0(42) | 5 | 37.6 µs | 13.4 µs | 55.8 µs |
| IQ3_XXS(18) / Q2_0(42) | 3 | 32.4 µs | 13.4 µs | 48.8 µs |
| IQ3_S(21) / Q2_0(42) | 1 | 34.6 µs | 13.4 µs | 51.1 µs |
| IQ4_XS(23) / IQ4_NL(20) | 1 | 31.4 µs | 25.8 µs | 57.8 µs |

加权：gu 34.69 µs、down 23.43 µs、合计 **59.52 µs/层**，有效带宽 **约 341 GB/s**（单卡峰值 ~859，40%）。

## 三、内核级 limiter（ncu，环境已透传）

| 内核 | 层数 | 加权 µs | 占比 | DRAM% | Regs | 块/SM 上限 | Waves/SM | 占用率 |
|---|---|---|---|---|---|---|---|---|
| down<20> IQ4_NL | 39 | 1002 | 36% | **50.9** | 48 | 5 | 1.00 | 54.6% |
| gu<22> IQ2_S | 20 | 740 | 27% | 28.8 | **92** | 2 | **1.25** | 21.6% |
| gu<18> IQ3_XXS | 17 | 549 | 20% | 44.5 | 80 | 3 | 0.83 | 28.8% |
| gu<21> IQ3_S | 10 | 346 | 12% | 43.3 | 80 | 3 | 0.83 | 28.8% |
| down<42> Q2_0 | 9 | 121 | 4% | 26.9 | 64 | 4 | 1.25 | 38.5% |

- **没有一个内核是带宽瓶颈**（最高的 down<20> 也只有 DRAM 50.9%）
- 主导 stall 是 `long_scoreboard`（全局访存延迟），gu<21> 3.47 / 其余 stall 均 <1
- gu<22> 是唯一有结构性硬伤的：92 寄存器 → 每 SM 只放 2 块 → 25% 占用率 + 200 块网格溢出到 1.25 波
- gu<18>/<21> 已是 0.83 波（200 块全部驻留），**降寄存器对它们无效**——网格就只有 200 块，
  放得下 3 块也没第 3 块可用。同理 down<20> 已是 1.00 波、5 块上限，也无处可改。

## 四、改动与 A/B

```c
// Volta, mode 8: the IQ2_S instantiation alone wants 92 registers, which caps the block at 2 per SM
// (25% occupancy) and pushes the 200-block grid to 1.25 waves.  Three blocks per SM keeps it at 80
// registers and the whole grid resident in one wave; STRATA_GU_FUSED_MINBLK=2 restores the old bounds.
#ifndef STRATA_GU_FUSED_MINBLK
#define STRATA_GU_FUSED_MINBLK 3
#endif
__global__ void __launch_bounds__(256, STRATA_GU_FUSED_MINBLK) native_gu_fused_kernel(...)
```

4 个二进制 × 7 组合 × 3 轮交叉（生产服务同卡运行，取每轮最小值压噪）：

| 变体 | gu<22> | gu<21> | gu<18> | 加权 gu | 加权合计 |
|---|---|---|---|---|---|
| 基线 | 92 regs | 80 | 80 | 34.69 µs | 59.52 µs |
| MINBLK=2 | 93 | **91** | **96** | +11.51% | **+7.82%** |
| **MINBLK=3** | **80**（溢出 24B） | 79 | 80 | **−4.39%** | **−3.33%** |
| MINBLK=4 | 64（溢出 112B） | 64 | 64 | +11.99% | **+7.42%** |

注意 MINBLK=2 **不等于基线**：`__launch_bounds__(256, 2)` 让 ptxas 把 gu<21>/<18> 的寄存器
从 80 抬到 91/96，反而慢 7.8%。所以最小块数这条旋钮是**非单调**的，只能靠实测定。

## 五、剩下的窗口② 空间

341 GB/s → 60–70% 峰值（515–600 GB/s）需要结构性改写，flag 级已经榨干：

- gu<18>/<21>（32% 的 MoE 时间）：瓶颈是**网格只有 200 块**，2.5 块/SM、18.4 warp 驻留。
  要提并行度只能细分——16 行/块（400 块，但寄存器 80 仍限 3 块 → 1.67 波，收益存疑）
  或沿 n_embd 做 split-K（需要在 q8_1 量化前做跨块归约，改动大、parity 风险高）。
- down<20>（36%）：已是 1.00 波 / 54.6% 占用，DRAM 50.9% 是最健康的一个。
  再往上要减 L1 扇区压力（当前每请求 5.3 个 32B 扇区，理想 4）或提高每 warp 的访存并行度。
- 收益上限参考：专家调用占 stage 20%（MoE 侧合计 37%）。专家调用降 30% 才换得到端到端 +6%。

## 六、端到端 A/B（重启实测，同会话双臂）

协议：同一台机器、同一 config，greedy、thinking off、固定 prompt、`max_tokens` 512，
热身 1 次后取 10 次中位数。空上下文单请求（非 32 窗长上下文场景）。

| 臂 | 二进制 | decode 中位数 | 10 次区间 |
|---|---|---|---|
| A | 新（MINBLK=3） | **162.80 tok/s** | 161.2 – 163.3 |
| B | 旧（`strata.prev`） | **162.65 tok/s** | 161.2 – 163.1 |

**差 +0.09%，在 ±0.7% 的噪声内分辨不出来。** 这不是"改动没生效"，是算术决定的：
162 tok/s ⇒ 6.2 ms/token，双卡 layer split 25 后每卡 ~24 层 ≈ 258 µs/层，
专家调用 59.5 µs 只占 **23%**；3.33% × 23% ≈ **0.77% ≈ 1.25 tok/s**，正好压在噪声地板上。

⇒ **端到端这条路已经测到极限了**：要验证窗口②的成果只能靠内核级 bench（精度 ~1%）
或相位级埋点，不能靠 tok/s。后续窗口②的改动都应该用 bench 判定，tok/s 只做"没变差"的护栏。

## 七、状态

- `v100-opt` 提交 `9b01cc68`，已 `git show` 复核
- 生产构建树 `strata-1771` 已打同一补丁，`build-cuda/strata` 已重编并**已上线**（`strata-sc117.service` 已重启验证）
- 回滚：`rm -f build-cuda/strata && cp build-cuda/strata.prev build-cuda/strata` 再重启
  （注意正在运行的 exe 不能直接覆盖，会 ETXTBSY，必须先 unlink）
- 服务启动约 30 s；`strata.new` 是新二进制的一份副本，确认稳定后可删
