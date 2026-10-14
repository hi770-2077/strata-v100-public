# grouped 内核对照实测：窗口① 收益为零（2026-10-14）

> 承接《decode 全局定量账与优化窗口排序》的「第一优先动作」。三个 parity bench 从**生产一致源码树**
> 重建后在真机跑完，结论：**窗口①（grouped 内核接线 +10–15%）不成立——收益为零，因为生产已经在跑
> 新内核，且新内核在生产主路径（VRAM 调用）上相对旧代际没有任何速度优势。**
> 所有数字为实测；测试形态与引擎调用形态逐字对应（见 §一.4）。

## 一、怎么跑的（可复现）

1. **二进制与生产同源**：fork `v100-opt` 分支的 `src` 子树哈希
   `c9e6f6e9c52d17bc36048828e7462653b31b2dc4`，与生产 commit（eb38b43c）的 `src` 树
   **完全一致**（`git rev-parse <ref>:src` 双边校验；两树根哈希不同仅因 docs/ 差异）。
   构建用 sm_70 生产同款配方：Ninja + g++-14 + CUDA 12.8，`-DCMAKE_CUDA_ARCHITECTURES=70
   -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON`，
   FetchContent 复用 strata-1771 的 llamacpp 源码，`TMPDIR` 指向大分区。增量构建 3 个目标 5 秒。
2. **env 必须复刻生产**：`STRATA_SM70_TABLE=1`。不带它，bench 测的是**非 Volta 路径**——
   `exp_mode()` 只在 sm_70 + 该变量非零时选 mode 8（grid + 组激活进 shared memory + SwiGLU/q8_1 融合），
   mmvq_il 的 Volta rows 表与 gr_fast 同样以它为门。生产服务该变量常开。
3. **GPU 空闲**：native bench 的 VRAM 行需要 3.7 GB 专家 arena（48 层 × 28 专家 × 2.73 MB），
   生产引擎占 29 GB/卡放不下 → 暂停生产服务 7 秒（16:56:45 stop → 16:56:52 start，模型在页缓存里）。
   重启后健康检查：6 token 生成 decode **109.1 tok/s**，与 110.8 基线一致。
4. **形态对应**（关键）：引擎 verify 路径的 VRAM 调用传 `grid_groups=0`，PCIe 调用传
   `kPcieGroupRows=4`——与 bench 三行的 gy 逐字相同。引擎**没有内核代际开关**
   （`STRATA_GROUPED_V1` 默认关、`moe_grouped_select_old` 默认 auto），
   所以「new」列就是生产在跑的配置，「v1」列是反事实对照。

## 二、native_grouped_parity --bench（IQ3_S gate/up × IQ4_XS down，2560×768）

48 次调用捕进一张 CUDA 图（= 一个验证窗口的层数），每层有**自己**的 28 个专家（权重从 DRAM 流读，
与引擎一致）；v1/new 交替发射各 40 对取中位数（前 20 对预热）。

| 调用形态（引擎里的名字） | gy | v1 µs | new µs | new 加速比 |
|---|---|---|---|---|
| no group（pcie_frac 0 时的 PCIe 调用） | 4 | 23.89 | 6.42 | **3.72×** |
| 1 组 2 条目（PCIe 调用） | 4 | 45.99 | 23.49 | **1.96×** |
| 28 组 40 条目（**VRAM 调用，生产主路径**） | 0 | 196.05 | 195.88 | **1.00×** |

- 正确性：10 种 gate/up × 5 种 down 共 55 个格式对（512×256）+ IQ3_S/IQ4_NL 2560×640 +
  IQ3_S/IQ4_XS 2560×768，每个 84 次调用覆盖 7 种 grid_groups × 2 种 base——**全部与 v1 bitwise
  相等，0 failures**。
- VRAM 调用的有效带宽：28 × 2.73 MB / 195.88 µs = **391 GB/s**（单卡实测峰值 858 GB/s 的 46%，
  见 §五）。

## 三、iq_multi_parity --bench

**iq_mmvq 8192×2560**（µs/次，old/new；old = 逐列 decode，new = 一次解完）：

| 类型 | nc=1 | nc=2 | nc=4 | nc=8 |
|---|---|---|---|---|
| IQ2_XXS | 19.8 / 19.6 | 35.9 / 26.4 | 67.1 / 38.5 | 131.0 / 65.3 |
| IQ2_XS | 23.1 / 22.5 | 41.9 / 29.5 | 78.9 / 42.2 | 155.7 / 68.6 |
| IQ3_XXS | 22.9 / 22.8 | 39.5 / 29.7 | 72.8 / 41.8 | 141.9 / 68.7 |
| IQ3_S | 26.8 / 27.3 | 47.4 / 33.9 | 102.5 / 52.1 | 171.6 / 73.6 |
| IQ2_S | 25.7 / 25.5 | 45.6 / 32.4 | 85.6 / 45.0 | 164.4 / 71.9 |
| IQ1_M | 20.2 / 20.2 | 36.9 / 26.7 | 70.0 / 39.8 | 137.4 / 66.7 |

nc=1（纯 decode）新旧持平；nc≥2 新代际全格式快 1.4–2.3×。

**grouped 2560×640 × Q2_0 down，16 组 × m 条目**（µs/次，old/new）：

| gate/up | m=1 | m=8 | m=8 加速比 |
|---|---|---|---|
| IQ2_XXS | 65.2 / 65.0 | 193.8 / 192.6 | 1.01× |
| IQ2_XS | 84.8 / 72.8 | 446.2 / 210.0 | **2.13×** |
| IQ3_XXS | 71.6 / 71.7 | 213.2 / 212.7 | 1.00× |
| IQ3_S | 75.3 / 75.5 | 213.3 / 212.3 | 1.00× |
| IQ2_S | 67.5 / 67.8 | 195.4 / 196.2 | 1.00× |
| IQ1_M | 76.9 / 69.3 | 403.8 / 199.4 | **2.02×** |

**只有 IQ2_XS 和 IQ1_M 受益；生产 pack 的 gate/up 类型（IQ2_S×20 层 / IQ3_XXS×17 /
IQ3_S×10 / IQ4_XS×1）一个都不受益。** 0 failures（各格式 67×2560 与 5×1024 bitwise、
grouped bitwise、q8_1 有限性钳制检查全过）。

## 四、s2_expert_grouped_parity --bench（Q2_0 专家 blob，H=2560/FF=640）

| 形态 | previous µs（GB/s） | new µs（GB/s） | 加速比 |
|---|---|---|---|
| moe_hit_grouped_s2, 9 hits | 77.4（161） | 38.1（327） | 2.03× |
| 5 hits | 52.5（132） | 28.3（245） | 1.86× |
| 3 hits | 37.0（112） | 24.4（170） | 1.52× |
| 2 hits | 30.3（91） | 22.1（125） | 1.37× |
| 1 hit | 24.3（57） | 19.9（69） | 1.22× |
| moe_grouped_s2, 4 tokens 无共享（40 组） | 144.7（382） | 90.6（611） | 1.60× |
| 4 tokens 30% 共享（30 组） | 126.4（333） | 80.3（525） | 1.58× |
| 8 tokens 30% 共享（60 组） | 223.0（374） | 141.2（590） | 1.58× |
| 8 tokens 全共享（10 组） | 146.4（94） | 88.0（157） | 1.66× |

0 failures：新旧 bitwise 一致，双精度 host 参考逐行核对 0 出界。
**注意适用范围**：s2 内核的 blob 几何是硬编码 Q2_0（2-bit 码 + fp16 scale）。本机生产是
**native IQ pack**（引擎日志 "native pack ... experts"），verify 路径走
`native_expert_grouped`；s2 这套 1.2–2.0× 属于 **Q2_0 pack**（如社区 GSQ-RCO Q2_0 pack），
对本机生产最多只可能涉及 MTP 那 1/49 层。

## 五、附带锚点：单卡读带宽实测 858 GB/s

grid-stride float4 读（1.34 GB / 0.81 GB buffer，L2 仅 6 MB）：**GPU0 859.0 GB/s、GPU1 857.1 GB/s**。
双卡聚合峰值 ≈ 1716 GB/s。由此：

- decode 有效带宽 319 GB/s = 双卡峰值的 **18.6%**——预算文档「18%」得到确认（不是带宽打满）；
- 专家 VRAM 调用 391 GB/s = 单卡峰值 **46%**（预算文档的引擎内估计 329 GB/s = 38%）——
  **还剩 ~2.2× 带宽空间，但它是 dequant 计算瓶颈，不是 launch 几何**：这正是窗口②的战场，
  不是窗口①。

## 六、窗口① 结论与排序变更

1. **生产早已跑新内核**：引擎无代际开关，默认新；`STRATA_SM70_TABLE=1` 已让 mode 8 常开。
   新内核在小编组调用上的 2–3.7× 优势**已经计入**当前 110 tok/s。
2. **生产主路径没有可赚的**：VRAM 调用 new == v1 = 196 µs（带宽/计算 bound，launch 几何无关）。
   即使能切代际也一分不赚。
3. **类型不匹配**：新 iq_mmvq 代际在 grouped 内只利好 IQ2_XS / IQ1_M，本 pack 不用。
4. **s2 红利属于 Q2_0 pack**，与本机 native IQ pack 无关。

⇒ **窗口① 关闭，预期增益 0%，成本已沉没。** 残余 caveat：48 调用图只测了 IQ3_S/IQ4_XS
一种格式对；IQ2_S×20 / IQ3_XXS×17 层的 VRAM 调用未逐一测图——但 §三 的 grouped 对照在同形状
（2560×640）下对这些类型同样是 1.00×，且 v1/new 差异是结构性的（发射几何）而非类型相关，
误判风险低。

**排序变更**：第一优先从①移到**② iq_mmvq 深度优化**（专家读 5.0 ms/前向，现 38–46% 峰值，
计算-bound，要拿到 60–70% 需重写 dequant/MAC 路径）；③ launch 间隙不变（新内核已把小编组调用
从 24–46 µs 压到 6–23 µs，剩余间隙在引擎调度层）；④⑤⑥⑦ 不变。

## 七、复现

```bash
# 构建（沙箱可做，无需 GPU）：fork 仓 + sm_70 配方，增量编 3 个 parity 目标
cd /home/shun/tmpwork/fork-build && TMPDIR=/home/shun/tmpwork ninja \
  iq_multi_parity native_grouped_parity s2_expert_grouped_parity
# 运行（真机，STRATA_SM70_TABLE=1 必须；native bench 需空闲 GPU）
STRATA_SM70_TABLE=1 LD_LIBRARY_PATH=/opt/cuda-12.8/lib64 ./<target> --bench
```

原始日志：`/home/shun/tmpwork/bench-run1.log`（iq_multi + s2）、`/home/shun/tmpwork/bench-run2.log`
（native，含服务暂停/恢复全过程）；带宽测试 `bw_test.cu` 同目录。
