# MoE4All 对照实验：哪些能吸收，哪些不能（2026-10-17）

> 结论先行：**MoE4All（Rust + Vulkan，三级存储跑超显存 MoE）与我们的 Strata fork 是同一模型、
> 同源量化的两个极端实现。逐条对照后，内核/调度/草稿三层我们全部已经走在前面或同等位置，
> 没有可直接搬的内核优化；真正被吸收的是三条负面结论（省未来的时间）和一套方法论
> （专家 miss 曲线 + 多卡配比公式，留给下一代模型）。唯一量化的新发现是：
> native IQ pack 结构性要求 `--spec>=2`（MTP 不可关），以及生产口径 prefill 基线
> **1516.8 tok/s @ ~9600 token prompt**。**

## 1. 对方是什么

- GitHub `Headmaster218/MoE4All`，基于 kryptic.sh 的 Pure-Rust Vulkan 引擎 infr，
  Apache-2.0。定位「让游戏显卡（AMD/NVIDIA/Intel）跑超显存 MoE」，Windows 免安装。
- **同一个模型、同源量化**：推荐下载 Qwen3.8-Flash-Next `AD-4.27bpw-Q4_K_M-M64`
  （AtomicChat GGUF，33 分片）= 我们的 acq4；架构标识 `qwen4exp`。
- 三级存储：SSD GGUF → 48 GiB RAM cache → GPU 弹性专家缓存；单卡 24 GiB 跑 49.22 GiB 专家。

## 2. 它自己测的同机对照（2026-10-15，RX 7900 XTX 24G）

| 场景 | MoE4All | Strata v0.1.44.1 HIP 版 |
|---|---:|---:|
| Flash-Next 30K decode x1 | **50.44** | 30.17（+67%） |
| Flash-Next 30K x2 | 66.19 | 48.14 |
| Flash-Next 150K x1 | 44.16 | 36.52 |
| Qwen3.6-35B 30K x1 | 120.98 | 19.37 |

条件不完全对等（Strata 47.17 GiB host arena + 13 GiB GPU cache、无 48 GiB RAM 预算），
但同硬件同模型，方向性结论：**同机下 MoE4All 的内核效率不低于 Strata HIP 版**。
我们的生产数字（双 V100 + 100% 常驻 + MTP 123 tok/s）更高，靠的是系统级配置
（双卡 / 零 miss / 接受率 0.97），不是内核效率——折算单卡等效后两者相当。

## 3. 逐条吸收对照（核心表）

| MoE4All 的发现（含实测） | Strata 现状 | 处置 |
|---|---|---|
| IQ kernel：mask 检查在 codebook 初始化**之后**，无效 group 白初始化（它把 mask 前提，单行 +0.6%） | `s2_expert_grouped.cu:827` mask 就是内核第一行（`if (g >= *n_groups) return`） | **无增量，已排除** |
| IQ2_S 多输出行 tree variants + NR=8：VERIFY device time −45.6%；IQ3_S 的 NR2/4/8 全被否（54.5→57.0 ms） | 已有 tree 变体（`gu_grouped_t_kernel`/`down_grouped_t_kernel`）+ GU_ROWS=32 / D_ROWS=64 多行 block，对齐即默认走 tree | **无增量，已排除**；且反向验证候选 A 应集中于 IQ2_S 路径 |
| MTP VERIFY 宽度按墙钟输出选（alpha 0.69 时 3 行≈4 行） | 我们 alpha 0.97、每行 4.50 ms 线性，T=4 已在最优区 | 不动 |
| draft 末步跳过无用的 HC head/vocab/argmax/readback | 我们每步都产 id+prob（窗口最后一个候选也要 id），设计自洽；且已有 CUDA graph 复用、逐 draft 步流水（写完即取）、PLE 预取、概率门控 | **我们的 draft 实现更细，无增量** |
| MTP prompt-prime 慢 9.5%（20K：781 MiB hidden D2H2D + 全行 VERIFY + m·vocab logits） | `MtpDrafter::prefill` 是专用 graph 路径 + **纯 D2D** 拷贝 + 一次性批处理记录（E-4，对比旧循环 32K 上 ~5500 次 host round trip）+ `first_needed` 剪枝 + async 不阻塞 | **我们的实现每个维度都更优**；绝对成本量化见 §4 |
| shared-slot 融合（单行 Decode 用，多行 MTP 更慢不用） | shared expert 是独立 kernel + `native_moe_combine` + `shared_early` overlap；我们以 MTP 多行窗口为主 | 收益存疑且架构不同，不动 |
| hidden readback 放 Readback 内存（5.04→0.016 ms/cycle） | Vulkan 内存放置特性 | CUDA 不适用 |
| 层同步并发生成（双请求合计 50.2→54.3 tok/s） | 引擎有 batch slots + `--batch-mtp`，但 serve/server.py 排队；实测并发 +1.1% | 已知结论（2-2f §5.5），引擎能力在，等有多 QPS 需求再接 |
| **负面结论①**：UG 后另发 D 慢 0.1–3.1 ms，保留完整 UGD 批次 | 我们 GU/D 分 launch 但非 tier 拆分，与它一致 | 省一次试错 |
| **负面结论②**：低权重专家跳过——删 <10% 权重专家 hidden 误差 max 9.34%（L0 反例 7.75% 权重影响是高权者的 2.42 倍），重归一化 154/156 层更差；未测速度、未启用 | — | **此路已死，别碰**（省一次大实验） |
| **负面结论③**：专家缓存按角色（Gate/Up/Down）固定配额不如全局 LRU（miss +5.72%）；SSD 盲预取需先在有序 route trace 上验 precision | 我们 100% 常驻、无缓存 | 不适用，记录备查 |
| 专家 miss 曲线（744 万次真实选择回放，27 段指数 R²=0.999999：15.5 GiB→12.65%、24→3.86%、32→1.06%、48→0.10%）+ 多卡配比 argmin Σ(rᵢ/Bᵢ)·M(49.22·Vᵢ/(E·rᵢ)) | 我们双卡分层各背一半专家，miss=0（nsys 解码期专家 HtoD=0） | **方法论吸收**：曲线证明"100% 常驻"是局部最优；公式留给专家量 >2×单卡显存的下一代模型 |

## 4. 我们的量化实测（2026-10-17）

### 4.1 生产口径 prefill 基线（serve A/B，Arm A）
同二进制同参数（只复制配置改端口 8082），8K nonce prompt ×4、max_tokens=1 纯预填充：

| 轮次 | prompt_n | prompt_tps |
|---|---:|---:|
| warmup | 9677 | 1492.9 |
| 1 | 9596 | 1517.1 |
| 2 | 9589 | 1515.9 |
| 3 | 9622 | 1516.8 |

**中位 1516.8 tok/s**（离散度 0.08%，极干净）。注意 prompt 实际 ~9600 token（生成器估算偏差），
高于历史 8K=1420 的口径——与 jmnargi/Strata-V100 观察一致（长 prompt 把 GEMM 摊大）。

### 4.2 结构性发现：MTP 在这个包上不可关
Arm B（`--spec 0`）启动即被引擎拒绝：
```
strata generate: .../pack is a native (IQ) pack: it needs --native SHARD1,
--spec T (T >= 2) and --prefill CHUNK
```
⇒ **MTP 是 native IQ pack 的格式要求，不是可选插件**。"关 MTP 测 prefill 基线"这条路不存在。
（另：CLI generate 模式不支持 `--layer-split`，它只在 `--serve` 模式可用 ⇒ CLI 无法复制生产双卡配置。）

### 4.3 草稿端计时诊断：此路在 serve 模式不通（负结果，记录备查）
想量化草稿端成本，给生产加 `STRATA_DRAFT_TIMING=1` dropin（引擎 mtp.cpp:1688 每 300 个
draft round 打印一轮分解：wall = round graph + steps GPU + host gaps + tail）。
连发 80 个请求（×128 token ≈ 3000+ rounds，远超 300 阈值），引擎日志零输出。
`/proc/<engine_pid>/environ` 确认 env 已到达引擎进程 ⇒ **serve 路径走的是 chain 流水线
（`prepare_chain`/`chain_launch`/`chain_poll`），不经过 `mtp.draft()` 里带计时的分支**，
该开关只覆盖 CLI / 非 chain 路径。dropin 已撤、生产已恢复干净。
教训：① 这个诊断开关对生产形态无效，别再把时间花在它上面；② 系统d dropin 写入后
**必须先 daemon-reload 再 restart**，否则第一次重启用的是旧单元定义（本次先踩后修）。

## 5. 结论

1. **不换框架、不搬内核**：三层对照证明 Strata 在同名技术上不落后；换 Vulkan 等于扔掉
   CUDA 内核资产（下投影 86% 带宽、Q8_0 交错门）和双卡分层，换来的是可移植性——我们不需要。
2. **吸收物是负面结论和方法论**：低权重专家跳过死路、UGD 不拆、角色配额失效；
   miss 曲线与多卡公式留给下一代（专家 >64 GiB）模型。
3. **候选 A 获得精确输入**：MoE4All 的 IQ3_S NR 全否 + IQ2_S 树有效，与我们的 nsys 账
   （门/上 IQ2_S·IQ3_S 64–81% vs 下投影 IQ4_NL 86%）互相印证——病根在 IQ2_S/IQ3_S 位拆解算力，
   下一步要动的是解包算法本身，不是 grid 几何（tree 已有）。
4. 待办：候选 A（门/上解包）与候选 B（dense Q8_0 算力路径，7.5 TFLOPS≈FP32 55%）不受本次对照影响，仍然有效。
