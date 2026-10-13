# GitHub 社区挖矿报告 II：MMQ 移植路径、MoE 优化项目全景、评论区实战结论

**日期**：2026-10-13
**范围**：Strata 上游（Niko1221/Strata）issue/PR 区、llama.cpp 上游、V100 专用 fork、MoE 加速周边项目
**主线问题**：① MMQ/tensor-core 移植路径社区做到哪了；② 还有哪些 MoE 加速方案；③ 评论区大佬的实测结论

---

## 〇、头号发现：Strata 是开源项目，issue 区就是 V100 金矿

本地仓库 README 指向 **github.com/Niko1221/Strata**（英文/中文/日/德/法/西/葡七语 README，RTX 5070 12GB 为参考机）。上游最新 tag 是 **v0.1.43.1**（10-12，我们跑 0.1.43）。issue 区有 313 条 volta/v100/mmq/tensor-core 相关讨论，其中至少 10 位用户在真实 V100 上做过同口径 A/B——**我们这台双 V100 的调优问题，上游社区已经系统性做过了**。

---

## 一、MMQ / tensor-core 移植路径全解（PR #1401 + PR #1771）

### 1.1 这条路径到底是什么

PR #1401《V100 (sm_70): prompt experts on FP16 tensor cores (+10% prompt) and three existing decode kernels as defaults (-6.8% GPU per window)》：

- **Prompt 侧**：新增 `gemm_iq_f16_grouped`——MMQ 专家组每次 launch 在 **FP16 tensor core（WMMA，FP32 累加）**上跑，每个权重 superblock 用 `dq_dispatch` 反量化一次进共享内存，FP16 激活流过，下一个 tile 预取。V100-SXM2 实测：26,359-token prompt **1,467→1,608 tok/s（+9.6%）**，6,345-token **1,390→1,534（+10.3%）**。
- **Decode 侧**：三个已存在于其他架构的内核搬到 sm_70（gfx906 的 expert mode 8、带 Volta rows 表的 `native_mmvq_il`、latency-hidden hyper-connection up），**逐位相同**，GPU work per verify window **22.80→21.24 ms（-6.8%）**。

### 1.2 当前状态：decode 三件套已合并（opt-in），prompt WMMA 未合并

| 部分 | 状态 | 我们的状态 |
|---|---|---|
| decode 三件套（mode 8 / mmvq_il / gr_fast） | 0.1.41 起合并，**`STRATA_SM70_TABLE=1` 开启**（无它 sm_70 跑 0.1.40.3 的老路） | ✅ **服务 env 已设 `STRATA_SM70_TABLE=1`，已在享受** |
| prompt WMMA（`STRATA_PF_WMMA`） | FP16 commit 曾被 revert，**至今未合并**（#1401 仍 open） | ❌ 未启用 |
| IQ4_NL down lane（我们的 pack 的 down 是 type 20） | PR #1771（branch `prefill-fp16-iq4nl`，head `0742b98`，5 commits）补了 WK=128 IQ4_NL 反量化 lane + 放宽 gate（`mmq_dt==42 \|\| ==20`），未合并 | ❌ 未启用 |

### 1.3 评论区：10+ 台真实 V100 的同口径实测（这是最值钱的部分）

#1401 的 16 条评论里，**9 位不同用户、10+ 种互不重叠的配置**独立复现了 decode 三件套的收益：

| 报告者 | 机器 | 结果 |
|---|---|---|
| rewin123（PR 作者） | V100（rig 见 #1079） | 192/192 bitwise；expert VRAM calls **-13.8%～-17.1%**；decode +5-7% |
| christopherrobertbrooks-tech | V100-**PCIe**-32GB 台式，Coder IQ1_M，专家全驻留 | decode **+4.4%～+5.3%**，16/16 请求输出逐字节相同 |
| justxiami | **4× V100-16GB NVLink 层切分** | decode **+5.6%**（87.5→92.4），归因：几乎全部来自 expert mode 8 |
| noahark | V100-PCIE TCC，Windows，双路 E5 | 接受；prefill 各尺寸**完全不变**（解码内核不碰 prompt 路径） |
| yudagua | **7× V100 Windows**，IQ3_S，524K YaRN | 接受，与 SXM2 数据一致 |
| taxah92 | V100-SXM2-**16GB**，专家仅 12-16% 驻留（流式） | **仅 +1.4%**——「win stops on a streaming rig」 |
| pavelrzn | 4× V100-SXM2，专家不驻留（host page cache） | 接受（AI 辅助整理，机器和日志是他的） |
| G0-0000 | V100-SXM2，UD-IQ4_XS | mode 8 + gr_fast 生效；**IL 路径对 Q8_0  dense 是 no-op**（不在白名单），自己写了 Q8_0 IL 特化 |
| Toastmaster9 | **V100 + A100 混合层切分**（单二进制 sm_70+sm_80） | V100 stage -6.4% GPU/window，A100 stage 不受影响 |
| aleesposito85 | V100+P100 多架构 `70;60` 构建 | **编译失败**（`nvcuda::` 未加 guard）→ 多架构构建的坑，我们只编 sm_70 无此问题 |

**评论区结论**：bitwise 已验证、prompt 路径无污染、收益集中在专家全驻留的机器（**我们正是 100% 驻留**）、多卡层切分/NVLink/混合卡全部安全。上游因没有 V100 保持 opt-in，等多方确认「final code」后转正——**我们已经在用，等于提前拿到了未来的默认值**。

### 1.4 prompt WMMA 对我们的真实预期（诚实版）

- PR #1771 自己的数据：`gemm down` 相位 **-24.9%**（5/5 轮），但流式配置下专家权重拷贝等待 **+17.8%**，**端到端约 flat**。
- 我们是 100% 专家驻留（无流式等待惩罚），-24.9% 的 down 相位有机会更多落到端到端——**但没人测过我们这个配置，这是 unknown，不是承诺**。
- 代价：FP16 激活（与 q8_1 不逐位相同，同 MMQ 一类的数值差异）；要 cherry-pick 未合并 PR 代码、sm_70 自编、跑 parity、A/B。
- 上游维护者对 WMMA 路径的态度谨慎（revert 过一次），且它只对 down 投影的 K=640 tiling 生效（#1771 补的 lane 就是为这个）。

---

## 二、其他 Strata 上游 Volta 相关项逐一评估

| 项 | 内容 | 对我们的判断 |
|---|---|---|
| **#1822** MMQ in-place gate/up | CMake 把 llama.cpp 的 mmq.cuh 拷进构建树打一个补丁（负 expert stride = 专家指针表），prompt 路径就地读专家权重，省掉 group buffer 拷贝。实测 copy16_group_kernel 75.5→31.9 ms，端到端 -0.7%（PCIe Gen2 机器，拷贝占比小） | 未合并。我们的拷贝占比未知，收益可能也在这个量级。**技术漂亮（这就是用户问的「llama.cpp mmq 移植」的标准做法），优先级低** |
| **#1100** QPN8 m8n8k4 GEMV | Volta mma 的 QPN8 专家 GEMV，M≥4 才赚（10x8 = 1.21×，10x4 = 0.88×，10x1 = 0.66×）。MTP 验证窗口 M≈5 正好在 0.88×～1.21× 之间的「不确定区」 | **跳过**：wash zone + 已知 layer-split 尺寸 bug（会 over-pick stage 导致 open 失败） |
| **#1098** GDN chain-split | prompt GDN  recurrence 四累加器版，kernel +7-15%，但 recurrence 只占 32K prefill 的 ~30ms/21s | **跳过**：端到端在噪声内 |
| **#2048** 4×V100 社区基准 | ① `-DSTRATA_MMQ_KQUANTS=ON` 在 sm_70 **prefill 零增益**（瓶颈是 PCIe 流式不是 GPU 反量化）；② 带宽比例层切分 +12.9%（我们是对称双卡，25/23 已扫过是最优）；③ **conversation-cache 对 fresh 长 prompt 白付 2.6-3.2s parking**（`reused_kv_bytes=0`），关掉后 81K prefill +10.5%；多轮续话则命中 68.7% 该开 | ①跳过（我们的瓶颈也不是反量化）；②不适用；③**我们 parking 只要 150ms/12.9K tokens 且多轮复用实测有效，保持开启**；但若将来跑大批 fresh 长文档，记得 `--conversation-cache-mib 0` |
| **#690** 二卡 AMD 层切分 prefill 崩 | `no kernel image`，仅 HIP 多架构 | 不适用（我们 NVIDIA 单架构） |
| **0.1.43.1 热修** | 修复 0.1.43 的 `STRATA_ADAPT_OVERLAP` 默认开启回归（桌面卡 -19%～-26%，P100 +4-6%） | **我们不急着升**：见下 |
| **#1884 route tail skip** | 0.1.43 起：专家驻留 >90% 时自动关闭（V100 32GB 94.8% 驻留实测 0.1.42 慢 1-2%） | ✅ 0.1.43 自动生效，无需动作 |

### 2.1 关键澄清：0.1.43 的 overlap 回归与我们无关

#2010 及其 15 条评论显示 0.1.43 在 Linux CUDA 把 `STRATA_ADAPT_OVERLAP` 默认打开，部分机器 -21%。但 **Krypto-Whitehat 在代码里指出：层切分下引擎先于 flag  consulted 就把它关掉**（`generate.cpp:8278`：`multi_gpu || !stages.empty() → "not on a layer split"`）。我在我们本地 0.1.43 源码 **8278 行原文核实了这段逻辑**——我们跑 `--layer-split 25`（且 `--pipeline-windows 2` 也单独禁用 overlap），**overlap 对我们本来就是 off，回归类问题不触及我们**。0.1.43.1 对我们唯一的意义是杂项修复（batch 35%、parking watchdog），可升可不升，答案逐字节相同。

---

## 三、llama.cpp 上游的相关进展

1. **b11335（2026-10-02）PR #29753：sm_70 路由到 Turing MMVQ nwarps 表**——Volta 没有自己的 MMVQ 参数表，掉进 GENERIC 兜底（K-quant batch-1 解码 nwarps=4），而 sm_70 与 Turing 同 tuning（nwarps=2）。单张 V100 32GB PCIe + Qwen3.8-27B Q4_K_M 实测 **+3.17%**（6 个 ABBA block 全正，perplexity 逐位相同），源自 V100 专用 fork **anyei/llamacpp-v100** 的 `MMVQ_PARAMETERS_VOLTA` 表。→ Strata 侧的对应物就是我们已在跑的 per-arch IL rows 表（`STRATA_SM70_TABLE`）。**同一个「老卡需要自己的 launch 配置」结论，两边互相印证。**
2. **issue #20605：PP graph reuse 关闭导致 V100 解码 -16%**（2× V100-SXM2 NVLink，Qwen3.5-35B-A3B MoE，91.1→83.2 tok/s）——不用 CUDA graph 每步重建图，CPU 开销吃掉 16%。→ **Strata 全程 CUDA graph（zero-doorbell），这类问题我们不存在**，再次验证引擎选型。
3. **硬件事实**：Volta tensor core 只有 FP16/FP32/INT8，**没有 INT4**（Turing 起才有）。所以 V100 上「INT4 走 tensor core」是软件模拟；可行的路就是 #1401 的「反量化进 FP16 再上 WMMA」。

---

## 四、V100 专用 fork 深度分析：anyei/llamacpp-v100（11★，10-02 仍活跃）

这位 anyei 在 **X99 + 单 V100 32GB + 2× E5-2690 v4（和我们同 CPU！）** 上跑 Flash-Next GSQ-RCO 3.5bit，`-ncmoe 48` 专家全卸到 CPU。他做了两轮「Strata 技术反向移植进 llama.cpp」，文档极其硬核：

**第一轮（strata-port-plan.md，DONE）**：
- 预算拆解：plain token 49.5ms = GPU 35.6（72%）+ **每层 GPU→CPU→GPU handoff 9ms（18%）** + CPU miss 5ms（10%）
- **Item 1：把 Strata 的 AVX2 Q2_0 CPU 内核（`src/kernels/cpu/q2_avx2.cpp`）移植进 llama.cpp**（上游 x86 Q2_0 是标量，11 GB/s vs AVX2 类型 59 GB/s）→ **decode +43%（94.6→64.0 ms/MTP step），prefill 4.1×**
- Item 3：Strata 的 MoE doorbell（消除 handoff）

**第二轮（strata-followups-plan.md，BUILT+MEASURED）**：-13.7% ms/MTP step（短 prompt decode +13%），28k/95k 下 -19%（decode +20%/+30%）：
- Item 1 稳定 CUDA graph（llama.cpp 的 qwen4exp 解码图每步重建，QSA/PLE input 没有 `can_reuse`）
- Item 2 gather-based QSA：发现 **llama.cpp 的 indexer 每步每层重池化全部原始 key（32k 时每层 ~120 MiB 流量，是注意力读取的两倍且随上下文增长）**；gather 方案在 width-4 verify 下约 break-even
- Item 3 CJK-aware 草稿词表：MTP 头 248,320 id 中 **Han token 全部在 95,726 以上**，first-N 子集会把中文全砍掉；做了按 script 配额子集——**结论：净收益不足，关闭**
- Item 4 K80 gate：doorbell 可行，Kepler 镜像需 FP16 guard

**对我们核实了两件事**：
1. **我们的 indexer 不吃这个亏**：Strata 的 `native_qsa_indexer.cu` 是「块完成时算一次 pooled key 并 append 进 pooled 缓存」（C-2 batched append，pooled buffer 按 `max_cells/R+1` 预分配），每步只池化当前尾块——**正是 anyei 想让 llama.cpp 做到而我们已有的样子**。
2. **我们的草稿词表没有 CJK 问题**：`draft_vocab.bin` 106,299 个 id 跨满 0..248,319，**Han 段（95,726-151,600）覆盖 94.5%**，不是 first-N 子集。我们中文接受率 89.6% 有原因。

---

## 五、其他 MoE 优化项目全景（用户点名的）

| 项目 | 核心招 | 对我们的判断 |
|---|---|---|
| **ktransformers**（清华 MADSys，SOSP'25，18.9k★） | 冷专家卸 CPU + **AMX 内核**（单路 Xeon 21.3 TFLOPS，3.9× PyTorch）+ NUMA 感知（双路上限 +63%）+ **expert deferral**（利用残差容忍把专家计算推到下一层与 GPU attention 重叠，decode +1.45×，精度变化 <0.5%）+ CUDA Graph 消 launch 开销。已支持 Qwen3-Next、unsloth 1.58/2.51bit、IQ1_S/FP8 混合 | **不适用**：我们 E5-2686 v4 是 Broadwell，**没有 AMX**；单路无 NUMA 问题；专家 100% 驻留无需卸载。唯一可借鉴的是 expert deferral 思路，但我们的 pipeline-windows 2 已在做跨窗口重叠。**结论：架构方向不同，没有可搬的东西** |
| **MoE-Infinity**（Edinburgh，arXiv 2401.14361） | activation-aware expert offloading：**request 级** Expert Activation Matrix 追踪（model 级计数会均匀化失效）+ 跨层并发预取 + 「早期层专家较少受益于预取故优先缓存」 | 它的前提是「有 miss 要预取」。我们 100% 驻留、routing-aware prefetch 已开、expert-profile 已按路由频率排序——**它的三个核心招我们都有对应物且更进一步**。若将来到 262K 满上下文把专家挤出显存，它的 EAM 思路值得回头翻 |
| **PowerInfer / PowerInfer-2**（上交） | hot/cold neuron 分离 + 异构 XPU + neuron-cluster 级 I/O-compute pipeline；需要模型侧可预测稀疏性（TurboSparse 系列） | 面向手机/边缘；要求重训稀疏化模型。**不适用** |
| **llama.cpp `--n-cpu-moe` / Apple LLM-in-a-flash / Mixtral-offloading** | 权重流式 vs 计算卸载两派 | 见 ktransformers 判断；我们 64GB 显存 + 125GB RAM 的组合下，全驻留就是最优解，这是 anyei 的预算模型和 #2048 的实测共同确认的 |

---

## 六、仍然开放的加速选项（按预期收益排序）

1. **cherry-pick PR #1771（prefill-fp16-iq4nl）+ #1401 的三个 FP16 prompt commit**，自编 sm_70，跑 `pf_wmma_parity`，A/B prompt 速度。预期：down 投影相位 -24.9%（V100-SXM2 UD-IQ4_XS 实测），端到端多少取决于我们 100% 驻留配置下拷贝等待是否真的消失——**这是社区数据指向的最后一个有大鱼嫌疑的项，但没人测过我们的配置**。工作量：cherry-pick（5 commits，10 files）→ 编 ~5-10 分钟 → parity → 2 次重启 A/B。风险可控（独立 exe + 备份回滚）。
2. ~~QPN8 / GDN chain-split / MMQ in-place / MMQ_KQUANTS~~——本轮全部排除，理由见上表。
3. **0.1.43.1 升级**：对我们无性能意义（overlap 本来就被层切分关掉），可选。
4. 文档债：把 §1.3 评论区结论、V100 官方文档（docs/NVIDIA_V100.md）要点并入我们的部署方案文档。

---

## 七、评论区方法论沉淀（下次测速用得上）

- **同口径 A/B 是唯一可信的证据形态**：同二进制同树，只变一个 env/flag；交替 ABAB；每臂 2+ 次全新启动；报告分差和 run-to-run 分布（noahark/mmbeanz 的 p 值和 CI 是线程里被点名表扬的做法）。
- **负载匹配才能比**：yannickloth 的教训——同一构建在负载 9～44 tok/s 间摆动，只有负载匹配窗口可比。
- **「prompt 路径不受影响」要用数据证明**：解码内核 A/B 要同时报告 prefill 各尺寸（noahark 的表）。
- **收益有适用边界**：taxah92 的 16GB 流式机只有 +1.4%，我们 100% 驻留才拿满——**社区数字不能直接搬，要匹配自己的驻留率/拓扑**。
- **未合并 PR 的数据要打折**：#1771 自己写明「win is in the phase, not the total」。

---

## 附：本轮涉及的一手材料

- Strata PR #1401（v100-decode-kernels，head 4bc57c9）+ 16 条评论
- Strata PR #1771（prefill-fp16-iq4nl，head 0742b98）+ 3 条评论
- Strata docs/NVIDIA_V100.md（官方 V100 构建 + 内核路由表 + 实测）
- Strata issue #2010（overlap 回归）+ 15 条评论、#2048、#1822、#1100、#1098、#690、#1884
- llama.cpp b11335 / PR #29753（sm_70→Turing MMVQ nwarps，+3.17%）、issue #20605（PP graph reuse -16%）
- anyei/llamacpp-v100：strata-port-plan.md、strata-followups-plan.md、commit 1300d2bef/d6c02899a
- ktransformers（SOSP'25 paper）、MoE-Infinity（arXiv 2401.14361）、PowerInfer-2
