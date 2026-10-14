# decode 全局定量账与优化窗口排序（2026-10-14 实测）

> 场景：一台双 V100 机器上的 125B 级 MoE 生产模型（48 层 / 512 专家 top-10 / embed 2560 / GQA 24:2 /
> GDN×36 + 全注意力×12 / indexer top-2048 / PLE 16 heads / 词表 248,320），引擎 strata-1771 +
> PF_WMMA=2 + spec 4 + mtp-q4 all + layer-split 25。
> 所有数字来自：引擎日志每请求计时行、`/metrics` 遥测（200 ms 粒度）、宿主机 `nvidia-smi` 双卡采样、
> GGUF 张量清单解析。**实测与建模分开标注。**

## 一、模型结构账（GGUF 解析，纠正一处旧误解）

- 分片：00001 主体 55.06 GB + 00002 PLE n-gram 表 28.8 GB（`--ple-io ram`，宿主机内存）
- 专家权重是**分层混合量化**（"GSQ-RCO"）：
  - gate/up：IQ2_S(2.5625 bpw)×20 层、IQ3_XXS(3.0625)×17、IQ3_S(3.4375)×10、IQ4_XS(4.25)×1
  - down：Q4_0(4.5 bpw)×39 层、Q2_0(2 bpw)×9 层
  - 合计 **50.1 GB**，与引擎日志 "50296.6 MB read" 精确吻合；每层均值 1.04 GB，每专家均值 1.97 MB
- 每专家 3×1.638M 参数（gate/up 2560×640、down 640×2560）；每层全部 512 专家 = 2.52B 参数
- dense 权重 ~4.1 GB/次前向：attn 1.16 + GDN 0.64 + 超连接(BF16) 1.26 + shexp 0.16 + lm_head(Q6_K) 0.52
  + 杂项 0.36。**超连接 BF16 与 lm_head 是 dense 流量大头**
- 纠正：旧 flag-ledger 曾称"本模型是纯 IQ3_S"——实际 gate/up 只有 10/48 层是 IQ3_S；
  type 码按本仓 vendored ggml.h：22=IQ2_S、23=IQ4_XS、42=Q2_0、2=Q4_0

## 二、decode 全局账（实测锚点）

| 指标 | 数值 | 来源 |
|---|---|---|
| decode 速度 | 86.6–134.6 tok/s（典型 100–125） | 引擎日志 60+ 请求 + 本轮 4 次受控请求 |
| 每次目标前向 | **18.0–19.5 ms**（1200 token 请求 483–854 次前向） | 日志 drafts/生成数反推 |
| 每前向 token 数 | 1.6–2.5（接受率 73–94%，随内容波动） | 日志 |
| 每 token HBM 流量 | 专家 **0.95 GB**（48 层×top-10×1.97 MB，剖析器相位折算）+ dense ~3.2 GB | 建模 + 剖析器相位折算互证 |
| 有效带宽 | **~450 GB/s = 单卡峰值 50%**（MoE 专家读：VRAM hits 相位 2.1 ms/窗/stage × 0.95 GB；窗口① native bench 391 GB/s 独立互证；**推翻旧假设 329 GB/s**） | 实测（双方法） |
| 双卡 util（decode 期） | 78–90% → 12–20% 空闲，**但非 launch 间隙**：剖析器 (gap) 相位 0.03–0.04 ms/窗 ≈ 0，主机侧 wait/plan/actq/jobs/CPU 全 0.00；真实空闲在窗口间调度（fresh 惩罚串行化） | nvidia-smi + STRATA_VERIFY_PROFILE |
| 时钟/功耗/温度 | 1380 MHz 满频、120–140 W / 250 W、40–43 °C，无降频 | nvidia-smi |
| prefill（7–14K fresh） | 1500–1815 tok/s | 日志（12.9K：1728–1815，8 次重复稳定） |
| 110K+ 长上下文 | decode 85–153 tok/s、每前向 ~25 ms（+35%）、fresh prefill 176–773 tok/s | 日志 |
| 短 prompt 固定开销 | 250–650 ms（快照恢复+环形设置+预热） | 日志 36–81 token 请求 |

每窗口 GPU 相位账（STRATA_VERIFY_PROFILE 剖析器，T=1.84：stage 0 = 10.41 ms/窗、stage 1 = 11.06 ms/窗，墙钟 18.07 ms/窗）：
- **MoE 专家分组 GEMV（VRAM hits 相位）= 2.1 ms/窗/stage，占 stage 20%**（MoE 侧合计 ~3.9 ms = 37%：
  含 router/plan/bell、shared+quant、VRAM hits、PCIe grp、copy+combine），有效带宽 ~450 GB/s = 单卡峰值 50%
- dense 侧 6.5 ms（63%）：hc0/hc1 读 2.1 + qkv gemv 1.7 + out-proj 1.2 + z/rec/conv 1.1 + attention 栈 1.3
- **(gap) 层间 GPU 空隙 = 0.03–0.04 ms/窗 ≈ 0；主机侧 wait/plan/actq/jobs/CPU 全 0.00**
⇒ 旧账「空闲间隙 3.4 ms」实为**窗口间调度空隙**（fresh 惩罚：undo+chain+双 stage 串行），不是内核 launch 间隙。
⇒ profiler 本身会制造额外空闲（unfused 内核串行化主机循环）：profiled 形态 GPU0 73.8%/GPU1 57%，
   干净生产形态 GPU0 83–86%/GPU1 78–84%——相位占比口径不受影响，绝对值略偏保守。
理论下限（全内核 70% 峰值）≈ 8.5 ms/前向。

## 三、窗口排序（decode 收益预期）

| # | 窗口 | 收益 | 成本 | 依据 |
|---|---|---|---|---|
| 1 | ~~grouped 内核接线~~ | **0%（已关闭）** | — | 2026-10-14 三 bench 实测：生产已走新内核（引擎无代际开关）；VRAM 调用 new==v1（196 µs，391 GB/s=单卡峰值 46%）；新 iq_mmvq 代际只利好 IQ2_XS/IQ1_M（本 pack 不用）。详见《grouped 内核对照实测》 |
| 2 | iq_mmvq 深度优化（50%→60–70%） | **+3–6%** | 高（CUDA 重写+parity+A/B） | 剖析器实测 450 GB/s = 单卡峰值 50%（旧 329 作废）；MoE 占 GPU 时间 37%；上游无更优可抄（sm_70 mma 路线已证伪） |
| 3 | ~~launch 间隙压缩（CUDA graph 等）~~ | **0%（已证伪关闭）** | — | STRATA_VERIFY_PROFILE：(gap) 相位 0.03–0.04 ms/窗 ≈ 0、主机侧相位全 0.00；GPU 时间线无 launch 间隙，NVML 12–20% 空闲在窗口间调度 |
| 4 | 草稿接受率 73–94%→95% | +10–20% | 很高（重训 MTP 头） | flag 级已榨干：spec 8 证伪、spec-min-p 0.8 甜点、mtp-q4 all 已采纳 |
| 5 | 长上下文状态读取优化 | 长上下文 +25–35% | 高、未归因 | 110K 每前向 25 ms vs 18.5 ms（KV/indexer/GDN 快照/PLE host 表候选） |
| 6 | prefill 残余 | <8% | 高 | 已 1500–1815 tok/s；PF_WMMA 仅 +4%；Volta dp4a 是上限路径 |
| 7 | lookup chain | ≈0（本负载） | — | 0/562 windows：设计要求「上下文+MTP草稿尾部」精确历史匹配，聊天/创作负载不可命中；代码/重复性负载才可能受益，非 bug |

叠加口径：窗口① 已关闭（0%）、窗口③ 已证伪（0%）⇒ **引擎级残余 = 窗口② 的 +3–6%**；
模型级（窗口④接受率）上限 ~1.6× 但需重训 MTP 头。
调度层（fresh 惩罚）理论 +10–20%：fresh 窗占 15/32（47%），每窗比 speculative 贵 +7.5 ms；
但**无便宜旋钮**——0d 三臂 A/B 证实调草稿结构只换配比不换每 token 成本（见第六节）。
**flag 级甜头已经没有了**：spec 8 证伪 / spec-min-p 0.8 甜点 / mtp-q4 all 已采纳 /
lookup-chain 4 vs 0 精确同速 / 0d 三臂（0.5 洗平、0.65 -2%、suffix-draft 0 洗平）。

## 四、权衡型杠杆（非纯 win，记录备查）

- down 投影 39 层已是 Q4_0(4.5 bpw)，gate/up 20 层 IQ2_S(2.56)——整体已压到底部区间，
  再降 = 质量风险，且专家分组 GEMV 仅占 GPU 时间 20%（MoE 侧合计 37%），收益 <5%
- 超连接 BF16 1.26 GB/前向 + lm_head Q6_K 0.52 GB/前向占 dense 流量 43%，
  量化它们可直接砍 dense 时间，但动质量核心，需专项评估
- 量化不改变 MAC 数：prefill 是算力瓶颈，低比特量化对 prefill 无加速

## 五、第一优先动作（2026-10-14 已执行，结论：窗口① 收益为零）

用仓内现有 `native_grouped_parity` / `s2_expert_grouped_parity` / `iq_multi_parity --bench`
对**生产模型的真实专家类型组合**跑 grouped vs old 对照（真机 SSH 跑，`STRATA_SM70_TABLE=1`，
native bench 需暂停生产服务 7 秒腾 3.7 GB arena）。完整数据与结论见
《grouped 内核对照实测：窗口① 收益为零》：**生产已在跑新内核且主路径 new==v1，
窗口① 关闭，第一优先移向窗口②（iq_mmvq 深度优化）**。

## 六、窗口②-0 相位分解 + flag 级排除实验全记录（2026-10-14）

口径：serve 路径 `STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1`（引擎只在 serve 路径打印，每次打印后清零）；
`split_` 仅是 `--spec-split` 开关（生产未开 → G=1 → 剖析器有效）；测量用同 prompt 二次请求
（51/56 复用 prompt-cache）获得近纯 decode 口径；每轮换唯一盐 prompt。

### 6.1 窗口结构定量（基线 = 生产参数 0.8/3/lc4，56 token，两次重复逐位一致）

| 指标 | 数值 |
|---|---|
| 总吞吐 | 106.8 tok/s，16.41 ms/窗（32 窗） |
| fresh 窗 | 15 窗 × 20.36 ms × 1.60 tok（B 未 on-path：未启动或被 rollback） |
| speculative 窗 | 17 窗 × 12.90 ms × 2.00 tok（只付 stage 1，单卡前向） |
| chain late | 12/32（stage 0 空等 MTP forced-chain 产出 B） |
| 草稿接受 | 26/27（96%）；B-on-path 59% |
| fresh 惩罚构成 | undo+chain(~2 ms) + stage0(9.5) + stage1(10.2) 串行 ≈ 20.4 ms；speculative 只付 stage1 |

### 6.2 flag 级三臂 A/B（同基线口径，每臂 warmup + 2 次测量）

| 臂 | 变体 | tok/s | ms/窗 | fresh 窗 | speculative 窗 | 草稿接受 | 结论 |
|---|---|---|---|---|---|---|---|
| 0d-a | --spec-min-p 0.5 | 105.9/106.6 | 21.14 | 16 × 25.08（2.31 tok） | 9 × 14.04（2.33） | 33/47（70%） | **完全洗平** |
| 0d-b | --spec-min-p 0.65 | 104.3/104.5 | 17.32 | 16 × 21.32（1.75） | 15 × 13.02（2.00） | 27/30（90%） | **-2%，两次重复稳定** |
| 0d-c | --suffix-draft 0 | 106.6/106.8 | 16.42 | 15 × 20.37（1.60） | 17 × 12.89（2.00） | 26/27（96%） | **精确洗平，与基线逐位同构** |

**每 token 成本不变式**：四臂 blended 成本均为 9.37–9.59 ms/token（56 token 总时 524–537 ms）。
两点线性拟合（0c vs 0d-a，仅两点、示结构不确定）：fresh 窗 ≈ 9.7 ms 固定 + 6.65 ms/token；
speculative 窗 ≈ 6.0 ms 固定 + 3.45 ms/token。
⇒ spec-min-p 0.5 把草稿量 27→47、接受率 96%→70%、fresh token 占比 41%→64%，
   配比剧变但每 token 成本精确不变——**调草稿结构只换 fresh/speculative 配比，不换 blended 成本**。
⇒ 想赢只有两条路：**砍 per-token 成本本身（窗口②内核，+3–6%）**，
   或**砍 fresh 窗固定开销/占比（调度层手术或模型级接受率，理论 +10–20%，均无便宜旋钮）**。

### 6.3 调度层侦察结论（STRATA_PIPELINE_TRACE 逐窗时间线，零重建，双请求逐位一致）

方法：`STRATA_PIPELINE_DEBUG=1 + STRATA_PIPELINE_TRACE=<file>`（循环每个事件带主机时间戳，L0/F0/L1/F1/V/CL/CD/CE/U），
生产参数原样运行，106.7/106.8 tok/s 与基线逐位一致（埋点零开销）。注意 **seq 会复用**：doomed B 滚回后
fresh 窗重用同一 seq（L0 双事件），分类须按「verdict 前最后一次 F0 的 spec 标志」。

**管线设备账（每请求 32 窗 / 525 ms，trace 内部账与 NVML 请求期采样互证）**：

| 设备 | 负载 | 忙碌率 | NVML 印证 |
|---|---|---|---|
| GPU0（层 0-24） | 44 次 stage-0 验证（32 A + 12 doomed B） | **83%**（424 ms） | 83–86% |
| GPU1（层 25-47 + MTP drafter） | 32 次 stage-1 + 31 条 chain | **81%**（stage1 64% + chain 17%） | 78–84% |

- 每窗成本：**s0 ≈ 6.4 + 1.7×T ms**（GPU0）、**s1 ≈ 6.5 + 1.6×T ms**（GPU1）；chain 1.5–5.1 ms（fresh 均值 2.3 / spec 3.3）
- chain 归属 GPU1 由双证据确认：trace 账（若在 GPU0 则超载 101%）+ `mtp.cpp:246`「drafter lives on the last stage's device」

**两类周期的结构地板（实测已贴地）**：
- **fresh 周期 20.35 ms** = chain 首输出等待 0.7 + s0_A（8.0–12.6）+ s1_A（8.1–12.3）+ 交接 0.3；
  T=1 窗实测 17.2 vs 地板 17.15——**误差 0.2 ms**。GPU1 空闲 9–11.5 ms/周期是**结构性**的：
  fresh 周期里 GPU1 唯一可干的活（A 的 stage 1）依赖 A 的 stage 0 输出，chain 已藏在这段空闲里
- **spec 周期 12.88 ms** = max(GPU1: s1_A + 0.85×chain 互损 ≈ 12.1，GPU0: chain 等待 + s0_B)；
  chain 与 stage 1 在 GPU1 并发互损（T=2 窗 s1 从干净 9.64 被拉到 12.20）；距地板 ~0.8 ms
- 主机侧全 0.00（plan/actq/jobs/CPU/wait），undo 0.15–0.45 ms——主机不是瓶颈

**手术面逐条定量排除**：
1. **chain 迁移到 GPU0**：GPU0 将 83%+17% > 100% 超载，且 fresh 周期失去 chain 藏身处——净负
2. **chain 降 stream priority**（保 stage 1 干净）：chain 拉长 → GPU0 的 B 启动等待同比拉长——净负
   （当前 0.85 互损系数下 GPU1 12.1 / GPU0 ~12.4 本是均衡态）
3. **θ 门限调整**（默认 0.20）：p_on 校准表非单调（0.4:4/6、0.5:1/4、0.6:4/7、0.7:1/3、0.8:2/2、0.9:5/5），
   0.4–0.7 区间估计量几乎无信息；抬 θ 把 on-path 的 B 变 fresh（每次 +7.5 ms）——净负
4. **B 的 T 封顶**（治机制 B：T_B=4 时 s0_B 尾巴拖到下一周期，GPU1 空等 ~5 ms × 4 窗 ≈ 20 ms）：
   与每 token 成本不变式冲突（低 T 窗固定开销占比升高），预期洗平
5. **undo / 主机侧**：合计 < 10 ms，无空间

**结论：调度层无可 profitable 的手术——系统在结构地板 ±5% 内运行，双设备负载 83/81% 已均衡，
fresh 周期 GPU1 空闲结构性无活可填。** 理论完美重叠地板 max(424, 412) = 424 ms vs 实际 508（1.20×），
差额绝大部分是 fresh 周期的串行结构。剩余大池子归属：**内核（s0+s1 占 spec 周期 79–84%）→ 窗口②；
接受率（15/32 fresh；完美接受率理论 +27%）→ 窗口④**。
