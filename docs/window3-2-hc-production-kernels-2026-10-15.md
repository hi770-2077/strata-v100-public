# 窗口 ③-2：超连接读的生产内核实测——六个候选全阴，块数的不对称成为主要结论（2026-10-15）

> ③-1 把「超连接投影 4.0 ms/窗（22%）」列为最大单一机会，但它背后的 bench 调的是
> `bf16_gemv_fp32_mmvf_multi`——**解码路径根本不用的内核**。本轮先把生产调用链核实清楚，
> 再对真正的生产内核（`fused_gr_read_multi` 的 norm/down/up 三件套）做候选改造实测。
> **结论：六个候选（含逐位一致的）全部更慢或只有噪声级收益；HC 内核微优化到此为止。
> 主要产出是一条不对称规律和一个被纠正的事实，剩余最大项回到窗口③（launch 间隙 19%）。**

---

## 1. 路线纠偏：生产到底调什么

③-1 的 bench（`win22_dense_bench hc`）量 `bf16_gemv_fp32_mmvf_multi`（`native_bf16.cu`），
而 `verify.cpp:1071` / `mtp.cpp` 走的是 `fused_gr_read_multi`。读 `launch_multi`
（`fused_gr.cu:1030`）的分支，本卡（V100 + `STRATA_SM70_TABLE=1`）实际启动的是：

| 组件 | 内核 | 网格 | 说明 |
|---|---|---|---|
| norm | **`gr_norm_split_kernel`** | n_tok × HC | `kHcStaged(3) >= kHcSplit(2)` 走 split 支；`norm_fast_on()` 只认 gfx12，V100 为 false；xn 写两趟（先 r*g，再读回乘 rs） |
| down | `gr_down_staged_kernel` | 41 block | 双缓冲 staged tile，sm_70 上 cp.async 退化为普通加载 |
| up | `gr_up_fast_kernel` | 160 block | 8 lane/行（xor8），一线程两行 |

> ⚠️ **纠正自己上一轮的两个误判**：①「norm 是 grid=n_tok 每 token 一个 block」——只对
> **plain 路径**成立（`gr_norm_fast_kernel`），staged 路径本来就走 split；②「独立 mmvf 内核
> 2.2× 加速对生产有意义」——不成立，生产不调它。引擎日志的
> "staged (the norm per token and stream, ...)" 一直是对的。

## 2. 修正后的基线（`w32_fgr_bench.cu`，event 计时，13 组冷权重轮换，T=1..8）

| 内核 | T=1 | T=2 | T=3 | T=4 | 备注 |
|---|---|---|---|---|---|
| norm（split，生产） | 5.3 | 5.4 | 5.6 | 5.4 | ~130 KiB/token，随 T 平坦 |
| norm（fast，plain 路径） | 6.7 | 6.8 | 6.8 | 6.9 | grid=n_tok，块更少 |
| down（staged，冷） | 18.4 | 21.3 | 21.7 | 25.8 | 6.25 MiB，310–360 GB/s |
| up（fast） | 14.2 | 16.9 | 19.1 | 21.7 | 6.25 MiB，300–460 GB/s |
| nop 地板 | 2.0 | 1.9 | 2.1 | 1.9 | 启动率下限 |

**引擎 stamp 数的修正**：`fused_gr_bench` 报的 norm 8.2 / down 19.5 / up 19.5 us 里，
**每个相位各含一个 `gpu_stamp` 内核**（`verify_kernels.cu:1344`，一个 1 线程的内核启动，
读 globaltimer 写缓冲）。生产路径 `prof_on_=false` 时 `stamp_buf=nullptr`，**不付这个代价**；
剖析模式的相位账每项要减 ~1.5 us。这是「launch 间隙」在微尺度的直接证据：一次 1 线程启动
在 GPU 时间线上就是 ~1–2 us。

## 3. 候选实测：六投六阴

所有逐位候选都先过 bitwise 再计时（rel_l1 双精度参考同量级）。

| 候选 | 逐位 | T=1 | T=2 | T=3 | 结论 |
|---|---|---|---|---|---|
| down ref（staged） | — | 16.8 | 19.5 | 21.0 | 基线 |
| down 深权重预取（2 tile in flight） | ✅ | 19.2 **+14%** | 23.5 **+20%** | 26.4 **+26%** | 寄存器 98→102（ref 74–84），T 越大越亏 |
| down 三缓冲（屏障减半） | ✅ | 17.6 +5% | 20.8 +7% | 21.4 +2% | 冷权重下 T≤3 快 1.4–3.3%，但 T=4 需 61 KiB smem（>48 KiB 默认），不可上线；折算 ~0.2% e2e |
| down K-split ×2/×4（更多 block） | ✗（设计坏） | 27.0 | 33.3 | 36.4 | slice 共用一块双缓冲互相踩；**坏着都已慢 40–70%**（同样的加载量） |
| down 4 warp/block（81 block） | ✅ | 17.0 +1% | 21.8 **+12%** | 26.3 **+25%** | 加块对 down 是反效果，见 §4 |
| down 2 warp/block（161 block） | ✅ | 17.5 +4% | 23.8 **+22%** | 31.0 **+48%** | 同上，更极端 |
| up ref（fast） | — | 14.2 | 16.9 | 19.1 | 基线 |
| up 8 列/block（320 block） | ✅ | 15.2 **+7%** | 18.0 **+7%** | 20.2 **+6%** | 寄存器减半（127→89）但一线程一行的 ILP 损失更大 |

ncu（`STRATA_SM70_TABLE=1`）：down DRAM 31.8% / compute 8.4% / 0.2 waves；
up DRAM 35.8% / compute 10.8%——都是延迟受限，但**不是「加块、加深预取、减寄存器」能救的那种**。

## 4. 主要结论：块数的不对称

norm 的消融给出全轮最干净的结构性事实：

| norm 变体 | 网格 | 时间 | 每 block 吞吐 |
|---|---|---|---|
| fast（grid=n_tok） | 1–4 block | 6.8 us | ~19 GB/s |
| split（生产，grid=T×HC） | 4–16 block | 5.4 us | ~6 GB/s |
| ldst-only（grid=T，无归约无屏障） | 1–4 block | 4.9 us | ~25 GB/s |
| wide（grid=T×8，无屏障，4× 重叠读） | 8–32 block | 2.5 us | ~24 GB/s |

- **norm（每 block 32 KiB、纯延迟、无内部流水线）：块数就是吞吐**——T=1（4 block）每 token
  5.3 us，T=8（32 block）每 token 1.3 us，4× 块数换 4× 吞吐。
- **down（每 block 160 KiB、8 tile 双缓冲流水线）：块数是反效果**——4 warp/block（81 block）
  在 T≥2 慢 12–25%。减线程让每次 `__syncthreads()` 等待更久，staging 的延迟隐藏变差。
- **屏障和串行收尾不是 norm 的成本**：并行 finalize + 单屏障版本与生产 split 同速（5.1 vs 5.4）。
- **norm 宽于 T×HC 会破坏逐位**：蝶形分组随线程→元素映射而变，rs 的最后几位不同。
  要更宽只有 opt-in（复刻蝶形树的 finalize + 原子/自旋跨块合并），多一次启动就吃掉收益
  （每次启动 ~2 us，见 §2）。

## 5. 可做但没做的（记录在案）

1. **norm 换 `gr_norm_split_fast_kernel`**（寄存器留存、xn 只写一趟）：逐位一致（T=1..8 全验），
   快 0.2–0.4 us/read ≈ 0.1% e2e——低于任何合理门槛，不值得一次引擎重建。
2. **norm 宽 split（T×8 block）**：bench 2.5 us vs 生产 5.4 us，理论 ~1.1% e2e，但非逐位，
   要走 V3 式 opt-in + 全量 parity，且多启动/自旋的代价可能吃掉大半。
3. **down 三缓冲**：见 §3，smem 在 T=4 越限。

## 6. 对窗口③的意义

本轮顺带证明了 launch 经济学：`gpu_stamp` 一个 1 线程内核就是 ~1–2 us 的 GPU 时间；
③-0/③-1 账里的「空闲间隙 3.4 ms（19%）」与每次启动 2–5 us × 每步数百次启动一致。
**HC 内核本身已优化到头（6 投 6 阴），下一个最大单项是 launch 间隙本身**——引擎级项目
（启动融合 / CUDA Graph 按 T 分桶捕获），预计 +8–12%，也是全预算里最后一块大的。

---

*bench：`src/kernels/w32_fgr_bench.cu`（v100-opt 私仓）。构建：`/home/shun/tmpwork/build_fgr_bench.sh`
（nvcc sm_70 + g++-14，链生产同款 `libstrata_kernels.a`）。运行：`/home/shun/tmpwork/w32_fgr_cand_run.sh`。
方法学教训两条：① up 内核的 `a.R`/`a.R_out` 别名同一块显存，跑一次会改掉自己的输入，
parity 快照必须分开存初始值；② 深预取的载入必须放在寄存器shift**之后**，否则会冲掉还没消费的 tile。*
