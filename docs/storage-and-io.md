# 存储与 I/O 核查：SATA 不是你的热路径，`--ple-io ram` 在 NVMe 上是个坑

> 缘起：2026-10-07，引擎 v0.1.40.1，内存 64 GB（待升 128 GB），模型 82.8 GiB IQ3_S。
> 问题：「SATA 还有优化空间吗？」——**基本没有，而且方向反了。**

这份核查的结论大多是否定性的，但**否定性结论在社区里更稀缺**：
「你的瓶颈不在这」比「照这个做」更难查到，也更省钱。

---

## 一、先看存储拓扑：模型根本不在 SATA 盘上

| 设备 | 类型 | 挂载 | 用途 |
|---|---|---|---|
| NVMe SSD 1TB | **NVMe** | `/home` | **模型 + PLE 表在此** |
| NVMe SSD 2TB | NVMe | 数据盘 | 空闲 |
| SATA 机械硬盘 1TB | **SATA（ROTA=1）** | 媒体盘 | **纯媒体，推理服务没碰** |

- 模型在 **NVMe** 上，比任何 SATA SSD 快一个量级。**把模型挪到 SATA 盘是降级。**
- 官方 `docs/DETAILS.md` 第 363 行原文：**「An NVMe SSD is strongly recommended」**；
  第 1244 行：默认 `--ple-io direct`「**made for SSDs**」，**机械盘才用 `--ple-io ram`**。
- 引擎日志确认：**routing-aware prefetch of the file tier 默认就是开的**（按下一层路由器预取），
  不是一个需要你去开的优化。

**结论：SATA 不是热路径，没有可优化的 SATA 开关。** 唯一沾边的只有 NVMe 上的
btrfs 透明压缩（对模型文件关掉，`btrfs property set compression none`，收益小）。

---

## 二、一次「588 秒偶发极慢」的根因：不是磁盘，是内存

升级前遇到过偶发的极端慢请求（588 秒）。根因不是磁盘速度：

**64 GB 内存装不下 82.8 GiB 的模型 → 页缓存被回收 → 每次冷读都从盘上拉。**

⇒ **加内存（容量）能根治；换更快的盘没用。**（后来的 128G 升级实测印证：
加载 50s→31s、prefill +8~10%，见 [`memory-and-thermals.md`](memory-and-thermals.md)）

---

## 三、`--ple-io` 三模式：NVMe 上 `direct` 就对，`ram` 有个隐蔽的坑

PLE 表（路由查找表）约 26.8 GiB，三种放置模式：

| 模式 | 行为 | 适用 |
|---|---|---|
| `direct`（默认） | 每个 token 的 PLE 查表付一次 SSD 读 | **SSD/NVMe（官方：made for SSDs）** |
| `ram` | 整表锁进内存 | **仅机械/慢盘**（官方明说） |
| `mmap` | 映射进 page cache 但不锁，热行常驻、冷行可回收 | 内存装不下全模型时 |

### 🔴 `ram` 模式在内存紧张的机器上是个坑（社区同机型实测）

issue #1300（**2× V100 + IQ3_S，与我们完全同型**）实测：`ram` 把 26.8 GiB 锁进内存后：

```
MemFree 掉到 603 MB
VmSwap 1.45 GiB
compact_stall 324,567     ← 内核在疯狂做内存压缩换页
```

**症状极其隐蔽：「引擎解码率正常，但客户端等几十秒」** —— 因为被回收的是进程自己的
匿名页，不是模型权重。看 decode 数字一切正常，看延迟才发现出事。

> 呼应我们的 [`memory-and-thermals.md`](memory-and-thermals.md)：内存只影响
> 「需要从存储读东西」的路径。PLE 表锁 RAM 是拿内存换磁盘读——**内存不够时，
> 这个交换会以更隐蔽的方式反向收账。**

NVMe 上 `direct` 与 `ram` 的分布完全重叠（社区 #1211：RX 9070 XT + 128 GB + NVMe），
NVMe 把 PLE 读模式「零开销吸收」了。**我们保持默认 `direct`。**

---

## 四、解码分析器：先量瓶颈，再谈优化（社区同款方法）

issue #1300 用了 0.1.40 新增的解码性能分析器，这也是回答「存储到底是不是瓶颈」的权威手段：

```bash
STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 ./strata ...
```

结果（同机型 2× V100，512K 上下文 + DDR3-1600 下）：

| 项 | 窗口时间占比 |
|---|---:|
| CPU experts | **0.5–4%** |
| PCIe | **0.01 ms** |

⇒ **即使在最苛刻配置下，存储/CPU 都不是瓶颈。吞吐由「每验证窗口接受的 token 数」
（即草稿接受率）决定** —— 这解释了为什么我们所有磁盘/内存相关的折腾在 decode 上全部归零
（详见 [`flag-ledger.md`](flag-ledger.md) 的证伪清单与 [`memory-and-thermals.md`](memory-and-thermals.md)）。

**这条方法论值得单独记：先花十分钟量出自己的瓶颈占比，再决定优化什么。
我们后来的「引擎侧再次见底」结论（[`engine-upgrade-notes.md`](engine-upgrade-notes.md)）
用的就是同款分析器思路。**

---

## 五、社区方案摘录与我们的后续闭环

按相关度排序（ issue 编号可溯源）：

| Issue | 主张 | 我们的后续 |
|---|---|---|
| #1300 | `--ple-io mmap` 是内存紧张时的正解；`--spec 4` > `--spec 8`（79.4 vs 72.9 tok/s @256K） | ✅ 与我们扫描一致（`--spec 4` 采纳，`--spec 8` 证伪，见 flag-ledger） |
| #1300 | `--batch` 净负、`--lookup-chain` 散文/代码上净负、`numactl --cpunodebind=0` 更慢（49–69 tok/s） | ✅ 与我们的证伪清单一致 |
| #1348 | 命中缺失每次吞 **34% 解码时间**；`--pcie-frac 0.2 --spec-min-p 0.70` 比 `0.5/0.5` **+16%** | ⚠️ 我们在 V100 上复测：**`--spec-min-p 0.70` 本身在噪声内**，真正的增量来自同期上线的 `--mtp-hnorm stream`（见 flag-ledger 的误归因警示）—— **同一旋钮，两套归因** |
| #1280 | `--pipeline-windows 2` 在慢 GPU 链路 **+7~20%**，快链路持平 | ✅ 采纳，生产在跑（PCIe 无 NVLink = 慢链路），prefill +16.6%，见 [`multi-gpu-and-interconnect.md`](multi-gpu-and-interconnect.md) |
| #1211 | NVMe 上 `direct`/`ram` 分布重叠 | ✅ 保持默认 `direct` |
| #1235 / #1349 | DDR3 速度不影响解码（CPU 专家池 0.5–4%）；DDR4-2666→3600 的 A/B | ✅ 印证我们「买容量不买速度」的内存决策（DDR3-1600 继续用，无升级理由） |
| #1206 | `STRATA_IO_PREFETCH`（DirectStorage 对应物，0.1.40.2 加入） | 后续版本已含；未单独 A/B（分析器显示存储非瓶颈，预期收益≈0） |

### 明确不要做的（社区已实测为负）

`--batch` · `--lookup-chain`（散文/代码）· `--ple-io ram`（NVMe 机器）·
`numactl --cpunodebind` 绑单 NUMA · 把模型挪到 SATA 盘。

---

## 六、一句话

**模型在 NVMe 上、预取默认开着、CPU/PCIe/磁盘都不是瓶颈 —— 存储侧没有可捡的果子。
偶发极慢是内存容量问题（换盘无用），PLE 表锁 RAM 在内存紧张的机器上会以「解码正常但客户端卡几十秒」
的方式反咬一口。先量瓶颈（分析器），再决定要不要动。**

**On an NVMe machine, storage is not your bottleneck. Measure before you tune —
and treat "your bottleneck is elsewhere" as a finding, not a non-answer.**

---

## 七、2026-10-14 复测：三层驻留现状（专家 100% VRAM / PLE mlock RAM / KV 98% VRAM）

128 GB 内存 + v0.1.43 之后重测「谁常驻在哪」，全部只读测量（进程画像 + 引擎日志 + `/metrics`）：

| 层 | 驻留位置 | 实测 |
|---|---|---|
| 专家权重 50.1 GB | **VRAM 100%** | CUDA0 12800 slots/23.08 GiB + CUDA1 11776 slots/23.75 GiB = profile 全部 24576 个 (layer,expert) 对；启动 8.9 s 预填（2773 MB/s）；请求级 `expert tiers: GPU 0 hits, RAM 0 blobs, files 0 blobs`、`0 major faults` |
| PLE n-gram 表 28.8 GB | **主机 RAM（mlock）** | `--ple-io ram`，1.1 s 锁定，320001536 rows，00002 分片 26.82 GiB mmap 常驻 |
| KV cache | **VRAM 98%** | `KV streaming: 97.98–98.24% of block reads hit VRAM`，仅 ~0.8 MiB/请求来自 RAM |
| 会话快照（预算 24 GiB） | 主机 RAM | 8 会话全驻留 1.9 GB，evictions 22–23（有压力未冻结） |

**每请求 PCIe 流量 = 噪声级**（−73~+19 MB，符号不稳；请求前后快照差分实测）⇒
专家/PLE/KV 每前向都不走 PCIe。启动时 stager 经 PCIe 灌 46.8 GB 进 VRAM 是一次性成本。
引擎自测 host→device 12.2 GB/s（据此把 pcie_frac 自动降为 0.33）。

⇒ 第三节「`--ple-io ram` 在内存紧张的机器上是坑」的边界更清楚了：
**125 GB 内存 + 100% page cache 的机器上 ram 模式无感**（PLE 表 1.1 s 锁完，
之后读取全命中）。真正的坑仍然是「内存 < 模型 + PLE 表」的机器。

**测量方法坑（重要）**：`/metrics` 的 `gpu_pcie_rx_mb` / `gpu_pcie_tx_mb` 是**累计值**——
必须请求前后快照差分；对 200ms 采样点直接求和会算出虚高一个数量级的假流量
（第一版脚本就这样报出了假的「并发 PCIe 1.2 GB」，差分重测后是噪声级。
已补进 [`how-we-measure.md`](how-we-measure.md) 的坑清单）。
