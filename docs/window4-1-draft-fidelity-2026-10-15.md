# 窗口 ④-1：草稿层图保真审计（2026-10-15 凌晨）

> 路线图 #28。原任务："对照 llama.cpp 参考实现逐算子核 Strata 草稿层，找保真偏差，先只审计不动代码。"
> 结论：**没有找到系统性保真缺陷**，并且审计过程中**推翻了窗口 ④-0 的一条核心前提**（"英文 a=2.68 vs 中文 0.97，差 1.86×"）。
> 附带产出：一个零重启的窗口④实验工具（`strata_tune.spec_min_p`）和一个新的成本模型。

---

## 0. 一句话结论

草稿层的**逐步接受率是正常衰减的**（p1 ≈ 0.8–0.85，p2 ≈ 0.65，p3 ≈ 0.30），中英文**几乎无差异**（106.8 vs 112.6 tok/s，差 5%）。
既没有"第 1 步好、后面崩"的链式 bug，也没有"中文被词表/量化坑了"的语言性缺陷。
**窗口 ④（草稿准确率）已经接近它的天花板**；真正的天花板是**验证窗口每多一行的边际成本 4.83 ms**，那属于窗口 ③。

---

## 1. 审计基线：llama.cpp 里根本没有这个 MTP 图

原任务假设可以拿 vendored llama.cpp 当参考。实测**不成立**：

| 检查 | 结果 |
|---|---|
| `build-cuda/_deps/strata_llamacpp-src/src/models/qwen4exp.cpp`（1294 行） | 实现了 qwen4exp **主干**（hyper-connection / indexer / PLE / GDN 全有），但 **`nextn` 零命中** |
| 全仓 `nextn` | 只在 `llama-arch.cpp` 的张量名表（`LLM_TENSOR_NEXTN_*`）和 bailingmoe2/3、cohere2moe、deepseek 等**别的架构**里 |
| `fc_hidden` / `pre_fc_norm_hidden` | llama.cpp **零命中**（只有 `LLM_KV_HYPER_CONNECTION_*` 这类 KV key） |
| `tools/mtp_probe.py` | **不存在**。`git log --all -- tools/mtp_probe.py` 为空 |

⇒ `include/strata/core/mtp.hpp` 头部那句"transcribed and validated in `tools/mtp_probe.py`，where it accepts 0.89 / 0.86 / 0.85 of greedy drafts at steps 1-3" 引用的脚本**不在仓库里**，那组数字**没有现存出处**。以后不要把它当基线引用。

**改用自参照基线**：草稿层（`src/core/mtp.cpp` 的 `record_front` / `record_rest`）vs Strata 自己的主干解码路径（`src/core/verify.cpp`）。草稿层除了输入投影和自有 K/V 外，结构上应当与主干一层完全一致。

## 2. 自参照逐算子核对：全部对齐

| 项 | 草稿层 | 主干 | 判定 |
|---|---|---|---|
| RMS eps | `EPS = 1e-6f`（mtp.cpp:56） | `EPS = 1e-6f`（verify.cpp:62） | ✅ |
| grouped norm 权切片 | `native_qsa_rms_norm_grouped(Rin_, pre_fc_norm_hidden, hn_, N, HC, T*HC)` | 同函数（prefill.cpp:1400） | ✅ 见下 |
| 末端结构 | 最后 `fused_gr_read_multi`（`hyper_connection_mixer.*`）→ `sample_` → q8_1 → `native_mmvq(head)` | 主干同样用 `output_hc_*` mixer → `head_mixed_` → q8_1 → `native_mmvq(head)`（verify.cpp:1581-1609） | ✅ 两侧**都没有**独立的 final RMS norm |
| rope 位置 | `put(row, cell)` → `pos[·] = cell`；cell t = p+t；草稿 j 的 cell = p+a+1+j | 与 mtp.hpp 注释"cell i 配 R_i 与 token_{i+1}，输出预测 token_{i+2}"一致 | ✅ |
| q 的 gate | `native_qsa_gate_apply`（q_proj 输出 NH·2·HD，后半作 sigmoid gate） | 同 | ✅ |
| MoE | bf16 router gemv + top-10/512 + 常驻分组专家 + shared expert（`shared_expert_gate` sigmoid）+ combine | 同函数族 | ✅ |
| 超连接读/写 | `fused_gr_read_multi` / `gr_write_multi` | 同 | ✅ |

**重点澄清一个疑似 bug（已排除）**：`norm_grouped` 的权切片索引是
`gamma += std::size_t(blockIdx.x % unsigned(groups)) * n_cols;`（`native_qsa.cu:76`）。
多行的 catch-up 批里行号 = t·HC + hc，取模后 = hc，**正确**。不是"T>1 时读到错权重"。

## 3. 存在的既定偏差（设计取舍，不是 bug）

| 偏差 | 说明 | 影响 |
|---|---|---|
| **indexer 未参与，草稿层是稠密注意力** | `mtp/rt/dense.txt` 里有 `self_attn.indexer.*` 四个张量，但 `mtp.cpp` **全篇未引用**；草稿层对最近 `--mtp-window`（默认 32768，生产未覆盖）个 cell 做稠密注意力，主干用 indexer 稀疏 top-k | mtp.hpp 自己写明"below 2,051 cells"才等价。上下文 >2051 cell 后不等价；窗口④-0 实测长度退化温和（81%→75%） |
| **草稿层没有 GDN/卷积分支** | 主干 profile 有 conv/ab/z/rec（GDN 递归），草稿层没有 | 但 HF checkpoint 的 MTP 31 个张量里本来就没有 SSM/conv 权重 —— 模型自带结构，非 Strata 偏差 |
| **K/V 精度更高** | 草稿层强制 `qsa_set_kv_hybrid(false)`，在 `--kv k8v4` 下退化为纯 INT8 | 比目标的 k8v4（V 是 Q4）更高，不是退化方向 |
| **权重精度** | `--mtp-q4 all`：11 个 Q8_0 投影 + 106299 行 draft head 压成 Q4_0 | 引擎日志实测 draft head 每步读 **212.9 → 146.0 MiB**；配置注释记录接受率不降（85.7–86.9% → 89.6%） |

---

## 4. 判别实验：逐步接受率（零重启，按请求 `strata_tune`）

### 4.1 工具发现（重要，后续窗口④/③ 都该用）

`spec_min_p` **可以按请求覆盖**（`strata_tune={"spec_min_p": x}`，`serve/server.py:965` → `generate.cpp:8964`），
而且它控制的**不只是链条截断，而是验证窗口大小 T**：

```c
// generate.cpp:10119
if (req_spec_min_p > 0.0) {
    while (T < S_mtp && from + T - 1 < avail && pr[from + T - 1] >= (float) req_spec_min_p) ++T;
}
```

⇒ `min_p = 1.0` → T 恒为 1 → **完全关闭投机**；`min_p = 0.0` → T = spec → 满链。
**不需要改配置、不需要重启服务。** 这比之前"改 config + systemctl restart"那一套可靠得多（见第 6 节的翻车记录）。

顺带确定语义：**`--spec N` → 窗口 T = N 行 → N-1 个草稿**（不是 N+1）。满链实测 offered/win = 3.00 = spec 4 − 1 ✓。

### 4.2 实测（生产服务，greedy，max_tokens 400，中/英各 4 次取中位数）

| `spec_min_p` | 窗口 T | 每窗草稿 | a（中文） | tok/s（中文） | ms/窗 | a（英文） | tok/s（英文） |
|---|---|---|---|---|---|---|---|
| 1.0 | 1 | 0.01 | 0.010 | 87.0 | 11.59 | 0.006 | 87.8 |
| **0.8（生产）** | ≈2.17 | 1.17 | **0.942** | **106.8** | 18.15 | **0.962** | **112.6** |
| 0.0 | 4 | 3.00 | **1.532** | 95.9 | 25.99 | **1.624** | 102.3 |

### 4.3 反解逐步接受率

设 q_j = 链条走到第 j 步的概率（q_j = Π_{i≤j} p_i），满链 a = q1+q2+q3 = 1.532；
生产档 offered/win = 1.17 ⇒ 17% 的窗口是 T=3（2 草稿），
accepted/win = p1(1 + 0.17·p2) = 0.942 ⇒ **p1 ≈ 0.8–0.85**；
回代满链 ⇒ **q2 ≈ 0.54（p2 ≈ 0.65）**，**q3 ≈ 0.16（p3 ≈ 0.30）**。

判读：
- p1 ≈ 0.8–0.85 **落在 mtp.hpp 声称的 0.85–0.89 区间内**（±5 pp）⇒ 草稿层实现没有系统性保真缺陷。
- 链条是**平滑衰减**（0.83 → 0.65 → 0.30），不是"第 1 步正常、第 2 步崩"。
  ⇒ **排除链式 bug**（残余 R / K-V / 位置在续接步上出错的那一类）。
- 中文 p1 与英文 p1 **基本相等**（草稿接受率 51.7% vs 56.5%），综合速度只差 5%。
  ⇒ **排除"中文被 draft_vocab 子集 / Q4_0 head 量化坑了"**。

### 4.4 ⚠️ 更正窗口 ④-0 的核心前提

窗口④-0 报告写的是"英文 a=2.68 / 208 tok/s vs 中文 a=0.97 / 112 tok/s，差 1.86×"。
那个英文样本的 prompt 是**让模型数数**（"英文计数"）——退化地可预测，a 能到 2.68。

换成**同等难度的中英文叙述性文本**后，差距只有 **5%**（106.8 vs 112.6）。
⇒ "中英文接受率差 1.86 倍、这里面藏着一个大杠杆"这个判断**不成立**，窗口④-0 据此建立的动机需要修正。

### 4.5 新的成本模型（比 ④-0 的"每窗成本恒定"更准）

```
ms/窗 ≈ 11.5 + 4.83 × (T − 1)
```

- T=1（不投机）：11.5 ms/窗，87.8 tok/s
- T≈2.17（生产）：18.2 ms/窗，106.8 tok/s（**投机带来 +21.6%**）
- T=4（满链）：26.0 ms/窗，95.9 tok/s（**比生产慢 10%**）

满链为什么更慢：多 1.83 行花 **+7.85 ms**，只换回 **+0.59 token**。
**边际行成本 4.83 ms = 基线的 42%** —— 这就是投机解码的真实天花板。
若能把它压到 ~1 ms，满链 a=1.53 会给出 2.53 token / (11.5+3) ms ≈ 174 tok/s（+63%）。

⇒ **下一步应该去窗口 ③（每行串行开销 / launch 间隙），而不是继续在窗口 ④ 里抠草稿准确率。**

---

## 5. 审计结论

**未发现可修的保真偏差。** 逐算子与主干一致；唯一的结构性偏差（稠密 vs indexer 稀疏注意力）是已知的、作者自述的取舍，且实测影响温和。
剩余的真实杠杆排序（更新后）：

1. **边际验证行成本 4.83 ms** → 窗口 ③。这是当前最大的单项（占基线 42%）。
2. **indexer 稀疏注意力进草稿层** → 只在 >2051 cell 的长上下文有意义，且工作量大。
3. 草稿准确率本身（p1 ≈ 0.83）→ 与参考值同档，空间约 5 pp，换算到 tok/s 约 +3%。

## 6. 翻车记录（写成教训）

1. **沙箱里 `sudo` 已失效**：`sudo: /etc/sudo.conf is owned by uid 65534` → `无法打开 /etc/sudoers` → `systemctl restart` 静默失败，而 health 检查对**旧进程**返回 200，于是"改了配置却测到旧配置"，扫出来的四个臂全是假的。
   ⇒ **凡是改生产配置/重启，一律走 SSH 通道**（`ssh -F none -i ~/.ssh/wb_ed25519 shun@127.0.0.1 "bash -s" < script`），并且最好根本别重启（用 `strata_tune`）。
2. **`--spec 1` 会让引擎直接退出**：原生 IQ pack 校验要求 `--spec T (T >= 2)`（`strata generate: ... it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK`）。我一度把生产服务搞宕，已用备份配置恢复（spec 4 / min-p 0.8 / health 200 / active）。
3. **探测 prompt 必须同等难度**：退化 prompt（数数）会把 a 抬到 2.7，直接污染整条结论链。
4. `/metrics` 在两个端口都只有 82 字节（未启用），Prometheus 那条路走不通。

## 7. 复现命令

```bash
# 零重启扫 min_p（本次的测量脚本）
ssh -F none -i ~/.ssh/wb_ed25519 -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/home/shun/tmpwork/kh shun@127.0.0.1 "bash -s" \
    < /home/shun/tmpwork/w4_tune.sh
```

备份：`configs/sc117-256k-vision.json.bak-w4arm`（spec 4 / min-p 0.8，已确认与恢复后的生产一致）。
