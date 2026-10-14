# 窗口 ③-1：把那 45% 的"其他"逐项命名（2026-10-15）

> 窗口 ③-0 留下一个残差：`ms/窗 = 7.79 + 4.49×T` 里，dense 实测只解释 3.98 + 0.55T，
> 剩下 4.24 ms 固定 + 3.94 ms/行 **没有名字**（合计约 45% 的窗口）。
> 本轮用引擎自带的相位剖析器 `STRATA_VERIFY_PROFILE=1` 把它拆开。
> **结论：两项——专家 VRAM 读（1.20 ms/行）和超连接投影（4.0 ms/窗）——是新的最大单项，
> 而超连接投影是全模型唯一没量化的读（BF16），只跑到 330 GB/s，是最值得动的一块。**

---

## 1. 方法

`STRATA_VERIFY_PROFILE=1` + `STRATA_DECODE_TIMING=1`（**后者是前者的打印门控**，只开前者不打印），
每个请求结束时往引擎 stderr 打一份逐相位账单（`verify.cpp:1801` `profile_report`）：

```
strata decode GPU stages (ms/window): GDN layers: <相位> <ms> ... | QSA layers: ... | total <ms>/window
```

⚠️ 剖析器会**关掉 shared-expert 分支和 df_branch**，所以**绝对值不是引擎的值**；能用的是**两个 T 之间的差**。
按请求覆盖 `spec_min_p` 零重启扫 T：`min_p=1.0` → T=1，`min_p=0.0` → T=4。一次重启拿两份账单。

另外把基准扩展成 `win22_dense_bench hc`，单独量超连接投影。

## 2. 相位账（双卡两 stage 相加；T=1 / T=4，ΔT=3.00）

| 相位 | T=1 | T=4 | Δ | **ms/行** |
|---|---|---|---|---|
| **VRAM hits**（专家 VRAM 读） | 3.02 | 6.63 | +3.61 | **1.20** |
| **q8+qkv/q-idx gemv** | 2.37 | 3.59 | +1.22 | **0.41** |
| hc0+hc1 **down** | 1.78 | 2.83 | +1.05 | **0.35** |
| shared+quant | 1.70 | 2.36 | +0.66 | 0.22 |
| hc0+hc1 **up** | 1.76 | 2.39 | +0.63 | 0.21 |
| out-proj | 1.48 | 1.90 | +0.42 | 0.14 |
| z（GDN 输出归一） | 0.99 | 1.34 | +0.35 | 0.12 |
| router+plan+bell | 1.33 | 1.67 | +0.34 | 0.11 |
| rec（GDN 递归） | 0.40 | 0.63 | +0.23 | 0.08 |
| hc0+hc1 norm | 1.02 | 1.20 | +0.18 | 0.06 |
| ab / k/v+norm-rope / copy+combine | 0.90 | 1.09 | +0.19 | 0.06 |
| head | 0.95 | 1.25 | +0.30 | 0.10 |
| attention | 0.43 | 0.44 | +0.01 | 0.00 |
| scores+topk / kv-resolve / gate | 0.35 | 0.37 | +0.02 | 0.01 |
| conv | 0.20 | 0.20 | 0.00 | 0.00 |
| **小计** | **19.12** | **28.39** | **+9.27** | **3.09** |
| 剖析器关掉的分支 | | | | **~1.40** |
| **引擎实测（窗口③-0）** | | | | **4.49** |

**读法**：
- **`attention`、`scores+topk`、`kv-resolve`、`conv` 的 Δ≈0** ⇒ 注意力这块在当前上下文长度下基本免费，
  "长上下文 +25–35%"的说法要重新审视（它是 KV 读，随 context 涨，不是随 T 涨）。
- **`VRAM hits` = 1.20 ms/行，是每行成本的最大单项（占剖析器每行成本的 39%）**，
  而且它随行数线性涨 —— 每一行都要读自己的 top-10 专家，不共享。
- 剩下的 ~1.40 ms/行是剖析器关掉的分支，**无法归因**，要另找手段。

## 3. 超连接投影：全模型唯一没量化的读

`native_dense.cpp` 的 `eligible()` 白名单只有 6 类张量，**hc_attn_/hc_ffn_ 的 up/down 不在里面**，
它们在 GGUF 里是 **BF16**，走 `bf16_gemv_fp32_mmvf_multi`：

| 矩阵 | n_in → n_out | MiB/层 | nc=1 | nc=2 | nc=3 | nc=4 |
|---|---|---|---|---|---|---|
| hc_attn_up / hc_ffn_up | 320 → 10240 | 6.25 ×2 | 337 GB/s | 231 | 207 | **194** |
| hc_attn_down / hc_ffn_down | 10240 → 320 | 6.25 ×2 | **742** | 581 | 433 | 350 |

**合计 25.0 MiB/层 = 1.17 GiB/前向**（比 native dense 的 2.26 GiB 少一半，但只量化了一半的效率）：

| | nc=1 | nc=2 | nc=3 | nc=4 |
|---|---|---|---|---|
| HC 合计 | 2.72 ms | **3.79** | 4.47 | 5.02 |
| 有效带宽 | 463 GB/s | **330** | 281 | 250 |

⇒ **T≈2.2 时 HC ≈ 4.0 ms，占生产窗口 22%，是仅次于 dense（31%）的第二大读**，
而它的有效带宽（330 GB/s）只有 dense 的一半、只有 Q8_0 `ssm_out` 单列（693 GB/s）的一半。

**两个方向**：
1. **把 HC 权重量化成 Q8_0**：字节减半（1.17 → 0.58 GiB），且能走已经跑得更好的 `native_mmvq`。
   估算 4.0 → ~1.2 ms ⇒ **+19%**。⚠️ **有损**（BF16→Q8_0 重量化），必须做输出质量评估。
   现成的 `STRATA_HC_Q8=1` 对我们**无效**：`native_dense.cpp:329` 要求 `tensor.type == 8`（Q8_0），
   而我们的 hc_* 是 BF16（type 30）——要自己造 Q8_0 副本。
2. **优化 bf16 GEMV 内核本身**：`hc_*_up`（320→10240）只跑 337 GB/s，
   而同为 bf16 的 `hc_*_down`（10240→320）跑 742 GB/s ⇒ 问题在"短 n_in + 多输出行"的访存模式，
   不是 bf16 本身。无损，但要写内核。

## 4. 更新后的窗口预算（T=2.22，17.76 ms）

| 组成 | ms/窗 | 占比 | 状态 |
|---|---|---|---|
| dense（6 个量化矩阵，native_mmvq） | 5.5 | 31% | 已实测（②-2） |
| MoE 专家读（48 层 × top-10） | 4.7 | 26% | 估（0.95 GB/行 ÷ 450） |
| **HC 投影（4 个 BF16 矩阵）** | **4.0** | **22%** | **本轮实测，带宽最差** |
| 注意力 + GDN + router + head + 间隙 | 3.5 | 20% | 本轮按相位拆分 |

⇒ **路线图再变**：HC 从"没人看过"变成"第二大项且效率最差"。
窗口 ②（dense）上限 +6~8%；**HC 上限 +12~19%，是现在最大的单一机会**。

## 5. 已上线 / 待决

- ✅ **已上线**：`kIlRows_70` 两格修正（`STRATA_MMVQ_IL_ROWS="14:2:0,2,2,1,2;12:2:0,2,4,2,2"`，
  drop-in `/etc/systemd/system/strata-sc117.service.d/30-ilrows.conf`）。
  内核级 −5.6 µs/层 × 48 = −0.27 ms/窗；端到端 110.0/110.1 tok/s（基线 110.1）——
  **1.5% 落在 ±1–2% 的噪声里，测不出来**，但 IL 输出与 `native_mmvq` 逐位相同，零风险故保留。
- ⬜ **待决（最大机会）**：HC 权重量化到 Q8_0（有损，需质量评估）或重写 bf16 GEMV（无损，需写内核）。
- ⬜ 每行成本里 ~1.40 ms 的"剖析器关掉的分支"仍无名字。

## 6. 复现

```bash
ssh -F none -i ~/.ssh/wb_ed25519 -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/home/shun/tmpwork/kh shun@127.0.0.1 "bash -s" < /home/shun/tmpwork/w31_profile.sh
ssh ... < /home/shun/tmpwork/w31_hc_bench.sh      # HC 投影带宽
bash /home/shun/tmpwork/build_win22_bench.sh      # 基准（含 hc 模式）
```

新增/改动：`src/kernels/win22_dense_bench.cpp`（+`hc` 模式）、`w31_profile.sh`、`w31_hc_bench.sh`、
`w22_ship.sh`（上线脚本）、`/home/shun/push_strata_repos.sh`（推送脚本，等 PAT）。
