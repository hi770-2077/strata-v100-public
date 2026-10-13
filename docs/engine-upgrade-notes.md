# 引擎升级实录：v0.1.41 → v0.1.43

> 日期：2026-10-13 · 机器：双 Tesla V100-32GB（layer-split 25）· 模型：Qwen3.8-Flash-Next GSQ-RCO IQ3_S
> 结论先行：**升级完成，已切生产。速度持平（decode ~100 tok/s），换来 NaN guard 安全网与 307 个修复。新开关全体检过，本机无增益，配置保持纯净。**

---

## 一、307 个提交里，与我们相关的只有三件

| 项 | 内容 |
|---|---|
| 版本 | v0.1.41 → **v0.1.43** |
| 提交跨度 | **307 个提交** |
| **与我们相关的** | ① **NaN guard**（非有限 logits 自动丢弃缓存重读，`STRATA_NAN_GUARD=0` 可关，代价 <0.2%）—— 长任务安全网；② prefill 分块记录走 pinned buffer；③ route tail skip 默认开启（对我们无增益，见第四节） |
| **与我们无关的** | AMD gfx12、Intel Battlemage（SYCL）、CUDA sm_86 Q3_K —— 占提交数 **90%+** |

**升级的第一课：提交数不等于收益面。** 307 个提交里九成是别的架构，我们的 sm_70 + IQ3_S + 双卡组合本来就不在受益面内。

---

## 二、A/B 测速（同机同配置，唯一变量 = 二进制）

| 指标 | v0.1.41 | v0.1.43 | 判定 |
|---|---|---|---|
| decode 中值（5 轮 × 512 tok） | 99.31 tok/s | **99.50 tok/s** | +0.2%，噪声内持平 |
| TTFT 中值 | 0.23 s | 0.28 s | 持平 |
| prefill（17.5K token 冷 prompt ×3） | ~1030 tok/s | ~1030 tok/s | 持平 |
| 长生成（600 tok） | 176 tok/s | 176~183 tok/s | 持平 |
| `max_tokens` 护栏 | — | 999999 → HTTP 200 正常回复 | 护栏仍在 |
| 稳定性 | — | 5 轮 bench + 3 轮 prefill + 冒烟，零报错 | 通过 |

**解读**：持平即预期。升级的真实收益是**安全**（NaN guard 对长任务、尤其是长时间 agent 循环是实打实的保险）和**修复**（通用层的边角问题）。

---

## 三、编译：两个新坑

1. 🔴 **v143 的 CMake 默认拒绝 sm_70**：不显式 `-DSTRATA_EXPERIMENTAL_SM60=ON` 直接
   FATAL_ERROR（"needs compute capability 7.5 or newer"）。v141 时代没这个门。
2. 🔴 **不加 `-DSTRATA_ENABLE_CUDA=ON` 会静默编出 CPU 版**：configure 不报错但不生成主目标。
   第一遍就这么白编了一次（26 秒完成还没觉得不对）。
   **验证 GPU 版的方法**：`nm -D <bin> | grep -c cudaMalloc` ≥ 6。
3. 老坑依旧：`CMAKE_CUDA_HOST_COMPILER` 指向 g++-14（nvcc 12.8 不认 gcc 16）。

完整编译参数见 [`scripts/build-sm70.sh`](../scripts/build-sm70.sh)。

---

## 四、新开关全体检（本机实测）

| 开关 | 别处收益 | 本机实测 | 处置 |
|---|---|---|---|
| `STRATA_ROUTE_TAIL_SKIP`（0.1.42 起默认开） | 单卡 +10~24% | 双卡 layer-split 专家命中率 99.7% 档位，官方同档数据 0.998 | 保持默认 |
| `--aux-cpus auto` | 别处 +5.5% | 99.83 vs 99.50 | 不采纳（噪声） |
| `STRATA_ROUTE_RESIDENT=1` | 单卡 +1.6~2.4% | 98.85 vs 99.50 | 不采纳（噪声） |
| `STRATA_ADAPT_LAG=2` | 混合 | 未测 —— 增益机制是「少等 miss 专家拷贝」，我们无 miss 可等 | 不采纳 |
| `STRATA_STAGE_PIN=1` | RTX5070 +1.2% | **🔴 严禁**：官方 0.1.41 发布门实测 IQ3_S 在 4096-token prompt 后解码损坏，正是我们的量化档 | 永久关闭 |

**为什么都没用**：这些优化的增益全部来自「等不在显存里的专家」—— 单卡小显存卡的痛点。
我们双卡 64GB + kv-resident 49152，几乎所有专家常驻，等待时间 ≈ 0，优化空间也 ≈ 0。

> **又一个「加内存 = 全面提升」的镜像版：「别人 +10~24% 的开关，我们没有可增益的空间」。**
> 开关台账里这类"不适用"比"有害"更常见 —— 详见 [`flag-ledger.md`](flag-ledger.md)。

---

## 五、部署与回滚

- 生产服务 config 的 exe 指向 v143，其余参数零改动
- 环境变量 drop-in（`STRATA_SM70_TABLE=1` / `STRATA_SPLIT_OWN=auto`）不受影响
- 旧二进制原地保留
- **一键回滚**：config 的 exe 改回旧路径 + 重启服务（约 90 秒）

**升级可以很激进，只要回滚很便宜。** 90 秒回滚是这次敢直接切生产的前提。

---

## 六、引擎侧结论：再次见底

本机的性能方程里，引擎通用内核已不是短板（decode ~100-183 tok/s 取决于生成长度，prefill ~1030 tok/s）。
进一步的收益只能来自：

1. **使用姿势**（prefix 复用，已做满 —— 见 [`client-call-patterns.md`](client-call-patterns.md)）
2. **任务拆解**（会话卫生）
3. **512K 上下文**（待真实需求）

---

## 七、什么时候该升一个「没有变快」的版本

这次升级的决策模式，比升级本身更值得记：

| 问题 | 本次的答案 |
|---|---|
| 性能变快了吗？ | 没有（噪声内持平） |
| 那为什么升？ | **安全网**（NaN guard）+ **修复**（307 提交里的通用层边角） |
| 风险是什么？ | 新默认值可能反向影响我们（route tail skip）→ 逐项体检 |
| 退路是什么？ | 90 秒一键回滚，旧二进制原地保留 |
| 结论 | **升，但体检完再把生产切过去** |

**A flat A/B is not a reason to stay on an old version — it is a reason to read the changelog for safety fixes.**
