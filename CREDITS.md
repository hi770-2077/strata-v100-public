# 致谢与出处

本仓库是对 [Niko1221/Strata](https://github.com/Niko1221/Strata) 社区的实测回馈，
所有内容基于上游开源代码（MIT License）。

## 上游项目

- **Strata** — https://github.com/Niko1221/Strata · MIT License
  - 本仓库的补丁针对 PR #1771（prompt 阶段的 WMMA tensor-core 路径）
  - 引用的官方文档：`docs/MULTI_GPU.md`、`docs/SECOND_GPU.md`、`docs/NVIDIA_V100.md`、
    `docs/OLDER_GPUS.md`、`docs/VRAM_ELASTIC.md`
  - 本仓库的 `LICENSE` 是上游的 MIT 许可（Copyright (c) 2026 Niko1221 and the Strata contributors）

## 参考的社区讨论

以下 issue 里的实测数据构成本仓库多卡建议矩阵的【社区】级证据：

| Issue | 主题 |
|---|---|
| [#1401](https://github.com/Niko1221/Strata/issues/1401) | V100 用户配置讨论（10+ 台机器） |
| [#1771](https://github.com/Niko1221/Strata/issues/1771) | prompt WMMA tensor-core（本仓库补丁针对的 PR） |
| [#2048](https://github.com/Niko1221/Strata/issues/2048) | 4× V100 基准 |
| #1884 / #1352 / #1447 / #854 | 多卡 / 专家缓存 / MMQ 相关 |

也参考了 [llama.cpp](https://github.com/ggml-org/llama.cpp)（Unsloth fork b10995）的
`qwen4exp` 架构支持与 mma 内核实现 —— 我们计划把它的 IQ3_S mma 路径移植进 Strata
（见 [`docs/llama-cpp-mining-notes.md`](docs/llama-cpp-mining-notes.md) 末节）。

## 本仓库的实测环境

2× Tesla V100-PCIE-32GB（GV100GL, sm_70）· 驱动 580.178.04 · CUDA 12.8 · CachyOS(x86_64)
模型：Qwen3.8-Flash-Next GSQ-RCO-abliterated IQ3_S（48 层 / 512 专家 top-10 / GDN+QSA 混合）

## 引用规范

本仓库的每个数字都标注了证据等级（【实测】带复算方法、【社区】附 issue 编号、【推导】带推理链）。
**转引时请连同出处一起转引**，不要只搬数字 —— 不同硬件拓扑下的数字不可直接比较。

## 联系我们

发现错误、有补充实测、或者补丁在你的机器上有问题，欢迎开 issue。
特别是 [`patches/v100-multigpu-wmma-fix.patch`](patches/v100-multigpu-wmma-fix.patch) ——
如果它对上游有价值，也欢迎直接提给 Niko1221。