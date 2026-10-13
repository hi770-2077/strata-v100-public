# 多卡 layer-split 下 WMMA 启动失败（已修）+ IQ3_S gate/up 实测数据

> 测试环境：2× Tesla V100-32GB（sm_70，GV100GL），驱动 580.178.04，CUDA 12.8，
> Strata v0.1.43 + PR #1771 的 head（0742b98）。
> 这份内容可直接用于向 [Niko1221/Strata](https://github.com/Niko1221/Strata) 反馈。
> 补丁文件：[`../patches/v100-multigpu-wmma-fix.patch`](../patches/v100-multigpu-wmma-fix.patch)
>
> **如果你只想知道结论**：第 1 节是 bug（10 行修复），第 2 节是实测数据，
> 第 3 节是我们的一个可选方案，第 4 节是正确性证据。

## 1. Bug：layer split 下第二张卡的 WMMA 启动必失败

`gemm_iq_f16_grouped` 里：

```cpp
static bool attr = false;
if (!attr) {
    cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem(256));
    ...
    attr = true;
}
```

`cudaFuncSetAttribute` 是**按设备（per-context）**的。这些内核要 83 KB 动态共享内存
（>48 KB 默认上限，必须 opt-in），于是：

- 单卡：第一个调用把属性设上，之后都能启动 —— 你们的基准机是单 V100，所以没暴露。
- 多卡 layer split（我们 25+23）：第一张卡（layers 0-24）的调用设上属性，
  第二张卡（layers 25-47）的调用**没有设过**，启动直接 `cudaErrorInvalidValue`，
  表现为 `prefill: the WMMA expert tiles could not launch`，第一个 prompt 请求就失败。

修复（按设备索引，已在我们 fork 里实测通过）：

```cpp
static bool attr[16] = {};   // per device: the attribute is per-context, a layer split launches on both cards
int dev = 0;
if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
const size_t di = (dev >= 0 && dev < 16) ? (size_t) dev : 0;
if (!attr[di]) {
    cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<256, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem(256));
    cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem(128));
    cudaFuncSetAttribute(gemm_iq_f16_grouped_kernel<128, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem(128));
    attr[di] = true;
}
```

## 2. 实测：IQ3_S gate/up 在 FP16 瓦片上比 MMQ 慢 ~20%（你们的 rig 没覆盖这个格式）

我们的包是 IQ3_S gate/up（1280×2560）+ IQ4_NL down（2560×640）。
prompt-iso（3141 token，`STRATA_PREFILL_TIMING=1` 相位表，ms/请求）：

| 阶段 | PF_WMMA=0（MMQ） | PF_WMMA=1（全 WMMA） |
|---|---:|---:|
| gemm gate/up | 238-255 | 287-300（**+20% 慢**） |
| gemm down | 129-131 | 102-109（**-20% 快**） |
| GPU timeline | 645-672 | 674-724 |

机制推测：Volta 的 MMQ 走 int8 mma（吞吐 2× FP16，激活 q8_1 仅 1 字节/元素），
IQ3_S 的 256 宽 superblock 反量化到共享内存也不便宜；而 down 的 K=640 太小，
MMQ 的 quantize + 每 product 启动开销占比过高，融合瓦片赢。
IQ4_NL down 的 WK=128 通道（ea2ecc6e）在我们包上确实如预期工作，parity 也过。

## 3. 我们的取法：down-only（PF_WMMA=2）

GU 保持 MMQ、只把 down 送上瓦片（`pf_wmma_down()`，约 40 行，核心是
`swiglu_split_f16` + 现有 down 调用，bounds 与 MMQ down 路径逐字相同）。
prompt-iso 三轮 ABAB（同二进制 env 切换，best-of-3 中值，引擎日志互证）：

- PF_WMMA=0：1081 / 1082 tok/s
- PF_WMMA=2：1127 / 1127 tok/s → **+4.1~4.3%**
- decode 不受影响（预填充专用路径）

如果你们觉得 down-only 有普适价值（Q2_0 down 的包应该同样受益，K=640 的
几何形状决定了开销结构），欢迎吸收进 PR；不吸收也没关系，我们 fork 自持。

## 4. 正确性证据

- `pf_wmma_parity`：8 种格式全过（IQ3_S 1280×2560 max err 1.3e-3 of 259；
  IQ4_NL 2560×640 max err 1.6e-4 of 65.4）。
- 新旧二进制在 PF_WMMA=0 下输出逐字节一致（非 WMMA 路径零改动）；
  同 arm 重跑逐字节一致（确定性）。
- PF_WMMA=2 输出通顺合理；逐 token argmax 翻转率与你们 teacher-forcing
  表里 MMQ vs WMMA 的 94.5-98.3% 一致率同量级（FP32 求和序差异，预期内）。

## 5. 环境备注

- 双卡 layer-split 25（CUDA0: layers 0-24，CUDA1: 25-47），专家 100% 常驻。
- 本机沙箱一度看不到 GPU（nvidia-smi failed），测试走 SSH 到宿主机执行。
- 编译配方：Ninja + gcc-14（CUDA 12.8 nvcc 不认 gcc 16）+
  `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70`。
