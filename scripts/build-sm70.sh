#!/bin/bash
# 构建 sm_70 (Tesla V100) 版 Strata — 与生产完全相同的配方。
# 用法: ./build-sm70.sh [源码目录=仓库根] [构建目录=源码目录/build-cuda]
# 产物: <build>/strata + <build>/pf_wmma_parity (parity 需要 GPU 才能跑)
#
# 前置: CUDA 12.8 + gcc-14 (nvcc 不认 gcc 16) + Ninja
#   Arch/CachyOS: pacman -S cuda gcc-14 ninja
#   Ubuntu 22.04: 从 NVIDIA CUDA apt 源装 cuda-12-8 + ninja-build
set -e
export TMPDIR="${TMPDIR:-/var/tmp}"     # 构建目录不要放/tmp（容器里 /tmp 常常很小，nvcc 会写爆）
export PATH=/opt/cuda-12.8/bin:$PATH
export CUDA_HOME=/opt/cuda-12.8

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"                    # 默认仓库根
[ $# -ge 1 ] && SRC="$(cd "$1" && pwd)"
BLD="${2:-$SRC/build-cuda}"
mkdir -p "$BLD"

# 三个坑 (2026-10-13 实测, 缺一个都不行):
#  1) v0.1.43 CMake 默认拒 sm_70, 必须 -DSTRATA_EXPERIMENTAL_SM60=ON
#  2) 不显式 -DSTRATA_ENABLE_CUDA=ON 会静默编出 CPU 版 (不报错, 但不生成主目标)
#     验证 GPU 版: nm -D build/strata | grep -c cudaMalloc >= 6
#  3) CUDA 12.8 nvcc 不认系统 gcc 16, 必须显式 -DCMAKE_CUDA_HOST_COMPILER=g++-14
cmake -S "$SRC" -B "$BLD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/opt/cuda-12.8/bin/nvcc \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-14 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DSTRATA_BUILD_TESTS=ON > "$BLD/configure.log" 2>&1 || { tail -40 "$BLD/configure.log"; exit 1; }
echo "=== configure OK, building (3m35s @ -j16) ==="
cmake --build "$BLD" --target strata pf_wmma_parity -j 16 > "$BLD/build.log" 2>&1 || { tail -60 "$BLD/build.log"; exit 1; }
echo "=== BUILD DONE ==="
ls -la "$BLD/strata" "$BLD/pf_wmma_parity"
echo "=== CUDA symbols: $(nm -D "$BLD/strata" 2>/dev/null | grep -c cudaMalloc) (>=6 才是 GPU 版) ==="
