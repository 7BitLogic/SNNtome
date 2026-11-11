#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_LIB="${SCRIPT_DIR}/libk44wavec.so"
SRC_CU="${SCRIPT_DIR}/k44_wave_executor.cu"

BACKEND="${BACKEND:-auto}"
if [[ "${BACKEND}" == "auto" ]]; then
    if [[ -x "/opt/rocm/bin/hipcc" ]] && /opt/rocm/bin/rocminfo >/dev/null 2>&1; then
        BACKEND="hip"
    elif command -v nvcc >/dev/null 2>&1 || [[ -x "/opt/cuda/bin/nvcc" ]]; then
        BACKEND="cuda"
    elif [[ -x "/opt/rocm/bin/hipcc" ]]; then
        BACKEND="hip"
    else
        echo "ERROR: Neither hipcc nor nvcc found." >&2
        exit 1
    fi
fi

if [[ "${BACKEND}" == "hip" ]]; then
    HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
    OFFLOAD_ARCH="${OFFLOAD_ARCH:-gfx1200}"
    echo "[wavec] Building ${OUT_LIB} with HIP (${HIPCC}, --offload-arch=${OFFLOAD_ARCH})..."
    "${HIPCC}" -O3 -std=c++17 --offload-arch="${OFFLOAD_ARCH}" -fPIC -shared -x hip "${SRC_CU}" -o "${OUT_LIB}"
elif [[ "${BACKEND}" == "cuda" ]]; then
    NVCC="${NVCC:-$(command -v nvcc || echo /opt/cuda/bin/nvcc)}"
    CUDA_ARCH="${CUDA_ARCH:-sm_89}"
    echo "[wavec] Building ${OUT_LIB} with CUDA (${NVCC}, -arch=${CUDA_ARCH})..."
    "${NVCC}" -O3 -std=c++17 -arch="${CUDA_ARCH}" -Xcompiler -fPIC -shared "${SRC_CU}" -o "${OUT_LIB}"
else
    echo "ERROR: Unknown BACKEND=${BACKEND} (expected hip or cuda)" >&2
    exit 1
fi

echo "[wavec] Built ${OUT_LIB} successfully."
