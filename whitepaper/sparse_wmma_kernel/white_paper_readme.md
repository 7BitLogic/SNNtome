# Kernel 44 (`sparse_wmma_kernel`) — Engine Profile & Whitepaper Reproduction Guide

This directory contains the self-contained reference implementation, verification suite, and benchmark harness for the **SNNtome Sparse WMMA Connectome Engine (Kernel 44)** as presented in the accompanying whitepaper:

- **Whitepaper PDF:** [`main.pdf`](main.pdf) (*"Full VRAM Sparse Neural Network: Overcoming the Memory Wall"*)

---

## Table of Contents

1. [Engine Profile (Steckbrief)](#1-engine-profile-steckbrief)
2. [Reproduction & Quickstart Guide](#2-reproduction--quickstart-guide)
3. [Kernel Files & Test Suite Overview](#3-kernel-files--test-suite-overview)
4. [3-Stage Numerical Verification](#4-3-stage-numerical-verification)
5. [Copyright](#5-copyright)

---

## 1. Engine Profile (Steckbrief)

| Property | Specification |
| :--- | :--- |
| **Engine ID** | `sparse_wmma_kernel` (`Kernel 44`) |
| **Topology Pattern** | Immutable Clustered Connectome ($S$ postsynaptic neurons share $B$ presynaptic index columns) |
| **Cluster Sizes ($S$)** | $S \in \{16, 32, 64, 128, 256, 512\}$ |
| **Synaptic Fan-In ($B$)** | $B = 1024$ (standard) / $B = 2048$ (high-density) |
| **Presynaptic Canvas ($C_{\text{in}}$)** | $1,024 \to 40,000,000$ neurons (up to $99.9974\%$ sparsity in 16 GB VRAM) |
| **Arithmetic Precision** | Pure FP16 inputs/weights with FP32 hardware accumulation (`v_wmma_f32_16x16x16_f16`) |
| **On-Chip Memory (LDS)** | `0 KB` for $S=16$ (direct register path); `8–16 KB` double-buffered cooperative B-tile broadcast for $S \ge 32$ |
| **Peak In-Cache Throughput** | **$31.10\text{ TSyn/s}$ ($62.20\text{ TFLOPS}$)** at $M=512$ / **$32.06\text{ TSyn/s}$ ($64.12\text{ TFLOPS}$)** at $M=1024$ |
| **Sustained VRAM-Streaming** | **$17.97\text{--}25.16\text{ TSyn/s}$ ($35.94\text{--}50.31\text{ TFLOPS}$)** at $C_{\text{in}} = 10\text{M}$ ($M=512$) |
| **Reference Hardware** | AMD Radeon RX 9060 XT (RDNA4 / `gfx1200`, 32 CUs, 16 GB GDDR6, 32 MB Infinity Cache) |

*(For the complete roofline curves, cache hierarchy diagrams, and latency analyses, see Figures 1–6 in [`main.pdf`](main.pdf)).*

---

## 2. Reproduction & Quickstart Guide

All benchmarks deterministically synthesize their connectome indices, FP16 weights, biases, and activations in memory using fixed PRNG seeds. No external dataset downloads are required.

### Prerequisites
- **Hardware:** AMD RDNA4 GPU (`gfx1200`, e.g., Radeon RX 9060 XT) with 16 GB VRAM.
- **Software:** ROCm 7.x (`hipcc`), `nlohmann-json`, GNU `make`, Python 3.9+ (`numpy`, `scipy`, `matplotlib`).

### Step 1: Compile the Benchmarks

```bash
cd whitepaper/sparse_wmma_kernel
make all
```

### Step 2: Run the Whitepaper Benchmarks

Each benchmark automatically creates the `data/` directory and exports its empirical telemetry as JSON:

```bash
# 1. Primary Cin & Sparsity Scaling Suite (S = 16..512, Cin up to 40M, plus 4 GiB boundary test)
#    Outputs: data/benchmark_results_k44.json
#             data/benchmark_results_k44_m128_40m.json
#             data/benchmark_k44_dual_buffer_s128_m128.json
./bin/benchmark_whitepaper_k44

# 2. Batch Dimension Scaling Suite (M = 16 .. 16384)
#    Outputs: data/benchmark_batch_scaling_results.json
./bin/benchmark_batch_scaling

# 3. 3-Stage Numerical Error Collapse Verification (FP16 GPU vs. FP64 CPU Ground Truth)
#    Outputs: data/error_collapse_results.json
./bin/error_collapse_test
```

### Step 3: Run Independent Python Verification & Generate Plots

```bash
# Verify 3-stage error collapse and Kolmogorov-Smirnov sparsity invariance
make verify

# Regenerate publication plots (Figures 3, 4, 5, and 6) into figures/
make plots
```

---

## 3. Kernel Files & Test Suite Overview

```text
whitepaper/sparse_wmma_kernel/
├── Makefile                                # Standalone build, verification & plot harness
├── white_paper_readme.md                   # This documentation file
├── main.pdf                                # Compiled Whitepaper PDF
├── kernels/
│   ├── kernel44_immutable.cuh              # Primary FP16 Sparse WMMA kernel (single-descriptor + 64-bit path)
│   └── kernel44_dual_buffer.cuh            # Diagnostic Dual-Buffer (rsrc_X0/X1) variant for Fig. 4 (4 GiB boundary)
├── tests/
│   ├── benchmark_common.cuh                # Shared 256 MiB multi-wave ring-buffer pool & CPU reference validator
│   ├── benchmark_whitepaper_k44.cpp        # Primary Cin scaling & 40M connectome benchmark runner
│   ├── benchmark_batch_scaling.cpp         # Batch scaling benchmark runner (M = 16 .. 16384)
│   └── error_collapse_test.cpp             # Rigorous 3-stage numerical error collapse test
└── scripts/
    ├── check_duplicates.py                 # Automated C++ code duplication & style linter
    ├── plot_scaling_all_clusters_loglog.py # Generator for Figure 3 (All-clusters log-log scaling)
    ├── plot_scaling_m128_40m_loglog.py     # Generator for Figure 4 (M=128 40M scaling & Dual-Buffer comparison)
    ├── plot_batch_scaling_loglog.py        # Generator for Figure 5 (Batch size scaling)
    ├── plot_distributions.py               # Generator for Figure 6 (Tensor distribution histograms)
    └── verify_whitepaper_error_collapse.py # Standalone Python 3-stage statistical verification
```

---

## 4. 3-Stage Numerical Verification

To prove that the hardware-accelerated FP16 WMMA pipeline introduces zero hidden approximation shortcuts, `error_collapse_test.cpp` and `verify_whitepaper_error_collapse.py` evaluate every output tile against three analytical reference stages across $C_{\text{in}} \in [1,024, 20,000,000]$:

```text
[Stage 1: Continuous Unquantized FP32]  ──▶  MAE ~ 3.9e-3 (RelMAE ~ 0.046% vs RMS(Y) ~ 10.7)
                    │
                    ▼  (1.31x Quantization Collapse)
[Stage 2: Exact FP16 Quantized CPU]     ──▶  MAE ~ 3.0e-3 (Accumulation Order Delta)
                    │
                    ▼  (WMMA Tile Verification)
[Stage 3: Stepwise WMMA Fragment Chunk] ──▶  MAE ~ 3.0e-3, MaxDiff ~ 0.031 (Invariant across Cin)
```

---

## 5. Copyright

Compare LICENSE.md
