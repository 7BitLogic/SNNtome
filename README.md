# SNNtome

This is the first publication of the SNNtome package. 

In its first stages this package is meant to give insights into biologically **connectome** inspired artificial Neural Networks. Since biological networks are sparse in nature I will call this package Sparse Neural Network Connectome or SNNtome in short.

Since neural cells in a brain can (theoretically) access any other neural cell, the SNNtome neurons can address any given neuron-output via all traditional layers. Since under these circumstances the term of "layers" becomes obsolete. I introduced a "wave" architecture. The waves describe a bunch of similar neurons firing at the same time. They have full access to the full substrate of all neural activities, so they can reproduce a layerd structure but also can have virtually any other layout. Therefore, the space of possible architectures is vastly larger.

Comming from neurological imaging and having a past with close hardware GPU programming, I always wanted to create a software which is as close as possible to real biological neurological systems, while being GPU fast.

With the advent of agentic (AI) programming I was able to develop a Kernel (current version is 44) which is able (for batch operations) to deliver up to 50% of the theoretical FP16 performance of my GPU. Which is very exciting since this allows to run normal NNs in a sparse setting. This allowed me to run LLMs at comparable speeds (within a few per cent) to a local Ollama server.
For more details on the current kernel have a closer look at the whitepaper "Full VRAM Sparse Neural Network: Overcoming the Memory Wall" in "whitepaper/sparse_wmma_kernel/".
You will also find some CUDA variants, which I tested on online server hardware. All currently in a quite early stage.

You will find a "compiler" which is able to convert traditional Neural Networks into SNNtomes, storing compiled plans in the tcff format (tagged connectome file format) that combines weights, biases, connections as well as the wave architecture.

All modules and functions are quite new and in an early experimental stage.

For all the licensing stuff please check the LICENSE.md so I do not have to reproduce it here. 

**Preview only (no permission to compile or execute, see LICENSE.md).**  

---

## Table of Contents

1. [Installation & Quickstart](#1-installation--quickstart)
2. [Available Kernel Engines & Wave Compiler](#2-available-kernel-engines--wave-compiler)
3. [Repository Structure](#3-repository-structure)
4. [Publications & Copyright](#4-publications--copyright)

---

## 1. Installation & Quickstart

### Prerequisites
- **Compiler & GPU Stack:**
  - *AMD Backend:* ROCm 7.x (`hipcc` at `/opt/rocm/bin/hipcc`), C++17 host compiler (`gcc` / `clang`), GNU `make`.
  - *NVIDIA Backend (early stage):* CUDA 12.x (`nvcc` for Ada `sm_89` / Blackwell `sm_100a`).
- **C++ Dependencies:** `nlohmann-json` (for whitepaper benchmark telemetry export).
  - Arch Linux: `sudo pacman -S nlohmann-json`
  - Ubuntu / Debian: `sudo apt install nlohmann-json3-dev`
- **Python Environment (for `wavec` examples, plot generation & verification scripts):**
  - Python 3.9+ with `numpy`, `scipy`, `matplotlib`.

### Initial Build & Run

```bash
# 1. Clone repository
git clone https://github.com/7BitLogic/SNNtome.git
cd SNNtome

# 2a. Compile and run the whitepaper benchmark & 3-stage numerical verification suite (ROCm)
make all
./whitepaper/sparse_wmma_kernel/bin/error_collapse_test

# 2b. Build the Wave Compiler shared library (auto-detects ROCm hipcc or NVIDIA nvcc)
make wavec
python3 wavec/examples/example_yolov8.py
python3 wavec/examples/example_llm.py
```

For full whitepaper reproduction instructions (benchmarks, telemetry generation, and publication plots), see [`whitepaper/sparse_wmma_kernel/white_paper_readme.md`](whitepaper/sparse_wmma_kernel/white_paper_readme.md).
For compiling and running neural networks with the Wave Compiler (`wavec`) and `.tcff` files, see [`wavec/README.md`](wavec/README.md).

---

## 2. Available Kernel Engines & Wave Compiler

`SNNtome` provides modular sparse compute engines ("kernels") and a 5-pass graph-to-wave compiler (`wavec`) that can be selected and combined depending on the network topology, sparsity regime, and hardware target:

| Component / Engine                   | Short Description                                                                                                          | Recommended Application                                                                                                                                              | Hardware Target                           | Documentation & Files                                                                                                                    |
| :----------------------------------- | :------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **`sparse_wmma_kernel` (Kernel 44)** | Immutable Clustered Connectome Engine (FP16 Wave32 WMMA, Cooperative LDS Broadcast & 2-Step Ping-Pong Pipeline)            | Static or structurally fixed ultra-sparse topologies ($C_{\text{in}}$ up to $40\text{M}$, sparsity $>99.99\%$) with bundled dendritic clusters ($S \in \{16..512\}$) | AMD RDNA4 (`gfx1200`)                     | [Engine Profile & Guide](whitepaper/sparse_wmma_kernel/white_paper_readme.md) · [Whitepaper PDF](whitepaper/sparse_wmma_kernel/main.pdf) |
| **`kernel44_rocm_pack`**             | Canonical RDNA4 pure-FP16 WMMA & Receptive-Field Token-Shift Kernel Pack (`connectome_k44_token_shift_f16_kernel`)         | $1\times 1$ pointwise projections, $3\times 3$ / $2\times 2$ spatial convolutions without `im2col`, and fused epilogues on `d_Canvas`                                | AMD RDNA4 (`gfx1200`)                     | [`kernels/rocm/kernel44_rocm_pack.cuh`](kernels/rocm/kernel44_rocm_pack.cuh)                                                             |
| **`kernel44_cuda_pack`**             | Canonical NVIDIA pure-FP16 Tensor Core & Token-Shift Kernel Pack (`k44_cuda_smlS`, `k44_cuda_lrgS`, `tcgen05`)             | Cross-vendor execution of clustered connectomes and receptive-field waves on NVIDIA Tensor Cores (early stage)                                                       | NVIDIA Ada (`sm_89`), Blackwell (`sm_100a`) | [`kernels/cuda/kernel44_cuda_pack.cuh`](kernels/cuda/kernel44_cuda_pack.cuh)                                                             |
| **`kernel44_valu_pack`**             | Shared HIP & CUDA 2D Spatial & 1D Transformer VALU Auxiliary Kernel Pack                                                   | In-place non-matrix operations on `d_Canvas` (`MaxPool`, `Upsample2x`, `S2D_2x2`, `Scatter_UpConv2x2`, `RMSNorm`, `SwiGLU`, `RoPE`, `GQA Causal Attention`)          | AMD ROCm & NVIDIA CUDA                    | [`kernels/common/kernel44_valu_pack.cuh`](kernels/common/kernel44_valu_pack.cuh)                                                         |
| **`wavec` (Wave Compiler)**          | 5-Pass Wave Graph Compiler, `.tcff` (Tagged Connectome File Format) Serializer & Unified ROCm/CUDA GPU Executor            | Translating traditional 2D CNNs (e.g. YOLOv8) and 1D Transformers/LLMs into zero-workspace `.tcff` wave plans                                                        | AMD ROCm & NVIDIA CUDA                    | [WaveC Guide](wavec/README.md) · [`wavec/`](wavec/)                                                                                      |

---

## 3. Repository Structure

```text
SNNtome/
├── Makefile                                            # Top-level build & style-check harness
├── README.md                                           # Main library overview & engine catalog
├── kernels/
│   ├── common/kernel44_valu_pack.cuh                   # Shared HIP/CUDA 2D spatial & 1D VALU auxiliary kernels
│   ├── rocm/kernel44_rocm_pack.cuh                     # Canonical RDNA4 (gfx1200) pure-FP16 WMMA kernel pack
│   └── cuda/kernel44_cuda_pack.cuh                     # Canonical NVIDIA Ada/Blackwell pure-FP16 Tensor Core pack
├── wavec/                                              # 5-Pass Wave Graph Compiler & Unified GPU Executor
│   ├── README.md                                       # Wave compiler documentation & CLI usage guide
│   ├── build_wavec.sh                                  # Shared library build script (ROCm hipcc / NVIDIA nvcc)
│   ├── k44_plan_types.hpp                              # Resolution descriptors, hardware profiles & WMMA/MMA packing
│   ├── k44_tcff_io.hpp                                 # TCFFv1.0 (TIFF-compatible) plan serializer & loader
│   ├── k44_wave_compiler.hpp                           # 5-Pass Wave Graph Compiler (K44WaveGraphCompiler)
│   ├── k44_wave_executor.cu                            # Unified ROCm/CUDA GPU executor & C-ABI frontend
│   ├── wavec.py                                        # Thin ctypes Python bindings & model loaders
│   └── examples/
│       ├── example_yolov8.py                           # End-to-end 2D CNN (YOLOv8) verification & NCNN loader
│       └── example_llm.py                              # End-to-end 1D Transformer/LLM verification & SafeTensors loader
└── whitepaper/
    └── sparse_wmma_kernel/                             # Engine 1: Immutable Clustered Connectome (K44)
        ├── Makefile                                    # Standalone whitepaper build harness
        ├── white_paper_readme.md                       # Detailed engine profile & reproduction guide
        ├── main.pdf                                    # Compiled whitepaper manuscript (with all figures)
        ├── kernels/
        │   ├── kernel44_immutable.cuh                  # Primary FP16 Sparse WMMA kernel
        │   └── kernel44_dual_buffer.cuh                # Auxiliary diagnostic kernel (4 GiB MUBUF boundary)
        ├── tests/
        │   ├── benchmark_common.cuh                    # Deterministic PRNG data generator & ring-buffer pool
        │   ├── benchmark_whitepaper_k44.cpp            # Cin & cluster sparsity scaling benchmark
        │   ├── benchmark_batch_scaling.cpp             # Batch dimension scaling benchmark (M = 16 .. 16384)
        │   └── error_collapse_test.cpp                 # 3-stage numerical FP16 vs. FP64 verification
        └── scripts/
            ├── check_duplicates.py                     # Automated code-reviewability & duplicate linter
            ├── plot_scaling_all_clusters_loglog.py     # Generates Figure 3 (All-clusters scaling)
            ├── plot_scaling_m128_40m_loglog.py         # Generates Figure 4 (40M scaling & 4 GiB boundary)
            ├── plot_batch_scaling_loglog.py            # Generates Figure 5 (Batch scaling)
            ├── plot_distributions.py                   # Generates Figure 6 (Input/weight distributions)
            └── verify_whitepaper_error_collapse.py     # Standalone Python 3-stage error collapse verifier
```

---

## 4. Publications & Copyright

- **Whitepaper:** Raoul-Amadeus Lorbeer, *"Full VRAM Sparse Neural Network: Overcoming the Memory Wall"* ([`whitepaper/sparse_wmma_kernel/main.pdf`](whitepaper/sparse_wmma_kernel/main.pdf)).
- *(if I manage to get published in arXiv, citation details will be added upon publication).*

Copyright compare LICENSE.md.
