# Kernel 44 Wave Compiler (`wavec`)

`wavec` translates standard neural network architectures (2D spatial Vision CNNs such as **YOLOv8** and 1D token Transformers/LLMs such as **Llama-3 / SmolLM2**) into **Kernel 44 Sparse Connectome Wave Plans (`.tcff` v1.0)** and executes them on a unified FP16 GPU substrate (`d_Canvas`) across:
- **AMD RDNA4 (`gfx1200`)** via `kernels/rocm/kernel44_rocm_pack.cuh` (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`)
- **NVIDIA Ada / Blackwell (`sm_89` / `sm_100a`)** via `kernels/cuda/kernel44_cuda_pack.cuh` (`mma.sync.aligned.m16n8k16`)
- **Shared HIP / CUDA Auxiliary Kernels** via `kernels/common/kernel44_valu_pack.cuh` (`MaxPool`, `Upsample2x`, `S2D_2x2`, `Scatter_UpConv2x2`, `RMSNorm`, `SwiGLU`, `RoPE`, `GQA Causal Attention`)

> **Zero Repository Weights Policy:** This repository contains **0 bytes** of pre-trained weights or biases. All example scripts either construct an architectural graph on the fly (to verify compilation, `.tcff` round-trip serialization, and bit-exact GPU vs. CPU FP32 accuracy) or load user-supplied model files from disk.

---

## 1. Building `libk44wavec.so`

```bash
# Auto-detects AMD ROCm (hipcc) or NVIDIA CUDA (nvcc):
./wavec/build_wavec.sh

# Or explicitly specify backend & architecture:
BACKEND=hip  OFFLOAD_ARCH=gfx1200 ./wavec/build_wavec.sh
BACKEND=cuda CUDA_ARCH=sm_89      ./wavec/build_wavec.sh
```

---

## 2. Architecture & Canonical Kernel Families

| Kernel Family | Target Operations | Hardware Path |
| :--- | :--- | :--- |
| **`k44` Token-Shift (`ROCm` / `CUDA` `smlS` & `lrgS`)** | $1\times 1$ Pointwise Convolutions, Linear Projections (`Q, K, V, O, Gate, Up, Down`), $3\times 3$ Spatial & $2\times 2$ Stride-2 Convolutions | 32-Bit 2-Byte Token-Shift Gather (`uint32_t cluster_indices_32bit[num_clusters * B]`) + Cooperative LDS/SMEM B-Tile Broadcast + Hardware Matrix Cores (WMMA / MMA) without `im2col` duplication |
| **`k44_valu` & 2D Spatial Aux** | Non-Matrix Operations (`MaxPool`, `Upsample2x`, `S2D_2x2`, `Scatter_UpConv2x2`, `Clear_2D_Halo`, `RMSNorm`, `SwiGLU`, `RoPE`, `GQA Causal Attention`) | Coalesced Vector-ALU / Shared-Memory Tiling directly on `d_Canvas` (`kernels/common/kernel44_valu_pack.cuh`) |

### Modular Header & Executor Layout (`wavec/` & `kernels/`)
- **`kernels/common/kernel44_valu_pack.cuh`**: Shared HIP/CUDA 2D spatial (`MaxPool`, `Upsample2x`, `S2D_2x2`, `Scatter_UpConv2x2`, `Clear_2D_Halo`) and 1D Transformer VALU kernels (`RMSNorm`, `SwiGLU`, `RoPE`, `GQA Causal Attention`).
- **`kernels/rocm/kernel44_rocm_pack.cuh`**: Canonical RDNA4 (`gfx1200`) pure-FP16 WMMA token-shift kernel pack (`connectome_k44_token_shift_f16_kernel`).
- **`kernels/cuda/kernel44_cuda_pack.cuh`**: Canonical NVIDIA Ada/Blackwell (`sm_89` / `sm_100a`) pure-FP16 Tensor Core kernel pack (`k44_cuda_smlS_f16_kernel`, `k44_cuda_lrgS_f16_kernel`, `k44_cuda_tcgen05_f3_kernel`).
- **`wavec/k44_plan_types.hpp`**: Hardware profiles (`K44HardwareProfile`), 1D/2D resolution descriptors (`K44ResDesc`), wave commands (`K44ExecCmd`), and WMMA/CUDA MMA weight layout bijections (`convert_raw_to_wmma`, `convert_wmma_to_raw`, `convert_wmma_to_cuda_mma`).
- **`wavec/k44_tcff_io.hpp`**: 256-byte aligned TIFF-compatible `TCFFv1.0` binary serializer and loader (`save_plan_to_tcff`, `load_weights_from_tcff`) with 64-bit FNV-1a topology verification.
- **`wavec/k44_wave_compiler.hpp`**: 5-Pass Wave Graph Compiler (`K44WaveGraphCompiler`).
- **`wavec/k44_wave_executor.cu`**: Unified ROCm/CUDA GPU substrate manager (`K44GpuExecutorState`) and C-ABI frontend (`wavec_*`).

### The 5-Pass Wave Compiler (`k44_wave_compiler.hpp`)
1. **Pass 1 (Shape & Topology Propagation — `pass1_infer_shapes`):** Tracks 2D (`[C, H, W]`) and 1D (`[D, 1, SeqLen]`) tensor shapes across the graph.
2. **Pass 2 (Compile-Time RMSNorm Gamma Folding — `pass2_fold_rmsnorm`):** Absorbs `RMSNorm` scale vectors $\gamma_i$ directly into downstream synaptic weights ($W_{\text{new}}[j, i] = W[j, i] \cdot \gamma_i$) before FP16 quantization.
3. **Pass 3 (Resolution Descriptors — `pass3_build_layouts`):** Computes padded halo strides (`H_pad`, `W_pad`), tile offset tables (`d_tile_off_X`, `d_tile_off_Y`), and optimal cluster sizes $S \in \{16, 32, 64, 128\}$.
4. **Pass 4 (`d_Canvas` Liveness & In-Place Allocator — `pass4_allocate_canvas`):** Packs all intermediate activations into a single contiguous FP16 substrate (`d_Canvas`) using best-fit interval coalescing.
5. **Pass 5 (Zero-Copy Virtual Addressing & Hardware Bijection — `pass5_emit_commands`):** Eliminates `Concat`, `Split`, `Slice`, and `Flatten` copies via compile-time channel offsets (`tensor_channel_offsets`), folds stride-2 $2\times 2$ Space-to-Depth (`emit_conv_waves`) and transposed convolutions (`emit_deconv_waves`), and interleaves weights into the exact 16-lane hardware Matrix Core layout (`pack_wave_buffers`).

---

## 3. Quickstart Examples

### A. YOLOv8 (2D Spatial Vision Backbone + C2f + SPPF + Detect)
Run the standalone verification & loader script:
```bash
python3 wavec/examples/example_yolov8.py
```
To test with your own exported YOLOv8 NCNN model (`yolov8n.ncnn.param` + `yolov8n.ncnn.bin`):
```bash
# 1. Export YOLOv8n from Ultralytics outside this repo:
#    yolo export model=yolov8n.pt format=ncnn imgsz=640
# 2. Compile & run on Kernel 44:
python3 wavec/examples/example_yolov8.py --ncnn-param /path/to/model.ncnn.param --ncnn-bin /path/to/model.ncnn.bin
```

### B. LLM / Transformer (`RMSNorm` + `RoPE` + `GQA Causal Attention` + `SwiGLU`)
Run the standalone Transformer block verification (compares GPU `k44` + `k44_valu` output against a pure NumPy FP32 reference):
```bash
python3 wavec/examples/example_llm.py
```
To load and benchmark a full HuggingFace SafeTensors model directory (e.g., `SmolLM2-135M` or `Llama-3.2-1B` containing `config.json` and `model.safetensors`):
```bash
python3 wavec/examples/example_llm.py --safetensors-dir /path/to/hf_model_dir --seq-len 64
```
