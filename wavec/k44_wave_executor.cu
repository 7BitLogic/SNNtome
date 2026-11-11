#include "k44_wave_compiler.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#include <chrono>

#ifdef __HIPCC__
#include "../kernels/rocm/kernel44_rocm_pack.cuh"
#ifndef CHECK_HIP
#define CHECK_HIP(call)                                                                            \
    do {                                                                                           \
        hipError_t err_ = (call);                                                                  \
        if (err_ != hipSuccess) {                                                                  \
            std::fprintf(stderr, "HIP Error %s at %s:%d\n", hipGetErrorString(err_), __FILE__,     \
                         __LINE__);                                                                \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)
#endif
#define K44_LAUNCH_KERNEL(kernel, grid, block, smem, stream, ...)                                  \
    hipLaunchKernelGGL(kernel, grid, block, smem, stream, __VA_ARGS__)

#elif defined(__CUDACC__)
#include "../kernels/cuda/kernel44_cuda_pack.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using hipStream_t = cudaStream_t;
using hipEvent_t = cudaEvent_t;
using hipError_t = cudaError_t;

#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipHostMalloc cudaMallocHost
#define hipHostFree cudaFreeHost
#define hipMemset cudaMemset
#define hipMemsetAsync cudaMemsetAsync
#define hipMemcpy cudaMemcpy
#define hipMemcpyAsync cudaMemcpyAsync
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamNonBlocking cudaStreamNonBlocking
#define hipStreamSynchronize cudaStreamSynchronize
#define hipStreamDestroy cudaStreamDestroy
#define hipStreamWaitEvent cudaStreamWaitEvent
#define hipEventCreate cudaEventCreate
#define hipEventDestroy cudaEventDestroy
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventElapsedTime cudaEventElapsedTime

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t _err = (call);                                                                 \
        if (_err != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "CUDA Error at %s:%d -> %s\n", __FILE__, __LINE__,                \
                         cudaGetErrorString(_err));                                                \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)
#define CHECK_HIP(call) CHECK_CUDA(call)
#define K44_LAUNCH_KERNEL(kernel, grid, block, smem, stream, ...)                                  \
    kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)
#endif

// =================================================================================================
// 1. Device Buffer Wrappers (K44ResDesc, K44ExecCmd, K44WavePlan)
// =================================================================================================

namespace k44 {

using K44CmdType = k44_host::K44CmdType;

struct K44ResDesc : public k44_host::K44ResDesc {
    uint32_t *d_tile_off_X = nullptr;
    uint32_t *d_tile_off_Y = nullptr;

    void upload_gpu() {
        if (d_tile_off_X) {
            CHECK_HIP(hipFree(d_tile_off_X));
        }
        if (d_tile_off_Y) {
            CHECK_HIP(hipFree(d_tile_off_Y));
        }
        CHECK_HIP(hipMalloc(&d_tile_off_X, num_tiles * sizeof(uint32_t)));
        CHECK_HIP(hipMalloc(&d_tile_off_Y, num_tiles * sizeof(uint32_t)));
        CHECK_HIP(hipMemcpy(d_tile_off_X, h_tile_off_X.data(), num_tiles * sizeof(uint32_t),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_tile_off_Y, h_tile_off_Y.data(), num_tiles * sizeof(uint32_t),
                            hipMemcpyHostToDevice));
    }

    void free_gpu() {
        if (d_tile_off_X) {
            CHECK_HIP(hipFree(d_tile_off_X));
            d_tile_off_X = nullptr;
        }
        if (d_tile_off_Y) {
            CHECK_HIP(hipFree(d_tile_off_Y));
            d_tile_off_Y = nullptr;
        }
    }
};

struct K44ExecCmd : public k44_host::K44ExecCmd {
    uint4 *d_w_wmma = nullptr;
    uint32_t *d_idx = nullptr;
    __half *d_bias = nullptr;
    uint16_t *d_cout_map = nullptr;

    void free_gpu() {
        if (d_w_wmma) {
            CHECK_HIP(hipFree(d_w_wmma));
            d_w_wmma = nullptr;
        }
        if (d_idx) {
            CHECK_HIP(hipFree(d_idx));
            d_idx = nullptr;
        }
        if (d_bias) {
            CHECK_HIP(hipFree(d_bias));
            d_bias = nullptr;
        }
        if (d_cout_map) {
            CHECK_HIP(hipFree(d_cout_map));
            d_cout_map = nullptr;
        }
    }
};

struct K44WavePlan : public k44_host::K44WavePlan {
    std::map<uint32_t, K44ResDesc> res_map;
    std::vector<K44ExecCmd> cmds;
    __half *d_Canvas = nullptr;

    void allocate_substrate() {
        if (d_Canvas) {
            CHECK_HIP(hipFree(d_Canvas));
        }
        CHECK_HIP(hipMalloc(&d_Canvas, canvas_elems * sizeof(__half)));
        CHECK_HIP(hipMemset(d_Canvas, 0, canvas_elems * sizeof(__half)));
    }

    void upload_input_f32(const std::vector<float> &h_in) {
        const auto &in_layout = res_map.at(input_res_key);
        std::vector<__half> h_pad((size_t)input_c * in_layout.channel_stride, __float2half(0.0f));
        for (uint32_t ch = 0; ch < input_c; ++ch) {
            for (uint32_t y = 0; y < input_h; ++y) {
                for (uint32_t x = 0; x < input_w; ++x) {
                    h_pad[in_layout.canvas_offset(ch, y, x)] =
                        __float2half(h_in[((size_t)ch * input_h + y) * input_w + x]);
                }
            }
        }
        CHECK_HIP(hipMemcpy(d_Canvas + input_off, h_pad.data(), h_pad.size() * sizeof(__half),
                            hipMemcpyHostToDevice));
    }

    std::vector<float> download_output_f32() const {
        const auto &out_layout = res_map.at(output_res_key);
        std::vector<__half> h_pad((size_t)output_c * out_layout.channel_stride);
        CHECK_HIP(hipMemcpy(h_pad.data(), d_Canvas + output_off, h_pad.size() * sizeof(__half),
                            hipMemcpyDeviceToHost));
        std::vector<float> out((size_t)output_c * output_h * output_w);
        for (uint32_t ch = 0; ch < output_c; ++ch) {
            for (uint32_t y = 0; y < output_h; ++y) {
                for (uint32_t x = 0; x < output_w; ++x) {
                    out[((size_t)ch * output_h + y) * output_w + x] =
                        __half2float(h_pad[out_layout.canvas_offset(ch, y, x)]);
                }
            }
        }
        return out;
    }
};

} // namespace k44

// =================================================================================================
// 2. Wave Command Dispatcher (launch_k44_wave)
// =================================================================================================

template <int CLUSTER_SIZE>
static inline uint32_t compute_mall_chunk_tiles(const k44::K44ExecCmd &cmd,
                                                const k44::K44ResDesc &res_layout,
                                                const k44_host::K44HardwareProfile &hw_prof) {
    (void)cmd;
    (void)hw_prof;
    return res_layout.num_tiles;
}

// Dispatches a single `K44_WAVE` command (`cmd`) across 64-element spatial/token tiles on
// `d_Canvas` for a compile-time cluster size `CLUSTER_SIZE` (`S` in `{16, 32, 64, 128}` output
// neurons per cluster).
// Parameters & Variables:
// - `cmd`: Device command descriptor with output offset `y_off`, residual offset `res_off`,
//   padded output channels `Cout`, synaptic fan-in in 16-element K-steps `B_vec16` (`B / 16`),
//   and device pointers `d_idx` (`[Cout / S, B]` gather indices), `d_w_wmma` (WMMA weights),
//   `d_bias` (`[Cout]` FP16 biases), and `d_cout_map` (optional output channel map).
// - `d_Canvas`: Base pointer to the unified FP16 GPU activation arena.
// - `x_base_off`: Base offset (in FP16 elements) added to `d_Canvas` to form `X_ptr`.
// - `res_layout`: Output `K44ResDesc` providing `num_tiles` (total 64-element tiles),
//   `channel_stride` (FP16 elements between consecutive output channels), `needs_mask` (`true`
//   when `W % 64 != 0` requires the 16-bit lane write mask in `d_tile_off_Y`), and device tile
//   offset tables `d_tile_off_X` and `d_tile_off_Y`.
// - `hw_prof`: Target `K44HardwareProfile`.
// - `stream`: HIP/CUDA execution stream.
// - `use_smlS`: On CUDA (`__CUDACC__`), selects the direct register-gather (`smlS`) kernel path
//   when `true` (1D token sequences or 64-frame image batches).
template <int CLUSTER_SIZE>
static inline void launch_k44_wave(const k44::K44ExecCmd &cmd, __half *d_Canvas,
                                   uint32_t x_base_off, const k44::K44ResDesc &res_layout,
                                   const k44_host::K44HardwareProfile &hw_prof,
                                   hipStream_t stream = nullptr, bool use_smlS = false) {
    __half *Y_ptr = d_Canvas + cmd.y_off;
    const __half *X_ptr = d_Canvas + x_base_off;
    const __half *R_ptr = d_Canvas + cmd.res_off;
    const uint32_t chunk_tiles = compute_mall_chunk_tiles<CLUSTER_SIZE>(cmd, res_layout, hw_prof);
    for (uint32_t tile_start = 0u; tile_start < res_layout.num_tiles; tile_start += chunk_tiles) {
        const uint32_t cur_tiles = std::min(chunk_tiles, res_layout.num_tiles - tile_start);
        const uint32_t *d_off_X = res_layout.d_tile_off_X + tile_start;
        const uint32_t *d_off_Y = res_layout.d_tile_off_Y + tile_start;
#ifdef __CUDACC__
        if (use_smlS) {
            if (res_layout.needs_mask) {
                K44_DISPATCH_SMLS_EPILOGUE_EX(CLUSTER_SIZE, true, cmd.has_bias, cmd.act,
                                              cmd.has_residual, cmd.has_cout_map, Y_ptr, X_ptr,
                                              cmd.d_idx, cmd.d_w_wmma, cmd.d_bias, d_off_X, d_off_Y,
                                              cur_tiles, cmd.Cout, res_layout.channel_stride,
                                              cmd.B_vec16, stream, R_ptr, cmd.d_cout_map);
            } else {
                K44_DISPATCH_SMLS_EPILOGUE_EX(CLUSTER_SIZE, false, cmd.has_bias, cmd.act,
                                              cmd.has_residual, cmd.has_cout_map, Y_ptr, X_ptr,
                                              cmd.d_idx, cmd.d_w_wmma, cmd.d_bias, d_off_X, d_off_Y,
                                              cur_tiles, cmd.Cout, res_layout.channel_stride,
                                              cmd.B_vec16, stream, R_ptr, cmd.d_cout_map);
            }
            continue;
        }
#else
        (void)use_smlS;
#endif
        if (res_layout.needs_mask) {
            K44_DISPATCH_EPILOGUE_EX(
                CLUSTER_SIZE, true, cmd.has_bias, cmd.act, cmd.has_residual, cmd.has_cout_map,
                Y_ptr, X_ptr, cmd.d_idx, cmd.d_w_wmma, cmd.d_bias, d_off_X, d_off_Y, cur_tiles,
                cmd.Cout, res_layout.channel_stride, cmd.B_vec16, stream, R_ptr, cmd.d_cout_map);
        } else {
            K44_DISPATCH_EPILOGUE_EX(
                CLUSTER_SIZE, false, cmd.has_bias, cmd.act, cmd.has_residual, cmd.has_cout_map,
                Y_ptr, X_ptr, cmd.d_idx, cmd.d_w_wmma, cmd.d_bias, d_off_X, d_off_Y, cur_tiles,
                cmd.Cout, res_layout.channel_stride, cmd.B_vec16, stream, R_ptr, cmd.d_cout_map);
        }
    }
}

// =================================================================================================
// 3. Persistent GPU Executor State (K44GpuExecutorState)
// =================================================================================================

namespace k44_host {

struct K44GpuExecutorState {
    k44::K44WavePlan gpu_plan;
    K44HardwareProfile hw_profile = K44HardwareProfile::from_preset("auto");
    uint32_t batch_size = 1;
    uint32_t wave_batch_size = 1;
    uint32_t llm_seq_len_per_req = 0;
    bool is_image_token64 = false;
    bool reuse_canvas = false;
    std::map<uint32_t, uint32_t> res_single_stride;
    std::vector<bool> cmd_clear_halo;
    std::vector<uint32_t> cmd_x_base_off;
    hipEvent_t ev_start = nullptr;
    hipEvent_t ev_stop = nullptr;
    bool initialized = false;

    ~K44GpuExecutorState() {
        free_all();
    }

    static void upload_cmd_to_gpu(const K44ExecCmd &host_cmd, k44::K44ExecCmd &gpu_cmd) {
        static_cast<k44_host::K44ExecCmd &>(gpu_cmd) = host_cmd;
        gpu_cmd.h_w_wmma.clear();
        gpu_cmd.h_idx.clear();
        gpu_cmd.h_bias.clear();
        gpu_cmd.h_cout_map.clear();

        if (!host_cmd.h_w_wmma.empty()) {
            size_t wb = host_cmd.h_w_wmma.size() * sizeof(uint4);
            CHECK_HIP(hipMalloc(&gpu_cmd.d_w_wmma, wb));
#ifdef __CUDACC__
            std::vector<uint4_host> h_w_cuda(host_cmd.h_w_wmma.size());
            convert_wmma_to_cuda_mma(h_w_cuda.data(), host_cmd.h_w_wmma.data(), host_cmd.Cout,
                                     host_cmd.B / 8u);
            CHECK_HIP(hipMemcpy(gpu_cmd.d_w_wmma, h_w_cuda.data(), wb, hipMemcpyHostToDevice));
#else
            CHECK_HIP(
                hipMemcpy(gpu_cmd.d_w_wmma, host_cmd.h_w_wmma.data(), wb, hipMemcpyHostToDevice));
#endif
        }
        if (!host_cmd.h_idx.empty()) {
            size_t ib = host_cmd.h_idx.size() * sizeof(uint32_t);
            CHECK_HIP(hipMalloc(&gpu_cmd.d_idx, ib));
            CHECK_HIP(hipMemcpy(gpu_cmd.d_idx, host_cmd.h_idx.data(), ib, hipMemcpyHostToDevice));
        }
        if (!host_cmd.h_bias.empty()) {
            size_t bb = host_cmd.h_bias.size() * sizeof(__half);
            CHECK_HIP(hipMalloc(&gpu_cmd.d_bias, bb));
            CHECK_HIP(hipMemcpy(gpu_cmd.d_bias, host_cmd.h_bias.data(), bb, hipMemcpyHostToDevice));
        }
        if (!host_cmd.h_cout_map.empty()) {
            size_t cb = host_cmd.h_cout_map.size() * sizeof(uint16_t);
            CHECK_HIP(hipMalloc(&gpu_cmd.d_cout_map, cb));
            CHECK_HIP(hipMemcpy(gpu_cmd.d_cout_map, host_cmd.h_cout_map.data(), cb,
                                hipMemcpyHostToDevice));
        }
    }

    void free_all() {
        for (auto &kv : gpu_plan.res_map) {
            kv.second.free_gpu();
        }
        for (auto &cmd : gpu_plan.cmds) {
            cmd.free_gpu();
        }
        if (gpu_plan.d_Canvas) {
            (void)hipFree(gpu_plan.d_Canvas);
            gpu_plan.d_Canvas = nullptr;
        }
        if (ev_start) {
            (void)hipEventDestroy(ev_start);
            ev_start = nullptr;
        }
        if (ev_stop) {
            (void)hipEventDestroy(ev_stop);
            ev_stop = nullptr;
        }
        gpu_plan.res_map.clear();
        gpu_plan.cmds.clear();
        res_single_stride.clear();
        cmd_clear_halo.clear();
        cmd_x_base_off.clear();
        initialized = false;
    }

    bool init(const K44WavePlan &host_plan) {
        free_all();
        hw_profile = host_plan.hw_profile;
        batch_size = std::max(1u, host_plan.batch_size);
        wave_batch_size = std::max(1u, host_plan.wave_batch_size);
        is_image_token64 = host_plan.is_image_token64;
        reuse_canvas = host_plan.reuse_canvas;

        static_cast<k44_host::K44WavePlan &>(gpu_plan) = host_plan;
        gpu_plan.output_c = host_plan.logical_output_c;
        gpu_plan.res_map.clear();
        gpu_plan.cmds.clear();

        for (const auto &kv : host_plan.res_map) {
            auto &gpu_layout = gpu_plan.res_map[kv.first];
            const auto &host_layout = kv.second;
            static_cast<k44_host::K44ResDesc &>(gpu_layout) = host_layout;
            gpu_layout.upload_gpu();
            res_single_stride[kv.first] = host_layout.single_stride;
        }

        gpu_plan.cmds.reserve(host_plan.cmds.size());
        cmd_clear_halo.reserve(host_plan.cmds.size());
        cmd_x_base_off.reserve(host_plan.cmds.size());
        for (const auto &host_cmd : host_plan.cmds) {
            cmd_clear_halo.push_back(host_cmd.needs_halo_clear);
            cmd_x_base_off.push_back(host_cmd.x_base_off);
            k44::K44ExecCmd gpu_cmd{};
            upload_cmd_to_gpu(host_cmd, gpu_cmd);
            gpu_plan.cmds.push_back(std::move(gpu_cmd));
        }

        gpu_plan.allocate_substrate();
        CHECK_HIP(hipEventCreate(&ev_start));
        CHECK_HIP(hipEventCreate(&ev_stop));
        initialized = true;
        return true;
    }

    // Zeros out the 2D padding halo of a destination region (`d_Canvas + cmd.y_off`) when Pass 4
    // reuses an arena interval that was previously written by another tensor, ensuring subsequent
    // 3x3 convolutions read exact `0.0f` boundary padding.
    // Variables (in 64-bit `uint64_t` / 4-FP16 quad units `>> 2`):
    // - `s_out`: Single-frame channel plane stride in FP16 elements (`single_stride`).
    // - `prefix_q` (`(W_pad + 4) / 4`): Top halo row (`W_pad` elements) plus first row's 4-element
    //   left halo, measured in 4-FP16 quads.
    // - `gap_q` (`(W_pad - W) / 4`): Inter-row right/left halo gap between row `y` and row `y + 1`.
    // - `mid_q` (`H * gap_q`): Total inter-row halo quads across all `H` rows.
    // - `halo_q` (`(s_out - H * W) / 4`): Total halo quads per single-frame channel plane.
    // - `tot_ch` (`cmd.Cout * wave_batch_size`): Total channel planes across the batch.
    // - `total_q` (`tot_ch * halo_q`): Total 64-bit halo words cleared by
    // `k44_clear_2d_halo_f16_kernel`.
    void clear_2d_halo(const k44::K44ExecCmd &cmd, hipStream_t stream) const {
        const auto &out_layout = gpu_plan.res_map.at(cmd.res_key_out);
        uint32_t s_out = res_single_stride.at(cmd.res_key_out);
        uint32_t prefix_q = (out_layout.W_pad + 4u) >> 2u;
        uint32_t gap_q = (out_layout.W_pad - out_layout.W) >> 2u;
        uint32_t mid_q = out_layout.H * gap_q;
        uint32_t halo_q = (s_out - out_layout.H * out_layout.W) >> 2u;
        uint32_t tot_ch = cmd.Cout * wave_batch_size;
        uint32_t total_q = tot_ch * halo_q;
        K44_LAUNCH_KERNEL(k44_clear_2d_halo_f16_kernel, dim3((total_q + 255u) / 256u, 1u, 1u),
                          dim3(256u), 0, stream, gpu_plan.d_Canvas + cmd.y_off, out_layout.H,
                          out_layout.W, out_layout.W_pad, s_out, prefix_q, gap_q, mid_q, halo_q,
                          total_q);
    }

    void dispatch_wave_by_cluster(const k44::K44ExecCmd &cmd, uint32_t xb,
                                  hipStream_t stream) const {
        const auto &out_layout = gpu_plan.res_map.at(cmd.res_key_out);
        const bool use_smlS = is_image_token64 || out_layout.is_1d_tokens;
        if (cmd.cluster_size == 128u) {
            launch_k44_wave<128>(cmd, gpu_plan.d_Canvas, xb, out_layout, hw_profile, stream,
                                 use_smlS);
        } else if (cmd.cluster_size == 64u) {
            launch_k44_wave<64>(cmd, gpu_plan.d_Canvas, xb, out_layout, hw_profile, stream,
                                use_smlS);
        } else if (cmd.cluster_size == 32u) {
            launch_k44_wave<32>(cmd, gpu_plan.d_Canvas, xb, out_layout, hw_profile, stream,
                                use_smlS);
        } else {
            launch_k44_wave<16>(cmd, gpu_plan.d_Canvas, xb, out_layout, hw_profile, stream,
                                use_smlS);
        }
    }

    template <int HEAD_DIM>
    void launch_rope_and_gqa(const k44::K44ExecCmd &cmd, uint32_t channel_stride, uint32_t num_t32,
                             uint32_t seq_req_eff, float neg_log_theta, uint32_t gqa_group,
                             float scale, hipStream_t stream) const {
        if (cmd.llama3_rope_scaling) {
            K44_LAUNCH_KERNEL((k44_valu_rope_tile32_1d_f16_kernel<HEAD_DIM, true>),
                              dim3(num_t32, cmd.h_q, 1u), dim3(32u, 8u, 1u), 0, stream,
                              gpu_plan.d_Canvas + cmd.src_off, channel_stride, seq_req_eff,
                              neg_log_theta);
            K44_LAUNCH_KERNEL((k44_valu_rope_tile32_1d_f16_kernel<HEAD_DIM, true>),
                              dim3(num_t32, cmd.h_kv, 1u), dim3(32u, 8u, 1u), 0, stream,
                              gpu_plan.d_Canvas + cmd.src2_off, channel_stride, seq_req_eff,
                              neg_log_theta);
        } else {
            K44_LAUNCH_KERNEL((k44_valu_rope_tile32_1d_f16_kernel<HEAD_DIM, false>),
                              dim3(num_t32, cmd.h_q, 1u), dim3(32u, 8u, 1u), 0, stream,
                              gpu_plan.d_Canvas + cmd.src_off, channel_stride, seq_req_eff,
                              neg_log_theta);
            K44_LAUNCH_KERNEL((k44_valu_rope_tile32_1d_f16_kernel<HEAD_DIM, false>),
                              dim3(num_t32, cmd.h_kv, 1u), dim3(32u, 8u, 1u), 0, stream,
                              gpu_plan.d_Canvas + cmd.src2_off, channel_stride, seq_req_eff,
                              neg_log_theta);
        }
        K44_LAUNCH_KERNEL((k44_valu_gqa_causal_attn_tile32_1d_f16_kernel<HEAD_DIM>),
                          dim3(num_t32, cmd.h_q, 1u), dim3(256u, 1u, 1u), 0, stream,
                          gpu_plan.d_Canvas + cmd.y_off, gpu_plan.d_Canvas + cmd.src_off,
                          gpu_plan.d_Canvas + cmd.src2_off, gpu_plan.d_Canvas + cmd.src3_off,
                          channel_stride, gqa_group, scale, seq_req_eff);
    }

    // Executes the `i`-th compiled command (`gpu_plan.cmds[i]`) on `stream` over
    // `gpu_plan.d_Canvas`, dispatching either a Matrix Core connectome wave (`K44_WAVE`) or an
    // auxiliary 1D/2D VALU kernel from `kernels/common/kernel44_valu_pack.cuh` (`S2D_2X2`,
    // `MAXPOOL_2X2`, `MAXPOOL_5X5`, `UPSAMPLE_2X`, `SCATTER_UP2X2`, `RMSNORM_1D`, `SWIGLU_MUL_1D`,
    // or `ROPE_GQA_ATTN_1D`).
    void execute_cmd(size_t i, hipStream_t stream = nullptr) const {
        const auto &cmd = gpu_plan.cmds[i];
        if (cmd_clear_halo[i]) {
            clear_2d_halo(cmd, stream);
        }
        switch (static_cast<K44CmdType>(static_cast<int>(cmd.type))) {
        case K44CmdType::K44_WAVE: {
            dispatch_wave_by_cluster(cmd, cmd_x_base_off[i], stream);
            break;
        }
        case K44CmdType::S2D_2X2: {
            const auto &in_layout = gpu_plan.res_map.at(cmd.res_key_in);
            const auto &out_layout = gpu_plan.res_map.at(cmd.res_key_out);
            uint32_t s_in = res_single_stride.at(cmd.res_key_in);
            uint32_t s_out = res_single_stride.at(cmd.res_key_out);
            uint32_t tot_1 = cmd.Cin * cmd.h_out * cmd.w_out;
            K44_LAUNCH_KERNEL(k44_s2d_2x2_batched_padded_f16_kernel,
                              dim3((tot_1 + 255u) / 256u, wave_batch_size, 1u), dim3(256u), 0,
                              stream, gpu_plan.d_Canvas + cmd.y_off,
                              gpu_plan.d_Canvas + cmd.src_off, cmd.Cin, cmd.h_out, cmd.w_out,
                              in_layout.W_pad, in_layout.channel_stride, s_in, out_layout.W_pad,
                              out_layout.channel_stride, s_out);
            break;
        }
        case K44CmdType::MAXPOOL_2X2: {
            const auto &in_layout = gpu_plan.res_map.at(cmd.res_key_in);
            const auto &out_layout = gpu_plan.res_map.at(cmd.res_key_out);
            uint32_t s_in = res_single_stride.at(cmd.res_key_in);
            uint32_t s_out = res_single_stride.at(cmd.res_key_out);
            uint32_t shift = (uint32_t)__builtin_ctz(cmd.w_out);
            uint32_t mask = cmd.w_out - 1u;
            uint32_t tot = (cmd.Cin * wave_batch_size) * cmd.h_out * cmd.w_out;
            K44_LAUNCH_KERNEL(k44::k44_maxpool2x2_padded_f16_kernel, dim3(tot / 256u), dim3(256u),
                              0, stream, gpu_plan.d_Canvas + cmd.y_off,
                              gpu_plan.d_Canvas + cmd.src_off, shift, mask, in_layout.W_pad, s_in,
                              out_layout.W_pad, s_out);
            break;
        }
        case K44CmdType::MAXPOOL_5X5: {
            const auto &out_layout = gpu_plan.res_map.at(cmd.res_key_out);
            uint32_t s_out = res_single_stride.at(cmd.res_key_out);
            uint32_t tot = (cmd.Cin * wave_batch_size) * cmd.h_out * cmd.w_out;
            K44_LAUNCH_KERNEL(k44::k44_maxpool5x5_s1_p2_padded_f16_kernel, dim3(tot / 256u),
                              dim3(256u), 0, stream, gpu_plan.d_Canvas + cmd.y_off,
                              gpu_plan.d_Canvas + cmd.src_off, cmd.h_out, cmd.w_out,
                              out_layout.W_pad, s_out);
            break;
        }
        case K44CmdType::UPSAMPLE_2X: {
            const auto &in_layout = gpu_plan.res_map.at(cmd.res_key_in);
            const auto &out_layout = gpu_plan.res_map.at(cmd.res_key_out);
            uint32_t s_in = res_single_stride.at(cmd.res_key_in);
            uint32_t s_out = res_single_stride.at(cmd.res_key_out);
            uint32_t tot = (cmd.Cin * wave_batch_size) * cmd.h_in * cmd.w_in;
            K44_LAUNCH_KERNEL(k44::k44_upsample2x_nearest_padded_f16_kernel, dim3(tot / 256u),
                              dim3(256u), 0, stream, gpu_plan.d_Canvas + cmd.y_off,
                              gpu_plan.d_Canvas + cmd.src_off, cmd.h_in, cmd.w_in, in_layout.W_pad,
                              s_in, out_layout.W_pad, s_out);
            break;
        }
        case K44CmdType::SCATTER_UP2X2: {
            const auto &in_layout = gpu_plan.res_map.at(cmd.res_key_in);
            const auto &out_layout = gpu_plan.res_map.at(cmd.res_key_out);
            uint32_t s_in = res_single_stride.at(cmd.res_key_in);
            uint32_t s_out = res_single_stride.at(cmd.res_key_out);
            uint32_t tot_1 = cmd.Cout * cmd.h_out * cmd.w_out;
            uint32_t shift = (uint32_t)__builtin_ctz(cmd.w_out);
            uint32_t mask = cmd.w_out - 1u;
            K44_LAUNCH_KERNEL(k44_scatter_upconv2x2_batched_padded_f16_kernel,
                              dim3(tot_1 / 256u, wave_batch_size, 1u), dim3(256u), 0, stream,
                              gpu_plan.d_Canvas + cmd.y_off, gpu_plan.d_Canvas + cmd.src_off,
                              cmd.Cout, shift, mask, in_layout.W_pad, in_layout.channel_stride,
                              s_in, out_layout.W_pad, out_layout.channel_stride, s_out);
            break;
        }
        case K44CmdType::RMSNORM_1D: {
            const auto &out_layout = gpu_plan.res_map.at(cmd.res_key_out);
            float inv_D = 1.0f / (float)cmd.Cin;
            uint32_t num_tiles64 = (cmd.w_out + 63u) / 64u;
            if (cmd.gamma_folded) {
                K44_LAUNCH_KERNEL(k44_valu_rmsnorm_tile64_f16_kernel<false>, dim3(num_tiles64),
                                  dim3(256u), 0, stream, gpu_plan.d_Canvas + cmd.y_off,
                                  gpu_plan.d_Canvas + cmd.src_off, nullptr, cmd.Cin,
                                  out_layout.channel_stride, inv_D, cmd.eps);
            } else {
                K44_LAUNCH_KERNEL(k44_valu_rmsnorm_tile64_f16_kernel<true>, dim3(num_tiles64),
                                  dim3(256u), 0, stream, gpu_plan.d_Canvas + cmd.y_off,
                                  gpu_plan.d_Canvas + cmd.src_off, cmd.d_bias, cmd.Cin,
                                  out_layout.channel_stride, inv_D, cmd.eps);
            }
            break;
        }
        case K44CmdType::SWIGLU_MUL_1D: {
            const auto &out_layout = gpu_plan.res_map.at(cmd.res_key_out);
            uint32_t tot = cmd.Cout * out_layout.channel_stride;
            K44_LAUNCH_KERNEL(k44_valu_swiglu_tile64_f16_kernel, dim3(tot / 1024u), dim3(128u), 0,
                              stream, gpu_plan.d_Canvas + cmd.y_off,
                              gpu_plan.d_Canvas + cmd.src_off, gpu_plan.d_Canvas + cmd.src2_off);
            break;
        }
        case K44CmdType::ROPE_GQA_ATTN_1D: {
            const auto &out_layout = gpu_plan.res_map.at(cmd.res_key_out);
            float scale = 1.0f / std::sqrt((float)cmd.head_dim);
            float neg_log_theta = -std::log(cmd.rope_theta);
            uint32_t gqa_group = cmd.h_q / cmd.h_kv;
            uint32_t seq_req = (llm_seq_len_per_req > 0u) ? llm_seq_len_per_req : cmd.w_out;
            uint32_t num_t32 = (cmd.w_out + 31u) / 32u;
            uint32_t seq_req_eff =
                (seq_req >= 32u) ? ((seq_req + 31u) & ~31u) : std::max(1u, seq_req);
            if (cmd.head_dim == 128u) {
                launch_rope_and_gqa<128>(cmd, out_layout.channel_stride, num_t32, seq_req_eff,
                                         neg_log_theta, gqa_group, scale, stream);
            } else {
                launch_rope_and_gqa<64>(cmd, out_layout.channel_stride, num_t32, seq_req_eff,
                                        neg_log_theta, gqa_group, scale, stream);
            }
            break;
        }
        default:
            break;
        }
    }

    void execute_plan(hipStream_t stream = nullptr) const {
        for (size_t i = 0; i < gpu_plan.cmds.size(); ++i) {
            execute_cmd(i, stream);
        }
    }

    int infer_f32(const float *h_in, int in_elems, float *h_out, int out_elems, float *out_gpu_ms) {
        if (!initialized) {
            return -1;
        }
        std::vector<float> in_vec(h_in, h_in + in_elems);
        gpu_plan.upload_input_f32(in_vec);
        CHECK_HIP(hipEventRecord(ev_start, nullptr));
        execute_plan(nullptr);
        CHECK_HIP(hipEventRecord(ev_stop, nullptr));
        CHECK_HIP(hipEventSynchronize(ev_stop));
        float ms = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&ms, ev_start, ev_stop));
        if (out_gpu_ms) {
            *out_gpu_ms = ms;
        }
        std::vector<float> out_vec = gpu_plan.download_output_f32();
        size_t copy_n = std::min((size_t)out_elems, out_vec.size());
        std::memcpy(h_out, out_vec.data(), copy_n * sizeof(float));
        return 0;
    }
};

static std::map<void *, K44GpuExecutorState *> g_gpu_executors;

} // namespace k44_host
#endif

// =================================================================================================
// 4. C-ABI Frontend (extern "C")
// =================================================================================================

static inline k44_host::K44WaveGraphCompiler *get_compiler(void *handle) {
    return static_cast<k44_host::K44WaveGraphCompiler *>(handle);
}

static inline k44_host::K44WaveGraphCompiler *get_compiled(void *handle) {
    auto *comp = get_compiler(handle);
    if (comp && !comp->is_compiled) {
        comp->compile();
    }
    return comp;
}

extern "C" {

void *wavec_create(int max_B_per_wave, int default_S, int kv_cache_max_seq,
                   int fold_rmsnorm_gamma) {
    auto *comp = new k44_host::K44WaveGraphCompiler();
    comp->opts.max_B_per_wave = (uint32_t)std::max(0, max_B_per_wave);
    comp->opts.force_cluster_size = (uint32_t)std::max(0, default_S);
    comp->opts.kv_cache_max_seq = (uint32_t)std::max(0, kv_cache_max_seq);
    comp->opts.fold_rmsnorm_gamma = (fold_rmsnorm_gamma != 0);
    comp->opts.keep_host_plan_buffers = true;
    return static_cast<void *>(comp);
}

void wavec_destroy(void *handle) {
    if (!handle) {
        return;
    }
#if defined(__HIPCC__) || defined(__CUDACC__)
    auto it = k44_host::g_gpu_executors.find(handle);
    if (it != k44_host::g_gpu_executors.end()) {
        delete it->second;
        k44_host::g_gpu_executors.erase(it);
    }
#endif
    delete get_compiler(handle);
}

int wavec_gpu_release(void *handle) {
    if (!handle) {
        return -1;
    }
#if defined(__HIPCC__) || defined(__CUDACC__)
    auto it = k44_host::g_gpu_executors.find(handle);
    if (it != k44_host::g_gpu_executors.end()) {
        delete it->second;
        k44_host::g_gpu_executors.erase(it);
    }
    return 0;
#else
    return -1;
#endif
}

int wavec_add_input(void *handle, const char *name, int c, int h, int w, int is_1d) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->add_input(name, (uint32_t)c, (uint32_t)h, (uint32_t)w, is_1d != 0);
    return 0;
}

int wavec_add_flatten(void *handle, const char *name, const char *in_tensor,
                      const char *out_tensor) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->add_flatten(name, in_tensor, out_tensor);
    return 0;
}

int wavec_add_conv2d(void *handle, const char *name, const char *in_tensor, const char *out_tensor,
                     int cin, int cout, int kh, int kw, int stride, int pad, int act,
                     const float *w_f32, const float *bias_f32, const char *residual_tensor) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->add_conv2d(name, in_tensor, out_tensor, (uint32_t)cin, (uint32_t)cout, (uint32_t)kh,
                     (uint32_t)kw, (uint32_t)stride, (uint32_t)pad,
                     static_cast<k44_host::K44Act>(act), w_f32, bias_f32,
                     residual_tensor ? std::string(residual_tensor) : std::string(""));
    return 0;
}

int wavec_add_conv2d_f16(void *handle, const char *name, const char *in_tensor,
                         const char *out_tensor, int cin, int cout, int kh, int kw, int stride,
                         int pad, int act, const uint16_t *w_f16, const float *bias_f32,
                         const char *residual_tensor) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->add_conv2d_f16(name, in_tensor, out_tensor, (uint32_t)cin, (uint32_t)cout, (uint32_t)kh,
                         (uint32_t)kw, (uint32_t)stride, (uint32_t)pad,
                         static_cast<k44_host::K44Act>(act), w_f16, bias_f32,
                         residual_tensor ? std::string(residual_tensor) : std::string(""));
    return 0;
}

int wavec_add_linear_f32(void *handle, const char *name, const char *in_tensor,
                         const char *out_tensor, int din, int dout, int act, const float *w_f32,
                         const float *bias_f32, const char *residual_tensor) {
    return wavec_add_conv2d(handle, name, in_tensor, out_tensor, din, dout, 1, 1, 1, 0, act, w_f32,
                            bias_f32, residual_tensor);
}

int wavec_add_linear_f16(void *handle, const char *name, const char *in_tensor,
                         const char *out_tensor, int din, int dout, int act, const uint16_t *w_f16,
                         const float *bias_f32, const char *residual_tensor) {
    return wavec_add_conv2d_f16(handle, name, in_tensor, out_tensor, din, dout, 1, 1, 1, 0, act,
                                w_f16, bias_f32, residual_tensor);
}

int wavec_add_linear_bf16(void *handle, const char *name, const char *in_tensor,
                          const char *out_tensor, int din, int dout, int act,
                          const uint16_t *w_bf16, const float *bias_f32,
                          const char *residual_tensor) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->add_linear_bf16(name, in_tensor, out_tensor, (uint32_t)din, (uint32_t)dout,
                          static_cast<k44_host::K44Act>(act), w_bf16, bias_f32,
                          residual_tensor ? std::string(residual_tensor) : std::string(""));
    return 0;
}

int wavec_add_upconv2x2(void *handle, const char *name, const char *in_tensor,
                        const char *out_tensor, int cin, int cout, const float *w_f32,
                        const float *bias_f32) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->add_upconv2x2(name, in_tensor, out_tensor, (uint32_t)cin, (uint32_t)cout, w_f32,
                        bias_f32);
    return 0;
}

int wavec_add_maxpool(void *handle, const char *name, const char *in_tensor, const char *out_tensor,
                      int k, int stride, int pad) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->add_maxpool(name, in_tensor, out_tensor, (uint32_t)k, (uint32_t)stride, (uint32_t)pad);
    return 0;
}

int wavec_add_upsample2x(void *handle, const char *name, const char *in_tensor,
                         const char *out_tensor) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->add_upsample2x(name, in_tensor, out_tensor);
    return 0;
}

int wavec_add_rmsnorm(void *handle, const char *name, const char *in_tensor, const char *out_tensor,
                      int dim, float eps, const float *gamma_f32, int pre_folded) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->add_rmsnorm(name, in_tensor, out_tensor, (uint32_t)dim, eps, gamma_f32, pre_folded != 0);
    return 0;
}

int wavec_add_swiglu(void *handle, const char *name, const char *gate_tensor, const char *up_tensor,
                     const char *out_tensor) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->add_swiglu_mul(name, gate_tensor, up_tensor, out_tensor);
    return 0;
}

int wavec_add_rope_gqa_attn(void *handle, const char *name, const char *q_tensor,
                            const char *k_tensor, const char *v_tensor, const char *out_tensor,
                            int h_q, int h_kv, int head_dim, float rope_theta, int llama3_scaling) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->add_rope_gqa_attention(name, q_tensor, k_tensor, v_tensor, out_tensor, (uint32_t)h_q,
                                 (uint32_t)h_kv, (uint32_t)head_dim, rope_theta,
                                 llama3_scaling != 0);
    return 0;
}

int wavec_add_split(void *handle, const char *name, const char *in_tensor, int num_outs,
                    const char **out_tensors_arr) {
    auto *comp = get_compiler(handle);
    if (!comp || num_outs <= 0) {
        return -1;
    }
    std::vector<std::string> out_tensors;
    out_tensors.reserve(num_outs);
    for (int i = 0; i < num_outs; ++i) {
        out_tensors.emplace_back(out_tensors_arr[i]);
    }
    comp->add_split(name, in_tensor, out_tensors);
    return 0;
}

int wavec_add_slice(void *handle, const char *name, const char *in_tensor, int num_outs,
                    const char **out_tensors_arr, const int *slice_sizes) {
    auto *comp = get_compiler(handle);
    if (!comp || num_outs <= 0) {
        return -1;
    }
    std::vector<std::string> out_tensors;
    std::vector<uint32_t> sizes;
    out_tensors.reserve(num_outs);
    for (int i = 0; i < num_outs; ++i) {
        out_tensors.emplace_back(out_tensors_arr[i]);
        if (slice_sizes) {
            sizes.push_back((uint32_t)slice_sizes[i]);
        }
    }
    comp->add_slice(name, in_tensor, out_tensors, sizes);
    return 0;
}

int wavec_add_concat(void *handle, const char *name, int num_ins, const char **in_tensors_arr,
                     const char *out_tensor) {
    auto *comp = get_compiler(handle);
    if (!comp || num_ins <= 0) {
        return -1;
    }
    std::vector<std::string> in_tensors;
    in_tensors.reserve(num_ins);
    for (int i = 0; i < num_ins; ++i) {
        in_tensors.emplace_back(in_tensors_arr[i]);
    }
    comp->add_concat(name, in_tensors, out_tensor);
    return 0;
}

int wavec_set_output(void *handle, const char *out_tensor) {
    auto *comp = get_compiler(handle);
    if (!comp || !out_tensor) {
        return -1;
    }
    comp->default_output_tensor = out_tensor;
    return 0;
}

int wavec_set_reuse_canvas(void *handle, int reuse_canvas) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->opts.reuse_canvas = (reuse_canvas != 0);
    comp->is_compiled = false;
    return 0;
}

int wavec_set_batch_size(void *handle, int batch_size) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->opts.batch_size = (uint32_t)std::max(1, batch_size);
    comp->is_compiled = false;
    return 0;
}

int wavec_set_target_preset(void *handle, const char *preset_name) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->opts.hw_profile = k44_host::K44HardwareProfile::from_preset(preset_name);
    comp->is_compiled = false;
    return 0;
}

int wavec_compile(void *handle) {
    auto *comp = get_compiler(handle);
    if (!comp) {
        return -1;
    }
    comp->compile();
    return 0;
}

const char *wavec_get_summary_json(void *handle) {
    auto *comp = get_compiled(handle);
    if (!comp) {
        return "{}";
    }
    return comp->cached_summary_json.c_str();
}

int wavec_save_tcff_ex(void *handle, const char *tcff_path, const char *model_prefix,
                       int include_canvas, int include_geometry) {
    auto *comp = get_compiled(handle);
    if (!comp || !tcff_path) {
        return -1;
    }
    (void)include_canvas;
    return k44_host::save_plan_to_tcff(comp->compiled_plan, tcff_path, model_prefix, false,
                                       include_geometry != 0, nullptr)
               ? 0
               : -1;
}

int wavec_save_tcff(void *handle, const char *tcff_path, const char *model_prefix) {
    return wavec_save_tcff_ex(handle, tcff_path, model_prefix, 1, 1);
}

int wavec_load_tcff_weights(void *handle, const char *tcff_path) {
    auto *comp = get_compiled(handle);
    if (!comp || !tcff_path) {
        return -1;
    }
    return k44_host::load_weights_from_tcff(comp->compiled_plan, tcff_path) ? 0 : -1;
}

int wavec_gpu_init_plan(void *handle) {
    auto *comp = get_compiled(handle);
    if (!comp) {
        return -1;
    }
#if defined(__HIPCC__) || defined(__CUDACC__)
    auto *&exec = k44_host::g_gpu_executors[handle];
    if (!exec) {
        exec = new k44_host::K44GpuExecutorState();
    }
    return exec->init(comp->compiled_plan) ? 0 : -1;
#else
    return -1;
#endif
}

int wavec_gpu_infer_f32(void *handle, const float *h_in, int in_elems, float *h_out, int out_elems,
                        float *out_gpu_ms) {
    if (!handle || !h_in || !h_out) {
        return -1;
    }
#if defined(__HIPCC__) || defined(__CUDACC__)
    auto it = k44_host::g_gpu_executors.find(handle);
    if (it == k44_host::g_gpu_executors.end() || !it->second) {
        return -1;
    }
    return it->second->infer_f32(h_in, in_elems, h_out, out_elems, out_gpu_ms);
#else
    return -1;
#endif
}

} // extern "C"
