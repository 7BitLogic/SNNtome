#pragma once
// =================================================================================================
// kernel44_valu_pack.cuh — Shared HIP & CUDA 2D Spatial & VALU Auxiliary Kernels for Kernel 44
//   - MaxPool (2x2 & 5x5 SPPF), Upsample2x, 2D Halo Clear, Space-to-Depth (S2D), Scatter UpConv2x2
//   - RMSNorm (Tile64), SwiGLU (Tile64), RoPE (Tile32), Grouped-Query Causal Attention (GQA Tile32)
//   - Status-Register Wave-Reduce-Max Epilogue (`k44_valu_status_reg_epilogue_f16_kernel`)
// =================================================================================================
#if !defined(K44_VALU_PACK_CUH)
#define K44_VALU_PACK_CUH
#include <cstdint>
__device__ __forceinline__ float k44_dev_shfl_xor(float v, int mask) {
#ifdef __CUDACC__
    return __shfl_xor_sync(0xffffffffu, v, mask);
#else
    return __shfl_xor(v, mask, 32);
#endif
} __device__ __forceinline__ float k44_dev_fmaxf(float a, float b) {
#ifdef __CUDACC__
    return fmaxf(a, b);
#else
    return __builtin_fmaxf(a, b);
#endif
} __device__ __forceinline__ float k44_dev_expf(float x) {
#ifdef __CUDACC__
    return __expf(x);
#else
    return __builtin_expf(x);
#endif
} namespace k44 {
__global__ void k44_maxpool2x2_padded_f16_kernel(__half *__restrict__ dst, const __half *__restrict__ src,
                                                 uint32_t w_out_shift, uint32_t w_out_mask, uint32_t W_pad_in, uint32_t ch_stride_in,
                                                 uint32_t W_pad_out, uint32_t ch_stride_out) {
    uint32_t tid = blockIdx.x * 256u + threadIdx.x; uint32_t x = tid & w_out_mask;
    uint32_t y = (tid >> w_out_shift) & w_out_mask; uint32_t c = tid >> (2u * w_out_shift);
    uint32_t in_r0 = c * ch_stride_in + (2u * y + 1u) * W_pad_in + 4u + 2u * x; uint32_t in_r1 = in_r0 + W_pad_in;
    float v00 = __half2float(src[in_r0]); float v01 = __half2float(src[in_r0 + 1u]);
    float v10 = __half2float(src[in_r1]); float v11 = __half2float(src[in_r1 + 1u]);
    float m = k44_dev_fmaxf(k44_dev_fmaxf(v00, v01), k44_dev_fmaxf(v10, v11));
    dst[c * ch_stride_out + (y + 1u) * W_pad_out + 4u + x] = __float2half(m); }
__global__ void k44_maxpool5x5_s1_p2_padded_f16_kernel(__half *__restrict__ dst, const __half *__restrict__ src, uint32_t H,
                                                       uint32_t W, uint32_t W_pad, uint32_t ch_stride) {
    uint32_t tid = blockIdx.x * 256u + threadIdx.x; uint32_t x = tid % W;
    uint32_t t = tid / W; uint32_t y = t % H;
    uint32_t c = t / H; const __half *ch_ptr = src + c * ch_stride;
    float max_v = -65504.0f;
#pragma unroll
    for (int dy = -2; dy <= 2; ++dy) { int yy = (int)y + dy;
        bool vy = (yy >= 0 && yy < (int)H); int cy = vy ? yy : 0;
        const __half *row_ptr = ch_ptr + (uint32_t)(cy + 1) * W_pad + 4u;
#pragma unroll
        for (int dx = -2; dx <= 2; ++dx) { int xx = (int)x + dx;
            bool v = vy && (xx >= 0 && xx < (int)W); int cx = v ? xx : 0;
            float val = v ? __half2float(row_ptr[cx]) : -65504.0f; max_v = k44_dev_fmaxf(max_v, val);
        } }
    dst[c * ch_stride + (y + 1u) * W_pad + 4u + x] = __float2half(max_v); }
__global__ void k44_upsample2x_nearest_padded_f16_kernel(
    __half *__restrict__ dst, const __half *__restrict__ src, uint32_t H_in, uint32_t W_in,
    uint32_t W_pad_in, uint32_t ch_stride_in, uint32_t W_pad_out, uint32_t ch_stride_out) {
    uint32_t tid = blockIdx.x * 256u + threadIdx.x; uint32_t x = tid % W_in;
    uint32_t t = tid / W_in; uint32_t y = t % H_in;
    uint32_t c = t / H_in; __half v = src[c * ch_stride_in + (y + 1u) * W_pad_in + 4u + x];
    uint32_t r0 = c * ch_stride_out + (2u * y + 1u) * W_pad_out + 4u + 2u * x; uint32_t r1 = r0 + W_pad_out;
    dst[r0 + 0u] = v; dst[r0 + 1u] = v;
    dst[r1 + 0u] = v; dst[r1 + 1u] = v;
} template <bool LLAMA3_SCALING>
__device__ __forceinline__ float k44_rope_inv_freq(uint32_t p, uint32_t head_dim, float neg_log_theta) {
    float inv_freq = expf((float)(2u * p) * (neg_log_theta / (float)head_dim)); if constexpr (LLAMA3_SCALING) {
        float ratio = 1303.7972938088068f * inv_freq; float smooth = fminf(1.0f, fmaxf(0.0f, (ratio - 1.0f) * 0.3333333432674408f));
        inv_freq = (1.0f - smooth) * (inv_freq * 0.03125f) + smooth * inv_freq; }
    return inv_freq; }
} // namespace k44
__global__ void k44_clear_2d_halo_f16_kernel(__half *__restrict__ dst, uint32_t H, uint32_t W, uint32_t W_pad, uint32_t single_stride, uint32_t prefix_q, uint32_t gap_q, uint32_t mid_q, uint32_t halo_q, uint32_t total_q) { uint32_t tid = blockIdx.x * 256u + threadIdx.x;
    uint32_t safe_tid = (tid < total_q) ? tid : 0u; uint32_t c = safe_tid / halo_q;
    uint32_t q = safe_tid % halo_q; bool in_prefix = (q < prefix_q);
    bool in_mid = (!in_prefix) & (q < (prefix_q + mid_q)); uint32_t mq = in_mid ? (q - prefix_q) : 0u;
    uint32_t y = mq / gap_q; uint32_t rem = mq % gap_q;
    uint32_t sq = q - (prefix_q + mid_q); uint32_t mid_off = ((y + 1u) * W_pad + 4u + W) >> 2u;
    uint32_t suf_off = ((H + 1u) * W_pad + 4u) >> 2u; uint32_t q_off = in_prefix ? q : (in_mid ? (mid_off + rem) : (suf_off + sq));
    uint64_t *ch_q = reinterpret_cast<uint64_t *>(dst + (size_t)c * single_stride); ch_q[q_off] = 0ULL;
} __global__ void k44_s2d_2x2_batched_padded_f16_kernel(
    __half *__restrict__ dst, const __half *__restrict__ src, uint32_t Cin, uint32_t H_out,
    uint32_t W_out, uint32_t W_pad_in, uint32_t ch_stride_in, uint32_t single_stride_in,
    uint32_t W_pad_out, uint32_t ch_stride_out, uint32_t single_stride_out) { uint32_t tid = blockIdx.x * 256u + threadIdx.x;
    uint32_t tot = Cin * H_out * W_out; uint32_t safe_tid = (tid < tot) ? tid : 0u;
    uint32_t b_frame = blockIdx.y; uint32_t x = safe_tid % W_out;
    uint32_t t = safe_tid / W_out; uint32_t y = t % H_out;
    uint32_t c = t / H_out; const __half *src_b = src + (size_t)b_frame * single_stride_in;
    __half *dst_b = dst + (size_t)b_frame * single_stride_out;
    size_t in_r0 = (size_t)c * ch_stride_in + (size_t)(2u * y + 1u) * W_pad_in + 4u + 2u * x; size_t in_r1 = in_r0 + W_pad_in;
    size_t out_base = (size_t)(4u * c) * ch_stride_out + (size_t)(y + 1u) * W_pad_out + 4u + x;
    dst_b[out_base + (size_t)0u * ch_stride_out] = src_b[in_r0 + 0u]; dst_b[out_base + (size_t)1u * ch_stride_out] = src_b[in_r0 + 1u];
    dst_b[out_base + (size_t)2u * ch_stride_out] = src_b[in_r1 + 0u]; dst_b[out_base + (size_t)3u * ch_stride_out] = src_b[in_r1 + 1u];
} __global__ void k44_scatter_upconv2x2_batched_padded_f16_kernel(
    __half *__restrict__ dst, const __half *__restrict__ src_quad, uint32_t Cout,
    uint32_t w_out_shift, uint32_t w_out_mask, uint32_t W_pad_in, uint32_t ch_stride_in,
    uint32_t single_stride_in, uint32_t W_pad_out, uint32_t ch_stride_out, uint32_t single_stride_out) {
    uint32_t tid = blockIdx.x * 256u + threadIdx.x; uint32_t b_frame = blockIdx.y;
    uint32_t x = tid & w_out_mask; uint32_t y = (tid >> w_out_shift) & w_out_mask;
    uint32_t c = tid >> (2u * w_out_shift); uint32_t q = ((y & 1u) << 1) | (x & 1u);
    uint32_t qc = q * Cout + c; const __half *src_b = src_quad + (size_t)b_frame * single_stride_in;
    __half *dst_b = dst + (size_t)b_frame * single_stride_out; __half val =
        src_b[(size_t)qc * ch_stride_in + (size_t)((y >> 1) + 1u) * W_pad_in + 4u + (x >> 1)];
    dst_b[(size_t)c * ch_stride_out + (size_t)(y + 1u) * W_pad_out + 4u + x] = val; }
template <bool HAS_GAMMA> __global__ __launch_bounds__(256) void k44_valu_rmsnorm_tile64_f16_kernel(
    __half *__restrict__ dst, const __half *__restrict__ src, const __half *__restrict__ gamma,
    uint32_t D, uint32_t stride, float inv_D, float eps) { __shared__ float s_warp_sq[8][8][8];
    const uint32_t tile_b = blockIdx.x; const uint32_t tid = threadIdx.x;
    const uint32_t t_sub = tid & 7u; const uint32_t c_row = tid >> 3u;
    const uint32_t t_base = tile_b * 64u + t_sub * 8u; float sum_sq[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t c = c_row; c < D; c += 32u) { const uint4 u4 = *reinterpret_cast<const uint4 *>(src + (size_t)c * stride + t_base);
        const __half *h = reinterpret_cast<const __half *>(&u4);
#pragma unroll
        for (int k = 0; k < 8; ++k) { float v = __half2float(h[k]);
            sum_sq[k] += v * v; }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) { sum_sq[k] += k44_dev_shfl_xor(sum_sq[k], 8);
        sum_sq[k] += k44_dev_shfl_xor(sum_sq[k], 16); s_warp_sq[tid >> 5u][t_sub][k] = sum_sq[k];
    } __syncthreads();
    float inv_rms[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { float tot = s_warp_sq[0][t_sub][k] + s_warp_sq[1][t_sub][k] + s_warp_sq[2][t_sub][k] +
                    s_warp_sq[3][t_sub][k] + s_warp_sq[4][t_sub][k] + s_warp_sq[5][t_sub][k] +
                    s_warp_sq[6][t_sub][k] + s_warp_sq[7][t_sub][k]; inv_rms[k] = rsqrtf(tot * inv_D + eps);
    } for (uint32_t c = c_row; c < D; c += 32u) {
        const uint4 u4 = *reinterpret_cast<const uint4 *>(src + (size_t)c * stride + t_base);
        const __half *h = reinterpret_cast<const __half *>(&u4); float g = 1.0f;
        if constexpr (HAS_GAMMA) { g = __half2float(gamma[c]);
        } uint4 out_u4;
        __half *out_h = reinterpret_cast<__half *>(&out_u4);
#pragma unroll
        for (int k = 0; k < 8; ++k) { out_h[k] = __float2half(__half2float(h[k]) * inv_rms[k] * g);
        } *reinterpret_cast<uint4 *>(dst + (size_t)c * stride + t_base) = out_u4;
    } }
__global__ __launch_bounds__(128) void k44_valu_swiglu_tile64_f16_kernel(
    __half *__restrict__ dst, const __half *__restrict__ gate_silu, const __half *__restrict__ up) {
    const uint32_t idx = blockIdx.x * 128u + threadIdx.x; const uint4 g4 = reinterpret_cast<const uint4 *>(gate_silu)[idx];
    const uint4 u4 = reinterpret_cast<const uint4 *>(up)[idx]; const __half *gh = reinterpret_cast<const __half *>(&g4);
    const __half *uh = reinterpret_cast<const __half *>(&u4); uint4 out4;
    __half *oh = reinterpret_cast<__half *>(&out4);
#pragma unroll
    for (int k = 0; k < 8; ++k) { oh[k] = __float2half(__half2float(gh[k]) * __half2float(uh[k]));
    } reinterpret_cast<uint4 *>(dst)[idx] = out4;
} template <uint32_t HEAD_DIM, bool LLAMA3_SCALING = false>
__global__ __launch_bounds__(256) void k44_valu_rope_tile32_1d_f16_kernel(
    __half *__restrict__ qk, uint32_t stride, uint32_t seq_len_per_req, float neg_log_theta = -9.21034037f) {
    constexpr uint32_t HALF_DIM = HEAD_DIM >> 1u; constexpr uint32_t PAIRS_PER_WARP = HALF_DIM / 8u;
    const uint32_t tile_t = blockIdx.x; const uint32_t h = blockIdx.y;
    const uint32_t l = threadIdx.x; const uint32_t p_sub = threadIdx.y;
    const uint32_t t = tile_t * 32u + l; const uint32_t pos = t % seq_len_per_req;
#pragma unroll
    for (uint32_t u = 0; u < PAIRS_PER_WARP; ++u) { const uint32_t p = p_sub + u * 8u;
        float freq = k44::k44_rope_inv_freq<LLAMA3_SCALING>(p, HEAD_DIM, neg_log_theta); float angle = (float)pos * freq;
        float cos_a = cosf(angle); float sin_a = sinf(angle);
        size_t idx0 = (size_t)(h * HEAD_DIM + p) * stride + t; size_t idx1 = (size_t)(h * HEAD_DIM + p + HALF_DIM) * stride + t;
        float x0 = __half2float(qk[idx0]); float x1 = __half2float(qk[idx1]);
        qk[idx0] = __float2half(x0 * cos_a - x1 * sin_a); qk[idx1] = __float2half(x0 * sin_a + x1 * cos_a);
    } }
template <uint32_t HEAD_DIM> __global__ __launch_bounds__(256) void k44_valu_gqa_causal_attn_tile32_1d_f16_kernel(
    __half *__restrict__ out, const __half *__restrict__ q, const __half *__restrict__ k,
    const __half *__restrict__ v, uint32_t stride, uint32_t gqa_group, float scale, uint32_t seq_len_per_req) {
    constexpr uint32_t U4_PER_ROW = HEAD_DIM / 8u; constexpr uint32_t U4_PER_THREAD = HEAD_DIM / 64u;
    __shared__ uint4 s_K_u4[32][U4_PER_ROW + 1u]; __shared__ uint4 s_V_u4[32][U4_PER_ROW + 1u];
    const uint32_t tile_t = blockIdx.x; const uint32_t h_q = blockIdx.y;
    const uint32_t h_kv = h_q / gqa_group; const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid >> 5u; const uint32_t lane = tid & 31u;
    const uint32_t q_in_w = lane >> 3u; const uint32_t d_lane = lane & 7u;
    const uint32_t q_local = (warp_id << 2u) + q_in_w; const uint32_t t = tile_t * 32u + q_local;
    const uint32_t req_start_tile = ((tile_t * 32u) / seq_len_per_req) * (seq_len_per_req >> 5u);
    const __half *q_head = q + (size_t)(h_q * HEAD_DIM) * stride + tile_t * 32u + lane;
#pragma unroll
    for (uint32_t d = warp_id; d < HEAD_DIM; d += 8u) { reinterpret_cast<__half *>(s_K_u4[lane])[d] = q_head[(size_t)d * stride];
    } __syncthreads();
    float q_reg[U4_PER_THREAD][8]; float acc[U4_PER_THREAD][8];
#pragma unroll
    for (uint32_t uv = 0; uv < U4_PER_THREAD; ++uv) { const uint4 qu4 = s_K_u4[q_local][uv * 8u + d_lane];
        const __half *qh = reinterpret_cast<const __half *>(&qu4);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            q_reg[uv][j] = __half2float(qh[j]) * scale;
            acc[uv][j] = 0.0f;
        }
    }
    __syncthreads();
    float m_prev = -65504.0f;
    float l_prev = 0.0f;
    const __half *k_head = k + (size_t)(h_kv * HEAD_DIM) * stride;
    const __half *v_head = v + (size_t)(h_kv * HEAD_DIM) * stride;
    for (uint32_t tau_tile = req_start_tile; tau_tile <= tile_t; ++tau_tile) {
        const uint32_t tau_col = tau_tile * 32u + lane;
#pragma unroll
        for (uint32_t d = warp_id; d < HEAD_DIM; d += 8u) {
            reinterpret_cast<__half *>(s_K_u4[lane])[d] = k_head[(size_t)d * stride + tau_col];
            reinterpret_cast<__half *>(s_V_u4[lane])[d] = v_head[(size_t)d * stride + tau_col];
        }
        __syncthreads();
        for (uint32_t u = 0; u < 32u; ++u) {
            const uint32_t tau = tau_tile * 32u + u;
            const bool valid = (tau <= t);
            float dot = 0.0f;
#pragma unroll
            for (uint32_t uv = 0; uv < U4_PER_THREAD; ++uv) {
                const uint4 ku4 = s_K_u4[u][uv * 8u + d_lane];
                const __half *kh = reinterpret_cast<const __half *>(&ku4);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    dot += q_reg[uv][j] * __half2float(kh[j]);
                }
            }
            dot += k44_dev_shfl_xor(dot, 1);
            dot += k44_dev_shfl_xor(dot, 2);
            dot += k44_dev_shfl_xor(dot, 4);
            float score = valid ? dot : -65504.0f;
            float m_new = k44_dev_fmaxf(m_prev, score);
            float exp_prev = valid ? k44_dev_expf(m_prev - m_new) : 1.0f;
            float exp_curr = valid ? k44_dev_expf(score - m_new) : 0.0f;
            l_prev = l_prev * exp_prev + exp_curr;
#pragma unroll
            for (uint32_t uv = 0; uv < U4_PER_THREAD; ++uv) {
                const uint4 vu4 = s_V_u4[u][uv * 8u + d_lane];
                const __half *vh = reinterpret_cast<const __half *>(&vu4);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[uv][j] = acc[uv][j] * exp_prev + exp_curr * __half2float(vh[j]);
                }
            }
            m_prev = m_new;
        }
        __syncthreads();
    }
    float inv_l = 1.0f / l_prev;
#pragma unroll
    for (uint32_t uv = 0; uv < U4_PER_THREAD; ++uv) {
        uint4 out_u4;
        __half *oh = reinterpret_cast<__half *>(&out_u4);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            oh[j] = __float2half(acc[uv][j] * inv_l);
        }
        s_K_u4[q_local][uv * 8u + d_lane] = out_u4;
    }
    __syncthreads();
    __half *out_head = out + (size_t)(h_q * HEAD_DIM) * stride + tile_t * 32u + lane;
#pragma unroll
    for (uint32_t d = warp_id; d < HEAD_DIM; d += 8u) {
        out_head[(size_t)d * stride] = reinterpret_cast<const __half *>(s_K_u4[lane])[d];
    }
}
// Rule 11 Branch-Free Status Register Epilogue Kernel (Zero `if` / `while`)
// Reduces active tokens `m in [0, valid_tokens)` (up to 64 tokens per first tile)
// for each control neuron `j = blockIdx.x` and writes into `d_status_regs[reg_offset + j]`.
__global__ __launch_bounds__(32) void k44_valu_status_reg_epilogue_f16_kernel(
    float *__restrict__ d_status_regs, const __half *__restrict__ d_Y, uint32_t channel_stride,
    uint32_t first_tile_y_off, uint32_t valid_tokens, uint32_t reg_offset) {
    const uint32_t j = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t last_m = valid_tokens - 1u;
    const uint32_t m0 = min(lane, last_m);
    const uint32_t m1 = min(lane + 32u, last_m);
    const __half *row_ptr = d_Y + (size_t)j * channel_stride + first_tile_y_off;
    float v0 = __half2float(row_ptr[m0]);
    float v1 = __half2float(row_ptr[m1]);
    float vmax = k44_dev_fmaxf(v0, v1);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        vmax = k44_dev_fmaxf(vmax, k44_dev_shfl_xor(vmax, mask));
    }
    d_status_regs[reg_offset + j] = vmax;
}
#endif // K44_VALU_PACK_CUH
