#pragma once
// ==============================================================================
// kernel44_cuda_pack.cuh — Kernel 44 FP16 CUDA Pack (Ultra-Sparse Connectome Engine)
// Targets: NVIDIA Ada Lovelace (sm_89), Hopper (sm_90) & Blackwell (sm_100a)
//
// Canonical Architecture (Vorspann -> Loop -> Abspann / `setup` -> `half_shader` -> `finish`):
//   - K44_CUDA_d2 / smlS: Direct Register Gather + `prmt.b32` + SMEM Broadcast
//   - K44_CUDA_lrgS:      Async DMA Bypass (`cp.async`) + `ldmatrix.m8n8.x4.trans`
//   - K44_CUDA_f3:        Blackwell 5th-Gen Tensor Core (`tcgen05.mma` + TMEM)
//
// Strict Compliance:
//   - Rule 11: ZERO runtime `if` or `while` branches inside GPU compute kernels.
// ==============================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
#if !defined(K44_CUDA_ACT_DEFINED)
#define K44_CUDA_ACT_DEFINED
enum class K44Act : int { NONE = 0, RELU = 1, SIGMOID = 2, SILU = 3, LEAKY_RELU = 4 };
#endif
// ==============================================================================
// 1. PTX Inline Assembly, Hardware Intrinsics & Shared WaveC Epilogue
// ==============================================================================
__device__ __forceinline__ uint32_t k44_smem_u32_addr(const void* smem_ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)); }
__device__ __forceinline__ uint32_t k44_pack_f16x2(float lo, float hi) {
    __half2 h2 = __halves2half2(__float2half_rn(lo), __float2half_rn(hi)); return *reinterpret_cast<const uint32_t*>(&h2);
} __device__ __forceinline__ void k44_mma_m16n8k16_f32_f16(
    float& d0, float& d1, float& d2, float& d3, const uint4& a, uint32_t b0, uint32_t b1 ) { asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1) );
} __device__ __forceinline__ void k44_compute_mma_subtile_16x16x16( const uint4& w, const uint4& b, float (&c_sub)[8] ) {
    k44_mma_m16n8k16_f32_f16(c_sub[0], c_sub[1], c_sub[2], c_sub[3], w, b.x, b.y);
    k44_mma_m16n8k16_f32_f16(c_sub[4], c_sub[5], c_sub[6], c_sub[7], w, b.z, b.w); }
__device__ __forceinline__ void k44_compute_mma_4x_f16( const uint4& w,
    const uint4& b0, const uint4& b1, const uint4& b2, const uint4& b3, float (&c)[4][8]
) { k44_compute_mma_subtile_16x16x16(w, b0, c[0]);
    k44_compute_mma_subtile_16x16x16(w, b1, c[1]); k44_compute_mma_subtile_16x16x16(w, b2, c[2]);
    k44_compute_mma_subtile_16x16x16(w, b3, c[3]); }
__device__ __forceinline__ void k44_ldmatrix_m8n8_x4_trans_b16(uint4& dst, uint32_t smem_addr) { asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w)
        : "r"(smem_addr) ); } __device__ __forceinline__ void k44_cp_async_cg_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile( "cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr) : "memory" );
} __device__ __forceinline__ void k44_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void k44_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void k44_store_global_v4_u32(void* ptr, const uint4& val, bool /*streaming*/) { asm volatile(
        "st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};\n" :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory" ); }
__device__ __forceinline__ void k44_store_global_cs_v4_u32(void* ptr, const uint4& val) { asm volatile(
        "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};\n" :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory" ); }
__device__ __forceinline__ uint4 k44_pack_b_frag_prmt(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) { return make_uint4(
        __byte_perm(v0, v1, 0x5410u), __byte_perm(v2, v3, 0x5410u), __byte_perm(v0, v1, 0x7632u), __byte_perm(v2, v3, 0x7632u)
    ); } template <K44Act ACT> __device__ __forceinline__ float k44_apply_act_cuda(float v) {
    if constexpr (ACT == K44Act::RELU) return fmaxf(v, 0.0f);
    else if constexpr (ACT == K44Act::SIGMOID) return 1.0f / (1.0f + __expf(-v));
    else if constexpr (ACT == K44Act::SILU) return v / (1.0f + __expf(-v));
    else if constexpr (ACT == K44Act::LEAKY_RELU) return fmaxf(v, 0.1f * v); else return v;
} __device__ __forceinline__ void k44_fence_mma_acc(float (&c)[4][8]) { asm volatile( ""
        : "+f"(c[0][0]), "+f"(c[1][0]), "+f"(c[2][0]), "+f"(c[3][0]), "+f"(c[0][4]), "+f"(c[1][4]), "+f"(c[2][4]), "+f"(c[3][4]) ); }
__device__ __forceinline__ uint4 k44_transpose_4x4_quad( uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t tid_in_group
) { const uint32_t b0 = tid_in_group & 1u;
    uint32_t s01 = b0 ? x0 : x1, k01 = b0 ? x1 : x0; uint32_t s23 = b0 ? x2 : x3, k23 = b0 ? x3 : x2;
    s01 = __shfl_xor_sync(0xffffffffu, s01, 1); s23 = __shfl_xor_sync(0xffffffffu, s23, 1);
    x0 = b0 ? s01 : k01; x1 = b0 ? k01 : s01; x2 = b0 ? s23 : k23; x3 = b0 ? k23 : s23;
    const uint32_t b1 = tid_in_group & 2u; uint32_t s02 = b1 ? x0 : x2, k02 = b1 ? x2 : x0;
    uint32_t s13 = b1 ? x1 : x3, k13 = b1 ? x3 : x1; s02 = __shfl_xor_sync(0xffffffffu, s02, 2);
    s13 = __shfl_xor_sync(0xffffffffu, s13, 2); x0 = b1 ? s02 : k02; x2 = b1 ? k02 : s02;
    x1 = b1 ? s13 : k13; x3 = b1 ? k13 : s13; return make_uint4(x0, x1, x2, x3); } template <
    bool USE_LDMATRIX_TRANS, int VALID_TOKENS, bool STREAMING_STORE, bool PER_TILE_MASK,
    bool HAS_RESIDUAL, K44Act EFFECTIVE_ACT, bool HAS_COUT_MAP >
__device__ __forceinline__ void k44_finish_wavec_epilogue( __half* base_Y_tile, const __half* base_R_tile, uint32_t j_base0,
    uint32_t tid_in_group, uint32_t valid_lanes, const float (&c)[4][8], uint32_t channel_stride_Y,
    const uint16_t* cout_map ) {
    #pragma unroll
    for (int nh = 0; nh < 2; ++nh) { const uint32_t logical_j = j_base0 + nh * 8u;
        uint32_t out_j = logical_j; if constexpr (HAS_COUT_MAP) { out_j = static_cast<uint32_t>(__ldg(cout_map + logical_j)); }
        const int c_idx_even = (nh == 0) ? 0 : 2; const int c_idx_odd  = (nh == 0) ? 4 : 6;
        #pragma unroll
        for (int sh = 0; sh < 2; ++sh) { const size_t token_base = (size_t)(tid_in_group * 16u + sh * 8u);
            uint4 y_u4; if constexpr (!USE_LDMATRIX_TRANS) { const int c_e = c_idx_even + sh; const int c_o = c_idx_odd  + sh;
                float v[8];
                #pragma unroll
                for (int q = 0; q < 4; ++q) { v[2 * q + 0] = c[q][c_e]; v[2 * q + 1] = c[q][c_o]; }
                float r_val[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; if constexpr (HAS_RESIDUAL) {
                    const uint4 r_u4 = __ldg(reinterpret_cast<const uint4*>(
                        base_R_tile + (size_t)out_j * channel_stride_Y + token_base ));
                    const __half* r_h = reinterpret_cast<const __half*>(&r_u4);
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) { r_val[t] = __half2float(r_h[t]); if constexpr (HAS_COUT_MAP) v[t] += r_val[t]; }
                }
                #pragma unroll
                for (int t = 0; t < 8; ++t) { v[t] = k44_apply_act_cuda<EFFECTIVE_ACT>(v[t]);
                    if constexpr (HAS_RESIDUAL && !HAS_COUT_MAP) v[t] += r_val[t]; } y_u4 = make_uint4( k44_pack_f16x2(v[0], v[1]),
                    k44_pack_f16x2(v[2], v[3]), k44_pack_f16x2(v[4], v[5]), k44_pack_f16x2(v[6], v[7]) );
            } else { const int cb = (sh == 0) ? c_idx_even : c_idx_odd; if constexpr (!HAS_RESIDUAL) { uint32_t pk[4];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) { float v0 = k44_apply_act_cuda<EFFECTIVE_ACT>(c[q][cb + 0]);
                        float v1 = k44_apply_act_cuda<EFFECTIVE_ACT>(c[q][cb + 1]); pk[q] = k44_pack_f16x2(v0, v1);
                    } y_u4 = k44_transpose_4x4_quad(pk[0], pk[1], pk[2], pk[3], tid_in_group);
                } else { uint4 f_even = k44_transpose_4x4_quad( __float_as_uint(c[0][cb + 0]), __float_as_uint(c[1][cb + 0]),
                        __float_as_uint(c[2][cb + 0]), __float_as_uint(c[3][cb + 0]), tid_in_group );
                    uint4 f_odd = k44_transpose_4x4_quad( __float_as_uint(c[0][cb + 1]),
                        __float_as_uint(c[1][cb + 1]), __float_as_uint(c[2][cb + 1]), __float_as_uint(c[3][cb + 1]), tid_in_group
                    ); float v[8] = {
                        __uint_as_float(f_even.x), __uint_as_float(f_odd.x), __uint_as_float(f_even.y), __uint_as_float(f_odd.y),
                        __uint_as_float(f_even.z), __uint_as_float(f_odd.z), __uint_as_float(f_even.w), __uint_as_float(f_odd.w)
                    }; const uint4 r_u4 = __ldg(reinterpret_cast<const uint4*>(
                        base_R_tile + (size_t)out_j * channel_stride_Y + token_base ));
                    const __half* r_h = reinterpret_cast<const __half*>(&r_u4);
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) { float rv = __half2float(r_h[t]);
                        if constexpr (HAS_COUT_MAP) v[t] += rv; v[t] = k44_apply_act_cuda<EFFECTIVE_ACT>(v[t]);
                        if constexpr (!HAS_COUT_MAP) v[t] += rv; } y_u4 = make_uint4( k44_pack_f16x2(v[0], v[1]),
                        k44_pack_f16x2(v[2], v[3]), k44_pack_f16x2(v[4], v[5]), k44_pack_f16x2(v[6], v[7]) );
                } } if constexpr (PER_TILE_MASK) { const uint32_t b_lo = tid_in_group * 4u + sh * 2u + 0u;
                const uint32_t b_hi = tid_in_group * 4u + sh * 2u + 1u; const uint32_t m_lo = 0u - ((valid_lanes >> b_lo) & 1u);
                const uint32_t m_hi = 0u - ((valid_lanes >> b_hi) & 1u); y_u4.x &= m_lo; y_u4.y &= m_lo;
                y_u4.z &= m_hi; y_u4.w &= m_hi; } else if constexpr (VALID_TOKENS < 64) {
                const uint32_t b_lo = tid_in_group * 4u + sh * 2u + 0u; const uint32_t b_hi = tid_in_group * 4u + sh * 2u + 1u;
                const uint32_t m_lo = 0u - static_cast<uint32_t>(b_lo < (uint32_t)(VALID_TOKENS >> 2));
                const uint32_t m_hi = 0u - static_cast<uint32_t>(b_hi < (uint32_t)(VALID_TOKENS >> 2)); y_u4.x &= m_lo; y_u4.y &= m_lo;
                y_u4.z &= m_hi; y_u4.w &= m_hi; }
            void* dst_ptr = base_Y_tile + (size_t)out_j * channel_stride_Y + token_base; if constexpr (STREAMING_STORE) {
                k44_store_global_cs_v4_u32(dst_ptr, y_u4); } else { k44_store_global_v4_u32(dst_ptr, y_u4, false); }
        } } }
// ==============================================================================
// 2. Kernel `d2` & `smlS` (K44CudaDTokenShiftPipeline & K44CudaSmlSPipeline):
//    2-Byte Token-Shift Addressing & Vectorized Aligned Gather
//    Structured into Vorspann (`setup`) -> Loop (`half_shader`) -> Abspann (`finish`)
// ==============================================================================
template < int CLUSTER_SIZE, int WARPS_PER_BLOCK, bool FLAT_BATCH_TILES = false,
    bool ALIGNED_GATHER = false, bool VEC_GATHER_S128 = false > struct K44CudaDTokenShiftPipeline {
    static constexpr int WARPS_PER_CLUSTER = (CLUSTER_SIZE / 16 < WARPS_PER_BLOCK) ? (CLUSTER_SIZE / 16) : WARPS_PER_BLOCK;
    static constexpr int CLUSTERS_PER_BLOCK = WARPS_PER_BLOCK / WARPS_PER_CLUSTER;
    static constexpr int K_CHUNK = (CLUSTER_SIZE >= 32) ? 2 : 1;
    static constexpr int GATHER_SUBTILES = (4 * K_CHUNK) / WARPS_PER_CLUSTER; static constexpr bool USE_SMEM = (WARPS_PER_CLUSTER > 1);
    static constexpr bool USE_VEC_GATHER = ALIGNED_GATHER && (VEC_GATHER_S128 ? (GATHER_SUBTILES >= 1) : (GATHER_SUBTILES >= 2));
    using WLoadElem = uint4; using GatherElem = typename std::conditional<GATHER_SUBTILES == 4, uint4,
                       typename std::conditional<GATHER_SUBTILES == 2, uint2, uint32_t>::type>::type;
    using SmemArray = uint4[2][USE_SMEM ? CLUSTERS_PER_BLOCK : 1][K_CHUNK][4][32];
    uint32_t lane, group_id, tid_in_group, k_sub, sub_tile, cluster_in_block, j_base0, valid_lanes; const uint2* idx_ptr;
    const WLoadElem* w_ptr; const uint16_t* base_X_u16; __half* base_Y_tile; const __half* base_R_tile;
    uint2 idx_next[2]; WLoadElem w_buf0, w_buf1; uint4 b_reg[4]; GatherElem gv_aligned[USE_VEC_GATHER ? 4 : 1];
    uint32_t gv_even[USE_VEC_GATHER ? 1 : GATHER_SUBTILES][4]; uint32_t gv_odd[USE_VEC_GATHER ? 1 : GATHER_SUBTILES][4];
    float c[4][8]; __device__ __forceinline__ uint32_t c_slot() const { return (CLUSTERS_PER_BLOCK > 1) ? cluster_in_block : 0u; }
    __device__ __forceinline__ void load_idx(uint2 (&ibuf)[2]) { ibuf[0] = __ldg(idx_ptr + tid_in_group);
        ibuf[1] = __ldg(idx_ptr + tid_in_group + 4); idx_ptr += 8 * K_CHUNK;
    } __device__ __forceinline__ void load_w(WLoadElem& w_buf) { w_buf = __ldg(w_ptr + lane); w_ptr += 32; }
    __device__ __forceinline__ void prefetch_gathers(const uint2 (&ibuf)[2]) {
        const uint32_t r[4] = {ibuf[0].x, ibuf[0].y, ibuf[1].x, ibuf[1].y}; if constexpr (USE_VEC_GATHER) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) { const size_t elem_off = FLAT_BATCH_TILES ? ((size_t)r[i] << 6u) : (size_t)r[i];
                gv_aligned[i] = __ldg(reinterpret_cast<const GatherElem*>(base_X_u16 + elem_off)); } } else {
            #pragma unroll
            for (int s = 0; s < GATHER_SUBTILES; ++s) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) { const size_t elem_off = FLAT_BATCH_TILES ? ((size_t)r[i] << 6u) : (size_t)r[i];
                    const uint16_t* p = base_X_u16 + elem_off + (s * 2); gv_even[s][i] = __ldg(p + 0); gv_odd[s][i]  = __ldg(p + 1); }
            } } } __device__ __forceinline__ void pack_and_sync_stage(uint32_t dst_stage, SmemArray& s_b_frag) {
        if constexpr (USE_VEC_GATHER) { if constexpr (GATHER_SUBTILES == 1) { s_b_frag[dst_stage][c_slot()][k_sub][sub_tile][lane] =
                    k44_pack_b_frag_prmt(gv_aligned[0], gv_aligned[1], gv_aligned[2], gv_aligned[3]);
            } else if constexpr (GATHER_SUBTILES == 2) { s_b_frag[dst_stage][c_slot()][k_sub][sub_tile + 0][lane] =
                    k44_pack_b_frag_prmt(gv_aligned[0].x, gv_aligned[1].x, gv_aligned[2].x, gv_aligned[3].x);
                s_b_frag[dst_stage][c_slot()][k_sub][sub_tile + 1][lane] =
                    k44_pack_b_frag_prmt(gv_aligned[0].y, gv_aligned[1].y, gv_aligned[2].y, gv_aligned[3].y); } else {
                b_reg[0] = k44_pack_b_frag_prmt(gv_aligned[0].x, gv_aligned[1].x, gv_aligned[2].x, gv_aligned[3].x);
                b_reg[1] = k44_pack_b_frag_prmt(gv_aligned[0].y, gv_aligned[1].y, gv_aligned[2].y, gv_aligned[3].y);
                b_reg[2] = k44_pack_b_frag_prmt(gv_aligned[0].z, gv_aligned[1].z, gv_aligned[2].z, gv_aligned[3].z);
                b_reg[3] = k44_pack_b_frag_prmt(gv_aligned[0].w, gv_aligned[1].w, gv_aligned[2].w, gv_aligned[3].w);
                if constexpr (USE_SMEM) { auto& smem_sub = s_b_frag[dst_stage][c_slot()][k_sub];
                    smem_sub[0][lane] = b_reg[0]; smem_sub[1][lane] = b_reg[1];
                    smem_sub[2][lane] = b_reg[2]; smem_sub[3][lane] = b_reg[3]; } } } else {
            #pragma unroll
            for (int s = 0; s < GATHER_SUBTILES; ++s) { uint4 frag = make_uint4(
                    __byte_perm(gv_even[s][0], gv_even[s][1], 0x5410u), __byte_perm(gv_even[s][2], gv_even[s][3], 0x5410u),
                    __byte_perm(gv_odd[s][0],  gv_odd[s][1],  0x5410u), __byte_perm(gv_odd[s][2],  gv_odd[s][3],  0x5410u)
                ); if constexpr (USE_SMEM) s_b_frag[dst_stage][c_slot()][k_sub][sub_tile + s][lane] = frag; else b_reg[s] = frag; }
        } if constexpr (USE_SMEM) __syncthreads(); } template <bool PREFETCH_W0>
    __device__ __forceinline__ void compute_smem_2k(uint32_t cur, SmemArray& s_b_frag) {
        uint4 b0 = s_b_frag[cur][c_slot()][0][0][lane], b1 = s_b_frag[cur][c_slot()][0][1][lane];
        uint4 b2 = s_b_frag[cur][c_slot()][0][2][lane], b3 = s_b_frag[cur][c_slot()][0][3][lane];
        k44_compute_mma_4x_f16(w_buf0, b0, b1, b2, b3, c); if constexpr (PREFETCH_W0) load_w(w_buf0);
        b0 = s_b_frag[cur][c_slot()][1][0][lane]; b1 = s_b_frag[cur][c_slot()][1][1][lane];
        b2 = s_b_frag[cur][c_slot()][1][2][lane]; b3 = s_b_frag[cur][c_slot()][1][3][lane];
        k44_compute_mma_4x_f16(w_buf1, b0, b1, b2, b3, c); }
    // --------------------------------------------------------------------------
    // 1. VORSPANN (Teil 1): `setup`
    // --------------------------------------------------------------------------
    template <bool HAS_BIAS, bool PER_TILE_MASK, bool HAS_RESIDUAL> __device__ __forceinline__ void setup(
        __half* __restrict__ Y, const __half* __restrict__ X, const uint32_t* __restrict__ cluster_indices_32bit,
        const uint4* __restrict__ cluster_weights_mma, const __half* __restrict__ neuron_bias,
        const uint32_t* __restrict__ tile_offsets_X, const uint32_t* __restrict__ tile_offsets_Y,
        uint32_t B_vec16, const __half* __restrict__ residual_X, uint32_t Cin_flat = 0u, uint32_t Cout_flat = 0u
    ) { lane = threadIdx.x & 31u; group_id = lane >> 2u; tid_in_group = lane & 3u;
        const uint32_t warp_id = threadIdx.x >> 5u, wg_idx = blockIdx.y, batch_b = blockIdx.x;
        constexpr uint32_t WARPS_PER_K = 4 / GATHER_SUBTILES;
        const uint32_t warp_in_clust = warp_id % WARPS_PER_CLUSTER, nb_w0 = wg_idx * WARPS_PER_BLOCK + warp_id;
        k_sub = (K_CHUNK == 2) ? (warp_in_clust / WARPS_PER_K) : 0u; sub_tile = (warp_in_clust % WARPS_PER_K) * GATHER_SUBTILES;
        cluster_in_block = warp_id / WARPS_PER_CLUSTER; j_base0 = (nb_w0 << 4u) + group_id;
        idx_ptr = reinterpret_cast<const uint2*>(cluster_indices_32bit + ((nb_w0 / (CLUSTER_SIZE / 16)) * B_vec16 + k_sub) * 16u);
        w_ptr = reinterpret_cast<const WLoadElem*>(cluster_weights_mma) + ((size_t)(nb_w0 * B_vec16) << 5u);
        size_t x_tile_half_off = 0, y_tile_half_off = 0; valid_lanes = 0xffffu;
        if constexpr (FLAT_BATCH_TILES) { x_tile_half_off = (size_t)batch_b * ((size_t)Cin_flat << 6u);
            y_tile_half_off = (size_t)batch_b * ((size_t)Cout_flat << 6u); } else {
            x_tile_half_off = static_cast<size_t>(__ldg(tile_offsets_X + batch_b));
            const uint32_t raw_y = __ldg(tile_offsets_Y + batch_b); y_tile_half_off = static_cast<size_t>(raw_y);
            if constexpr (PER_TILE_MASK) { y_tile_half_off = static_cast<size_t>((raw_y & 0xffffu) << 2u);
                valid_lanes = raw_y >> 16u; }
        } base_X_u16 = reinterpret_cast<const uint16_t*>(X) + x_tile_half_off + (group_id << 3u) + (sub_tile << 1u);
        base_Y_tile = Y + y_tile_half_off; if constexpr (HAS_RESIDUAL) base_R_tile = residual_X + y_tile_half_off;
        float bias0 = 0.0f, bias1 = 0.0f; if constexpr (HAS_BIAS) {
            bias0 = __half2float(__ldg(neuron_bias + j_base0)); bias1 = __half2float(__ldg(neuron_bias + j_base0 + 8u)); }
        #pragma unroll
        for (int q = 0; q < 4; ++q) { c[q][0] = bias0; c[q][1] = bias0; c[q][4] = bias0; c[q][5] = bias0;
            c[q][2] = bias1; c[q][3] = bias1; c[q][6] = bias1; c[q][7] = bias1; } load_idx(idx_next); }
    // --------------------------------------------------------------------------
    // 2. HAUPTSCHLEIFE / PROLOG / DRAIN: `half_shader`
    // --------------------------------------------------------------------------
    template <bool DO_COMPUTE, bool DO_GATHER, bool DO_LOOKAHEAD, bool USE_FENCE = false>
    __device__ __forceinline__ void half_shader(uint32_t cur_slot, uint32_t next_slot, SmemArray& s_b_frag) {
        if constexpr (!DO_COMPUTE) { prefetch_gathers(idx_next); if constexpr (DO_LOOKAHEAD) load_idx(idx_next); load_w(w_buf0);
            pack_and_sync_stage(next_slot, s_b_frag); } else if constexpr (DO_GATHER) { load_w(w_buf1); prefetch_gathers(idx_next);
            if constexpr (DO_LOOKAHEAD) load_idx(idx_next); if constexpr (USE_SMEM) {
                compute_smem_2k<true>(cur_slot, s_b_frag); if constexpr (USE_FENCE) k44_fence_mma_acc(c);
                pack_and_sync_stage(next_slot, s_b_frag); } else {
                k44_compute_mma_4x_f16(w_buf0, b_reg[0], b_reg[1], b_reg[2], b_reg[3], c);
                if constexpr (USE_FENCE) k44_fence_mma_acc(c); pack_and_sync_stage(0u, s_b_frag); w_buf0 = w_buf1; }
        } else { if constexpr (USE_SMEM) { load_w(w_buf1); compute_smem_2k<false>(cur_slot, s_b_frag);
            } else { k44_compute_mma_4x_f16(w_buf0, b_reg[0], b_reg[1], b_reg[2], b_reg[3], c); } }
    }
    // --------------------------------------------------------------------------
    // 3. ABSPANN (Teil 2): `finish`
    // --------------------------------------------------------------------------
    template < int VALID_TOKENS, bool STREAMING_STORE, bool PER_TILE_MASK,
        bool HAS_RESIDUAL, K44Act EFFECTIVE_ACT, bool HAS_COUT_MAP >
    __device__ __forceinline__ void finish( uint32_t channel_stride_Y, const uint16_t* cout_map ) const { k44_finish_wavec_epilogue<
            false, VALID_TOKENS, STREAMING_STORE, PER_TILE_MASK, HAS_RESIDUAL, EFFECTIVE_ACT, HAS_COUT_MAP
        >(base_Y_tile, base_R_tile, j_base0, tid_in_group, valid_lanes, c, channel_stride_Y, cout_map); }
}; template <int CLUSTER_SIZE, int WARPS_PER_BLOCK, bool FLAT_BATCH_TILES = false>
using K44CudaSmlSPipeline = K44CudaDTokenShiftPipeline<CLUSTER_SIZE, WARPS_PER_BLOCK, FLAT_BATCH_TILES, true, true>; template <
    int CLUSTER_SIZE = 128, int WARPS_PER_BLOCK = (CLUSTER_SIZE >= 64 ? 8 : 4), bool HAS_BIAS = true, bool APPLY_RELU = false,
    bool APPLY_SIGMOID = false, int VALID_TOKENS = 64, bool STREAMING_STORE = false, bool APPLY_SILU = false,
    bool PER_TILE_MASK = false, bool HAS_RESIDUAL = false, K44Act ACT_ENUM = K44Act::NONE, bool HAS_COUT_MAP = false,
    bool FLAT_BATCH_TILES = false, bool ALIGNED_GATHER = false > __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
__global__ void k44_cuda_d2_token_shift_f16_kernel( __half* __restrict__ Y,
    const __half* __restrict__ X, const uint32_t* __restrict__ cluster_indices_32bit,
    const uint4* __restrict__ cluster_weights_mma, const __half* __restrict__ neuron_bias,
    const uint32_t* __restrict__ tile_offsets_X, const uint32_t* __restrict__ tile_offsets_Y,
    uint32_t channel_stride_Y, uint32_t B_vec16,
    const __half* __restrict__ residual_X = nullptr, const uint16_t* __restrict__ cout_map = nullptr,
    uint32_t Cin_flat = 0u, uint32_t Cout_flat = 0u
) { using Pipe = K44CudaDTokenShiftPipeline<CLUSTER_SIZE, WARPS_PER_BLOCK, FLAT_BATCH_TILES, ALIGNED_GATHER, false>;
    constexpr K44Act EFFECTIVE_ACT = (ACT_ENUM != K44Act::NONE) ? ACT_ENUM :
        (APPLY_SILU ? K44Act::SILU : (APPLY_SIGMOID ? K44Act::SIGMOID : (APPLY_RELU ? K44Act::RELU : K44Act::NONE)));
    __shared__ typename Pipe::SmemArray s_b_frag; Pipe p;
    p.template setup<HAS_BIAS, PER_TILE_MASK, HAS_RESIDUAL>( Y, X, cluster_indices_32bit, cluster_weights_mma, neuron_bias,
        tile_offsets_X, tile_offsets_Y, B_vec16, residual_X, Cin_flat, Cout_flat );
    p.template half_shader<false, true, true, true>(1u, 0u, s_b_frag); const uint32_t num_steps = B_vec16 / Pipe::K_CHUNK;
    for (uint32_t s = 0; s < num_steps - 2u; ++s) { p.template half_shader<true, true, true, true>(s & 1u, (s + 1u) & 1u, s_b_frag);
    } p.template half_shader<true, true, false, true>((num_steps - 2u) & 1u, (num_steps - 1u) & 1u, s_b_frag);
    p.template half_shader<true, false, false, true>((num_steps - 1u) & 1u, 0u, s_b_frag);
    p.template finish<VALID_TOKENS, STREAMING_STORE, PER_TILE_MASK, HAS_RESIDUAL, EFFECTIVE_ACT, HAS_COUT_MAP>(
        channel_stride_Y, cout_map ); } template <
    int CLUSTER_SIZE = 128, int WARPS_PER_BLOCK = (CLUSTER_SIZE >= 64 ? 8 : 4), bool HAS_BIAS = true, bool APPLY_RELU = false,
    bool APPLY_SIGMOID = false, int VALID_TOKENS = 64, bool STREAMING_STORE = false, bool APPLY_SILU = false,
    bool PER_TILE_MASK = false, bool HAS_RESIDUAL = false, K44Act ACT_ENUM = K44Act::NONE, bool HAS_COUT_MAP = false,
    bool FLAT_BATCH_TILES = false > __launch_bounds__(WARPS_PER_BLOCK * 32, 2) __global__ void k44_cuda_smlS_f16_kernel(
    __half* __restrict__ Y, const __half* __restrict__ X,
    const uint32_t* __restrict__ cluster_indices_32bit, const uint4* __restrict__ cluster_weights_mma,
    const __half* __restrict__ neuron_bias, const uint32_t* __restrict__ tile_offsets_X,
    const uint32_t* __restrict__ tile_offsets_Y, uint32_t channel_stride_Y,
    uint32_t B_vec16, const __half* __restrict__ residual_X = nullptr,
    const uint16_t* __restrict__ cout_map = nullptr, uint32_t Cin_flat = 0u, uint32_t Cout_flat = 0u ) {
    using Pipe = K44CudaSmlSPipeline<CLUSTER_SIZE, WARPS_PER_BLOCK, FLAT_BATCH_TILES>;
    constexpr K44Act EFFECTIVE_ACT = (ACT_ENUM != K44Act::NONE) ? ACT_ENUM :
        (APPLY_SILU ? K44Act::SILU : (APPLY_SIGMOID ? K44Act::SIGMOID : (APPLY_RELU ? K44Act::RELU : K44Act::NONE)));
    __shared__ typename Pipe::SmemArray s_b_frag; Pipe p;
    p.template setup<HAS_BIAS, PER_TILE_MASK, HAS_RESIDUAL>( Y, X, cluster_indices_32bit, cluster_weights_mma, neuron_bias,
        tile_offsets_X, tile_offsets_Y, B_vec16, residual_X, Cin_flat, Cout_flat );
    p.template half_shader<false, true, true, true>(1u, 0u, s_b_frag); const uint32_t num_steps = B_vec16 / Pipe::K_CHUNK;
    for (uint32_t s = 0; s < num_steps - 2u; ++s) { p.template half_shader<true, true, true, true>(s & 1u, (s + 1u) & 1u, s_b_frag);
    } p.template half_shader<true, true, false, true>((num_steps - 2u) & 1u, (num_steps - 1u) & 1u, s_b_frag);
    p.template half_shader<true, false, false, true>((num_steps - 1u) & 1u, 0u, s_b_frag);
    p.template finish<VALID_TOKENS, STREAMING_STORE, PER_TILE_MASK, HAS_RESIDUAL, EFFECTIVE_ACT, HAS_COUT_MAP>(
        channel_stride_Y, cout_map ); }
// ==============================================================================
// 3. K44_CUDA_lrgS (Large-S cp.async + ldmatrix.m8n8.x4.trans + WaveC Epilogue)
//    Structured into Vorspann (`setup`) -> Loop (`half_shader`) -> Abspann (`finish`)
// ==============================================================================
template <int CLUSTER_SIZE, int WARPS_PER_BLOCK, bool FLAT_BATCH_TILES = false> struct K44CudaLrgSPipeline {
    static constexpr int WARPS_PER_CLUSTER  = (CLUSTER_SIZE / 16 < WARPS_PER_BLOCK) ? (CLUSTER_SIZE / 16) : WARPS_PER_BLOCK;
    static constexpr int CLUSTERS_PER_BLOCK = WARPS_PER_BLOCK / WARPS_PER_CLUSTER;
    static constexpr int K_CHUNK            = (CLUSTER_SIZE >= 32) ? 2 : 1;
    static constexpr int GATHER_PASSES      = (4 * K_CHUNK) / WARPS_PER_CLUSTER;
    using SmemRawArray = uint4[2][CLUSTERS_PER_BLOCK][K_CHUNK][16][8];
    uint32_t lane, group_id, tid_in_group, col_u4, row_in_pass, k_sub, row_base, cluster_in_block, j_base0, valid_lanes;
    const uint32_t* idx_ptr; const uint4* w_ptr; const uint16_t* base_X_u16; __half* base_Y_tile;
    const __half* base_R_tile; uint32_t idx_next[GATHER_PASSES]; uint4 w_buf0, w_buf1; float c[4][8];
    __device__ __forceinline__ uint32_t c_slot() const { return (CLUSTERS_PER_BLOCK > 1) ? cluster_in_block : 0u; }
    __device__ __forceinline__ uint32_t smem_offset_bytes(uint32_t stage, uint32_t kk, uint32_t r, uint32_t col) const {
        return (((((stage * CLUSTERS_PER_BLOCK + c_slot()) * K_CHUNK + kk) * 16u + r) * 8u + col) << 4u); }
    __device__ __forceinline__ void load_idx(uint32_t (&ibuf)[GATHER_PASSES]) {
        #pragma unroll
        for (int p = 0; p < GATHER_PASSES; ++p) ibuf[p] = __ldg(idx_ptr + p * 4 + row_in_pass); idx_ptr += 16 * K_CHUNK;
    } __device__ __forceinline__ void load_w(uint4& w_buf) { w_buf = __ldg(w_ptr + lane); w_ptr += 32; }
    __device__ __forceinline__ void issue_cp_async_gathers(uint32_t dst_stage, const uint32_t (&ibuf)[GATHER_PASSES]) {
        #pragma unroll
        for (int p = 0; p < GATHER_PASSES; ++p) { const uint32_t r = row_base + p * 4 + row_in_pass;
            const uint32_t smem_dst = smem_offset_bytes(dst_stage, k_sub, r, col_u4 ^ (r & 7u));
            const size_t elem_off = FLAT_BATCH_TILES ? ((size_t)ibuf[p] << 6u) : (size_t)ibuf[p];
            k44_cp_async_cg_16(smem_dst, base_X_u16 + elem_off); } k44_cp_async_commit(); }
    __device__ __forceinline__ void compute_ldmatrix_1k(uint32_t cur, uint32_t kk, const uint4& w_cur) {
        const uint32_t r_ldm = lane & 15u, half_col = lane >> 4u, r_swz = r_ldm & 7u; uint4 b0, b1, b2, b3;
        k44_ldmatrix_m8n8_x4_trans_b16(b0, smem_offset_bytes(cur, kk, r_ldm, (0u | half_col) ^ r_swz));
        k44_ldmatrix_m8n8_x4_trans_b16(b1, smem_offset_bytes(cur, kk, r_ldm, (2u | half_col) ^ r_swz));
        k44_ldmatrix_m8n8_x4_trans_b16(b2, smem_offset_bytes(cur, kk, r_ldm, (4u | half_col) ^ r_swz));
        k44_ldmatrix_m8n8_x4_trans_b16(b3, smem_offset_bytes(cur, kk, r_ldm, (6u | half_col) ^ r_swz));
        k44_compute_mma_4x_f16(w_cur, b0, b1, b2, b3, c); }
    template <bool PREFETCH_W0> __device__ __forceinline__ void compute_smem_step(uint32_t cur) {
        compute_ldmatrix_1k(cur, 0u, w_buf0); if constexpr (PREFETCH_W0) load_w(w_buf0);
        if constexpr (K_CHUNK == 2) compute_ldmatrix_1k(cur, 1u, w_buf1); }
    // --------------------------------------------------------------------------
    // 1. VORSPANN (Teil 1): `setup`
    // --------------------------------------------------------------------------
    template <bool HAS_BIAS, bool PER_TILE_MASK, bool HAS_RESIDUAL> __device__ __forceinline__ void setup(
        __half* __restrict__ Y, const __half* __restrict__ X, const uint32_t* __restrict__ cluster_indices_32bit,
        const uint4* __restrict__ cluster_weights_mma, const __half* __restrict__ neuron_bias,
        const uint32_t* __restrict__ tile_offsets_X, const uint32_t* __restrict__ tile_offsets_Y,
        uint32_t B_vec16, const __half* __restrict__ residual_X, uint32_t Cin_flat = 0u, uint32_t Cout_flat = 0u
    ) { lane = threadIdx.x & 31u; group_id = lane >> 2u; tid_in_group = lane & 3u; col_u4 = lane & 7u; row_in_pass = lane >> 3u;
        const uint32_t warp_id = threadIdx.x >> 5u, wg_idx = blockIdx.y, batch_b = blockIdx.x;
        constexpr uint32_t WARPS_PER_K = 4 / GATHER_PASSES;
        const uint32_t warp_in_clust = warp_id % WARPS_PER_CLUSTER, nb_w0 = wg_idx * WARPS_PER_BLOCK + warp_id;
        k_sub = (K_CHUNK == 2) ? (warp_in_clust / WARPS_PER_K) : 0u; row_base = (warp_in_clust % WARPS_PER_K) * (GATHER_PASSES * 4);
        cluster_in_block = warp_id / WARPS_PER_CLUSTER; j_base0 = (nb_w0 << 4u) + group_id;
        idx_ptr = cluster_indices_32bit + ((nb_w0 / (CLUSTER_SIZE / 16)) * B_vec16 + k_sub) * 16u + row_base;
        w_ptr = cluster_weights_mma + ((size_t)(nb_w0 * B_vec16) << 5u); size_t x_tile_half_off = 0, y_tile_half_off = 0;
        valid_lanes = 0xffffu; if constexpr (FLAT_BATCH_TILES) { x_tile_half_off = (size_t)batch_b * ((size_t)Cin_flat << 6u);
            y_tile_half_off = (size_t)batch_b * ((size_t)Cout_flat << 6u); } else {
            x_tile_half_off = static_cast<size_t>(__ldg(tile_offsets_X + batch_b));
            const uint32_t raw_y = __ldg(tile_offsets_Y + batch_b); y_tile_half_off = static_cast<size_t>(raw_y);
            if constexpr (PER_TILE_MASK) { y_tile_half_off = static_cast<size_t>((raw_y & 0xffffu) << 2u);
                valid_lanes = raw_y >> 16u; } } base_X_u16 = reinterpret_cast<const uint16_t*>(X) + x_tile_half_off + (col_u4 << 3u);
        base_Y_tile = Y + y_tile_half_off; if constexpr (HAS_RESIDUAL) base_R_tile = residual_X + y_tile_half_off;
        float bias0 = 0.0f, bias1 = 0.0f; if constexpr (HAS_BIAS) {
            bias0 = __half2float(__ldg(neuron_bias + j_base0)); bias1 = __half2float(__ldg(neuron_bias + j_base0 + 8u)); }
        #pragma unroll
        for (int q = 0; q < 4; ++q) { c[q][0] = bias0; c[q][1] = bias0; c[q][4] = bias0; c[q][5] = bias0;
            c[q][2] = bias1; c[q][3] = bias1; c[q][6] = bias1; c[q][7] = bias1; } load_idx(idx_next); }
    // --------------------------------------------------------------------------
    // 2. HAUPTSCHLEIFE / PROLOG / DRAIN: `half_shader`
    // --------------------------------------------------------------------------
    template <bool DO_COMPUTE, bool DO_GATHER, bool DO_LOOKAHEAD>
    __device__ __forceinline__ void half_shader(uint32_t cur_slot, uint32_t next_slot) { if constexpr (!DO_COMPUTE) {
            issue_cp_async_gathers(next_slot, idx_next); if constexpr (DO_LOOKAHEAD) load_idx(idx_next);
            load_w(w_buf0); k44_cp_async_wait<0>(); __syncthreads(); } else if constexpr (DO_GATHER) {
            if constexpr (K_CHUNK == 2) load_w(w_buf1); issue_cp_async_gathers(next_slot, idx_next);
            if constexpr (DO_LOOKAHEAD) load_idx(idx_next); compute_smem_step<true>(cur_slot);
            k44_fence_mma_acc(c); k44_cp_async_wait<0>(); __syncthreads(); } else {
            if constexpr (K_CHUNK == 2) load_w(w_buf1); compute_smem_step<false>(cur_slot); } }
    // --------------------------------------------------------------------------
    // 3. ABSPANN (Teil 2): `finish`
    // --------------------------------------------------------------------------
    template < int VALID_TOKENS, bool STREAMING_STORE, bool PER_TILE_MASK,
        bool HAS_RESIDUAL, K44Act EFFECTIVE_ACT, bool HAS_COUT_MAP >
    __device__ __forceinline__ void finish( uint32_t channel_stride_Y, const uint16_t* cout_map ) const { k44_finish_wavec_epilogue<
            true, VALID_TOKENS, STREAMING_STORE, PER_TILE_MASK, HAS_RESIDUAL, EFFECTIVE_ACT, HAS_COUT_MAP
        >(base_Y_tile, base_R_tile, j_base0, tid_in_group, valid_lanes, c, channel_stride_Y, cout_map); } }; template <
    int CLUSTER_SIZE = 128, int WARPS_PER_BLOCK = (CLUSTER_SIZE >= 64 ? 8 : 4), bool HAS_BIAS = true, bool APPLY_RELU = false,
    bool APPLY_SIGMOID = false, int VALID_TOKENS = 64, bool STREAMING_STORE = false, bool APPLY_SILU = false,
    bool PER_TILE_MASK = false, bool HAS_RESIDUAL = false, K44Act ACT_ENUM = K44Act::NONE, bool HAS_COUT_MAP = false,
    bool FLAT_BATCH_TILES = false > __launch_bounds__(WARPS_PER_BLOCK * 32, 2) __global__ void k44_cuda_lrgS_f16_kernel(
    __half* __restrict__ Y, const __half* __restrict__ X,
    const uint32_t* __restrict__ cluster_indices_32bit, const uint4* __restrict__ cluster_weights_mma,
    const __half* __restrict__ neuron_bias, const uint32_t* __restrict__ tile_offsets_X,
    const uint32_t* __restrict__ tile_offsets_Y, uint32_t channel_stride_Y,
    uint32_t B_vec16, const __half* __restrict__ residual_X = nullptr,
    const uint16_t* __restrict__ cout_map = nullptr, uint32_t Cin_flat = 0u, uint32_t Cout_flat = 0u ) {
    using Pipe = K44CudaLrgSPipeline<CLUSTER_SIZE, WARPS_PER_BLOCK, FLAT_BATCH_TILES>;
    constexpr K44Act EFFECTIVE_ACT = (ACT_ENUM != K44Act::NONE) ? ACT_ENUM :
        (APPLY_SILU ? K44Act::SILU : (APPLY_SIGMOID ? K44Act::SIGMOID : (APPLY_RELU ? K44Act::RELU : K44Act::NONE)));
    __shared__ typename Pipe::SmemRawArray s_raw; asm volatile("" :: "l"(&s_raw));
    Pipe p; p.template setup<HAS_BIAS, PER_TILE_MASK, HAS_RESIDUAL>( Y, X, cluster_indices_32bit, cluster_weights_mma, neuron_bias,
        tile_offsets_X, tile_offsets_Y, B_vec16, residual_X, Cin_flat, Cout_flat );
    p.template half_shader<false, true, true>(1u, 0u); const uint32_t num_steps = B_vec16 / Pipe::K_CHUNK;
    for (uint32_t s = 0; s < num_steps - 2u; ++s) { p.template half_shader<true, true, true>(s & 1u, (s + 1u) & 1u);
    } p.template half_shader<true, true, false>((num_steps - 2u) & 1u, (num_steps - 1u) & 1u);
    p.template half_shader<true, false, false>((num_steps - 1u) & 1u, 0u);
    p.template finish<VALID_TOKENS, STREAMING_STORE, PER_TILE_MASK, HAS_RESIDUAL, EFFECTIVE_ACT, HAS_COUT_MAP>(
        channel_stride_Y, cout_map ); }
// ==============================================================================
// 4. Offline Weight Interleaving for CUDA mma.sync.aligned.m16n8k16 (Host & GPU)
// ==============================================================================
__global__ void interleave_weights_cluster16_mma_cuda_kernel( uint4* __restrict__ dst_mma,
    const uint32_t* __restrict__ src_compact_u32, uint32_t num_output_neurons, uint32_t B_vec16 ) {
    const uint32_t total_elements = (num_output_neurons / 16u) * B_vec16 * 32u; const uint32_t row_stride_u32 = B_vec16 * 8u;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const uint32_t tile_idx = idx >> 5u, lane = idx & 31u, group_id = lane >> 2u, tid_in_group = lane & 3u;
        const uint32_t n_base = (tile_idx / B_vec16) * 16u, k_base_u32 = (tile_idx % B_vec16) * 8u + tid_in_group;
        dst_mma[idx] = make_uint4( src_compact_u32[(n_base + group_id + 0u) * row_stride_u32 + k_base_u32 + 0u],
            src_compact_u32[(n_base + group_id + 8u) * row_stride_u32 + k_base_u32 + 0u],
            src_compact_u32[(n_base + group_id + 0u) * row_stride_u32 + k_base_u32 + 4u],
            src_compact_u32[(n_base + group_id + 8u) * row_stride_u32 + k_base_u32 + 4u] ); } }
inline void interleave_weights_cluster16_mma_cuda_host( uint4* dst_mma, const __half* src_compact, uint32_t num_output_neurons,
    uint32_t B_vec16 ) { const uint32_t* src_u32 = reinterpret_cast<const uint32_t*>(src_compact);
    const uint32_t row_stride_u32 = B_vec16 * 8u, num_tiles = (num_output_neurons / 16u) * B_vec16;
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const uint32_t n_base = (tile_idx / B_vec16) * 16u, k_base_u32 = (tile_idx % B_vec16) * 8u;
        for (uint32_t lane = 0; lane < 32u; ++lane) { const uint32_t group_id = lane >> 2u, tid_in_group = lane & 3u;
            dst_mma[tile_idx * 32u + lane] = make_uint4(
                src_u32[(n_base + group_id + 0u) * row_stride_u32 + k_base_u32 + tid_in_group + 0u],
                src_u32[(n_base + group_id + 8u) * row_stride_u32 + k_base_u32 + tid_in_group + 0u],
                src_u32[(n_base + group_id + 0u) * row_stride_u32 + k_base_u32 + tid_in_group + 4u],
                src_u32[(n_base + group_id + 8u) * row_stride_u32 + k_base_u32 + tid_in_group + 4u] ); } }
}
// ==============================================================================
// 5. NVIDIA Blackwell (sm_100a) 5th-Gen Tensor Core (`k44_cuda_tcgen05_f3_kernel`)
// ==============================================================================
__device__ __forceinline__ uint64_t k44_make_desc_w_swz32(uint32_t smem_u32_addr) { uint64_t d = 0;
    d |= (uint64_t)((smem_u32_addr & 0x3ffffu) >> 4); d |= (uint64_t)(16ULL   >> 4) << 16;
    d |= (uint64_t)(256ULL  >> 4) << 32; d |= (3ULL << 62); return d; }
__device__ __forceinline__ uint64_t k44_make_desc_x_swz128(uint32_t smem_u32_addr) { uint64_t d = 0;
    d |= (uint64_t)((smem_u32_addr & 0x3ffffu) >> 4); d |= (uint64_t)(2048ULL >> 4) << 16;
    d |= (uint64_t)(1024ULL >> 4) << 32; d |= (1ULL << 62); return d; }
__global__ void interleave_weights_tcgen05_cuda_kernel( uint4* __restrict__ dst_tc,
    const uint32_t* __restrict__ src_compact_u32, uint32_t num_output_neurons, uint32_t B_vec16 ) {
    const uint4* src_u4 = reinterpret_cast<const uint4*>(src_compact_u32);
    const uint32_t total_u4 = (num_output_neurons / 128u) * B_vec16 * 256u; const uint32_t row_stride_u4 = B_vec16 * 2u;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_u4;
         idx += gridDim.x * blockDim.x) { const uint32_t lane     = idx & 31u;
        const uint32_t s_step   = (idx >> 5u) % B_vec16; const uint32_t w_glob   = (idx >> 5u) / B_vec16;
        const uint32_t wg_idx   = w_glob >> 3u; const uint32_t warp_id  = w_glob & 7u;
        const uint32_t t_in_cta = warp_id * 32u + lane; const uint32_t j_in_wg  = t_in_cta >> 1u;
        const uint32_t u4_slot  = t_in_cta & 1u; const uint32_t k_half   = u4_slot ^ (j_in_wg & 1u);
        const uint32_t neuron_j = wg_idx * 128u + j_in_wg;
        dst_tc[idx] = src_u4[(size_t)neuron_j * row_stride_u4 + s_step * 2u + k_half]; } } template <
    int CLUSTER_SIZE, int N_BLOCKS = 2, int M_TILES = 4, uint32_t NUM_STAGES = 6,
    uint32_t CP_ASYNC_PREFETCH = 4, bool HAS_BIAS = true, bool STREAMING_STORE = true >
__global__ __launch_bounds__(288, 1) void k44_cuda_tcgen05_f3_kernel( __half* __restrict__ Y,
    const __half* __restrict__ X, const uint32_t* __restrict__ cluster_indices_32bit,
    const uint4* __restrict__ cluster_weights_tc, const __half* __restrict__ neuron_bias, uint32_t Cin, uint32_t Cout,
    uint32_t B_vec16 ) {
#if __CUDA_ARCH__ >= 1000
    static_assert(N_BLOCKS == 1 || N_BLOCKS == 2, "N_BLOCKS must be 1 or 2");
    static_assert(M_TILES == 1 || M_TILES == 2 || M_TILES == 4, "M_TILES must be 1, 2, or 4");
    static_assert(CP_ASYNC_PREFETCH >= 2 && CP_ASYNC_PREFETCH < NUM_STAGES, "Invalid CP_ASYNC_PREFETCH");
    constexpr uint32_t X_BYTES_PER_KSUB = M_TILES * 2048u; constexpr uint32_t X_BYTES_STAGE    = 2u * X_BYTES_PER_KSUB;
    constexpr uint32_t W_BYTES_STAGE    = N_BLOCKS * 8192u; constexpr uint32_t STAGE_BYTES      = X_BYTES_STAGE + W_BYTES_STAGE;
    constexpr uint32_t TMEM_COLS        = (N_BLOCKS * M_TILES * 64u <= 128u) ? 128u : (N_BLOCKS * M_TILES * 64u);
    extern __shared__ __align__(1024) uint4 s_dyn[];
    uint64_t* s_mbar_full  = reinterpret_cast<uint64_t*>(s_dyn + NUM_STAGES * (STAGE_BYTES / 16u));
    uint64_t* s_mbar_empty = s_mbar_full + NUM_STAGES; uint32_t* p_tmem_addr  = reinterpret_cast<uint32_t*>(s_mbar_empty + NUM_STAGES);
    constexpr uint32_t NEURONS_PER_CTA  = N_BLOCKS * 128u;
    constexpr uint32_t CTAS_PER_CLUSTER = (CLUSTER_SIZE >= (int)NEURONS_PER_CTA) ? (CLUSTER_SIZE / NEURONS_PER_CTA) : 1u;
    const uint32_t tid        = threadIdx.x; const uint32_t lane       = tid & 31u;
    const uint32_t warp_id    = tid >> 5u; const uint32_t batch_grp  = blockIdx.x / CTAS_PER_CLUSTER;
    const uint32_t c_sub      = blockIdx.x % CTAS_PER_CLUSTER; const uint32_t cluster_id = blockIdx.y;
    const uint32_t cta_n_idx  = cluster_id * CTAS_PER_CLUSTER + c_sub;
    const uint32_t mbar_full_base  = static_cast<uint32_t>(__cvta_generic_to_shared(s_mbar_full));
    const uint32_t mbar_empty_base = static_cast<uint32_t>(__cvta_generic_to_shared(s_mbar_empty)); if (tid < NUM_STAGES) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;\n" :: "r"(mbar_full_base  + tid * 8u));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"   :: "r"(mbar_empty_base + tid * 8u)); }
    if (tid < 32) { uint32_t ptr_smem = static_cast<uint32_t>(__cvta_generic_to_shared(p_tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(ptr_smem), "n"(TMEM_COLS)); }
    __syncthreads(); const uint32_t tmem_d        = *p_tmem_addr;
    const uint32_t smem_base_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(&s_dyn[0]));
    const uint32_t num_steps     = B_vec16 / 2u; if (warp_id < 8u) {
        const uint32_t k_sub          = warp_id >> 2u; const uint32_t row_base       = (warp_id & 3u) * 4u;
        const uint32_t num_batch_grps = gridDim.x / CTAS_PER_CLUSTER; const uint32_t M_u4           = num_batch_grps * (M_TILES * 8u);
        const uint4* w0_ptr = cluster_weights_tc + ((size_t)((cta_n_idx * N_BLOCKS + 0u) * 8u + warp_id) * B_vec16 << 5u) + lane;
        const uint4* w1_ptr = (N_BLOCKS >= 2) ? (cluster_weights_tc + ((size_t)((cta_n_idx * N_BLOCKS + 1u) * 8u + warp_id) * B_vec16 << 5u) + lane) : nullptr;
        if constexpr (M_TILES == 4) { const uint32_t m_sub  = lane >> 3u;
            const uint32_t col_u4 = lane & 7u;
            const uint32_t x_off0 = m_sub * 2048u + (((row_base + 0u) * 8u + (col_u4 ^ ((row_base + 0u) & 7u))) << 4u);
            const uint32_t x_off1 = m_sub * 2048u + (((row_base + 1u) * 8u + (col_u4 ^ ((row_base + 1u) & 7u))) << 4u);
            const uint32_t x_off2 = m_sub * 2048u + (((row_base + 2u) * 8u + (col_u4 ^ ((row_base + 2u) & 7u))) << 4u);
            const uint32_t x_off3 = m_sub * 2048u + (((row_base + 3u) * 8u + (col_u4 ^ ((row_base + 3u) & 7u))) << 4u);
            const uint4* idx4_ptr  = reinterpret_cast<const uint4*>(cluster_indices_32bit + (cluster_id * B_vec16 + k_sub) * 16u + row_base);
            const uint4* base_X_u4 = reinterpret_cast<const uint4*>(X) + (size_t)(batch_grp * 32u) + lane;
            auto issue_stage_mbar = [&](uint32_t st, uint4 idx4) {
                uint32_t st_off = smem_base_u32 + st * STAGE_BYTES, ks_off = k_sub * X_BYTES_PER_KSUB;
                k44_cp_async_cg_16(st_off + ks_off + x_off0, base_X_u4 + (size_t)idx4.x * M_u4);
                k44_cp_async_cg_16(st_off + ks_off + x_off1, base_X_u4 + (size_t)idx4.y * M_u4);
                k44_cp_async_cg_16(st_off + ks_off + x_off2, base_X_u4 + (size_t)idx4.z * M_u4);
                k44_cp_async_cg_16(st_off + ks_off + x_off3, base_X_u4 + (size_t)idx4.w * M_u4);
                uint32_t w_base = X_BYTES_STAGE + ((warp_id * 32u + lane) << 4u); k44_cp_async_cg_16(st_off + w_base + 0u,    w0_ptr);
                k44_cp_async_cg_16(st_off + w_base + 4096u, w0_ptr + 32u); w0_ptr += 64u;
                if constexpr (N_BLOCKS >= 2) { k44_cp_async_cg_16(st_off + w_base + 8192u,  w1_ptr);
                    k44_cp_async_cg_16(st_off + w_base + 12288u, w1_ptr + 32u); w1_ptr += 64u;
                } asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(mbar_full_base + st * 8u) : "memory");
            }; uint4 idx_q0 = __ldg(idx4_ptr), idx_q1 = __ldg(idx4_ptr + 8u), idx_q2 = __ldg(idx4_ptr + 16u);
            idx4_ptr += 24u;
            #pragma unroll
            for (uint32_t st = 0; st < NUM_STAGES; ++st) { issue_stage_mbar(st, idx_q0);
                idx_q0 = idx_q1; idx_q1 = idx_q2; idx_q2 = __ldg(idx4_ptr); idx4_ptr += 8u; }
            uint32_t dst_st = 0u, empty_phase_mask = 0u; for (uint32_t s = NUM_STAGES; s < num_steps; ++s) {
                uint32_t wait_mbar = mbar_empty_base + dst_st * 8u, ph = (empty_phase_mask >> dst_st) & 1u; asm volatile(
                    "{\n.reg .pred p_done;\nWAIT_EMPTY_4_%=:\n" "mbarrier.try_wait.parity.shared::cta.b64 p_done, [%0], %1;\n"
                    "@!p_done bra WAIT_EMPTY_4_%=;\n}\n" :: "r"(wait_mbar), "r"(ph) : "memory" );
                empty_phase_mask ^= (1u << dst_st); issue_stage_mbar(dst_st, idx_q0);
                idx_q0 = idx_q1; idx_q1 = idx_q2; idx4_ptr += (s + 3u < num_steps) ? 8u : 0u;
                idx_q2 = __ldg(idx4_ptr - 8u); dst_st = (dst_st + 1u == NUM_STAGES) ? 0u : (dst_st + 1u);
            } } else {
            const uint32_t col_u4 = lane & 7u, row_in_pass = lane >> 3u, r_g = row_base + row_in_pass, col_swz_g = col_u4 ^ (r_g & 7u);
            const uint32_t* idx_ptr = cluster_indices_32bit + (cluster_id * B_vec16 + k_sub) * 16u + r_g;
            const uint4* base_X_u4  = reinterpret_cast<const uint4*>(X) + (size_t)(batch_grp * M_TILES * 8u) + col_u4;
            auto issue_stage_mbar = [&](uint32_t st, uint32_t idx) {
                uint32_t st_off = smem_base_u32 + st * STAGE_BYTES, x_off = ((r_g * 8u + col_swz_g) << 4u), ks_off = k_sub * X_BYTES_PER_KSUB;
                const uint4* src_x = base_X_u4 + (size_t)idx * M_u4; k44_cp_async_cg_16(st_off + ks_off + 0u + x_off, src_x + 0u);
                if constexpr (M_TILES >= 2) k44_cp_async_cg_16(st_off + ks_off + 2048u + x_off, src_x + 8u);
                uint32_t w_base = X_BYTES_STAGE + ((warp_id * 32u + lane) << 4u); k44_cp_async_cg_16(st_off + w_base + 0u,    w0_ptr);
                k44_cp_async_cg_16(st_off + w_base + 4096u, w0_ptr + 32u); w0_ptr += 64u;
                if constexpr (N_BLOCKS >= 2) { k44_cp_async_cg_16(st_off + w_base + 8192u,  w1_ptr);
                    k44_cp_async_cg_16(st_off + w_base + 12288u, w1_ptr + 32u); w1_ptr += 64u;
                } asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(mbar_full_base + st * 8u) : "memory");
            }; uint32_t idx_q0 = __ldg(idx_ptr), idx_q1 = __ldg(idx_ptr + 32u), idx_q2 = __ldg(idx_ptr + 64u);
            idx_ptr += 96u;
            #pragma unroll
            for (uint32_t st = 0; st < NUM_STAGES; ++st) { issue_stage_mbar(st, idx_q0);
                idx_q0 = idx_q1; idx_q1 = idx_q2; idx_q2 = __ldg(idx_ptr); idx_ptr += 32u; }
            uint32_t dst_st = 0u, empty_phase_mask = 0u; for (uint32_t s = NUM_STAGES; s < num_steps; ++s) {
                uint32_t wait_mbar = mbar_empty_base + dst_st * 8u, ph = (empty_phase_mask >> dst_st) & 1u; asm volatile(
                    "{\n.reg .pred p_done;\nWAIT_EMPTY_12_%=:\n" "mbarrier.try_wait.parity.shared::cta.b64 p_done, [%0], %1;\n"
                    "@!p_done bra WAIT_EMPTY_12_%=;\n}\n" :: "r"(wait_mbar), "r"(ph) : "memory" );
                empty_phase_mask ^= (1u << dst_st); issue_stage_mbar(dst_st, idx_q0);
                idx_q0 = idx_q1; idx_q1 = idx_q2; idx_ptr += (s + 3u < num_steps) ? 32u : 0u;
                idx_q2 = __ldg(idx_ptr - 32u); dst_st = (dst_st + 1u == NUM_STAGES) ? 0u : (dst_st + 1u);
            } }
    } else if (tid == 256u) { constexpr uint32_t N_MMA = (M_TILES == 1) ? 64u : 128u;
        constexpr uint32_t idesc = (1u << 4) | (1u << 16) | ((N_MMA >> 3) << 17) | ((128u >> 4) << 24);
        uint32_t cur_st = 0u, full_phase_mask = 0u; for (uint32_t s = 0; s < num_steps; ++s) {
            uint32_t full_mbar = mbar_full_base + cur_st * 8u, ph = (full_phase_mask >> cur_st) & 1u; asm volatile(
                "{\n.reg .pred p_done;\nWAIT_FULL_%=:\n" "mbarrier.try_wait.parity.shared::cta.b64 p_done, [%0], %1;\n"
                "@!p_done bra WAIT_FULL_%=;\n}\n" :: "r"(full_mbar), "r"(ph) : "memory" );
            full_phase_mask ^= (1u << cur_st); uint32_t st_off     = smem_base_u32 + cur_st * STAGE_BYTES;
            uint32_t empty_mbar = mbar_empty_base + cur_st * 8u; uint32_t first_step = (s == 0u) ? 0u : 1u;
            uint64_t desc_x01_k0 = k44_make_desc_x_swz128(st_off + 0u);
            uint64_t desc_x01_k1 = k44_make_desc_x_swz128(st_off + X_BYTES_PER_KSUB);
            uint64_t desc_w0_k0  = k44_make_desc_w_swz32(st_off + X_BYTES_STAGE + 0u);
            uint64_t desc_w0_k1  = k44_make_desc_w_swz32(st_off + X_BYTES_STAGE + 4096u);
            if constexpr (N_BLOCKS == 1 && M_TILES <= 2) { asm volatile(
                    "{\n.reg .pred p0, p1;\nsetp.ne.b32 p0, %6, 0;\nsetp.ne.b32 p1, 1, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %3, %5, p0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %2, %4, %5, p1;\n"
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%7];\n}\n"
                    :: "r"(tmem_d), "l"(desc_w0_k0), "l"(desc_w0_k1),
                       "l"(desc_x01_k0), "l"(desc_x01_k1), "r"(idesc), "r"(first_step), "r"(empty_mbar) );
            } else if constexpr (N_BLOCKS == 1 && M_TILES == 4) { uint64_t desc_x23_k0 = k44_make_desc_x_swz128(st_off + 4096u);
                uint64_t desc_x23_k1 = k44_make_desc_x_swz128(st_off + X_BYTES_PER_KSUB + 4096u); uint32_t t1 = tmem_d + 128u;
                asm volatile( "{\n.reg .pred p0, p1;\nsetp.ne.b32 p0, %10, 0;\nsetp.ne.b32 p1, 1, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %2, %4, %8, p0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%1], %2, %5, %8, p0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %3, %6, %8, p1;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%1], %3, %7, %8, p1;\n"
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%9];\n}\n"
                    :: "r"(tmem_d), "r"(t1), "l"(desc_w0_k0), "l"(desc_w0_k1),
                       "l"(desc_x01_k0), "l"(desc_x23_k0), "l"(desc_x01_k1), "l"(desc_x23_k1),
                       "r"(idesc), "r"(empty_mbar), "r"(first_step) );
            } else if constexpr (N_BLOCKS == 2 && M_TILES <= 2) {
                uint64_t desc_w1_k0 = k44_make_desc_w_swz32(st_off + X_BYTES_STAGE + 8192u);
                uint64_t desc_w1_k1 = k44_make_desc_w_swz32(st_off + X_BYTES_STAGE + 12288u); uint32_t t1 = tmem_d + 128u;
                asm volatile( "{\n.reg .pred p0, p1;\nsetp.ne.b32 p0, %10, 0;\nsetp.ne.b32 p1, 1, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %2, %6, %8, p0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%1], %4, %6, %8, p0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %3, %7, %8, p1;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%1], %5, %7, %8, p1;\n"
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%9];\n}\n" :: "r"(tmem_d), "r"(t1),
                       "l"(desc_w0_k0), "l"(desc_w0_k1), "l"(desc_w1_k0), "l"(desc_w1_k1), "l"(desc_x01_k0), "l"(desc_x01_k1),
                       "r"(idesc), "r"(empty_mbar), "r"(first_step) );
            } else { uint64_t desc_x23_k0 = k44_make_desc_x_swz128(st_off + 4096u);
                uint64_t desc_x23_k1 = k44_make_desc_x_swz128(st_off + X_BYTES_PER_KSUB + 4096u);
                uint64_t desc_w1_k0  = k44_make_desc_w_swz32(st_off + X_BYTES_STAGE + 8192u);
                uint64_t desc_w1_k1  = k44_make_desc_w_swz32(st_off + X_BYTES_STAGE + 12288u);
                uint32_t t0 = tmem_d + 0u, t1 = tmem_d + 128u, t2 = tmem_d + 256u, t3 = tmem_d + 384u; asm volatile(
                    "{\n.reg .pred p0, p1;\nsetp.ne.b32 p0, %14, 0;\nsetp.ne.b32 p1, 1, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %4, %8,  %12, p0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%1], %4, %9,  %12, p0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], %6, %8,  %12, p0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%3], %6, %9,  %12, p0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %5, %10, %12, p1;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%1], %5, %11, %12, p1;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], %7, %10, %12, p1;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%3], %7, %11, %12, p1;\n"
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%13];\n}\n" :: "r"(t0), "r"(t1), "r"(t2), "r"(t3),
                       "l"(desc_w0_k0), "l"(desc_w0_k1), "l"(desc_w1_k0), "l"(desc_w1_k1),
                       "l"(desc_x01_k0), "l"(desc_x23_k0), "l"(desc_x01_k1), "l"(desc_x23_k1),
                       "r"(idesc), "r"(empty_mbar), "r"(first_step) );
            } if (s + 1u == num_steps) {
                uint32_t last_empty_ph = ((num_steps - 1u) / NUM_STAGES) & 1u; asm volatile(
                    "{\n.reg .pred p_done;\nWAIT_FINAL_EMPTY_%=:\n" "mbarrier.try_wait.parity.shared::cta.b64 p_done, [%0], %1;\n"
                    "@!p_done bra WAIT_FINAL_EMPTY_%=;\n}\n" :: "r"(empty_mbar), "r"(last_empty_ph) : "memory" );
            } cur_st = (cur_st + 1u == NUM_STAGES) ? 0u : (cur_st + 1u);
        } }
    __syncthreads(); if (tid < 128) {
        #pragma unroll
        for (int nb = 0; nb < N_BLOCKS; ++nb) { const uint32_t neuron_j = (cta_n_idx * N_BLOCKS + nb) * 128u + tid;
            float bias_val = 0.0f; if constexpr (HAS_BIAS) {
                bias_val = __half2float(__ldg(neuron_bias + neuron_j)); }
            #pragma unroll
            for (int bt = 0; bt < M_TILES; ++bt) { __half* dst_y = Y + ((size_t)(batch_grp * M_TILES + bt) * Cout + neuron_j) * 64u;
                #pragma unroll
                for (int c_blk = 0; c_blk < 8; ++c_blk) { uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                    uint32_t t_addr = tmem_d + (nb * ((M_TILES <= 2) ? 128u : 256u)) + (bt * 64u) + (c_blk * 8u); asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                        "tcgen05.wait::ld.sync.aligned;\n"
                        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7) : "r"(t_addr)
                    ); uint4 packed = make_uint4(
                        k44_pack_f16x2(__uint_as_float(r0) + bias_val, __uint_as_float(r1) + bias_val),
                        k44_pack_f16x2(__uint_as_float(r2) + bias_val, __uint_as_float(r3) + bias_val),
                        k44_pack_f16x2(__uint_as_float(r4) + bias_val, __uint_as_float(r5) + bias_val),
                        k44_pack_f16x2(__uint_as_float(r6) + bias_val, __uint_as_float(r7) + bias_val) );
                    if constexpr (STREAMING_STORE) { k44_store_global_cs_v4_u32(dst_y + c_blk * 8, packed);
                    } else { k44_store_global_v4_u32(dst_y + c_blk * 8, packed, false);
                    } }
            } }
    } __syncthreads();
    if (tid < 32) { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" :: "r"(tmem_d), "n"(TMEM_COLS)); }
#else
    (void)Y; (void)X; (void)cluster_indices_32bit; (void)cluster_weights_tc; (void)neuron_bias; (void)Cin; (void)Cout; (void)B_vec16;
#endif
}
// =================================================================================================
// 6. K44 CUDA PACK LAUNCHERS & EPILOGUE DISPATCH MACROS
// =================================================================================================
template <int CLUSTER_SIZE, bool HAS_BIAS = true, bool APPLY_RELU = false,
          bool APPLY_SIGMOID = false, int VALID_TOKENS = 64, bool STREAMING_STORE = false,
          bool APPLY_SILU = false, bool PER_TILE_MASK = false, bool HAS_RESIDUAL = false,
          int ACT_ENUM = 0, bool HAS_COUT_MAP = false>
static inline void launch_k44_token_shift_f16_forward(
    __half *d_Y, const __half *d_X, const uint32_t *d_cluster_indices_32bit,
    const uint4 *d_cluster_weights_mma, const __half *d_neuron_bias,
    const uint32_t *d_tile_offsets_X, const uint32_t *d_tile_offsets_Y, uint32_t num_batch_tiles,
    uint32_t num_output_neurons, uint32_t channel_stride_Y, uint32_t B_vec16,
    cudaStream_t stream = nullptr, const __half *d_residual_X = nullptr,
    const uint16_t *d_cout_map = nullptr) {
    constexpr int DEFAULT_WAVES = (CLUSTER_SIZE >= 64) ? 8 : 4;
    constexpr int SINGLE_WAVES = CLUSTER_SIZE / 16;
    if ((num_output_neurons >= (uint32_t)(DEFAULT_WAVES * 16)) &&
        (num_output_neurons % (uint32_t)(DEFAULT_WAVES * 16) == 0u)) {
        uint32_t neuron_blocks = num_output_neurons / (DEFAULT_WAVES * 16);
        dim3 grid(num_batch_tiles, neuron_blocks, 1);
        dim3 block(DEFAULT_WAVES * 32, 1, 1);
        k44_cuda_d2_token_shift_f16_kernel<CLUSTER_SIZE, DEFAULT_WAVES, HAS_BIAS, APPLY_RELU,
                                           APPLY_SIGMOID, VALID_TOKENS, STREAMING_STORE, APPLY_SILU,
                                           PER_TILE_MASK, HAS_RESIDUAL,
                                           static_cast<K44Act>(ACT_ENUM), HAS_COUT_MAP>
            <<<grid, block, 0, stream>>>(d_Y, d_X, d_cluster_indices_32bit, d_cluster_weights_mma,
                                         d_neuron_bias, d_tile_offsets_X, d_tile_offsets_Y,
                                         channel_stride_Y, B_vec16, d_residual_X, d_cout_map);
    } else {
        uint32_t neuron_blocks =
            (num_output_neurons + (SINGLE_WAVES * 16) - 1) / (SINGLE_WAVES * 16);
        dim3 grid(num_batch_tiles, neuron_blocks, 1);
        dim3 block(SINGLE_WAVES * 32, 1, 1);
        k44_cuda_d2_token_shift_f16_kernel<CLUSTER_SIZE, SINGLE_WAVES, HAS_BIAS, APPLY_RELU,
                                           APPLY_SIGMOID, VALID_TOKENS, STREAMING_STORE, APPLY_SILU,
                                           PER_TILE_MASK, HAS_RESIDUAL,
                                           static_cast<K44Act>(ACT_ENUM), HAS_COUT_MAP>
            <<<grid, block, 0, stream>>>(d_Y, d_X, d_cluster_indices_32bit, d_cluster_weights_mma,
                                         d_neuron_bias, d_tile_offsets_X, d_tile_offsets_Y,
                                         channel_stride_Y, B_vec16, d_residual_X, d_cout_map);
    }
}
#define K44_DISPATCH_EPILOGUE_EX(CLUSTER_SZ, MASK_BOOL, BIAS_BOOL, ACT_VAL, RES_BOOL, CMAP_BOOL,   \
                                 ...)                                                              \
    do {                                                                                           \
        auto launch_act_res_cmap = [&](auto bias_tag, auto res_tag, auto cmap_tag) {               \
            constexpr bool HB = decltype(bias_tag)::value;                                         \
            constexpr bool HR = decltype(res_tag)::value;                                          \
            constexpr bool HC = decltype(cmap_tag)::value;                                         \
            switch (static_cast<int>(ACT_VAL)) {                                                   \
            case 1:                                                                                \
                launch_k44_token_shift_f16_forward<CLUSTER_SZ, HB, true, false, 64, false, false,  \
                                                   MASK_BOOL, HR, 1, HC>(__VA_ARGS__);             \
                break;                                                                             \
            case 2:                                                                                \
                launch_k44_token_shift_f16_forward<CLUSTER_SZ, HB, false, true, 64, false, false,  \
                                                   MASK_BOOL, HR, 2, HC>(__VA_ARGS__);             \
                break;                                                                             \
            case 3:                                                                                \
                launch_k44_token_shift_f16_forward<CLUSTER_SZ, HB, false, false, 64, false, true,  \
                                                   MASK_BOOL, HR, 3, HC>(__VA_ARGS__);             \
                break;                                                                             \
            case 4:                                                                                \
                launch_k44_token_shift_f16_forward<CLUSTER_SZ, HB, false, false, 64, false, false, \
                                                   MASK_BOOL, HR, 4, HC>(__VA_ARGS__);             \
                break;                                                                             \
            default:                                                                               \
                launch_k44_token_shift_f16_forward<CLUSTER_SZ, HB, false, false, 64, false, false, \
                                                   MASK_BOOL, HR, 0, HC>(__VA_ARGS__);             \
                break;                                                                             \
            }                                                                                      \
        };                                                                                         \
        auto launch_res_cmap = [&](auto bias_tag) {                                                \
            if (CMAP_BOOL) {                                                                       \
                if (RES_BOOL)                                                                      \
                    launch_act_res_cmap(bias_tag, std::true_type{}, std::true_type{});             \
                else                                                                               \
                    launch_act_res_cmap(bias_tag, std::false_type{}, std::true_type{});            \
            } else {                                                                               \
                if (RES_BOOL)                                                                      \
                    launch_act_res_cmap(bias_tag, std::true_type{}, std::false_type{});            \
                else                                                                               \
                    launch_act_res_cmap(bias_tag, std::false_type{}, std::false_type{});           \
            }                                                                                      \
        };                                                                                         \
        if (BIAS_BOOL)                                                                             \
            launch_res_cmap(std::true_type{});                                                     \
        else                                                                                       \
            launch_res_cmap(std::false_type{});                                                    \
    } while (0)
template <int CLUSTER_SIZE, bool HAS_BIAS = true, bool APPLY_RELU = false,
          bool APPLY_SIGMOID = false, int VALID_TOKENS = 64, bool STREAMING_STORE = false,
          bool APPLY_SILU = false, bool PER_TILE_MASK = false, bool HAS_RESIDUAL = false,
          int ACT_ENUM = 0, bool HAS_COUT_MAP = false>
static inline void launch_k44_cuda_smlS_f16_forward(
    __half *d_Y, const __half *d_X, const uint32_t *d_cluster_indices_32bit,
    const uint4 *d_cluster_weights_mma, const __half *d_neuron_bias,
    const uint32_t *d_tile_offsets_X, const uint32_t *d_tile_offsets_Y, uint32_t num_batch_tiles,
    uint32_t num_output_neurons, uint32_t channel_stride_Y, uint32_t B_vec16,
    cudaStream_t stream = nullptr, const __half *d_residual_X = nullptr,
    const uint16_t *d_cout_map = nullptr) {
    constexpr int DEFAULT_WAVES = (CLUSTER_SIZE >= 64) ? 8 : 4;
    constexpr int SINGLE_WAVES = CLUSTER_SIZE / 16;
    if ((num_output_neurons >= (uint32_t)(DEFAULT_WAVES * 16)) &&
        (num_output_neurons % (uint32_t)(DEFAULT_WAVES * 16) == 0u)) {
        uint32_t neuron_blocks = num_output_neurons / (DEFAULT_WAVES * 16);
        dim3 grid(num_batch_tiles, neuron_blocks, 1);
        dim3 block(DEFAULT_WAVES * 32, 1, 1);
        k44_cuda_smlS_f16_kernel<CLUSTER_SIZE, DEFAULT_WAVES, HAS_BIAS, APPLY_RELU, APPLY_SIGMOID,
                                 VALID_TOKENS, STREAMING_STORE, APPLY_SILU, PER_TILE_MASK,
                                 HAS_RESIDUAL, static_cast<K44Act>(ACT_ENUM), HAS_COUT_MAP>
            <<<grid, block, 0, stream>>>(d_Y, d_X, d_cluster_indices_32bit, d_cluster_weights_mma,
                                         d_neuron_bias, d_tile_offsets_X, d_tile_offsets_Y,
                                         channel_stride_Y, B_vec16, d_residual_X, d_cout_map);
    } else {
        uint32_t neuron_blocks =
            (num_output_neurons + (SINGLE_WAVES * 16) - 1) / (SINGLE_WAVES * 16);
        dim3 grid(num_batch_tiles, neuron_blocks, 1);
        dim3 block(SINGLE_WAVES * 32, 1, 1);
        k44_cuda_smlS_f16_kernel<CLUSTER_SIZE, SINGLE_WAVES, HAS_BIAS, APPLY_RELU, APPLY_SIGMOID,
                                 VALID_TOKENS, STREAMING_STORE, APPLY_SILU, PER_TILE_MASK,
                                 HAS_RESIDUAL, static_cast<K44Act>(ACT_ENUM), HAS_COUT_MAP>
            <<<grid, block, 0, stream>>>(d_Y, d_X, d_cluster_indices_32bit, d_cluster_weights_mma,
                                         d_neuron_bias, d_tile_offsets_X, d_tile_offsets_Y,
                                         channel_stride_Y, B_vec16, d_residual_X, d_cout_map);
    }
}
template <int CLUSTER_SIZE, bool HAS_BIAS = true, bool APPLY_RELU = false,
          bool APPLY_SIGMOID = false, int VALID_TOKENS = 64, bool STREAMING_STORE = false,
          bool APPLY_SILU = false, bool PER_TILE_MASK = false, bool HAS_RESIDUAL = false,
          int ACT_ENUM = 0, bool HAS_COUT_MAP = false>
static inline void launch_k44_cuda_lrgS_f16_forward(
    __half *d_Y, const __half *d_X, const uint32_t *d_cluster_indices_32bit,
    const uint4 *d_cluster_weights_mma, const __half *d_neuron_bias,
    const uint32_t *d_tile_offsets_X, const uint32_t *d_tile_offsets_Y, uint32_t num_batch_tiles,
    uint32_t num_output_neurons, uint32_t channel_stride_Y, uint32_t B_vec16,
    cudaStream_t stream = nullptr, const __half *d_residual_X = nullptr,
    const uint16_t *d_cout_map = nullptr) {
    constexpr int WAVES = 8;
    uint32_t neuron_blocks = (num_output_neurons + (WAVES * 16) - 1) / (WAVES * 16);
    dim3 grid(num_batch_tiles, neuron_blocks, 1);
    dim3 block(WAVES * 32, 1, 1);
    k44_cuda_lrgS_f16_kernel<CLUSTER_SIZE, WAVES, HAS_BIAS, APPLY_RELU, APPLY_SIGMOID, VALID_TOKENS,
                             STREAMING_STORE, APPLY_SILU, PER_TILE_MASK, HAS_RESIDUAL,
                             static_cast<K44Act>(ACT_ENUM), HAS_COUT_MAP>
        <<<grid, block, 0, stream>>>(d_Y, d_X, d_cluster_indices_32bit, d_cluster_weights_mma,
                                     d_neuron_bias, d_tile_offsets_X, d_tile_offsets_Y,
                                     channel_stride_Y, B_vec16, d_residual_X, d_cout_map);
}
template <int CLUSTER_SIZE, bool HAS_BIAS = true, bool APPLY_RELU = false,
          bool APPLY_SIGMOID = false, int VALID_TOKENS = 64, bool STREAMING_STORE = false,
          bool APPLY_SILU = false, bool PER_TILE_MASK = false, bool HAS_RESIDUAL = false,
          int ACT_ENUM = 0, bool HAS_COUT_MAP = false>
static inline void
launch_k44_cuda_f16_forward(__half *d_Y, const __half *d_X, const uint32_t *d_cluster_indices_32bit,
                            const uint4 *d_cluster_weights_mma, const __half *d_neuron_bias,
                            const uint32_t *d_tile_offsets_X, const uint32_t *d_tile_offsets_Y,
                            uint32_t num_batch_tiles, uint32_t num_output_neurons,
                            uint32_t channel_stride_Y, uint32_t B_vec16,
                            cudaStream_t stream = nullptr, const __half *d_residual_X = nullptr,
                            const uint16_t *d_cout_map = nullptr) {
    if constexpr (CLUSTER_SIZE <= 64) {
        launch_k44_cuda_smlS_f16_forward<CLUSTER_SIZE, HAS_BIAS, APPLY_RELU, APPLY_SIGMOID,
                                         VALID_TOKENS, STREAMING_STORE, APPLY_SILU, PER_TILE_MASK,
                                         HAS_RESIDUAL, ACT_ENUM, HAS_COUT_MAP>(
            d_Y, d_X, d_cluster_indices_32bit, d_cluster_weights_mma, d_neuron_bias,
            d_tile_offsets_X, d_tile_offsets_Y, num_batch_tiles, num_output_neurons,
            channel_stride_Y, B_vec16, stream, d_residual_X, d_cout_map);
    } else {
        launch_k44_cuda_lrgS_f16_forward<CLUSTER_SIZE, HAS_BIAS, APPLY_RELU, APPLY_SIGMOID,
                                         VALID_TOKENS, STREAMING_STORE, APPLY_SILU, PER_TILE_MASK,
                                         HAS_RESIDUAL, ACT_ENUM, HAS_COUT_MAP>(
            d_Y, d_X, d_cluster_indices_32bit, d_cluster_weights_mma, d_neuron_bias,
            d_tile_offsets_X, d_tile_offsets_Y, num_batch_tiles, num_output_neurons,
            channel_stride_Y, B_vec16, stream, d_residual_X, d_cout_map);
    }
}
#define K44_DISPATCH_SMLS_EPILOGUE_EX(CLUSTER_SZ, MASK_BOOL, BIAS_BOOL, ACT_VAL, RES_BOOL,         \
                                      CMAP_BOOL, ...)                                              \
    do {                                                                                           \
        auto launch_act_res_cmap = [&](auto bias_tag, auto res_tag, auto cmap_tag) {               \
            constexpr bool HB = decltype(bias_tag)::value;                                         \
            constexpr bool HR = decltype(res_tag)::value;                                          \
            constexpr bool HC = decltype(cmap_tag)::value;                                         \
            switch (static_cast<int>(ACT_VAL)) {                                                   \
            case 1:                                                                                \
                launch_k44_cuda_f16_forward<CLUSTER_SZ, HB, true, false, 64, false, false,         \
                                            MASK_BOOL, HR, 1, HC>(__VA_ARGS__);                    \
                break;                                                                             \
            case 2:                                                                                \
                launch_k44_cuda_f16_forward<CLUSTER_SZ, HB, false, true, 64, false, false,         \
                                            MASK_BOOL, HR, 2, HC>(__VA_ARGS__);                    \
                break;                                                                             \
            case 3:                                                                                \
                launch_k44_cuda_f16_forward<CLUSTER_SZ, HB, false, false, 64, false, true,         \
                                            MASK_BOOL, HR, 3, HC>(__VA_ARGS__);                    \
                break;                                                                             \
            case 4:                                                                                \
                launch_k44_cuda_f16_forward<CLUSTER_SZ, HB, false, false, 64, false, false,        \
                                            MASK_BOOL, HR, 4, HC>(__VA_ARGS__);                    \
                break;                                                                             \
            default:                                                                               \
                launch_k44_cuda_f16_forward<CLUSTER_SZ, HB, false, false, 64, false, false,        \
                                            MASK_BOOL, HR, 0, HC>(__VA_ARGS__);                    \
                break;                                                                             \
            }                                                                                      \
        };                                                                                         \
        auto launch_res_cmap = [&](auto bias_tag) {                                                \
            if (CMAP_BOOL) {                                                                       \
                if (RES_BOOL)                                                                      \
                    launch_act_res_cmap(bias_tag, std::true_type{}, std::true_type{});             \
                else                                                                               \
                    launch_act_res_cmap(bias_tag, std::false_type{}, std::true_type{});            \
            } else {                                                                               \
                if (RES_BOOL)                                                                      \
                    launch_act_res_cmap(bias_tag, std::true_type{}, std::false_type{});            \
                else                                                                               \
                    launch_act_res_cmap(bias_tag, std::false_type{}, std::false_type{});           \
            }                                                                                      \
        };                                                                                         \
        if (BIAS_BOOL)                                                                             \
            launch_res_cmap(std::true_type{});                                                     \
        else                                                                                       \
            launch_res_cmap(std::false_type{});                                                    \
    } while (0)
#include "../common/kernel44_valu_pack.cuh"
