#pragma once
// ==============================================================================
// kernel44_dual_buffer.cuh — Auxiliary Diagnostic Kernel (Dual-V# Workaround)
// Target: AMD Radeon RX 9060 XT (Navi 44 / GFX1200 / RDNA4)
//
// DISCLAIMER:
//   This file is an AUXILIARY diagnostic kernel created solely to evaluate a
//   single hardware special case in Section 4.1 (Figure 4) of the whitepaper.
//   It is NOT the primary production kernel (see `kernel44_immutable.cuh`).
//
// Architectural Background (33.55M < Cin <= 67.1M Neurons):
//   Standard RDNA4 MUBUF instructions (`buffer_load_*`) use a 32-bit unsigned
//   byte offset (`voffset`) per 128-bit Buffer Resource Descriptor (`V#`),
//   limiting a single descriptor to 4 GiB = 2^25 Token64 input neurons
//   (33,554,432 neurons). In the primary kernel (`kernel44_immutable.cuh`),
//   exceeding 33.55M neurons automatically switches to 64-bit FLAT/GLOBAL
//   pointers (`global_load_*`), which supports arbitrary 64-bit address spaces
//   cleanly but bypasses the Texture Address Unit (TA) buffer path.
//
//   To prove empirically that the throughput step at Cin = 33.55M in Figure 4
//   stems from the 32-bit MUBUF descriptor boundary rather than VRAM bandwidth
//   exhaustion, this auxiliary variant pairs two adjacent 4 GiB descriptors
//   (`rsrc_X0` at base_X + 0, `rsrc_X1` at base_X + 4 GiB) configured with
//   hardware out-of-bounds suppression (`OOB_SELECT = 3`, `num_records =
//   0xFFFFFFFC`). Bit 25 of the neuron index forces the inactive descriptor's
//   `voffset` to `0xFFFFFFFF` (Out-Of-Bounds), causing the RDNA4 TA unit to
//   discard the unselected load in hardware. This extends MUBUF addressing up
//   to 2^26 = 67,108,864 neurons (8 GiB per 64-batch tile) at the cost of a
//   second 128-bit SGPR descriptor and doubled MUBUF instruction issue slots.
// ==============================================================================

#include "kernel44_immutable.cuh"

// --- 1. Dual-V# Hardware OOB-Select Gather Subroutine (up to 64 Mi Neurons) ---
template <int GATHER_SUBTILES, typename GatherElem>
__device__ __forceinline__ void prefetch_gathers_dual_vsharp_mubuf(
    GatherElem (&gv)[8], GatherElem (&gv_hi)[8], const uint4_v (&c_buf)[2],
    uint32_t m_off, __amdgpu_buffer_rsrc_t rsrc_X0, __amdgpu_buffer_rsrc_t rsrc_X1
) {
    const uint32_t* cols = reinterpret_cast<const uint32_t*>(c_buf);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t c_lo = cols[2 * p], c_hi = cols[2 * p + 1];
        const uint32_t base_lo = (c_lo << 7) | m_off, base_hi = (c_hi << 7) | m_off;
        uint32_t mask_lo, mask_hi;
        // Sign-extend bit 25 of col: 0x00000000 if col < 2^25, 0xFFFFFFFF if col >= 2^25
        asm("v_bfe_i32 %0, %2, 25, 1\n\tv_bfe_i32 %1, %3, 25, 1"
            : "=v"(mask_lo), "=v"(mask_hi) : "v"(c_lo), "v"(c_hi));
        const uint32_t off0_lo = base_lo | mask_lo,  off0_hi = base_hi | mask_hi;
        const uint32_t off1_lo = base_lo | ~mask_lo, off1_hi = base_hi | ~mask_hi;
        if constexpr (GATHER_SUBTILES == 1) {
            asm volatile(
                "buffer_load_u16 %0, %4, %8, null offen\n\tbuffer_load_u16 %1, %5, %8, null offen\n\t"
                "buffer_load_u16 %2, %6, %9, null offen\n\tbuffer_load_u16 %3, %7, %9, null offen"
                : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]), "=&v"(gv_hi[2 * p]), "=&v"(gv_hi[2 * p + 1])
                : "v"(off0_lo), "v"(off0_hi), "v"(off1_lo), "v"(off1_hi), "s"(rsrc_X0), "s"(rsrc_X1) : "memory");
        } else if constexpr (GATHER_SUBTILES == 2) {
            asm volatile(
                "buffer_load_b32 %0, %4, %8, null offen\n\tbuffer_load_b32 %1, %5, %8, null offen\n\t"
                "buffer_load_b32 %2, %6, %9, null offen\n\tbuffer_load_b32 %3, %7, %9, null offen"
                : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]), "=&v"(gv_hi[2 * p]), "=&v"(gv_hi[2 * p + 1])
                : "v"(off0_lo), "v"(off0_hi), "v"(off1_lo), "v"(off1_hi), "s"(rsrc_X0), "s"(rsrc_X1) : "memory");
        } else {
            asm volatile(
                "buffer_load_b64 %0, %4, %8, null offen\n\tbuffer_load_b64 %1, %5, %8, null offen\n\t"
                "buffer_load_b64 %2, %6, %9, null offen\n\tbuffer_load_b64 %3, %7, %9, null offen"
                : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]), "=&v"(gv_hi[2 * p]), "=&v"(gv_hi[2 * p + 1])
                : "v"(off0_lo), "v"(off0_hi), "v"(off1_lo), "v"(off1_hi), "s"(rsrc_X0), "s"(rsrc_X1) : "memory");
        }
    }
}

// --- 2. Kernel 44 Dual-Buffer Pipeline ----------------------------------------
template <int CLUSTER_SIZE, int WAVES_PER_WG>
struct K44DualBufferPipeline {
    static constexpr int WAVES_PER_CLUSTER = (CLUSTER_SIZE / 16 < WAVES_PER_WG) ? (CLUSTER_SIZE / 16) : WAVES_PER_WG;
    static constexpr int CLUSTERS_PER_WG   = WAVES_PER_WG / WAVES_PER_CLUSTER;
    static constexpr int K_CHUNK           = (CLUSTER_SIZE >= 32) ? 2 : 1;
    static constexpr int GATHER_SUBTILES   = (4 * K_CHUNK) / WAVES_PER_CLUSTER;
    static constexpr bool USE_LDS          = (WAVES_PER_CLUSTER > 1);
    using GatherElem = typename std::conditional<GATHER_SUBTILES == 4, uint2, uint32_t>::type;
    using LDSArray   = half8[2][USE_LDS ? CLUSTERS_PER_WG : 1][K_CHUNK][4][32];

    uint32_t lane, k_sub, sub_tile, cluster_in_wg, lane_idx_off, lane_w_off, m_off, j_base0;
    const char *base_idx_ptr, *base_w0_ptr, *base_bias_ptr;
    __amdgpu_buffer_rsrc_t rsrc_X0, rsrc_X1;
    Token64* tile_Y;
    uint4_v c_buf_next[2];
    half8 w_buf0, w_buf1, b_reg0, b_reg1, b_reg2, b_reg3;
    GatherElem gv[8], gv_hi[8];
    float8 c0, c1, c2, c3;

    __device__ __forceinline__ uint32_t c_slot() const { return (CLUSTERS_PER_WG > 1) ? cluster_in_wg : 0; }
    __device__ __forceinline__ void load_idx(uint4_v (&c_buf)[2]) {
        load_global_b128_sgpr(c_buf[0], base_idx_ptr, lane_idx_off + 0);
        load_global_b128_sgpr(c_buf[1], base_idx_ptr, lane_idx_off + 16);
        base_idx_ptr += 64 * K_CHUNK;
    }
    __device__ __forceinline__ void load_w(half8& w_buf) {
        load_global_b128_sgpr(w_buf, base_w0_ptr, lane_w_off);
        base_w0_ptr += 32 * sizeof(half8);
    }
    __device__ __forceinline__ void prefetch_gathers(const uint4_v (&c_buf)[2]) {
        prefetch_gathers_dual_vsharp_mubuf<GATHER_SUBTILES, GatherElem>(gv, gv_hi, c_buf, m_off, rsrc_X0, rsrc_X1);
    }

    template <int WAIT_V, bool HAS_NEXT_IDX> __device__ __forceinline__ void pack_and_sync_stage(uint32_t dst_stage, const half8& w_dep, half8& w_tail, LDSArray& s_b_frag) {
        half8 b_dep;
        if constexpr (WAIT_V == 3) asm volatile("s_wait_loadcnt 0x3" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]), "+v"(gv_hi[0]), "+v"(gv_hi[1]), "+v"(gv_hi[2]), "+v"(gv_hi[3]), "+v"(gv_hi[4]), "+v"(gv_hi[5]), "+v"(gv_hi[6]), "+v"(gv_hi[7]) : "v"(w_dep), "v"(c0[0]) : "memory");
        else if constexpr (WAIT_V == 2) asm volatile("s_wait_loadcnt 0x2" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]), "+v"(gv_hi[0]), "+v"(gv_hi[1]), "+v"(gv_hi[2]), "+v"(gv_hi[3]), "+v"(gv_hi[4]), "+v"(gv_hi[5]), "+v"(gv_hi[6]), "+v"(gv_hi[7]) : "v"(w_dep), "v"(c0[0]) : "memory");
        else if constexpr (WAIT_V == 1) asm volatile("s_wait_loadcnt 0x1" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]), "+v"(gv_hi[0]), "+v"(gv_hi[1]), "+v"(gv_hi[2]), "+v"(gv_hi[3]), "+v"(gv_hi[4]), "+v"(gv_hi[5]), "+v"(gv_hi[6]), "+v"(gv_hi[7]) : "v"(w_dep), "v"(c0[0]) : "memory");
        else asm volatile("s_wait_loadcnt 0x0" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]), "+v"(gv_hi[0]), "+v"(gv_hi[1]), "+v"(gv_hi[2]), "+v"(gv_hi[3]), "+v"(gv_hi[4]), "+v"(gv_hi[5]), "+v"(gv_hi[6]), "+v"(gv_hi[7]) : "v"(w_dep), "v"(c0[0]) : "memory");

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if constexpr (GATHER_SUBTILES == 4) {
                gv[i].x |= gv_hi[i].x;
                gv[i].y |= gv_hi[i].y;
            } else {
                gv[i] |= gv_hi[i];
            }
        }

        if constexpr (GATHER_SUBTILES == 1) {
            uint32_t* p0 = reinterpret_cast<uint32_t*>(&b_dep);
            #pragma unroll
            for (int p = 0; p < 4; ++p) asm("v_pack_b32_f16 %0, %1, %2" : "=v"(p0[p]) : "v"(gv[2 * p]), "v"(gv[2 * p + 1]));
            s_b_frag[dst_stage][c_slot()][k_sub][sub_tile][lane] = b_dep;
        } else if constexpr (GATHER_SUBTILES == 2) {
            half8 b0;
            uint32_t *p0 = (uint32_t*)&b0, *p1 = reinterpret_cast<uint32_t*>(&b_dep);
            #pragma unroll
            for (int p = 0; p < 4; ++p) pack_pair_f16(p0[p], p1[p], gv[2 * p], gv[2 * p + 1]);
            s_b_frag[dst_stage][c_slot()][k_sub][sub_tile + 0][lane] = b0;
            s_b_frag[dst_stage][c_slot()][k_sub][sub_tile + 1][lane] = b_dep;
        } else {
            uint32_t *p0 = (uint32_t*)&b_reg0, *p1 = (uint32_t*)&b_reg1, *p2 = (uint32_t*)&b_reg2, *p3 = (uint32_t*)&b_dep;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                pack_pair_f16(p0[p], p1[p], gv[2 * p].x, gv[2 * p + 1].x);
                pack_pair_f16(p2[p], p3[p], gv[2 * p].y, gv[2 * p + 1].y);
            }
            if constexpr (USE_LDS) {
                auto& lds_sub = s_b_frag[dst_stage][c_slot()][k_sub];
                lds_sub[0][lane] = b_reg0; lds_sub[1][lane] = b_reg1; lds_sub[2][lane] = b_reg2; lds_sub[3][lane] = b_dep;
            } else b_reg3 = b_dep;
        }
        if constexpr (HAS_NEXT_IDX) asm volatile("s_wait_loadcnt 0x0" : "+v"(c_buf_next[0]), "+v"(c_buf_next[1]), "+v"(w_tail) : "v"(b_dep) : "memory");
        else asm volatile("s_wait_loadcnt 0x0" : "+v"(w_tail) : "v"(b_dep) : "memory");
        if constexpr (USE_LDS) __syncthreads();
    }

    template <int WAIT_W1, bool PREFETCH_W0> __device__ __forceinline__ void compute_lds_2k(uint32_t cur, LDSArray& s_b_frag) {
        half8 b0, b1, b2, b3;
        read_lds_b_tile_4x(b0, b1, b2, b3, s_b_frag[cur][c_slot()][0], lane);
        __builtin_amdgcn_sched_barrier(0);
        compute_wmma_4x_f16(w_buf0, b0, b1, b2, b3, c0, c1, c2, c3);
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (PREFETCH_W0) load_w(w_buf0);
        read_lds_b_tile_4x(b0, b1, b2, b3, s_b_frag[cur][c_slot()][1], lane);
        if constexpr (WAIT_W1 == 11) asm volatile("s_wait_loadcnt 0x13" : "+v"(w_buf1) : "v"(c0[0]), "v"(w_buf0), "v"(b0) : "memory");
        else if constexpr (WAIT_W1 == 9) asm volatile("s_wait_loadcnt 0x11" : "+v"(w_buf1) : "v"(c0[0]), "v"(w_buf0), "v"(b0) : "memory");
        else asm volatile("s_wait_loadcnt 0x0" : "+v"(w_buf1) : "v"(c0[0]), "v"(b0) : "memory");
        __builtin_amdgcn_sched_barrier(0);
        compute_wmma_4x_f16(w_buf1, b0, b1, b2, b3, c0, c1, c2, c3);
    }

    template <bool HAS_BIAS> __device__ __forceinline__ void init_and_prime(
        __half* __restrict__ Y, const __half* __restrict__ X, const uint32_t* __restrict__ cluster_indices_32bit,
        const uint4* __restrict__ cluster_weights_wmma, const __half* __restrict__ neuron_bias, uint32_t Cin, uint32_t Cout, uint32_t B_vec16, LDSArray& s_b_frag
    ) {
        lane = threadIdx.x & 31;
        const uint32_t wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
        const uint32_t wg_idx  = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const uint32_t batch_b = __builtin_amdgcn_readfirstlane(blockIdx.x);
        constexpr uint32_t WAVES_PER_K = 4 / GATHER_SUBTILES;
        const uint32_t wave_in_clust = wave_id % WAVES_PER_CLUSTER, nb_w0 = wg_idx * WAVES_PER_WG + wave_id;
        k_sub         = (K_CHUNK == 2) ? (wave_in_clust / WAVES_PER_K) : 0;
        sub_tile      = (wave_in_clust % WAVES_PER_K) * GATHER_SUBTILES;
        cluster_in_wg = wave_id / WAVES_PER_CLUSTER;
        lane_idx_off  = (lane & 16) << 1; lane_w_off = lane << 4; m_off = (lane & 15) << 3;
        j_base0       = (nb_w0 << 4) + ((lane & 16) >> 1);
        base_bias_ptr = to_sgpr_ptr(neuron_bias);
        base_idx_ptr  = to_sgpr_ptr(reinterpret_cast<const char*>(cluster_indices_32bit) + ((nb_w0 / (CLUSTER_SIZE / 16)) * B_vec16 + k_sub) * 64);
        base_w0_ptr   = to_sgpr_ptr(reinterpret_cast<const half8*>(cluster_weights_wmma) + ((nb_w0 * B_vec16) << 5));
        const char* base_X = reinterpret_cast<const char*>(X) + (size_t)batch_b * ((size_t)Cin << 7) + (sub_tile << 1);
        rsrc_X0 = __builtin_amdgcn_make_buffer_rsrc((void*)base_X, 0, 0xfffffffc, 0x31004000);
        rsrc_X1 = __builtin_amdgcn_make_buffer_rsrc((void*)(base_X + (1ULL << 32)), 0, 0xfffffffc, 0x31004000);
        tile_Y = reinterpret_cast<Token64*>(Y) + (size_t)batch_b * Cout;

        uint4_v c_buf[2];
        load_idx(c_buf);
        if constexpr (HAS_BIAS) load_global_b128_sgpr(w_buf0, base_bias_ptr, j_base0 * sizeof(__half));
        asm volatile("s_wait_loadcnt 0x0" : "+v"(c_buf[0]), "+v"(c_buf[1]), "+v"(w_buf0) :: "memory");
        prefetch_gathers(c_buf);
        load_idx(c_buf_next);
        if constexpr (HAS_BIAS) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) c0[i] = c1[i] = c2[i] = c3[i] = static_cast<float>(w_buf0[i]);
        } else c0 = c1 = c2 = c3 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        load_w(w_buf0);
        pack_and_sync_stage<3, true>(0, w_buf0, w_buf0, s_b_frag);
    }

    template <int WAIT_W1, int WAIT_V, bool PREFETCH_NEXT_IDX> __device__ __forceinline__ void step(uint32_t s, LDSArray& s_b_frag) {
        load_w(w_buf1);
        prefetch_gathers(c_buf_next);
        if constexpr (PREFETCH_NEXT_IDX) load_idx(c_buf_next);
        if constexpr (USE_LDS) {
            compute_lds_2k<WAIT_W1, true>(s & 1, s_b_frag);
            pack_and_sync_stage<WAIT_V, PREFETCH_NEXT_IDX>((s + 1) & 1, w_buf1, w_buf0, s_b_frag);
        } else {
            __builtin_amdgcn_sched_barrier(0);
            compute_wmma_4x_f16(w_buf0, b_reg0, b_reg1, b_reg2, b_reg3, c0, c1, c2, c3);
            pack_and_sync_stage<WAIT_V, PREFETCH_NEXT_IDX>(0, w_buf1, w_buf1, s_b_frag);
            w_buf0 = w_buf1;
        }
    }
};

template <int CLUSTER_SIZE = 128, int WAVES_PER_WG = (CLUSTER_SIZE >= 64 ? 8 : 4), bool HAS_BIAS = true, bool STREAMING_STORE = false>
__launch_bounds__(WAVES_PER_WG * 32, 2)
__attribute__((amdgpu_flat_work_group_size(WAVES_PER_WG * 32, WAVES_PER_WG * 32)))
__global__ void connectome_k44_dual_buffer_lds_f16_kernel(
    __half* __restrict__ Y, const __half* __restrict__ X, const uint32_t* __restrict__ cluster_indices_32bit,
    const uint4* __restrict__ cluster_weights_wmma, const __half* __restrict__ neuron_bias, uint32_t Cin, uint32_t Cout, uint32_t B_vec16
) {
    using Pipe = K44DualBufferPipeline<CLUSTER_SIZE, WAVES_PER_WG>;
    __shared__ typename Pipe::LDSArray s_b_frag;
    Pipe p;
    p.template init_and_prime<HAS_BIAS>(Y, X, cluster_indices_32bit, cluster_weights_wmma, neuron_bias, Cin, Cout, B_vec16, s_b_frag);
    const uint32_t num_steps = B_vec16 / Pipe::K_CHUNK;
    for (uint32_t s = 0; s < num_steps - 2; ++s) p.template step<11, (Pipe::USE_LDS ? 3 : 2), true>(s, s_b_frag);
    p.template step<9, (Pipe::USE_LDS ? 1 : 0), false>(num_steps - 2, s_b_frag);
    if constexpr (Pipe::USE_LDS) {
        p.load_w(p.w_buf1);
        p.template compute_lds_2k<0, false>((num_steps - 1) & 1, s_b_frag);
    } else compute_wmma_4x_f16(p.w_buf0, p.b_reg0, p.b_reg1, p.b_reg2, p.b_reg3, p.c0, p.c1, p.c2, p.c3);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint2 y_vec = make_uint2(cvt_pk_f16_f32(p.c0[i], p.c1[i]), cvt_pk_f16_f32(p.c2[i], p.c3[i]));
        const void* ptr = &p.tile_Y[p.j_base0 + i][(p.lane & 15) * 4];
        if constexpr (STREAMING_STORE) asm volatile("global_store_b64 %0, %1, off th:TH_STORE_NT" :: "v"(ptr), "v"(y_vec) : "memory");
        else *reinterpret_cast<uint2*>(const_cast<void*>(ptr)) = y_vec;
    }
}

template <int CLUSTER_SIZE = 128, bool STREAMING_STORE = false>
inline void launch_k44_dual_buffer_f16_forward(
    __half* Y, const __half* X, const uint32_t* cluster_indices_32bit, const uint4* cluster_weights_wmma,
    const __half* neuron_bias, uint32_t batch_size, uint32_t Cin, uint32_t Cout, uint32_t B_vec16, hipStream_t stream = nullptr
) {
    constexpr int WAVES_PER_WG = (CLUSTER_SIZE >= 64) ? 8 : 4;
    const uint32_t total_batch_tiles = (batch_size + 63) / 64, num_neuron_blocks = (Cout + WAVES_PER_WG * 16 - 1) / (WAVES_PER_WG * 16);
    dim3 grid(total_batch_tiles, num_neuron_blocks, 1);
    auto kernel = connectome_k44_dual_buffer_lds_f16_kernel<CLUSTER_SIZE, WAVES_PER_WG, true, STREAMING_STORE>;
    hipLaunchKernelGGL(kernel, grid, dim3(WAVES_PER_WG * 32, 1, 1), 0, stream, Y, X, cluster_indices_32bit, cluster_weights_wmma, neuron_bias, Cin, Cout, B_vec16);
}
