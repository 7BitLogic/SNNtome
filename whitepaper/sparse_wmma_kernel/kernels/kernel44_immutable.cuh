#pragma once
// ==============================================================================
// kernel44_immutable.cuh — Kernel 44 FP16 (Ultra-Sparse Connectome Engine)
// Target: AMD Radeon RX 9060 XT (Navi 44 / GFX1200 / RDNA4, HIP_VISIBLE_DEVICES=1)
// Unified Cooperative LDS B-Tile Broadcast & Ping-Pong Weight Pipeline across S:
//   - S in {128, 256}: 8 Waves/WG, 2 K-Steps/Barrier, u16 Gather (8 KB LDS)
//   - S == 64:         8 Waves/WG, 2 K-Steps/Barrier, b32 Gather (16 KB LDS)
//   - S == 32:         4 Waves/WG, 2 K-Steps/Barrier, b64 Gather (16 KB LDS)
//   - S == 16:         4 Waves/WG, 1 K-Step/Stage,    b64 Gather (0 KB LDS, Reg)
// ==============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <type_traits>

#if !defined(HALF8_FLOAT8_DEFINED)
#define HALF8_FLOAT8_DEFINED
typedef __fp16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef unsigned int uint4_v __attribute__((ext_vector_type(4)));
#endif
#if !defined(TOKEN64_DEFINED)
#define TOKEN64_DEFINED
typedef __fp16 Token64[64];
#endif

// --- 1. Hardware Utility Functions & Inline Assembly (GFX1200 / RDNA4) --------
__device__ __forceinline__ const char* to_sgpr_ptr(const void* ptr) {
    uintptr_t u = reinterpret_cast<uintptr_t>(ptr);
    uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u));
    uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u >> 32));
    return reinterpret_cast<const char*>((static_cast<uintptr_t>(hi) << 32) | lo);
}

template <typename T> __device__ __forceinline__ void load_global_b128_sgpr(T& dst, const char* base_sgpr, uint32_t offset_vgpr) {
    asm volatile("global_load_b128 %0, %1, %2" : "=v"(dst) : "v"(offset_vgpr), "s"(base_sgpr) : "memory");
}

__device__ __forceinline__ float8 __organic_wmma_f32_16x16x16_f16(half8 a, half8 b, float8 c) {
    float8 d = c;
#if defined(__gfx1200__)
    asm("v_wmma_f32_16x16x16_f16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
#endif
    return d;
}

__device__ __forceinline__ uint32_t cvt_pk_f16_f32(float a, float b) {
    uint32_t res;
    asm("v_cvt_pk_rtz_f16_f32 %0, %1, %2" : "=v"(res) : "v"(a), "v"(b));
    return res;
}

__device__ __forceinline__ void compute_wmma_4x_f16(half8 w, half8 b0, half8 b1, half8 b2, half8 b3, float8& c0, float8& c1, float8& c2, float8& c3) {
    c0 = __organic_wmma_f32_16x16x16_f16(w, b0, c0); c1 = __organic_wmma_f32_16x16x16_f16(w, b1, c1);
    c2 = __organic_wmma_f32_16x16x16_f16(w, b2, c2); c3 = __organic_wmma_f32_16x16x16_f16(w, b3, c3);
}

__device__ __forceinline__ void pack_pair_f16(uint32_t& d0, uint32_t& d1, uint32_t lo, uint32_t hi) {
    asm("v_pack_b32_f16 %0, %1, %2"         : "=v"(d0) : "v"(lo), "v"(hi));
    asm("v_perm_b32 %0, %1, %2, 0x07060302" : "=v"(d1) : "v"(hi), "v"(lo));
}

__device__ __forceinline__ void read_lds_b_tile_4x(half8& b0, half8& b1, half8& b2, half8& b3, const half8 (&lds_sub)[4][32], uint32_t lane) {
    b0 = lds_sub[0][lane]; b1 = lds_sub[1][lane]; b2 = lds_sub[2][lane]; b3 = lds_sub[3][lane];
    asm volatile("" : "+v"(b0), "+v"(b1), "+v"(b2), "+v"(b3) :: "memory");
}

// --- 1b. Token Gather Addressing Subroutines (32-bit MUBUF vs 64-bit GLOBAL) --
// Variant A: 32-bit Hardware Buffer Descriptor (MUBUF, <= 4 GiB / 2^25 = 33.55M neurons per 64-batch tile)
template <int GATHER_SUBTILES, typename GatherElem>
__device__ __forceinline__ void prefetch_gathers_32bit_mubuf(
    GatherElem (&gv)[8], const uint4_v (&c_buf)[2], uint32_t m_off, __amdgpu_buffer_rsrc_t rsrc_X
) {
    const uint32_t* cols = reinterpret_cast<const uint32_t*>(c_buf);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t lo = (cols[2 * p] << 7) | m_off, hi = (cols[2 * p + 1] << 7) | m_off;
        if constexpr (GATHER_SUBTILES == 1) asm volatile("buffer_load_u16 %0, %2, %4, null offen\n\tbuffer_load_u16 %1, %3, %4, null offen" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(lo), "v"(hi), "s"(rsrc_X) : "memory");
        else if constexpr (GATHER_SUBTILES == 2) asm volatile("buffer_load_b32 %0, %2, %4, null offen\n\tbuffer_load_b32 %1, %3, %4, null offen" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(lo), "v"(hi), "s"(rsrc_X) : "memory");
        else asm volatile("buffer_load_b64 %0, %2, %4, null offen\n\tbuffer_load_b64 %1, %3, %4, null offen" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(lo), "v"(hi), "s"(rsrc_X) : "memory");
    }
}

// Variant B: 64-bit Global Pointer via v_mad_co_u64_u32 (FLAT/GLOBAL, > 4 GiB / > 33.55M neurons per 64-batch tile)
template <int GATHER_SUBTILES, typename GatherElem>
__device__ __forceinline__ void prefetch_gathers_64bit_global(
    GatherElem (&gv)[8], const uint4_v (&c_buf)[2], uint64_t lane_base_X
) {
    const uint32_t* cols = reinterpret_cast<const uint32_t*>(c_buf);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint64_t a0, a1;
        asm("v_mad_co_u64_u32 %0, null, %2, 0x80, %4\n\t"
            "v_mad_co_u64_u32 %1, null, %3, 0x80, %4"
            : "=&v"(a0), "=&v"(a1)
            : "v"(cols[2 * p]), "v"(cols[2 * p + 1]), "v"(lane_base_X));
        if constexpr (GATHER_SUBTILES == 1) asm volatile("global_load_u16 %0, %2, off\n\tglobal_load_u16 %1, %3, off" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(a0), "v"(a1) : "memory");
        else if constexpr (GATHER_SUBTILES == 2) asm volatile("global_load_b32 %0, %2, off\n\tglobal_load_b32 %1, %3, off" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(a0), "v"(a1) : "memory");
        else asm volatile("global_load_b64 %0, %2, off\n\tglobal_load_b64 %1, %3, off" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(a0), "v"(a1) : "memory");
    }
}

// --- 2. Kernel 44 FP16: Unified Cooperative LDS Pipeline ----------------------
template <int CLUSTER_SIZE, int WAVES_PER_WG, bool ADDR_64BIT = false>
struct K44Pipeline {
    static constexpr int WAVES_PER_CLUSTER = (CLUSTER_SIZE / 16 < WAVES_PER_WG) ? (CLUSTER_SIZE / 16) : WAVES_PER_WG;
    static constexpr int CLUSTERS_PER_WG   = WAVES_PER_WG / WAVES_PER_CLUSTER;
    static constexpr int K_CHUNK           = (CLUSTER_SIZE >= 32) ? 2 : 1;
    static constexpr int GATHER_SUBTILES   = (4 * K_CHUNK) / WAVES_PER_CLUSTER;
    static constexpr bool USE_LDS          = (WAVES_PER_CLUSTER > 1);
    using GatherElem = typename std::conditional<GATHER_SUBTILES == 4, uint2, uint32_t>::type;
    using LDSArray   = half8[2][USE_LDS ? CLUSTERS_PER_WG : 1][K_CHUNK][4][32];

    uint32_t lane, k_sub, sub_tile, cluster_in_wg, lane_idx_off, lane_w_off, m_off, j_base0;
    const char *base_idx_ptr, *base_w0_ptr, *base_bias_ptr;
    typename std::conditional<ADDR_64BIT, uint64_t, __amdgpu_buffer_rsrc_t>::type addr_X;
    Token64* tile_Y;
    uint4_v c_buf_next[2];
    half8 w_buf0, w_buf1, b_reg0, b_reg1, b_reg2, b_reg3;
    GatherElem gv[8]; float8 c0, c1, c2, c3;

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
        if constexpr (ADDR_64BIT) prefetch_gathers_64bit_global<GATHER_SUBTILES, GatherElem>(gv, c_buf, addr_X);
        else prefetch_gathers_32bit_mubuf<GATHER_SUBTILES, GatherElem>(gv, c_buf, m_off, addr_X);
    }

    template <int WAIT_V, bool HAS_NEXT_IDX> __device__ __forceinline__ void pack_and_sync_stage(uint32_t dst_stage, const half8& w_dep, half8& w_tail, LDSArray& s_b_frag) {
        half8 b_dep;
        if constexpr (WAIT_V == 3) asm volatile("s_wait_loadcnt 0x3" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]) : "v"(w_dep), "v"(c0[0]) : "memory");
        else if constexpr (WAIT_V == 2) asm volatile("s_wait_loadcnt 0x2" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]) : "v"(w_dep), "v"(c0[0]) : "memory");
        else if constexpr (WAIT_V == 1) asm volatile("s_wait_loadcnt 0x1" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]) : "v"(w_dep), "v"(c0[0]) : "memory");
        else asm volatile("s_wait_loadcnt 0x0" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]) : "v"(w_dep), "v"(c0[0]) : "memory");

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
        if constexpr (WAIT_W1 == 11) asm volatile("s_wait_loadcnt 0xb" : "+v"(w_buf1) : "v"(c0[0]), "v"(w_buf0), "v"(b0) : "memory");
        else if constexpr (WAIT_W1 == 9) asm volatile("s_wait_loadcnt 0x9" : "+v"(w_buf1) : "v"(c0[0]), "v"(w_buf0), "v"(b0) : "memory");
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
        if constexpr (ADDR_64BIT) addr_X = reinterpret_cast<uint64_t>(base_X) + m_off;
        else addr_X = __builtin_amdgcn_make_buffer_rsrc((void*)base_X, 0, 0xffffffff, 0x31004000);
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

template <int CLUSTER_SIZE = 128, int WAVES_PER_WG = (CLUSTER_SIZE >= 64 ? 8 : 4), bool HAS_BIAS = true, bool STREAMING_STORE = false, bool ADDR_64BIT = false>
__launch_bounds__(WAVES_PER_WG * 32, 2)
__attribute__((amdgpu_flat_work_group_size(WAVES_PER_WG * 32, WAVES_PER_WG * 32)))
__global__ void connectome_k44_lds_f16_kernel(
    __half* __restrict__ Y, const __half* __restrict__ X, const uint32_t* __restrict__ cluster_indices_32bit,
    const uint4* __restrict__ cluster_weights_wmma, const __half* __restrict__ neuron_bias, uint32_t Cin, uint32_t Cout, uint32_t B_vec16
) {
    using Pipe = K44Pipeline<CLUSTER_SIZE, WAVES_PER_WG, ADDR_64BIT>;
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

// --- 3. Weight Interleaving, Initialization & Host Launchers ------------------
__global__ void interleave_weights_cluster16_wmma_gpu_kernel(uint4* __restrict__ dst_wmma, const uint4* __restrict__ src_compact, uint32_t num_output_neurons, uint32_t B_vec8) {
    uint32_t B_vec16 = B_vec8 / 2, total_elements = (num_output_neurons / 16) * B_vec16 * 32, global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= total_elements) return;
    uint32_t tile_idx = global_idx / 32, lane = global_idx % 32;
    dst_wmma[global_idx] = src_compact[((tile_idx / B_vec16) * 16 + (lane & 15)) * B_vec8 + (2 * (tile_idx % B_vec16) + (lane >> 4))];
}

__global__ void init_x_gpu_kernel(uint4* __restrict__ X_u4, size_t num_u4) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < num_u4; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t m = (uint32_t)(i & 15);
        X_u4[i] = make_uint4(0x2c002c00 + m, 0x30003000 + m, 0x34003400 + m, 0x38003800 + m);
    }
}

template <int CLUSTER_SIZE = 128, bool STREAMING_STORE = false>
inline void launch_k44_f16_forward(
    __half* Y, const __half* X, const uint32_t* cluster_indices_32bit, const uint4* cluster_weights_wmma,
    const __half* neuron_bias, uint32_t batch_size, uint32_t Cin, uint32_t Cout, uint32_t B_vec16, hipStream_t stream = nullptr
) {
    constexpr int WAVES_PER_WG = (CLUSTER_SIZE >= 64) ? 8 : 4;
    const uint32_t total_batch_tiles = (batch_size + 63) / 64, num_neuron_blocks = (Cout + WAVES_PER_WG * 16 - 1) / (WAVES_PER_WG * 16);
    const bool use_64bit = (Cin > (1u << 25)); // Switch > 33,554,432 neurons (4 GiB per 64-batch tile)
    dim3 grid(total_batch_tiles, num_neuron_blocks, 1);
    auto kernel = use_64bit
        ? connectome_k44_lds_f16_kernel<CLUSTER_SIZE, WAVES_PER_WG, true, STREAMING_STORE, true>
        : connectome_k44_lds_f16_kernel<CLUSTER_SIZE, WAVES_PER_WG, true, STREAMING_STORE, false>;
    hipLaunchKernelGGL(kernel, grid, dim3(WAVES_PER_WG * 32, 1, 1), 0, stream, Y, X, cluster_indices_32bit, cluster_weights_wmma, neuron_bias, Cin, Cout, B_vec16);
}
