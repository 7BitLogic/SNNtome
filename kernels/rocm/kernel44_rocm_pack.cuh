#pragma once
// ==============================================================================
// kernel44_rocm_pack.cuh — Kernel 44 FP16 ROCm Pack (Standard & 2-Byte Token-Shift)
// Target: AMD Radeon RX 9060 XT (Navi 44 / GFX1200 / RDNA4)
// Unified Cooperative LDS B-Tile Broadcast & Ping-Pong Weight Pipeline across S:
//   - S in {128, 256}: 8 Waves/WG, 2 K-Steps/Barrier, u16 Gather (8 KB LDS)
//   - S == 64:         8 Waves/WG, 2 K-Steps/Barrier, b32 Gather (16 KB LDS)
//   - S == 32:         4 Waves/WG, 2 K-Steps/Barrier, b64 Gather (16 KB LDS)
//   - S == 16:         4 Waves/WG, 1 K-Step/Stage,    b64 Gather (0 KB LDS, Reg)
// Architecture: Vorspann–Loop–Abspann (`setup` -> `half_shader` -> `finish`)
// ==============================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <type_traits>
#if !defined(HALF8_FLOAT8_DEFINED)
#define HALF8_FLOAT8_DEFINED
typedef __fp16 half8 __attribute__((ext_vector_type(8))); typedef float float8 __attribute__((ext_vector_type(8)));
typedef unsigned int uint4_v __attribute__((ext_vector_type(4)));
#endif
#if !defined(TOKEN64_DEFINED)
#define TOKEN64_DEFINED
typedef __fp16 Token64[64];
#endif
// --- 1. Hardware Utility Functions & Inline Assembly (GFX1200 / RDNA4) --------
#if !defined(K44_HW_UTILS_DEFINED)
#define K44_HW_UTILS_DEFINED
__device__ __forceinline__ const char* to_sgpr_ptr(const void* ptr) { uintptr_t u = reinterpret_cast<uintptr_t>(ptr);
    uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u));
    uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u >> 32));
    return reinterpret_cast<const char*>((static_cast<uintptr_t>(hi) << 32) | lo); }
template <typename T> __device__ __forceinline__ void load_global_b128_sgpr(T& dst, const char* base_sgpr, uint32_t offset_vgpr) {
    asm volatile("global_load_b128 %0, %1, %2" : "=v"(dst) : "v"(offset_vgpr), "s"(base_sgpr) : "memory"); }
template <typename T> __device__ __forceinline__ void load_global_b64_sgpr(T& dst, const char* base_sgpr, uint32_t offset_vgpr) {
    asm volatile("global_load_b64 %0, %1, %2" : "=v"(dst) : "v"(offset_vgpr), "s"(base_sgpr) : "memory"); }
template <typename T> __device__ __forceinline__ void load_global_b32_sgpr(T& dst, const char* base_sgpr, uint32_t offset_vgpr) {
    asm volatile("global_load_b32 %0, %1, %2" : "=v"(dst) : "v"(offset_vgpr), "s"(base_sgpr) : "memory"); }
__device__ __forceinline__ uint32_t cvt_pk_f16_f32(float a, float b) { uint32_t lo, hi;
    asm("v_cvt_f16_f32 %0, %2\n\t" "v_cvt_f16_f32 %1, %3\n\t"
        "v_pack_b32_f16 %0, %0, %1" : "=&v"(lo), "=&v"(hi) : "v"(a), "v"(b));
    return lo; }
__device__ __forceinline__ float8 __organic_wmma_f32_16x16x16_f16(half8 a, half8 b, float8 c) { float8 d = c;
#if defined(__gfx1200__)
    asm("v_wmma_f32_16x16x16_f16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
#endif
    return d; }
__device__ __forceinline__ void compute_wmma_4x_f16(half8 w, half8 b0, half8 b1, half8 b2, half8 b3, float8& c0, float8& c1, float8& c2, float8& c3) {
    c0 = __organic_wmma_f32_16x16x16_f16(w, b0, c0); c1 = __organic_wmma_f32_16x16x16_f16(w, b1, c1);
    c2 = __organic_wmma_f32_16x16x16_f16(w, b2, c2); c3 = __organic_wmma_f32_16x16x16_f16(w, b3, c3); }
__device__ __forceinline__ void pack_pair_f16(uint32_t& d0, uint32_t& d1, uint32_t lo, uint32_t hi) {
    asm("v_pack_b32_f16 %0, %1, %2"         : "=v"(d0) : "v"(lo), "v"(hi));
    asm("v_perm_b32 %0, %1, %2, 0x07060302" : "=v"(d1) : "v"(hi), "v"(lo)); }
__device__ __forceinline__ void read_lds_b_tile_4x(half8& b0, half8& b1, half8& b2, half8& b3, const half8 (&lds_sub)[4][32], uint32_t lane) {
    b0 = lds_sub[0][lane]; b1 = lds_sub[1][lane]; b2 = lds_sub[2][lane]; b3 = lds_sub[3][lane];
    asm volatile("" : "+v"(b0), "+v"(b1), "+v"(b2), "+v"(b3) :: "memory"); }
#endif
// --- 1b. Token-Shift Gather Addressing Subroutines (2-Byte / __half Offset) ---
template <int GATHER_SUBTILES, typename GatherElem> __device__ __forceinline__ void prefetch_gathers_32bit_mubuf_shift(
    GatherElem (&gv)[8], const uint4_v (&c_buf)[2], uint32_t m_off, __amdgpu_buffer_rsrc_t rsrc_X ) {
    const uint32_t* cols = reinterpret_cast<const uint32_t*>(c_buf);
    #pragma unroll
    for (int p = 0; p < 4; ++p) { const uint32_t lo = (cols[2 * p] << 1) + m_off, hi = (cols[2 * p + 1] << 1) + m_off;
        if constexpr (GATHER_SUBTILES == 1) asm volatile("buffer_load_u16 %0, %2, %4, null offen\n\tbuffer_load_u16 %1, %3, %4, null offen" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(lo), "v"(hi), "s"(rsrc_X) : "memory");
        else if constexpr (GATHER_SUBTILES == 2) asm volatile("buffer_load_b32 %0, %2, %4, null offen\n\tbuffer_load_b32 %1, %3, %4, null offen" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(lo), "v"(hi), "s"(rsrc_X) : "memory");
        else asm volatile("buffer_load_b64 %0, %2, %4, null offen\n\tbuffer_load_b64 %1, %3, %4, null offen" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(lo), "v"(hi), "s"(rsrc_X) : "memory");
    } }
template <int GATHER_SUBTILES, typename GatherElem> __device__ __forceinline__ void prefetch_gathers_64bit_global_shift(
    GatherElem (&gv)[8], const uint4_v (&c_buf)[2], uint64_t lane_base_X ) {
    const uint32_t* cols = reinterpret_cast<const uint32_t*>(c_buf);
    #pragma unroll
    for (int p = 0; p < 4; ++p) { uint64_t a0, a1;
        asm("v_mad_co_u64_u32 %0, null, %2, 2, %4\n\t" "v_mad_co_u64_u32 %1, null, %3, 2, %4"
            : "=&v"(a0), "=&v"(a1) : "v"(cols[2 * p]), "v"(cols[2 * p + 1]), "v"(lane_base_X));
        if constexpr (GATHER_SUBTILES == 1) asm volatile("global_load_u16 %0, %2, off\n\tglobal_load_u16 %1, %3, off" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(a0), "v"(a1) : "memory");
        else if constexpr (GATHER_SUBTILES == 2) asm volatile("global_load_b32 %0, %2, off\n\tglobal_load_b32 %1, %3, off" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(a0), "v"(a1) : "memory");
        else asm volatile("global_load_b64 %0, %2, off\n\tglobal_load_b64 %1, %3, off" : "=&v"(gv[2 * p]), "=&v"(gv[2 * p + 1]) : "v"(a0), "v"(a1) : "memory");
    } }
// --- 2. Unified Compile-Time Epilogue Activation Enum & Helper ----------------
enum class K44Act : int { NONE = 0, RELU = 1, SIGMOID = 2, SILU = 3, LEAKY_RELU = 4 }; template <K44Act ACT>
__device__ __forceinline__ float k44_apply_act(float v) { if constexpr (ACT == K44Act::RELU) return __builtin_fmaxf(v, 0.0f);
    else if constexpr (ACT == K44Act::SIGMOID) return 1.0f / (1.0f + __builtin_expf(-v));
    else if constexpr (ACT == K44Act::SILU) return v / (1.0f + __builtin_expf(-v));
    else if constexpr (ACT == K44Act::LEAKY_RELU) return __builtin_fmaxf(v, 0.1f * v); else return v;
}
// --- 3. Kernel 44 Token-Shift: Unified Cooperative LDS Pipeline (Pure FP16) ---
//     Structured into Vorspann (`setup`) -> Loop (`half_shader`) -> Abspann (`finish`)
template <int CLUSTER_SIZE, int WAVES_PER_WG, bool ADDR_64BIT = false> struct K44TokenShiftPipeline {
    static constexpr int WAVES_PER_CLUSTER = (CLUSTER_SIZE / 16 < WAVES_PER_WG) ? (CLUSTER_SIZE / 16) : WAVES_PER_WG;
    static constexpr int CLUSTERS_PER_WG   = WAVES_PER_WG / WAVES_PER_CLUSTER;
    static constexpr int K_CHUNK           = (CLUSTER_SIZE >= 32) ? 2 : 1;
    static constexpr int GATHER_SUBTILES   = (4 * K_CHUNK) / WAVES_PER_CLUSTER;
    static constexpr bool USE_LDS          = (WAVES_PER_CLUSTER > 1);
    using GatherElem = typename std::conditional<GATHER_SUBTILES == 4, uint2, uint32_t>::type; using WLoadElem  = half8;
    using LDSArray   = half8[2][USE_LDS ? CLUSTERS_PER_WG : 1][K_CHUNK][4][32];
    uint32_t lane, k_sub, sub_tile, cluster_in_wg, lane_idx_off, lane_w_off, m_off, j_base0, valid_lanes;
    const char *base_idx_ptr, *base_w0_ptr, *base_bias_ptr;
    typename std::conditional<ADDR_64BIT, uint64_t, __amdgpu_buffer_rsrc_t>::type addr_X; __half* base_Y_tile;
    const __half* base_R_tile; uint4_v c_buf_next[2];
    half8 bias_init; WLoadElem w_buf0, w_buf1;
    half8 b_reg0, b_reg1, b_reg2, b_reg3; GatherElem gv[8]; float8 c0, c1, c2, c3;
    __device__ __forceinline__ uint32_t c_slot() const { return (CLUSTERS_PER_WG > 1) ? cluster_in_wg : 0; }
    __device__ __forceinline__ void load_idx(uint4_v (&c_buf)[2]) { load_global_b128_sgpr(c_buf[0], base_idx_ptr, lane_idx_off + 0);
        load_global_b128_sgpr(c_buf[1], base_idx_ptr, lane_idx_off + 16); base_idx_ptr += 64 * K_CHUNK;
    } __device__ __forceinline__ void load_w(WLoadElem& w_buf) {
        load_global_b128_sgpr(w_buf, base_w0_ptr, lane_w_off); base_w0_ptr += 32 * sizeof(half8);
    } __device__ __forceinline__ void prefetch_gathers(const uint4_v (&c_buf)[2]) {
        if constexpr (ADDR_64BIT) prefetch_gathers_64bit_global_shift<GATHER_SUBTILES, GatherElem>(gv, c_buf, addr_X);
        else prefetch_gathers_32bit_mubuf_shift<GATHER_SUBTILES, GatherElem>(gv, c_buf, m_off, addr_X); }
    template <int WAIT_V, bool HAS_NEXT_IDX> __device__ __forceinline__ void pack_and_sync_stage(uint32_t dst_stage, const WLoadElem& w_dep, WLoadElem& w_tail, LDSArray& s_b_frag) {
        half8 b_dep;
        if constexpr (WAIT_V == 3) asm volatile("s_wait_loadcnt 0x3" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]) : "v"(w_dep), "v"(c0[0]) : "memory");
        else if constexpr (WAIT_V == 2) asm volatile("s_wait_loadcnt 0x2" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]) : "v"(w_dep), "v"(c0[0]) : "memory");
        else if constexpr (WAIT_V == 1) asm volatile("s_wait_loadcnt 0x1" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]) : "v"(w_dep), "v"(c0[0]) : "memory");
        else asm volatile("s_wait_loadcnt 0x0" : "+v"(gv[0]), "+v"(gv[1]), "+v"(gv[2]), "+v"(gv[3]), "+v"(gv[4]), "+v"(gv[5]), "+v"(gv[6]), "+v"(gv[7]) : "v"(w_dep), "v"(c0[0]) : "memory");
        if constexpr (GATHER_SUBTILES == 1) { uint32_t* p0 = reinterpret_cast<uint32_t*>(&b_dep);
            #pragma unroll
            for (int p = 0; p < 4; ++p) asm("v_pack_b32_f16 %0, %1, %2" : "=v"(p0[p]) : "v"(gv[2 * p]), "v"(gv[2 * p + 1]));
            s_b_frag[dst_stage][c_slot()][k_sub][sub_tile][lane] = b_dep; } else if constexpr (GATHER_SUBTILES == 2) {
            half8 b0; uint32_t *p0 = (uint32_t*)&b0, *p1 = reinterpret_cast<uint32_t*>(&b_dep);
            #pragma unroll
            for (int p = 0; p < 4; ++p) pack_pair_f16(p0[p], p1[p], gv[2 * p], gv[2 * p + 1]);
            s_b_frag[dst_stage][c_slot()][k_sub][sub_tile + 0][lane] = b0;
            s_b_frag[dst_stage][c_slot()][k_sub][sub_tile + 1][lane] = b_dep; } else {
            uint32_t *p0 = (uint32_t*)&b_reg0, *p1 = (uint32_t*)&b_reg1, *p2 = (uint32_t*)&b_reg2, *p3 = (uint32_t*)&b_dep;
            #pragma unroll
            for (int p = 0; p < 4; ++p) { pack_pair_f16(p0[p], p1[p], gv[2 * p].x, gv[2 * p + 1].x);
                pack_pair_f16(p2[p], p3[p], gv[2 * p].y, gv[2 * p + 1].y); }
            if constexpr (USE_LDS) { auto& lds_sub = s_b_frag[dst_stage][c_slot()][k_sub];
                lds_sub[0][lane] = b_reg0; lds_sub[1][lane] = b_reg1; lds_sub[2][lane] = b_reg2; lds_sub[3][lane] = b_dep;
            } else b_reg3 = b_dep; }
        if constexpr (HAS_NEXT_IDX) asm volatile("s_wait_loadcnt 0x0" : "+v"(c_buf_next[0]), "+v"(c_buf_next[1]), "+v"(w_tail) : "v"(b_dep) : "memory");
        else asm volatile("s_wait_loadcnt 0x0" : "+v"(w_tail) : "v"(b_dep) : "memory"); if constexpr (USE_LDS) __syncthreads();
    } template <int WAIT_W1, bool PREFETCH_W0> __device__ __forceinline__ void compute_lds_2k(uint32_t cur, LDSArray& s_b_frag) {
        half8 b0, b1, b2, b3; read_lds_b_tile_4x(b0, b1, b2, b3, s_b_frag[cur][c_slot()][0], lane);
        __builtin_amdgcn_sched_barrier(0); compute_wmma_4x_f16(w_buf0, b0, b1, b2, b3, c0, c1, c2, c3);
        __builtin_amdgcn_sched_barrier(0); if constexpr (PREFETCH_W0) load_w(w_buf0);
        read_lds_b_tile_4x(b0, b1, b2, b3, s_b_frag[cur][c_slot()][1], lane);
        if constexpr (WAIT_W1 == 11) asm volatile("s_wait_loadcnt 0xb" : "+v"(w_buf1) : "v"(c0[0]), "v"(w_buf0), "v"(b0) : "memory");
        else if constexpr (WAIT_W1 == 9) asm volatile("s_wait_loadcnt 0x9" : "+v"(w_buf1) : "v"(c0[0]), "v"(w_buf0), "v"(b0) : "memory");
        else asm volatile("s_wait_loadcnt 0x0" : "+v"(w_buf1) : "v"(c0[0]), "v"(b0) : "memory"); __builtin_amdgcn_sched_barrier(0);
        compute_wmma_4x_f16(w_buf1, b0, b1, b2, b3, c0, c1, c2, c3); }
    // --------------------------------------------------------------------------
    // 1. VORSPANN (Teil 1): `setup` — Pointer, Strides, Offsets, Bias & `idx_0`
    // --------------------------------------------------------------------------
    template <bool HAS_BIAS, bool PER_TILE_MASK = false, bool HAS_RESIDUAL = false> __device__ __forceinline__ void setup(
        __half* __restrict__ Y, const __half* __restrict__ X, const uint32_t* __restrict__ cluster_indices_32bit,
        const uint4* __restrict__ cluster_weights_wmma, const __half* __restrict__ neuron_bias,
        const uint32_t* __restrict__ tile_offsets_X, const uint32_t* __restrict__ tile_offsets_Y,
        uint32_t B_vec16, const __half* __restrict__ residual_X = nullptr ) {
        lane = threadIdx.x & 31; const uint32_t wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
        const uint32_t wg_idx  = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const uint32_t batch_b = __builtin_amdgcn_readfirstlane(blockIdx.x); constexpr uint32_t WAVES_PER_K = 4 / GATHER_SUBTILES;
        const uint32_t wave_in_clust = wave_id % WAVES_PER_CLUSTER, nb_w0 = wg_idx * WAVES_PER_WG + wave_id;
        k_sub         = (K_CHUNK == 2) ? (wave_in_clust / WAVES_PER_K) : 0;
        sub_tile      = (wave_in_clust % WAVES_PER_K) * GATHER_SUBTILES; cluster_in_wg = wave_id / WAVES_PER_CLUSTER;
        lane_idx_off  = (lane & 16) << 1; lane_w_off    = lane * sizeof(WLoadElem);
        m_off         = (lane & 15) << 3; j_base0       = (nb_w0 << 4) + ((lane & 16) >> 1);
        base_bias_ptr = to_sgpr_ptr(neuron_bias);
        base_idx_ptr  = to_sgpr_ptr(reinterpret_cast<const char*>(cluster_indices_32bit) + ((nb_w0 / (CLUSTER_SIZE / 16)) * B_vec16 + k_sub) * 64);
        base_w0_ptr   = to_sgpr_ptr(reinterpret_cast<const WLoadElem*>(cluster_weights_wmma) + ((size_t)(nb_w0 * B_vec16) << 5));
        const uint32_t x_tile_half_off = __builtin_amdgcn_readfirstlane(tile_offsets_X[batch_b]);
        const uint32_t raw_y           = __builtin_amdgcn_readfirstlane(tile_offsets_Y[batch_b]);
        uint32_t y_tile_half_off       = raw_y; if constexpr (PER_TILE_MASK) {
            y_tile_half_off = (raw_y & 0xffffu) << 2; valid_lanes     = raw_y >> 16;
        } const char* base_X = reinterpret_cast<const char*>(X) + ((size_t)x_tile_half_off << 1) + (sub_tile << 1);
        if constexpr (ADDR_64BIT) addr_X = reinterpret_cast<uint64_t>(base_X) + m_off;
        else addr_X = __builtin_amdgcn_make_buffer_rsrc((void*)base_X, 0, 0xffffffff, 0x31004000); base_Y_tile = Y + y_tile_half_off;
        if constexpr (HAS_RESIDUAL) base_R_tile = residual_X + y_tile_half_off;
        bias_init = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; load_idx(c_buf_next);
        if constexpr (HAS_BIAS) load_global_b128_sgpr(bias_init, base_bias_ptr, j_base0 * sizeof(__half));
        asm volatile("s_wait_loadcnt 0x0" : "+v"(c_buf_next[0]), "+v"(c_buf_next[1]), "+v"(bias_init) :: "memory"); }
    // --------------------------------------------------------------------------
    // 2. HAUPTSCHLEIFE / PROLOG / DRAIN: `half_shader`
    // --------------------------------------------------------------------------
    template <bool DO_COMPUTE, bool DO_GATHER, bool DO_LOOKAHEAD, bool HAS_BIAS = false>
    __device__ __forceinline__ void half_shader(uint32_t cur, uint32_t nxt, LDSArray& s_b_frag) {
        if constexpr (!DO_COMPUTE && DO_GATHER) {
            // Prolog Stage 0 (`cur=1, nxt=0`): Gathers + idx_1 + Bias-Init + W_0 + LDS-Pack
            prefetch_gathers(c_buf_next); if constexpr (DO_LOOKAHEAD) load_idx(c_buf_next);
            if constexpr (HAS_BIAS) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) c0[i] = c1[i] = c2[i] = c3[i] = static_cast<float>(bias_init[i]); } else {
                c0 = c1 = c2 = c3 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; }
            load_w(w_buf0); pack_and_sync_stage<3, DO_LOOKAHEAD>(0u, w_buf0, w_buf0, s_b_frag);
        } else if constexpr (DO_COMPUTE && DO_GATHER) {
            // Steady-State (`DO_LOOKAHEAD=true`) & Vorletzter Half-Step (`DO_LOOKAHEAD=false`)
            constexpr int WAIT_W1 = DO_LOOKAHEAD ? 11 : 9;
            constexpr int WAIT_V  = DO_LOOKAHEAD ? (USE_LDS ? 3 : 2) : (USE_LDS ? 1 : 0); load_w(w_buf1);
            prefetch_gathers(c_buf_next); if constexpr (DO_LOOKAHEAD) load_idx(c_buf_next);
            if constexpr (USE_LDS) { compute_lds_2k<WAIT_W1, true>(cur, s_b_frag);
                pack_and_sync_stage<WAIT_V, DO_LOOKAHEAD>(nxt, w_buf1, w_buf0, s_b_frag); } else {
                __builtin_amdgcn_sched_barrier(0); compute_wmma_4x_f16(w_buf0, b_reg0, b_reg1, b_reg2, b_reg3, c0, c1, c2, c3);
                pack_and_sync_stage<WAIT_V, DO_LOOKAHEAD>(0u, w_buf1, w_buf1, s_b_frag); w_buf0 = w_buf1;
            } } else if constexpr (DO_COMPUTE && !DO_GATHER) {
            // Letzter Half-Step (Drain): Nur noch Compute der verbleibenden Stage `cur`
            if constexpr (USE_LDS) { load_w(w_buf1);
                compute_lds_2k<0, false>(cur, s_b_frag); } else {
                compute_wmma_4x_f16(w_buf0, b_reg0, b_reg1, b_reg2, b_reg3, c0, c1, c2, c3); }
        } }
    // --------------------------------------------------------------------------
    // 3. ABSPANN (Teil 2): `finish` — Activation, Residual, Masking & Writeback
    // --------------------------------------------------------------------------
    template < bool APPLY_RELU = false,
        bool APPLY_SIGMOID = false, int VALID_TOKENS = 64,
        bool STREAMING_STORE = false, bool APPLY_SILU = false,
        bool PER_TILE_MASK = false, bool HAS_RESIDUAL = false,
        K44Act ACT_ENUM = K44Act::NONE, bool HAS_COUT_MAP = false
    > __device__ __forceinline__ void finish(uint32_t channel_stride_Y, const uint16_t* __restrict__ cout_map = nullptr) {
        uint32_t lane_valid_mask = 0xffffffffu; if constexpr (PER_TILE_MASK) {
            lane_valid_mask = ((valid_lanes >> (lane & 15)) & 1u) ? 0xffffffffu : 0x00000000u;
        } else if constexpr (VALID_TOKENS < 64) { lane_valid_mask = ((lane & 15) < (VALID_TOKENS >> 2)) ? 0xffffffffu : 0x00000000u;
        } constexpr K44Act EFFECTIVE_ACT = (ACT_ENUM != K44Act::NONE) ? ACT_ENUM :
            (APPLY_SILU ? K44Act::SILU : (APPLY_SIGMOID ? K44Act::SIGMOID : (APPLY_RELU ? K44Act::RELU : K44Act::NONE)));
        uint4_v dst_j_vec = {0u, 0u, 0u, 0u}; if constexpr (HAS_COUT_MAP) {
            const char* base_cout_ptr = to_sgpr_ptr(cout_map);
            load_global_b128_sgpr(dst_j_vec, base_cout_ptr, j_base0 * sizeof(uint16_t));
            asm volatile("s_wait_loadcnt 0x0" : "+v"(dst_j_vec) :: "memory"); }
        #pragma unroll
        for (int i = 0; i < 8; ++i) { const uint32_t out_j = HAS_COUT_MAP
                ? ((dst_j_vec[i >> 1] >> ((i & 1) * 16)) & 0xffffu) : (j_base0 + i);
            float c0_i = c0[i], c1_i = c1[i], c2_i = c2[i], c3_i = c3[i]; float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
            if constexpr (HAS_RESIDUAL) { const uint2 r_vec = *reinterpret_cast<const uint2*>(
                    base_R_tile + (size_t)out_j * channel_stride_Y + (lane & 15) * 4 );
                const __half* r_h = reinterpret_cast<const __half*>(&r_vec); r0 = __half2float(r_h[0]);
                r1 = __half2float(r_h[1]); r2 = __half2float(r_h[2]);
                r3 = __half2float(r_h[3]); if constexpr (HAS_COUT_MAP) {
                    c0_i += r0; c1_i += r1; c2_i += r2; c3_i += r3; }
            } float v0 = k44_apply_act<EFFECTIVE_ACT>(c0_i);
            float v1 = k44_apply_act<EFFECTIVE_ACT>(c1_i); float v2 = k44_apply_act<EFFECTIVE_ACT>(c2_i);
            float v3 = k44_apply_act<EFFECTIVE_ACT>(c3_i); if constexpr (HAS_RESIDUAL && !HAS_COUT_MAP) {
                v0 += r0; v1 += r1; v2 += r2; v3 += r3; }
            uint2 y_vec = make_uint2(cvt_pk_f16_f32(v0, v1), cvt_pk_f16_f32(v2, v3));
            if constexpr (PER_TILE_MASK || (VALID_TOKENS < 64)) { y_vec.x &= lane_valid_mask; y_vec.y &= lane_valid_mask;
            } const void* ptr = base_Y_tile + (size_t)out_j * channel_stride_Y + (lane & 15) * 4;
            if constexpr (STREAMING_STORE) asm volatile("global_store_b64 %0, %1, off th:TH_STORE_NT" :: "v"(ptr), "v"(y_vec) : "memory");
            else *reinterpret_cast<uint2*>(const_cast<void*>(ptr)) = y_vec; }
    } };
template < int CLUSTER_SIZE = 128,
    int WAVES_PER_WG = (CLUSTER_SIZE >= 64 ? 8 : 4), bool HAS_BIAS = true,
    bool APPLY_RELU = false, bool APPLY_SIGMOID = false,
    int VALID_TOKENS = 64, bool STREAMING_STORE = false,
    bool ADDR_64BIT = false, bool APPLY_SILU = false,
    bool PER_TILE_MASK = false, bool HAS_RESIDUAL = false,
    K44Act ACT_ENUM = K44Act::NONE, bool HAS_COUT_MAP = false
> __launch_bounds__(WAVES_PER_WG * 32, 2)
__attribute__((amdgpu_flat_work_group_size(WAVES_PER_WG * 32, WAVES_PER_WG * 32)))
__global__ void connectome_k44_token_shift_f16_kernel( __half* __restrict__ Y,
    const __half* __restrict__ X, const uint32_t* __restrict__ cluster_indices_32bit,
    const uint4* __restrict__ cluster_weights_wmma,
    const __half* __restrict__ neuron_bias,
    const uint32_t* __restrict__ tile_offsets_X,
    const uint32_t* __restrict__ tile_offsets_Y,
    uint32_t channel_stride_Y,
    uint32_t B_vec16,
    const __half* __restrict__ residual_X = nullptr,
    const uint16_t* __restrict__ cout_map = nullptr
) {
    using Pipe = K44TokenShiftPipeline<CLUSTER_SIZE, WAVES_PER_WG, ADDR_64BIT>;
    __shared__ typename Pipe::LDSArray s_b_frag;
    Pipe pipe;
    // --- 1. VORSPANN (Teil 1: Pointer, Strides & initialer Index-Load) ---
    pipe.template setup<HAS_BIAS, PER_TILE_MASK, HAS_RESIDUAL>(
        Y, X, cluster_indices_32bit, cluster_weights_wmma, neuron_bias,
        tile_offsets_X, tile_offsets_Y, B_vec16, residual_X
    );
    // --- 1. VORSPANN (Teil 2: Prolog über half_shader ohne Compute) ---
    pipe.template half_shader<false, true, true, HAS_BIAS>(1u, 0u, s_b_frag);
    // --- 2. HAUPTSCHLEIFE (Steady-State Doppelpuffer-Ping-Pong) ---
    const uint32_t num_steps = B_vec16 / Pipe::K_CHUNK;
    for (uint32_t s = 0; s < num_steps - 2u; ++s) {
        pipe.template half_shader<true, true, true, false>(s & 1u, (s + 1u) & 1u, s_b_frag);
    }
    // --- 3. ABSPANN (Teil 1: Drain der letzten beiden Half-Steps) ---
    pipe.template half_shader<true, true, false, false>((num_steps - 2u) & 1u, (num_steps - 1u) & 1u, s_b_frag);
    pipe.template half_shader<true, false, false, false>((num_steps - 1u) & 1u, num_steps & 1u, s_b_frag);
    // --- 3. ABSPANN (Teil 2: Epilog & Writeback) ---
    pipe.template finish<
        APPLY_RELU, APPLY_SIGMOID, VALID_TOKENS, STREAMING_STORE,
        APPLY_SILU, PER_TILE_MASK, HAS_RESIDUAL, ACT_ENUM, HAS_COUT_MAP
    >(channel_stride_Y, cout_map);
}
// --- 4. Weight Interleaving & Host Launchers ----------------------------------
#if !defined(K44_INTERLEAVE_DEFINED)
#define K44_INTERLEAVE_DEFINED
__global__ void interleave_weights_cluster16_wmma_gpu_kernel(uint4* __restrict__ dst_wmma, const uint4* __restrict__ src_compact, uint32_t num_output_neurons, uint32_t B_vec8) {
    uint32_t B_vec16 = B_vec8 / 2, total_elements = (num_output_neurons / 16) * B_vec16 * 32, global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= total_elements) return;
    uint32_t tile_idx = global_idx / 32, lane = global_idx % 32;
    dst_wmma[global_idx] = src_compact[((tile_idx / B_vec16) * 16 + (lane & 15)) * B_vec8 + (2 * (tile_idx % B_vec16) + (lane >> 4))];
}
template <typename ChunkT>
__global__ void interleave_weights_cluster16_wmma_generic_gpu_kernel(ChunkT* __restrict__ dst_wmma, const ChunkT* __restrict__ src_compact, uint32_t num_output_neurons, uint32_t B_vec8) {
    uint32_t B_vec16 = B_vec8 / 2, total_elements = (num_output_neurons / 16) * B_vec16 * 32, global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= total_elements) return;
    uint32_t tile_idx = global_idx / 32, lane = global_idx % 32;
    dst_wmma[global_idx] = src_compact[((tile_idx / B_vec16) * 16 + (lane & 15)) * B_vec8 + (2 * (tile_idx % B_vec16) + (lane >> 4))];
}
#endif
template <
    int CLUSTER_SIZE = 128,
    bool HAS_BIAS = true,
    bool APPLY_RELU = false,
    bool APPLY_SIGMOID = false,
    int VALID_TOKENS = 64,
    bool STREAMING_STORE = false,
    bool APPLY_SILU = false,
    bool PER_TILE_MASK = false,
    bool HAS_RESIDUAL = false,
    K44Act ACT_ENUM = K44Act::NONE,
    bool HAS_COUT_MAP = false
>
inline void launch_k44_token_shift_f16_forward(
    __half* Y,
    const __half* X,
    const uint32_t* cluster_indices_32bit,
    const uint4* cluster_weights_wmma,
    const __half* neuron_bias,
    const uint32_t* tile_offsets_X,
    const uint32_t* tile_offsets_Y,
    uint32_t num_tiles,
    uint32_t Cout,
    uint32_t channel_stride_Y,
    uint32_t B_vec16,
    hipStream_t stream = nullptr,
    const __half* residual_X = nullptr,
    const uint16_t* cout_map = nullptr
) {
    constexpr int DEFAULT_WAVES = (CLUSTER_SIZE >= 64) ? 8 : 4;
    constexpr int SINGLE_CLUSTER_WAVES = CLUSTER_SIZE / 16;
    if ((Cout < (uint32_t)(DEFAULT_WAVES * 16)) || (Cout % (uint32_t)(DEFAULT_WAVES * 16) != 0)) {
        const uint32_t num_neuron_blocks = (Cout + SINGLE_CLUSTER_WAVES * 16 - 1) / (SINGLE_CLUSTER_WAVES * 16);
        dim3 grid(num_tiles, num_neuron_blocks, 1);
        auto kernel = connectome_k44_token_shift_f16_kernel<
            CLUSTER_SIZE, SINGLE_CLUSTER_WAVES, HAS_BIAS, APPLY_RELU, APPLY_SIGMOID,
            VALID_TOKENS, STREAMING_STORE, false, APPLY_SILU, PER_TILE_MASK, HAS_RESIDUAL, ACT_ENUM, HAS_COUT_MAP
        >;
        hipLaunchKernelGGL(
            kernel, grid, dim3(SINGLE_CLUSTER_WAVES * 32, 1, 1), 0, stream,
            Y, X, cluster_indices_32bit, cluster_weights_wmma, neuron_bias,
            tile_offsets_X, tile_offsets_Y, channel_stride_Y, B_vec16, residual_X, cout_map
        );
    } else {
        const uint32_t num_neuron_blocks = (Cout + DEFAULT_WAVES * 16 - 1) / (DEFAULT_WAVES * 16);
        dim3 grid(num_tiles, num_neuron_blocks, 1);
        auto kernel = connectome_k44_token_shift_f16_kernel<
            CLUSTER_SIZE, DEFAULT_WAVES, HAS_BIAS, APPLY_RELU, APPLY_SIGMOID,
            VALID_TOKENS, STREAMING_STORE, false, APPLY_SILU, PER_TILE_MASK, HAS_RESIDUAL, ACT_ENUM, HAS_COUT_MAP
        >;
        hipLaunchKernelGGL(
            kernel, grid, dim3(DEFAULT_WAVES * 32, 1, 1), 0, stream,
            Y, X, cluster_indices_32bit, cluster_weights_wmma, neuron_bias,
            tile_offsets_X, tile_offsets_Y, channel_stride_Y, B_vec16, residual_X, cout_map
        );
    }
}
#define K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, B_FLAG, ACT_V, R_FLAG, ...) \
    case (((B_FLAG) ? 1 : 0) | ((static_cast<int>(ACT_V) & 7) << 1) | ((R_FLAG) ? 16 : 0)): \
        launch_k44_token_shift_f16_forward< \
            CLUSTER_SIZE, B_FLAG, (ACT_V == K44Act::RELU), (ACT_V == K44Act::SIGMOID), \
            64, false, (ACT_V == K44Act::SILU), MASK_FLAG, R_FLAG, ACT_V, false \
        >(__VA_ARGS__); \
        break
#define K44_DISPATCH_EPILOGUE(CLUSTER_SIZE, MASK_FLAG, HAS_B, ACT_V, HAS_R, ...) \
    do { \
        const int _k44_ep_key = ((HAS_B) ? 1 : 0) | ((static_cast<int>(ACT_V) & 7) << 1) | ((HAS_R) ? 16 : 0); \
        switch (_k44_ep_key) { \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::NONE,    false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::NONE,    false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::RELU,    false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::RELU,    false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::SIGMOID, false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::SIGMOID, false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::SILU,    false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::SILU,    false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::SILU,    true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::SILU,    true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::NONE,    true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::NONE,    true,  __VA_ARGS__); \
            default: break; \
        } \
    } while (0)
#define K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, B_FLAG, ACT_V, R_FLAG, CMAP_FLAG, ...) \
    case (((B_FLAG) ? 1 : 0) | ((static_cast<int>(ACT_V) & 7) << 1) | ((R_FLAG) ? 16 : 0) | ((CMAP_FLAG) ? 32 : 0)): \
        launch_k44_token_shift_f16_forward< \
            CLUSTER_SIZE, B_FLAG, (ACT_V == K44Act::RELU), (ACT_V == K44Act::SIGMOID), \
            64, false, (ACT_V == K44Act::SILU), MASK_FLAG, R_FLAG, ACT_V, CMAP_FLAG \
        >(__VA_ARGS__); \
        break
#define K44_DISPATCH_EPILOGUE_EX(CLUSTER_SIZE, MASK_FLAG, HAS_B, ACT_V, HAS_R, HAS_CMAP, ...) \
    do { \
        const int _k44_ep_key = ((HAS_B) ? 1 : 0) | ((static_cast<int>(ACT_V) & 7) << 1) | ((HAS_R) ? 16 : 0) | ((HAS_CMAP) ? 32 : 0); \
        switch (_k44_ep_key) { \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::NONE,    false, false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::NONE,    false, false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::RELU,    false, false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::RELU,    false, false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::SIGMOID, false, false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::SIGMOID, false, false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::SILU,    false, false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::SILU,    false, false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::SILU,    true,  false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::SILU,    true,  false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::NONE,    true,  false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::NONE,    true,  false, __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::NONE,    false, true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::NONE,    false, true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::RELU,    false, true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::RELU,    false, true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::SIGMOID, false, true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::SIGMOID, false, true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::SILU,    false, true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::SILU,    false, true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::SILU,    true,  true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::SILU,    true,  true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, false, K44Act::NONE,    true,  true,  __VA_ARGS__); \
            K44_DISPATCH_EPILOGUE_EX_CASE(CLUSTER_SIZE, MASK_FLAG, true,  K44Act::NONE,    true,  true,  __VA_ARGS__); \
            default: break; \
        } \
    } while (0)
#include "../common/kernel44_valu_pack.cuh"
