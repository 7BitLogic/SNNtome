#pragma once
// ==============================================================================
// tests/benchmark_common.cuh
//
// Shared Benchmark & Verification Harness for AMD RDNA4 (Navi 44 / GFX1200)
// Pure FP16 Architecture with Multi-Wave Ring-Buffer VRAM-Streaming Pool.
// ==============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../kernels/kernel44_immutable.cuh"
#include "../kernels/kernel44_dual_buffer.cuh"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <random>
#include <cmath>

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at line %d\n", hipGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)
#endif

// -----------------------------------------------------------------------------
// 1. GPU Pseudo-Random Linear Congruential Generator & Input Tensor Init
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t lcg_rand(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

__global__ void init_x_continuous_f16_gpu_kernel(uint4* __restrict__ X_u4, size_t num_u4, uint32_t seed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < num_u4; i += stride) {
        uint32_t state = (uint32_t)(i ^ ((size_t)seed * 0x9e3779b97f4a7c15ULL));
        __half h[8];
        for (int k = 0; k < 8; ++k) {
            uint32_t r = lcg_rand(state);
            float f = ((float)(r & 0x7fffffff) * (2.0f / 2147483647.0f)) - 1.0f;
            h[k] = __float2half(f);
        }
        uint4 out;
        out.x = *reinterpret_cast<uint32_t*>(&h[0]);
        out.y = *reinterpret_cast<uint32_t*>(&h[2]);
        out.z = *reinterpret_cast<uint32_t*>(&h[4]);
        out.w = *reinterpret_cast<uint32_t*>(&h[6]);
        X_u4[i] = out;
    }
}

__global__ void gather_validation_cols_f16_kernel(
    uint4* __restrict__ dst_cols_u4,
    const uint4* __restrict__ X_u4,
    const uint32_t* __restrict__ cols,
    uint32_t B
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_u4 = B * 8; // 64 halfs = 8 uint4 per column
    if (idx < total_u4) {
        uint32_t s = idx >> 3;
        uint32_t vec_in_col = idx & 7;
        uint32_t col = cols[s];
        dst_cols_u4[idx] = X_u4[(size_t)col * 8 + vec_in_col];
    }
}

// -----------------------------------------------------------------------------
// 2. Deterministic Pseudo-Random Connectome & Tensor Generation
// -----------------------------------------------------------------------------
static inline std::vector<uint32_t> generate_indices(uint32_t Cin, uint32_t num_clusters, uint32_t B, uint64_t seed) {
    std::vector<uint32_t> idx((size_t)num_clusters * B);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, Cin - 1);
    const uint32_t base_clusters = std::min<uint32_t>(num_clusters, 512u);
    for (uint32_t c = 0; c < base_clusters; ++c) {
        uint32_t* row = idx.data() + (size_t)c * B;
        for (uint32_t s = 0; s < B; ++s) {
            row[s] = dist(rng);
        }
        std::sort(row, row + B);
    }
    for (uint32_t c = base_clusters; c < num_clusters; ++c) {
        const uint32_t* src = idx.data() + (size_t)(c % base_clusters) * B;
        uint32_t* dst = idx.data() + (size_t)c * B;
        uint32_t shift = dist(rng);
        uint32_t split = B;
        for (uint32_t s = 0; s < B; ++s) {
            uint32_t v = src[s] + shift;
            if (v >= Cin) {
                if (split == B) split = s;
                v -= Cin;
            }
            dst[s] = v;
        }
        if (split < B) {
            std::rotate(dst, dst + split, dst + B);
        }
    }
    return idx;
}

static inline std::vector<uint16_t> generate_uniform_f16(size_t count, float low, float high, uint64_t seed) {
    std::vector<uint16_t> v(count);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < count; ++i) {
        __half h = __float2half(dist(rng));
        v[i] = *reinterpret_cast<uint16_t*>(&h);
    }
    return v;
}

static inline std::vector<uint16_t> generate_weights_f16(uint32_t Cout, uint32_t B, uint64_t seed = 1337) {
    return generate_uniform_f16((size_t)Cout * B, -1.0f, 1.0f, seed);
}

static inline std::vector<uint16_t> generate_biases(uint32_t Cout, uint64_t seed = 2026) {
    return generate_uniform_f16(Cout, -0.5f, 0.5f, seed);
}

// -----------------------------------------------------------------------------
// 3. GPU Clock Stabilization & Sustained Warmup Harness
// -----------------------------------------------------------------------------
template <typename F>
static void run_gpu_sustained_warmup(F&& launcher, int target_ms = 200, hipStream_t stream = nullptr) {
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start, stream));
    for (int i = 0; i < 40; ++i) launcher();
    CHECK_HIP(hipEventRecord(stop, stream));
    CHECK_HIP(hipEventSynchronize(stop));

    float calib_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&calib_ms, start, stop));
    int burst_reps = std::max(40, (int)std::ceil((float)target_ms / std::max(0.001f, calib_ms / 40.0f)));

    CHECK_HIP(hipEventRecord(start, stream));
    for (int i = 0; i < burst_reps; ++i) launcher();
    CHECK_HIP(hipEventRecord(stop, stream));
    CHECK_HIP(hipEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    printf("[WARMUP] %u launches in %.3f ms continuous GPU full-load (Target >= %d ms) completed.\n",
           burst_reps, elapsed_ms, target_ms);
}

// -----------------------------------------------------------------------------
// 4. Weight Interleaving for Matrix Core WMMA Architecture
// -----------------------------------------------------------------------------
static inline void interleave_weights_f16(
    void* d_W_wmma,
    const void* d_W_compact,
    uint32_t Cout,
    uint32_t B
) {
    uint32_t B_vec8 = B / 8;
    uint32_t total_elements = (Cout / 16) * (B_vec8 / 2) * 32;
    uint32_t grid_dim = (total_elements + 255) / 256;
    interleave_weights_cluster16_wmma_gpu_kernel<<<grid_dim, 256>>>(
        reinterpret_cast<uint4*>(d_W_wmma),
        reinterpret_cast<const uint4*>(d_W_compact),
        Cout, B_vec8
    );
    CHECK_HIP(hipDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// 5. Multi-Wave Ring-Buffer Pool (256 MiB Weights, 64 MiB Indices, 256 MiB Out)
//    Prevents artificial L2 cache residency of weights & indices across waves.
//    Pre-allocates validation scratch buffers so zero hipMalloc/hipFree occurs
//    inside the inner measurement loops (preserving 2 MB VRAM hugepages & clocks).
// -----------------------------------------------------------------------------
struct RingBufferPool {
    static constexpr size_t W_POOL_BYTES = 256ULL << 20; // 256 MiB (8x 32 MB L2)
    static constexpr size_t C_POOL_BYTES = 64ULL  << 20; // 64 MiB
    static constexpr size_t Y_POOL_BYTES = 256ULL << 20; // 256 MiB

    uint4*    d_W_pool     = nullptr;
    uint4*    d_W_val_b1k  = nullptr; // Pre-interleaved Cout=8192, B=1024 validation weights
    uint4*    d_W_val_b2k  = nullptr; // Pre-interleaved Cout=4096, B=2048 validation weights
    uint32_t* d_C_pool     = nullptr;
    __half*   d_Y_pool     = nullptr;
    __half*   d_bias       = nullptr;
    uint4*    d_val_cols   = nullptr; // Scratch for fast single-DMA column gather (2048 * 64 halfs)

    std::vector<uint16_t> h_W_ref;
    std::vector<uint16_t> h_B_ref;

    void init(uint32_t max_cout = 8192, uint32_t max_b = 2048) {
        CHECK_HIP(hipMalloc(&d_W_pool,    W_POOL_BYTES));
        CHECK_HIP(hipMalloc(&d_C_pool,    C_POOL_BYTES));
        CHECK_HIP(hipMalloc(&d_Y_pool,    Y_POOL_BYTES));
        CHECK_HIP(hipMalloc(&d_bias,      max_cout * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_val_cols,  max_b * 64 * sizeof(__half)));

        size_t w_wmma_bytes = (size_t)(8192 / 16) * (1024 / 16) * 32 * sizeof(uint4);
        CHECK_HIP(hipMalloc(&d_W_val_b1k, w_wmma_bytes));
        CHECK_HIP(hipMalloc(&d_W_val_b2k, w_wmma_bytes));

        init_x_continuous_f16_gpu_kernel<<<1024, 256>>>(
            d_W_pool, W_POOL_BYTES / sizeof(uint4), 54321u
        );
        CHECK_HIP(hipDeviceSynchronize());

        h_W_ref = generate_weights_f16(max_cout, max_b, 1337);
        h_B_ref = generate_biases(max_cout, 2026);
        CHECK_HIP(hipMemcpy(d_bias, h_B_ref.data(), max_cout * sizeof(uint16_t), hipMemcpyHostToDevice));

        // Pre-interleave validation weights once at startup
        void* d_W_compact = nullptr;
        CHECK_HIP(hipMalloc(&d_W_compact, (size_t)max_cout * max_b * sizeof(uint16_t)));
        CHECK_HIP(hipMemcpy(d_W_compact, h_W_ref.data(), (size_t)max_cout * max_b * sizeof(uint16_t), hipMemcpyHostToDevice));
        interleave_weights_f16(d_W_val_b1k, d_W_compact, 8192, 1024);
        interleave_weights_f16(d_W_val_b2k, d_W_compact, 4096, 2048);
        CHECK_HIP(hipFree(d_W_compact));
    }

    const uint4* get_val_weights(uint32_t B) const {
        return (B == 2048) ? d_W_val_b2k : d_W_val_b1k;
    }

    void warmup(int target_ms = 300) {
        run_gpu_sustained_warmup([&]() {
            launch_k44_f16_forward<256, false>(
                d_Y_pool, reinterpret_cast<const __half*>(d_W_pool),
                d_C_pool, d_W_pool, d_bias, 512, 8192, 8192, 1024 / 16
            );
        }, target_ms);
    }

    void destroy() {
        if (d_W_pool)    { CHECK_HIP(hipFree(d_W_pool));    d_W_pool    = nullptr; }
        if (d_W_val_b1k) { CHECK_HIP(hipFree(d_W_val_b1k)); d_W_val_b1k = nullptr; }
        if (d_W_val_b2k) { CHECK_HIP(hipFree(d_W_val_b2k)); d_W_val_b2k = nullptr; }
        if (d_C_pool)    { CHECK_HIP(hipFree(d_C_pool));    d_C_pool    = nullptr; }
        if (d_Y_pool)    { CHECK_HIP(hipFree(d_Y_pool));    d_Y_pool    = nullptr; }
        if (d_bias)      { CHECK_HIP(hipFree(d_bias));      d_bias      = nullptr; }
        if (d_val_cols)  { CHECK_HIP(hipFree(d_val_cols));  d_val_cols  = nullptr; }
    }
};

// -----------------------------------------------------------------------------
// 6. Fast Pointwise CPU Reference Validation (Single-DMA Gather, < 0.2 ms CPU)
//    Validates 3 representative neurons [3/3] across all 64 tokens without
//    letting the GPU shader clock drop between measurements.
// -----------------------------------------------------------------------------
template <int CLUSTER_SIZE, bool USE_DUAL_BUFFER = false>
static bool run_falsification_wave0_f16(
    uint32_t Cin, uint32_t Cout, uint32_t B,
    const void* d_X, const std::vector<uint32_t>& h_C,
    RingBufferPool& pool,
    double& out_max_diff
) {
    out_max_diff = 0.0;
    const uint32_t B_vec16 = B / 16;
    const uint4* d_W_val = pool.get_val_weights(B);

    if constexpr (USE_DUAL_BUFFER) {
        launch_k44_dual_buffer_f16_forward<CLUSTER_SIZE, false>(
            pool.d_Y_pool, reinterpret_cast<const __half*>(d_X),
            pool.d_C_pool, d_W_val, pool.d_bias, 64, Cin, Cout, B_vec16, nullptr
        );
    } else {
        launch_k44_f16_forward<CLUSTER_SIZE, false>(
            pool.d_Y_pool, reinterpret_cast<const __half*>(d_X),
            pool.d_C_pool, d_W_val, pool.d_bias, 64, Cin, Cout, B_vec16, nullptr
        );
    }

    uint32_t total_u4 = B * 8;
    gather_validation_cols_f16_kernel<<<(total_u4 + 255) / 256, 256>>>(
        pool.d_val_cols, reinterpret_cast<const uint4*>(d_X), pool.d_C_pool, B
    );
    CHECK_HIP(hipDeviceSynchronize());

    std::vector<uint16_t> h_Y_gpu((size_t)CLUSTER_SIZE * 64);
    std::vector<uint16_t> h_cols((size_t)B * 64);
    CHECK_HIP(hipMemcpy(h_Y_gpu.data(), pool.d_Y_pool, h_Y_gpu.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_cols.data(),  pool.d_val_cols, h_cols.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));

    // Validate 3 representative neurons across Cluster 0: first, middle, last [3/3]
    const uint32_t test_neurons[3] = {0, (uint32_t)(CLUSTER_SIZE / 2), (uint32_t)(CLUSTER_SIZE - 1)};
    for (int idx = 0; idx < 3; ++idx) {
        uint32_t j = test_neurons[idx];
        float bias = __half2float(*reinterpret_cast<const __half*>(&pool.h_B_ref[j]));
        for (int t = 0; t < 64; ++t) {
            double acc = (double)bias;
            for (uint32_t s = 0; s < B; ++s) {
                float w = __half2float(*reinterpret_cast<const __half*>(&pool.h_W_ref[(size_t)j * B + s]));
                float x = __half2float(*reinterpret_cast<const __half*>(&h_cols[(size_t)s * 64 + t]));
                acc += (double)w * (double)x;
            }
            float ref = (float)acc;
            float gpu_v = __half2float(*reinterpret_cast<const __half*>(&h_Y_gpu[(size_t)j * 64 + t]));
            double diff = std::abs((double)gpu_v - ref);
            if (diff > out_max_diff) out_max_diff = diff;
            if (diff > 0.06 && (diff / (std::abs(ref) + 1e-5)) > 0.025) {
                fprintf(stderr, "[FALSIFICATION CONTRADICTION] FP16 S=%d Cin=%u Neuron %u: Diff=%f\n",
                        CLUSTER_SIZE, Cin, j, diff);
                return false;
            }
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// 7. Multi-Wave Ring-Buffer Benchmark Run (VRAM-Streaming Pool + Validation)
// -----------------------------------------------------------------------------
struct LatencyStats {
    float lat_mean_us;
    float lat_std_us;
    float lat_min_us;
    float lat_max_us;
    double tsyns_mean;
    double tsyns_std;
    double tsyns_min;
    double tsyns_max;
};

template <typename F>
static LatencyStats measure_ring_stats(F&& launch_pass, int num_waves, double syn_per_wave, int iters = 4) {
    for (int w = 0; w < 4; ++w) launch_pass();
    CHECK_HIP(hipDeviceSynchronize());

    const int reps = std::max(1, 16 / num_waves);
    std::vector<float> lats(iters);
    std::vector<double> tsyns(iters);

    hipEvent_t ev0, ev1;
    CHECK_HIP(hipEventCreate(&ev0));
    CHECK_HIP(hipEventCreate(&ev1));

    for (int it = 0; it < iters; ++it) {
        CHECK_HIP(hipEventRecord(ev0));
        for (int r = 0; r < reps; ++r) launch_pass();
        CHECK_HIP(hipEventRecord(ev1));
        CHECK_HIP(hipEventSynchronize(ev1));
        float ms = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&ms, ev0, ev1));
        int total_launches = num_waves * reps;
        float lat_us = (ms * 1000.0f) / (float)total_launches;
        double total_syn = syn_per_wave * (double)total_launches;
        double t_syn = total_syn / ((double)ms * 1e-3 * 1e12);
        lats[it] = lat_us;
        tsyns[it] = t_syn;
    }

    CHECK_HIP(hipEventDestroy(ev0));
    CHECK_HIP(hipEventDestroy(ev1));

    std::vector<size_t> order(iters);
    for (int i = 0; i < iters; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return tsyns[a] < tsyns[b]; });

    int start_idx = (iters >= 4) ? 1 : 0;
    int count = iters - start_idx;

    float lat_min = lats[order[iters - 1]];
    float lat_max = lats[order[start_idx]];
    double tsyn_min = tsyns[order[start_idx]];
    double tsyn_max = tsyns[order[iters - 1]];

    double sum_lat = 0.0, sum_tsyn = 0.0;
    for (int i = start_idx; i < iters; ++i) {
        sum_lat += lats[order[i]];
        sum_tsyn += tsyns[order[i]];
    }
    float lat_mean = (float)(sum_lat / count);
    double tsyn_mean = sum_tsyn / count;

    double var_lat = 0.0, var_tsyn = 0.0;
    for (int i = start_idx; i < iters; ++i) {
        double dl = lats[order[i]] - lat_mean;
        var_lat += dl * dl;
        double dt = tsyns[order[i]] - tsyn_mean;
        var_tsyn += dt * dt;
    }
    float lat_std = (float)std::sqrt(var_lat / count);
    double tsyn_std = std::sqrt(var_tsyn / count);

    return {lat_mean, lat_std, lat_min, lat_max, tsyn_mean, tsyn_std, tsyn_min, tsyn_max};
}

template <int CLUSTER_SIZE, bool USE_DUAL_BUFFER = false>
static nlohmann::json run_ring_benchmark_f16(
    const char* tag,
    uint32_t Cin, uint32_t Cout, uint32_t B, uint32_t M,
    const void* d_X,
    RingBufferPool& pool,
    bool run_validation = true
) {
    const uint32_t B_vec16 = B / 16;
    const uint32_t num_tiles = (M + 63) / 64;
    const size_t x_bytes = (size_t)num_tiles * (size_t)Cin * 64 * sizeof(__half);

    const size_t wave_w_u4  = (size_t)(Cout / 16) * (B / 16) * 32;
    const size_t wave_c_u32 = (size_t)(Cout / CLUSTER_SIZE) * B;
    const size_t wave_y_h   = (size_t)Cout * num_tiles * 64;

    const int max_w_waves = (int)((RingBufferPool::W_POOL_BYTES / sizeof(uint4)) / wave_w_u4);
    const int max_c_waves = (int)((RingBufferPool::C_POOL_BYTES / sizeof(uint32_t)) / wave_c_u32);
    const int num_waves   = std::min(16, std::min(max_w_waves, max_c_waves));
    const int y_slots     = std::max<int>(1, (int)((RingBufferPool::Y_POOL_BYTES / sizeof(__half)) / wave_y_h));

    auto h_C = generate_indices(Cin, (Cout / CLUSTER_SIZE) * num_waves, B, 42ULL + CLUSTER_SIZE * 100 + B + Cin);
    CHECK_HIP(hipMemcpy(pool.d_C_pool, h_C.data(), h_C.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipDeviceSynchronize());

    auto launch_pass = [&]() {
        for (int w = 0; w < num_waves; ++w) {
            __half* y_ptr = pool.d_Y_pool + (w % y_slots) * wave_y_h;
            const __half* x_ptr = reinterpret_cast<const __half*>(d_X);
            const uint32_t* c_ptr = pool.d_C_pool + w * wave_c_u32;
            const uint4* w_ptr = pool.d_W_pool + w * wave_w_u4;
            if constexpr (USE_DUAL_BUFFER) {
                launch_k44_dual_buffer_f16_forward<CLUSTER_SIZE, false>(
                    y_ptr, x_ptr, c_ptr, w_ptr, pool.d_bias, M, Cin, Cout, B_vec16, nullptr
                );
            } else {
                launch_k44_f16_forward<CLUSTER_SIZE, false>(
                    y_ptr, x_ptr, c_ptr, w_ptr, pool.d_bias, M, Cin, Cout, B_vec16, nullptr
                );
            }
        }
    };

    double syn_per_wave = (double)Cout * (double)B * (double)M;
    auto stats = measure_ring_stats(launch_pass, num_waves, syn_per_wave, 4);

    double max_diff = 0.0;
    bool fals_ok = true;
    if (run_validation) {
        fals_ok = run_falsification_wave0_f16<CLUSTER_SIZE, USE_DUAL_BUFFER>(
            Cin, Cout, B, d_X, h_C, pool, max_diff
        );
    }

    float best_lat = stats.lat_mean_us;
    double tsyns = stats.tsyns_mean;
    double tflops = tsyns * 2.0;

    return nlohmann::json{
        {"format", "FP16"},
        {"tag", tag},
        {"variant", USE_DUAL_BUFFER ? "DualBuffer" : (Cin > 33554432u ? "Global64" : "SingleBuffer32")},
        {"S", CLUSTER_SIZE},
        {"Cin", Cin},
        {"Cout", Cout},
        {"B", B},
        {"M", M},
        {"sparsity_pct", 100.0 * (1.0 - (double)B / (double)Cin)},
        {"x_vram_gb", (double)x_bytes / (1024.0 * 1024.0 * 1024.0)},
        {"lat_res_us", stats.lat_mean_us},
        {"lat_str_us", stats.lat_mean_us},
        {"best_lat_us", best_lat},
        {"best_tsyns", tsyns},
        {"best_tflops", tflops},
        {"lat_mean_us", stats.lat_mean_us},
        {"lat_std_us", stats.lat_std_us},
        {"lat_min_us", stats.lat_min_us},
        {"lat_max_us", stats.lat_max_us},
        {"tsyns_mean", stats.tsyns_mean},
        {"tsyns_std", stats.tsyns_std},
        {"tsyns_min", stats.tsyns_min},
        {"tsyns_max", stats.tsyns_max},
        {"max_diff", max_diff},
        {"falsification_passed", fals_ok}
    };
}
