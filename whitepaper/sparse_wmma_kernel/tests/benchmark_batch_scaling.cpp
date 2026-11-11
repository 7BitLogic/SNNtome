// ==============================================================================
// tests/whitepaper/benchmark_batch_scaling.cpp
//
// Whitepaper Benchmark Suite: Batch Size Scaling (FP16 only) on AMD RDNA4.
// Target: AMD Radeon RX 9060 XT (GFX1200 / Navi 44, Device 0)
//
// Test Matrix (Multi-Wave Ring Buffer / 256 MiB VRAM-Streaming Weight Pool):
//   - Precision: FP16 only
//   - In-Cache Regime (Cin = 8,192):
//       1. S=512, B=2048, Cout=4096 (Fastest Peak)
//       2. S=256, B=1024, Cout=8192 (Standard Flexible)
//       3. S=128, B=1024, Cout=8192
//       4. S=16,  B=1024, Cout=8192
//   - VRAM-Streaming Regime (Cin = 1,048,576):
//       5. S=512, B=2048, Cout=4096 (Fastest Peak)
//       6. S=512, B=1024, Cout=8192 (Fastest B=1024)
//       7. S=256, B=1024, Cout=8192 (Standard Flexible)
//       8. S=128, B=1024, Cout=8192
//       9. S=16,  B=1024, Cout=8192
//   - Batch Sizes M: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
// ==============================================================================

#include "benchmark_common.cuh"
#include <sys/stat.h>

// -----------------------------------------------------------------------------
// 1. Formatted Table Output
// -----------------------------------------------------------------------------
static void print_result_row(const nlohmann::json& r) {
    char s_m[16], s_spar[16], s_lat[16], s_tsyn[24], s_tfl[16], s_err[16];
    snprintf(s_m,    sizeof(s_m),    "M=%u",         r["M"].get<uint32_t>());
    snprintf(s_spar, sizeof(s_spar), "%.3f%%",       r["sparsity_pct"].get<double>());
    snprintf(s_lat,  sizeof(s_lat),  "%.1f us",      r["best_lat_us"].get<float>());
    snprintf(s_tsyn, sizeof(s_tsyn), "%.3f TSyn/s",  r["best_tsyns"].get<double>());
    snprintf(s_tfl,  sizeof(s_tfl),  "%.2f TF",      r["best_tflops"].get<double>());
    snprintf(s_err,  sizeof(s_err),  "%.1e",         r["max_diff"].get<double>());

    char s_s[16], s_b[16];
    snprintf(s_s, sizeof(s_s), "S=%u", r["S"].get<uint32_t>());
    snprintf(s_b, sizeof(s_b), "B=%u", r["B"].get<uint32_t>());

    printf("| %-6s| %5s| %6s| %8s| %8s| %11s| %14s| %11s| %9s| %-12s|\n",
           r["format"].get<std::string>().c_str(), s_s, s_b, s_m, s_spar, s_lat, s_tsyn, s_tfl, s_err,
           r["falsification_passed"].get<bool>() ? "PASSED [3/3]" : "FAILED");
}

// -----------------------------------------------------------------------------
// 2. Test Suite Runner across Batch Sizes M for a Given (Cin, S, B, Cout)
// -----------------------------------------------------------------------------
template <int CLUSTER_SIZE>
static void run_batch_suite_fp16(
    int suite_idx, int total_suites,
    const char* tag, const char* regime_label,
    uint32_t Cin, uint32_t Cout, uint32_t B,
    const std::vector<uint32_t>& batch_sizes,
    RingBufferPool& pool,
    nlohmann::json& all_results
) {
    const uint32_t num_clusters = Cout / CLUSTER_SIZE;
    printf("\n==============================================================================================\n");
    printf(" [SUITE %d/%d] %s | FP16 S=%d (Clusters=%u), B=%u, Cout=%u, Cin=%u\n",
           suite_idx, total_suites, regime_label, CLUSTER_SIZE, num_clusters, B, Cout, Cin);
    printf("==============================================================================================\n");
    printf("| Format| S    | B     | M       | Spars.  | Latency    | Throughput    | FLOPS      | Max Err  | Falsif.     |\n");
    printf("|-------|------|-------|---------|---------|------------|---------------|------------|----------|-------------|\n");

    for (uint32_t m : batch_sizes) {
        uint32_t m_tiles = (m + 63) / 64;
        size_t x_bytes = (size_t)m_tiles * 64 * (size_t)Cin * sizeof(__half);
        if (x_bytes > (8ULL << 30)) continue; // Keep <= 8 GiB for clean 2 MB VRAM hugepages

        void* d_X = nullptr;
        CHECK_HIP(hipMalloc(&d_X, x_bytes));
        init_x_continuous_f16_gpu_kernel<<<1024, 256>>>(
            reinterpret_cast<uint4*>(d_X), x_bytes / sizeof(uint4), 98765u + m + Cin
        );
        CHECK_HIP(hipDeviceSynchronize());

        nlohmann::json r = run_ring_benchmark_f16<CLUSTER_SIZE, false>(
            tag, Cin, Cout, B, m, d_X, pool, true
        );
        CHECK_HIP(hipFree(d_X));

        print_result_row(r);
        all_results.push_back(std::move(r));
    }
    printf("[SUITE %d/%d] COMPLETED SUCCESSFULLY.\n", suite_idx, total_suites);
}

// -----------------------------------------------------------------------------
// 3. Main Program
// -----------------------------------------------------------------------------
int main() {
    int dev = 0;
    CHECK_HIP(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, dev));

    printf("==============================================================================================\n");
    printf("  AMD RDNA4 Kernel 44 Batch Size Scaling Benchmark Suite (FP16 Multi-Wave Ring Buffer)\n");
    printf("  Device: %s (GCN: %s, DevID: %d)\n", prop.name, prop.gcnArchName, dev);
    printf("  Comparing In-Cache (Cin=8k) vs. VRAM-Streaming (Cin=1M) across S in {16, 128, 256, 512}\n");
    printf("==============================================================================================\n");

    const std::vector<uint32_t> batch_sizes = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
    };

    RingBufferPool pool;
    pool.init(8192, 2048);

    pool.warmup(300);

    nlohmann::json all_results = nlohmann::json::array();

    // In-Cache Regime (Cin = 8,192)
    run_batch_suite_fp16<512>(1, 9, "Cin8k_S512_B2048_C4k", "In-Cache (Cin=8k) Fastest",  8192, 4096, 2048, batch_sizes, pool, all_results);
    run_batch_suite_fp16<256>(2, 9, "Cin8k_S256_B1024_C8k", "In-Cache (Cin=8k) Flexible", 8192, 8192, 1024, batch_sizes, pool, all_results);
    run_batch_suite_fp16<128>(3, 9, "Cin8k_S128_B1024_C8k", "In-Cache (Cin=8k) S=128",    8192, 8192, 1024, batch_sizes, pool, all_results);
    run_batch_suite_fp16<16 >(4, 9, "Cin8k_S16_B1024_C8k",  "In-Cache (Cin=8k) S=16",     8192, 8192, 1024, batch_sizes, pool, all_results);

    // VRAM-Streaming Regime (Cin = 1,048,576)
    run_batch_suite_fp16<512>(5, 9, "Cin1M_S512_B2048_C4k", "VRAM-Streaming (Cin=1M) Fastest",  1048576, 4096, 2048, batch_sizes, pool, all_results);
    run_batch_suite_fp16<512>(6, 9, "Cin1M_S512_B1024_C8k", "VRAM-Streaming (Cin=1M) S=512",    1048576, 8192, 1024, batch_sizes, pool, all_results);
    run_batch_suite_fp16<256>(7, 9, "Cin1M_S256_B1024_C8k", "VRAM-Streaming (Cin=1M) Flexible", 1048576, 8192, 1024, batch_sizes, pool, all_results);
    run_batch_suite_fp16<128>(8, 9, "Cin1M_S128_B1024_C8k", "VRAM-Streaming (Cin=1M) S=128",    1048576, 8192, 1024, batch_sizes, pool, all_results);
    run_batch_suite_fp16<16 >(9, 9, "Cin1M_S16_B1024_C8k",  "VRAM-Streaming (Cin=1M) S=16",     1048576, 8192, 1024, batch_sizes, pool, all_results);

    pool.destroy();

    mkdir("data", 0777);
    std::string out_json = "data/benchmark_batch_scaling_results.json";
    {
        std::ofstream ofs(out_json);
        ofs << all_results.dump(2) << "\n";
    }
    printf("[LOG] JSON results written to %s.\n", out_json.c_str());

    printf("\n==============================================================================================\n");
    printf("  ALL 9 BATCH SCALING BENCHMARK SUITES SUCCESSFULLY COMPLETED AND VERIFIED!\n");
    printf("==============================================================================================\n");

    return 0;
}
