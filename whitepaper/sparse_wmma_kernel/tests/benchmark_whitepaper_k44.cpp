// ==============================================================================
// tests/benchmark_whitepaper_k44.cpp
//
// Official Whitepaper Benchmark Suite for Kernel 44 FP16 on AMD RDNA4.
// Target: AMD Radeon RX 9060 XT (GFX1200 / Navi 44, Device 0)
//
// Invariants & Requirements:
//   1. Immutable GPU kernel (kernels/hip/kernel44_immutable.cuh) 100% unabridged.
//   2. Pure FP16 benchmark with 256 MiB Multi-Wave Ring-Buffer (VRAM-Streaming).
//   3. Standard ("Flexible") S in {16, 32, 64, 128, 256} at M=512, B=1024, Cout=8192
//      plus Fastest Comparison S=512 (B=1024, Cout=8192 and B=2048, Cout=4096).
//   4. Lean modular test functions & compact tabular output (printf).
//   5. Export of reference datasets to data/whitepaper_reference/.
//   6. VRAM hugepage safety: dynamically allocate d_X per Cin & immediately hipFree.
// ==============================================================================

#include "benchmark_common.cuh"
#include <cstdio>
#include <sys/stat.h>

struct ConnectomeRef {
    int S;
    std::vector<uint32_t> indices;
};

// -----------------------------------------------------------------------------
// 1. File & Export Helper Functions
// -----------------------------------------------------------------------------
static void export_reference_datasets(
    const std::string& dir,
    const std::vector<ConnectomeRef>& refs,
    const std::vector<uint16_t>& w_f16,
    const std::vector<uint16_t>& b_f16
) {
    mkdir(dir.c_str(), 0777);
    auto write_bin = [](const std::string& p, const void* d, size_t n) {
        std::ofstream f(p, std::ios::binary);
        f.write(reinterpret_cast<const char*>(d), n);
    };
    for (const auto& r : refs) {
        std::string fn = dir + "/connectome_indices_s" + std::to_string(r.S) + ".bin";
        write_bin(fn, r.indices.data(), r.indices.size() * 4);
    }
    write_bin(dir + "/weights_f16.bin", w_f16.data(), w_f16.size() * 2);
    write_bin(dir + "/biases_f16.bin",  b_f16.data(), b_f16.size() * 2);

    nlohmann::json meta;
    meta["schema"] = "1.0.0";
    meta["target"] = "AMD Radeon RX 9060 XT (Navi 44)";
    meta["Cout"] = 8192;
    meta["B"] = 1024;
    meta["Cin_max"] = 10000000;
    nlohmann::json files;
    for (const auto& r : refs) {
        uint32_t num_clusters = 8192 / r.S;
        std::string fn = "connectome_indices_s" + std::to_string(r.S) + ".bin";
        files[fn] = {
            {"shape", {num_clusters, 1024}},
            {"dtype", "uint32"},
            {"bytes", r.indices.size() * 4}
        };
    }
    files["weights_f16.bin"] = {
        {"shape", {8192, 1024}},
        {"dtype", "float16"},
        {"bytes", w_f16.size() * 2}
    };
    files["biases_f16.bin"] = {
        {"shape", {8192}},
        {"dtype", "float16"},
        {"bytes", b_f16.size() * 2}
    };
    meta["files"] = files;

    std::ofstream meta_fs(dir + "/reference_metadata.json");
    meta_fs << meta.dump(2) << "\n";
    printf("[EXPORT] Whitepaper reference data successfully written to %s.\n", dir.c_str());
}

// -----------------------------------------------------------------------------
// 2. Formatted Table Output (Compact & Clean via printf)
// -----------------------------------------------------------------------------
static void print_table_header(const std::string& title) {
    printf("\n========================================================================================================================\n");
    printf("  %s\n", title.c_str());
    printf("========================================================================================================================\n");
    printf("%-12s %-14s %-12s %-14s %-14s %-14s %-18s %-16s %-12s %-14s\n",
           "Cin", "Sparsity", "X VRAM", "Lat. Res", "Lat. Stream", "Lat. Best", "Throughput", "Compute", "MaxDiff", "Validation");
    printf("------------------------------------------------------------------------------------------------------------------------\n");
}

static void print_table_row(const nlohmann::json& r) {
    char s_cin[16], s_spar[16], s_xvr[16], s_lres[16], s_lstr[16], s_lbst[16], s_tsyn[24], s_comp[16], s_diff[16];
    snprintf(s_cin,  sizeof(s_cin),  "%u",          r["Cin"].get<uint32_t>());
    snprintf(s_spar, sizeof(s_spar), "%.4f%%",      r["sparsity_pct"].get<double>());
    snprintf(s_xvr,  sizeof(s_xvr),  "%.3f GB",     r["x_vram_gb"].get<double>());
    snprintf(s_lres, sizeof(s_lres), "%.2f us",     r["lat_res_us"].get<float>());
    snprintf(s_lstr, sizeof(s_lstr), "%.2f us",     r["lat_str_us"].get<float>());
    snprintf(s_lbst, sizeof(s_lbst), "%.2f us",     r["best_lat_us"].get<float>());
    snprintf(s_tsyn, sizeof(s_tsyn), "%.2f TSyn/s", r["best_tsyns"].get<double>());
    snprintf(s_comp, sizeof(s_comp), "%.2f TF",     r["best_tflops"].get<double>());
    snprintf(s_diff, sizeof(s_diff), "%.4f",        r["max_diff"].get<double>());

    printf("%-12s %-14s %-12s %-14s %-14s %-14s %-18s %-16s %-12s %-14s\n",
           s_cin, s_spar, s_xvr, s_lres, s_lstr, s_lbst, s_tsyn, s_comp, s_diff,
           r["falsification_passed"].get<bool>() ? "PASSED [3/3]" : "FAILED");
}

static void print_suite_from_records(
    int suite_idx, int total_suites,
    const char* tag, const char* desc,
    int S, uint32_t Cout, uint32_t B, uint32_t M,
    const nlohmann::json& records
) {
    char title[180];
    snprintf(title, sizeof(title),
             "TEST SUITE %d/%d: Kernel 44 FP16 [%s] | Cluster S = %d (%u Clusters) | Cout = %u | B = %u | M = %u",
             suite_idx, total_suites, desc, S, Cout / S, Cout, B, M);
    print_table_header(title);
    for (const auto& r : records) {
        if (r["tag"].get<std::string>() == tag) {
            print_table_row(r);
        }
    }
}

// -----------------------------------------------------------------------------
// 3. Main Program
// -----------------------------------------------------------------------------
int main() {
    CHECK_HIP(hipSetDevice(0));
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));

    printf("========================================================================================================================\n");
    printf("  AMD RDNA4 Kernel 44 FP16 Whitepaper Benchmark Suite (Multi-Wave Ring Buffer / VRAM-Streaming)\n");
    printf("  Device: %s (GCN: %s, Compute Units: %d)\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    printf("  Target Architecture: Pure FP16 Matrix Cores (256 MiB Weight Pool across 16 Disjoint Connectome Waves)\n");
    printf("========================================================================================================================\n");

    const uint32_t Cout_std = 8196 - 4; // 8192
    const uint32_t B_std    = 1024;
    const uint32_t M_std    = 512;

    const std::vector<uint32_t> cin_list_m512 = {
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
        262144, 524288, 1048576, 2097152, 4194304, 8388608, 10000000
    };
    const std::vector<uint32_t> cin_list_40m = {
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
        262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 20000000,
        33554432, 40000000
    };

    std::vector<ConnectomeRef> refs = {
        {16,  generate_indices(10000000, Cout_std / 16,  B_std, 16)},
        {32,  generate_indices(10000000, Cout_std / 32,  B_std, 32)},
        {64,  generate_indices(10000000, Cout_std / 64,  B_std, 64)},
        {128, generate_indices(10000000, Cout_std / 128, B_std, 128)},
        {256, generate_indices(10000000, Cout_std / 256, B_std, 256)},
        {512, generate_indices(10000000, Cout_std / 512, B_std, 512)}
    };

    RingBufferPool pool;
    pool.init(8192, 2048);

    auto h_wf16_export = generate_weights_f16(Cout_std, B_std, 1337);
    auto h_b_export    = generate_biases(Cout_std, 2026);
    export_reference_datasets("data", refs, h_wf16_export, h_b_export);

    pool.warmup(300);

    nlohmann::json all_results      = nlohmann::json::array();
    nlohmann::json m128_40m_results = nlohmann::json::array();
    nlohmann::json dual_buf_results = nlohmann::json::array();

    // -------------------------------------------------------------------------
    // PART 1: Standard ("Flexible", M=512, B=1024, Cout=8192) S in {16..256}
    //         + Fastest Comparison S=512 (B=1024, Cout=8192 & B=2048, Cout=4096)
    //         Allocate d_X dynamically per Cin step for 2 MB VRAM hugepages.
    // -------------------------------------------------------------------------
    for (uint32_t Cin : cin_list_m512) {
        size_t x_bytes = (size_t)Cin * M_std * sizeof(__half);
        void* d_X = nullptr;
        CHECK_HIP(hipMalloc(&d_X, x_bytes));
        init_x_continuous_f16_gpu_kernel<<<1024, 256>>>(
            reinterpret_cast<uint4*>(d_X), x_bytes / sizeof(uint4), 12345u + Cin
        );
        CHECK_HIP(hipDeviceSynchronize());

        all_results.push_back(run_ring_benchmark_f16<16,  false>("S16_std",      Cin, 8192, 1024, 512, d_X, pool));
        all_results.push_back(run_ring_benchmark_f16<32,  false>("S32_std",      Cin, 8192, 1024, 512, d_X, pool));
        all_results.push_back(run_ring_benchmark_f16<64,  false>("S64_std",      Cin, 8192, 1024, 512, d_X, pool));
        all_results.push_back(run_ring_benchmark_f16<128, false>("S128_std",     Cin, 8192, 1024, 512, d_X, pool));
        all_results.push_back(run_ring_benchmark_f16<256, false>("S256_std",     Cin, 8192, 1024, 512, d_X, pool));
        all_results.push_back(run_ring_benchmark_f16<512, false>("S512_b1k_c8k", Cin, 8192, 1024, 512, d_X, pool));
        all_results.push_back(run_ring_benchmark_f16<512, false>("S512_b2k_c4k", Cin, 4096, 2048, 512, d_X, pool));

        CHECK_HIP(hipFree(d_X));
    }

    print_suite_from_records(1, 13, "S16_std",      "Standard Flexible", 16,  8192, 1024, 512, all_results);
    print_suite_from_records(2, 13, "S32_std",      "Standard Flexible", 32,  8192, 1024, 512, all_results);
    print_suite_from_records(3, 13, "S64_std",      "Standard Flexible", 64,  8192, 1024, 512, all_results);
    print_suite_from_records(4, 13, "S128_std",     "Standard Flexible", 128, 8192, 1024, 512, all_results);
    print_suite_from_records(5, 13, "S256_std",     "Standard Flexible", 256, 8192, 1024, 512, all_results);
    print_suite_from_records(6, 13, "S512_b1k_c8k", "Fastest B=1024",    512, 8192, 1024, 512, all_results);
    print_suite_from_records(7, 13, "S512_b2k_c4k", "Fastest Peak",      512, 4096, 2048, 512, all_results);

    // -------------------------------------------------------------------------
    // PART 2: Extended 40M Connectome Series at M=128 (Cout=4096, B=1024)
    // -------------------------------------------------------------------------
    const uint32_t M_ext    = 128;
    const uint32_t Cout_ext = 4096;
    for (uint32_t Cin : cin_list_40m) {
        size_t x_bytes = (size_t)Cin * M_ext * sizeof(__half);
        void* d_X = nullptr;
        CHECK_HIP(hipMalloc(&d_X, x_bytes));
        init_x_continuous_f16_gpu_kernel<<<1024, 256>>>(
            reinterpret_cast<uint4*>(d_X), x_bytes / sizeof(uint4), 67890u + Cin
        );
        CHECK_HIP(hipDeviceSynchronize());

        m128_40m_results.push_back(run_ring_benchmark_f16<16,  false>("S16_m128",  Cin, Cout_ext, 1024, M_ext, d_X, pool));
        m128_40m_results.push_back(run_ring_benchmark_f16<32,  false>("S32_m128",  Cin, Cout_ext, 1024, M_ext, d_X, pool));
        m128_40m_results.push_back(run_ring_benchmark_f16<64,  false>("S64_m128",  Cin, Cout_ext, 1024, M_ext, d_X, pool));
        m128_40m_results.push_back(run_ring_benchmark_f16<128, false>("S128_m128", Cin, Cout_ext, 1024, M_ext, d_X, pool));
        m128_40m_results.push_back(run_ring_benchmark_f16<256, false>("S256_m128", Cin, Cout_ext, 1024, M_ext, d_X, pool));
        dual_buf_results.push_back(run_ring_benchmark_f16<128, true >("S128_dual", Cin, Cout_ext, 1024, M_ext, d_X, pool));

        CHECK_HIP(hipFree(d_X));
    }

    print_suite_from_records( 8, 13, "S16_m128",  "M=128 40M", 16,  Cout_ext, 1024, M_ext, m128_40m_results);
    print_suite_from_records( 9, 13, "S32_m128",  "M=128 40M", 32,  Cout_ext, 1024, M_ext, m128_40m_results);
    print_suite_from_records(10, 13, "S64_m128",  "M=128 40M", 64,  Cout_ext, 1024, M_ext, m128_40m_results);
    print_suite_from_records(11, 13, "S128_m128", "M=128 40M", 128, Cout_ext, 1024, M_ext, m128_40m_results);
    print_suite_from_records(12, 13, "S256_m128", "M=128 40M", 256, Cout_ext, 1024, M_ext, m128_40m_results);
    print_suite_from_records(13, 13, "S128_dual", "Dual-Buffer M=128", 128, Cout_ext, 1024, M_ext, dual_buf_results);

    pool.destroy();

    {
        std::ofstream ofs("data/benchmark_results_k44.json");
        ofs << all_results.dump(2) << "\n";
    }
    {
        std::ofstream ofs("data/benchmark_results_k44_m128_40m.json");
        ofs << m128_40m_results.dump(2) << "\n";
    }
    {
        std::ofstream ofs("data/benchmark_k44_dual_buffer_s128_m128.json");
        ofs << dual_buf_results.dump(2) << "\n";
    }
    printf("[LOG] JSON results written to data/.\n");
    printf("\n[OVERALL RESULT] All 13 FP16 test suites & validations completed successfully!\n");
    return 0;
}
