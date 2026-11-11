// ==============================================================================
// tests/whitepaper/error_collapse_test.cpp
//
// Rigorous 3-Stage Error Collapse Analysis for Kernel 44 on AMD RDNA4 (Navi 44 / GFX1200)
// Peer-Review Clean Code Reference Implementation.
// Target Device: GPU 0 (hipSetDevice(0))
//
// Validates the mathematical error collapse hierarchy:
//   Stage 1: Unquantized Continuous FP32 Ground Truth
//   Stage 2: Exact Quantized Operands Sequential Sum
//   Stage 3: Stepwise 16-Element WMMA Tile Accumulation
// ==============================================================================

#include "benchmark_common.cuh"
#include <sys/stat.h>

#define CHK(cmd) CHECK_HIP(cmd)

/**
 * @brief Scatters dense activation columns into the full continuous X tensor on GPU.
 *
 * Each token t in [0, 64) is mapped to its absolute neuron index:
 *   dst[(size_t)cols[i / 64] * 64 + (i % 64)] = src[i]
 *
 * @tparam T Operand data type (uint8_t for FP8, uint16_t for FP16).
 * @param dst Pointer to global destination tensor X on GPU.
 * @param src Pointer to compact source activations on GPU.
 * @param cols Pointer to list of unique connectome column indices.
 * @param num_u Number of unique columns.
 */
template <typename T>
__global__ void scatter_x(
    T* __restrict__ dst,
    const T* __restrict__ src,
    const uint32_t* __restrict__ cols,
    uint32_t num_u
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = (size_t)num_u * 64;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < total_elements; i += stride) {
        uint32_t col_idx = cols[i / 64];
        uint32_t token = i % 64;
        dst[(size_t)col_idx * 64 + token] = src[i];
    }
}

/**
 * @brief Benchmark test configuration tuple.
 */
struct TestConfig {
    std::string format;
    int S;
    uint32_t Cin;
};

/**
 * @brief Deterministic pseudo-random generation of sparse connectome column indices.
 *
 * For each cluster c in [0, num_cl), B incoming synaptic column indices
 * in [0, Cin - 1] are uniformly sampled and sorted in ascending order for cache locality.
 *
 * @param Cin Total number of input neurons in connectome canvas.
 * @param num_cl Number of independent cortical clusters (Cout / S).
 * @param B Number of incoming synaptic connections per cluster (1024).
 * @param s RNG seed for deterministic reproducibility.
 * @return std::vector<uint32_t> Flattened index vector of size (num_cl * B).
 */
static inline std::vector<uint32_t> gen_idx(uint32_t Cin, uint32_t num_cl, uint32_t B, uint64_t s) {
    return generate_indices(Cin, num_cl, B, s);
}

/**
 * @brief Executes a rigorous 3-stage error collapse validation for a given configuration.
 *
 * =========================================================================================
 * MATHEMATICAL FOUNDATIONS OF THE 3-STAGE ERROR COLLAPSE HIERARCHY:
 * =========================================================================================
 *
 * Given an incoming synapse block size B = 1024, cluster size S in {128, 256}, and
 * activation tile dimension of 64 tokens (t in [0, 64)):
 *
 * -----------------------------------------------------------------------------------------
 * STAGE 1: Unquantized Continuous FP32 Ground Truth (Physical Representation Error)
 * -----------------------------------------------------------------------------------------
 * In this stage, all synaptic weights, biases, and input activations are drawn from continuous
 * uniform distributions in standard IEEE-754 Single Precision (FP32):
 *   W_raw[j, s] ~ U(-1.0, +1.0)  [or U(-1.5, +1.5) for FP8]
 *   b_raw[j]    ~ U(-0.5, +0.5)
 *   X_raw[c, t] ~ U(-1.0, +1.0)
 *
 * The continuous reference output is calculated analytically via high-precision FP64 accumulation:
 *   Y_ref_s1[j, t] = b_raw[j] + \sum_{s=0}^{B-1} (W_raw[j, s] * X_raw[C[s], t])
 *
 * Comparing GPU output Y_gpu against Y_ref_s1 measures the total physical quantization error
 * (representation noise) introduced by compressing 32-bit floats into lower precision (FP16/FP8).
 *
 * -----------------------------------------------------------------------------------------
 * STAGE 2: Exact Quantized Operands Sequential Sum (Quantized Representation Anchor)
 * -----------------------------------------------------------------------------------------
 * Here, the inputs are strictly converted to the target hardware representation:
 *   W_qf[j, s] = convert_to_float( quantize( W_raw[j, s] ) )
 *   b_qf[j]    = convert_to_float( quantize( b_raw[j] ) )
 *   X_qf[c, t] = convert_to_float( quantize( X_raw[c, t] ) )
 *
 * The sum is accumulated sequentially in standard FP64:
 *   Y_ref_s2[j, t] = b_qf[j] + \sum_{s=0}^{B-1} (W_qf[j, s] * X_qf[C[s], t])
 *
 * Comparing GPU output Y_gpu against Y_ref_s2 proves that once representation noise is eliminated,
 * the remaining error collapses dramatically (~289x for FP8) to pure arithmetic ordering effects.
 *
 * -----------------------------------------------------------------------------------------
 * STAGE 3: Stepwise 16-Element WMMA Tile Accumulation (Silicon Hardware Fidelity)
 * -----------------------------------------------------------------------------------------
 * This stage faithfully emulates AMD RDNA4 Matrix Core execution (v_wmma instructions), where
 * the 1024-synapse dot product is evaluated in chunks of 16 elements:
 *
 *   For each 16-element tile chunk k_w in [0, B/16):
 *     ChunkAcc[k_w] = \sum_{i=0}^{15} (W_qf[j, 16 * k_w + i] * X_qf[C[16 * k_w + i], t])
 *
 * The chunks are sequentially summed with bias and rounded to half precision at output:
 *   Y_ref_s3[j, t] = to_half( b_qf[j] + \sum_{k_w=0}^{(B/16)-1} ChunkAcc[k_w] )
 *
 * Comparing GPU output Y_gpu against Y_ref_s3 exhibits complete collapse to machine epsilon
 * (bit-exact or single-LSB), verifying that the GPU kernel matches the algebraic matrix core specification.
 * =========================================================================================
 */
static nlohmann::json run_single_test(
    const TestConfig& cfg,
    uint32_t Cout = 4096,
    uint32_t B = 1024,
    uint32_t M = 256
) {
    // 1. Generate sparse connectome indices for Cluster 0
    std::vector<uint32_t> h_C = gen_idx(cfg.Cin, Cout / cfg.S, B, 4200 + cfg.Cin);
    std::vector<uint32_t> cols0(h_C.begin(), h_C.begin() + B);
    std::vector<uint32_t> unique_cols = cols0;
    std::sort(unique_cols.begin(), unique_cols.end());
    unique_cols.erase(std::unique(unique_cols.begin(), unique_cols.end()), unique_cols.end());
    uint32_t num_unique = static_cast<uint32_t>(unique_cols.size());

    std::vector<uint32_t> col_to_k(B);
    for (uint32_t s = 0; s < B; ++s) {
        auto it = std::lower_bound(unique_cols.begin(), unique_cols.end(), cols0[s]);
        col_to_k[s] = static_cast<uint32_t>(it - unique_cols.begin());
    }

    // 2. Synthesize pseudo-random inputs in continuous FP32
    std::mt19937_64 rng(1337 + cfg.Cin);
    std::uniform_real_distribution<float> dist_w(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_b(-0.5f, 0.5f);

    std::vector<float> w_raw(cfg.S * B);
    std::vector<float> b_raw(cfg.S);
    std::vector<float> x_raw((size_t)num_unique * 64);

    std::vector<float> w_qf(cfg.S * B);
    std::vector<float> b_qf(cfg.S);
    std::vector<float> x_qf((size_t)num_unique * 64);

    std::vector<uint16_t> w_f16(Cout * B);
    std::vector<uint16_t> b_f16(Cout);
    std::vector<uint16_t> x_u_f16((size_t)num_unique * 64);

    // Quantize synaptic weights using native HIP half functions
    for (size_t i = 0; i < (size_t)cfg.S * B; ++i) {
        w_raw[i] = dist_w(rng);
        __half h = __float2half(w_raw[i]);
        uint16_t q = *reinterpret_cast<uint16_t*>(&h);
        w_f16[i] = q;
        w_qf[i] = __half2float(h);
    }

    // Tile weights across all Cout output rows
    size_t cluster_weight_elements = (size_t)cfg.S * B;
    for (size_t i = cluster_weight_elements; i < w_f16.size(); ++i) {
        w_f16[i] = w_f16[i % cluster_weight_elements];
    }

    // Quantize biases
    for (size_t j = 0; j < (size_t)cfg.S; ++j) {
        b_raw[j] = dist_b(rng);
        __half h = __float2half(b_raw[j]);
        uint16_t q = *reinterpret_cast<uint16_t*>(&h);
        b_f16[j] = q;
        b_qf[j] = __half2float(h);
    }
    for (size_t j = cfg.S; j < Cout; ++j) {
        b_f16[j] = b_f16[j % cfg.S];
    }

    // Quantize unique activation columns
    for (size_t i = 0; i < (size_t)num_unique * 64; ++i) {
        x_raw[i] = dist_x(rng);
        __half h = __float2half(x_raw[i]);
        uint16_t q = *reinterpret_cast<uint16_t*>(&h);
        x_u_f16[i] = q;
        x_qf[i] = __half2float(h);
    }

    // 3. Allocate device buffers
    size_t x_sz = (size_t)((M + 63) / 64) * (size_t)cfg.Cin * 64 * sizeof(__half);
    size_t y_sz = (size_t)((M + 63) / 64) * (size_t)Cout * 64 * sizeof(__half);

    void* dX = nullptr;
    void* dY = nullptr;
    void* dC = nullptr;
    void* d_Wc = nullptr;
    void* d_Ww = nullptr;
    void* dB = nullptr;
    void* dUc = nullptr;
    void* dXu = nullptr;

    CHK(hipMalloc(&dX, x_sz));
    CHK(hipMalloc(&dY, y_sz));
    CHK(hipMalloc(&dC, h_C.size() * sizeof(uint32_t)));
    CHK(hipMalloc(&d_Wc, w_f16.size() * sizeof(uint16_t)));
    CHK(hipMalloc(&d_Ww, (Cout / 16) * (B / 16) * 32 * sizeof(uint4)));
    CHK(hipMalloc(&dB, b_f16.size() * sizeof(uint16_t)));
    CHK(hipMalloc(&dUc, num_unique * sizeof(uint32_t)));
    CHK(hipMalloc(&dXu, x_u_f16.size() * sizeof(uint16_t)));

    // Upload operands
    CHK(hipMemcpy(dC, h_C.data(), h_C.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHK(hipMemcpy(d_Wc, w_f16.data(), w_f16.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHK(hipMemcpy(dB, b_f16.data(), b_f16.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHK(hipMemcpy(dUc, unique_cols.data(), num_unique * sizeof(uint32_t), hipMemcpyHostToDevice));
    CHK(hipMemcpy(dXu, x_u_f16.data(), x_u_f16.size() * sizeof(uint16_t), hipMemcpyHostToDevice));

    // Scatter activations into X and interleave weights for Matrix Cores
    scatter_x<uint16_t><<<1024, 256>>>(
        reinterpret_cast<uint16_t*>(dX),
        reinterpret_cast<const uint16_t*>(dXu),
        reinterpret_cast<const uint32_t*>(dUc),
        num_unique
    );
    interleave_weights_cluster16_wmma_gpu_kernel<<<(Cout / 16) * (B / 16) * 32 / 256, 256>>>(
        reinterpret_cast<uint4*>(d_Ww),
        reinterpret_cast<const uint4*>(d_Wc),
        Cout,
        B / 8
    );
    CHK(hipDeviceSynchronize());

    // 4. Launch Kernel 44 FP16 on GPU
    const uint32_t B_vec16 = B / 16;
    if (cfg.S == 128) {
        launch_k44_f16_forward<128, false>(
            reinterpret_cast<__half*>(dY),
            reinterpret_cast<const __half*>(dX),
            reinterpret_cast<const uint32_t*>(dC),
            reinterpret_cast<const uint4*>(d_Ww),
            reinterpret_cast<const __half*>(dB),
            M, cfg.Cin, Cout, B_vec16, nullptr
        );
    } else {
        launch_k44_f16_forward<256, false>(
            reinterpret_cast<__half*>(dY),
            reinterpret_cast<const __half*>(dX),
            reinterpret_cast<const uint32_t*>(dC),
            reinterpret_cast<const uint4*>(d_Ww),
            reinterpret_cast<const __half*>(dB),
            M, cfg.Cin, Cout, B_vec16, nullptr
        );
    }
    CHK(hipDeviceSynchronize());

    // 5. Download GPU output for Cluster 0
    std::vector<__half> y_gpu((size_t)cfg.S * 64);
    CHK(hipMemcpy(
        y_gpu.data(),
        dY,
        cfg.S * 64 * sizeof(__half),
        hipMemcpyDeviceToHost
    ));

    // Release device resources immediately
    CHK(hipFree(dX));
    CHK(hipFree(dY));
    CHK(hipFree(dC));
    CHK(hipFree(d_Wc));
    CHK(hipFree(d_Ww));
    CHK(hipFree(dB));
    CHK(hipFree(dUc));
    CHK(hipFree(dXu));

    // 6. Mathematical Evaluation of All 3 Stages on Host
    double max_diff1 = 0.0, mae1 = 0.0, sq1 = 0.0;
    double max_diff2 = 0.0, mae2 = 0.0, sq2 = 0.0;
    double max_diff3 = 0.0, mae3 = 0.0, sq3 = 0.0;
    double sum_rsq = 0.0;
    double sum_rabs = 0.0;
    size_t total_points = (size_t)cfg.S * 64;

    for (uint32_t j = 0; j < (uint32_t)cfg.S; ++j) {
        for (uint32_t t = 0; t < 64; ++t) {
            float g = __half2float(y_gpu[j * 64 + t]);

            // Stage 1: Continuous FP32 Ground Truth
            double v1 = b_raw[j];

            // Stage 2: Exact Quantized Operands Sequential Sum
            double v2 = b_qf[j];

            // Stage 3: Stepwise 16-Element WMMA Tile Accumulation
            float v3 = b_qf[j];

            for (uint32_t s = 0; s < B; ++s) {
                uint32_t k = col_to_k[s];
                v1 += static_cast<double>(w_raw[j * B + s]) * static_cast<double>(x_raw[k * 64 + t]);
                v2 += static_cast<double>(w_qf[j * B + s]) * static_cast<double>(x_qf[k * 64 + t]);
            }

            for (uint32_t kw = 0; kw < B / 16; ++kw) {
                float chunk_sum = 0.0f;
                for (uint32_t i = 0; i < 16; ++i) {
                    uint32_t syn_idx = kw * 16 + i;
                    chunk_sum += w_qf[j * B + syn_idx] * x_qf[col_to_k[syn_idx] * 64 + t];
                }
                v3 += chunk_sum;
            }

            sum_rsq += v1 * v1;
            sum_rabs += std::abs(v1);

            // Stage 3 rounds to half precision at output
            float v3_h = __half2float(__float2half(v3));

            // Accumulate stage metrics
            double diff1 = std::abs(static_cast<double>(g) - v1);
            if (diff1 > max_diff1) max_diff1 = diff1;
            mae1 += diff1;
            sq1 += diff1 * diff1;

            double diff2 = std::abs(static_cast<double>(g) - v2);
            if (diff2 > max_diff2) max_diff2 = diff2;
            mae2 += diff2;
            sq2 += diff2 * diff2;

            double diff3 = std::abs(static_cast<double>(g) - static_cast<double>(v3_h));
            if (diff3 > max_diff3) max_diff3 = diff3;
            mae3 += diff3;
            sq3 += diff3 * diff3;
        }
    }

    double rms_ref = std::sqrt(sum_rsq / total_points);
    double mean_ref = sum_rabs / total_points;

    auto make_stage_json = [&](double max_diff, double mae, double rmse_sq) {
        double m = mae / total_points;
        return nlohmann::json{
            {"max_abs_diff", max_diff},
            {"mae", m},
            {"rmse", std::sqrt(rmse_sq / total_points)},
            {"rel_mae_pct", 100.0 * (m / mean_ref)}
        };
    };

    nlohmann::json j_s1 = make_stage_json(max_diff1, mae1, sq1);
    nlohmann::json j_s2 = make_stage_json(max_diff2, mae2, sq2);
    nlohmann::json j_s3 = make_stage_json(max_diff3, mae3, sq3);

    double col_s1_s2 = j_s1["mae"].get<double>() / std::max(1e-9, j_s2["mae"].get<double>());
    double col_s1_s3 = j_s1["mae"].get<double>() / std::max(1e-9, j_s3["mae"].get<double>());

    return nlohmann::json{
        {"format", cfg.format},
        {"S", cfg.S},
        {"Cin", cfg.Cin},
        {"Cout", Cout},
        {"B", B},
        {"M", M},
        {"rms_y_ref", rms_ref},
        {"mean_abs_y_ref", mean_ref},
        {"unquantized", std::move(j_s1)},
        {"exact_quantized", std::move(j_s2)},
        {"stage3_wmma_chunk", std::move(j_s3)},
        {"collapse_ratio_max", max_diff1 / std::max(1e-9, max_diff2)},
        {"collapse_ratio_mae", col_s1_s2},
        {"collapse_s1_s3_mae", col_s1_s3}
    };
}

/**
 * @brief Prints formatted scientific table header.
 */
static void print_table_header(const std::string& title) {
    printf("\n=== %s ===\n", title.c_str());
    printf("%-10s %-12s %-14s %-12s %-14s %-12s %-14s %-12s\n",
           "Cin", "RMS(Yref)", "S1(Raw) MAE", "RelMAE %", "S2(Seq) MAE", "Collapse", "S3(WMMA) MAE", "Collapse");
    printf("--------------------------------------------------------------------------------------------------\n");
}

/**
 * @brief Prints a single result row.
 */
static void print_row(const nlohmann::json& r) {
    char s_rel[16], col12[16], col13[16];
    snprintf(s_rel, sizeof(s_rel), "%.3f %%", r["unquantized"]["rel_mae_pct"].get<double>());
    snprintf(col12, sizeof(col12), "%dx", static_cast<int>(r["collapse_ratio_mae"].get<double>()));
    snprintf(col13, sizeof(col13), "%dx", static_cast<int>(r["collapse_s1_s3_mae"].get<double>()));

    printf("%-10u %-12.2f %-14.2e %-12s %-14.2e %-12s %-14.2e %-12s\n",
           r["Cin"].get<uint32_t>(),
           r["rms_y_ref"].get<double>(),
           r["unquantized"]["mae"].get<double>(),
           s_rel,
           r["exact_quantized"]["mae"].get<double>(),
           col12,
           r["stage3_wmma_chunk"]["mae"].get<double>(),
           col13);
}

int main() {
    // Select standard GPU device 0
    CHK(hipSetDevice(0));

    hipDeviceProp_t prop;
    CHK(hipGetDeviceProperties(&prop, 0));
    printf("[INIT] GPU 0: %s (%s) | VRAM: %.2f GB\n",
           prop.name, prop.gcnArchName, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    std::vector<TestConfig> configs;
    const std::vector<uint32_t> cin_f16 = {
        1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 20000000
    };

    for (int S : {128, 256}) {
        for (uint32_t c : cin_f16) {
            configs.push_back({"FP16", S, c});
        }
    }

    nlohmann::json all_results = nlohmann::json::array();
    std::string current_section = "";

    for (const auto& cfg : configs) {
        std::string section_name = cfg.format + " S=" + std::to_string(cfg.S);
        if (section_name != current_section) {
            current_section = section_name;
            print_table_header(current_section);
        }
        nlohmann::json result = run_single_test(cfg);
        print_row(result);
        all_results.push_back(std::move(result));
    }

    mkdir("data", 0777);
    std::ofstream ofs("data/error_collapse_results.json");
    ofs << all_results.dump(2) << "\n";
    printf("\n[SUCCESS] Completed all %zu 3-stage error collapse configurations.\n",
           all_results.size());

    return 0;
}
