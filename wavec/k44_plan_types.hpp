#pragma once
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace k44_host {

// =================================================================================================
// 1. FP16/BF16 Conversion & Alignment Helpers
// =================================================================================================

static inline uint16_t float_to_half(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(uint32_t));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x007fffffu;
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant |= 0x00800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half_mant & 1u))) {
            half_mant++;
        }
        return (uint16_t)(sign | half_mant);
    } else if (exp == 0xff - 127 + 15) {
        if (mant == 0) {
            return (uint16_t)(sign | 0x7c00u);
        }
        return (uint16_t)(sign | 0x7c00u | (mant >> 13) | 1u);
    } else {
        uint32_t half_mant = mant >> 13;
        uint32_t rem = mant & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1u))) {
            half_mant++;
            if (half_mant == 0x0400u) {
                half_mant = 0;
                exp++;
            }
        }
        if (exp >= 31) {
            return (uint16_t)(sign | 0x7c00u);
        }
        return (uint16_t)(sign | ((uint32_t)exp << 10) | half_mant);
    }
}

static inline float half_to_float(uint16_t half_bits) {
    uint32_t sign = (uint32_t)(half_bits & 0x8000u) << 16;
    uint32_t exp = (half_bits >> 10) & 0x1fu;
    uint32_t mant = half_bits & 0x03ffu;
    uint32_t out_bits;
    if (exp == 0) {
        if (mant == 0) {
            out_bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            out_bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out_bits = sign | 0x7f800000u | (mant << 13);
    } else {
        out_bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float val;
    std::memcpy(&val, &out_bits, sizeof(float));
    return val;
}

static inline float bf16_to_float(uint16_t bf16_bits) {
    uint32_t f32_bits = ((uint32_t)bf16_bits) << 16;
    float val;
    std::memcpy(&val, &f32_bits, sizeof(float));
    return val;
}

struct alignas(16) uint4_host {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
};

enum class K44Act : int { NONE = 0, RELU = 1, SIGMOID = 2, SILU = 3, LEAKY_RELU = 4 };

enum class K44CmdType : int {
    K44_WAVE = 0,
    S2D_2X2 = 1,
    MAXPOOL_2X2 = 2,
    MAXPOOL_5X5 = 3,
    UPSAMPLE_2X = 4,
    SCATTER_UP2X2 = 5,
    RMSNORM_1D = 7,
    SWIGLU_MUL_1D = 8,
    ROPE_GQA_ATTN_1D = 9
};

static inline const char *cmd_type_str(K44CmdType cmd_type) {
    switch (cmd_type) {
    case K44CmdType::K44_WAVE:
        return "K44_WAVE";
    case K44CmdType::S2D_2X2:
        return "S2D_2X2";
    case K44CmdType::MAXPOOL_2X2:
        return "MAXPOOL_2X2";
    case K44CmdType::MAXPOOL_5X5:
        return "MAXPOOL_5X5";
    case K44CmdType::UPSAMPLE_2X:
        return "UPSAMPLE_2X";
    case K44CmdType::SCATTER_UP2X2:
        return "SCATTER_UP2X2";
    case K44CmdType::RMSNORM_1D:
        return "RMSNORM_1D";
    case K44CmdType::SWIGLU_MUL_1D:
        return "SWIGLU_MUL_1D";
    case K44CmdType::ROPE_GQA_ATTN_1D:
        return "ROPE_GQA_ATTN_1D";
    }
    return "UNKNOWN";
}

static inline const char *act_str(K44Act act) {
    switch (act) {
    case K44Act::NONE:
        return "NONE";
    case K44Act::RELU:
        return "RELU";
    case K44Act::SIGMOID:
        return "SIGMOID";
    case K44Act::SILU:
        return "SILU";
    case K44Act::LEAKY_RELU:
        return "LEAKY_RELU";
    }
    return "NONE";
}

static inline uint32_t round_up_16(uint32_t val) {
    uint32_t aligned = ((val + 15u) & ~15u);
    return (aligned < 16u) ? 16u : aligned;
}

static inline uint32_t round_up_32(uint32_t val) {
    uint32_t aligned = ((val + 31u) & ~31u);
    return (aligned < 64u) ? 64u : aligned;
}

static inline uint32_t choose_cluster_size(uint32_t cout, uint32_t num_tiles,
                                           uint32_t force_cluster_size = 0u) {
    if (force_cluster_size != 0u) {
        for (uint32_t cluster_sz = force_cluster_size; cluster_sz >= 16u; cluster_sz >>= 1u) {
            if (cout % cluster_sz == 0u) {
                return cluster_sz;
            }
        }
    }
    if ((cout % 128u == 0u) && (cout >= 640u || num_tiles <= 64u)) {
        return 128u;
    }
    if (cout % 64u == 0u) {
        return 64u;
    }
    if (cout % 32u == 0u) {
        return 32u;
    }
    return 16u;
}

// =================================================================================================
// 2. Hardware Profiles & Wave Tiling Configuration
// =================================================================================================

struct K44HardwareProfile {
    std::string name = "rx9060xt";
    uint32_t num_cus = 32u;
    uint32_t max_clusters_cta = 2u;
    double r_tc_syn_s = 27.72e12;
    double bw_vram_bytes_s = 320.0e9;
    double bw_cache_bytes_s = 1400.0e9;
    double bw_l1_bytes_s = 4500.0e9;
    double v_cache_win = 24.0 * 1024.0 * 1024.0;
    double v_cache_budget = 16.0 * 1024.0 * 1024.0;
    double cache_gamma = 2.0;
    double k_pipe_half = 32.0;
    double w_cta_half = 0.80;
    double occ_waves_half = 4.0;
    double chunk_launch_s = 3.5e-6;
    double gather_subtile_pen = 0.04;
    uint32_t max_b_per_wave = 0u;

    static K44HardwareProfile from_preset_base(const char *preset_str) {
        std::string preset = preset_str ? preset_str : "auto";
        for (char &ch : preset) {
            ch = (char)std::tolower((unsigned char)ch);
        }
        if (preset == "auto" || preset.empty()) {
#ifdef __CUDACC__
            int dev = 0;
            cudaDeviceProp prop{};
            if (cudaGetDevice(&dev) == cudaSuccess &&
                cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
                if (prop.major >= 10) {
                    return from_preset_base("b200");
                }
                if (prop.major == 9) {
                    return from_preset_base("h100");
                }
                if (prop.major == 8 && prop.minor == 9) {
                    return (prop.multiProcessorCount >= 130) ? from_preset_base("rtx6000ada")
                                                             : from_preset_base("rtx4090");
                }
                K44HardwareProfile hw_prof;
                hw_prof.name = prop.name;
                hw_prof.num_cus = (uint32_t)std::max(1, prop.multiProcessorCount);
                hw_prof.max_clusters_cta = 3u;
                hw_prof.r_tc_syn_s = (double)hw_prof.num_cus * 0.95e12;
                hw_prof.bw_vram_bytes_s = 800.0e9;
                hw_prof.bw_cache_bytes_s = 3000.0e9;
                hw_prof.bw_l1_bytes_s = 8000.0e9;
                double l2_bytes =
                    (prop.l2CacheSize > 0) ? (double)prop.l2CacheSize : (48.0 * 1024.0 * 1024.0);
                hw_prof.v_cache_win = 0.75 * l2_bytes;
                hw_prof.v_cache_budget = 0.50 * l2_bytes;
                return hw_prof;
            }
            return from_preset_base("rtx6000ada");
#else
            return from_preset_base("rx9060xt");
#endif
        }
        if (preset == "rtx6000ada" || preset == "rtx6000_ada" || preset == "rtx6000" ||
            preset == "ada" || preset == "sm_89" || preset == "l40s") {
            return {"rtx6000ada",
                    142u,
                    3u,
                    115.0e12,
                    795.6e9,
                    4078.7e9,
                    14000.0e9,
                    64.0 * 1024.0 * 1024.0,
                    48.0 * 1024.0 * 1024.0};
        }
        if (preset == "rtx4090" || preset == "4090") {
            return {"rtx4090",
                    128u,
                    3u,
                    125.0e12,
                    1008.0e9,
                    4200.0e9,
                    12800.0e9,
                    54.0 * 1024.0 * 1024.0,
                    36.0 * 1024.0 * 1024.0};
        }
        if (preset == "h100" || preset == "sm_90" || preset == "hopper") {
            return {"h100",
                    132u,
                    3u,
                    350.0e12,
                    3350.0e9,
                    5500.0e9,
                    18000.0e9,
                    38.0 * 1024.0 * 1024.0,
                    28.0 * 1024.0 * 1024.0};
        }
        if (preset == "b200" || preset == "blackwell" || preset == "sm_100" ||
            preset == "sm_100a") {
            return {"b200",
                    148u,
                    3u,
                    900.0e12,
                    8000.0e9,
                    14000.0e9,
                    32000.0e9,
                    96.0 * 1024.0 * 1024.0,
                    64.0 * 1024.0 * 1024.0};
        }
        return {"rx9060xt",
                32u,
                2u,
                27.72e12,
                320.0e9,
                1400.0e9,
                4500.0e9,
                24.0 * 1024.0 * 1024.0,
                16.0 * 1024.0 * 1024.0};
    }

    static K44HardwareProfile from_preset(const char *preset_str) {
        return from_preset_base(preset_str);
    }
};

struct K44WaveConfig {
    uint32_t S1 = 64u;
    uint32_t Cout_pad1 = 64u;
    uint32_t S2 = 0u;
    uint32_t Cout_pad2 = 0u;
    uint32_t num_sub_b = 1u;

    uint32_t total_cout_pad() const {
        return Cout_pad1 + Cout_pad2;
    }

    bool is_split() const {
        return (S2 > 0u) && (Cout_pad2 > 0u);
    }
};

static inline K44WaveConfig select_optimal_wave_config(
    uint32_t cin, uint32_t cout, uint32_t kh, uint32_t kw, uint32_t spatial_in,
    uint32_t spatial_out, uint32_t force_cluster_size = 0u,
    const K44HardwareProfile &hw_prof = K44HardwareProfile::from_preset("rx9060xt"),
    uint32_t force_max_b = 0u) {
    (void)cin;
    (void)kh;
    (void)kw;
    (void)spatial_in;
    (void)hw_prof;
    (void)force_max_b;
    uint32_t base_cout_pad = round_up_16(cout);
    uint32_t cluster_sz =
        choose_cluster_size(base_cout_pad, std::max(1u, spatial_out), force_cluster_size);
    return {cluster_sz, base_cout_pad, 0u, 0u, 1u};
}

// =================================================================================================
// 3. WMMA & MMA Weight Layout Interleaving
// =================================================================================================

// Packs row-major FP16 weights (`src_compact`) into RDNA4 Wave32 WMMA 16x16x16 hardware register
// layout (`dst_wmma`).
// Variables:
// - `dst_wmma`: Output buffer of 128-bit `uint4_host` vectors (8 FP16 weights each) in WMMA order.
// - `src_compact`: Input row-major FP16 weight matrix of shape `[num_output_neurons, B]`.
// - `num_output_neurons` (`Cout_pad`): Padded output neuron count (multiple of 16).
// - `B`: Padded synaptic fan-in per output neuron (multiple of 32, `B >= 64`).
// - `B_vec8` (`B / 8`): Number of 8-FP16 `uint4_host` vectors per neuron row in `src_compact`.
// - `B_vec16` (`B / 16`): Number of 16-synapse WMMA K-tiles per neuron row.
// - `tile_idx` (`global_idx / 32`): Index of the 16x16 WMMA tile across `[num_output_neurons/16,
// B_vec16]`.
// - `lane` (`global_idx % 32`): Wave32 thread lane `0..31`, where `lane & 15` selects the output
//   neuron row `0..15` within the tile and `lane >> 4` (`0` or `1`) selects the lower (`0..7`) or
//   upper (`8..15`) 8-FP16 half of the 16-synapse K-slice.
static inline void convert_raw_to_wmma(uint4_host *dst_wmma, const uint4_host *src_compact,
                                       uint32_t num_output_neurons, uint32_t B_vec8) {
    uint32_t B_vec16 = B_vec8 / 2u;
    uint32_t total_elements = (num_output_neurons / 16u) * B_vec16 * 32u;
#pragma omp parallel for schedule(static) if (total_elements >= 4096)
    for (uint32_t global_idx = 0; global_idx < total_elements; ++global_idx) {
        uint32_t tile_idx = global_idx / 32u;
        uint32_t lane = global_idx % 32u;
        uint32_t src_idx = ((tile_idx / B_vec16) * 16u + (lane & 15u)) * B_vec8 +
                           (2u * (tile_idx % B_vec16) + (lane >> 4u));
        dst_wmma[global_idx] = src_compact[src_idx];
    }
}

// Inverse bijection of `convert_raw_to_wmma`: restores row-major `[num_output_neurons, B]` FP16
// weight order (`dst_compact`) from RDNA4 Wave32 WMMA tile layout (`src_wmma`), where
// `num_output_neurons` is the padded output channel count (`Cout_pad`), `B_vec8 = B / 8` is the
// row stride in 128-bit `uint4_host` units, `B_vec16 = B / 16` is the number of 16-synapse K-tiles,
// `tile_idx` is the 16x16 tile index, and `lane` (`0..31`) is the Wave32 thread lane index.
static inline void convert_wmma_to_raw(uint4_host *dst_compact, const uint4_host *src_wmma,
                                       uint32_t num_output_neurons, uint32_t B_vec8) {
    uint32_t B_vec16 = B_vec8 / 2u;
    uint32_t total_elements = (num_output_neurons / 16u) * B_vec16 * 32u;
#pragma omp parallel for schedule(static) if (total_elements >= 4096)
    for (uint32_t global_idx = 0; global_idx < total_elements; ++global_idx) {
        uint32_t tile_idx = global_idx / 32u;
        uint32_t lane = global_idx % 32u;
        uint32_t src_idx = ((tile_idx / B_vec16) * 16u + (lane & 15u)) * B_vec8 +
                           (2u * (tile_idx % B_vec16) + (lane >> 4u));
        dst_compact[src_idx] = src_wmma[global_idx];
    }
}

// Converts RDNA4 Wave32 WMMA weights (`src_wmma`) into NVIDIA PTX
// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` A-matrix fragment layout (`dst_cuda_mma`).
// Variables:
// - `num_output_neurons` (`Cout_pad`): Padded output neuron count (MMA M-dimension, multiple of
// 16).
// - `B_vec8` (`B / 8`): Number of 8-FP16 `uint4_host` vectors per row along the MMA K-dimension
// (`B`).
// - `B_vec16` (`B / 16`): Number of 16-synapse K-tiles per 16-neuron block.
// - `n_blk` (`0 .. num_output_neurons/16 - 1`): 16-neuron output block index (`n_base = n_blk *
// 16`).
// - `k_blk` (`0 .. B_vec16 - 1`): 16-synapse K-tile index (`k_base_u32 = k_blk * 8` in FP16x2
// units).
// - `lane` (`0..31`): CUDA warp thread lane, decomposed into `group_id = lane >> 2` (`0..7`,
//   selecting output neuron rows `n_base + group_id` and `n_base + group_id + 8`) and
//   `tid_in_group = lane & 3` (`0..3`, selecting 32-bit FP16x2 synapse pairs at `k_base_u32 +
//   tid_in_group` and `k_base_u32 + tid_in_group + 4` to form the 4x32-bit fragment `{a0, a1, a2,
//   a3}`).
static inline void convert_wmma_to_cuda_mma(uint4_host *dst_cuda_mma, const uint4_host *src_wmma,
                                            uint32_t num_output_neurons, uint32_t B_vec8) {
    uint32_t B_vec16 = B_vec8 / 2u;
    uint32_t num_tiles = (num_output_neurons / 16u) * B_vec16;
    std::vector<uint4_host> compact(num_tiles * 32u);
    convert_wmma_to_raw(compact.data(), src_wmma, num_output_neurons, B_vec8);
    const uint32_t *src_u32 = reinterpret_cast<const uint32_t *>(compact.data());
    const uint32_t row_stride_u32 = B_vec16 * 8u;
#pragma omp parallel for schedule(static) if (num_tiles >= 128)
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint32_t n_blk = tile_idx / B_vec16;
        uint32_t k_blk = tile_idx % B_vec16;
        uint32_t n_base = n_blk * 16u;
        uint32_t k_base_u32 = k_blk * 8u;
        for (uint32_t lane = 0; lane < 32u; ++lane) {
            uint32_t group_id = lane >> 2u;
            uint32_t tid_in_group = lane & 3u;
            uint32_t a0 =
                src_u32[(n_base + group_id + 0u) * row_stride_u32 + k_base_u32 + tid_in_group + 0u];
            uint32_t a1 =
                src_u32[(n_base + group_id + 8u) * row_stride_u32 + k_base_u32 + tid_in_group + 0u];
            uint32_t a2 =
                src_u32[(n_base + group_id + 0u) * row_stride_u32 + k_base_u32 + tid_in_group + 4u];
            uint32_t a3 =
                src_u32[(n_base + group_id + 8u) * row_stride_u32 + k_base_u32 + tid_in_group + 4u];
            dst_cuda_mma[tile_idx * 32u + lane] = {a0, a1, a2, a3};
        }
    }
}

// =================================================================================================
// 4. Execution Plan & IR Data Structures
// =================================================================================================

struct K44ResDesc {
    uint32_t H = 0;
    uint32_t W = 0;
    uint32_t H_pad = 0;
    uint32_t W_pad = 0;
    uint32_t batch_size = 1;
    uint32_t tiles_per_row = 0;
    uint32_t tiles_per_frame = 0;
    uint32_t num_tiles = 0;
    uint32_t single_stride = 0;
    uint32_t channel_stride = 0;
    bool needs_mask = false;
    bool is_1d_tokens = false;
    bool is_image_token64 = false;
    std::vector<uint32_t> h_tile_off_X;
    std::vector<uint32_t> h_tile_off_Y;
    std::vector<uint32_t> h_tile_off_X_s2;

    // Initializes spatial/token tile offset tables (`h_tile_off_X`, `h_tile_off_Y`) across 3 modes:
    // Parameters & Member Variables:
    // - `height` (`H`): Spatial feature-map height in pixels (or `1` for 1D token sequences).
    // - `width` (`W`): Spatial feature-map width in pixels (or sequence length `SeqLen` for 1D
    // tokens).
    // - `is_1d` (`is_1d_tokens`): `true` for 1D token tensors `[D, 1, SeqLen]` (where `D` is the
    //   embedding dimension and `SeqLen` is token count); `false` for 2D image tensors `[C, H, W]`
    //   (where `C` is channel count).
    // - `b_frames` (`batch_size`): Number of batch frames processed concurrently in one wave.
    // - `W_pad`, `H_pad`: Padded width and height in FP16 elements including zero-halo borders.
    // - `single_stride`: FP16 elements per single-frame channel plane (`H_pad * W_pad` in 2D).
    // - `channel_stride`: FP16 elements between consecutive channels `c` and `c + 1`
    //   (`batch_size * single_stride`).
    // - `num_tiles`: Total 64-element WMMA B-matrix tiles across the batch (`batch_size *
    // tiles_per_frame`). Modes: 1) 1D token sequence (`is_1d_tokens == true`): Contiguous 64-token
    // tiles (`H_pad = 1`,
    //    `W_pad = single_stride = ceil(W / 64) * 64`) with no 2D spatial halo.
    // 2) Aligned 2D grid (`W % 64 == 0`): `W_pad = W + 8` (4 FP16 left + 4 FP16 right zero-halo)
    //    and `H_pad = H + 2` (1 top + 1 bottom zero-halo row). `h_tile_off_X` is pre-shifted to
    //    `(ty - 1, tx*64 - 1)` so 3x3 convolutions (`dy, dx in {-1, 0, +1}`) execute branch-free
    //    vector loads without boundary checks.
    // 3) Arbitrary 2D grid (`W % 64 != 0`): `W_pad = W + 4` and `H_pad = H + 2`. Packs the
    //    `uint64_t` quad-FP16 write offset (`y_off_b >> 2`) into the lower 16 bits of
    //    `h_tile_off_Y` and a 16-bit lane write mask (`mask16`, where bit `lane` in `0..15` is `1`
    //    iff pixel quad
    //    `(ty, tx_col)` is within `H x W`) into the upper 16 bits to prevent overwriting halos.
    void init(uint32_t height, uint32_t width, bool is_1d = false, uint32_t b_frames = 1u) {
        H = height;
        W = width;
        is_1d_tokens = is_1d;
        batch_size = std::max(1u, b_frames);
        is_image_token64 = (!is_1d_tokens) && (batch_size >= 64u) && (batch_size % 64u == 0u);
        h_tile_off_X_s2.clear();
        if (is_1d_tokens) {
            needs_mask = false;
            H_pad = 1u;
            tiles_per_frame = (W + 63u) / 64u;
            tiles_per_row = tiles_per_frame;
            single_stride = tiles_per_frame * 64u;
            W_pad = single_stride;
            num_tiles = batch_size * tiles_per_frame;
            channel_stride = batch_size * single_stride;
            h_tile_off_X.resize(num_tiles);
            h_tile_off_Y.resize(num_tiles);
            for (uint32_t bf = 0; bf < batch_size; ++bf) {
                uint32_t frame_off = bf * single_stride;
                for (uint32_t tile_idx = 0; tile_idx < tiles_per_frame; ++tile_idx) {
                    uint32_t flat_tile = bf * tiles_per_frame + tile_idx;
                    h_tile_off_X[flat_tile] = frame_off + tile_idx * 64u;
                    h_tile_off_Y[flat_tile] = frame_off + tile_idx * 64u;
                }
            }
        } else if (W % 64u == 0u) {
            needs_mask = false;
            tiles_per_row = W / 64u;
            W_pad = W + 8u;
            H_pad = H + 2u;
            single_stride = H_pad * W_pad;
            tiles_per_frame = H * tiles_per_row;
            num_tiles = batch_size * tiles_per_frame;
            channel_stride = batch_size * single_stride;
            h_tile_off_X.resize(num_tiles);
            h_tile_off_Y.resize(num_tiles);
            for (uint32_t bf = 0; bf < batch_size; ++bf) {
                uint32_t frame_off = bf * single_stride;
                for (uint32_t ty = 0; ty < H; ++ty) {
                    for (uint32_t tx = 0; tx < tiles_per_row; ++tx) {
                        uint32_t tile_idx = ty * tiles_per_row + tx;
                        uint32_t flat_tile = bf * tiles_per_frame + tile_idx;
                        h_tile_off_X[flat_tile] = frame_off + ty * W_pad + (4u + tx * 64u - 1u);
                        h_tile_off_Y[flat_tile] = frame_off + (ty + 1u) * W_pad + (4u + tx * 64u);
                    }
                }
            }
        } else {
            needs_mask = true;
            W_pad = W + 4u;
            H_pad = H + 2u;
            tiles_per_frame = (H * W_pad + 63u) / 64u;
            tiles_per_row = 0u;
            single_stride = ((2u * W_pad + tiles_per_frame * 64u + 63u) & ~63u);
            num_tiles = batch_size * tiles_per_frame;
            channel_stride = batch_size * single_stride;
            h_tile_off_X.resize(num_tiles);
            h_tile_off_Y.resize(num_tiles);
            for (uint32_t bf = 0; bf < batch_size; ++bf) {
                uint32_t frame_off = bf * single_stride;
                for (uint32_t tile_idx = 0; tile_idx < tiles_per_frame; ++tile_idx) {
                    uint32_t flat_tile = bf * tiles_per_frame + tile_idx;
                    uint32_t y_off_1 = W_pad + 4u + tile_idx * 64u;
                    uint32_t y_off_b = frame_off + y_off_1;
                    h_tile_off_X[flat_tile] = frame_off + (y_off_1 - W_pad - 1u);
                    uint32_t mask16 = 0u;
                    for (uint32_t lane = 0; lane < 16u; ++lane) {
                        uint32_t rel = tile_idx * 64u + lane * 4u;
                        uint32_t ty = rel / W_pad;
                        uint32_t tx_col = rel % W_pad;
                        if (ty < H && tx_col < W) {
                            mask16 |= (1u << lane);
                        }
                    }
                    h_tile_off_Y[flat_tile] = ((y_off_b >> 2) & 0xffffu) | (mask16 << 16);
                }
            }
        }
    }

    // Returns the linear FP16 element offset inside a tensor buffer for channel `ch` (`0 .. C-1`),
    // vertical row `y` (`0 .. H-1`), horizontal column or token index `x` (`0 .. W-1`), and batch
    // frame `bf` (`0 .. batch_size-1`). In standard 2D mode, `y + 1` skips the 1-row top zero-halo
    // and `+ 4` skips the 4-element left zero-halo along row stride `W_pad`.
    uint32_t canvas_offset(uint32_t ch, uint32_t y, uint32_t x, uint32_t bf = 0u) const {
        if (is_1d_tokens) {
            return ch * channel_stride + bf * single_stride + x;
        }
        if (is_image_token64) {
            return ch * channel_stride + (bf / 64u) * single_stride +
                   ((y + 1u) * W_pad + (x + 1u)) * 64u + (bf & 63u);
        }
        return ch * channel_stride + bf * single_stride + (y + 1u) * W_pad + 4u + x;
    }

    // Computes the connectome gather index (`h_idx` entry, in FP16 elements) for an input channel
    // starting at `ch_base_half` in `d_Canvas` and a receptive-field spatial tap `(dy, dx)`
    // (where `dy` is vertical filter offset and `dx` is horizontal filter offset, e.g. `-1, 0, +1`
    // for 3x3 convolutions). Because `h_tile_off_X` is pre-anchored at `(y - 1, x - 1)`, adding
    // `(1 + dy) * W_pad + (1 + dx)` yields a non-negative unsigned offset for all `dy, dx >= -1`.
    uint32_t tap_offset(uint32_t ch_base_half, int dy, int dx) const {
        if (is_1d_tokens) {
            return ch_base_half + (uint32_t)dx;
        }
        if (is_image_token64) {
            return ch_base_half + ((uint32_t)(1 + dy) * W_pad + (uint32_t)(1 + dx)) * 64u;
        }
        return ch_base_half + (uint32_t)(1 + dy) * W_pad + (uint32_t)(1 + dx);
    }
};

struct K44ExecCmd {
    K44CmdType type = K44CmdType::K44_WAVE;
    std::string name;
    uint32_t Cout = 0;
    uint32_t logical_Cout = 0;
    uint32_t Cin = 0;
    uint32_t cluster_size = 64;
    uint32_t B = 0;
    uint32_t B_raw = 0;
    uint32_t B_vec16 = 0;
    bool has_bias = true;
    K44Act act = K44Act::NONE;
    bool has_residual = false;
    bool has_cout_map = false;
    bool has_self_read = false;
    bool needs_halo_clear = false;
    uint32_t subwave_idx = 0;
    uint32_t num_subwaves = 1;
    uint32_t weight_bits = 16;
    uint32_t y_off = 0;
    uint32_t src_off = 0;
    uint32_t src2_off = 0;
    uint32_t src3_off = 0;
    uint32_t res_off = 0;
    uint32_t x_base_off = 0;
    uint32_t res_key_in = 0;
    uint32_t res_key_out = 0;
    uint32_t h_in = 0;
    uint32_t w_in = 0;
    uint32_t h_out = 0;
    uint32_t w_out = 0;
    uint32_t h_q = 0;
    uint32_t h_kv = 0;
    uint32_t head_dim = 64;
    uint32_t kv_k_off = 0;
    uint32_t kv_v_off = 0;
    uint32_t kv_max_seq = 0;
    uint32_t attn_probs_off = 0;
    float eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool llama3_rope_scaling = false;
    bool gamma_folded = false;
    double gflops = 0.0;
    std::vector<uint4_host> h_w_wmma;
    std::vector<uint32_t> h_idx;
    std::vector<uint16_t> h_bias;
    std::vector<uint16_t> h_cout_map;
};

struct K44CompileOptions {
    uint32_t max_B_per_wave = 0;
    bool fold_rmsnorm_gamma = true;
    bool keep_host_plan_buffers = true;
    bool reuse_canvas = false;
    uint32_t force_cluster_size = 0;
    uint32_t kv_cache_max_seq = 0;
    uint32_t batch_size = 1;
    K44HardwareProfile hw_profile = K44HardwareProfile::from_preset("auto");
};

struct K44WavePlan {
    uint32_t batch_size = 1;
    uint32_t wave_batch_size = 1;
    bool is_image_token64 = false;
    size_t canvas_elems = 0;
    size_t kv_cache_elems = 0;
    size_t total_w_bytes = 0;
    size_t total_idx_bytes = 0;
    double single_frame_gflops = 0;
    double total_gflops = 0.0;
    int num_wave_dispatches = 0;
    int num_folded_rmsnorms = 0;
    int num_eliminated_concats = 0;
    int num_eliminated_splits = 0;
    int num_zero_copy_flattens = 0;
    uint32_t input_off = 0;
    uint32_t input_c = 0;
    uint32_t input_h = 0;
    uint32_t input_w = 0;
    uint32_t input_res_key = 0;
    uint32_t output_off = 0;
    uint32_t output_c = 0;
    uint32_t logical_output_c = 0;
    uint32_t output_h = 0;
    uint32_t output_w = 0;
    uint32_t output_res_key = 0;
    bool reuse_canvas = false;
    K44HardwareProfile hw_profile = K44HardwareProfile::from_preset("auto");
    std::map<uint32_t, K44ResDesc> res_map;
    std::vector<K44ExecCmd> cmds;
};

struct K44IRNode {
    std::string op_type;
    std::string name;
    std::vector<std::string> in_tensors;
    std::vector<std::string> out_tensors;
    std::vector<uint32_t> slice_sizes;
    uint32_t cin = 0;
    uint32_t cout = 0;
    uint32_t kh = 1;
    uint32_t kw = 1;
    uint32_t stride = 1;
    uint32_t pad = 0;
    uint32_t h_in = 0;
    uint32_t w_in = 0;
    uint32_t h_out = 0;
    uint32_t w_out = 0;
    uint32_t h_q = 0;
    uint32_t h_kv = 0;
    uint32_t head_dim = 64;
    float eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool llama3_rope_scaling = false;
    bool is_1d_tokens = false;
    bool gamma_folded = false;
    K44Act act = K44Act::NONE;
    std::string residual_tensor;
    std::vector<float> weights_f32;
    std::vector<uint16_t> weights_f16;
    std::vector<float> bias_f32;
    std::vector<uint16_t> bias_f16;
};

} // namespace k44_host
