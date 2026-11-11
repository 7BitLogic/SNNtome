#pragma once
#include "k44_plan_types.hpp"

namespace k44_host {

// =================================================================================================
// TCFFv1.0 Binary Serialization & Deserialization
// =================================================================================================

static inline uint64_t fnv1a_hash64(const uint8_t *data, size_t len) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 0x00000100000001b3ULL;
    }
    return hash;
}

static inline uint32_t align_up_u32(uint32_t off, uint32_t align) {
    return ((off + align - 1u) / align) * align;
}

static constexpr uint16_t TIFF_BYTE = 1;
static constexpr uint16_t TIFF_ASCII = 2;
static constexpr uint16_t TIFF_SHORT = 3;
static constexpr uint16_t TIFF_LONG = 4;
static constexpr uint16_t TIFF_RATIONAL = 5;
static constexpr uint16_t TAG_TCFF_IMAGE_DEPTH = 0x80E5;
static constexpr uint16_t TAG_TCFF_WAVE_MAGIC = 0x8440;
static constexpr uint16_t TAG_TCFF_WAVE_PARAMS = 0x8441;
static constexpr uint16_t TAG_TCFF_AUX_OFFSETS = 0x8442;
static constexpr uint16_t TAG_TCFF_DATA_HASH64 = 0x8443;
static constexpr uint16_t TAG_TCFF_WAVE_GEOMETRY = 0x8444;
static constexpr uint32_t TCFF_MAGIC_WAVE = 0x45564157u;
static constexpr uint32_t TCFF_MAGIC_CANV = 0x564E4143u;
static constexpr uint32_t TCFF_FLAG_TRAILING_IFD_HASH = 0x00000001u;
static constexpr uint32_t TCFF_FLAG_HAS_CANVAS = 0x00000004u;

struct TiffIFDEntry {
    uint16_t tag;
    uint16_t typ;
    uint32_t count;
    uint32_t val_or_off;
};

struct ByteWriter {
    std::vector<uint8_t> buf;

    void u16(uint16_t val) {
        buf.push_back(static_cast<uint8_t>(val & 0xffu));
        buf.push_back(static_cast<uint8_t>((val >> 8) & 0xffu));
    }

    void u32(uint32_t val) {
        buf.push_back(static_cast<uint8_t>(val & 0xffu));
        buf.push_back(static_cast<uint8_t>((val >> 8) & 0xffu));
        buf.push_back(static_cast<uint8_t>((val >> 16) & 0xffu));
        buf.push_back(static_cast<uint8_t>((val >> 24) & 0xffu));
    }

    void u64(uint64_t val) {
        u32(static_cast<uint32_t>(val & 0xffffffffULL));
        u32(static_cast<uint32_t>((val >> 32) & 0xffffffffULL));
    }

    void raw(const void *ptr, size_t num_bytes) {
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(ptr);
        buf.insert(buf.end(), bytes, bytes + num_bytes);
    }

    void patch_u32(uint32_t off, uint32_t val) {
        buf[off + 0] = static_cast<uint8_t>(val & 0xffu);
        buf[off + 1] = static_cast<uint8_t>((val >> 8) & 0xffu);
        buf[off + 2] = static_cast<uint8_t>((val >> 16) & 0xffu);
        buf[off + 3] = static_cast<uint8_t>((val >> 24) & 0xffu);
    }

    uint32_t pad(uint32_t align) {
        uint32_t target = align_up_u32(static_cast<uint32_t>(buf.size()), align);
        buf.resize(target, 0u);
        return target;
    }
};

// Serializes all `K44_WAVE` commands of a compiled `K44WavePlan` (`plan`) into a 256-byte aligned,
// TIFF-compatible `TCFFv1.0` binary container at `tcff_path`.
// Parameters & Payload Fields:
// - `plan`: Compiled wave execution plan containing `cmds` and `res_map`.
// - `tcff_path`: Output file path on disk.
// - `model_prefix`: ASCII model identifier prefix stored in each IFD page name (tag `285`).
// - `include_canvas`: Sets `TCFF_FLAG_HAS_CANVAS` in the master header with `canvas_bytes`
//   (`plan.canvas_elems * 2` bytes).
// - `include_geometry`: Emits tag `0x8444` (`TAG_TCFF_WAVE_GEOMETRY`) recording logical dimensions
//   `[d0..d3]`, physical strides `[s0..s3]`, and 2D halo offset (`W_pad + 4`).
// - Per-wave IFD page (`w_idx = 0 .. num_waves - 1`): Stores 256-byte aligned blocks for
//   `cmd.h_w_wmma` (WMMA-interleaved FP16 weights of size `Cout * B` halfs, where `Cout` is padded
//   output channels and `B` is padded synaptic fan-in per neuron), `cmd.h_idx` (`[Cout / S, B]`
//   connectome gather offsets in FP16 elements, where `S = cmd.cluster_size`), and `cmd.h_bias`
//   (`[Cout]` FP16 biases), protected by a 64-bit FNV-1a payload checksum (`wave_hash64`).
static bool save_plan_to_tcff(const K44WavePlan &plan, const char *tcff_path,
                              const char *model_prefix = "K44Model", bool include_canvas = true,
                              bool include_geometry = true,
                              const uint16_t *live_canvas_f16 = nullptr) {
    (void)live_canvas_f16;
    std::vector<const K44ExecCmd *> wave_cmds;
    for (const auto &cmd : plan.cmds) {
        if (cmd.type == K44CmdType::K44_WAVE) {
            wave_cmds.push_back(&cmd);
        }
    }
    const uint32_t num_waves = static_cast<uint32_t>(wave_cmds.size());
    const uint32_t stride_alignment = 256u;
    const uint32_t canvas_bytes = static_cast<uint32_t>(plan.canvas_elems * sizeof(uint16_t));
    const uint32_t total_pages = num_waves + (include_canvas ? 1u : 0u);
    const uint32_t master_flags =
        TCFF_FLAG_TRAILING_IFD_HASH | (include_canvas ? TCFF_FLAG_HAS_CANVAS : 0u);

    ByteWriter writer;
    writer.buf.reserve(plan.total_w_bytes + plan.total_idx_bytes +
                       (include_canvas ? canvas_bytes : 0u) + 65536u);
    writer.buf.push_back('I');
    writer.buf.push_back('I');
    writer.u16(42u);
    writer.u32(0u);
    writer.raw("TCFFv1.0", 8);
    writer.u32(stride_alignment);
    writer.u32(num_waves);
    writer.u32(canvas_bytes);
    writer.u32(master_flags);
    uint64_t master_hash = fnv1a_hash64(writer.buf.data() + 8, 24);
    writer.u64(master_hash);
    writer.pad(stride_alignment);

    uint32_t prev_next_ifd_ptr_off = 4u;
    const char *pfx = (model_prefix && model_prefix[0]) ? model_prefix : "K44Model";
    for (uint32_t w_idx = 0; w_idx < num_waves; ++w_idx) {
        const K44ExecCmd &cmd = *wave_cmds[w_idx];
        uint32_t wave_data_start = static_cast<uint32_t>(writer.buf.size());
        uint32_t w_offset = static_cast<uint32_t>(writer.buf.size());
        uint32_t w_bytes = static_cast<uint32_t>(cmd.h_w_wmma.size() * sizeof(uint4_host));
        writer.raw(cmd.h_w_wmma.data(), w_bytes);
        writer.pad(stride_alignment);
        uint32_t idx_offset = static_cast<uint32_t>(writer.buf.size());
        uint32_t idx_bytes = static_cast<uint32_t>(cmd.h_idx.size() * sizeof(uint32_t));
        writer.raw(cmd.h_idx.data(), idx_bytes);
        writer.pad(stride_alignment);
        uint32_t bias_offset = static_cast<uint32_t>(writer.buf.size());
        uint32_t bias_bytes = static_cast<uint32_t>(cmd.h_bias.size() * sizeof(uint16_t));
        writer.raw(cmd.h_bias.data(), bias_bytes);
        writer.pad(stride_alignment);
        uint32_t wave_data_end = static_cast<uint32_t>(writer.buf.size());
        uint64_t wave_hash64 =
            fnv1a_hash64(writer.buf.data() + wave_data_start, wave_data_end - wave_data_start);

        uint32_t ifd_offset = static_cast<uint32_t>(writer.buf.size());
        writer.patch_u32(prev_next_ifd_ptr_off, ifd_offset);
        const uint16_t num_tags = 21u + (include_geometry ? 1u : 0u);
        uint32_t ifd_table_bytes = 2u + num_tags * 12u + 4u;
        uint32_t ext_cursor = ifd_offset + ifd_table_bytes;
        ByteWriter ext;
        auto add_ext_bytes = [&](const void *ptr, size_t num_bytes) -> uint32_t {
            if (ext.buf.size() & 1u) {
                ext.buf.push_back(0u);
            }
            uint32_t off = ext_cursor + static_cast<uint32_t>(ext.buf.size());
            ext.raw(ptr, num_bytes);
            return off;
        };
        auto add_ext_ascii = [&](const std::string &str, uint32_t &out_len) -> uint32_t {
            out_len = static_cast<uint32_t>(str.size() + 1u);
            return add_ext_bytes(str.c_str(), out_len);
        };
        auto add_ext_u32_arr = [&](const std::vector<uint32_t> &arr,
                                   uint32_t &out_cnt) -> uint32_t {
            out_cnt = static_cast<uint32_t>(arr.size());
            return add_ext_bytes(arr.data(), arr.size() * sizeof(uint32_t));
        };

        const char *act_name = act_str(cmd.act);
        char page_buf[128];
        std::snprintf(page_buf, sizeof(page_buf), "%s_Wave_%03u_%ux%u_S%u", pfx, w_idx, cmd.B,
                      cmd.Cout, cmd.cluster_size);
        char desc_buf[320];
        std::snprintf(desc_buf, sizeof(desc_buf),
                      "TCFF %s Wave %u/%u | Cin=%u Cout=%u(log=%u) B=%u S=%u Act=%s Res=%d | "
                      "W=0x%x Idx=0x%x Bias=0x%x | FNV1a64=0x%016llx",
                      pfx, w_idx, num_waves, cmd.Cin, cmd.Cout, cmd.logical_Cout, cmd.B,
                      cmd.cluster_size, act_name, (int)cmd.has_residual, w_offset, idx_offset,
                      bias_offset, (unsigned long long)wave_hash64);

        uint32_t desc_len = 0;
        uint32_t page_len = 0;
        uint32_t dummy_cnt = 0;
        uint32_t wp_cnt = 0;
        uint32_t aux_cnt = 0;
        uint32_t geom_cnt = 0;
        uint32_t desc_off = add_ext_ascii(desc_buf, desc_len);
        uint32_t page_off = add_ext_ascii(page_buf, page_len);
        uint32_t res_rat_off = add_ext_u32_arr({1u, 1u}, dummy_cnt);
        uint32_t wp_off =
            add_ext_u32_arr({cmd.Cin, cmd.Cout, cmd.B, cmd.cluster_size, cmd.weight_bits, 3u,
                             static_cast<uint32_t>(cmd.act),
                             static_cast<uint32_t>(cmd.has_residual), cmd.h_out, cmd.w_out},
                            wp_cnt);
        uint32_t aux_off =
            add_ext_u32_arr({w_offset, w_bytes, idx_offset, idx_bytes, bias_offset, bias_bytes,
                             cmd.src_off, cmd.y_off, wave_data_start, wave_data_end, cmd.res_off},
                            aux_cnt);
        uint32_t hash_off =
            add_ext_u32_arr({static_cast<uint32_t>(wave_hash64 & 0xffffffffULL),
                             static_cast<uint32_t>((wave_hash64 >> 32) & 0xffffffffULL)},
                            dummy_cnt);
        uint32_t geom_off = 0u;
        if (include_geometry) {
            uint32_t ndims = 3u;
            uint32_t d0 = 1u;
            uint32_t d1 = cmd.logical_Cout;
            uint32_t d2 = cmd.h_out;
            uint32_t d3 = cmd.w_out;
            uint32_t s0 = cmd.Cout;
            uint32_t s1 = 1u;
            uint32_t s2 = 1u;
            uint32_t s3 = 1u;
            uint32_t pad_off = 0u;
            auto rit = plan.res_map.find(cmd.res_key_out);
            if (rit != plan.res_map.end()) {
                const auto &out_layout = rit->second;
                if (out_layout.is_1d_tokens) {
                    uint32_t num_tokens = std::max(1u, cmd.w_out);
                    ndims = (num_tokens > 1u) ? 2u : 1u;
                    d0 = num_tokens;
                    d1 = cmd.logical_Cout;
                    d2 = 1u;
                    d3 = 1u;
                    s0 = 1u;
                    s1 = out_layout.channel_stride;
                    s2 = 1u;
                    s3 = 1u;
                    pad_off = 0u;
                } else {
                    ndims = 3u;
                    d0 = 1u;
                    d1 = cmd.logical_Cout;
                    d2 = cmd.h_out;
                    d3 = cmd.w_out;
                    s0 = cmd.Cout * out_layout.channel_stride;
                    s1 = out_layout.channel_stride;
                    s2 = out_layout.W_pad;
                    s3 = 1u;
                    pad_off = out_layout.W_pad + 4u;
                }
            }
            geom_off = add_ext_u32_arr({ndims, d0, d1, d2, d3, s0, s1, s2, s3, pad_off}, geom_cnt);
        }

        std::vector<TiffIFDEntry> entries = {{254, TIFF_LONG, 1, 2u},
                                             {256, TIFF_LONG, 1, cmd.B},
                                             {257, TIFF_LONG, 1, cmd.Cout},
                                             {258, TIFF_SHORT, 1, cmd.weight_bits},
                                             {259, TIFF_SHORT, 1, 1u},
                                             {262, TIFF_SHORT, 1, 1u},
                                             {266, TIFF_SHORT, 1, 1u},
                                             {270, TIFF_ASCII, desc_len, desc_off},
                                             {273, TIFF_LONG, 1, w_offset},
                                             {277, TIFF_SHORT, 1, 1u},
                                             {278, TIFF_LONG, 1, cmd.Cout},
                                             {279, TIFF_LONG, 1, w_bytes},
                                             {282, TIFF_RATIONAL, 1, res_rat_off},
                                             {283, TIFF_RATIONAL, 1, res_rat_off},
                                             {285, TIFF_ASCII, page_len, page_off},
                                             {297, TIFF_SHORT, 2, w_idx | (total_pages << 16)},
                                             {339, TIFF_SHORT, 1, 3u},
                                             {TAG_TCFF_WAVE_MAGIC, TIFF_LONG, 1, TCFF_MAGIC_WAVE},
                                             {TAG_TCFF_WAVE_PARAMS, TIFF_LONG, wp_cnt, wp_off},
                                             {TAG_TCFF_AUX_OFFSETS, TIFF_LONG, aux_cnt, aux_off},
                                             {TAG_TCFF_DATA_HASH64, TIFF_LONG, 2, hash_off}};
        if (include_geometry) {
            entries.push_back({TAG_TCFF_WAVE_GEOMETRY, TIFF_LONG, geom_cnt, geom_off});
        }
        writer.u16(static_cast<uint16_t>(entries.size()));
        for (const auto &entry : entries) {
            writer.u16(entry.tag);
            writer.u16(entry.typ);
            writer.u32(entry.count);
            writer.u32(entry.val_or_off);
        }
        prev_next_ifd_ptr_off = static_cast<uint32_t>(writer.buf.size());
        writer.u32(0u);
        writer.raw(ext.buf.data(), ext.buf.size());
        writer.pad(stride_alignment);
    }

    FILE *fp = std::fopen(tcff_path, "wb");
    if (!fp) {
        return false;
    }
    size_t written = std::fwrite(writer.buf.data(), 1, writer.buf.size(), fp);
    std::fclose(fp);
    return written == writer.buf.size();
}

static inline uint16_t read_u16_le(const uint8_t *ptr) {
    return (uint16_t)ptr[0] | ((uint16_t)ptr[1] << 8);
}

static inline uint32_t read_u32_le(const uint8_t *ptr) {
    return (uint32_t)ptr[0] | ((uint32_t)ptr[1] << 8) | ((uint32_t)ptr[2] << 16) |
           ((uint32_t)ptr[3] << 24);
}

static inline uint64_t read_u64_le(const uint8_t *ptr) {
    return (uint64_t)read_u32_le(ptr) | ((uint64_t)read_u32_le(ptr + 4) << 32);
}

// Deserializes pre-interleaved WMMA weights (`cmd.h_w_wmma`), FP16 biases (`cmd.h_bias`), and
// connectome gather indices (`cmd.h_idx`, copied when `plan.batch_size == 1` and the canvas byte
// size `plan.canvas_elems * 2` matches `tcff_canvas_bytes` in the file header) from `tcff_path`
// into `plan`, verifying both the 64-bit FNV-1a master header hash and each per-wave IFD hash.
static bool load_weights_from_tcff(K44WavePlan &plan, const char *tcff_path) {
    FILE *fp = std::fopen(tcff_path, "rb");
    if (!fp) {
        std::fprintf(stderr, "Failed to open TCFF file: %s\n", tcff_path);
        return false;
    }
    std::fseek(fp, 0, SEEK_END);
    long file_size = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (file_size < 256) {
        std::fclose(fp);
        return false;
    }
    std::vector<uint8_t> buf((size_t)file_size);
    if (std::fread(buf.data(), 1, (size_t)file_size, fp) != (size_t)file_size) {
        std::fclose(fp);
        return false;
    }
    std::fclose(fp);
    if (buf[0] != 'I' || buf[1] != 'I' || read_u16_le(buf.data() + 2) != 42u) {
        std::fprintf(stderr, "Invalid TIFF header in TCFF file: %s\n", tcff_path);
        return false;
    }
    if (std::memcmp(buf.data() + 8, "TCFFv1.0", 8) != 0) {
        std::fprintf(stderr, "Invalid TCFFv1.0 magic in: %s\n", tcff_path);
        return false;
    }
    uint64_t expected_master_hash = read_u64_le(buf.data() + 32);
    uint64_t actual_master_hash = fnv1a_hash64(buf.data() + 8, 24);
    if (expected_master_hash != actual_master_hash) {
        std::fprintf(stderr, "Master header FNV-1a hash mismatch in: %s\n", tcff_path);
        return false;
    }
    uint32_t tcff_canvas_bytes = read_u32_le(buf.data() + 24);
    bool use_tcff_idx =
        (plan.batch_size == 1u) &&
        ((uint64_t)plan.canvas_elems * sizeof(uint16_t) == (uint64_t)tcff_canvas_bytes);
    std::vector<K44ExecCmd *> wave_cmds;
    for (auto &cmd : plan.cmds) {
        if (cmd.type == K44CmdType::K44_WAVE) {
            wave_cmds.push_back(&cmd);
        }
    }
    uint32_t ifd_off = read_u32_le(buf.data() + 4);
    size_t w_idx = 0;
    while (ifd_off != 0u && w_idx < wave_cmds.size()) {
        if (ifd_off + 2u > buf.size()) {
            return false;
        }
        uint16_t num_tags = read_u16_le(buf.data() + ifd_off);
        uint32_t entries_off = ifd_off + 2u;
        uint32_t wave_magic = 0u;
        uint32_t aux_off = 0u;
        uint32_t aux_cnt = 0u;
        uint32_t hash_off = 0u;
        for (uint16_t tag_idx = 0; tag_idx < num_tags; ++tag_idx) {
            const uint8_t *entry_ptr = buf.data() + entries_off + tag_idx * 12u;
            uint16_t tag = read_u16_le(entry_ptr);
            uint32_t cnt = read_u32_le(entry_ptr + 4);
            uint32_t val = read_u32_le(entry_ptr + 8);
            if (tag == TAG_TCFF_WAVE_MAGIC) {
                wave_magic = val;
            } else if (tag == TAG_TCFF_AUX_OFFSETS) {
                aux_off = val;
                aux_cnt = cnt;
            } else if (tag == TAG_TCFF_DATA_HASH64) {
                hash_off = val;
            }
        }
        if (wave_magic == TCFF_MAGIC_CANV) {
            ifd_off = read_u32_le(buf.data() + entries_off + num_tags * 12u);
            continue;
        }
        if (aux_off == 0u || aux_cnt < 10u || hash_off == 0u) {
            return false;
        }
        const uint8_t *aux_ptr = buf.data() + aux_off;
        uint32_t w_offset = read_u32_le(aux_ptr + 0);
        uint32_t w_bytes = read_u32_le(aux_ptr + 4);
        uint32_t idx_offset = read_u32_le(aux_ptr + 8);
        uint32_t idx_bytes = read_u32_le(aux_ptr + 12);
        uint32_t bias_offset = read_u32_le(aux_ptr + 16);
        uint32_t bias_bytes = read_u32_le(aux_ptr + 20);
        uint32_t wave_data_start = read_u32_le(aux_ptr + 32);
        uint32_t wave_data_end = read_u32_le(aux_ptr + 36);
        uint64_t expected_wave_hash = read_u64_le(buf.data() + hash_off);
        uint64_t actual_wave_hash =
            fnv1a_hash64(buf.data() + wave_data_start, wave_data_end - wave_data_start);
        if (expected_wave_hash != actual_wave_hash) {
            std::fprintf(stderr, "Wave %zu FNV-1a hash mismatch in: %s\n", w_idx, tcff_path);
            return false;
        }
        K44ExecCmd &cmd = *wave_cmds[w_idx];
        size_t expected_w_bytes = (size_t)cmd.Cout * (size_t)cmd.B * sizeof(uint16_t);
        size_t alloc_w_bytes = std::max(expected_w_bytes, (size_t)w_bytes);
        cmd.h_w_wmma.assign(alloc_w_bytes / sizeof(uint4_host), {0u, 0u, 0u, 0u});
        std::memcpy(cmd.h_w_wmma.data(), buf.data() + w_offset,
                    std::min(expected_w_bytes, (size_t)w_bytes));
        if (use_tcff_idx && idx_bytes == cmd.h_idx.size() * sizeof(uint32_t)) {
            std::memcpy(cmd.h_idx.data(), buf.data() + idx_offset, idx_bytes);
        }
        size_t expected_bias_len = std::max<size_t>(cmd.Cout, bias_bytes / sizeof(uint16_t));
        cmd.h_bias.assign(expected_bias_len, 0u);
        std::memcpy(cmd.h_bias.data(), buf.data() + bias_offset,
                    std::min((size_t)cmd.Cout * sizeof(uint16_t), (size_t)bias_bytes));
        ifd_off = read_u32_le(buf.data() + entries_off + num_tags * 12u);
        w_idx++;
    }
    return w_idx == wave_cmds.size();
}

} // namespace k44_host
