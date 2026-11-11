#pragma once
#include "k44_plan_types.hpp"
#include "k44_tcff_io.hpp"

namespace k44_host {

// =================================================================================================
// Wave Graph Compiler (K44WaveGraphCompiler)
// =================================================================================================

class K44WaveGraphCompiler {
  private:
    struct Shape3D {
        uint32_t c;
        uint32_t h;
        uint32_t w;
        bool is_1d;
    };

    struct PhysAlloc {
        uint32_t res_key;
        uint32_t num_ch;
        uint32_t canvas_off;
    };

    static inline uint32_t make_res_key(uint32_t height, uint32_t width, bool is_1d) {
        return is_1d ? (0x80000000u | width)
                     : ((height == width) ? height : ((height << 14) ^ width));
    }

    static void init_node_bias(K44IRNode &node, uint32_t cout, const float *bias_f32) {
        node.bias_f32.assign(cout, 0.0f);
        node.bias_f16.assign(cout, 0u);
        if (bias_f32) {
            std::memcpy(node.bias_f32.data(), bias_f32, cout * sizeof(float));
            for (uint32_t i = 0; i < cout; ++i) {
                node.bias_f16[i] = float_to_half(bias_f32[i]);
            }
        }
    }

    static K44ExecCmd make_base_cmd(const K44IRNode &node, K44CmdType cmd_type, uint32_t rkey_in,
                                    uint32_t rkey_out, uint32_t src_off, uint32_t dst_off) {
        K44ExecCmd cmd{};
        cmd.name = node.name;
        cmd.type = cmd_type;
        cmd.Cin = (node.cin > 0u) ? node.cin : node.cout;
        cmd.Cout = node.cout;
        cmd.logical_Cout = node.cout;
        cmd.h_in = node.h_in;
        cmd.w_in = node.w_in;
        cmd.h_out = node.h_out;
        cmd.w_out = node.w_out;
        cmd.res_key_in = rkey_in;
        cmd.res_key_out = rkey_out;
        cmd.src_off = src_off;
        cmd.y_off = dst_off;
        return cmd;
    }

    // Pass 1: Propagates 2D (`[C, H, W]`) and 1D (`[D, 1, SeqLen]`) tensor shapes across the graph.
    // Variables & Tensor Dimensions:
    // - `plan`: Target `K44WavePlan` recording input dimensions and zero-copy optimization counts.
    // - `shapes`: Map from tensor name to `Shape3D {c, h, w, is_1d}`.
    // - `C` (`c`, `node.cin`, `node.cout`): Number of feature-map channels (or embedding dimension
    //   `D` in 1D token mode).
    // - `H` (`h`, `node.h_in`, `node.h_out`): Spatial feature-map height in pixels (`1` in 1D
    // mode).
    // - `W` (`w`, `node.w_in`, `node.w_out`): Spatial feature-map width in pixels (or token
    // sequence
    //   length `SeqLen` in 1D mode).
    // - `kh`, `kw`: Convolution/Pooling spatial filter height and width in pixels.
    // - `stride`: Spatial downsampling factor (for `Convolution`/`Pooling`) or upsampling factor
    //   (for `Deconvolution`).
    // Also auto-inserts a zero-copy `Flatten` view when a `1x1` (`kh == 1, kw == 1`) dense linear
    // layer consumes a 2D tensor (`node.cin == c * h * w`), and records zero-copy topology ops
    // (`Split`, `Slice`, `Concat`, `Flatten`) for compile-time elimination in Pass 5.
    void pass1_infer_shapes(K44WavePlan &plan, std::map<std::string, Shape3D> &shapes) {
        if (!nodes.empty() && nodes[0].op_type == "Input" && nodes[0].is_1d_tokens &&
            opts.batch_size > 1u && nodes[0].w_out == 1u) {
            nodes[0].w_in = opts.batch_size;
            nodes[0].w_out = opts.batch_size;
        }
        std::vector<K44IRNode> expanded_nodes;
        expanded_nodes.reserve(nodes.size() + 8);
        for (auto &node : nodes) {
            if (node.op_type == "Input") {
                shapes[node.out_tensors[0]] = {node.cout, node.h_out, node.w_out,
                                               node.is_1d_tokens};
                plan.input_c = node.cout;
                plan.input_h = node.h_out;
                plan.input_w = node.w_out;
                plan.input_res_key = node.is_1d_tokens ? (0x80000000u | node.w_out) : node.h_out;
            } else if (node.op_type == "Flatten") {
                Shape3D shape = shapes.at(node.in_tensors[0]);
                node.cin = shape.c;
                node.cout = shape.c * shape.h * shape.w;
                node.h_in = shape.h;
                node.w_in = shape.w;
                node.h_out = 1u;
                node.w_out = 1u;
                node.is_1d_tokens = true;
                shapes[node.out_tensors[0]] = {node.cout, 1u, 1u, true};
                plan.num_zero_copy_flattens++;
            } else if (node.op_type == "Split") {
                Shape3D shape = shapes.at(node.in_tensors[0]);
                node.is_1d_tokens = shape.is_1d;
                for (const auto &out_name : node.out_tensors) {
                    shapes[out_name] = shape;
                }
                plan.num_eliminated_splits++;
            } else if (node.op_type == "Slice") {
                Shape3D shape = shapes.at(node.in_tensors[0]);
                node.is_1d_tokens = shape.is_1d;
                if (node.slice_sizes.empty()) {
                    uint32_t part = shape.c / (uint32_t)node.out_tensors.size();
                    for (const auto &out_name : node.out_tensors) {
                        shapes[out_name] = {part, shape.h, shape.w, shape.is_1d};
                    }
                } else {
                    for (size_t k = 0; k < node.out_tensors.size(); ++k) {
                        shapes[node.out_tensors[k]] = {node.slice_sizes[k], shape.h, shape.w,
                                                       shape.is_1d};
                    }
                }
                plan.num_eliminated_splits++;
            } else if (node.op_type == "Concat") {
                Shape3D first_shape = shapes.at(node.in_tensors[0]);
                node.is_1d_tokens = first_shape.is_1d;
                uint32_t ch_sum = 0;
                for (const auto &in_name : node.in_tensors) {
                    ch_sum += shapes.at(in_name).c;
                }
                shapes[node.out_tensors[0]] = {ch_sum, first_shape.h, first_shape.w,
                                               first_shape.is_1d};
                plan.num_eliminated_concats++;
            } else if (node.op_type == "Convolution") {
                Shape3D shape = shapes.at(node.in_tensors[0]);
                if (!shape.is_1d && node.kh == 1u && node.kw == 1u &&
                    node.cin == shape.c * shape.h * shape.w &&
                    (shape.h > 1u || shape.w > 1u || !shape.is_1d) && node.cin != shape.c) {
                    std::string flat_tensor = node.in_tensors[0] + "_flat_view";
                    K44IRNode flat_node;
                    flat_node.op_type = "Flatten";
                    flat_node.name = node.name + "_auto_flatten";
                    flat_node.in_tensors = {node.in_tensors[0]};
                    flat_node.out_tensors = {flat_tensor};
                    flat_node.cin = shape.c;
                    flat_node.cout = shape.c * shape.h * shape.w;
                    flat_node.h_in = shape.h;
                    flat_node.w_in = shape.w;
                    flat_node.h_out = 1u;
                    flat_node.w_out = 1u;
                    flat_node.is_1d_tokens = true;
                    expanded_nodes.push_back(std::move(flat_node));
                    node.in_tensors[0] = flat_tensor;
                    shape = {shape.c * shape.h * shape.w, 1u, 1u, true};
                    shapes[flat_tensor] = shape;
                    plan.num_zero_copy_flattens++;
                }
                node.is_1d_tokens = shape.is_1d;
                node.h_in = shape.h;
                node.w_in = shape.w;
                node.h_out = shape.is_1d ? 1u : (shape.h / node.stride);
                node.w_out = shape.is_1d ? shape.w : (shape.w / node.stride);
                shapes[node.out_tensors[0]] = {node.cout, node.h_out, node.w_out, shape.is_1d};
            } else if (node.op_type == "Deconvolution") {
                Shape3D shape = shapes.at(node.in_tensors[0]);
                node.is_1d_tokens = false;
                node.h_in = shape.h;
                node.w_in = shape.w;
                node.h_out = shape.h * node.stride;
                node.w_out = shape.w * node.stride;
                shapes[node.out_tensors[0]] = {node.cout, node.h_out, node.w_out, false};
            } else if (node.op_type == "Pooling") {
                Shape3D shape = shapes.at(node.in_tensors[0]);
                node.cin = shape.c;
                node.cout = shape.c;
                node.h_in = shape.h;
                node.w_in = shape.w;
                node.h_out = shape.h / node.stride;
                node.w_out = shape.w / node.stride;
                shapes[node.out_tensors[0]] = {node.cout, node.h_out, node.w_out, false};
            } else if (node.op_type == "Interp") {
                Shape3D shape = shapes.at(node.in_tensors[0]);
                node.cin = shape.c;
                node.cout = shape.c;
                node.h_in = shape.h;
                node.w_in = shape.w;
                node.h_out = shape.h * 2u;
                node.w_out = shape.w * 2u;
                shapes[node.out_tensors[0]] = {node.cout, node.h_out, node.w_out, false};
            } else if (node.op_type == "RMSNorm") {
                Shape3D shape = shapes.at(node.in_tensors[0]);
                node.is_1d_tokens = shape.is_1d;
                node.h_in = shape.h;
                node.h_out = shape.h;
                node.w_in = shape.w;
                node.w_out = shape.w;
                shapes[node.out_tensors[0]] = shape;
            } else if (node.op_type == "SwiGLU") {
                Shape3D shape = shapes.at(node.in_tensors[0]);
                node.is_1d_tokens = shape.is_1d;
                node.cin = shape.c;
                node.cout = shape.c;
                node.h_in = shape.h;
                node.h_out = shape.h;
                node.w_in = shape.w;
                node.w_out = shape.w;
                shapes[node.out_tensors[0]] = shape;
            } else if (node.op_type == "RoPEAttention") {
                Shape3D shape = shapes.at(node.in_tensors[0]);
                node.is_1d_tokens = shape.is_1d;
                node.h_in = shape.h;
                node.h_out = shape.h;
                node.w_in = shape.w;
                node.w_out = shape.w;
                shapes[node.out_tensors[0]] = {node.cout, shape.h, shape.w, shape.is_1d};
            }
            expanded_nodes.push_back(std::move(node));
        }
        nodes = std::move(expanded_nodes);
    }

    // Pass 2: Folds `RMSNorm` per-channel scale weights (`gamma[ch]`, stored in
    // `nodes[i].bias_f32`) into the synaptic weights of all downstream `1x1` (`kh == 1, kw == 1`)
    // linear projections:
    //   `W_folded[n_row, ch] = W[n_row, ch] * gamma[ch]`
    // where `n_row` (`0 .. cons_node.cout - 1`) is the output neuron row index and `ch`
    // (`0 .. cons_node.cin - 1`) is the input channel index. Folding is applied iff every consumer
    // of the normalized tensor (including aliases created by `Split`) is a `1x1` projection and the
    // normalized tensor is never used as an unscaled residual skip connection (`residual_tensor`).
    void pass2_fold_rmsnorm(K44WavePlan &plan) {
        if (!opts.fold_rmsnorm_gamma) {
            return;
        }
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].op_type != "RMSNorm") {
                continue;
            }
            if (nodes[i].gamma_folded) {
                plan.num_folded_rmsnorms++;
                continue;
            }
            std::set<std::string> aliases = {nodes[i].out_tensors[0]};
            bool all_linear_consumers = true;
            std::vector<size_t> consumers;
            for (size_t j = i + 1; j < nodes.size(); ++j) {
                if (nodes[j].op_type == "Split" && aliases.count(nodes[j].in_tensors[0])) {
                    for (const auto &out_name : nodes[j].out_tensors) {
                        aliases.insert(out_name);
                    }
                    continue;
                }
                for (const auto &in_name : nodes[j].in_tensors) {
                    if (aliases.count(in_name)) {
                        if (nodes[j].op_type == "Convolution" && nodes[j].kh == 1u &&
                            nodes[j].kw == 1u) {
                            consumers.push_back(j);
                        } else {
                            all_linear_consumers = false;
                        }
                    }
                }
                if (!nodes[j].residual_tensor.empty() && aliases.count(nodes[j].residual_tensor)) {
                    all_linear_consumers = false;
                }
            }
            if (all_linear_consumers && !consumers.empty()) {
                const auto &gamma = nodes[i].bias_f32;
                for (size_t cons_idx : consumers) {
                    auto &cons_node = nodes[cons_idx];
                    if (!cons_node.weights_f32.empty()) {
                        for (uint32_t n_row = 0; n_row < cons_node.cout; ++n_row) {
                            for (uint32_t ch = 0; ch < cons_node.cin; ++ch) {
                                float folded =
                                    cons_node.weights_f32[(size_t)n_row * cons_node.cin + ch] *
                                    gamma[ch];
                                cons_node.weights_f32[(size_t)n_row * cons_node.cin + ch] = folded;
                                cons_node.weights_f16[(size_t)n_row * cons_node.cin + ch] =
                                    float_to_half(folded);
                            }
                        }
                    } else {
                        for (uint32_t n_row = 0; n_row < cons_node.cout; ++n_row) {
                            for (uint32_t ch = 0; ch < cons_node.cin; ++ch) {
                                float folded =
                                    half_to_float(
                                        cons_node.weights_f16[(size_t)n_row * cons_node.cin + ch]) *
                                    gamma[ch];
                                cons_node.weights_f16[(size_t)n_row * cons_node.cin + ch] =
                                    float_to_half(folded);
                            }
                        }
                    }
                }
                nodes[i].gamma_folded = true;
                plan.num_folded_rmsnorms++;
            }
        }
    }

    // Pass 3: Builds `K44ResDesc` resolution descriptors (`plan.res_map`) and wave tiling configs
    // (`node_wave_cfg`).
    // Variables:
    // - `plan`: Target `K44WavePlan` storing `res_map`, `batch_size`, and `wave_batch_size`.
    // - `shapes`: Map of tensor names to `Shape3D {c, h, w, is_1d}` from Pass 1.
    // - `node_wave_cfg`: Output vector of `K44WaveConfig` indexed by node `step` (`0 ..
    // nodes.size()-1`),
    //   storing cluster size `S1` (`S` in `{16, 32, 64, 128}` output neurons per cluster) and
    //   16-neuron-aligned output channel count `Cout_pad1`.
    // - `B_frames` (`plan.wave_batch_size`): Number of frames processed per wave dispatch.
    // - `spatial_in`, `spatial_out`: Number of 64-element spatial/token tiles (1D) or pixels (2D)
    //   used to balance cluster size `S` against CU occupancy.
    void pass3_build_layouts(K44WavePlan &plan, const std::map<std::string, Shape3D> &shapes,
                             std::vector<K44WaveConfig> &node_wave_cfg) {
        const bool graph_is_1d =
            (!nodes.empty() && nodes[0].op_type == "Input" && nodes[0].is_1d_tokens);
        const bool is_image_token64 =
            (!graph_is_1d) && (opts.batch_size >= 64u) && (opts.batch_size % 64u == 0u);
        uint32_t B_frames =
            graph_is_1d ? 1u
                        : (is_image_token64 ? 64u : std::min(std::max(1u, opts.batch_size), 8u));
        if (is_image_token64) {
            const char *env_wb = std::getenv("WAVEC_WAVE_BATCH");
            if (env_wb && std::atoi(env_wb) >= 64 && (std::atoi(env_wb) % 64 == 0)) {
                B_frames = std::min(opts.batch_size, (uint32_t)std::atoi(env_wb));
            } else if (opts.reuse_canvas && opts.batch_size <= 256u) {
                B_frames = opts.batch_size;
            }
        }
        plan.batch_size = std::max(1u, opts.batch_size);
        plan.wave_batch_size = B_frames;
        plan.is_image_token64 = is_image_token64;

        for (const auto &kv : shapes) {
            uint32_t key = make_res_key(kv.second.h, kv.second.w, kv.second.is_1d);
            if (!plan.res_map.count(key)) {
                plan.res_map[key].init(kv.second.h, kv.second.w, kv.second.is_1d, B_frames);
            }
        }
        plan.hw_profile = opts.hw_profile;

        const uint32_t num_img_blks = is_image_token64 ? (B_frames / 64u) : 1u;
        node_wave_cfg.assign(nodes.size(), {16u, 16u, 0u, 0u, 1u});
        for (size_t step = 0; step < nodes.size(); ++step) {
            const auto &node = nodes[step];
            if (node.op_type == "Convolution") {
                uint32_t spatial_in =
                    node.is_1d_tokens
                        ? std::max(
                              1u,
                              plan.res_map.at(make_res_key(node.h_in, node.w_in, true)).num_tiles)
                        : (std::max(1u, node.h_in * node.w_in) * num_img_blks);
                uint32_t spatial_out =
                    node.is_1d_tokens
                        ? std::max(
                              1u,
                              plan.res_map.at(make_res_key(node.h_out, node.w_out, true)).num_tiles)
                        : (std::max(1u, node.h_out * node.w_out) * num_img_blks);
                node_wave_cfg[step] = select_optimal_wave_config(
                    node.cin, node.cout, node.kh, node.kw, spatial_in, spatial_out,
                    opts.force_cluster_size, opts.hw_profile, opts.max_B_per_wave);
            }
        }
    }

    // Pass 4: Computes tensor liveness across zero-copy aliases (`Split`, `Slice`, `Concat`,
    // `Flatten`) and packs all intermediate activations, auxiliary scratchpads, and KV caches into
    // a single contiguous 64-element-aligned FP16 arena (`d_Canvas`) using best-fit free-interval
    // allocation with adjacent-interval coalescing.
    // Parameters & Variables:
    // - `plan`: Target `K44WavePlan` receiving `canvas_elems`, `kv_cache_elems`, and `input_off`.
    // - `node_wave_cfg`: Per-node wave configs (`total_cout_pad()` gives padded output channels).
    // - `allocs`: Output map from physical allocation ID (`aid`) to `PhysAlloc {res_key, num_ch,
    //   canvas_off}`, where `res_key` (`rkey`) identifies the `K44ResDesc` resolution layout,
    //   `num_ch` is the allocated channel count, and `canvas_off` is the FP16 element offset in
    //   `d_Canvas`.
    // - `node_out_aid`: Output vector mapping each node index `step` to its output `aid` (`-1` for
    //   zero-copy view ops).
    // - `s2d_base_off`: Output map from output resolution key `rkey` to the `d_Canvas` scratchpad
    //   offset reserved for `S2D_2X2` (`4 * cin` polyphase channels).
    // - `upquad_base_off`: Output map from input resolution key `rkey` to the `d_Canvas` scratchpad
    //   offset reserved for `Deconvolution` (`4 * round_up_16(cout)` quadrant channels).
    // - `node_kv_offs`: Output map from `RoPEAttention` node `step` to `{k_off, v_off}` KV-cache
    //   offsets in `d_Canvas`.
    // - `attn_probs_off`: Output `d_Canvas` offset for causal attention probability workspace.
    void pass4_allocate_canvas(K44WavePlan &plan, const std::vector<K44WaveConfig> &node_wave_cfg,
                               std::map<int, PhysAlloc> &allocs, std::vector<int> &node_out_aid,
                               std::map<uint32_t, uint32_t> &s2d_base_off,
                               std::map<uint32_t, uint32_t> &upquad_base_off,
                               std::map<size_t, std::pair<uint32_t, uint32_t>> &node_kv_offs,
                               uint32_t &attn_probs_off) {
        std::map<std::string, std::set<int>> tensor_alloc_ids;
        node_out_aid.assign(nodes.size(), -1);
        int next_aid = 0;
        for (size_t step = 0; step < nodes.size(); ++step) {
            const auto &node = nodes[step];
            if (node.op_type == "Split" || node.op_type == "Slice" || node.op_type == "Flatten") {
                const auto &src_aids = tensor_alloc_ids.at(node.in_tensors[0]);
                for (const auto &out_name : node.out_tensors) {
                    tensor_alloc_ids[out_name] = src_aids;
                }
            } else if (node.op_type == "Concat") {
                std::set<int> merged_aids;
                for (const auto &in_name : node.in_tensors) {
                    const auto &in_aids = tensor_alloc_ids.at(in_name);
                    merged_aids.insert(in_aids.begin(), in_aids.end());
                }
                tensor_alloc_ids[node.out_tensors[0]] = merged_aids;
            } else {
                int aid = next_aid++;
                uint32_t rkey = make_res_key(node.h_out, node.w_out, node.is_1d_tokens);
                uint32_t alloc_ch = (node.op_type == "Convolution")
                                        ? node_wave_cfg[step].total_cout_pad()
                                        : node.cout;
                allocs[aid] = {rkey, alloc_ch, 0u};
                tensor_alloc_ids[node.out_tensors[0]] = {aid};
                node_out_aid[step] = aid;
            }
        }

        std::map<int, size_t> last_use;
        last_use[0] = nodes.size();
        if (tensor_alloc_ids.count(default_output_tensor)) {
            for (int aid : tensor_alloc_ids.at(default_output_tensor)) {
                last_use[aid] = nodes.size();
            }
        }
        for (size_t step = 0; step < nodes.size(); ++step) {
            const auto &node = nodes[step];
            for (const auto &in_name : node.in_tensors) {
                if (tensor_alloc_ids.count(in_name)) {
                    for (int aid : tensor_alloc_ids.at(in_name)) {
                        last_use[aid] = std::max(last_use[aid], step);
                    }
                }
            }
            if (!node.residual_tensor.empty() && tensor_alloc_ids.count(node.residual_tensor)) {
                for (int aid : tensor_alloc_ids.at(node.residual_tensor)) {
                    last_use[aid] = std::max(last_use[aid], step);
                }
            }
        }

        plan.reuse_canvas = opts.reuse_canvas;
        uint32_t cur_canvas_off = 0u;
        uint32_t s2_subwave_peak_elems = 0u;
        std::map<uint32_t, uint32_t> s2d_peak_ch;
        std::map<uint32_t, uint32_t> upquad_peak_ch;
        for (size_t step = 0; step < nodes.size(); ++step) {
            const auto &node = nodes[step];
            if (node.op_type == "Convolution" && node.stride == 2u) {
                if (!plan.is_image_token64) {
                    uint32_t rkey = make_res_key(node.h_out, node.w_out, false);
                    s2d_peak_ch[rkey] = std::max(s2d_peak_ch[rkey], 4u * node.cin);
                } else if (node_wave_cfg[step].num_sub_b > 1u) {
                    uint32_t rkey_in = make_res_key(node.h_in, node.w_in, false);
                    uint32_t elems = node_wave_cfg[step].total_cout_pad() *
                                     plan.res_map.at(rkey_in).channel_stride;
                    s2_subwave_peak_elems = std::max(s2_subwave_peak_elems, elems);
                }
            } else if (node.op_type == "Deconvolution") {
                uint32_t rkey = make_res_key(node.h_in, node.w_in, false);
                upquad_peak_ch[rkey] = std::max(upquad_peak_ch[rkey], 4u * round_up_16(node.cout));
            }
        }

        struct ElemInterval {
            uint32_t elem_start;
            uint32_t elem_len;
        };
        std::vector<ElemInterval> global_free_pool;
        uint32_t peak_canvas_elems = 0u;
        std::map<int, uint32_t> aid_elem_start;
        std::map<int, uint32_t> aid_elem_len;
        std::set<int> active_aids;

        for (size_t step = 0; step < nodes.size(); ++step) {
            int aid = node_out_aid[step];
            if (aid != -1) {
                uint32_t rkey = allocs[aid].res_key;
                uint32_t raw_need = allocs[aid].num_ch * plan.res_map.at(rkey).channel_stride;
                uint32_t need = (raw_need + 63u) & ~63u;
                aid_elem_len[aid] = need;
                int best_idx = -1;
                for (size_t k = 0; k < global_free_pool.size(); ++k) {
                    if (global_free_pool[k].elem_len >= need) {
                        if (best_idx == -1 ||
                            global_free_pool[k].elem_len < global_free_pool[best_idx].elem_len) {
                            best_idx = (int)k;
                        }
                    }
                }
                if (best_idx != -1) {
                    aid_elem_start[aid] = global_free_pool[best_idx].elem_start;
                    if (global_free_pool[best_idx].elem_len == need) {
                        global_free_pool.erase(global_free_pool.begin() + best_idx);
                    } else {
                        global_free_pool[best_idx].elem_start += need;
                        global_free_pool[best_idx].elem_len -= need;
                    }
                } else if (!global_free_pool.empty() &&
                           global_free_pool.back().elem_start + global_free_pool.back().elem_len ==
                               peak_canvas_elems) {
                    uint32_t tail_start = global_free_pool.back().elem_start;
                    global_free_pool.pop_back();
                    aid_elem_start[aid] = tail_start;
                    peak_canvas_elems = tail_start + need;
                } else {
                    aid_elem_start[aid] = peak_canvas_elems;
                    peak_canvas_elems += need;
                }
                allocs[aid].canvas_off = aid_elem_start[aid];
                active_aids.insert(aid);
            }
            bool retired_any = false;
            for (auto it = active_aids.begin(); it != active_aids.end();) {
                int active_aid = *it;
                if (last_use[active_aid] <= step) {
                    global_free_pool.push_back(
                        {aid_elem_start[active_aid], aid_elem_len[active_aid]});
                    it = active_aids.erase(it);
                    retired_any = true;
                } else {
                    ++it;
                }
            }
            if (retired_any && global_free_pool.size() > 1) {
                std::sort(global_free_pool.begin(), global_free_pool.end(),
                          [](const ElemInterval &iv_a, const ElemInterval &iv_b) {
                              return iv_a.elem_start < iv_b.elem_start;
                          });
                std::vector<ElemInterval> merged;
                for (const auto &iv : global_free_pool) {
                    if (!merged.empty() &&
                        merged.back().elem_start + merged.back().elem_len == iv.elem_start) {
                        merged.back().elem_len += iv.elem_len;
                    } else {
                        merged.push_back(iv);
                    }
                }
                global_free_pool = std::move(merged);
            }
        }

        cur_canvas_off = (peak_canvas_elems + 63u) & ~63u;
        uint32_t max_s2d_elems = 0u;
        for (const auto &kv : s2d_peak_ch) {
            s2d_base_off[kv.first] = cur_canvas_off;
            uint32_t elems = kv.second * plan.res_map.at(kv.first).channel_stride;
            max_s2d_elems = std::max(max_s2d_elems, elems);
        }
        cur_canvas_off += (max_s2d_elems + 63u) & ~63u;

        uint32_t max_upquad_elems = s2_subwave_peak_elems;
        for (const auto &kv : upquad_peak_ch) {
            upquad_base_off[kv.first] = cur_canvas_off;
            uint32_t elems = kv.second * plan.res_map.at(kv.first).channel_stride;
            max_upquad_elems = std::max(max_upquad_elems, elems);
        }
        cur_canvas_off += (max_upquad_elems + 63u) & ~63u;

        attn_probs_off = 0u;
        size_t kv_elems_total = 0;
        if (opts.kv_cache_max_seq > 0u) {
            uint32_t max_h_q = 0u;
            for (size_t step = 0; step < nodes.size(); ++step) {
                const auto &node = nodes[step];
                if (node.op_type == "RoPEAttention") {
                    uint32_t kv_elems = node.h_kv * node.head_dim * opts.kv_cache_max_seq;
                    uint32_t k_off = cur_canvas_off;
                    cur_canvas_off += kv_elems;
                    uint32_t v_off = cur_canvas_off;
                    cur_canvas_off += kv_elems;
                    node_kv_offs[step] = {k_off, v_off};
                    kv_elems_total += (size_t)2u * kv_elems;
                    max_h_q = std::max(max_h_q, node.h_q);
                }
            }
            if (max_h_q > 0u) {
                attn_probs_off = cur_canvas_off;
                cur_canvas_off += max_h_q * opts.kv_cache_max_seq;
            }
        }
        cur_canvas_off = ((cur_canvas_off + 127u) / 128u) * 128u;
        if (cur_canvas_off == 0u) {
            cur_canvas_off = 128u;
        }
        plan.kv_cache_elems = kv_elems_total;
        plan.canvas_elems = cur_canvas_off;
        plan.input_off = allocs.empty() ? 0u : allocs[0].canvas_off;
    }

    // Emits a `2x2` stride-2 transposed convolution (`Deconvolution`) as a `1x1` `K44_WAVE` command
    // followed by a `SCATTER_UP2X2` deinterleaving command.
    // Parameters & Variables:
    // - `node`: `Deconvolution` IR node with `cin` input channels and `cout` logical output
    // channels.
    // - `rkey_in`, `rkey_out`: Resolution keys for input (`h_in x w_in`) and 2x upsampled output
    //   (`h_out = 2*h_in, w_out = 2*w_in`).
    // - `dst_off`: Final `d_Canvas` output offset (in FP16 elements) at resolution `rkey_out`.
    // - `quad_off`: Intermediate scratchpad offset in `d_Canvas` at input resolution `rkey_in`.
    // - `in_chs`: Per-channel `d_Canvas` base offsets for each input channel `ch` (`0 .. cin-1`).
    // - `cout_aligned`: `round_up_16(node.cout)`, 16-neuron aligned output channels per quadrant.
    // - `Cout_quad`: `4 * cout_aligned`, total output channels across all 4 sub-pixel quadrants
    //   `quad` in `{0, 1, 2, 3}` (`(top-left, top-right, bottom-left, bottom-right)`).
    // - `B`: `round_up_32(node.cin)`, padded synaptic fan-in per output neuron.
    void emit_deconv_waves(K44IRNode &node, uint32_t rkey_in, uint32_t rkey_out, uint32_t dst_off,
                           uint32_t quad_off, const std::vector<uint32_t> &in_chs,
                           K44WavePlan &plan) {
        const auto &in_layout = plan.res_map.at(rkey_in);
        const uint32_t B_frames = plan.wave_batch_size;
        uint32_t cout_aligned = round_up_16(node.cout);
        uint32_t Cout_quad = 4u * cout_aligned;
        uint32_t spatial_tiles_1f = in_layout.is_image_token64
                                        ? ((in_layout.H * (in_layout.W + 4u) + 63u) / 64u)
                                        : in_layout.tiles_per_frame;
        uint32_t cluster_sz =
            choose_cluster_size(Cout_quad, spatial_tiles_1f, opts.force_cluster_size);
        uint32_t num_clusters = Cout_quad / cluster_sz;
        uint32_t B = round_up_32(node.cin);
        std::vector<uint32_t> h_idx((size_t)num_clusters * B, 0u);
        for (uint32_t cl = 0; cl < num_clusters; ++cl) {
            for (uint32_t ch = 0; ch < node.cin; ++ch) {
                h_idx[(size_t)cl * B + ch] = in_layout.tap_offset(in_chs[ch], 0, 0);
            }
        }
        std::vector<uint16_t> h_w_compact;
        std::vector<uint16_t> h_bias_quad(Cout_quad, 0u);
        if (!node.weights_f16.empty()) {
            h_w_compact.assign((size_t)Cout_quad * B, 0u);
            for (uint32_t quad = 0; quad < 4u; ++quad) {
                for (uint32_t j = 0; j < node.cout; ++j) {
                    uint32_t row = quad * cout_aligned + j;
                    if (j < node.bias_f16.size()) {
                        h_bias_quad[row] = node.bias_f16[j];
                    }
                    for (uint32_t ch = 0; ch < node.cin; ++ch) {
                        h_w_compact[(size_t)row * B + ch] =
                            node.weights_f16[((size_t)j * node.cin + ch) * 4u + quad];
                    }
                }
            }
        }
        K44ExecCmd wcmd = make_base_cmd(node, K44CmdType::K44_WAVE, rkey_in, rkey_in, 0u, quad_off);
        wcmd.Cout = Cout_quad;
        wcmd.logical_Cout = 4u * node.cout;
        wcmd.cluster_size = cluster_sz;
        wcmd.B = B;
        wcmd.B_raw = node.cin;
        wcmd.B_vec16 = B / 16u;
        wcmd.has_bias = true;
        wcmd.act = K44Act::NONE;
        wcmd.has_residual = false;
        wcmd.h_out = node.h_in;
        wcmd.w_out = node.w_in;
        double sf_gflops =
            2.0 * (double)node.cin * (double)Cout_quad * (double)(node.h_in * node.w_in) / 1e9;
        wcmd.gflops = (double)B_frames * sf_gflops;
        plan.single_frame_gflops += sf_gflops;
        plan.total_gflops += wcmd.gflops;
        pack_wave_buffers(wcmd, h_w_compact, h_idx, h_bias_quad, plan);
        plan.cmds.push_back(std::move(wcmd));
        plan.num_wave_dispatches++;

        K44ExecCmd scmd =
            make_base_cmd(node, K44CmdType::SCATTER_UP2X2, rkey_in, rkey_out, quad_off, dst_off);
        scmd.name = node.name + "_scatter2x2";
        scmd.Cin = Cout_quad;
        plan.cmds.push_back(std::move(scmd));
    }

    // Emits 1D/2D `Convolution` waves using virtual input channel offsets from
    // `tensor_channel_offsets`.
    // Parameters & Variables:
    // - `node`: `Convolution` IR node (`cin`, `cout`, kernel size `kh x kw`, `stride`).
    // - `wcfg`: `K44WaveConfig` specifying cluster sizes (`S1, S2`) and padded output channels
    //   (`Cout_pad1, Cout_pad2`).
    // - `rkey_in`, `rkey_out`: Input and output resolution layout keys in `plan.res_map`.
    // - `dst_off`: Destination `d_Canvas` offset in FP16 elements.
    // - `s2d_base_off`: Map from `rkey_out` to the `S2D_2X2` scratchpad offset in `d_Canvas`.
    // - `tensor_channel_offsets`: Map from tensor name to per-channel `d_Canvas` offsets
    // (`in_chs`).
    // - Polyphase `2x2` Space-to-Depth (`stride == 2` in 2D mode): First emits `S2D_2X2` to pack
    //   each `2x2` input pixel block of input channel `ch` (`0 .. cin-1`) into 4 half-resolution
    //   polyphase channels `4*ch + 2*py + px`, where `py` (`0` or `1`) is the vertical sub-pixel
    //   parity and `px` (`0` or `1`) is the horizontal sub-pixel parity. For filter tap `(ky, kx)`
    //   (`ky in 0..kh-1, kx in 0..kw-1`), this reduces the stride-2 tap to a stride-1 shift
    //   `dy = (ky == 0) ? -1 : 0`, `dx = (kx == 0) ? -1 : 0` on polyphase channel `4*ch + 2*py +
    //   px`.
    void emit_conv_waves(K44IRNode &node, const K44WaveConfig &wcfg, uint32_t rkey_in,
                         uint32_t rkey_out, uint32_t dst_off,
                         const std::map<uint32_t, uint32_t> &s2d_base_off,
                         const std::map<std::string, std::vector<uint32_t>> &tensor_channel_offsets,
                         K44WavePlan &plan) {
        const auto &in_layout = plan.res_map.at(rkey_in);
        const auto &out_layout = plan.res_map.at(rkey_out);
        const uint32_t B_frames = plan.wave_batch_size;
        const bool is_image_token64 = plan.is_image_token64;
        uint32_t Cout_pad = wcfg.total_cout_pad();
        const auto &in_chs = tensor_channel_offsets.at(node.in_tensors[0]);
        bool has_res = !node.residual_tensor.empty();
        uint32_t res_off = has_res ? tensor_channel_offsets.at(node.residual_tensor)[0] : 0u;
        if (has_res) {
            size_t min_res_end =
                (size_t)res_off + (size_t)Cout_pad * (size_t)out_layout.channel_stride;
            if (min_res_end > plan.canvas_elems) {
                plan.canvas_elems = ((min_res_end + 127ULL) / 128ULL) * 128ULL;
            }
        }
        uint32_t taps = node.kh * node.kw;
        uint32_t B_raw_total = taps * node.cin;
        uint32_t s2d_off = 0u;
        if (node.stride == 2u && !is_image_token64) {
            s2d_off = s2d_base_off.at(rkey_out);
            K44ExecCmd s2d_cmd =
                make_base_cmd(node, K44CmdType::S2D_2X2, rkey_in, rkey_out, in_chs[0], s2d_off);
            s2d_cmd.name = node.name + "_s2d_2x2";
            s2d_cmd.Cout = 4u * node.cin;
            s2d_cmd.logical_Cout = 4u * node.cin;
            plan.cmds.push_back(std::move(s2d_cmd));
        }
        auto tap_canvas_off = [&](uint32_t ch, uint32_t ky, uint32_t kx) -> uint32_t {
            if (node.stride == 2u && !is_image_token64) {
                int dy = (ky == 0u) ? -1 : 0;
                uint32_t py = (ky == 1u) ? 0u : 1u;
                int dx = (kx == 0u) ? -1 : 0;
                uint32_t px = (kx == 1u) ? 0u : 1u;
                uint32_t poly_ch_off =
                    s2d_off + (4u * ch + 2u * py + px) * out_layout.channel_stride;
                return out_layout.tap_offset(poly_ch_off, dy, dx);
            } else if (node.stride == 2u && is_image_token64) {
                int dy = (int)ky - (int)(node.kh / 2u);
                int dx = (int)kx - (int)(node.kw / 2u);
                return in_layout.tap_offset(in_chs[ch], dy, dx);
            } else if (node.kh == 1u && node.kw == 1u) {
                return out_layout.tap_offset(in_chs[ch], 0, 0);
            } else {
                int dy = (int)ky - (int)(node.kh / 2u);
                int dx = (int)kx - (int)(node.kw / 2u);
                return out_layout.tap_offset(in_chs[ch], dy, dx);
            }
        };
        if (wcfg.num_sub_b <= 1u) {
            uint32_t B = round_up_32(B_raw_total);
            uint32_t num_s_splits = wcfg.is_split() ? 2u : 1u;
            for (uint32_t sp = 0; sp < num_s_splits; ++sp) {
                uint32_t S_sp = (sp == 0u) ? wcfg.S1 : wcfg.S2;
                uint32_t Cout_pad_sp = (sp == 0u) ? wcfg.Cout_pad1 : wcfg.Cout_pad2;
                uint32_t c_out_start = (sp == 0u) ? 0u : wcfg.Cout_pad1;
                uint32_t cout_log_sp =
                    (c_out_start < node.cout) ? std::min(Cout_pad_sp, node.cout - c_out_start) : 0u;
                uint32_t num_cl_sp = Cout_pad_sp / S_sp;
                std::vector<uint32_t> h_idx((size_t)num_cl_sp * B, 0u);
                for (uint32_t cl = 0; cl < num_cl_sp; ++cl) {
                    for (uint32_t ch = 0; ch < node.cin; ++ch) {
                        for (uint32_t ky = 0; ky < node.kh; ++ky) {
                            for (uint32_t kx = 0; kx < node.kw; ++kx) {
                                uint32_t syn = ch * taps + ky * node.kw + kx;
                                h_idx[(size_t)cl * B + syn] = tap_canvas_off(ch, ky, kx);
                            }
                        }
                    }
                }
                std::vector<uint16_t> h_bias_pad(Cout_pad_sp, 0u);
                for (uint32_t j = 0; j < cout_log_sp && (c_out_start + j) < node.bias_f16.size();
                     ++j) {
                    h_bias_pad[j] = node.bias_f16[c_out_start + j];
                }
                std::vector<uint16_t> h_w_compact;
                if (!node.weights_f16.empty()) {
                    h_w_compact.assign((size_t)Cout_pad_sp * B, 0u);
                    for (uint32_t j = 0; j < cout_log_sp; ++j) {
                        std::memcpy(&h_w_compact[(size_t)j * B],
                                    &node.weights_f16[(size_t)(c_out_start + j) * B_raw_total],
                                    B_raw_total * sizeof(uint16_t));
                    }
                }
                uint32_t w_rkey_in = (node.stride == 2u && is_image_token64) ? rkey_in : rkey_out;
                uint32_t w_src_off = (node.stride == 2u && !is_image_token64) ? s2d_off : in_chs[0];
                uint32_t w_dst_off = dst_off + c_out_start * out_layout.channel_stride;
                K44ExecCmd wcmd = make_base_cmd(node, K44CmdType::K44_WAVE, w_rkey_in, rkey_out,
                                                w_src_off, w_dst_off);
                wcmd.name = wcfg.is_split() ? (node.name + (sp == 0u ? "_s1" : "_s2")) : node.name;
                wcmd.Cout = Cout_pad_sp;
                wcmd.logical_Cout = cout_log_sp;
                wcmd.cluster_size = S_sp;
                wcmd.B = B;
                wcmd.B_raw = B_raw_total;
                wcmd.B_vec16 = B / 16u;
                wcmd.has_bias = true;
                wcmd.act = node.act;
                wcmd.has_residual = has_res;
                wcmd.res_off = has_res ? (res_off + c_out_start * out_layout.channel_stride) : 0u;
                double sf_gflops = 2.0 * (double)node.cin * (double)cout_log_sp * (double)taps *
                                   (double)(node.h_out * node.w_out) / 1e9;
                wcmd.gflops = (double)B_frames * sf_gflops;
                plan.single_frame_gflops += sf_gflops;
                plan.total_gflops += wcmd.gflops;
                pack_wave_buffers(wcmd, h_w_compact, h_idx, h_bias_pad, plan);
                plan.cmds.push_back(std::move(wcmd));
                plan.num_wave_dispatches++;
            }
        }
        std::vector<uint16_t>().swap(node.weights_f16);
    }

    // Pass 5: Resolves `Split`, `Slice`, `Concat`, and `Flatten` as zero-copy virtual channel
    // offset lists in `tensor_channel_offsets` (no GPU copy kernels emitted), and emits the linear
    // execution stream (`plan.cmds`) of `K44_WAVE` and `k44_valu` dispatches.
    // Parameters & Variables:
    // - `plan`: Target `K44WavePlan` receiving `cmds` and final output tensor metadata.
    // - `shapes`: Map from tensor name to `Shape3D {c, h, w, is_1d}`.
    // - `node_wave_cfg`: Per-node wave tiling configuration (`S1, S2, Cout_pad1, Cout_pad2`).
    // - `allocs`: Map from physical allocation ID (`aid`) to `PhysAlloc {res_key, num_ch,
    // canvas_off}`.
    // - `node_out_aid`: Output allocation ID (`aid`) for each node `step`.
    // - `s2d_base_off`, `upquad_base_off`: Scratchpad offsets in `d_Canvas` for `S2D_2X2` and
    //   `SCATTER_UP2X2`.
    // - `node_kv_offs`, `attn_probs_off`: KV-cache and attention probability offsets in `d_Canvas`.
    // - `tensor_channel_offsets`: Map from tensor name to a `std::vector<uint32_t>` where entry
    //   `ch` holds the exact FP16 element offset of channel `ch` inside `d_Canvas`.
    void pass5_emit_commands(K44WavePlan &plan, const std::map<std::string, Shape3D> &shapes,
                             const std::vector<K44WaveConfig> &node_wave_cfg,
                             const std::map<int, PhysAlloc> &allocs,
                             const std::vector<int> &node_out_aid,
                             const std::map<uint32_t, uint32_t> &s2d_base_off,
                             const std::map<uint32_t, uint32_t> &upquad_base_off,
                             const std::map<size_t, std::pair<uint32_t, uint32_t>> &node_kv_offs,
                             uint32_t attn_probs_off) {
        const uint32_t B_frames = plan.wave_batch_size;
        std::map<std::string, std::vector<uint32_t>> tensor_channel_offsets;

        auto assign_out_channels = [&](const K44IRNode &node, int aid,
                                       uint32_t num_ch) -> uint32_t {
            const auto &alloc = allocs.at(aid);
            uint32_t stride = plan.res_map.at(alloc.res_key).channel_stride;
            std::vector<uint32_t> chs(num_ch);
            for (uint32_t ch = 0; ch < num_ch; ++ch) {
                chs[ch] = alloc.canvas_off + ch * stride;
            }
            tensor_channel_offsets[node.out_tensors[0]] = std::move(chs);
            return alloc.canvas_off;
        };

        for (size_t step = 0; step < nodes.size(); ++step) {
            auto &node = nodes[step];
            if (node.op_type == "Input") {
                assign_out_channels(node, node_out_aid[step], node.cout);
            } else if (node.op_type == "Flatten") {
                const auto &in_shape = shapes.at(node.in_tensors[0]);
                uint32_t rkey_in = make_res_key(in_shape.h, in_shape.w, in_shape.is_1d);
                const auto &in_layout = plan.res_map.at(rkey_in);
                const auto &in_chs = tensor_channel_offsets.at(node.in_tensors[0]);
                std::vector<uint32_t> flat_chs;
                flat_chs.reserve((size_t)in_shape.c * in_shape.h * in_shape.w);
                for (uint32_t ch = 0; ch < in_shape.c; ++ch) {
                    for (uint32_t y = 0; y < in_shape.h; ++y) {
                        for (uint32_t x = 0; x < in_shape.w; ++x) {
                            uint32_t off = in_shape.is_1d
                                               ? (in_chs[ch] + x)
                                               : (in_chs[ch] + (y + 1u) * in_layout.W_pad + 4u + x);
                            flat_chs.push_back(off);
                        }
                    }
                }
                tensor_channel_offsets[node.out_tensors[0]] = std::move(flat_chs);
            } else if (node.op_type == "Split") {
                const auto &chs = tensor_channel_offsets.at(node.in_tensors[0]);
                for (const auto &out_name : node.out_tensors) {
                    tensor_channel_offsets[out_name] = chs;
                }
            } else if (node.op_type == "Slice") {
                const auto &chs = tensor_channel_offsets.at(node.in_tensors[0]);
                if (node.slice_sizes.empty()) {
                    size_t part = chs.size() / node.out_tensors.size();
                    for (size_t k = 0; k < node.out_tensors.size(); ++k) {
                        tensor_channel_offsets[node.out_tensors[k]] = std::vector<uint32_t>(
                            chs.begin() + k * part, chs.begin() + (k + 1) * part);
                    }
                } else {
                    size_t cur = 0;
                    for (size_t k = 0; k < node.out_tensors.size(); ++k) {
                        tensor_channel_offsets[node.out_tensors[k]] = std::vector<uint32_t>(
                            chs.begin() + cur, chs.begin() + cur + node.slice_sizes[k]);
                        cur += node.slice_sizes[k];
                    }
                }
            } else if (node.op_type == "Concat") {
                std::vector<uint32_t> cat;
                for (const auto &in_name : node.in_tensors) {
                    const auto &chs = tensor_channel_offsets.at(in_name);
                    cat.insert(cat.end(), chs.begin(), chs.end());
                }
                tensor_channel_offsets[node.out_tensors[0]] = std::move(cat);
            } else if (node.op_type == "Pooling" || node.op_type == "Interp") {
                int aid = node_out_aid[step];
                uint32_t rkey_in = make_res_key(node.h_in, node.w_in, false);
                uint32_t rkey_out = allocs.at(aid).res_key;
                uint32_t dst_off = assign_out_channels(node, aid, node.cout);
                uint32_t src_off = tensor_channel_offsets.at(node.in_tensors[0])[0];
                K44CmdType ctype =
                    (node.op_type == "Interp")
                        ? K44CmdType::UPSAMPLE_2X
                        : ((node.kh == 2u) ? K44CmdType::MAXPOOL_2X2 : K44CmdType::MAXPOOL_5X5);
                plan.cmds.push_back(
                    make_base_cmd(node, ctype, rkey_in, rkey_out, src_off, dst_off));
            } else if (node.op_type == "RMSNorm") {
                int aid = node_out_aid[step];
                uint32_t rkey = allocs.at(aid).res_key;
                uint32_t dst_off = assign_out_channels(node, aid, node.cout);
                uint32_t src_off = tensor_channel_offsets.at(node.in_tensors[0])[0];
                K44ExecCmd cmd =
                    make_base_cmd(node, K44CmdType::RMSNORM_1D, rkey, rkey, src_off, dst_off);
                cmd.eps = node.eps;
                cmd.gamma_folded = node.gamma_folded;
                if (!node.gamma_folded) {
                    cmd.h_bias = node.bias_f16;
                }
                plan.cmds.push_back(std::move(cmd));
            } else if (node.op_type == "SwiGLU") {
                int aid = node_out_aid[step];
                uint32_t rkey = allocs.at(aid).res_key;
                uint32_t dst_off = assign_out_channels(node, aid, node.cout);
                uint32_t gate_off = tensor_channel_offsets.at(node.in_tensors[0])[0];
                uint32_t up_off = tensor_channel_offsets.at(node.in_tensors[1])[0];
                K44ExecCmd cmd =
                    make_base_cmd(node, K44CmdType::SWIGLU_MUL_1D, rkey, rkey, gate_off, dst_off);
                cmd.src2_off = up_off;
                plan.cmds.push_back(std::move(cmd));
            } else if (node.op_type == "RoPEAttention") {
                int aid = node_out_aid[step];
                uint32_t rkey = allocs.at(aid).res_key;
                uint32_t dst_off = assign_out_channels(node, aid, node.cout);
                uint32_t q_off = tensor_channel_offsets.at(node.in_tensors[0])[0];
                uint32_t k_off = tensor_channel_offsets.at(node.in_tensors[1])[0];
                uint32_t v_off = tensor_channel_offsets.at(node.in_tensors[2])[0];
                K44ExecCmd cmd =
                    make_base_cmd(node, K44CmdType::ROPE_GQA_ATTN_1D, rkey, rkey, q_off, dst_off);
                cmd.Cin = node.cout;
                cmd.h_q = node.h_q;
                cmd.h_kv = node.h_kv;
                cmd.head_dim = node.head_dim;
                cmd.rope_theta = node.rope_theta;
                cmd.llama3_rope_scaling = node.llama3_rope_scaling;
                cmd.src2_off = k_off;
                cmd.src3_off = v_off;
                if (opts.kv_cache_max_seq > 0u) {
                    const auto &kv_pair = node_kv_offs.at(step);
                    cmd.kv_k_off = kv_pair.first;
                    cmd.kv_v_off = kv_pair.second;
                    cmd.kv_max_seq = opts.kv_cache_max_seq;
                    cmd.attn_probs_off = attn_probs_off;
                }
                double sf_gflops =
                    2.0 * (double)node.cout * (double)node.w_out * (double)(node.w_out + 1u) / 1e9;
                cmd.gflops = (double)B_frames * sf_gflops;
                plan.single_frame_gflops += sf_gflops;
                plan.total_gflops += cmd.gflops;
                plan.cmds.push_back(std::move(cmd));
            } else if (node.op_type == "Deconvolution") {
                int aid = node_out_aid[step];
                uint32_t rkey_in = make_res_key(node.h_in, node.w_in, false);
                uint32_t rkey_out = allocs.at(aid).res_key;
                uint32_t dst_off = assign_out_channels(node, aid, node.cout);
                uint32_t quad_off = upquad_base_off.at(rkey_in);
                emit_deconv_waves(node, rkey_in, rkey_out, dst_off, quad_off,
                                  tensor_channel_offsets.at(node.in_tensors[0]), plan);
            } else if (node.op_type == "Convolution") {
                int aid = node_out_aid[step];
                uint32_t rkey_in = make_res_key(node.h_in, node.w_in, node.is_1d_tokens);
                uint32_t rkey_out = allocs.at(aid).res_key;
                uint32_t dst_off = assign_out_channels(node, aid, node.cout);
                emit_conv_waves(node, node_wave_cfg[step], rkey_in, rkey_out, dst_off, s2d_base_off,
                                tensor_channel_offsets, plan);
            }
        }
        if (tensor_channel_offsets.count(default_output_tensor)) {
            const auto &out_shape = shapes.at(default_output_tensor);
            plan.output_off = tensor_channel_offsets.at(default_output_tensor)[0];
            plan.logical_output_c = out_shape.c;
            plan.output_c = out_shape.c;
            plan.output_h = out_shape.h;
            plan.output_w = out_shape.w;
            plan.output_res_key = make_res_key(out_shape.h, out_shape.w, out_shape.is_1d);
        }
    }

  public:
    std::vector<K44IRNode> nodes;
    std::string default_output_tensor = "out0";
    K44CompileOptions opts;
    K44WavePlan compiled_plan;
    bool is_compiled = false;
    std::string cached_summary_json;

    void clear() {
        nodes.clear();
        default_output_tensor = "out0";
        is_compiled = false;
        cached_summary_json.clear();
    }

    // Appends a newly constructed `K44IRNode` (`node`) to the compiler's execution graph (`nodes`).
    // When `update_default_output` is `true` and `node.out_tensors` is non-empty, registers the
    // node's primary output tensor (`node.out_tensors[0]`) as `default_output_tensor` so the last
    // added layer automatically serves as the network output unless overridden by
    // `set_output(...)`.
    void append_node_to_graph(K44IRNode &&node, bool update_default_output = true) {
        if (update_default_output && !node.out_tensors.empty()) {
            default_output_tensor = node.out_tensors[0];
        }
        nodes.push_back(std::move(node));
    }

    // Interleaves compact row-major FP16 weights (`h_w_compact`) into RDNA4 Wave32 WMMA tile layout
    // (`cmd.h_w_wmma`) via `convert_raw_to_wmma`, and relativizes connectome gather indices
    // (`cmd.h_idx`) against a 64-element-aligned base offset (`cmd.x_base_off`) when the maximum
    // gather offset (`max_v`) reaches `0x70000000` FP16 elements (preventing 32-bit unsigned byte
    // offset overflow in RDNA4 MUBUF instructions).
    // Parameters:
    // - `cmd`: Target `K44ExecCmd` with padded output channels `Cout`, padded synaptic fan-in `B`,
    //   unpadded active synapses `B_raw`, and cluster size `cluster_size` (`S`).
    // - `h_w_compact`: Row-major `[Cout, B]` FP16 weight matrix (`B_vec8 = B / 8` `uint4_host` per
    // row).
    // - `h_idx`: `[Cout / S, B]` cluster gather offset table in FP16 elements.
    // - `h_bias`: `[Cout]` FP16 neuron bias vector.
    // - `plan`: `K44WavePlan` accumulating `total_w_bytes` and `total_idx_bytes`.
    // - `h_cout_map`: Optional `[Cout]` logical output channel remapping table.
    static void pack_wave_buffers(K44ExecCmd &cmd, const std::vector<uint16_t> &h_w_compact,
                                  const std::vector<uint32_t> &h_idx,
                                  const std::vector<uint16_t> &h_bias, K44WavePlan &plan,
                                  const std::vector<uint16_t> &h_cout_map = {}) {
        size_t w_bytes = (size_t)cmd.Cout * cmd.B * sizeof(uint16_t);
        size_t idx_bytes = h_idx.size() * sizeof(uint32_t);
        size_t b_bytes = h_bias.size() * sizeof(uint16_t);
        size_t cmap_bytes = h_cout_map.size() * sizeof(uint16_t);
        plan.total_w_bytes += w_bytes;
        plan.total_idx_bytes += (idx_bytes + b_bytes + cmap_bytes);
        if (!h_w_compact.empty()) {
            cmd.h_w_wmma.resize(w_bytes / sizeof(uint4_host));
            uint32_t B_vec8 = cmd.B / 8u;
            convert_raw_to_wmma(cmd.h_w_wmma.data(),
                                reinterpret_cast<const uint4_host *>(h_w_compact.data()), cmd.Cout,
                                B_vec8);
        }
        cmd.h_idx = h_idx;
        cmd.x_base_off = 0u;
        if (!cmd.h_idx.empty() && cmd.B > 0u && cmd.B_raw > 0u) {
            uint32_t num_clusters = (uint32_t)(cmd.h_idx.size() / cmd.B);
            uint32_t min_v = 0xffffffffu;
            uint32_t max_v = 0u;
            for (uint32_t cl = 0; cl < num_clusters; ++cl) {
                for (uint32_t syn = 0; syn < cmd.B_raw; ++syn) {
                    uint32_t val = cmd.h_idx[(size_t)cl * cmd.B + syn];
                    min_v = std::min(min_v, val);
                    max_v = std::max(max_v, val);
                }
            }
            if (max_v >= 0x70000000u && min_v != 0xffffffffu) {
                uint32_t x_base = min_v & ~63u;
                cmd.x_base_off = x_base;
                for (uint32_t cl = 0; cl < num_clusters; ++cl) {
                    for (uint32_t syn = 0; syn < cmd.B_raw; ++syn) {
                        cmd.h_idx[(size_t)cl * cmd.B + syn] -= x_base;
                    }
                    for (uint32_t syn = cmd.B_raw; syn < cmd.B; ++syn) {
                        cmd.h_idx[(size_t)cl * cmd.B + syn] = 0u;
                    }
                }
            }
        }
        cmd.h_bias = h_bias;
        cmd.h_cout_map = h_cout_map;
    }

    void add_input(const std::string &name, uint32_t ch, uint32_t height, uint32_t width,
                   bool is_1d = false) {
        K44IRNode node;
        node.op_type = "Input";
        node.name = name;
        node.out_tensors = {name};
        node.cin = ch;
        node.cout = ch;
        node.h_in = height;
        node.h_out = height;
        node.w_in = width;
        node.w_out = width;
        node.is_1d_tokens = is_1d || (height == 1u && (width < 4u || (width % 4u != 0u)));
        append_node_to_graph(std::move(node));
    }

    void add_flatten(const std::string &name, const std::string &in_tensor,
                     const std::string &out_tensor) {
        K44IRNode node;
        node.op_type = "Flatten";
        node.name = name;
        node.in_tensors = {in_tensor};
        node.out_tensors = {out_tensor};
        append_node_to_graph(std::move(node));
    }

    void add_rmsnorm(const std::string &name, const std::string &in_tensor,
                     const std::string &out_tensor, uint32_t dim, float eps, const float *gamma_f32,
                     bool pre_folded = false) {
        K44IRNode node;
        node.op_type = "RMSNorm";
        node.name = name;
        node.in_tensors = {in_tensor};
        node.out_tensors = {out_tensor};
        node.cin = dim;
        node.cout = dim;
        node.eps = eps;
        node.gamma_folded = pre_folded;
        node.bias_f32.resize(dim);
        node.bias_f16.resize(dim);
        for (uint32_t i = 0; i < dim; ++i) {
            float g_val = gamma_f32 ? gamma_f32[i] : 1.0f;
            node.bias_f32[i] = g_val;
            node.bias_f16[i] = float_to_half(g_val);
        }
        append_node_to_graph(std::move(node));
    }

    void add_conv2d(const std::string &name, const std::string &in_tensor,
                    const std::string &out_tensor, uint32_t cin, uint32_t cout, uint32_t kh,
                    uint32_t kw, uint32_t stride, uint32_t pad, K44Act act, const float *w_f32,
                    const float *b_f32, const std::string &residual_tensor = "") {
        K44IRNode node;
        node.op_type = "Convolution";
        node.name = name;
        node.in_tensors = {in_tensor};
        node.out_tensors = {out_tensor};
        node.cin = cin;
        node.cout = cout;
        node.kh = kh;
        node.kw = kw;
        node.stride = stride;
        node.pad = pad;
        node.act = act;
        node.residual_tensor = residual_tensor;
        size_t wsize = (size_t)cout * cin * kh * kw;
        node.weights_f32.resize(wsize);
        node.weights_f16.resize(wsize);
        if (w_f32) {
            std::memcpy(node.weights_f32.data(), w_f32, wsize * sizeof(float));
            for (size_t i = 0; i < wsize; ++i) {
                node.weights_f16[i] = float_to_half(w_f32[i]);
            }
        }
        init_node_bias(node, cout, b_f32);
        append_node_to_graph(std::move(node));
    }

    void add_conv2d_f16(const std::string &name, const std::string &in_tensor,
                        const std::string &out_tensor, uint32_t cin, uint32_t cout, uint32_t kh,
                        uint32_t kw, uint32_t stride, uint32_t pad, K44Act act,
                        const uint16_t *w_f16, const float *b_f32,
                        const std::string &residual_tensor = "") {
        K44IRNode node;
        node.op_type = "Convolution";
        node.name = name;
        node.in_tensors = {in_tensor};
        node.out_tensors = {out_tensor};
        node.cin = cin;
        node.cout = cout;
        node.kh = kh;
        node.kw = kw;
        node.stride = stride;
        node.pad = pad;
        node.act = act;
        node.residual_tensor = residual_tensor;
        size_t wsize = (size_t)cout * cin * kh * kw;
        node.weights_f16.resize(wsize, 0u);
        if (w_f16) {
            std::memcpy(node.weights_f16.data(), w_f16, wsize * sizeof(uint16_t));
        }
        init_node_bias(node, cout, b_f32);
        append_node_to_graph(std::move(node));
    }

    void add_linear_bf16(const std::string &name, const std::string &in_tensor,
                         const std::string &out_tensor, uint32_t din, uint32_t dout, K44Act act,
                         const uint16_t *w_bf16, const float *b_f32,
                         const std::string &residual_tensor = "") {
        K44IRNode node;
        node.op_type = "Convolution";
        node.name = name;
        node.in_tensors = {in_tensor};
        node.out_tensors = {out_tensor};
        node.cin = din;
        node.cout = dout;
        node.act = act;
        node.residual_tensor = residual_tensor;
        size_t wsize = (size_t)dout * din;
        node.weights_f16.resize(wsize, 0u);
        if (w_bf16) {
            for (size_t i = 0; i < wsize; ++i) {
                node.weights_f16[i] = float_to_half(bf16_to_float(w_bf16[i]));
            }
        }
        init_node_bias(node, dout, b_f32);
        append_node_to_graph(std::move(node));
    }

    void add_upconv2x2(const std::string &name, const std::string &in_tensor,
                       const std::string &out_tensor, uint32_t cin, uint32_t cout,
                       const float *w_f32, const float *b_f32) {
        K44IRNode node;
        node.op_type = "Deconvolution";
        node.name = name;
        node.in_tensors = {in_tensor};
        node.out_tensors = {out_tensor};
        node.cin = cin;
        node.cout = cout;
        node.kh = 2;
        node.kw = 2;
        node.stride = 2;
        size_t wsize = (size_t)cout * cin * 4u;
        node.weights_f32.resize(wsize, 0.0f);
        node.weights_f16.resize(wsize, 0u);
        if (w_f32) {
            std::memcpy(node.weights_f32.data(), w_f32, wsize * sizeof(float));
            for (size_t i = 0; i < wsize; ++i) {
                node.weights_f16[i] = float_to_half(w_f32[i]);
            }
        }
        init_node_bias(node, cout, b_f32);
        append_node_to_graph(std::move(node));
    }

    void add_maxpool(const std::string &name, const std::string &in_tensor,
                     const std::string &out_tensor, uint32_t kernel_sz, uint32_t stride,
                     uint32_t pad) {
        K44IRNode node;
        node.op_type = "Pooling";
        node.name = name;
        node.in_tensors = {in_tensor};
        node.out_tensors = {out_tensor};
        node.kh = kernel_sz;
        node.kw = kernel_sz;
        node.stride = stride;
        node.pad = pad;
        append_node_to_graph(std::move(node));
    }

    void add_upsample2x(const std::string &name, const std::string &in_tensor,
                        const std::string &out_tensor) {
        K44IRNode node;
        node.op_type = "Interp";
        node.name = name;
        node.in_tensors = {in_tensor};
        node.out_tensors = {out_tensor};
        append_node_to_graph(std::move(node));
    }

    void add_split(const std::string &name, const std::string &in_tensor,
                   const std::vector<std::string> &out_tensors) {
        K44IRNode node;
        node.op_type = "Split";
        node.name = name;
        node.in_tensors = {in_tensor};
        node.out_tensors = out_tensors;
        append_node_to_graph(std::move(node), false);
    }

    void add_slice(const std::string &name, const std::string &in_tensor,
                   const std::vector<std::string> &out_tensors,
                   const std::vector<uint32_t> &slice_sizes = {}) {
        K44IRNode node;
        node.op_type = "Slice";
        node.name = name;
        node.in_tensors = {in_tensor};
        node.out_tensors = out_tensors;
        node.slice_sizes = slice_sizes;
        append_node_to_graph(std::move(node), false);
    }

    void add_concat(const std::string &name, const std::vector<std::string> &in_tensors,
                    const std::string &out_tensor) {
        K44IRNode node;
        node.op_type = "Concat";
        node.name = name;
        node.in_tensors = in_tensors;
        node.out_tensors = {out_tensor};
        append_node_to_graph(std::move(node));
    }

    void add_swiglu_mul(const std::string &name, const std::string &gate_tensor,
                        const std::string &up_tensor, const std::string &out_tensor) {
        K44IRNode node;
        node.op_type = "SwiGLU";
        node.name = name;
        node.in_tensors = {gate_tensor, up_tensor};
        node.out_tensors = {out_tensor};
        append_node_to_graph(std::move(node));
    }

    void add_rope_gqa_attention(const std::string &name, const std::string &q_tensor,
                                const std::string &k_tensor, const std::string &v_tensor,
                                const std::string &out_tensor, uint32_t h_q, uint32_t h_kv,
                                uint32_t head_dim, float rope_theta = 10000.0f,
                                bool llama3_rope_scaling = false) {
        K44IRNode node;
        node.op_type = "RoPEAttention";
        node.name = name;
        node.in_tensors = {q_tensor, k_tensor, v_tensor};
        node.out_tensors = {out_tensor};
        node.h_q = h_q;
        node.h_kv = h_kv;
        node.head_dim = head_dim;
        node.rope_theta = rope_theta;
        node.llama3_rope_scaling = llama3_rope_scaling;
        node.cin = h_q * head_dim;
        node.cout = h_q * head_dim;
        append_node_to_graph(std::move(node));
    }

    const K44WavePlan &compile() {
        K44WavePlan plan;
        std::map<std::string, Shape3D> shapes;
        std::vector<K44WaveConfig> node_wave_cfg;
        std::map<int, PhysAlloc> allocs;
        std::vector<int> node_out_aid;
        std::map<uint32_t, uint32_t> s2d_base_off;
        std::map<uint32_t, uint32_t> upquad_base_off;
        std::map<size_t, std::pair<uint32_t, uint32_t>> node_kv_offs;
        uint32_t attn_probs_off = 0u;

        pass1_infer_shapes(plan, shapes);
        pass2_fold_rmsnorm(plan);
        pass3_build_layouts(plan, shapes, node_wave_cfg);
        pass4_allocate_canvas(plan, node_wave_cfg, allocs, node_out_aid, s2d_base_off,
                              upquad_base_off, node_kv_offs, attn_probs_off);
        pass5_emit_commands(plan, shapes, node_wave_cfg, allocs, node_out_aid, s2d_base_off,
                            upquad_base_off, node_kv_offs, attn_probs_off);

        compiled_plan = std::move(plan);
        is_compiled = true;
        build_summary_json();
        return compiled_plan;
    }

    void build_summary_json() {
        const auto &plan = compiled_plan;
        char hdr[1536];
        std::snprintf(
            hdr, sizeof(hdr),
            "{\n"
            "  \"batch_size\": %u,\n"
            "  \"num_waves\": %d,\n"
            "  \"num_cmds\": %zu,\n"
            "  \"canvas_elems\": %zu,\n"
            "  \"canvas_bytes\": %llu,\n"
            "  \"kv_cache_elems\": %zu,\n"
            "  \"total_w_bytes\": %zu,\n"
            "  \"total_idx_bytes\": %zu,\n"
            "  \"single_frame_gflops\": %.6g,\n"
            "  \"total_gflops\": %.6g,\n"
            "  \"single_frame_gsyn\": %.6g,\n"
            "  \"total_gsyn\": %.6g,\n"
            "  \"num_folded_rmsnorms\": %d,\n"
            "  \"num_eliminated_concats\": %d,\n"
            "  \"num_eliminated_splits\": %d,\n"
            "  \"num_zero_copy_flattens\": %d,\n"
            "  \"input_shape\": [%u, %u, %u],\n"
            "  \"output_shape\": [%u, %u, %u],\n"
            "  \"max_B_per_wave\": %u,\n"
            "  \"target_profile\": \"%s\",\n"
            "  \"target_num_cus\": %u,\n"
            "  \"persistent_neurons\": %s,\n"
            "  \"reuse_canvas\": %s,\n"
            "  \"cmds\": [\n",
            plan.batch_size, plan.num_wave_dispatches, plan.cmds.size(), plan.canvas_elems,
            (unsigned long long)(plan.canvas_elems * 2ULL), plan.kv_cache_elems, plan.total_w_bytes,
            plan.total_idx_bytes, plan.single_frame_gflops, plan.total_gflops,
            plan.single_frame_gflops * 0.5, plan.total_gflops * 0.5, plan.num_folded_rmsnorms,
            plan.num_eliminated_concats, plan.num_eliminated_splits, plan.num_zero_copy_flattens,
            plan.input_c, plan.input_h, plan.input_w, plan.logical_output_c, plan.output_h,
            plan.output_w, opts.max_B_per_wave, opts.hw_profile.name.c_str(),
            opts.hw_profile.num_cus, (!opts.reuse_canvas ? "true" : "false"),
            (opts.reuse_canvas ? "true" : "false"));

        cached_summary_json.clear();
        cached_summary_json.reserve(2048 + plan.cmds.size() * 256);
        cached_summary_json.append(hdr);
        char line[512];
        for (size_t i = 0; i < plan.cmds.size(); ++i) {
            const auto &cmd = plan.cmds[i];
            std::snprintf(
                line, sizeof(line),
                "    {\"idx\": %zu, \"name\": \"%s\", \"type\": \"%s\", \"Cin\": %u, \"Cout\": %u, "
                "\"logical_Cout\": %u, \"B\": %u, \"B_raw\": %u, \"S\": %u, \"act\": \"%s\", "
                "\"has_residual\": %s, \"has_self_read\": %s, \"subwave\": [%u, %u], \"h_out\": "
                "%u, \"w_out\": %u, \"src_off\": %u, \"y_off\": %u, \"gflops\": %.6g}%s\n",
                i, cmd.name.c_str(), cmd_type_str(cmd.type), cmd.Cin, cmd.Cout, cmd.logical_Cout,
                cmd.B, cmd.B_raw, cmd.cluster_size, act_str(cmd.act),
                (cmd.has_residual ? "true" : "false"), (cmd.has_self_read ? "true" : "false"),
                cmd.subwave_idx, cmd.num_subwaves, cmd.h_out, cmd.w_out, cmd.src_off, cmd.y_off,
                cmd.gflops, (i + 1 < plan.cmds.size() ? "," : ""));
            cached_summary_json.append(line);
        }
        cached_summary_json.append("  ]\n}\n");
    }
};

} // namespace k44_host
