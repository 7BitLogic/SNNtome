#!/usr/bin/env python3
"""
example_yolov8.py — YOLOv8 Architecture Compilation, TCFF v1.0 Serialization & GPU Verification

Usage:
  1. Self-contained architectural & numerical verification (zero external files required):
       python3 wavec/examples/example_yolov8.py
  2. Load your own exported YOLOv8 NCNN model (.param + .bin) and run GPU inference:
       python3 wavec/examples/example_yolov8.py --ncnn-param /path/to/yolov8n.param --ncnn-bin /path/to/yolov8n.bin
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavec import WaveGraph


def ref_silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def ref_conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, pad: int = 0, act: str = "silu") -> np.ndarray:
    cin, h, w_in = x.shape
    cout, _, kh, kw = w.shape
    x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="constant") if pad > 0 else x
    h_out = (h + 2 * pad - kh) // stride + 1
    w_out = (w_in + 2 * pad - kw) // stride + 1
    out = np.zeros((cout, h_out, w_out), dtype=np.float32)
    for dy in range(kh):
        for dx in range(kw):
            patch = x_pad[:, dy : dy + h_out * stride : stride, dx : dx + w_out * stride : stride]
            out += np.einsum("oi,iyx->oyx", w[:, :, dy, dx], patch)
    out += b[:, None, None]
    return ref_silu(out) if act == "silu" else out


def ref_maxpool5x5_s1_p2(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=-65504.0)
    out = np.full_like(x, -65504.0)
    for dy in range(5):
        for dx in range(5):
            out = np.maximum(out, xp[:, dy : dy + h, dx : dx + w])
    return out


def run_synthetic_yolov8_block_test() -> None:
    print("======================================================================")
    print(" [1/2] YOLOv8 Building-Block Verification (Stem + C2f + SPPF)")
    print("       Testing Zero-Copy Split/Concat, 3x3 k44_shift & 1x1 k44 Waves")
    print("======================================================================")
    rng = np.random.default_rng(42)
    H, W = 64, 64
    x_in = (rng.standard_normal((16, H, W)) * 0.25).astype(np.float16).astype(np.float32)

    # 1. Stem 3x3 Conv (16 -> 32, k=3, s=1, p=1, SiLU) -> exercises k44_shift
    w_stem = (rng.standard_normal((32, 16, 3, 3)) * 0.1).astype(np.float16).astype(np.float32)
    b_stem = (rng.standard_normal((32,)) * 0.05).astype(np.float16).astype(np.float32)

    # 2. C2f Block:
    #    cv1 (1x1, 32 -> 32) -> Split into [c2f_a (16), c2f_b (16)]
    #    Bottleneck on c2f_b: bn1 (3x3, 16->16) -> bn2 (3x3, 16->16 + residual c2f_b)
    #    Concat([c2f_a, c2f_b, bn_out]) (48 ch) -> cv2 (1x1, 48 -> 32)
    w_cv1 = (rng.standard_normal((32, 32, 1, 1)) * 0.1).astype(np.float16).astype(np.float32)
    b_cv1 = (rng.standard_normal((32,)) * 0.05).astype(np.float16).astype(np.float32)
    w_bn1 = (rng.standard_normal((16, 16, 3, 3)) * 0.1).astype(np.float16).astype(np.float32)
    b_bn1 = (rng.standard_normal((16,)) * 0.05).astype(np.float16).astype(np.float32)
    w_bn2 = (rng.standard_normal((16, 16, 3, 3)) * 0.1).astype(np.float16).astype(np.float32)
    b_bn2 = (rng.standard_normal((16,)) * 0.05).astype(np.float16).astype(np.float32)
    w_cv2 = (rng.standard_normal((32, 48, 1, 1)) * 0.1).astype(np.float16).astype(np.float32)
    b_cv2 = (rng.standard_normal((32,)) * 0.05).astype(np.float16).astype(np.float32)

    # 3. SPPF Block:
    #    sppf_cv1 (1x1, 32 -> 16) -> m1=MaxPool5x5 -> m2=MaxPool5x5 -> m3=MaxPool5x5
    #    Concat([sppf_cv1, m1, m2, m3]) (64 ch) -> sppf_cv2 (1x1, 64 -> 32)
    w_sp1 = (rng.standard_normal((16, 32, 1, 1)) * 0.1).astype(np.float16).astype(np.float32)
    b_sp1 = (rng.standard_normal((16,)) * 0.05).astype(np.float16).astype(np.float32)
    w_sp2 = (rng.standard_normal((32, 64, 1, 1)) * 0.1).astype(np.float16).astype(np.float32)
    b_sp2 = (rng.standard_normal((32,)) * 0.05).astype(np.float16).astype(np.float32)

    g = WaveGraph(max_B_per_wave=288, reuse_canvas=True)
    g.add_input("images", c=16, h=H, w=W, is_1d=False)
    g.add_conv2d("stem", "images", "stem_out", 16, 32, 3, 3, w_stem, b_stem, stride=1, pad=1, act="silu")
    g.add_conv2d("c2f_cv1", "stem_out", "c2f_cv1_out", 32, 32, 1, 1, w_cv1, b_cv1, stride=1, pad=0, act="silu")
    g.add_slice("c2f_split", "c2f_cv1_out", ["c2f_a", "c2f_b"], [16, 16])
    g.add_conv2d("c2f_bn1", "c2f_b", "c2f_bn1_out", 16, 16, 3, 3, w_bn1, b_bn1, stride=1, pad=1, act="silu")
    g.add_conv2d("c2f_bn2", "c2f_bn1_out", "c2f_bn2_out", 16, 16, 3, 3, w_bn2, b_bn2, stride=1, pad=1, act="none", res_blob="c2f_b")
    g.add_concat("c2f_cat", ["c2f_a", "c2f_b", "c2f_bn2_out"], "c2f_cat_out")
    g.add_conv2d("c2f_cv2", "c2f_cat_out", "c2f_out", 48, 32, 1, 1, w_cv2, b_cv2, stride=1, pad=0, act="silu")

    g.add_conv2d("sppf_cv1", "c2f_out", "sp1", 32, 16, 1, 1, w_sp1, b_sp1, stride=1, pad=0, act="silu")
    g.add_maxpool("sppf_m1", "sp1", "m1", kernel_size=5, stride=1, pad=2)
    g.add_maxpool("sppf_m2", "m1", "m2", kernel_size=5, stride=1, pad=2)
    g.add_maxpool("sppf_m3", "m2", "m3", kernel_size=5, stride=1, pad=2)
    g.add_concat("sppf_cat", ["sp1", "m1", "m2", "m3"], "sp_cat")
    g.add_conv2d("sppf_cv2", "sp_cat", "out", 64, 32, 1, 1, w_sp2, b_sp2, stride=1, pad=0, act="silu")
    g.set_output("out")

    plan = g.compile()
    print(f"  Compiled Waves       : {plan.num_wave_dispatches}")
    print(f"  Eliminated Splits    : {plan.summary.get('num_eliminated_splits', 0)} (Zero-Copy)")
    print(f"  Eliminated Concats   : {plan.summary.get('num_eliminated_concats', 0)} (Zero-Copy)")
    print(f"  d_Canvas Substrate   : {plan.canvas_bytes / 1024.0:.1f} KB")

    with tempfile.TemporaryDirectory() as tmpdir:
        tcff_path = os.path.join(tmpdir, "yolov8_subgraph.tcff")
        num_ifds = plan.save_tcff(tcff_path, model_prefix="YOLOv8_Subgraph")
        tcff_kb = os.path.getsize(tcff_path) / 1024.0
        plan.load_tcff_weights(tcff_path)
        print(f"  TCFF v1.0 Roundtrip  : {num_ifds} IFDs ({tcff_kb:.1f} KB) verified")

    y_gpu, gpu_ms = plan.infer_f32(x_in)

    # Pure NumPy FP32 Reference
    s_out = ref_conv2d(x_in, w_stem, b_stem, stride=1, pad=1, act="silu")
    cv1_out = ref_conv2d(s_out, w_cv1, b_cv1, stride=1, pad=0, act="silu")
    ca, cb = cv1_out[:16], cv1_out[16:]
    bn1 = ref_conv2d(cb, w_bn1, b_bn1, stride=1, pad=1, act="silu")
    bn2 = ref_conv2d(bn1, w_bn2, b_bn2, stride=1, pad=1, act="none") + cb
    c2f_out = ref_conv2d(np.concatenate([ca, cb, bn2], axis=0), w_cv2, b_cv2, stride=1, pad=0, act="silu")
    sp1 = ref_conv2d(c2f_out, w_sp1, b_sp1, stride=1, pad=0, act="silu")
    m1 = ref_maxpool5x5_s1_p2(sp1)
    m2 = ref_maxpool5x5_s1_p2(m1)
    m3 = ref_maxpool5x5_s1_p2(m2)
    y_ref = ref_conv2d(np.concatenate([sp1, m1, m2, m3], axis=0), w_sp2, b_sp2, stride=1, pad=0, act="silu")

    max_err = float(np.max(np.abs(y_gpu - y_ref)))
    cos_sim = float(np.dot(y_gpu.ravel(), y_ref.ravel()) / (np.linalg.norm(y_gpu) * np.linalg.norm(y_ref)))
    print(f"  GPU Execution Time   : {gpu_ms:.3f} ms")
    print(f"  Max Abs Error vs FP32: {max_err:.4e} (Cosine Similarity: {cos_sim:.7f})")
    assert max_err < 1e-2 and cos_sim > 0.9999, f"Accuracy check failed: max_err={max_err}"
    print("  [PASS] YOLOv8 building blocks verified on GPU!")


def run_external_ncnn_yolov8(param_path: str, bin_path: str) -> None:
    print("======================================================================")
    print(f" [2/2] Loading External YOLOv8 NCNN Model: {param_path}")
    print("======================================================================")
    g = WaveGraph(max_B_per_wave=288, reuse_canvas=True)
    g.load_ncnn(param_path, bin_path, in_c=3, in_h=640, in_w=640)
    plan = g.compile()
    print(f"  Compiled Waves     : {plan.num_wave_dispatches}")
    print(f"  Single-Frame GFLOPs: {plan.total_gflops:.2f} GFLOPs")
    print(f"  d_Canvas Substrate : {plan.canvas_bytes / (1024*1024):.2f} MB")

    dummy_rgb = np.full((640, 640, 3), 114, dtype=np.uint8)
    boxes, gpu_ms = plan.infer_rgb8_letterbox(dummy_rgb, conf_thresh=0.25, iou_thresh=0.45)
    tflops = (plan.total_gflops * 1e-3) / (gpu_ms * 1e-3) if gpu_ms > 0 else 0.0
    print(f"  GPU Inference Time : {gpu_ms:.3f} ms ({1000.0 / gpu_ms:.1f} FPS, {tflops:.2f} TFLOPS)")
    print(f"  Detections on blank: {len(boxes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 WaveCompiler Verification & Loader")
    parser.add_argument("--ncnn-param", type=str, default=None, help="Path to exported YOLOv8 .ncnn.param")
    parser.add_argument("--ncnn-bin", type=str, default=None, help="Path to exported YOLOv8 .ncnn.bin")
    args = parser.parse_args()

    run_synthetic_yolov8_block_test()
    if args.ncnn_param and args.ncnn_bin:
        run_external_ncnn_yolov8(args.ncnn_param, args.ncnn_bin)
