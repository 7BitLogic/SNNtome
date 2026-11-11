#!/usr/bin/env python3
"""
example_llm.py — LLM / Transformer Compilation, TCFF v1.0 Serialization & GPU Verification

Usage:
  1. Self-contained Transformer Decoder Block verification (RMSNorm + RoPE + GQA + SwiGLU):
       python3 wavec/examples/example_llm.py
  2. Load a full HuggingFace SafeTensors LLM directory (e.g. SmolLM2-135M / Llama-3.2-1B):
       python3 wavec/examples/example_llm.py --safetensors-dir /path/to/hf_model_dir --seq-len 64
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavec import WaveGraph


def ref_rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=0, keepdims=True) + eps)
    return (x / rms) * gamma[:, None]


def ref_rope_inplace(qk: np.ndarray, h_count: int, head_dim: int, theta: float = 10000.0) -> np.ndarray:
    out = qk.copy()
    seq_len = qk.shape[1]
    half_d = head_dim // 2
    p = np.arange(half_d, dtype=np.float32)
    inv_freq = np.exp(2.0 * p * (-np.log(theta) / float(head_dim)))
    pos = np.arange(seq_len, dtype=np.float32)
    angles = pos[None, :] * inv_freq[:, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    for h in range(h_count):
        base = h * head_dim
        x0 = out[base : base + half_d, :].copy()
        x1 = out[base + half_d : base + head_dim, :].copy()
        out[base : base + half_d, :] = x0 * cos_a - x1 * sin_a
        out[base + half_d : base + head_dim, :] = x0 * sin_a + x1 * cos_a
    return out


def ref_gqa_causal_attn(q: np.ndarray, k: np.ndarray, v: np.ndarray, h_q: int, h_kv: int, head_dim: int) -> np.ndarray:
    seq_len = q.shape[1]
    gqa_group = h_q // h_kv
    scale = 1.0 / np.sqrt(float(head_dim))
    out = np.zeros_like(q)
    mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
    for h in range(h_q):
        kv_h = h // gqa_group
        qh = q[h * head_dim : (h + 1) * head_dim, :].T
        kh = k[kv_h * head_dim : (kv_h + 1) * head_dim, :].T
        vh = v[kv_h * head_dim : (kv_h + 1) * head_dim, :].T
        scores = (qh @ kh.T) * scale + mask
        scores -= np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        out[h * head_dim : (h + 1) * head_dim, :] = (probs @ vh).T
    return out


def run_synthetic_transformer_test() -> None:
    print("======================================================================")
    print(" [1/2] Transformer Decoder Layer Verification (Llama-3 / SmolLM2 GQA)")
    print("       Testing Pass-2 Gamma Folding, k44 QKV/O/MLP & k44_valu RoPE+GQA")
    print("======================================================================")
    rng = np.random.default_rng(7)
    D = 256
    H_Q, H_KV, HEAD_DIM = 4, 2, 64
    D_KV = H_KV * HEAD_DIM
    D_FFN = 512
    SEQ_LEN = 64

    x_in = (rng.standard_normal((D, 1, SEQ_LEN)) * 0.25).astype(np.float16).astype(np.float32)
    gamma1 = (1.0 + rng.standard_normal((D,)) * 0.1).astype(np.float16).astype(np.float32)
    gamma2 = (1.0 + rng.standard_normal((D,)) * 0.1).astype(np.float16).astype(np.float32)

    wq = (rng.standard_normal((D, D)) * 0.08).astype(np.float16).astype(np.float32)
    wk = (rng.standard_normal((D_KV, D)) * 0.08).astype(np.float16).astype(np.float32)
    wv = (rng.standard_normal((D_KV, D)) * 0.08).astype(np.float16).astype(np.float32)
    wo = (rng.standard_normal((D, D)) * 0.08).astype(np.float16).astype(np.float32)

    w_gate = (rng.standard_normal((D_FFN, D)) * 0.08).astype(np.float16).astype(np.float32)
    w_up   = (rng.standard_normal((D_FFN, D)) * 0.08).astype(np.float16).astype(np.float32)
    w_down = (rng.standard_normal((D, D_FFN)) * 0.08).astype(np.float16).astype(np.float32)

    g = WaveGraph(max_B_per_wave=512, default_S=64, fold_rmsnorm_gamma=True, reuse_canvas=True)
    g.add_input("tokens_in", c=D, h=1, w=SEQ_LEN, is_1d=True)

    g.add_rmsnorm("attn_norm", "tokens_in", "x_norm1", D, gamma1, eps=1e-5)
    g.add_linear("q_proj", "x_norm1", "q", D, D, wq, act="none")
    g.add_linear("k_proj", "x_norm1", "k", D, D_KV, wk, act="none")
    g.add_linear("v_proj", "x_norm1", "v", D, D_KV, wv, act="none")
    g.add_rope_gqa_attn("rope_gqa", "q", "k", "v", "attn_out", H_Q, H_KV, HEAD_DIM, rope_theta=10000.0)
    g.add_linear("o_proj", "attn_out", "h_mid", D, D, wo, act="none", res_blob="tokens_in")

    g.add_rmsnorm("ffn_norm", "h_mid", "x_norm2", D, gamma2, eps=1e-5)
    g.add_linear("gate_proj", "x_norm2", "gate", D, D_FFN, w_gate, act="silu")
    g.add_linear("up_proj", "x_norm2", "up", D, D_FFN, w_up, act="none")
    g.add_swiglu("swiglu", "gate", "up", "ffn_act", D_FFN)
    g.add_linear("down_proj", "ffn_act", "tokens_out", D_FFN, D, w_down, act="none", res_blob="h_mid")
    g.set_output("tokens_out")

    plan = g.compile()
    print(f"  Compiled Waves       : {plan.num_wave_dispatches}")
    print(f"  Folded RMSNorm Gammas: {plan.summary.get('num_folded_rmsnorms', 0)} (Pass-2 Zero-Overhead)")
    print(f"  d_Canvas Substrate   : {plan.canvas_bytes / 1024.0:.1f} KB")

    with tempfile.TemporaryDirectory() as tmpdir:
        tcff_path = os.path.join(tmpdir, "llm_decoder_block.tcff")
        num_ifds = plan.save_tcff(tcff_path, model_prefix="LLM_Block")
        tcff_kb = os.path.getsize(tcff_path) / 1024.0
        plan.load_tcff_weights(tcff_path)
        print(f"  TCFF v1.0 Roundtrip  : {num_ifds} IFDs ({tcff_kb:.1f} KB) verified")

    y_gpu, gpu_ms = plan.infer_f32(x_in)

    x2d = x_in[:, 0, :]
    xn1 = ref_rmsnorm(x2d, gamma1, eps=1e-5)
    q = ref_rope_inplace(wq @ xn1, H_Q, HEAD_DIM, 10000.0)
    k = ref_rope_inplace(wk @ xn1, H_KV, HEAD_DIM, 10000.0)
    v = wv @ xn1
    attn = ref_gqa_causal_attn(q, k, v, H_Q, H_KV, HEAD_DIM)
    h_mid = (wo @ attn) + x2d
    xn2 = ref_rmsnorm(h_mid, gamma2, eps=1e-5)
    gate = w_gate @ xn2
    gate_silu = gate / (1.0 + np.exp(-gate))
    up = w_up @ xn2
    y_ref = ((w_down @ (gate_silu * up)) + h_mid)[:, None, :]

    max_err = float(np.max(np.abs(y_gpu - y_ref)))
    cos_sim = float(np.dot(y_gpu.ravel(), y_ref.ravel()) / (np.linalg.norm(y_gpu) * np.linalg.norm(y_ref)))
    print(f"  GPU Execution Time   : {gpu_ms:.3f} ms")
    print(f"  Max Abs Error vs FP32: {max_err:.4e} (Cosine Similarity: {cos_sim:.7f})")
    assert max_err < 2e-2 and cos_sim > 0.9999, f"Accuracy check failed: max_err={max_err}"
    print("  [PASS] Transformer GQA + SwiGLU block verified on GPU!")


def run_external_safetensors_llm(model_dir: str, seq_len: int) -> None:
    print("======================================================================")
    print(f" [2/2] Loading HuggingFace SafeTensors LLM: {model_dir} (seq_len={seq_len})")
    print("======================================================================")
    plan = WaveGraph.from_safetensors_llm(model_dir, seq_len=seq_len, reuse_canvas=True)
    in_c, in_h, in_w = plan.summary["input_shape"]
    print(f"  Compiled Waves     : {plan.num_wave_dispatches}")
    print(f"  Total GFLOPs       : {plan.total_gflops:.2f} GFLOPs")
    print(f"  d_Canvas Substrate : {plan.canvas_bytes / (1024*1024):.2f} MB")

    x = (np.random.default_rng(0).standard_normal((in_c, in_h, in_w)) * 0.05).astype(np.float32)
    y, gpu_ms = plan.infer_f32(x)
    tflops = (plan.total_gflops * 1e-3) / (gpu_ms * 1e-3) if gpu_ms > 0 else 0.0
    print(f"  GPU Inference Time : {gpu_ms:.3f} ms ({tflops:.2f} TFLOPS / {tflops/2.0:.2f} TSyn/s)")
    print(f"  Output Tensor Shape: {y.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM WaveCompiler Verification & SafeTensors Loader")
    parser.add_argument("--safetensors-dir", type=str, default=None, help="Path to HF SafeTensors model directory")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length (multiple of 64 recommended)")
    args = parser.parse_args()

    run_synthetic_transformer_test()
    if args.safetensors_dir:
        run_external_safetensors_llm(args.safetensors_dir, args.seq_len)
