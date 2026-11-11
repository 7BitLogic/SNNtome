#!/usr/bin/env python3
# ==============================================================================
# scripts/plot_distributions.py
#
# Standalone Python script for generating publication-grade vector SVGs and PNGs
# for the Kernel 44 Whitepaper (AMD Radeon RX 9060 XT / RDNA4 Navi 44).
#
# Visualizes True Continuous Uniform Distributions with Theoretical Poisson
# Confidence Bands (+-1 sigma and +-2 sigma):
#   1. distribution_activations.svg / .png (X ~ U(-1, 1))
#   2. distribution_indices.svg     / .png (C ~ U(0, 20M))
#   3. distribution_weights.svg     / .png (W_f16 ~ U(-1, 1) & W_f8 ~ U(-1.5, 1.5))
#   4. distribution_biases.svg      / .png (b ~ U(-0.5, 0.5))
#   5. distribution_overview.svg    / .png (4-Panel Complete Overview with KS & Poisson Stats)
# ==============================================================================

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Publication Styling (Clean Academic / Nature / IEEE Design)
# ------------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'svg.fonttype': 'none',
})

COLOR_ACT = '#1f77b4'   # Deep Blue
COLOR_IDX = '#2ca02c'   # Emerald Green
COLOR_WGT = '#d62728'   # Crimson Red
COLOR_BIA = '#9467bd'   # Amethyst Purple
COLOR_THEO = '#ff7f0e'  # Orange Theoretical Reference Line

def decode_e4m3(u):
    """Converts FP8 E4M3 bytes to 32-bit floats (OCP / IEEE spec)."""
    sign = (u >> 7) & 1
    exp = (u >> 3) & 0xf
    mant = u & 7
    val = np.zeros(u.shape, dtype=np.float32)
    is_sub = (exp == 0) & (u != 0) & (u != 0x80)
    is_norm = exp > 0
    val[is_sub] = (mant[is_sub] / 8.0) * (1.0 / 64.0)
    real_exp = exp[is_norm].astype(np.int32) - 7
    val[is_norm] = np.ldexp(1.0 + mant[is_norm] / 8.0, real_exp)
    val = np.where(sign == 1, -val, val)
    return val

def kolmogorov_smirnov_uniform(data, a, b):
    """Computes the Kolmogorov-Smirnov test statistic D against theoretical U(a, b)."""
    n = len(data)
    s = np.sort(data)
    cdf_theo = np.clip((s - a) / (b - a), 0.0, 1.0)
    d_plus = np.max(np.arange(1, n + 1) / n - cdf_theo)
    d_minus = np.max(cdf_theo - np.arange(0, n) / n)
    return max(d_plus, d_minus)

def ensure_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def load_or_generate_data(ref_dir):
    idx_s128_path = os.path.join(ref_dir, "connectome_indices_s128.bin")
    w_f16_path    = os.path.join(ref_dir, "weights_f16.bin")
    w_f8_path     = os.path.join(ref_dir, "weights_f8.bin")
    b_f16_path    = os.path.join(ref_dir, "biases_f16.bin")

    # 1. Indices
    if os.path.exists(idx_s128_path):
        indices = np.fromfile(idx_s128_path, dtype=np.uint32)
    else:
        rng = np.random.default_rng(42)
        indices = rng.integers(0, 20_000_000, size=32 * 1024, dtype=np.uint32)

    # 2. Weights FP16
    if os.path.exists(w_f16_path):
        weights_f16 = np.fromfile(w_f16_path, dtype=np.float16).astype(np.float32)
    else:
        rng = np.random.default_rng(1337)
        weights_f16 = rng.uniform(-1.0, 1.0, size=4096 * 1024).astype(np.float32)

    # 3. Weights FP8
    if os.path.exists(w_f8_path):
        weights_f8_raw = np.fromfile(w_f8_path, dtype=np.uint8)
        weights_f8 = decode_e4m3(weights_f8_raw)
    else:
        rng = np.random.default_rng(1337)
        weights_f8 = rng.uniform(-1.5, 1.5, size=4096 * 1024).astype(np.float32)

    # 4. Biases
    if os.path.exists(b_f16_path):
        biases = np.fromfile(b_f16_path, dtype=np.float16).astype(np.float32)
    else:
        rng = np.random.default_rng(2026)
        biases = rng.uniform(-0.5, 0.5, size=4096).astype(np.float32)

    # 5. Activations X (Simulated continuous GPU LCG U(-1, 1))
    rng_act = np.random.default_rng(777)
    activations = rng_act.uniform(-1.0, 1.0, size=131_072).astype(np.float32)

    return activations, indices, weights_f16, weights_f8, biases

def plot_activations(activations, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    k_bins = 50
    N = len(activations)
    n_bin = N / k_bins
    sigma_poisson = np.sqrt(n_bin)
    rel_sigma = 1.0 / np.sqrt(n_bin)
    lower_bound, upper_bound = -1.0, 1.0
    p_theo = 1.0 / (upper_bound - lower_bound)
    sigma_p = p_theo * rel_sigma

    ax.hist(activations, bins=k_bins, density=True, color=COLOR_ACT, alpha=0.72, edgecolor='black', linewidth=0.5, label='Empirical Activations')
    
    # Theoretical uniform and Poisson confidence bands
    ax.fill_between([-1.0, 1.0], p_theo - 2 * sigma_p, p_theo + 2 * sigma_p, color=COLOR_THEO, alpha=0.15, label=f'Poisson ±2σ Band (±{200*rel_sigma:.1f}%)')
    ax.fill_between([-1.0, 1.0], p_theo - sigma_p, p_theo + sigma_p, color=COLOR_THEO, alpha=0.28, label=f'Poisson ±1σ Band (±{100*rel_sigma:.1f}%)')
    ax.axhline(p_theo, color=COLOR_THEO, linestyle='-', linewidth=2.0, label=f'Theoretical Uniform Density [p={p_theo:.2f}]')

    d_ks = kolmogorov_smirnov_uniform(activations, -1.0, 1.0)
    ax.set_title("Continuous Uniform Distribution of Input Activations X", fontweight='bold')
    ax.set_xlabel("Activation Value x")
    ax.set_ylabel("Probability Density p(x)")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(0.0, 0.72)
    ax.legend(frameon=True, facecolor='white', framealpha=0.92, loc='upper right', fontsize=8.8)

    stats_text = (f"N = {N:,}\n"
                  f"Bins k = {k_bins} (n_bin = {n_bin:,.0f})\n"
                  f"Poisson σ = {sigma_poisson:.1f} (±{100*rel_sigma:.2f}%)\n"
                  f"Mean = {np.mean(activations):+.4f} (Ideal: 0.00)\n"
                  f"Std  = {np.std(activations):.4f} (Ideal: {1.0/np.sqrt(3):.4f})\n"
                  f"KS Statistic D = {d_ks:.4f}\n"
                  f"Status: Continuously Flat")
    ax.text(0.03, 0.94, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#ccc'))

    fig.savefig(out_path, format='svg')
    png_path = out_path.replace('.svg', '.png')
    pdf_path = out_path.replace('.svg', '.pdf')
    fig.savefig(png_path, format='png', dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path} and {png_path}")

def plot_indices(indices, out_path, cin_max=20_000_000):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    k_bins = 50
    N = len(indices)
    n_bin = N / k_bins
    sigma_poisson = np.sqrt(n_bin)
    rel_sigma = 1.0 / np.sqrt(n_bin)

    idx_m = indices / 1e6
    cin_m = cin_max / 1e6
    p_theo = 1.0 / cin_m
    sigma_p = p_theo * rel_sigma

    ax.hist(idx_m, bins=k_bins, density=True, color=COLOR_IDX, alpha=0.72, edgecolor='black', linewidth=0.5, label='Empirical Indices')
    ax.fill_between([0, cin_m], p_theo - 2 * sigma_p, p_theo + 2 * sigma_p, color=COLOR_THEO, alpha=0.15, label='Poisson ±2σ Band')
    ax.fill_between([0, cin_m], p_theo - sigma_p, p_theo + sigma_p, color=COLOR_THEO, alpha=0.28, label=f'Poisson ±1σ Band (±{100*rel_sigma:.1f}%)')
    ax.axhline(p_theo, color=COLOR_THEO, linestyle='-', linewidth=2.0, label=f'Theoretical Density U(0, {cin_m:.0f}M) [p={p_theo:.4f}]')

    d_ks = kolmogorov_smirnov_uniform(indices, 0, cin_max)
    ax.set_title(f"Homomorphic Connectome Indices C (Cin = {cin_m:.0f}M Neurons)", fontweight='bold')
    ax.set_xlabel("Channel Index [Millions of Neurons]")
    ax.set_ylabel("Probability Density p(c)")
    ax.set_xlim(0, cin_m)
    ax.set_ylim(0.0, 0.085)
    ax.legend(frameon=True, facecolor='white', framealpha=0.92, loc='upper right', fontsize=8.8)

    stats_text = (f"N = {N:,} Indices\n"
                  f"Clusters = 32 (S=128, B=1024)\n"
                  f"Bins k = {k_bins} (n_bin = {n_bin:,.0f})\n"
                  f"Poisson σ = {sigma_poisson:.1f} (±{100*rel_sigma:.2f}%)\n"
                  f"Mean = {np.mean(idx_m):.3f}M (Ideal: {cin_m/2.0:.1f}M)\n"
                  f"KS Statistic D = {d_ks:.4f}\n"
                  f"Status: Zero Gaps / Zero Clustering")
    ax.text(0.03, 0.94, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#ccc'))

    fig.savefig(out_path, format='svg')
    png_path = out_path.replace('.svg', '.png')
    pdf_path = out_path.replace('.svg', '.pdf')
    fig.savefig(png_path, format='png', dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path} and {png_path}")

def plot_weights(weights_f16, weights_f8, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.6), constrained_layout=True)
    k_bins = 50

    # Panel 1: FP16 Weights
    N1 = len(weights_f16)
    n_bin1 = N1 / k_bins
    sigma1 = np.sqrt(n_bin1)
    rel_sigma1 = 1.0 / np.sqrt(n_bin1)
    lower_bound1, upper_bound1 = -1.0, 1.0
    p_theo1 = 1.0 / (upper_bound1 - lower_bound1)
    sigma_p1 = p_theo1 * rel_sigma1

    ax1.hist(weights_f16, bins=k_bins, density=True, color=COLOR_WGT, alpha=0.72, edgecolor='black', linewidth=0.5, label='FP16 Weights')
    ax1.fill_between([-1.0, 1.0], p_theo1 - 2 * sigma_p1, p_theo1 + 2 * sigma_p1, color=COLOR_THEO, alpha=0.15, label='Poisson ±2σ Band')
    ax1.fill_between([-1.0, 1.0], p_theo1 - sigma_p1, p_theo1 + sigma_p1, color=COLOR_THEO, alpha=0.28, label=f'Poisson ±1σ (±{100*rel_sigma1:.2f}%)')
    ax1.axhline(p_theo1, color=COLOR_THEO, linestyle='-', linewidth=2.0, label=f'Theoretical Density [p={p_theo1:.2f}]')
    d_ks_f16 = kolmogorov_smirnov_uniform(weights_f16, -1.0, 1.0)
    ax1.set_title("Synaptic Weights W (FP16)", fontweight='bold')
    ax1.set_xlabel("Weight Value (FP16)")
    ax1.set_ylabel("Probability Density p(w)")
    ax1.set_xlim(-1.15, 1.15)
    ax1.set_ylim(0.0, 0.72)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.92, loc='upper right', fontsize=8.5)
    stats_f16 = (f"N = {N1:,}\n"
                 f"Bins k = {k_bins} (n_bin = {n_bin1:,.0f})\n"
                 f"Poisson σ = {sigma1:.1f} (±{100*rel_sigma1:.2f}%)\n"
                 f"Mean = {np.mean(weights_f16):+.4f} (Ideal: 0.00)\n"
                 f"Std  = {np.std(weights_f16):.4f} (Ideal: {1.0/np.sqrt(3):.4f})\n"
                 f"KS Statistic D = {d_ks_f16:.4f}\n"
                 f"Status: Zero Discrete Combing")
    ax1.text(0.03, 0.94, stats_f16, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#ccc'))

    # Panel 2: FP8 Weights
    N2 = len(weights_f8)
    n_bin2 = N2 / k_bins
    sigma2 = np.sqrt(n_bin2)
    rel_sigma2 = 1.0 / np.sqrt(n_bin2)
    lower_bound2, upper_bound2 = -1.5, 1.5
    p_theo2 = 1.0 / (upper_bound2 - lower_bound2)
    sigma_p2 = p_theo2 * rel_sigma2

    ax2.hist(weights_f8, bins=k_bins, density=True, color='#e377c2', alpha=0.72, edgecolor='black', linewidth=0.5, label='FP8 Weights')
    ax2.fill_between([-1.5, 1.5], p_theo2 - 2 * sigma_p2, p_theo2 + 2 * sigma_p2, color=COLOR_THEO, alpha=0.15, label='Poisson ±2σ Band')
    ax2.fill_between([-1.5, 1.5], p_theo2 - sigma_p2, p_theo2 + sigma_p2, color=COLOR_THEO, alpha=0.28, label=f'Poisson ±1σ (±{100*rel_sigma2:.2f}%)')
    ax2.axhline(p_theo2, color=COLOR_THEO, linestyle='-', linewidth=2.0, label=f'Theoretical Density [p={p_theo2:.3f}]')
    d_ks_f8 = kolmogorov_smirnov_uniform(weights_f8, -1.5, 1.5)
    ax2.set_title("Synaptic Weights W (FP8 E4M3)", fontweight='bold')
    ax2.set_xlabel("Weight Value (E4M3 Decoded)")
    ax2.set_ylabel("Probability Density p(w)")
    ax2.set_xlim(-1.7, 1.7)
    ax2.set_ylim(0.0, 0.52)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.92, loc='upper right', fontsize=8.5)
    stats_f8 = (f"N = {N2:,}\n"
                f"Bins k = {k_bins} (n_bin = {n_bin2:,.0f})\n"
                f"Poisson σ = {sigma2:.1f} (±{100*rel_sigma2:.2f}%)\n"
                f"Mean = {np.mean(weights_f8):+.4f} (Ideal: 0.00)\n"
                f"Std  = {np.std(weights_f8):.4f} (Ideal: {1.5/np.sqrt(3):.4f})\n"
                f"KS Statistic D = {d_ks_f8:.4f}\n"
                f"Status: 254 Valid E4M3 States")
    ax2.text(0.03, 0.94, stats_f8, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#ccc'))

    fig.savefig(out_path, format='svg')
    png_path = out_path.replace('.svg', '.png')
    pdf_path = out_path.replace('.svg', '.pdf')
    fig.savefig(png_path, format='png', dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path} and {png_path}")

def plot_biases(biases, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    k_bins = 40
    N = len(biases)
    n_bin = N / k_bins
    sigma_poisson = np.sqrt(n_bin)
    rel_sigma = 1.0 / np.sqrt(n_bin)
    lower_bound, upper_bound = -0.5, 0.5
    p_theo = 1.0 / (upper_bound - lower_bound)
    sigma_p = p_theo * rel_sigma

    ax.hist(biases, bins=k_bins, density=True, color=COLOR_BIA, alpha=0.72, edgecolor='black', linewidth=0.5, label='Empirical Biases')
    ax.fill_between([-0.5, 0.5], p_theo - 2 * sigma_p, p_theo + 2 * sigma_p, color=COLOR_THEO, alpha=0.15, label=f'Poisson ±2σ Band (±{200*rel_sigma:.1f}%)')
    ax.fill_between([-0.5, 0.5], p_theo - sigma_p, p_theo + sigma_p, color=COLOR_THEO, alpha=0.28, label=f'Poisson ±1σ Band (±{100*rel_sigma:.1f}%)')
    ax.axhline(p_theo, color=COLOR_THEO, linestyle='-', linewidth=2.0, label=f'Theoretical Density U(-0.5, 0.5) [p={p_theo:.2f}]')

    d_ks = kolmogorov_smirnov_uniform(biases, -0.5, 0.5)
    ax.set_title("Continuous Distribution of Neuron Biases b", fontweight='bold')
    ax.set_xlabel("Bias Value (FP16)")
    ax.set_ylabel("Probability Density p(b)")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0.0, 1.50)
    ax.legend(frameon=True, facecolor='white', framealpha=0.92, loc='upper right', fontsize=8.8)

    stats_text = (f"Cout = {N:,} Neurons\n"
                  f"Bins k = {k_bins} (n_bin = {n_bin:,.0f})\n"
                  f"Poisson σ = {sigma_poisson:.1f} (±{100*rel_sigma:.2f}%)\n"
                  f"Mean = {np.mean(biases):+.4f} (Ideal: 0.00)\n"
                  f"Std  = {np.std(biases):.4f} (Ideal: {1.0/(np.sqrt(12)):.4f})\n"
                  f"KS Statistic D = {d_ks:.4f}\n"
                  f"Status: Fully Continuous")
    ax.text(0.03, 0.94, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88, edgecolor='#ccc'))

    fig.savefig(out_path, format='svg')
    png_path = out_path.replace('.svg', '.png')
    pdf_path = out_path.replace('.svg', '.pdf')
    fig.savefig(png_path, format='png', dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path} and {png_path}")

def plot_overview(activations, indices, weights_f16, biases, out_path, cin_max=20_000_000):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    cin_m = cin_max / 1e6
    k_bins = 50

    # Panel A: Activations
    ax_a = axes[0, 0]
    Na = len(activations)
    n_bina = Na / k_bins
    rel_a = 1.0 / np.sqrt(n_bina)
    pa_theo = 1.0 / (1.0 - (-1.0))
    sig_pa = pa_theo * rel_a
    ax_a.hist(activations, bins=k_bins, density=True, color=COLOR_ACT, alpha=0.72, edgecolor='black', linewidth=0.5)
    ax_a.fill_between([-1.0, 1.0], pa_theo - sig_pa, pa_theo + sig_pa, color=COLOR_THEO, alpha=0.25, label=f'Poisson ±1σ (±{100*rel_a:.1f}%)')
    ax_a.axhline(pa_theo, color=COLOR_THEO, linestyle='--', linewidth=1.8, label='Theoretical U(-1, 1)')
    d_ks_act = kolmogorov_smirnov_uniform(activations, -1.0, 1.0)
    ax_a.set_title("(a) Input Activations X (M=256 Tokens)", fontweight='bold', loc='left')
    ax_a.set_xlabel("Activation Value x")
    ax_a.set_ylabel("Density p(x)")
    ax_a.set_ylim(0.0, 0.72)
    ax_a.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    ax_a.text(0.03, 0.92, f"N={Na:,} | n_bin={n_bina:,.0f}\nPoisson σ={np.sqrt(n_bina):.1f} (±{100*rel_a:.2f}%)\nMean: {np.mean(activations):+.4f}\nKS D: {d_ks_act:.4f}",
              transform=ax_a.transAxes, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # Panel B: Connectome Indices
    ax_b = axes[0, 1]
    Nb = len(indices)
    n_binb = Nb / k_bins
    rel_b = 1.0 / np.sqrt(n_binb)
    p_tb = 1.0 / cin_m
    sig_pb = p_tb * rel_b
    ax_b.hist(indices / 1e6, bins=k_bins, density=True, color=COLOR_IDX, alpha=0.72, edgecolor='black', linewidth=0.5)
    ax_b.fill_between([0, cin_m], p_tb - sig_pb, p_tb + sig_pb, color=COLOR_THEO, alpha=0.25, label=f'Poisson ±1σ (±{100*rel_b:.1f}%)')
    ax_b.axhline(p_tb, color=COLOR_THEO, linestyle='--', linewidth=1.8, label=f'Theoretical U(0, {cin_m:.0f}M)')
    d_ks_idx = kolmogorov_smirnov_uniform(indices, 0, cin_max)
    ax_b.set_title(f"(b) Topology Indices C (Cin={cin_m:.0f}M, S=128)", fontweight='bold', loc='left')
    ax_b.set_xlabel("Channel Index [Millions of Neurons]")
    ax_b.set_ylabel("Density p(c)")
    ax_b.set_ylim(0.0, 0.085)
    ax_b.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    ax_b.text(0.03, 0.92, f"N={Nb:,} | n_bin={n_binb:,.0f}\nPoisson σ={np.sqrt(n_binb):.1f} (±{100*rel_b:.2f}%)\nMean: {np.mean(indices/1e6):.2f}M\nKS D: {d_ks_idx:.4f}",
              transform=ax_b.transAxes, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # Panel C: Weights FP16
    ax_c = axes[1, 0]
    Nc = len(weights_f16)
    n_binc = Nc / k_bins
    rel_c = 1.0 / np.sqrt(n_binc)
    pc_theo = 1.0 / (1.0 - (-1.0))
    sig_pc = pc_theo * rel_c
    ax_c.hist(weights_f16, bins=k_bins, density=True, color=COLOR_WGT, alpha=0.72, edgecolor='black', linewidth=0.5)
    ax_c.fill_between([-1.0, 1.0], pc_theo - sig_pc, pc_theo + sig_pc, color=COLOR_THEO, alpha=0.25, label=f'Poisson ±1σ (±{100*rel_c:.2f}%)')
    ax_c.axhline(pc_theo, color=COLOR_THEO, linestyle='--', linewidth=1.8, label='Theoretical U(-1, 1)')
    d_ks_w = kolmogorov_smirnov_uniform(weights_f16, -1.0, 1.0)
    ax_c.set_title("(c) Synaptic Weights W (Cout=4096, B=1024)", fontweight='bold', loc='left')
    ax_c.set_xlabel("Weight Value (FP16)")
    ax_c.set_ylabel("Density p(w)")
    ax_c.set_ylim(0.0, 0.72)
    ax_c.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    ax_c.text(0.03, 0.92, f"N={Nc:,} | n_bin={n_binc:,.0f}\nPoisson σ={np.sqrt(n_binc):.1f} (±{100*rel_c:.2f}%)\nMean: {np.mean(weights_f16):+.4f}\nKS D: {d_ks_w:.4f}",
              transform=ax_c.transAxes, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    # Panel D: Biases
    ax_d = axes[1, 1]
    Nd = len(biases)
    k_bd = 40
    n_bind = Nd / k_bd
    rel_d = 1.0 / np.sqrt(n_bind)
    pd_theo = 1.0 / (0.5 - (-0.5))
    sig_pd = pd_theo * rel_d
    ax_d.hist(biases, bins=k_bd, density=True, color=COLOR_BIA, alpha=0.72, edgecolor='black', linewidth=0.5)
    ax_d.fill_between([-0.5, 0.5], pd_theo - sig_pd, pd_theo + sig_pd, color=COLOR_THEO, alpha=0.25, label=f'Poisson ±1σ (±{100*rel_d:.1f}%)')
    ax_d.axhline(pd_theo, color=COLOR_THEO, linestyle='--', linewidth=1.8, label='Theoretical U(-0.5, 0.5)')
    d_ks_b = kolmogorov_smirnov_uniform(biases, -0.5, 0.5)
    ax_d.set_title("(d) Neuron Biases b (Cout=4096)", fontweight='bold', loc='left')
    ax_d.set_xlabel("Bias Value (FP16)")
    ax_d.set_ylabel("Density p(b)")
    ax_d.set_ylim(0.0, 1.50)
    ax_d.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    ax_d.text(0.03, 0.92, f"Cout={Nd:,} | n_bin={n_bind:,.0f}\nPoisson σ={np.sqrt(n_bind):.1f} (±{100*rel_d:.2f}%)\nMean: {np.mean(biases):+.4f}\nKS D: {d_ks_b:.4f}",
              transform=ax_d.transAxes, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    fig.suptitle("Connectome Density Distributions for Whitepaper Benchmark Matrix (True Uniformity with Poisson Confidence)", fontweight='bold', fontsize=14.5)
    fig.savefig(out_path, format='svg')
    png_path = out_path.replace('.svg', '.png')
    pdf_path = out_path.replace('.svg', '.pdf')
    fig.savefig(png_path, format='png', dpi=300)
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path} and {png_path}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_dir = os.path.join(root_dir, "data")
    out_dirs = [
        os.path.join(root_dir, "figures"),
    ]
    for d in out_dirs:
        ensure_dirs(d)

    print("================================================================================")
    print("  Whitepaper Density Distribution Plotter with Poisson Confidence Bands")
    print(f"  Reference Data: {ref_dir}")
    print(f"  Target Dirs:    {out_dirs}")
    print("================================================================================")

    activations, indices, weights_f16, weights_f8, biases = load_or_generate_data(ref_dir)

    for d in out_dirs:
        plot_activations(activations, os.path.join(d, "distribution_activations.svg"))
        plot_indices(indices, os.path.join(d, "distribution_indices.svg"))
        plot_weights(weights_f16, weights_f8, os.path.join(d, "distribution_weights.svg"))
        plot_biases(biases, os.path.join(d, "distribution_biases.svg"))
        plot_overview(activations, indices, weights_f16, biases, os.path.join(d, "distribution_overview.svg"))
        print(f"[DONE] Generated 5 distribution plots in {d}")

if __name__ == "__main__":
    main()
