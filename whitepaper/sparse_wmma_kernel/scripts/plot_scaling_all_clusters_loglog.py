#!/usr/bin/env python3
# ==============================================================================
# scripts/plot_scaling_all_clusters_loglog.py
#
# Scientific Log(Y) - Log(X) Scaling Plot for Kernel 44 (FP16 only) on AMD RDNA4.
# Target: AMD Radeon RX 9060 XT (RDNA4 Navi 44 / GFX1200).
#
# Publication Specifications:
#   - Format: Pure FP16 (Multi-Wave Ring Buffer / 256 MiB VRAM-Streaming Pool)
#   - Standard ("Flexible") Series: S in {16, 32, 64, 128, 256} at M=512, B=1024, Cout=8192
#   - Fastest Comparison Series: S=512 (B=1024, Cout=8192 and B=2048, Cout=4096) at M=512
#   - Legend: Strictly confined to lower-left corner, zero data overlap
# ==============================================================================

import os
import sys

import numpy as np
import matplotlib

import json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 18,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 12.0,
    'figure.titlesize': 26,
    'figure.dpi': 300,
    'axes.edgecolor': '#222222',
    'axes.linewidth': 1.6,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'svg.fonttype': 'none',
})

THEO_FP16_MATRIX = 51.5

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(root_dir, "data", "benchmark_results_k44.json")
    if not os.path.exists(json_path):
        print(f"[ERROR] Result file {json_path} not found!")
        sys.exit(1)

    with open(json_path, 'r') as f:
        records = json.load(f)

    series_cfg = [
        ("S512_b2k_c4k", "FP16 S=512, B=2048, Cout=4096 (Fastest Peak)",      "#0d47a1", "*", "-",  3.4, 12.5, 14),
        ("S512_b1k_c8k", "FP16 S=512, B=1024, Cout=8192 (Fastest B=1024)",    "#1976d2", "P", "--", 3.0, 9.5,  13),
        ("S256_std",     "FP16 S=256, B=1024, Cout=8192 (Standard Flexible)", "#b71c1c", "D", "-",  3.2, 9.0,  12),
        ("S128_std",     "FP16 S=128, B=1024, Cout=8192 (64 Clusters)",       "#e65100", "s", "--", 2.8, 8.5,  10),
        ("S64_std",      "FP16 S=64,  B=1024, Cout=8192 (128 Clusters)",      "#f57c00", "^", "-.", 2.6, 8.5,   9),
        ("S32_std",      "FP16 S=32,  B=1024, Cout=8192 (256 Clusters)",      "#ffb300", "v", ":",  2.6, 8.5,   8),
        ("S16_std",      "FP16 S=16,  B=1024, Cout=8192 (512 Clusters)",      "#7b1fa2", "o", ":",  2.6, 8.5,   7),
    ]

    grouped = {tag: [] for tag, *_ in series_cfg}
    for r in records:
        tag = r.get("tag", "")
        if tag in grouped:
            grouped[tag].append(r)

    fig, ax = plt.subplots(figsize=(15.5, 10.8))
    fig.subplots_adjust(bottom=0.15, top=0.70, left=0.11, right=0.89)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(800, 16_000_000)
    ax.set_ylim(0.75, 60.0)

    refs = [
        {"y": 42.34, "label": "Pure-Reg WMMA B=1024 (0 Mem, 42.34 TSyn/s / 84.7 TF)", "color": "#455a64", "ls": ":", "lw": 2.2},
        {"y": 22.26, "label": "Pure-Reg VOPD Dual-Issue B=1024 (0 Mem, 22.26 TSyn/s / 44.5 TF)", "color": "#78909c", "ls": ":", "lw": 2.0},
    ]
    for ref in refs:
        ax.axhline(ref["y"], color=ref["color"], linestyle=ref["ls"], linewidth=ref["lw"], alpha=0.85)
        ax.text(14_500_000, ref["y"] * 1.04, ref["label"], color=ref["color"],
                fontsize=12.0, fontweight='bold', horizontalalignment='right',
                bbox=dict(boxstyle='square,pad=0.18', facecolor='white', alpha=0.85, edgecolor='none'))

    ax.axvline(32768, color="#2e7d32", linestyle="--", linewidth=2.0, alpha=0.75)
    ax.text(32768 * 1.06, 39.0, "32 MB Infinity Cache Boundary\n(X = 32 MiB at M=512)",
            color="#1b5e20", fontsize=11.5, fontweight='bold', ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#e8f5e9', edgecolor='#2e7d32', alpha=0.92))

    plot_handles = []
    for tag, label, color, marker, ls, lw, ms, z in series_cfg:
        pts = sorted(grouped[tag], key=lambda x: x["Cin"])
        if not pts:
            continue
        cin_arr = np.array([p["Cin"] for p in pts])
        tsyn_arr = np.array([p.get("tsyns_mean", p["best_tsyns"]) for p in pts])
        tmin_arr = np.array([p.get("tsyns_min", tsyn_arr[i] * 0.98) for i, p in enumerate(pts)])
        tmax_arr = np.array([p.get("tsyns_max", tsyn_arr[i] * 1.02) for i, p in enumerate(pts)])
        yerr = [np.maximum(0.0, tsyn_arr - tmin_arr), np.maximum(0.0, tmax_arr - tsyn_arr)]

        eb = ax.errorbar(cin_arr, tsyn_arr, yerr=yerr, label=label,
                         color=color, marker=marker, linestyle=ls,
                         linewidth=lw, markersize=ms, capsize=4.0,
                         capthick=1.4, elinewidth=1.4, alpha=0.95, zorder=z)
        plot_handles.append(eb)

    ax.set_xlabel("Input Neurons Cin [Logarithmic Scale]", fontweight='bold', labelpad=12)
    ax.set_ylabel("Inference Throughput [TSyn/s]", fontweight='bold', labelpad=12)

    # Space ticks cleanly (1k .. 4M, 10M) so 8M and 10M tick labels never touch
    cin_ticks = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 10000000]
    cin_labels = ["1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M", "2M", "4M", "10M"]
    ax.set_xticks(cin_ticks)
    ax.set_xticklabels(cin_labels, rotation=42, ha='right')

    y_ticks = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y)}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis='y', which='minor', left=False)

    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(cin_ticks)
    top_labels = []
    for c in cin_ticks:
        sp = 100.0 * (1.0 - 1024.0 / c)
        if c == 1024:
            top_labels.append("0.0%")
        elif c < 100000:
            top_labels.append(f"{sp:.1f}%")
        elif c < 1000000:
            top_labels.append(f"{sp:.2f}%")
        else:
            top_labels.append(f"{sp:.3f}%")
    ax_top.set_xticklabels(top_labels, rotation=42, ha='left', fontsize=13.5)
    ax_top.set_xlabel("Connectome Sparsity [1 - B / Cin] (Percent, for B=1024)", fontweight='bold', labelpad=18)

    ax_right = ax.twinx()
    ax_right.set_yscale('log')
    ax_right.set_ylim((0.75 / THEO_FP16_MATRIX) * 100.0, (60.0 / THEO_FP16_MATRIX) * 100.0)
    ax_right.set_ylabel("% vendor specified matrix peak FLOPS FP16", fontweight='bold', labelpad=12)
    util_ticks = [2, 5, 10, 20, 30, 40, 50, 60, 80, 100]
    ax_right.set_yticks(util_ticks)
    ax_right.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val)}%"))
    ax_right.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax_right.tick_params(axis='y', which='minor', right=False)

    leg = ax.legend(handles=plot_handles, loc='lower left', ncol=1,
                    bbox_to_anchor=(0.01, 0.01),
                    framealpha=0.96, facecolor='white',
                    edgecolor='#888888', fontsize=10.8,
                    title="Multi-Wave Ring Buffer (M=512, 256 MiB VRAM-Streaming Pool)",
                    title_fontsize=11.4)
    leg.get_frame().set_linewidth(1.4)

    fig.suptitle("Scaling Behavior of DG-WMMA (FP16): Standard Flexible (S=256) vs. Fastest (S=512)\nAMD Radeon RX 9060 XT (Navi 44 / GFX1200 / RDNA4)",
                 fontweight='bold', fontsize=21, y=0.97)

    out_dirs = [
        os.path.join(root_dir, "figures"),
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "scaling_cin_sparsity_all_clusters_loglog.svg"), format='svg')
        fig.savefig(os.path.join(d, "scaling_cin_sparsity_all_clusters_loglog.png"), format='png', dpi=300)
        fig.savefig(os.path.join(d, "scaling_cin_sparsity_all_clusters_loglog.pdf"), format='pdf')
        print(f"[PLOT] All-Clusters Log-Log Scaling Plot successfully exported to {d}")
    plt.close(fig)

if __name__ == "__main__":
    main()
