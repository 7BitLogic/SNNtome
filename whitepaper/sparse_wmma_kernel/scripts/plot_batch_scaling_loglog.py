#!/usr/bin/env python3
# ==============================================================================
# scripts/plot_batch_scaling_loglog.py
#
# Scientific Log(Y) - Log(X) Batch Size Scaling Plot for Kernel 44 (FP16 only)
# Target: AMD Radeon RX 9060 XT (RDNA4 Navi 44 / GFX1200)
#
# Parameters:
#   - Precision: FP16 only (Multi-Wave Ring Buffer / 256 MiB VRAM-Streaming Pool)
#   - Compares In-Cache (Cin=8k) vs. VRAM-Streaming (Cin=1M) across S in {16, 128, 256, 512}
#   - Sweeps Batch Size M from 1 to 16,384
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
    'legend.fontsize': 11.5,
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
    json_path = os.path.join(root_dir, "data", "benchmark_batch_scaling_results.json")
    out_dirs = [
        os.path.join(root_dir, "figures"),
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"[ERROR] Result file {json_path} not found!")
        sys.exit(1)

    with open(json_path, 'r') as f:
        records = json.load(f)

    series_cfg = [
        ("Cin8k_S512_B2048_C4k", "Cin=8k (In-Cache): S=512, B=2048, Cout=4k (Fastest)",         "#0d47a1", "*", "-",  3.2, 11.5, 14),
        ("Cin8k_S256_B1024_C8k", "Cin=8k (In-Cache): S=256, B=1024, Cout=8k (Flexible)",        "#1565c0", "D", "-",  3.0, 8.8,  13),
        ("Cin8k_S128_B1024_C8k", "Cin=8k (In-Cache): S=128, B=1024, Cout=8k",                   "#00838f", "s", "--", 2.6, 8.2,  11),
        ("Cin1M_S512_B2048_C4k", "Cin=1M (VRAM-Streaming): S=512, B=2048, Cout=4k (Fastest)",   "#2e7d32", "P", "-",  3.2, 10.0, 12),
        ("Cin1M_S512_B1024_C8k", "Cin=1M (VRAM-Streaming): S=512, B=1024, Cout=8k",             "#558b2f", "^", "-.", 2.7, 8.5,  10),
        ("Cin1M_S256_B1024_C8k", "Cin=1M (VRAM-Streaming): S=256, B=1024, Cout=8k (Flexible)",  "#b71c1c", "D", "-",  3.0, 8.8,  12),
        ("Cin1M_S128_B1024_C8k", "Cin=1M (VRAM-Streaming): S=128, B=1024, Cout=8k",             "#e65100", "s", "--", 2.6, 8.2,   9),
        ("Cin8k_S16_B1024_C8k",  "Cin=8k (In-Cache): S=16,  B=1024, Cout=8k",                   "#7b1fa2", "o", ":",  2.5, 8.0,   8),
        ("Cin1M_S16_B1024_C8k",  "Cin=1M (VRAM-Streaming): S=16,  B=1024, Cout=8k",             "#ad1457", "v", ":",  2.5, 8.0,   7),
    ]

    grouped = {tag: [] for tag, *_ in series_cfg}
    for r in records:
        tag = r.get("tag", "")
        if tag in grouped:
            grouped[tag].append(r)

    fig, ax = plt.subplots(figsize=(15.5, 10.5))
    fig.subplots_adjust(bottom=0.15, top=0.88, left=0.11, right=0.89)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlim(0.7, 24000)
    ax.set_ylim(0.08, 65.0)

    refs = [
        {"y": 42.34, "label": "Pure-Reg WMMA B=1024 (0 Mem, 42.34 TSyn/s / 84.7 TF)", "color": "#455a64", "ls": ":", "lw": 2.2},
        {"y": 22.26, "label": "Pure-Reg VOPD Dual-Issue B=1024 (0 Mem, 22.26 TSyn/s / 44.5 TF)", "color": "#78909c", "ls": ":", "lw": 2.0},
    ]
    for ref in refs:
        ax.axhline(ref["y"], color=ref["color"], linestyle=ref["ls"], linewidth=ref["lw"], alpha=0.85)
        ax.text(0.82, ref["y"] * 1.05, ref["label"], color=ref["color"],
                fontsize=11.0, fontweight='bold', horizontalalignment='left',
                bbox=dict(boxstyle='square,pad=0.14', facecolor='white', alpha=0.88, edgecolor='none'))

    plot_handles = []
    for tag, label, color, marker, ls, lw, ms, z in series_cfg:
        pts = sorted(grouped[tag], key=lambda x: x["M"])
        if not pts:
            continue
        m_arr = np.array([p["M"] for p in pts])
        tsyn_arr = np.array([p.get("tsyns_mean", p["best_tsyns"]) for p in pts])
        tmin_arr = np.array([p.get("tsyns_min", tsyn_arr[i] * 0.98) for i, p in enumerate(pts)])
        tmax_arr = np.array([p.get("tsyns_max", tsyn_arr[i] * 1.02) for i, p in enumerate(pts)])
        yerr = [np.maximum(0.0, tsyn_arr - tmin_arr), np.maximum(0.0, tmax_arr - tsyn_arr)]

        eb = ax.errorbar(m_arr, tsyn_arr, yerr=yerr, label=label,
                         color=color, marker=marker, linestyle=ls,
                         linewidth=lw, markersize=ms, capsize=4.0,
                         capthick=1.4, elinewidth=1.4, alpha=0.95, zorder=z)
        plot_handles.append(eb)

    ax.set_xlabel("Batch Size M (Number of Tokens) [Log2 Scale]", fontweight='bold', labelpad=12)
    ax.set_ylabel("Inference Throughput [TSyn/s]", fontweight='bold', labelpad=12)

    m_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    m_labels = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1k", "2k", "4k", "8k", "16k"]
    ax.set_xticks(m_ticks)
    ax.set_xticklabels(m_labels, rotation=35, ha='right')

    y_ticks = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis='y', which='minor', left=False)

    ax_right = ax.twinx()
    ax_right.set_yscale('log')
    ax_right.set_ylim((0.08 / THEO_FP16_MATRIX) * 100.0, (65.0 / THEO_FP16_MATRIX) * 100.0)
    ax_right.set_ylabel("% vendor specified matrix peak FLOPS FP16", fontweight='bold', labelpad=12)
    util_ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100]
    ax_right.set_yticks(util_ticks)
    ax_right.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{val:g}%"))
    ax_right.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax_right.tick_params(axis='y', which='minor', right=False)

    leg = ax.legend(handles=plot_handles, loc='lower right', framealpha=0.96, facecolor='white',
                    edgecolor='#888888', fontsize=11.5,
                    title="Multi-Wave Ring Buffer (256 MiB VRAM-Streaming Pool)",
                    title_fontsize=12.5)
    leg.get_frame().set_linewidth(1.4)

    fig.suptitle("Batch Size Scaling of DG-WMMA (FP16): In-Cache (Cin=8k) vs. VRAM-Streaming (Cin=1M)\nAMD Radeon RX 9060 XT (Navi 44 / GFX1200 / RDNA4)",
                 fontweight='bold', fontsize=20.5, y=0.96)

    for d in out_dirs:
        out_svg = os.path.join(d, "batch_scaling_loglog.svg")
        out_png = os.path.join(d, "batch_scaling_loglog.png")
        out_pdf = os.path.join(d, "batch_scaling_loglog.pdf")
        fig.savefig(out_svg, format='svg')
        fig.savefig(out_png, format='png', dpi=300)
        fig.savefig(out_pdf, format='pdf')
        print(f"[PLOT] Batch scaling plot successfully exported to {d}")
    plt.close(fig)

if __name__ == "__main__":
    main()
