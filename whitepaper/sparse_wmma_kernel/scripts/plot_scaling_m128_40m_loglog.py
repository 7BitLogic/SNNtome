#!/usr/bin/env python3
# ==============================================================================
# scripts/plot_scaling_m128_40m_loglog.py
#
# Scientific Log(Y) - Log(X) Scaling Plot for Kernel 44 at M=128 up to Cin=40M.
# Target: AMD Radeon RX 9060 XT (RDNA4 Navi 44 / GFX1200).
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
    'legend.fontsize': 14.5,
    'figure.titlesize': 26,
    'figure.dpi': 300,
    'axes.edgecolor': '#222222',
    'axes.linewidth': 1.6,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'svg.fonttype': 'none',
})

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(root_dir, "data", "benchmark_results_k44_m128_40m.json")
    out_dir = os.path.join(root_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"[ERROR] Result file {json_path} not found!")
        sys.exit(1)

    with open(json_path, 'r') as f:
        records = json.load(f)

    series = {
        ("FP16", 256): {"cin": [], "tsyns": [], "tsyns_min": [], "tsyns_max": [], "label": "FP16 S=256 (16 Clusters, M=128)", "color": "#b71c1c", "marker": "D", "ls": "-"},
        ("FP16", 128): {"cin": [], "tsyns": [], "tsyns_min": [], "tsyns_max": [], "label": "FP16 S=128 (32 Clusters, M=128)", "color": "#e65100", "marker": "s", "ls": "--"},
        ("FP16", 64):  {"cin": [], "tsyns": [], "tsyns_min": [], "tsyns_max": [], "label": "FP16 S=64 (64 Clusters, M=128)",  "color": "#f57c00", "marker": "^", "ls": "-."},
        ("FP16", 32):  {"cin": [], "tsyns": [], "tsyns_min": [], "tsyns_max": [], "label": "FP16 S=32 (128 Clusters, M=128)", "color": "#ffb300", "marker": "v", "ls": ":"},
        ("FP16", 16):  {"cin": [], "tsyns": [], "tsyns_min": [], "tsyns_max": [], "label": "FP16 S=16 (256 Clusters, M=128)", "color": "#7b1fa2", "marker": "o", "ls": ":"},
    }

    for r in records:
        if r.get("format") != "FP16":
            continue
        key = ("FP16", r["S"])
        if key in series:
            series[key]["cin"].append(r["Cin"])
            mean_v = r.get("tsyns_mean", r["best_tsyns"])
            min_v = r.get("tsyns_min", mean_v * 0.98)
            max_v = r.get("tsyns_max", mean_v * 1.02)
            series[key]["tsyns"].append(mean_v)
            series[key]["tsyns_min"].append(min_v)
            series[key]["tsyns_max"].append(max_v)

    fig, ax = plt.subplots(figsize=(15.5, 10.8))
    fig.subplots_adjust(bottom=0.15, top=0.70, left=0.11, right=0.89)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(800, 58_000_000)
    ax.set_ylim(1.8, 60.0)

    THEO_FP16_MATRIX = 51.5
    refs = [
        {"y": 42.34, "label": "Pure-Reg WMMA B=1024, M=256 (0 Mem, 42.34 TSyn/s / 84.7 TF)", "color": "#455a64", "ls": ":", "lw": 2.2},
        {"y": 37.57, "label": "Pure-Reg WMMA B=1024, M=128 (0 Mem, 37.57 TSyn/s / 75.1 TF)", "color": "#607d8b", "ls": ":", "lw": 2.2},
        {"y": 22.26, "label": "Pure-Reg VOPD Dual-Issue B=1024 (0 Mem, 22.26 TSyn/s / 44.5 TF)", "color": "#78909c", "ls": ":", "lw": 2.0},
    ]

    for ref in refs:
        ax.axhline(ref["y"], color=ref["color"], linestyle=ref["ls"], linewidth=ref["lw"], alpha=0.85)
        y_text = ref["y"] * (1.035 if ref["y"] != 37.57 else 0.90)
        ax.text(53_000_000, y_text, ref["label"], color=ref["color"],
                fontsize=12.5, fontweight='bold', horizontalalignment='right',
                bbox=dict(boxstyle='square,pad=0.18', facecolor='white', alpha=0.85, edgecolor='none'))

    # Vertical line at 32-bit Buffer Descriptor limit (2^25 = 33,554,432 neurons = 4 GiB / tile)
    ax.axvline(33_554_432, color="#c62828", linestyle="--", linewidth=2.0, alpha=0.80)
    ax.text(30_500_000, 16.8, "32-bit MUBUF Tile Limit ($2^{25} = 33.55\\mathrm{M}$ / 4 GiB)\n$\\rightarrow$ Switch to 64-bit $\\mathtt{global\\_load}$",
            color="#c62828", fontsize=11.0, fontweight='bold', ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#ffebee', edgecolor='#c62828', alpha=0.94))

    plot_handles = []
    for key, data in series.items():
        if not data["cin"]:
            continue
        cin_arr = np.array(data["cin"])
        tsyn_arr = np.array(data["tsyns"])
        tmin_arr = np.array(data["tsyns_min"])
        tmax_arr = np.array(data["tsyns_max"])
        idx_sort = np.argsort(cin_arr)

        cin_sorted = cin_arr[idx_sort]
        tsyn_sorted = tsyn_arr[idx_sort]
        tmin_sorted = tmin_arr[idx_sort]
        tmax_sorted = tmax_arr[idx_sort]

        yerr_lower = np.maximum(0.0, tsyn_sorted - tmin_sorted)
        yerr_upper = np.maximum(0.0, tmax_sorted - tsyn_sorted)
        yerr = [yerr_lower, yerr_upper]

        eb = ax.errorbar(cin_sorted, tsyn_sorted, yerr=yerr, label=data["label"],
                         color=data["color"], marker=data["marker"],
                         linestyle=data["ls"], linewidth=2.8, markersize=8.5,
                         capsize=4.5, capthick=1.5, elinewidth=1.5, alpha=0.95)
        plot_handles.append(eb)

    # Demo Curve: Kernel 44 Dual-Buffer Workaround (S=128, M=128)
    dual_json = os.path.join(root_dir, "data", "benchmark_k44_dual_buffer_s128_m128.json")
    if os.path.exists(dual_json):
        with open(dual_json, 'r') as f:
            dual_records = json.load(f)
        d_cin = np.array([r["Cin"] for r in dual_records])
        d_mean = np.array([r.get("tsyns_mean", r["best_tsyns"]) for r in dual_records])
        d_min = np.array([r.get("tsyns_min", d_mean[i] * 0.98) for i, r in enumerate(dual_records)])
        d_max = np.array([r.get("tsyns_max", d_mean[i] * 1.02) for i, r in enumerate(dual_records)])
        idx_d = np.argsort(d_cin)
        d_cin, d_mean, d_min, d_max = d_cin[idx_d], d_mean[idx_d], d_min[idx_d], d_max[idx_d]
        d_yerr = [np.maximum(0.0, d_mean - d_min), np.maximum(0.0, d_max - d_mean)]

        eb_dual = ax.errorbar(
            d_cin, d_mean, yerr=d_yerr,
            label="FP16 S=128 Dual-Buffer Workaround (Dual-V#, max 64M)",
            color="#00838f", marker="P", linestyle="-.",
            linewidth=2.8, markersize=9.5, capsize=4.5, capthick=1.5, elinewidth=1.5, alpha=0.95, zorder=12
        )
        plot_handles.append(eb_dual)

        # Callout at 40M comparing 64-bit global_load vs Dual-Buffer workaround
        dual_40m = float(d_mean[-1])
        ax.annotate(
            f"Dual-Buffer Workaround (S=128):\n{dual_40m:.2f} TSyn/s at 40M (limited to $\\leq$64M)",
            xy=(40_000_000, dual_40m),
            xytext=(5_500_000, 12.2),
            va='center', ha='center', fontsize=11.0, fontweight='bold', color="#006064",
            bbox=dict(boxstyle="round,pad=0.22", fc="#e0f7fa", ec="#00838f", lw=1.5, alpha=0.95),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.10", color="#00838f", lw=1.8),
            zorder=15
        )

    ax.set_xlabel("Input Neurons Cin [Logarithmic Scale]", fontweight='bold', labelpad=12)
    ax.set_ylabel("Inference Throughput [TSyn/s]", fontweight='bold', labelpad=12)

    cin_ticks = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 40000000]
    cin_labels = ["1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M", "2M", "4M", "8M", "16M", "40M"]
    ax.set_xticks(cin_ticks)
    ax.set_xticklabels(cin_labels, rotation=42, ha='right')

    y_ticks = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60]
    ax.set_ylim(0.85, 60.0)
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
            top_labels.append(f"{sp:.4f}%")
    ax_top.set_xticklabels(top_labels, rotation=42, ha='left', fontsize=13.5)
    ax_top.set_xlabel("Connectome Sparsity [1 - B / Cin] (Percent)", fontweight='bold', labelpad=18)

    ax_right = ax.twinx()
    ax_right.set_yscale('log')
    ymin_pct = (0.85 / THEO_FP16_MATRIX) * 100.0
    ymax_pct = (60.0 / THEO_FP16_MATRIX) * 100.0
    ax_right.set_ylim(ymin_pct, ymax_pct)
    ax_right.set_ylabel("% vendor specified matrix peak FLOPS FP16", fontweight='bold', labelpad=12)

    util_ticks = [2, 5, 10, 20, 30, 40, 50, 60, 80, 100]
    ax_right.set_yticks(util_ticks)
    ax_right.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val)}%"))
    ax_right.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax_right.tick_params(axis='y', which='minor', right=False)

    leg = ax.legend(handles=plot_handles, loc='lower left', ncol=1,
                    bbox_to_anchor=(0.012, 0.015),
                    framealpha=0.96, facecolor='white',
                    edgecolor='#888888', fontsize=11.8,
                    title="FP16 Series (Cout=4096, B=1024, M=128)",
                    title_fontsize=12.8)
    leg.get_frame().set_linewidth(1.4)

    fig.suptitle("Scaling Behavior of DG-WMMA at M=128 (up to Cin=40M): Log-Log Invariance\nAMD Radeon RX 9060 XT (Navi 44 / GFX1200 / RDNA4)",
                 fontweight='bold', fontsize=22, y=0.97)

    out_dirs = [
        os.path.join(root_dir, "figures"),
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "scaling_cin_sparsity_m128_40m_loglog.svg"), format='svg')
        fig.savefig(os.path.join(d, "scaling_cin_sparsity_m128_40m_loglog.png"), format='png', dpi=300)
        fig.savefig(os.path.join(d, "scaling_cin_sparsity_m128_40m_loglog.pdf"), format='pdf')

    plt.close(fig)
    print(f"[SUCCESS] M=128 40M Log-Log Plot exported.")

if __name__ == "__main__":
    main()
