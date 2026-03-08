#!/usr/bin/env python3
"""Analyze results/summary.csv: compute stats, run Wilcoxon, save report and boxplot.
"""

from __future__ import annotations

import csv
from pathlib import Path
import math
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt


def read_summary(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(x):
    if x is None:
        return math.nan
    try:
        return float(x)
    except Exception:
        return math.nan


def main():
    repo_root = Path(__file__).resolve().parents[1]
    summary = repo_root / "results" / "summary.csv"
    if not summary.exists():
        raise SystemExit(f"summary not found: {summary}")

    rows = read_summary(summary)

    uni_f1 = [to_float(r.get("unimodal_best_f1")) for r in rows]
    uni_acc = [to_float(r.get("unimodal_best_acc")) for r in rows]
    multi_f1 = [to_float(r.get("multimodal_best_f1")) for r in rows]
    multi_acc = [to_float(r.get("multimodal_best_acc")) for r in rows]

    uni_f1_arr = np.array(uni_f1, dtype=float)
    uni_acc_arr = np.array(uni_acc, dtype=float)
    multi_f1_arr = np.array(multi_f1, dtype=float)
    multi_acc_arr = np.array(multi_acc, dtype=float)

    # stats (ignore NaNs)
    def stats(a):
        mask = ~np.isnan(a)
        if mask.sum() == 0:
            return math.nan, math.nan
        return float(np.nanmean(a)), float(np.nanstd(a, ddof=0))

    uni_f1_mean, uni_f1_std = stats(uni_f1_arr)
    multi_f1_mean, multi_f1_std = stats(multi_f1_arr)
    uni_acc_mean, uni_acc_std = stats(uni_acc_arr)
    multi_acc_mean, multi_acc_std = stats(multi_acc_arr)

    # paired Wilcoxon: only where both present
    def paired_test(a, b):
        mask = ~np.isnan(a) & ~np.isnan(b)
        a2 = a[mask]
        b2 = b[mask]
        if a2.size == 0:
            return None
        try:
            stat, p = wilcoxon(a2, b2)
            return stat, p
        except Exception:
            return None

    f1_test = paired_test(uni_f1_arr, multi_f1_arr)
    acc_test = paired_test(uni_acc_arr, multi_acc_arr)

    # write report
    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.txt"
    with report_path.open("w", encoding="utf-8") as rf:
        rf.write("Summary of unimodal vs multimodal\n")
        rf.write("--------------------------------\n")
        rf.write(f"Unimodal F1: mean={uni_f1_mean:.4f} std={uni_f1_std:.4f}\n")
        rf.write(f"Multimodal F1: mean={multi_f1_mean:.4f} std={multi_f1_std:.4f}\n")
        if f1_test is None:
            rf.write("Wilcoxon F1: not enough paired data or test failed\n")
        else:
            rf.write(f"Wilcoxon F1: stat={f1_test[0]:.4f} p={f1_test[1]:.6e}\n")

        rf.write("\n")
        rf.write(f"Unimodal Acc: mean={uni_acc_mean:.4f} std={uni_acc_std:.4f}\n")
        rf.write(f"Multimodal Acc: mean={multi_acc_mean:.4f} std={multi_acc_std:.4f}\n")
        if acc_test is None:
            rf.write("Wilcoxon Acc: not enough paired data or test failed\n")
        else:
            rf.write(f"Wilcoxon Acc: stat={acc_test[0]:.4f} p={acc_test[1]:.6e}\n")

    # boxplot for F1
    # Use only paired entries where both F1 available
    mask = ~np.isnan(uni_f1_arr) & ~np.isnan(multi_f1_arr)
    uni_plot = uni_f1_arr[mask]
    multi_plot = multi_f1_arr[mask]
    plt.figure(figsize=(6, 6))
    plt.boxplot([uni_plot.tolist(), multi_plot.tolist()], labels=["unimodal", "multimodal"])
    plt.ylabel("Macro F1")
    plt.title("Unimodal vs Multimodal Macro F1")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plot_path = out_dir / "f1_boxplot.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
