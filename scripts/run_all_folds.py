#!/usr/bin/env python3
"""Run training for folds 1..5 (unimodal and multimodal) and summarize results.

For each fold i the script expects:
  processed/splits_by_image/fold_i/train.jsonl
  processed/splits_by_image/fold_i/val.jsonl

It runs two trainings per fold:
  1) unimodal (resnet18)
  2) multimodal (resnet18 + distilbert..., gated, freeze_text_encoder)

Outputs are written to `runs/unimodal_by_image/fold_i` and
`runs/multimodal_by_image/fold_i`.

After each training the script reads `train_log.jsonl` and extracts the best
`val_macro_f1` (and corresponding `val_acc`). At the end writes
`results/summary.csv` with columns:
  fold, unimodal_best_f1, unimodal_best_acc, multimodal_best_f1, multimodal_best_acc, delta_f1, delta_acc
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List


ROOT = Path(__file__).resolve().parents[1]


def run_training(cmd: List[str], env: Dict[str, str] = None) -> int:
    print("Running:", " ".join(cmd))
    try:
        res = subprocess.run(cmd, check=False, env=env)
        return res.returncode
    except Exception as e:
        print("Training call failed:", e)
        return 1


def read_best_from_log(log_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not log_path.exists():
        return None, None
    best_f1 = None
    best_acc = None
    best_epoch = None
    with log_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            f1 = rec.get("val_macro_f1")
            acc = rec.get("val_acc")
            if f1 is None:
                continue
            try:
                f1 = float(f1)
            except Exception:
                continue
            if best_f1 is None or f1 > best_f1:
                best_f1 = f1
                best_acc = float(acc) if acc is not None else None
                best_epoch = rec.get("epoch")
    return best_f1, best_acc


def main():
    processed_root = Path("/home/maria-luiza-duda/datasets/campylaspis/processed")
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.csv"
    rows = []

    # Ensure PYTHONPATH points to src
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    for fold in range(1, 6):
        print(f"\n=== Fold {fold} ===")
        split_dir = processed_root / "splits_by_image" / f"fold_{fold}"
        train_json = split_dir / "train.jsonl"
        val_json = split_dir / "val.jsonl"

        unimodal_out = ROOT / "runs" / "unimodal_by_image" / f"fold_{fold}"
        multimodal_out = ROOT / "runs" / "multimodal_by_image" / f"fold_{fold}"
        unimodal_out.mkdir(parents=True, exist_ok=True)
        multimodal_out.mkdir(parents=True, exist_ok=True)

        # Common args
        common = [sys.executable, str(ROOT / "scripts" / "train.py"), "--manifest", str(processed_root / "manifest.jsonl"), "--image_size", "224"]

        # 1) unimodal
        cmd_uni = common + [
            "--model_type", "unimodal",
            "--train_jsonl", str(train_json),
            "--val_jsonl", str(val_json),
            "--outdir", str(unimodal_out),
            "--backbone_name", "resnet18",
            "--epochs", "10",
            "--batch_size", "4",
            "--lr", "1e-4",
            "--max_images", "1",
        ]

        rc = run_training(cmd_uni, env=env)
        if rc != 0:
            print(f"Unimodal training fold {fold} exited with code {rc}")

        best_uni_f1, best_uni_acc = read_best_from_log(unimodal_out / "train_log.jsonl")

        # 2) multimodal
        cmd_multi = common + [
            "--model_type", "multimodal",
            "--train_jsonl", str(train_json),
            "--val_jsonl", str(val_json),
            "--outdir", str(multimodal_out),
            "--backbone_name", "resnet18",
            "--text_model_name", "distilbert-base-multilingual-cased",
            "--fusion", "gated",
            "--freeze_text_encoder",
            "--epochs", "10",
            "--batch_size", "2",
            "--lr", "1e-4",
            "--max_images", "1",
        ]

        rc = run_training(cmd_multi, env=env)
        if rc != 0:
            print(f"Multimodal training fold {fold} exited with code {rc}")

        best_multi_f1, best_multi_acc = read_best_from_log(multimodal_out / "train_log.jsonl")

        # compute deltas
        delta_f1 = None
        delta_acc = None
        if best_uni_f1 is not None and best_multi_f1 is not None:
            delta_f1 = best_multi_f1 - best_uni_f1
        if best_uni_acc is not None and best_multi_acc is not None:
            delta_acc = best_multi_acc - best_uni_acc

        rows.append({
            "fold": fold,
            "unimodal_best_f1": best_uni_f1,
            "unimodal_best_acc": best_uni_acc,
            "multimodal_best_f1": best_multi_f1,
            "multimodal_best_acc": best_multi_acc,
            "delta_f1": delta_f1,
            "delta_acc": delta_acc,
        })

    # write CSV
    fieldnames = ["fold", "unimodal_best_f1", "unimodal_best_acc", "multimodal_best_f1", "multimodal_best_acc", "delta_f1", "delta_acc"]
    with summary_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Summary written to", summary_path)


if __name__ == "__main__":
    main()
