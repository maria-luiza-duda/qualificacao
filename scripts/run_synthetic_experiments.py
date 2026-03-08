#!/usr/bin/env python3
"""
Run experiments for three data conditions:
 - real_only (baseline)
 - real_plus_synth
 - synth_only (optional ablation)

This script wraps scripts/train.py and creates separate outdirs per condition and seed.
"""

import argparse
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True, type=Path)
    p.add_argument("--val_jsonl", required=True, type=Path)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--label_map_json", type=Path, default=None)
    p.add_argument("--synth_jsonl", type=Path, default=None)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--backbone_name", required=True)
    p.add_argument("--text_model_name", default=None)
    p.add_argument("--model_type", choices=["unimodal", "multimodal"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--synth_keep_frac", type=float, default=1.0)
    p.add_argument("--conditions", nargs="*", choices=["real_only", "real_plus_synth", "synth_only"], default=["real_only", "real_plus_synth"], help="Which conditions to run")
    return p.parse_args()


def call_train(cmd_args):
    print("Running:", " ".join(str(x) for x in cmd_args))
    subprocess.check_call([str(x) for x in cmd_args])


def main():
    args = parse_args()
    train_py = Path(__file__).parent / "train.py"

    for cond in args.conditions:
        cmd = ["python3", str(train_py),
               "--model_type", args.model_type,
               "--train_jsonl", str(args.train_jsonl),
               "--val_jsonl", str(args.val_jsonl),
               "--outdir", str(args.outdir),
               "--backbone_name", args.backbone_name,
               "--epochs", str(args.epochs),
               "--batch_size", str(args.batch_size),
               "--seed", str(args.seed),
               "--data_condition", cond,
               "--synth_keep_frac", str(args.synth_keep_frac)
               ]

        if args.manifest is not None:
            cmd += ["--manifest", str(args.manifest)]
        if args.label_map_json is not None:
            cmd += ["--label_map_json", str(args.label_map_json)]
        if args.text_model_name is not None:
            cmd += ["--text_model_name", args.text_model_name]
        if args.synth_jsonl is not None:
            cmd += ["--synth_jsonl", str(args.synth_jsonl)]

        call_train(cmd)


if __name__ == "__main__":
    main()
