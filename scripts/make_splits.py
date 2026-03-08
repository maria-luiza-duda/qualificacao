#!/usr/bin/env python3
"""Gerar splits estratificados por espécie a partir de um manifest.jsonl.

Lê um `manifest.jsonl` (uma linha JSON por espécie) e cria `splits/` com k
folds. Para cada fold salva:
 - `train_all.jsonl`, `val_all.jsonl`
 - `train_image.jsonl`, `val_image.jsonl`  (apenas registros com has_image==1)
 - `train_text.jsonl`, `val_text.jsonl`    (apenas registros com has_text==1)
 - `train_multimodal.jsonl`, `val_multimodal.jsonl` (registros com has_image==1 and has_text==1)

Loga contagens por fold.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List


def read_manifest(manifest_path: Path) -> List[Dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    records = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            records.append(json.loads(ln))
    return records


def write_jsonl(path: Path, records: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_folds(species_list: List[str], k: int, seed: int = 42) -> List[List[str]]:
    rnd = random.Random(seed)
    items = list(species_list)
    rnd.shuffle(items)
    folds: List[List[str]] = [[] for _ in range(k)]
    for i, s in enumerate(items):
        folds[i % k].append(s)
    return folds


def build_index(records: List[Dict]) -> Dict[str, Dict]:
    idx = {}
    for r in records:
        species = r.get("species")
        if species is None:
            continue
        idx[species] = r
    return idx


def split_and_save(records: List[Dict], manifest_path: Path, k: int, seed: int):
    out_root = manifest_path.parent / "splits"
    idx = build_index(records)
    species_list = sorted(idx.keys())
    folds = make_folds(species_list, k=k, seed=seed)

    for fold_idx, val_species in enumerate(folds, start=1):
        val_set = set(val_species)
        train_records = [r for s, r in idx.items() if s not in val_set]
        val_records = [idx[s] for s in val_species]

        # subsets
        train_image = [r for r in train_records if int(r.get("has_image", 0)) == 1]
        val_image = [r for r in val_records if int(r.get("has_image", 0)) == 1]

        train_text = [r for r in train_records if int(r.get("has_text", 0)) == 1]
        val_text = [r for r in val_records if int(r.get("has_text", 0)) == 1]

        train_multimodal = [r for r in train_records if int(r.get("has_image", 0)) == 1 and int(r.get("has_text", 0)) == 1]
        val_multimodal = [r for r in val_records if int(r.get("has_image", 0)) == 1 and int(r.get("has_text", 0)) == 1]

        fold_dir = out_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        write_jsonl(fold_dir / "train_all.jsonl", train_records)
        write_jsonl(fold_dir / "val_all.jsonl", val_records)

        write_jsonl(fold_dir / "train_image.jsonl", train_image)
        write_jsonl(fold_dir / "val_image.jsonl", val_image)

        write_jsonl(fold_dir / "train_text.jsonl", train_text)
        write_jsonl(fold_dir / "val_text.jsonl", val_text)

        write_jsonl(fold_dir / "train_multimodal.jsonl", train_multimodal)
        write_jsonl(fold_dir / "val_multimodal.jsonl", val_multimodal)

        logging.info(
            "Fold %d: train_all=%d, val_all=%d, train_image=%d, val_image=%d, train_text=%d, val_text=%d, train_multimodal=%d, val_multimodal=%d",
            fold_idx,
            len(train_records),
            len(val_records),
            len(train_image),
            len(val_image),
            len(train_text),
            len(val_text),
            len(train_multimodal),
            len(val_multimodal),
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Criar k-fold splits a partir de manifest.jsonl")
    p.add_argument("--manifest", required=True, type=Path, help="Caminho para manifest.jsonl")
    p.add_argument("--k", type=int, default=5, help="Número de folds (default: 5)")
    p.add_argument("--seed", type=int, default=42, help="Seed para shuffle (default: 42)")
    p.add_argument("--verbose", action="store_true", help="Ativa logging DEBUG")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    try:
        records = read_manifest(args.manifest)
    except Exception as e:
        logging.error("Erro lendo manifest: %s", e)
        return 2

    if not records:
        logging.error("Nenhum registro encontrado no manifest: %s", args.manifest)
        return 2

    split_and_save(records, args.manifest, k=args.k, seed=args.seed)
    logging.info("Splits gerados em: %s", str(args.manifest.parent / "splits"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
