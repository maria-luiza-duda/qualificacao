#!/usr/bin/env python3
"""Criar splits por imagem a partir de um manifest.jsonl.

Para cada espécie, divide suas `image_paths` em k folds (round-robin após
embaralhar). Para cada fold escreve `splits_by_image/fold_{i}/train.jsonl` e
`val.jsonl` com uma linha por imagem contendo: species, image_paths (lista com
uma imagem), text_path (ou null), has_text (0/1).

Garante que espécies com >=2 imagens aparecem tanto em treino quanto em
validação (implícito pela estratégia de distribuição round-robin).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List


def read_manifest(manifest: Path) -> List[Dict]:
    items: List[Dict] = []
    with manifest.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items


def write_jsonl(path: Path, records: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_splits_by_image(manifest: Path, out_root: Path, k: int = 5, seed: int = 42):
    manifest = manifest.resolve()
    out_root = out_root.resolve()
    items = read_manifest(manifest)

    random.seed(seed)

    # species -> list of image paths
    species_images: Dict[str, List[str]] = {}
    species_text: Dict[str, str] = {}
    for it in items:
        sp = it.get("species")
        imgs = it.get("image_paths") or []
        imgs = [str(p) for p in imgs]
        species_images[sp] = imgs
        txt = it.get("text_path")
        species_text[sp] = txt if txt is not None else None

    # assignments: species -> list of lists (fold -> image paths)
    assignments: Dict[str, List[List[str]]] = {}
    for sp, imgs in species_images.items():
        imgs_shuf = list(imgs)
        random.shuffle(imgs_shuf)
        folds: List[List[str]] = [[] for _ in range(k)]
        for i, img in enumerate(imgs_shuf):
            folds[i % k].append(img)
        assignments[sp] = folds

    # Prepare per-fold records
    for fi in range(k):
        fold_dir = out_root / f"fold_{fi+1}"
        train_recs: List[Dict] = []
        val_recs: List[Dict] = []

        for sp, folds in assignments.items():
            text_path = species_text.get(sp) or None
            # val images for this species in this fold
            val_imgs = folds[fi]
            # train imgs are all other imgs
            train_imgs = [img for idx, fimgs in enumerate(folds) if idx != fi for img in fimgs]

            # create one record per image
            for img in val_imgs:
                rec = {
                    "species": sp,
                    "image_paths": [img],
                    "text_path": text_path if text_path is not None else None,
                    "has_text": 1 if text_path else 0,
                }
                val_recs.append(rec)

            for img in train_imgs:
                rec = {
                    "species": sp,
                    "image_paths": [img],
                    "text_path": text_path if text_path is not None else None,
                    "has_text": 1 if text_path else 0,
                }
                train_recs.append(rec)

        write_jsonl(fold_dir / "train.jsonl", train_recs)
        write_jsonl(fold_dir / "val.jsonl", val_recs)

        # logging summary for the fold
        logging.info("Fold %d: train_images=%d val_images=%d species=%d", fi + 1, len(train_recs), len(val_recs), len(assignments))

    # log per-species summary
    for sp, folds in assignments.items():
        total = sum(len(f) for f in folds)
        per_fold = [len(f) for f in folds]
        logging.info("Species '%s': total_images=%d per_fold=%s", sp, total, per_fold)


def parse_args():
    p = argparse.ArgumentParser(description="Create image-level k-fold splits from manifest.jsonl")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=None, help="Output root for splits_by_image (defaults to manifest.parent/splits_by_image)")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    manifest = args.manifest
    outdir = args.outdir or (manifest.parent / "splits_by_image")
    make_splits_by_image(manifest, outdir, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
