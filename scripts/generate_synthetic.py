#!/usr/bin/env python3
"""
Generate synthetic manifest and images.

Reads: manifest.jsonl, train.jsonl, val.jsonl
Writes: synthetic_manifest.jsonl (default) and PNG images into datasets/campylaspis/synthetic/<species>/<id>.png

Each synthetic record will have fields:
- id
- species
- image_path
- is_synthetic: true
- source_text_id
- prompt
- seed
- generator
- quality_scores: {clip_score, classifier_confidence}
- body_part (optional)

The original manifest.jsonl is NOT modified.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any

TRANSPARENT_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAn0B9pY3kAAAAABJRU5ErkJggg=="


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def deterministic_color(seed_str: str) -> (int, int, int):
    h = hashlib.md5(seed_str.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # lighten so not too dark
    return (r | 0x40) & 0xFF, (g | 0x40) & 0xFF, (b | 0x40) & 0xFF


def save_png(path: Path, color: tuple = (200, 200, 200), size: int = 128):
    # Try PIL first, otherwise write a tiny 1x1 PNG (transparent)
    try:
        from PIL import Image
    except Exception:
        img_bytes = base64.b64decode(TRANSPARENT_PNG_B64)
        path.write_bytes(img_bytes)
        return
    img = Image.new("RGB", (size, size), color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def make_id(species: str, idx: int, seed: int) -> str:
    safe_species = species.replace(" ", "_")
    return f"synth_{safe_species}_{seed}_{idx:04d}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True, help="Original manifest.jsonl (read-only)")
    p.add_argument("--train_jsonl", type=Path, required=True)
    p.add_argument("--val_jsonl", type=Path, required=True)
    p.add_argument("--out_manifest", type=Path, default=Path("synthetic_manifest.jsonl"))
    p.add_argument("--out_image_dir", type=Path, default=Path("datasets/campylaspis/synthetic"))
    p.add_argument("--n_per_species", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generator", type=str, default="mock-generator-v1")
    p.add_argument("--prompt_template", type=str, default="Generate an image of a {species}.")
    p.add_argument("--include_body_part", action="store_true", help="Randomly include a body_part field for some examples")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Read inputs (but do not modify them)
    if not args.manifest.exists():
        raise RuntimeError(f"manifest not found: {args.manifest}")
    train_items = read_jsonl(args.train_jsonl)
    val_items = read_jsonl(args.val_jsonl)

    # Collect species from train+val
    species_set = set()
    for it in train_items + val_items:
        sp = it.get("species")
        if sp:
            species_set.add(sp)

    species_list = sorted(species_set)

    out_manifest_path = args.out_manifest
    out_image_dir = args.out_image_dir

    body_parts = ["head", "thorax", "abdomen", "leg", "wing"]

    synth_records: List[Dict[str, Any]] = []

    # Precompute source_text_ids pool (items that have an id or text)
    source_pool = [it.get("id") or it.get("text") for it in (train_items + val_items) if (it.get("id") or it.get("text"))]

    for sp in species_list:
        for i in range(args.n_per_species):
            sid = make_id(sp, i, args.seed)
            rel_img_path = out_image_dir / sp / f"{sid}.png"
            prompt = args.prompt_template.format(species=sp)

            # deterministic pseudo-random quality scores using hash of sid
            h = hashlib.sha256(sid.encode("utf-8") + str(args.seed).encode("utf-8")).hexdigest()
            clip_score = (int(h[0:8], 16) % 10000) / 10000.0
            classifier_confidence = (int(h[8:16], 16) % 10000) / 10000.0

            source_text_id = None
            if source_pool:
                source_text_id = random.choice(source_pool)

            rec = {
                "id": sid,
                "species": sp,
                "image_path": str(rel_img_path),
                "is_synthetic": True,
                "source_text_id": source_text_id,
                "prompt": prompt,
                "seed": args.seed,
                "generator": args.generator,
                "quality_scores": {"clip_score": clip_score, "classifier_confidence": classifier_confidence},
            }

            # optionally add body_part sometimes
            if args.include_body_part and random.random() < 0.4:
                rec["body_part"] = random.choice(body_parts)

            synth_records.append(rec)

            # save image deterministically
            color = deterministic_color(sid)
            img_path = rel_img_path
            img_path.parent.mkdir(parents=True, exist_ok=True)
            save_png(img_path, color=color, size=128)

    # Write manifest lines (do not modify original manifest.jsonl)
    with open(out_manifest_path, "w", encoding="utf-8") as outf:
        for r in synth_records:
            outf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(synth_records)} synthetic records to {out_manifest_path}")
    print(f"Saved images under {out_image_dir}")


if __name__ == "__main__":
    main()
