#!/usr/bin/env python3
"""Gerar `manifest.jsonl` a partir de um diretório processado.

Entrada: --processed_root contendo subpastas por species, cada uma podendo ter
images/ e/ou text/. Gera `manifest.jsonl` em `processed_root` com uma linha por
espécie contendo: species, image_paths (lista), text_path (string ou null),
has_image (0/1), has_text (0/1), num_images.

Aceita imagens com extensões jpg/png/tif (case-insensitive). Loga resumo por
espécie e totais.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional


ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_image_files(images_dir: Path) -> List[str]:
    if not images_dir.exists() or not images_dir.is_dir():
        return []
    files = []
    for p in sorted(images_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in ALLOWED_IMAGE_EXTS:
            files.append(str(p.resolve()))
    return files


def find_text_file(text_dir: Path) -> Optional[str]:
    if not text_dir.exists() or not text_dir.is_dir():
        return None
    # Accept any file in text/ and pick the first (sorted) one
    files = [p for p in sorted(text_dir.iterdir()) if p.is_file()]
    return str(files[0].resolve()) if files else None


def process_root(processed_root: Path, out_name: str = "manifest.jsonl") -> Path:
    processed_root = processed_root.resolve()
    manifest_path = processed_root / out_name

    species_dirs = [p for p in sorted(processed_root.iterdir()) if p.is_dir()]

    totals = {
        "species_count": 0,
        "species_with_image": 0,
        "species_with_text": 0,
        "total_images": 0,
    }

    with manifest_path.open("w", encoding="utf-8") as out_f:
        for species_dir in species_dirs:
            species_name = species_dir.name
            images_dir = species_dir / "images"
            text_dir = species_dir / "text"

            image_paths = find_image_files(images_dir)
            text_path = find_text_file(text_dir)

            has_image = 1 if image_paths else 0
            has_text = 1 if text_path else 0
            num_images = len(image_paths)

            totals["species_count"] += 1
            totals["species_with_image"] += has_image
            totals["species_with_text"] += has_text
            totals["total_images"] += num_images

            record = {
                "species": species_name,
                "image_paths": image_paths,
                "text_path": text_path if text_path is not None else None,
                "has_image": has_image,
                "has_text": has_text,
                "num_images": num_images,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logging.info(
                "Species '%s': images=%d, has_text=%s, text_path=%s",
                species_name,
                num_images,
                "yes" if has_text else "no",
                text_path or "-",
            )

    logging.info(
        "Manifest written to %s — species=%d, species_with_image=%d, species_with_text=%d, total_images=%d",
        manifest_path,
        totals["species_count"],
        totals["species_with_image"],
        totals["species_with_text"],
        totals["total_images"],
    )

    return manifest_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Criar manifest.jsonl a partir de um diretório processado por espécie"
    )
    p.add_argument("--processed_root", required=True, type=Path, help="Caminho para o diretório root processado")
    p.add_argument("--output", default="manifest.jsonl", help="Nome do arquivo de saída dentro de processed_root")
    p.add_argument("--verbose", action="store_true", help="Ativa logging DEBUG")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    processed_root: Path = args.processed_root
    if not processed_root.exists():
        logging.error("processed_root não existe: %s", processed_root)
        return 2
    if not processed_root.is_dir():
        logging.error("processed_root não é um diretório: %s", processed_root)
        return 2

    try:
        process_root(processed_root, out_name=args.output)
    except Exception as e:
        logging.exception("Erro ao gerar o manifest: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
