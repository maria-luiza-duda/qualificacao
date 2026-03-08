from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterable

import torch
from PIL import Image
from torchvision import transforms as T


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    p = Path(path)
    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            records.append(json.loads(ln))
    return records


def build_label_map(manifest_or_split_items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    species = [r.get("species") for r in manifest_or_split_items if r.get("species") is not None]
    unique = sorted(set(species))
    return {s: i for i, s in enumerate(unique)}


class CampylaspisDataset:
    """Dataset que lê um JSONL (manifest ou split) e fornece seleções por amostra.

    Retorna por item um dict com:
      - `images_paths_sel` (list[str])
      - `text` (str)
      - `has_text` (0/1)
      - `label_id` (int)
      - `species` (str)
    """

    def __init__(
        self,
        jsonl_path: Path,
        label_map: Dict[str, int],
        image_size: int = 224,
        max_images: int = 8,
        mode: str = "image",
    ) -> None:
        self.records = load_jsonl(Path(jsonl_path))
        self.label_map = label_map
        self.image_size = int(image_size)
        self.max_images = int(max_images)
        if mode not in {"image", "text", "multimodal"}:
            raise ValueError("mode must be one of 'image','text','multimodal'")
        self.mode = mode

    def __len__(self) -> int:
        return len(self.records)

    def _read_text(self, text_path: Optional[str]) -> str:
        if not text_path:
            return ""
        p = Path(text_path)
        if not p.exists() or not p.is_file():
            return ""
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return ""

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        species = r.get("species")
        label_id = int(self.label_map.get(species, -1))

        image_paths: List[str] = r.get("image_paths") or []
        # keep only existing files
        image_paths = [p for p in image_paths if Path(p).exists() and Path(p).is_file()]

        # sample up to max_images
        if len(image_paths) > self.max_images:
            image_paths_sel = random.sample(image_paths, k=self.max_images)
        else:
            image_paths_sel = list(image_paths)

        text = ""
        text_path = r.get("text_path") or r.get("text")
        if text_path:
            text = self._read_text(text_path)

        has_text = 1 if text else 0

        return {
            "images_paths_sel": image_paths_sel,
            "text": text or "",
            "has_text": int(has_text),
            "label_id": label_id,
            "species": species,
        }


def make_collate_fn(image_size: int = 224, max_images: int = 8, device: Optional[str] = None):
    """Factory que retorna uma `collate_fn(batch)` para DataLoader.

    Transform: Resize -> CenterCrop -> ToTensor -> Normalize (ImageNet stats).
    Retorna dicionário com: images (B,N,C,H,W), image_mask (B,N), texts (list[str]), has_text (B), labels (B)
    """

    transform = T.Compose([
        T.Resize(int(image_size)),
        T.CenterCrop(int(image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def collate_fn(batch: List[Dict[str, Any]]):
        B = len(batch)
        N = int(max_images)
        C = 3
        H = int(image_size)
        W = int(image_size)

        images = torch.zeros((B, N, C, H, W), dtype=torch.float32)
        image_mask = torch.zeros((B, N), dtype=torch.uint8)

        texts: List[str] = []
        has_text = []
        labels = []

        for i, item in enumerate(batch):
            labels.append(int(item.get("label_id", -1)))
            txt = item.get("text", "")
            texts.append(txt or "")
            has_text.append(int(bool(item.get("has_text", 0))))

            paths = item.get("images_paths_sel", []) or []
            # ensure deterministic order in collate: sort paths
            # (they may be sampled already in dataset)
            for j in range(min(len(paths), N)):
                p = paths[j]
                try:
                    img = Image.open(p).convert("RGB")
                    t = transform(img)
                    if t.shape[0] != C or t.shape[1] != H or t.shape[2] != W:
                        # resize/crop should ensure shape, but guard
                        t = T.Resize((H, W))(img)
                        t = T.ToTensor()(t)
                        t = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t)
                    images[i, j] = t
                    image_mask[i, j] = 1
                except Exception:
                    # skip broken image
                    continue

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        has_text_tensor = torch.tensor(has_text, dtype=torch.uint8)

        if device is not None:
            dev = torch.device(device)
            images = images.to(dev)
            image_mask = image_mask.to(dev)
            labels_tensor = labels_tensor.to(dev)
            has_text_tensor = has_text_tensor.to(dev)

        return {
            "images": images,
            "image_mask": image_mask,
            "texts": texts,
            "has_text": has_text_tensor,
            "labels": labels_tensor,
        }

    return collate_fn


__all__ = ["load_jsonl", "build_label_map", "CampylaspisDataset", "make_collate_fn"]
__all__ = ["load_jsonl", "CampylaspisDataset", "make_collate_fn"]
