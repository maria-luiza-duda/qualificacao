#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer

from campylaspis.data import CampylaspisDataset, build_label_map, make_collate_fn, load_jsonl
from campylaspis.models_unimodal import build_unimodal
import timm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["unimodal", "multimodal"], required=True)
    p.add_argument("--train_jsonl", required=True, type=Path)
    p.add_argument("--val_jsonl", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--backbone_name", required=True)
    p.add_argument("--text_model_name", default=None)
    p.add_argument("--fusion", choices=["concat", "gated"], default="concat")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_images", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_condition", choices=["real_only", "real_plus_synth", "synth_only"], default="real_only", help="Which data condition to run: baseline or include synthetic")
    p.add_argument("--synth_jsonl", type=Path, default=None, help="Path to synthetic examples jsonl (each entry same format as train.jsonl)")
    p.add_argument("--synth_keep_frac", type=float, default=1.0, help="Fraction of synthetic examples to keep (for simple filtering/ablation)")
    p.add_argument("--manifest", type=Path, default=None, help="Path to manifest.jsonl to build a global label_map")
    p.add_argument("--label_map_json", type=Path, default=None, help="Optional JSON file with precomputed label_map {species: id}")
    p.add_argument("--freeze_text_encoder", action="store_true", help="If set and multimodal, freeze text encoder weights and set it to eval()")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultimodalModel(nn.Module):
    def __init__(self, backbone_name: str, text_model_name: str, fusion: str, num_classes: int, pretrained: bool = True, fusion_dim: int = 512):
        super().__init__()
        # image backbone returns feature vector per image
        self.img_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.img_dim = getattr(self.img_backbone, "num_features", None)
        if self.img_dim is None:
            # infer
            with torch.no_grad():
                o = self.img_backbone(torch.zeros(1, 3, 224, 224))
            self.img_dim = o.shape[1]

        # text encoder
        self.text_model = AutoModel.from_pretrained(text_model_name)
        # text dim
        self.text_dim = getattr(self.text_model.config, "hidden_size", None)
        if self.text_dim is None:
            # try running dummy
            with torch.no_grad():
                inp = torch.zeros(1, 1, dtype=torch.long)
                o = self.text_model(inputs_embeds=torch.zeros(1, 1, self.text_model.config.hidden_size))
            self.text_dim = o.last_hidden_state.shape[-1]

        self.fusion = fusion
        # projection dimension for fusion
        self.fusion_dim = int(fusion_dim)
        # Projection layers to align image/text dims
        self.img_proj = nn.Linear(self.img_dim, self.fusion_dim)
        self.txt_proj = nn.Linear(self.text_dim, self.fusion_dim)

        if fusion == "concat":
            # concat of projected vectors
            self.classifier = nn.Linear(self.fusion_dim * 2, num_classes)
            self.gate = None
        else:  # gated
            # compute gate from elementwise sum of projected vectors
            self.gate = nn.Linear(self.fusion_dim, 1)
            self.classifier = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, images: torch.Tensor, image_mask: torch.Tensor, input_ids=None, attention_mask=None, has_text=None):
        # images: (B, N, C, H, W)
        B, N, C, H, W = images.shape
        imgs = images.view(B * N, C, H, W)
        feats = self.img_backbone(imgs)
        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])
        feats = feats.view(B, N, -1)  # (B,N,Dimg)
        mask = image_mask.to(dtype=feats.dtype).unsqueeze(-1)
        sums = (feats * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        img_emb = sums / counts  # (B, Dimg)

        if input_ids is None:
            txt_emb = torch.zeros((B, self.text_dim), device=img_emb.device, dtype=img_emb.dtype)
        else:
            out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            # use pooler_output if available
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                txt_emb = out.pooler_output
            else:
                # mean pooling
                last = out.last_hidden_state  # (B, L, H)
                att = attention_mask.unsqueeze(-1).to(dtype=last.dtype)
                txt_emb = (last * att).sum(dim=1) / (att.sum(dim=1).clamp(min=1.0))

        # If has_text is provided, zero-out text embeddings for samples without text
        if has_text is not None:
            ht = has_text.view(B, -1).to(dtype=txt_emb.dtype)
            txt_emb = txt_emb * ht

        # Project image/text embeddings to fusion_dim
        img_proj = self.img_proj(img_emb)
        txt_proj = self.txt_proj(txt_emb)

        if self.fusion == "concat":
            fused = torch.cat([img_proj, txt_proj], dim=1)
            logits = self.classifier(fused)
        else:
            # gated fusion: compute gate from elementwise sum of projected vectors
            summed = img_proj + txt_proj
            alpha = torch.sigmoid(self.gate(summed))  # (B,1)
            fused = alpha * img_proj + (1 - alpha) * txt_proj
            logits = self.classifier(fused)

        return logits


def evaluate(model, dataloader, device, tokenizer=None):
    model.eval()
    ys = []
    preds = []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            image_mask = batch["image_mask"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["texts"]

            if tokenizer is not None:
                enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
                has_text_batch = batch.get("has_text")
                if has_text_batch is not None:
                    has_text_batch = has_text_batch.to(device)
                logits = model(images, image_mask, input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], has_text=has_text_batch)
            else:
                logits = model(images, image_mask)

            # model may return (logits, aux); ensure logits is a tensor
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            count += labels.size(0)

            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y = labels.cpu().numpy()
            preds.extend(pred.tolist())
            ys.extend(y.tolist())

    val_loss = total_loss / max(1, count)
    macro_f1 = f1_score(ys, preds, average="macro", zero_division=0)
    acc = np.mean(np.array(ys) == np.array(preds))
    return val_loss, macro_f1, acc


def main():
    args = parse_args()
    # Create nested run dir for reproducibility: separate folder per condition and seed
    base_out = args.outdir
    run_dir = base_out / f"{args.data_condition}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.outdir = run_dir
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Save args for reproducibility
    with open(args.outdir / "args.json", "w", encoding="utf-8") as af:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, af, ensure_ascii=False, indent=2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device %s", device)

    # If CUDA is available, clear cache and reset peak stats to avoid stale allocations
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            logging.exception("torch.cuda.empty_cache() failed")
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            logging.exception("torch.cuda.reset_peak_memory_stats() failed")
        try:
            dev_name = torch.cuda.get_device_name(0)
            mem_info = torch.cuda.mem_get_info()
            logging.info("CUDA device: %s", dev_name)
            logging.info("CUDA mem_get_info (free, total): %s", str(mem_info))
        except Exception:
            logging.exception("Failed to query CUDA device name or memory info")

    set_seed(args.seed)

    # load train/val items
    train_items = load_jsonl(args.train_jsonl)
    val_items = load_jsonl(args.val_jsonl)

    # Optionally incorporate synthetic items into the training set according to data_condition
    if args.data_condition != "real_only":
        if args.synth_jsonl is None:
            raise RuntimeError("--synth_jsonl must be provided when using synthetic conditions")
        if not args.synth_jsonl.exists():
            raise RuntimeError(f"synth_jsonl not found: {args.synth_jsonl}")
        synth_items = load_jsonl(args.synth_jsonl)
        # simple filtering / ablation by sampling fraction
        if args.synth_keep_frac < 1.0:
            import random as _rnd
            _rnd.seed(args.seed)
            k = int(len(synth_items) * float(args.synth_keep_frac))
            synth_items = _rnd.sample(synth_items, k)

        if args.data_condition == "real_plus_synth":
            train_items = list(train_items) + list(synth_items)
        else:  # synth_only
            train_items = list(synth_items)

    # build or load global label_map
    if args.label_map_json is not None:
        if not args.label_map_json.exists():
            raise RuntimeError(f"label_map_json not found: {args.label_map_json}")
        with open(args.label_map_json, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        # ensure keys are strings and values ints
        label_map = {str(k): int(v) for k, v in label_map.items()}
    elif args.manifest is not None:
        if not args.manifest.exists():
            raise RuntimeError(f"manifest not found: {args.manifest}")
        manifest_items = load_jsonl(args.manifest)
        label_map = build_label_map(manifest_items)
    else:
        raise RuntimeError("Either --manifest or --label_map_json must be provided to build/load a global label_map")

    num_classes = len(label_map)

    # validate that train/val species appear in the global label_map
    train_species = {r.get("species") for r in train_items if r.get("species")}
    val_species = {r.get("species") for r in val_items if r.get("species")}
    missing = (train_species | val_species) - set(label_map.keys())
    if missing:
        raise RuntimeError(f"Species in train/val not found in label_map: {sorted(list(missing))}")

    # datasets and loaders
    collate = make_collate_fn(image_size=args.image_size, max_images=args.max_images, device=None)
    # Create dataset from in-memory train_items by writing a temporary jsonl when necessary
    # The CampylaspisDataset accepts a path; to avoid changing its API we will write a temp file under outdir
    train_jsonl_path = args.outdir / "_train_items.jsonl"
    with open(train_jsonl_path, "w", encoding="utf-8") as tf:
        for it in train_items:
            tf.write(json.dumps(it, ensure_ascii=False) + "\n")

    train_ds = CampylaspisDataset(train_jsonl_path, label_map, image_size=args.image_size, max_images=args.max_images)
    val_ds = CampylaspisDataset(args.val_jsonl, label_map, image_size=args.image_size, max_images=args.max_images)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=4)

    # build model
    if args.model_type == "unimodal":
        model = build_unimodal(args.backbone_name, num_classes=num_classes, pretrained=True)
        model = model.to(device)
        tokenizer = None
    else:
        if not args.text_model_name:
            raise RuntimeError("--text_model_name is required for multimodal")
        model = MultimodalModel(args.backbone_name, args.text_model_name, fusion=args.fusion, num_classes=num_classes, pretrained=True)
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

    # Optionally freeze text encoder for multimodal
    if args.model_type == "multimodal" and getattr(args, "freeze_text_encoder", False):
        if hasattr(model, "text_model") and model.text_model is not None:
            logging.info("Freezing text encoder parameters and setting eval()")
            for p in model.text_model.parameters():
                p.requires_grad = False
            try:
                model.text_model.eval()
            except Exception:
                logging.exception("Failed to set text_model to eval()")
        else:
            logging.warning("--freeze_text_encoder set but model has no attribute text_model")

    # Build optimizer with only parameters that require grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for optimizer (all parameters have requires_grad=False)")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    log_path = args.outdir / "train_log.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for batch in train_loader:
            images = batch["images"].to(device)
            image_mask = batch["image_mask"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["texts"]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                if tokenizer is not None:
                    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
                    has_text_batch = batch.get("has_text")
                    if has_text_batch is not None:
                        has_text_batch = has_text_batch.to(device)
                    logits = model(images, image_mask, input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], has_text=has_text_batch)
                else:
                    logits = model(images, image_mask)

                # model may return (logits, aux); ensure logits is a tensor
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            running_loss += float(loss.item()) * bs
            n_seen += bs

        train_loss = running_loss / max(1, n_seen)

        # eval
        val_loss, val_f1, val_acc = evaluate(model, val_loader, device, tokenizer)

        # save best
        saved = False
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = args.outdir / "best.pt"
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "val_f1": val_f1}, save_path)
            saved = True

        # log epoch
        info = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": val_f1, "val_acc": val_acc, "saved_best": saved}
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(info, ensure_ascii=False) + "\n")

        logging.info("Epoch %d: train_loss=%.4f val_loss=%.4f val_f1=%.4f val_acc=%.4f saved=%s", epoch, train_loss, val_loss, val_f1, val_acc, saved)

    # final save last
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": args.epochs}, args.outdir / "last.pt")


if __name__ == "__main__":
    main()
