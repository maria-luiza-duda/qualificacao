#!/usr/bin/env python3
"""
CPU-based specimen photograph generation from illustration + taxonomy description.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from campylaspis.generation import MorphologyParser
from campylaspis.generation.morphology_parser import build_morphology_prompt
from campylaspis.generation.diffusion_renderer import MorphologyGuidedRenderer


def infer_species_from_illustration_path(illustration_path: str) -> Optional[str]:
    path = Path(illustration_path)
    parts = path.parts
    try:
        processed_idx = parts.index("processed")
        if processed_idx + 1 < len(parts):
            return parts[processed_idx + 1]
    except ValueError:
        return None
    return None


def load_species_description_text(species: str, dataset_root: str) -> Tuple[str, str]:
    text_dir = Path(dataset_root) / "processed" / species / "text"
    if not text_dir.exists():
        raise FileNotFoundError(f"Species text directory not found: {text_dir}")

    txt_files = sorted(text_dir.glob("*.txt"))
    if len(txt_files) == 0:
        raise FileNotFoundError(f"No .txt files found in: {text_dir}")
    if len(txt_files) > 1:
        raise ValueError(f"Multiple .txt files found in {text_dir}: {[f.name for f in txt_files]}")

    text_file = txt_files[0]
    return text_file.read_text(encoding="utf-8").strip(), str(text_file)


def build_prompt_from_description(
    description_text: str,
    illustration_path: str,
    species_name: str,
    max_prompt_tokens: int = 62,
) -> Tuple[str, Dict[str, Any]]:
    prompt_info: Dict[str, Any] = {
        "prompt_builder": "fallback_raw_description",
        "token_count": None,
        "truncated": None,
    }

    try:
        parser = MorphologyParser()
        constraints = parser.parse_taxonomic_description(description_text)
        if constraints:
            prompt_result = build_morphology_prompt(
                species_name=species_name,
                constraints=constraints,
                illustration_path=illustration_path,
                max_clip_tokens=max_prompt_tokens,
                return_metadata=True,
            )
            prompt = prompt_result["prompt"]
            prompt_info = {
                "prompt_builder": "build_morphology_prompt",
                "token_count": prompt_result.get("token_count"),
                "truncated": prompt_result.get("truncated", False),
                "included_body_parts": prompt_result.get("included_body_parts", []),
                "dropped_body_parts": prompt_result.get("dropped_body_parts", []),
                "compression_strategy": prompt_result.get("compression_strategy", "none"),
            }
            return prompt, prompt_info
    except Exception as exc:
        prompt_info["prompt_builder_error"] = str(exc)

    fallback_prompt = (
        f"{species_name} lateral view. "
        "Exact illustration geometry: contour, appendages, segmentation. "
        "Preserve exact body proportions and segment ratios from the reference illustration. "
        f"Morphology: {description_text} "
        "Realistic exoskeleton texture, natural lighting, corporeal biological specimen."
    )
    return fallback_prompt, prompt_info


def compute_structure_iou(reference_image: Image.Image, generated_image: Image.Image) -> float:
    """Compute edge-overlap IoU as a proxy for morphology/proportion preservation."""
    try:
        import cv2
        import numpy as np

        ref = np.array(reference_image.convert("L"))
        gen = np.array(generated_image.convert("L"))

        if ref.shape != gen.shape:
            gen = cv2.resize(gen, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        ref_edges = cv2.Canny(ref, 50, 150) > 0
        gen_edges = cv2.Canny(gen, 50, 150) > 0

        intersection = int((ref_edges & gen_edges).sum())
        union = int((ref_edges | gen_edges).sum())
        if union == 0:
            return 0.0
        return float(intersection / union)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate high-quality specimen photographs on CPU from illustration + species description"
    )
    parser.add_argument("--illustration", type=str, required=True, help="Path to scientific illustration")
    parser.add_argument("--description", type=str, default=None, help="Taxonomic description text")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/maria-luiza-duda/datasets/campylaspis",
        help="Dataset root used for auto-loading text when --description is not provided",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--appearance-prior", type=str, default=None, help="Appearance-prior photo directory")
    parser.add_argument("--num-images", type=int, default=2, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--guidance-scale", type=float, default=4.5, help="Classifier-free guidance scale")
    parser.add_argument("--num-steps", type=int, default=30, help="Diffusion inference steps")
    parser.add_argument("--strength", type=float, default=0.35, help="img2img denoising strength (lower preserves geometry)")
    parser.add_argument("--controlnet-scale", type=float, default=1.80, help="ControlNet conditioning scale (higher preserves structure)")
    parser.add_argument("--image-size", type=int, default=512, help="Output width/height")
    parser.add_argument(
        "--morph-prompt-max-tokens",
        type=int,
        default=62,
        help="Token budget reserved for morphology prompt before appearance-prior augmentation",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable safety checker (default: disabled)",
    )
    parser.add_argument(
        "--stable-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable numerically conservative mode (default: off for better quality)",
    )
    parser.add_argument(
        "--strict-morphology",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce conservative parameters to preserve proportions/silhouette (default: on)",
    )
    parser.add_argument(
        "--min-edge-iou",
        type=float,
        default=0.20,
        help="Minimum structure IoU threshold for accepting generated images in strict mode",
    )
    parser.add_argument(
        "--reject-low-structure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop images that fail structure IoU threshold in strict mode (default: off - save all for visual review)",
    )

    args = parser.parse_args()

    effective_strength = args.strength
    effective_controlnet_scale = args.controlnet_scale
    effective_guidance_scale = args.guidance_scale

    if args.strict_morphology:
        max_strength = 0.45  # Allow significant texture transformation while preserving structure
        max_guidance = 5.0   # Moderate guidance: balance between ControlNet and prompt
        min_controlnet = 1.7 # Keep structure lock strong but not excessive
        if effective_strength > max_strength:
            print(f"⚠️  strict-morphology: limiting strength {effective_strength:.3f} -> {max_strength:.3f}")
            effective_strength = max_strength
        if effective_controlnet_scale < min_controlnet:
            print(f"⚠️  strict-morphology: increasing controlnet-scale {effective_controlnet_scale:.3f} -> {min_controlnet:.3f}")
            effective_controlnet_scale = min_controlnet
        if effective_guidance_scale > max_guidance:
            print(f"⚠️  strict-morphology: limiting guidance-scale {effective_guidance_scale:.3f} -> {max_guidance:.1f}")
            effective_guidance_scale = max_guidance

    illustration_path = Path(args.illustration)
    if not illustration_path.exists():
        raise FileNotFoundError(f"Illustration not found: {illustration_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inferred_species = infer_species_from_illustration_path(args.illustration)
    description_text = args.description
    description_source = "cli"

    if description_text is None and inferred_species is not None:
        try:
            description_text, description_path = load_species_description_text(
                inferred_species,
                dataset_root=args.dataset_root,
            )
            description_source = description_path
            print(f"✅ Loaded taxonomic description for species: {inferred_species}")
        except Exception as exc:
            print(f"⚠️  Could not auto-load description for species '{inferred_species}': {exc}")

    if not description_text:
        description_text = "Morphologically accurate crustacean specimen consistent with the reference illustration."
        description_source = "fallback"

    species_name = f"Campylaspis {inferred_species}" if inferred_species else "Campylaspis specimen"
    prompt, prompt_info = build_prompt_from_description(
        description_text=description_text,
        illustration_path=args.illustration,
        species_name=species_name,
        max_prompt_tokens=args.morph_prompt_max_tokens,
    )

    print("\n" + "=" * 72)
    print("🧬 CPU SPECIMEN PHOTOGRAPH GENERATION")
    print("=" * 72)
    print(f"📍 Illustration: {args.illustration}")
    print(f"🧪 Species: {species_name}")
    print(f"📝 Description source: {description_source}")
    print(f"🎯 Prompt builder: {prompt_info.get('prompt_builder')}")
    print(
        f"⚙️  Steps={args.num_steps} | Guidance={effective_guidance_scale} | "
        f"Strength={effective_strength} | ControlNet={effective_controlnet_scale}"
    )
    print(f"🧷 Strict morphology lock={args.strict_morphology}")
    print(f"🖥️  Device=cpu | Stable mode={args.stable_mode}")
    print("=" * 72 + "\n")

    try:
        renderer = MorphologyGuidedRenderer(
            device="cpu",
            appearance_prior_dir=args.appearance_prior,
            disable_safety_checker=args.disable_safety_checker,
            stable_mode=args.stable_mode,
        )

        generated_images, used_prompt, metadata = renderer.render_morphology_preserved(
            illustration_path=args.illustration,
            prompt=prompt,
            controlnet_conditioning_scale=effective_controlnet_scale,
            strength=effective_strength,
            guidance_scale=effective_guidance_scale,
            num_inference_steps=args.num_steps,
            image_size=(args.image_size, args.image_size),
            num_images=args.num_images,
            seed=args.seed,
        )

        reference_image = Image.open(args.illustration).convert("RGB").resize(
            (args.image_size, args.image_size),
            Image.Resampling.LANCZOS,
        )

        scored_images: List[Tuple[int, Image.Image, float]] = []
        for idx, image in enumerate(generated_images, start=1):
            structure_iou = compute_structure_iou(reference_image, image)
            scored_images.append((idx, image, structure_iou))
            print(f"🧪 structure_iou candidate_{idx:02d} = {structure_iou:.4f}")

        scored_images.sort(key=lambda item: item[2], reverse=True)

        if args.strict_morphology and args.reject_low_structure:
            filtered = [item for item in scored_images if item[2] >= args.min_edge_iou]
            if not filtered:
                best_iou = scored_images[0][2] if scored_images else 0.0
                raise RuntimeError(
                    f"All generated images failed structure IoU threshold ({args.min_edge_iou:.2f}). "
                    f"Best IoU={best_iou:.4f}. Try lower --strength (e.g. 0.15) and higher --controlnet-scale (e.g. 1.45)."
                )
            if len(filtered) < len(scored_images):
                print(
                    f"⚠️  strict-morphology: dropped {len(scored_images) - len(filtered)} image(s) "
                    f"below IoU threshold {args.min_edge_iou:.2f}"
                )
            scored_images = filtered

        for save_idx, (_, image, structure_iou) in enumerate(scored_images, start=1):
            image_path = output_dir / f"specimen_{save_idx:02d}.png"
            image.save(image_path)
            print(f"📸 Saved: {image_path} (structure_iou={structure_iou:.4f})")

        metadata_path = output_dir / "generation_metadata.json"
        metadata_to_save = {
            "illustration_path": args.illustration,
            "species_name": species_name,
            "inferred_species": inferred_species,
            "description_source": description_source,
            "description_text": description_text,
            "prompt": used_prompt,
            "prompt_info": prompt_info,
            "num_images_requested": args.num_images,
            "num_images": len(scored_images),
            "seed": args.seed,
            "device": "cpu",
            "stable_mode": args.stable_mode,
            "strict_morphology": args.strict_morphology,
            "reject_low_structure": args.reject_low_structure,
            "min_edge_iou": args.min_edge_iou,
            "guidance_scale": effective_guidance_scale,
            "num_steps": args.num_steps,
            "strength": effective_strength,
            "controlnet_scale": effective_controlnet_scale,
            "image_size": args.image_size,
            "appearance_prior": args.appearance_prior,
            "structure_iou_scores": [
                {
                    "original_candidate_index": original_idx,
                    "structure_iou": structure_iou,
                }
                for original_idx, _, structure_iou in scored_images
            ],
            "renderer_metadata": metadata,
        }
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata_to_save, file, indent=2, ensure_ascii=False)

        prompt_path = output_dir / "05_prompt.txt"
        prompt_path.write_text(used_prompt, encoding="utf-8")

        print("\n✅ Generation complete")
        print(f"📁 Outputs: {output_dir}")
        print(f"📝 Prompt file: {prompt_path}")
        print(f"📋 Metadata file: {metadata_path}")

    except Exception as exc:
        print(f"\n❌ Generation failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
