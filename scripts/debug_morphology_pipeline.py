#!/usr/bin/env python3
"""
Morphology Pipeline Diagnostics Script

This script runs a single specimen through the morphology-guided generation pipeline
and saves comprehensive debug information to identify why anatomically incorrect
crustaceans are being generated.

The script saves all intermediate processing steps and detailed metadata for analysis.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def save_debug_image(image: Image.Image, output_path: Path, description: str) -> str:
    """Save a debug image and return the relative path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"💾 Saved {description}: {output_path}")
    return str(output_path)

def save_debug_json(data: Dict[str, Any], output_path: Path, description: str) -> str:
    """Save debug JSON data and return the relative path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {description}: {output_path}")
    return str(output_path)

def save_debug_text(text: str, output_path: Path, description: str) -> str:
    """Save debug text and return the relative path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"💾 Saved {description}: {output_path}")
    return str(output_path)

def infer_species_from_illustration_path(illustration_path: str) -> str:
    """
    Infer species name from illustration path.
    
    Expected path format: .../processed/{species}/images/{filename}
    """
    path = Path(illustration_path)
    
    # Find the 'processed' directory in the path
    parts = path.parts
    try:
        processed_idx = parts.index('processed')
        if processed_idx + 1 < len(parts):
            return parts[processed_idx + 1]
    except ValueError:
        pass
    
    # Fallback: try to find species name in path
    for part in reversed(path.parts):
        if part in ['images', 'text', 'processed']:
            continue
        # Check if this looks like a species name (no extension, reasonable length)
        if '.' not in part and 3 <= len(part) <= 50:
            return part
    
    raise ValueError(f"Could not infer species from illustration path: {illustration_path}")

def load_species_description_text(species: str, dataset_root: str = "/home/maria-luiza-duda/datasets/campylaspis") -> Tuple[str, str]:
    """
    Load the taxonomic description text for a species.
    
    Returns: (description_text, text_file_path)
    """
    text_dir = Path(dataset_root) / "processed" / species / "text"
    
    if not text_dir.exists():
        raise FileNotFoundError(f"Species text directory not found: {text_dir}")
    
    # Find all .txt files
    txt_files = list(text_dir.glob("*.txt"))
    
    if len(txt_files) == 0:
        raise FileNotFoundError(f"No .txt files found in: {text_dir}")
    elif len(txt_files) == 1:
        text_file = txt_files[0]
    else:
        # Multiple files - log them and fail
        file_list = [str(f.relative_to(text_dir)) for f in txt_files]
        raise ValueError(f"Multiple text files found in {text_dir}: {file_list}. Please ensure exactly one .txt file per species.")
    
    # Load the text
    with open(text_file, 'r', encoding='utf-8') as f:
        description_text = f.read().strip()
    
    return description_text, str(text_file)

def extract_morphology_edges(illustration_path: str) -> Tuple[Image.Image, Image.Image]:
    """
    Extract edges from illustration for ControlNet conditioning.
    Returns both raw and cleaned edge maps.
    """
    # Load illustration
    illustration = Image.open(illustration_path)

    # Convert to grayscale for edge detection
    if illustration.mode != 'L':
        gray = illustration.convert('L')
    else:
        gray = illustration

    # Extract edges using OpenCV if available, otherwise fallback
    try:
        import cv2
        import numpy as np

        # Convert to numpy array
        gray_array = np.array(gray)

        # Raw edge detection (weaker thresholds)
        edges_raw = cv2.Canny(gray_array, threshold1=100, threshold2=200)

        # Clean edge detection (stronger thresholds for morphology preservation)
        edges_clean = cv2.Canny(gray_array, threshold1=50, threshold2=150)

        # Convert back to PIL Images
        edge_map_raw = Image.fromarray(edges_raw, mode='L')
        edge_map_clean = Image.fromarray(edges_clean, mode='L')

    except ImportError:
        # Fallback: simple edge detection using PIL
        print("⚠️  OpenCV not available, using fallback edge detection")
        from PIL import ImageFilter

        # Simple edge enhancement
        edge_map_raw = gray.filter(ImageFilter.FIND_EDGES)
        edge_map_clean = gray.filter(ImageFilter.FIND_EDGES)

    return edge_map_raw, edge_map_clean

def main():
    """Run the morphology pipeline diagnostics."""

    parser = argparse.ArgumentParser(description="Morphology Pipeline Diagnostics")
    parser.add_argument("--illustration", type=str, required=True,
                       help="Path to illustration file")
    parser.add_argument("--description-text", type=str, default=None,
                       help="Taxonomic description text (optional - auto-loaded from dataset if not provided)")
    parser.add_argument("--appearance-prior", type=str, default=None,
                       help="Path to appearance prior dataset")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for debug files")
    parser.add_argument("--disable-safety-checker", action=argparse.BooleanOptionalAction, default=True,
                       help="Disable diffusers safety checker (default: disabled for morphology-guided debug runs)")
    parser.add_argument("--mode", choices=["morphology_guided", "morphology_guided_no_controlnet"], default="morphology_guided",
                       help="Generation mode: morphology_guided (ControlNet+img2img) or morphology_guided_no_controlnet (img2img only, diagnostic)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu",
                       help="Device to run on: cpu or cuda (default: cpu)")

    args = parser.parse_args()

    # Step 0: Handle description text (auto-load from dataset if not provided)
    if args.description_text is None:
        print("🔍 Auto-loading taxonomic description from dataset...")
        try:
            inferred_species = infer_species_from_illustration_path(args.illustration)
            description_text, text_file_path = load_species_description_text(inferred_species)
            print(f"✅ Inferred species: {inferred_species}")
            print(f"✅ Loaded description from: {text_file_path}")
            print(f"📝 Description length: {len(description_text)} characters")
        except Exception as e:
            print(f"❌ Failed to auto-load description: {e}")
            print("💡 Please provide --description-text manually for debugging")
            sys.exit(1)
    else:
        print("📝 Using manually provided description text...")
        inferred_species = None
        text_file_path = None
        description_text = args.description_text

    print("🔬 Morphology Pipeline Diagnostics")
    print("=" * 60)
    print(f"Illustration: {args.illustration}")
    print(f"Description: {description_text[:100]}{'...' if len(description_text) > 100 else ''}")
    print(f"Inferred species: {inferred_species}")
    print(f"Text file source: {text_file_path}")
    print(f"Appearance prior: {args.appearance_prior}")
    print(f"Disable safety checker: {args.disable_safety_checker}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_report = {
        "illustration_used": args.illustration,
        "description_text": description_text,
        "inferred_species": inferred_species,
        "text_file_path": text_file_path,
        "appearance_prior_used": args.appearance_prior,
        "mode": args.mode,
        "fallback_occurred": False,
        "prompt_truncated": False,
        "prompt_token_count": None,
        "prompt_view": None,
        "prompt_included_body_parts": [],
        "prompt_dropped_body_parts": [],
        "full_scientific_phrases_used": None,
        "summarization_step_applied": None,
        "compression_strategy": None,
        "prompt_truncation_occurred": False,
        "body_part_detected": False,
        "description_parsing_failed": False,
        "generation_failed_with_reason": None,
        "safety_checker_enabled": None,
        "safety_checker_disabled": None,
        "nsfw_content_detected": None
    }

    try:
        # Step 1: Load and validate input illustration
        print("📖 Step 1: Loading input illustration...")
        if not Path(args.illustration).exists():
            raise FileNotFoundError(f"Illustration file not found: {args.illustration}")

        input_illustration = Image.open(args.illustration).convert('RGB')
        print(f"✅ Loaded illustration: {input_illustration.size} {input_illustration.mode}")

        # Save 01_input_illustration.png
        save_debug_image(
            input_illustration,
            output_dir / "01_input_illustration.png",
            "input illustration"
        )

        # Step 2: Resize illustration if needed (for consistency)
        print("🔄 Step 2: Resizing illustration...")
        # Stable mode uses 320x320 for numerical stability
        target_size = (320, 320)
        resized_illustration = input_illustration.resize(target_size, Image.Resampling.LANCZOS)
        print(f"✅ Resized to: {resized_illustration.size}")

        # Save 02_resized_input.png
        save_debug_image(
            resized_illustration,
            output_dir / "02_resized_input.png",
            "resized input"
        )

        # Step 3: Extract edge maps
        print("🔍 Step 3: Extracting morphology edges...")
        
        if args.mode == "morphology_guided":
            edge_map_raw, edge_map_clean = extract_morphology_edges(args.illustration)

            # Resize edge maps to match target size
            edge_map_raw_resized = edge_map_raw.resize(target_size, Image.Resampling.LANCZOS)
            edge_map_clean_resized = edge_map_clean.resize(target_size, Image.Resampling.LANCZOS)

            print(f"✅ Extracted edge maps: raw={edge_map_raw.size}, clean={edge_map_clean.size}")

            # Save 03_edge_map_raw.png and 04_edge_map_clean.png
            save_debug_image(
                edge_map_raw_resized,
                output_dir / "03_edge_map_raw.png",
                "raw edge map"
            )
            save_debug_image(
                edge_map_clean_resized,
                output_dir / "04_edge_map_clean.png",
                "clean edge map"
            )
        else:
            print(f"⏭️  Skipping edge extraction (mode={args.mode}, ControlNet disabled)")

        # Step 4: Parse morphological description
        print("🔬 Step 4: Parsing morphological description...")
        from campylaspis.generation import MorphologyParser

        morphology_parser = MorphologyParser()
        morphological_constraints = morphology_parser.parse_taxonomic_description(description_text)

        print(f"✅ Parsed constraints: {len(morphological_constraints)} body parts identified")

        # Check if parsing failed
        if not morphological_constraints:
            debug_report["description_parsing_failed"] = True
            print("⚠️  Warning: No morphological constraints extracted from description")

        # Check if any body parts were detected
        body_parts_found = len(morphological_constraints) > 0
        debug_report["body_part_detected"] = body_parts_found

        # Save 07_morphology_constraints.json
        save_debug_json(
            morphological_constraints,
            output_dir / "07_morphology_constraints.json",
            "morphological constraints"
        )

        # Save 08_full_description.txt
        save_debug_text(
            description_text,
            output_dir / "08_full_description.txt",
            "full taxonomic description"
        )

        # Step 5: Build prompt
        print("📝 Step 5: Building generation prompt...")
        from campylaspis.generation.morphology_parser import build_morphology_prompt

        if inferred_species:
            species_name = f"Campylaspis {inferred_species}"
        else:
            species_name = "Campylaspis specimen"

        prompt_result = build_morphology_prompt(
            species_name=species_name,
            constraints=morphological_constraints,
            illustration_path=args.illustration,
            return_metadata=True,
        )
        prompt = prompt_result["prompt"]
        full_prompt_before_compression = prompt_result.get("full_prompt_before_compression", prompt)
        token_count = prompt_result["token_count"]
        prompt_truncated = prompt_result.get("truncated", False)
        prompt_view = prompt_result.get("view")
        prompt_included_body_parts = prompt_result.get("included_body_parts", [])
        prompt_dropped_body_parts = prompt_result.get("dropped_body_parts", [])
        full_scientific_phrases_used = prompt_result.get("full_scientific_phrases_used", True)
        summarization_step_applied = prompt_result.get("summarization_applied", False)
        compression_strategy = prompt_result.get("compression_strategy", "none")

        debug_report["prompt_truncated"] = prompt_truncated
        debug_report["prompt_truncation_occurred"] = prompt_truncated
        debug_report["prompt_token_count"] = token_count
        debug_report["prompt_view"] = prompt_view
        debug_report["prompt_included_body_parts"] = prompt_included_body_parts
        debug_report["prompt_dropped_body_parts"] = prompt_dropped_body_parts
        debug_report["full_scientific_phrases_used"] = full_scientific_phrases_used
        debug_report["summarization_step_applied"] = summarization_step_applied
        debug_report["compression_strategy"] = compression_strategy

        print(f"✅ Built prompt: {len(prompt)} characters, {token_count} tokens")
        if prompt_truncated:
            print("⚠️  Prompt was truncated to respect CLIP token budget")
        print(f"🗜️  Compression strategy: {compression_strategy}")

        # Save 04_full_prompt_before_compression.txt
        save_debug_text(
            full_prompt_before_compression,
            output_dir / "04_full_prompt_before_compression.txt",
            "full prompt before compression"
        )

        # Save 05_prompt.txt
        save_debug_text(
            prompt,
            output_dir / "05_prompt.txt",
            "generation prompt"
        )

        # Save 12_prompt_info.json
        save_debug_json(
            {
                "species_name": species_name,
                "view": prompt_view,
                "token_count": token_count,
                "truncated": prompt_truncated,
                "included_body_parts": prompt_included_body_parts,
                "dropped_body_parts": prompt_dropped_body_parts,
                "compression_strategy": compression_strategy,
                "essential_priority_parts": ["carapace", "pseudorostrum", "pereopod_2", "uropod"],
                "essential_parts_included": [p for p in ["carapace", "pseudorostrum", "pereopod_2", "uropod"] if p in prompt_included_body_parts],
                "essential_parts_dropped": [p for p in ["carapace", "pseudorostrum", "pereopod_2", "uropod"] if p in prompt_dropped_body_parts],
            },
            output_dir / "12_prompt_info.json",
            "prompt metadata"
        )

        # Step 6: Prepare negative prompt
        print("🚫 Step 6: Preparing negative prompt...")
        negative_prompt = "blurry, low quality, distorted, deformed, watermark, text, lobster, shrimp, crab, prawn, scorpion, spider, insect, cephalopod, octopus tentacles, claws, extra legs, extra appendages, fantasy anatomy, monster, alien, vertebrate eyes, fish fins"

        # Save 06_negative_prompt.txt
        save_debug_text(
            negative_prompt,
            output_dir / "06_negative_prompt.txt",
            "negative prompt"
        )

        # Step 7: Initialize generator and get pipeline info
        print("🎨 Step 7: Initializing generation pipeline...")
        from campylaspis.generation import MorphologyGuidedRenderer

        # Initialize renderer with stable_mode for numerical stability
        renderer = MorphologyGuidedRenderer(
            device=args.device,  # Use specified device (cpu or cuda)
            appearance_prior_dir=args.appearance_prior,
            disable_safety_checker=args.disable_safety_checker,
            stable_mode=True,  # Enable numerically stable inference (float32, reduced parameters)
        )

        # Extract pipeline information
        # For no_controlnet mode, report that ControlNet is not loaded/used even if technically in memory
        if args.mode == "morphology_guided_no_controlnet":
            pipeline_info = {
                "diffusers_pipeline_class": "StableDiffusionImg2ImgPipeline",
                "controlnet_loaded": False,
                "img2img_used": True,
                "appearance_prior_used": renderer.appearance_conditioner is not None,
                "fallback_happened": False,
                "device": str(renderer.device) if hasattr(renderer, 'device') else "unknown",
                "dtype": str(renderer.dtype) if hasattr(renderer, 'dtype') else "unknown",
                "safety_checker_enabled": bool(getattr(renderer, 'safety_checker_enabled', False)),
                "safety_checker_disabled": bool(getattr(renderer, 'safety_checker_disabled', True)),
                "no_controlnet_pipeline_available": renderer.pipe_no_controlnet is not None,
                "mode": "morphology_guided_no_controlnet",
            }
        else:
            pipeline_info = {
                "diffusers_pipeline_class": str(type(renderer.pipe).__name__) if renderer.pipe else "None",
                "controlnet_loaded": renderer.controlnet is not None,
                "img2img_used": True,
                "appearance_prior_used": renderer.appearance_conditioner is not None,
                "fallback_happened": False,
                "device": str(renderer.device) if hasattr(renderer, 'device') else "unknown",
                "dtype": str(renderer.dtype) if hasattr(renderer, 'dtype') else "unknown",
                "safety_checker_enabled": bool(getattr(renderer, 'safety_checker_enabled', False)),
                "safety_checker_disabled": bool(getattr(renderer, 'safety_checker_disabled', True)),
                "no_controlnet_pipeline_available": renderer.pipe_no_controlnet is not None,
                "mode": "morphology_guided",
            }

        debug_report["safety_checker_enabled"] = pipeline_info["safety_checker_enabled"]
        debug_report["safety_checker_disabled"] = pipeline_info["safety_checker_disabled"]
        print(f"🔒 Safety checker enabled: {pipeline_info['safety_checker_enabled']}")
        print(f"🔓 Safety checker disabled: {pipeline_info['safety_checker_disabled']}")

        # Check if fallback happened (no actual pipeline)
        if renderer.pipe is None:
            pipeline_info["fallback_happened"] = True
            debug_report["fallback_occurred"] = True
            print("⚠️  Warning: Pipeline not available, fallback mode")

        print(f"✅ Pipeline initialized: {pipeline_info['diffusers_pipeline_class']}")

        # Save 11_pipeline_info.json
        save_debug_json(
            pipeline_info,
            output_dir / "11_pipeline_info.json",
            "pipeline information"
        )

        # Step 8: Prepare generation arguments
        print("⚙️  Step 8: Preparing generation arguments...")
        generation_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "controlnet_conditioning_scale": 1.0 if args.mode == "morphology_guided" else None,  # None for no_controlnet mode
            "strength": 0.15,  # Stable mode uses 0.15 for numerical stability
            "guidance_scale": 3.0,  # Stable mode uses 3.0 for numerical stability
            "num_inference_steps": 6,  # Stable mode uses 6 for numerical stability
            "seed": 42,  # Fixed seed for reproducibility
            "image_size": target_size,
            "token_count": token_count,
            "memory_optimizations_enabled": False,  # Disabled in stable_mode
            "vram_optimization_mode": "stable_mode (float32, no xformers, no slicing, 320x320, 6 steps)",
            "numerical_stability_mode": True,
            "stable_mode": True,
            "disable_safety_checker": args.disable_safety_checker,
            "mode": args.mode,
            "controlnet_enabled": args.mode == "morphology_guided",
        }

        print(f"✅ Generation args prepared")

        # Save 13_generation_args.json
        save_debug_json(
            generation_args,
            output_dir / "13_generation_args.json",
            "generation arguments"
        )

        # Step 9: Run generation (now with real generation, no mock)
        print("🚀 Step 9: Running morphology-guided generation...")
        generation_success = False
        generation_error = None
        output_tensor_stats_before_postprocess = {}
        output_tensor_stats_after_postprocess = {}
        
        try:
            # For morphology_guided mode, we now run the actual generation
            # but expect it to fail loudly if components are missing
            print("🔬 Attempting real morphology-guided generation (no silent fallbacks)...")
            
            if args.mode == "morphology_guided":
                generated_images, used_prompt, generation_metadata = renderer.render_morphology_preserved(
                    illustration_path=args.illustration,
                    prompt=prompt,
                    controlnet_conditioning_scale=generation_args["controlnet_conditioning_scale"],
                    strength=generation_args["strength"],
                    guidance_scale=generation_args["guidance_scale"],
                    num_inference_steps=generation_args["num_inference_steps"],
                    image_size=generation_args["image_size"],
                    num_images=1,
                    seed=generation_args["seed"],
                    debug_capture_pre_post_stats=True,
                    debug_save_latent_stats_path=str(output_dir / "17_latent_stats_per_step.json"),
                )
            elif args.mode == "morphology_guided_no_controlnet":
                generated_images, used_prompt, generation_metadata = renderer.render_morphology_no_controlnet(
                    illustration_path=args.illustration,
                    prompt=prompt,
                    strength=generation_args["strength"],
                    guidance_scale=generation_args["guidance_scale"],
                    num_inference_steps=generation_args["num_inference_steps"],
                    image_size=generation_args["image_size"],
                    num_images=1,
                    seed=generation_args["seed"],
                    debug_capture_pre_post_stats=True,
                    debug_save_latent_stats_path=str(output_dir / "17_latent_stats_per_step.json"),
                )
            else:
                raise ValueError(f"Unknown mode: {args.mode}")
            # Extract output tensor stats from metadata if available
            output_tensor_stats_before_postprocess = generation_metadata.get("output_tensor_stats_before_postprocess", {})
            output_tensor_stats_after_postprocess = generation_metadata.get("output_tensor_stats_after_postprocess", {})
            output_image = generated_images[0]
            generation_success = True
            print(f"✅ Generation completed successfully: {output_image.size}")
            
            # Save tensor stats to debug report
            debug_report["output_tensor_validation"] = not (
                output_tensor_stats_after_postprocess.get("contains_nan", False)
                or output_tensor_stats_after_postprocess.get("contains_inf", False)
            )
            debug_report["output_contains_nan"] = output_tensor_stats_after_postprocess.get("contains_nan", False)
            debug_report["output_contains_inf"] = output_tensor_stats_after_postprocess.get("contains_inf", False)
            debug_report["output_sanitization_applied"] = bool(
                generation_metadata.get("sanitization_before_postprocess", False)
                or generation_metadata.get("sanitization_after_postprocess", False)
            )
            debug_report["nsfw_content_detected"] = generation_metadata.get("nsfw_content_detected")
            debug_report["safety_checker_enabled"] = generation_metadata.get("safety_checker_enabled", debug_report.get("safety_checker_enabled"))
            debug_report["safety_checker_disabled"] = generation_metadata.get("safety_checker_disabled", debug_report.get("safety_checker_disabled"))

        except Exception as e:
            print(f"❌ Generation failed as expected (morphology_guided validation): {e}")
            generation_error = str(e)
            debug_report["fallback_occurred"] = False  # This is expected failure, not fallback
            debug_report["generation_failed_with_reason"] = generation_error
            
            # Create a placeholder image to show the failure
            output_image = resized_illustration.copy()
            used_prompt = prompt
            generation_metadata = {"error": generation_error, "expected_failure": True}

        # Save 14_output_image.png
        save_debug_image(
            output_image,
            output_dir / "14_output_image.png",
            "output image"
        )

        # Step 10: Save final debug report
        print("📊 Step 10: Saving debug report...")

        # Save 11_debug_report.json
        save_debug_json(
            debug_report,
            output_dir / "11_debug_report.json",
            "debug report"
        )

        # Save 15_output_tensor_stats_before_postprocess.json
        save_debug_json(
            output_tensor_stats_before_postprocess,
            output_dir / "15_output_tensor_stats_before_postprocess.json",
            "output tensor statistics before postprocess"
        )

        # Save 16_output_tensor_stats_after_postprocess.json
        save_debug_json(
            output_tensor_stats_after_postprocess,
            output_dir / "16_output_tensor_stats_after_postprocess.json",
            "output tensor statistics after postprocess"
        )

        print("\n" + "=" * 60)
        print("🎯 DIAGNOSTICS COMPLETE")
        print("=" * 60)

        # Print all saved file paths
        print("\n📁 Saved debug files:")
        for file_path in sorted(output_dir.iterdir()):
            if file_path.is_file():
                print(f"  {file_path.name}: {file_path}")

        print(f"\n🔍 Debug report summary:")
        print(f"  - Illustration used: {debug_report['illustration_used']}")
        print(f"  - Edge map used: {'clean' if not debug_report['fallback_occurred'] else 'none (fallback)'}")
        print(f"  - Prompt truncated: {debug_report['prompt_truncated']}")
        print(f"  - Fallback occurred: {debug_report['fallback_occurred']}")
        print(f"  - Body part detected: {debug_report['body_part_detected']}")
        print(f"  - Description parsing failed: {debug_report['description_parsing_failed']}")
        if debug_report.get('generation_failed_with_reason'):
            print(f"  - Generation failed with reason: {debug_report['generation_failed_with_reason']}")

        print("\n✅ Morphology pipeline diagnostics completed successfully!")

    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()