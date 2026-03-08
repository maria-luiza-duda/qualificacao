#!/usr/bin/env python3
"""
Smoke test for scientific image generation pipeline.

This script runs a minimal end-to-end test of the generation pipeline
using a single specimen with minimal parameters.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def main():
    """Run the smoke test."""
    
    parser = argparse.ArgumentParser(description="Smoke test for scientific image generation pipeline")
    parser.add_argument("--mode", choices=["text_only", "structure_guided", "body_part"], 
                       default="text_only", help="Generation mode")
    parser.add_argument("--structure-strength", type=float, default=0.9,
                       help="ControlNet conditioning strength for structure_guided mode")
    parser.add_argument("--illustration", type=str, default=None,
                       help="Path to illustration file for structure_guided or body_part modes")
    parser.add_argument("--appearance-prior", type=str, default=None,
                       help="Path to appearance prior dataset (real crustacean photos)")
    
    args = parser.parse_args()

    print("🔬 Scientific Image Generation Pipeline - Smoke Test")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Appearance prior: {args.appearance_prior}")
    if args.appearance_prior:
        print("📸 Loading appearance prior dataset...")
    if args.mode in ["structure_guided", "body_part"]:
        print(f"Structure strength: {args.structure_strength}")
        print(f"CLI Illustration argument: {args.illustration}")
        
        # Debug: Check if illustration file exists
        if args.illustration:
            illustration_path_obj = Path(args.illustration)
            print(f"Illustration path exists: {illustration_path_obj.exists()}")
            if illustration_path_obj.exists():
                print(f"Illustration file size: {illustration_path_obj.stat().st_size} bytes")
                try:
                    from PIL import Image
                    with Image.open(args.illustration) as img:
                        print(f"PIL can open illustration: {img.format} {img.size} {img.mode}")
                except Exception as e:
                    print(f"PIL cannot open illustration: {e}")

    # Test parameters
    illustration_path = args.illustration
    taxonomic_description = {
        "carapace": "Smooth carapace",
        "pseudorostrum": "elongated pseudorostrum", 
        "pereopods": "slender pereopods",
        "uropods": "uropods with elongated exopod and shorter endopod"
    }

    # Convert description dict to string format expected by pipeline
    description_text = ". ".join([f"{key}: {value}" for key, value in taxonomic_description.items()])

    output_dir = "outputs/smoke_test"
    
    # Create debug directory
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Illustration path: {illustration_path}")
    print(f"📝 Description: {description_text}")
    print(f"🎯 Output directory: {output_dir}")
    print()

    try:
        # Import the pipeline function
        from campylaspis.generation import generate_dataset_from_taxonomy
        print("✅ Pipeline module imported successfully")

        # Run the pipeline with parameters based on mode
        print("🚀 Running generation pipeline...")
        
        illustration_paths = [args.illustration] if args.mode in ["structure_guided", "body_part"] and args.illustration else None
        
        metadata = generate_dataset_from_taxonomy(
            illustration_paths=illustration_paths,
            description_texts=[description_text],
            output_dir=output_dir,
            num_images_per_specimen=1,  # Only 1 image
            seed=42,
            device="cpu",  # Force CPU to avoid CUDA memory issues
            generation_mode=args.mode,
            structure_strength=args.structure_strength,
            appearance_prior_dir=args.appearance_prior
        )

        print("✅ Pipeline completed successfully!")
        print()

        # Extract and display required information
        specimen = metadata["specimens"][0]

        # 1. Final prompt
        final_prompt = specimen.get("generation_prompt", "N/A")
        print(f"1️⃣  Final prompt: {final_prompt}")

        # 2. Path of generated image
        if "images" in specimen and len(specimen["images"]) > 0:
            image_info = specimen["images"][0]
            image_path = Path(output_dir) / image_info["path"]
            print(f"2️⃣  Generated image: {image_path}")
        else:
            print("2️⃣  Generated image: No image generated")

        # 3. Path of metadata.json
        metadata_path = Path(output_dir) / "metadata.json"
        print(f"3️⃣  Metadata JSON: {metadata_path}")

        # Debug files
        debug_dir = Path(output_dir) / "debug"
        input_illustration = debug_dir / "input_illustration.png"
        edge_map = debug_dir / "edge_map.png"
        
        if input_illustration.exists():
            print(f"🔍 Debug input illustration: {input_illustration}")
        if edge_map.exists():
            print(f"🔍 Debug edge map: {edge_map}")

        # 4. Generation mode and conditioning info
        generation_mode = metadata.get("generation_config", {}).get("generation_mode", "unknown")
        conditioning_status = specimen.get("conditioning_status", {})
        structure_used = conditioning_status.get("structure_conditioning_used", False)
        body_part_used = generation_mode == "body_part"
        print(f"4️⃣  Generation mode: {generation_mode}")
        print(f"5️⃣  Structure conditioning used: {'Yes' if structure_used else 'No'}")
        print(f"6️⃣  Body part generation: {'Yes' if body_part_used else 'No'}")
        
        # Show appearance conditioning info
        generation_config = metadata.get("generation_config", {})
        appearance_prior_used = generation_config.get("appearance_prior_used", False)
        appearance_prior_dataset = generation_config.get("appearance_prior_dataset", None)
        reference_images_sampled = generation_config.get("reference_images_sampled", 0)
        color_palette_used = generation_config.get("color_palette_used", [])
        print(f"7️⃣  Appearance prior used: {'Yes' if appearance_prior_used else 'No'}")
        if appearance_prior_dataset:
            print(f"   Appearance prior dataset: {appearance_prior_dataset}")
            print(f"   Reference images sampled: {reference_images_sampled}")
            if color_palette_used:
                print(f"   Color palette used: {color_palette_used}")
        
        # Show conditioning details
        if args.mode in ["structure_guided", "body_part"]:
            print(f"   Illustration provided: {conditioning_status.get('illustration_provided', False)}")
            print(f"   Structure conditioning used: {conditioning_status.get('structure_conditioning_used', False)}")
            print(f"   Control image available: {conditioning_status.get('control_image_available', False)}")
            if conditioning_status.get("error"):
                print(f"   Conditioning error: {conditioning_status['error']}")
        
        # Show body part info if available
        if body_part_used and "body_part" in specimen:
            body_part_info = specimen["body_part"]
            print(f"8️⃣  Detected body part: {body_part_info.get('identifier', 'unknown')} ({body_part_info.get('description', 'unknown')})")

        print()
        print("🎉 Smoke test completed successfully!")
        print(f"📊 Generated {metadata['dataset_info']['total_images']} image(s)")
        print(f"📂 Results saved to: {output_dir}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This is expected if required dependencies are not installed.")
        print("Missing dependencies: torch, diffusers, numpy, opencv-python, etc.")

    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        print("This may be due to missing dependencies or invalid test data.")

        # Still show expected output paths
        print()
        print("Expected outputs:")
        print(f"1️⃣  Final prompt: N/A (pipeline failed)")
        print(f"2️⃣  Generated image: {output_dir}/images/specimen_000_000.png")
        print(f"3️⃣  Metadata JSON: {output_dir}/metadata.json")
        print(f"4️⃣  Structure conditioning used: Yes (intended)")

if __name__ == "__main__":
    main()