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
    parser.add_argument("--mode", choices=["text_only", "structure_guided", "body_part_guided", "render_preserving_morphology", "morphology_guided"], 
                       default="text_only", help="Generation mode")
    parser.add_argument("--structure-strength", type=float, default=0.9,
                       help="ControlNet conditioning strength for structure_guided mode")
    parser.add_argument("--illustration", type=str, default=None,
                       help="Path to illustration file for structure_guided or render_preserving_morphology modes")
    parser.add_argument("--appearance-prior", type=str, default=None,
                       help="Path to appearance prior dataset (real crustacean photos)")
    parser.add_argument("--annotations-csv", type=str, default=None,
                       help="Path to body part annotations CSV (required for body_part_guided mode)")
    parser.add_argument("--species", type=str, nargs='+', default=None,
                       help="Species names to process (required for body_part_guided mode)")
    parser.add_argument("--morphology-strength", type=float, default=0.95,
                       help="ControlNet conditioning strength for render_preserving_morphology (0.0-1.0)")
    parser.add_argument("--denoising-strength", type=float, default=0.1,
                       help="Image-to-image denoising strength for render_preserving_morphology (0.0-1.0)")
    parser.add_argument("--controlnet-scale", type=float, default=1.0,
                       help="ControlNet conditioning strength for morphology_guided mode (0.0-1.0)")
    parser.add_argument("--img2img-strength", type=float, default=0.4,
                       help="img2img denoising strength for morphology_guided mode (0.0-1.0)")
    
    args = parser.parse_args()

    print("🔬 Scientific Image Generation Pipeline - Smoke Test")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Appearance prior: {args.appearance_prior}")
    if args.appearance_prior:
        print("📸 Loading appearance prior dataset...")
    if args.mode == "structure_guided":
        print(f"Structure strength: {args.structure_strength}")
        print(f"Illustration: {args.illustration}")
        
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
    elif args.mode == "render_preserving_morphology":
        print(f"Morphology strength: {args.morphology_strength}")
        print(f"Denoising strength: {args.denoising_strength}")
        print(f"Illustration: {args.illustration}")
        
        # Validate required arguments for render_preserving_morphology mode
        if not args.illustration:
            print("❌ --illustration is required for render_preserving_morphology mode")
            sys.exit(1)
        
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
    elif args.mode == "morphology_guided":
        print(f"ControlNet conditioning scale: {args.controlnet_scale}")
        print(f"img2img strength: {args.img2img_strength}")
        print(f"Illustration: {args.illustration}")
        
        # Validate required arguments for morphology_guided mode
        if not args.illustration:
            print("❌ --illustration is required for morphology_guided mode")
            sys.exit(1)
        
        # Validate parameter ranges
        if not (0.0 <= args.controlnet_scale <= 1.0):
            print(f"❌ --controlnet-scale must be between 0.0 and 1.0, got {args.controlnet_scale}")
            sys.exit(1)
        if not (0.0 <= args.img2img_strength <= 1.0):
            print(f"❌ --img2img-strength must be between 0.0 and 1.0, got {args.img2img_strength}")
            sys.exit(1)
        
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
    elif args.mode == "morphology_guided":
        print(f"ControlNet scale: {args.controlnet_scale}")
        print(f"img2img strength: {args.img2img_strength}")
        print(f"Illustration: {args.illustration}")
        
        # Validate required arguments for morphology_guided mode
        if not args.illustration:
            print("❌ --illustration is required for morphology_guided mode")
            sys.exit(1)
        
        # Validate parameter ranges
        if not (0.0 <= args.controlnet_scale <= 1.0):
            print(f"❌ --controlnet-scale must be between 0.0 and 1.0, got {args.controlnet_scale}")
            sys.exit(1)
        if not (0.0 <= args.img2img_strength <= 1.0):
            print(f"❌ --img2img-strength must be between 0.0 and 1.0, got {args.img2img_strength}")
            sys.exit(1)
        
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
        
        if args.mode == "body_part_guided":
            # Body-part guided mode uses different parameters
            metadata = generate_dataset_from_taxonomy(
                illustration_paths=None,  # Not used in body_part_guided mode
                description_texts=[],  # Not used in body_part_guided mode
                output_dir=output_dir,
                num_images_per_specimen=1,  # Only 1 image per body part
                seed=42,
                device="cpu",  # Force CPU to avoid CUDA memory issues
                generation_mode=args.mode,
                structure_strength=args.structure_strength,
                appearance_prior_dir=args.appearance_prior,
                annotations_csv=args.annotations_csv,
                species_list=args.species
            )
        elif args.mode == "morphology_guided":
            # Morphology-guided mode uses illustration as both img2img base and ControlNet conditioning
            metadata = generate_dataset_from_taxonomy(
                illustration_paths=[args.illustration],  # Illustration for both img2img and ControlNet
                description_texts=[description_text],
                output_dir=output_dir,
                num_images_per_specimen=1,  # Only 1 image
                seed=42,
                device="cpu",  # Force CPU to avoid CUDA memory issues
                generation_mode=args.mode,
                controlnet_conditioning_scale=args.controlnet_scale,
                img2img_strength=args.img2img_strength,
                appearance_prior_dir=args.appearance_prior
            )
        else:
            # Original modes
            illustration_paths = [args.illustration] if args.mode == "structure_guided" and args.illustration else None
            
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
        if args.mode == "body_part_guided":
            # For body_part_guided mode, show summary of all generated specimens
            total_specimens = len(metadata["specimens"])
            successful_specimens = len([s for s in metadata["specimens"] if "error" not in s])
            
            print(f"📊 Body-part guided generation completed!")
            print(f"   Total body parts processed: {total_specimens}")
            print(f"   Successfully generated: {successful_specimens}")
            
            if successful_specimens > 0:
                # Show first successful specimen as example
                specimen = next(s for s in metadata["specimens"] if "error" not in s)
                
                # 1. Final prompt
                final_prompt = specimen.get("generation_prompt", "N/A")
                print(f"1️⃣  Example final prompt: {final_prompt}")
                
                # 2. Path of generated image
                if "images" in specimen and len(specimen["images"]) > 0:
                    image_info = specimen["images"][0]
                    image_path = Path(output_dir) / image_info["path"]
                    print(f"2️⃣  Example generated image: {image_path}")
                else:
                    print("2️⃣  Generated image: No image generated")
                
                # 3. Species and body part info
                species = specimen.get("species", "unknown")
                body_part = specimen.get("body_part", "unknown")
                illustration_used = specimen.get("illustration_used", "none")
                description_used = specimen.get("description_used", "none")
                print(f"   Species: {species}")
                print(f"   Body part: {body_part}")
                print(f"   Illustration used: {illustration_used}")
                print(f"   Description used: {description_used[:60]}{'...' if len(description_used) > 60 else ''}")
            else:
                print("❌ No successful generations")
                return
        # Debug output paths
        input_illustration = debug_dir / "input_illustration.png"
        edge_map = debug_dir / "edge_map.png"
        morphology_edges = debug_dir / "morphology_edges.png"
        morphology_comparison = debug_dir / "morphology_comparison.png"
        rendered_result = debug_dir / "rendered_result.png"
        
        if input_illustration.exists():
            print(f"🔍 Debug input illustration: {input_illustration}")
        if edge_map.exists():
            print(f"🔍 Debug edge map: {edge_map}")
        if morphology_edges.exists():
            print(f"🔍 Debug morphology edges: {morphology_edges}")
        if morphology_comparison.exists():
            print(f"🔍 Debug morphology comparison: {morphology_comparison}")
        if rendered_result.exists():
            print(f"🔍 Debug rendered result: {rendered_result}")

        # 4. Generation mode and conditioning info
        generation_mode = metadata.get("generation_config", {}).get("generation_mode", "unknown")
        print(f"4️⃣  Generation mode: {generation_mode}")
        
        if args.mode == "body_part_guided":
            # Show appearance conditioning info for body-part mode
            generation_config = metadata.get("generation_config", {})
            appearance_prior_used = generation_config.get("appearance_prior_used", False)
            appearance_prior_dataset = generation_config.get("appearance_prior_dataset", None)
            reference_images_sampled = generation_config.get("reference_images_sampled", 0)
            color_palette_used = generation_config.get("color_palette_used", [])
            print(f"5️⃣  Appearance prior used: {'Yes' if appearance_prior_used else 'No'}")
            if appearance_prior_dataset:
                print(f"   Appearance prior dataset: {appearance_prior_dataset}")
                print(f"   Reference images sampled: {reference_images_sampled}")
                if color_palette_used:
                    print(f"   Color palette used: {color_palette_used}")
        elif args.mode == "render_preserving_morphology":
            # Show morphology-preserving parameters
            generation_config = metadata.get("generation_config", {})
            morphology_strength = generation_config.get("morphology_strength", "N/A")
            denoising_strength = generation_config.get("denoising_strength", "N/A")
            appearance_prior_used = generation_config.get("appearance_prior_used", False)
            print(f"5️⃣  Morphology strength: {morphology_strength}")
            print(f"6️⃣  Denoising strength: {denoising_strength}")
            print(f"7️⃣  Appearance prior used: {'Yes' if appearance_prior_used else 'No'}")
            
            # Show comparison outputs
            if morphology_comparison.exists():
                print(f"   Morphology comparison: {morphology_comparison}")
            if rendered_result.exists():
                print(f"   Rendered result: {rendered_result}")
        elif args.mode == "morphology_guided":
            # Show morphology-guided parameters
            generation_config = metadata.get("generation_config", {})
            controlnet_scale = generation_config.get("controlnet_conditioning_scale", "N/A")
            img2img_strength = generation_config.get("img2img_strength", "N/A")
            appearance_prior_used = generation_config.get("appearance_prior_used", False)
            print(f"5️⃣  ControlNet conditioning scale: {controlnet_scale}")
            print(f"6️⃣  img2img strength: {img2img_strength}")
            print(f"7️⃣  Appearance prior used: {'Yes' if appearance_prior_used else 'No'}")
            
            # Show comparison outputs
            morphology_guided_comparison = debug_dir / "morphology_guided_comparison.png"
            morphology_guided_rendered = debug_dir / "morphology_guided_rendered.png"
            if morphology_guided_comparison.exists():
                print(f"   Morphology-guided comparison: {morphology_guided_comparison}")
            if morphology_guided_rendered.exists():
                print(f"   Morphology-guided rendered: {morphology_guided_rendered}")
        else:
            # Original mode info
            conditioning_status = specimen.get("conditioning_status", {})
            structure_used = conditioning_status.get("structure_conditioning_used", False)
            print(f"5️⃣  Structure conditioning used: {'Yes' if structure_used else 'No'}")
            
            # Show appearance conditioning info
            generation_config = metadata.get("generation_config", {})
            appearance_prior_used = generation_config.get("appearance_prior_used", False)
            appearance_prior_dataset = generation_config.get("appearance_prior_dataset", None)
            reference_images_sampled = generation_config.get("reference_images_sampled", 0)
            color_palette_used = generation_config.get("color_palette_used", [])
            print(f"6️⃣  Appearance prior used: {'Yes' if appearance_prior_used else 'No'}")
            if appearance_prior_dataset:
                print(f"   Appearance prior dataset: {appearance_prior_dataset}")
                print(f"   Reference images sampled: {reference_images_sampled}")
                if color_palette_used:
                    print(f"   Color palette used: {color_palette_used}")
            
            # Show conditioning details
            if args.mode == "structure_guided":
                print(f"   Illustration provided: {conditioning_status.get('illustration_provided', False)}")
                print(f"   Structure conditioning used: {conditioning_status.get('structure_conditioning_used', False)}")
                print(f"   Control image available: {conditioning_status.get('control_image_available', False)}")
                if conditioning_status.get("error"):
                    print(f"   Conditioning error: {conditioning_status['error']}")


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
        if args.mode == "body_part_guided":
            print(f"1️⃣  Example final prompt: N/A (pipeline failed)")
            print(f"2️⃣  Example generated image: {output_dir}/images/<species>_<body_part>_000.png")
            print(f"3️⃣  Metadata JSON: {output_dir}/metadata.json")
            print(f"4️⃣  Appearance prior used: Yes (intended)")
        elif args.mode == "morphology_guided":
            print(f"1️⃣  Final prompt: N/A (pipeline failed)")
            print(f"2️⃣  Morphology-guided rendered: {output_dir}/debug/morphology_guided_rendered.png")
            print(f"3️⃣  Morphology-guided comparison: {output_dir}/debug/morphology_guided_comparison.png")
            print(f"4️⃣  Metadata JSON: {output_dir}/metadata.json")
            print(f"5️⃣  ControlNet conditioning scale: {args.controlnet_scale} (intended)")
            print(f"6️⃣  img2img strength: {args.img2img_strength} (intended)")
        else:
            print(f"1️⃣  Final prompt: N/A (pipeline failed)")
            print(f"2️⃣  Generated image: {output_dir}/images/000_000.png")
            print(f"3️⃣  Metadata JSON: {output_dir}/metadata.json")
            print(f"4️⃣  Structure conditioning used: Yes (intended)")

if __name__ == "__main__":
    main()