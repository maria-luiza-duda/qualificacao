#!/usr/bin/env python3
"""
Batch generation test for scientific image generation pipeline.

Generates images in three modes for manual inspection:
1. text_only: Text prompt only
2. illustration_only: Illustration conditioning only
3. text_plus_illustration: Combined text and illustration conditioning
"""

import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def main():
    """Run the batch generation test."""

    print("🔬 Scientific Image Generation - Batch Test")
    print("=" * 60)

    # Test specimens
    specimens = [
        {
            "id": 0,
            "species": "Campylaspis_sp_01",
            "illustration": "data/illustrations/campylaspis_01.png",
            "description": {
                "carapace": "Smooth carapace",
                "pseudorostrum": "elongated pseudorostrum",
                "pereopods": "slender pereopods",
                "uropods": "uropods with elongated exopod and shorter endopod"
            }
        },
        {
            "id": 1,
            "species": "Campylaspis_sp_02",
            "illustration": "data/illustrations/campylaspis_02.png",
            "description": {
                "carapace": "Oval carapace",
                "pseudorostrum": "short pseudorostrum",
                "pereopods": "robust pereopods",
                "uropods": "broad uropods with equal rami"
            }
        },
        {
            "id": 2,
            "species": "Campylaspis_sp_03",
            "illustration": "data/illustrations/campylaspis_03.png",
            "description": {
                "carapace": "Rounded carapace with fine setae",
                "pseudorostrum": "curved pseudorostrum",
                "pereopods": "long pereopods with spines",
                "uropods": "asymmetrical uropods"
            }
        }
    ]

    seeds = [42, 43]
    num_images_per_specimen = 2

    print(f"📊 Testing {len(specimens)} specimens × {num_images_per_specimen} images × {len(seeds)} seeds × 3 modes")
    print(f"🎯 Total images to generate: {len(specimens) * num_images_per_specimen * len(seeds) * 3}")
    print()

    # Initialize components
    try:
        from campylaspis.generation import ScientificImageGenerator
        generator = ScientificImageGenerator(device="cpu")  # Force CPU to avoid CUDA issues
        print("✅ ScientificImageGenerator imported successfully")
    except ImportError as e:
        print(f"❌ Cannot import ScientificImageGenerator: {e}")
        return

    try:
        from campylaspis.generation import ScientificPromptProcessor
        prompt_processor = ScientificPromptProcessor()
        print("✅ ScientificPromptProcessor imported successfully")
    except ImportError as e:
        print(f"❌ Cannot import ScientificPromptProcessor: {e}")
        return

    try:
        from campylaspis.generation import ScientificConditioningProcessor
        conditioning_processor = ScientificConditioningProcessor()
        print("✅ ScientificConditioningProcessor imported successfully")
    except ImportError as e:
        print(f"⚠️  ScientificConditioningProcessor not available: {e}")
        print("Illustration-based modes will be skipped.")
        conditioning_processor = None

    # CSV data collection
    csv_data = []
    save_errors = []

    # Process each mode
    modes = [
        ("text_only", "Text prompt only"),
        ("illustration_only", "Illustration conditioning only"),
        ("text_plus_illustration", "Combined text and illustration")
    ]

    for mode_name, mode_description in modes:
        print(f"🎨 Processing mode: {mode_name} - {mode_description}")

        output_dir = Path(f"outputs/batch_{mode_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        image_counter = 0

        for specimen in specimens:
            print(f"  🧬 Processing specimen {specimen['id']}: {specimen['species']}")

            # Initialize conditioning variables
            structure_image = None
            conditioning_error = None

            # Prepare prompt and conditioning based on mode
            if mode_name == "text_only":
                # Text only - no structural conditioning
                description_text = ". ".join([f"{k}: {v}" for k, v in specimen['description'].items()])
                prompt = prompt_processor.build_taxonomic_prompt(specimen['species'], specimen['description'])
                structure_image = None

            elif mode_name == "illustration_only":
                # Illustration only - generic prompt with structural conditioning
                if conditioning_processor is None:
                    print(f"    ⚠️  Skipping {mode_name} mode: conditioning processor not available")
                    continue
                prompt = f"A realistic biological specimen of {specimen['species']}."
                try:
                    control_map = conditioning_processor.prepare_structure_condition(specimen['illustration'])
                    # Convert to PIL Image for generator
                    if hasattr(control_map, 'shape'):
                        from PIL import Image
                        import numpy as np
                        if hasattr(control_map, 'cpu'):
                            control_map = control_map.cpu().numpy()
                        control_image = Image.fromarray((control_map.squeeze() * 255).astype('uint8'), mode='L')
                    else:
                        control_image = control_map
                    structure_image = control_image
                except Exception as e:
                    conditioning_error = str(e)
                    print(f"    ⚠️  Could not load illustration {specimen['illustration']}: {conditioning_error}")
                    structure_image = None

            elif mode_name == "text_plus_illustration":
                # Combined - text prompt with structural conditioning
                if conditioning_processor is None:
                    print(f"    ⚠️  Skipping {mode_name} mode: conditioning processor not available")
                    continue
                description_text = ". ".join([f"{k}: {v}" for k, v in specimen['description'].items()])
                prompt = prompt_processor.build_taxonomic_prompt(specimen['species'], specimen['description'])
                try:
                    control_map = conditioning_processor.prepare_structure_condition(specimen['illustration'])
                    # Convert to PIL Image for generator
                    if hasattr(control_map, 'shape'):
                        from PIL import Image
                        import numpy as np
                        if hasattr(control_map, 'cpu'):
                            control_map = control_map.cpu().numpy()
                        control_image = Image.fromarray((control_map.squeeze() * 255).astype('uint8'), mode='L')
                    else:
                        control_image = control_map
                    structure_image = control_image
                except Exception as e:
                    conditioning_error = str(e)
                    print(f"    ⚠️  Could not load illustration {specimen['illustration']}: {conditioning_error}")
                    structure_image = None

            # Generate images for each seed
            for seed in seeds:
                try:
                    print(f"    🎲 Generating with seed {seed}...")

                    images = generator.generate_realistic_specimen(
                        prompt=prompt,
                        structure_image=structure_image,
                        num_images=num_images_per_specimen,
                        seed=seed
                    )

                    # Save images
                    for img_idx, image in enumerate(images):
                        image_filename = f"{image_counter:06d}.png"
                        image_path = images_dir / image_filename
                        
                        # Ensure image path has valid extension
                        if not image_path.suffix:
                            image_path = image_path.with_suffix(".png")
                        
                        # Ensure parent directories exist
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Enhanced debug logging
                        print(f"      📋 Specimen {specimen['id']} | Mode: {mode_name} | Seed: {seed}")
                        print(f"      💾 Saving image to: {image_path}")
                        print(f"      📎 Using suffix: {image_path.suffix}")
                        
                        try:
                            image.save(image_path)
                            print(f"      ✅ Saved successfully")
                            csv_image_path = str(image_path.relative_to(project_root))
                        except Exception as save_error:
                            print(f"      ❌ Save failed: {save_error}")
                            print(f"      📍 Path: {image_path} | Suffix: {image_path.suffix}")
                            # Add to error tracking
                            save_errors.append({
                                "specimen_id": specimen['id'],
                                "species": specimen['species'],
                                "mode": mode_name,
                                "seed": seed,
                                "image_index": img_idx,
                                "intended_path": str(image_path),
                                "suffix": image_path.suffix,
                                "error": str(save_error),
                                "timestamp": str(datetime.now())
                            })
                            csv_image_path = f"ERROR: {save_error}"

                        # Add to CSV data
                        csv_data.append({
                            "specimen_id": specimen['id'],
                            "species": specimen['species'],
                            "mode": mode_name,
                            "seed": seed,
                            "prompt": prompt.replace('\n', ' | '),  # Replace newlines for CSV compatibility
                            "image_path": csv_image_path,
                            "conditioning_used": structure_image is not None,
                            "conditioning_error": conditioning_error
                        })

                        image_counter += 1

                    print(f"    ✅ Generated {len(images)} images")

                except Exception as e:
                    print(f"    ❌ Generation failed for seed {seed}: {e}")

                    # Add error entries to CSV
                    for img_idx in range(num_images_per_specimen):
                        csv_data.append({
                            "specimen_id": specimen['id'],
                            "species": specimen['species'],
                            "mode": mode_name,
                            "seed": seed,
                            "prompt": prompt.replace('\n', ' | '),  # Replace newlines for CSV compatibility
                            "image_path": f"ERROR: {str(e)}",
                            "conditioning_used": structure_image is not None,
                            "conditioning_error": conditioning_error
                        })

        print(f"📊 Mode {mode_name} complete: {image_counter} images generated")
        print()

    # Save CSV summary
    csv_path = Path("outputs/batch_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['specimen_id', 'species', 'mode', 'seed', 'prompt', 'image_path', 'conditioning_used', 'conditioning_error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    # Save error report
    if save_errors:
        error_path = Path("outputs/batch_small/errors.json")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(save_errors, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Error report saved to: {error_path}")
        print(f"🚨 Total save errors: {len(save_errors)}")

    print("📋 Batch generation complete!")
    print(f"📊 Total entries in CSV: {len(csv_data)}")
    print(f"📄 CSV saved to: {csv_path}")
    print()

    # Print summary by mode
    print("📈 Generation Summary:")
    for mode_name, _ in modes:
        mode_count = len([row for row in csv_data if row['mode'] == mode_name and not row['image_path'].startswith('ERROR')])
        print(f"  {mode_name}: {mode_count} successful generations")

    total_successful = len([row for row in csv_data if not row['image_path'].startswith('ERROR')])
    print(f"  Total: {total_successful}/{len(csv_data)} successful")

if __name__ == "__main__":
    main()