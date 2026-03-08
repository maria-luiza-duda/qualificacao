#!/usr/bin/env python3
"""
Example usage of the scientific image generation pipeline.

This script demonstrates how to use the generate_dataset_from_taxonomy function
to create a dataset of realistic specimen images from taxonomic descriptions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from campylaspis.generation import generate_dataset_from_taxonomy, AppearancePrior, AppearanceConditioner

def main():
    """Example pipeline usage."""

    # Example illustration paths (these would be actual image files)
    illustration_paths = [
        "data/illustrations/campylaspis_01.png",
        "data/illustrations/campylaspis_02.png",
    ]

    # Example taxonomic descriptions
    description_texts = [
        {
            "carapace": "Smooth carapace with rounded margins",
            "pseudorostrum": "elongated pseudorostrum",
            "pereopods": "slender pereopods with long setae",
            "uropods": "uropods with elongated exopod and shorter endopod"
        },
        {
            "carapace": "Oval carapace",
            "pseudorostrum": "short pseudorostrum",
            "pereopods": "robust pereopods",
            "uropods": "broad uropods with equal rami"
        }
    ]

    # Convert dict descriptions to strings for the pipeline
    description_strings = []
    for desc_dict in description_texts:
        desc_parts = [f"{key}: {value}" for key, value in desc_dict.items()]
        description_strings.append(". ".join(desc_parts))

    # Output directory
    output_dir = "generated_dataset"

    # Run the pipeline with structure-guided generation
    try:
        metadata = generate_dataset_from_taxonomy(
            illustration_paths=illustration_paths,
            description_texts=description_strings,
            output_dir=output_dir,
            num_images_per_specimen=2,  # Generate 2 images per specimen
            seed=42,
            generation_mode="structure_guided",  # Use structure-guided generation
            structure_strength=0.95,  # High conditioning strength
            appearance_prior_dir="datasets/appearance_prior/cumacea_photos"  # Optional appearance conditioning
        )

        print("Pipeline completed successfully!")
        print(f"Generated {metadata['dataset_info']['total_images']} images")
        print(f"Generation mode: {metadata['generation_config']['generation_mode']}")
        print(f"Structure strength: {metadata['generation_config']['structure_strength']}")
        print(f"Appearance prior used: {metadata['generation_config']['appearance_prior_used']}")
        print(f"Appearance prior dataset: {metadata['generation_config']['appearance_prior_dataset']}")
        print(f"Reference images sampled: {metadata['generation_config']['reference_images_sampled']}")
        print(f"Color palette used: {metadata['generation_config']['color_palette_used']}")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        print("This is expected if required dependencies (torch, diffusers, etc.) are not installed")


def example_body_part_generation():
    """Example of body-part specific generation."""
    
    # Example body part illustration paths with specific naming
    body_part_illustrations = [
        "data/illustrations/species_p1.png",        # First pereopod
        "data/illustrations/species_carapace.png",  # Carapace
        "data/illustrations/species_uropod.png",    # Uropod
    ]
    
    # Body part descriptions (can be minimal since body part is detected from filename)
    body_part_descriptions = [
        "First pereopod with specialized setae",
        "Carapace dorsal view with surface texture",
        "Uropod biramous structure"
    ]
    
    # Output directory for body parts
    output_dir = "generated_body_parts"
    
    try:
        metadata = generate_dataset_from_taxonomy(
            illustration_paths=body_part_illustrations,
            description_texts=body_part_descriptions,
            output_dir=output_dir,
            num_images_per_specimen=3,  # Generate 3 images per body part
            seed=123,
            generation_mode="body_part",  # Use body-part specific generation
            structure_strength=0.8,  # Moderate conditioning for body parts
            appearance_prior_dir="datasets/appearance_prior/cumacea_photos"  # Optional appearance conditioning
        )
        
        print("Body part generation completed successfully!")
        print(f"Generated {metadata['dataset_info']['total_images']} images")
        print(f"Generation mode: {metadata['generation_config']['generation_mode']}")
        
        # Show detected body parts
        for specimen in metadata['specimens']:
            if 'body_part' in specimen:
                bp = specimen['body_part']
                print(f"Body part: {bp['identifier']} - {bp['description']}")
        
        print(f"Appearance prior used: {metadata['generation_config']['appearance_prior_used']}")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Body part generation failed: {e}")
        print("This is expected if required dependencies are not installed")


def example_appearance_conditioning():
    """Example of appearance-conditioned generation using real photograph colors."""
    
    # Load appearance prior dataset
    try:
        prior = AppearancePrior("datasets/appearance_prior/cumacea_photos")
        conditioner = AppearanceConditioner(prior)
        print(f"Loaded appearance prior with {len(prior.images)} reference images")
        
        # Generate appearance modifier
        appearance_modifier = conditioner.build_appearance_prompt_modifier()
        print(f"Generated appearance modifier: {appearance_modifier}")
        
        # Example illustration paths
        illustration_paths = [
            "data/illustrations/campylaspis_01.png",
        ]
        
        # Base taxonomic description
        base_description = "Smooth carapace with rounded margins, elongated pseudorostrum, slender pereopods with long setae"
        
        # Combine with appearance conditioning
        conditioned_description = f"{base_description}, {appearance_modifier}"
        
        # Output directory
        output_dir = "generated_appearance_conditioned"
        
        metadata = generate_dataset_from_taxonomy(
            illustration_paths=illustration_paths,
            description_texts=[conditioned_description],
            output_dir=output_dir,
            num_images_per_specimen=2,
            seed=456,
            generation_mode="structure_guided",
            structure_strength=0.9,
            appearance_prior_dir="datasets/appearance_prior/cumacea_photos"  # Enable appearance conditioning
        )
        
        print("Appearance-conditioned generation completed successfully!")
        print(f"Generated {metadata['dataset_info']['total_images']} images")
        print(f"Appearance modifier applied: {appearance_modifier}")
        print(f"Appearance prior used: {metadata['generation_config']['appearance_prior_used']}")
        print(f"Appearance prior dataset: {metadata['generation_config']['appearance_prior_dataset']}")
        print(f"Reference images sampled: {metadata['generation_config']['reference_images_sampled']}")
        print(f"Color palette used: {metadata['generation_config']['color_palette_used']}")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Appearance conditioning example failed: {e}")
        print("This may be due to missing appearance prior dataset or dependencies")


if __name__ == "__main__":
    print("Running structure-guided generation example...")
    main()
    print("\n" + "="*60 + "\n")
    print("Running body-part generation example...")
    example_body_part_generation()
    print("\n" + "="*60 + "\n")
    print("Running appearance conditioning example...")
    example_appearance_conditioning()