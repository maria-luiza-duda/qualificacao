"""
Part Annotations Module for Campylaspis Body-Part Aware Generation

This module loads and manages annotations for body parts from scientific illustrations,
enabling body-part specific image generation.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def load_part_annotations(csv_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Load part annotations from CSV file.

    CSV format:
    image_path,image_name,species,body_part,view,source

    Returns:
    {
        species: {
            body_part: [image_paths]
        }
    }
    """
    annotations = defaultdict(lambda: defaultdict(list))

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            species = row['species']
            body_part = row['body_part']
            image_path = row['image_path']

            annotations[species][body_part].append(image_path)

    # Convert defaultdict to regular dict for cleaner output
    return {species: dict(parts) for species, parts in annotations.items()}


def get_part_images(species: str, body_part: str, annotations: Dict[str, Dict[str, List[str]]]) -> List[str]:
    """
    Get all image paths for a specific species and body part.

    Args:
        species: Species name (e.g., 'aculeata')
        body_part: Body part identifier (e.g., 'pereopod_1')
        annotations: Loaded annotations dictionary

    Returns:
        List of image paths for the specified species and body part
    """
    if species not in annotations:
        return []

    if body_part not in annotations[species]:
        return []

    return annotations[species][body_part]


def get_species_parts(species: str, annotations: Dict[str, Dict[str, List[str]]]) -> List[str]:
    """
    Get all available body parts for a species.

    Args:
        species: Species name
        annotations: Loaded annotations dictionary

    Returns:
        List of body part identifiers available for the species
    """
    if species not in annotations:
        return []

    return list(annotations[species].keys())


def get_all_species(annotations: Dict[str, Dict[str, List[str]]]) -> List[str]:
    """
    Get all species available in the annotations.

    Args:
        annotations: Loaded annotations dictionary

    Returns:
        List of all species names
    """
    return list(annotations.keys())


def validate_annotations(annotations: Dict[str, Dict[str, List[str]]]) -> Dict[str, str]:
    """
    Validate the loaded annotations for consistency and completeness.

    Args:
        annotations: Loaded annotations dictionary

    Returns:
        Dictionary with validation results
    """
    validation_results = {}

    total_species = len(annotations)
    total_parts = sum(len(parts) for parts in annotations.values())
    total_images = sum(len(images) for parts in annotations.values() for images in parts.values())

    validation_results['total_species'] = f"{total_species} species found"
    validation_results['total_body_parts'] = f"{total_parts} body part categories"
    validation_results['total_images'] = f"{total_images} images total"

    # Check for common body parts across species
    all_body_parts = set()
    for parts in annotations.values():
        all_body_parts.update(parts.keys())

    validation_results['unique_body_parts'] = f"{len(all_body_parts)} unique body parts: {sorted(all_body_parts)}"

    # Check for species with missing common parts
    common_parts = {'carapace', 'pereopod_1', 'uropod'}  # Most common parts
    for species, parts in annotations.items():
        missing_parts = common_parts - set(parts.keys())
        if missing_parts:
            validation_results[f'{species}_missing'] = f"Missing parts: {missing_parts}"

    return validation_results


def print_annotations_summary(annotations: Dict[str, Dict[str, List[str]]]):
    """
    Print a summary of the loaded annotations.

    Args:
        annotations: Loaded annotations dictionary
    """
    print("📊 Part Annotations Summary")
    print("=" * 40)

    validation = validate_annotations(annotations)
    for key, value in validation.items():
        print(f"{key}: {value}")

    print("\n📋 Detailed breakdown:")
    for species, parts in sorted(annotations.items()):
        print(f"\n{species}:")
        for part, images in sorted(parts.items()):
            print(f"  {part}: {len(images)} images")
            if len(images) <= 3:  # Show image paths if few
                for img in images:
                    print(f"    - {Path(img).name}")


if __name__ == "__main__":
    # Example usage
    csv_path = "datasets/campylaspis_parts/annotations.csv"

    try:
        annotations = load_part_annotations(csv_path)
        print_annotations_summary(annotations)

        # Example queries
        print("\n🔍 Example queries:")
        print(f"Aculeata parts: {get_species_parts('aculeata', annotations)}")
        print(f"Pereopod_1 images for aculeata: {len(get_part_images('aculeata', 'pereopod_1', annotations))}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Make sure the annotations CSV exists at the expected location.")