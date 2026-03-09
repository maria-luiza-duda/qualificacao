#!/usr/bin/env python3
"""
Test script for the Morphology Parser

Demonstrates the functionality of the new morphology parser module.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from campylaspis.generation.morphology_parser import (
    MorphologyParser,
    build_morphology_prompt,
    parse_morphology_from_text
)

def test_morphology_parser():
    """Test the morphology parser with sample taxonomic descriptions."""

    print("🔬 Morphology Parser Test")
    print("=" * 50)

    # Sample taxonomic description
    sample_description = """
    carapace: Smooth carapace with rounded margins.
    pseudorostrum: elongated pseudorostrum extending forward.
    pereopods: slender pereopods adapted for swimming.
    uropods: uropods with elongated exopod and shorter endopod.
    antennae: long segmented antennae for sensory functions.
    """

    print("📝 Input taxonomic description:")
    print(sample_description.strip())
    print()

    # Test parsing
    print("🔍 Parsing morphological constraints...")
    constraints = parse_morphology_from_text(sample_description)

    print("📊 Extracted morphological constraints:")
    for body_part, descriptors in constraints.items():
        print(f"   {body_part}: {', '.join(descriptors)}")
    print()

    # Test prompt building
    species_name = "Campylaspis aculeata"
    print(f"🎨 Building prompt for species: {species_name}")
    prompt = build_morphology_prompt(species_name, constraints)

    print("📝 Generated morphological prompt:")
    print(prompt)
    print()

    # Test validation
    parser = MorphologyParser()
    validation = parser.validate_constraints(constraints)

    print("✅ Validation results:")
    print(f"   Valid: {validation['valid']}")
    if validation['warnings']:
        print("   Warnings:")
        for warning in validation['warnings']:
            print(f"     - {warning}")
    if validation['suggestions']:
        print("   Suggestions:")
        for suggestion in validation['suggestions']:
            print(f"     - {suggestion}")

    print("\n🎉 Morphology parser test completed successfully!")

if __name__ == "__main__":
    test_morphology_parser()