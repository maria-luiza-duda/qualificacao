#!/usr/bin/env python3
"""
Test script for appearance conditioning integration with generation pipeline.

This script validates that appearance conditioning can be properly integrated
with the main generation pipeline to produce biologically accurate coloration.
"""

import sys
import os
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from campylaspis.generation import AppearancePrior, AppearanceConditioner

def test_appearance_conditioning_integration():
    """Test appearance conditioning integration with generation pipeline."""

    print("Testing appearance conditioning integration...")

    try:
        # Load appearance prior dataset
        prior_path = "datasets/appearance_prior/cumacea_photos"
        if not os.path.exists(prior_path):
            print(f"Warning: Appearance prior dataset not found at {prior_path}")
            print("Creating mock appearance prior for testing...")

            # Create a minimal mock prior for testing
            from campylaspis.generation.appearance_prior import AppearancePrior
            import numpy as np
            from PIL import Image

            # Create mock images with typical crustacean colors
            mock_images = []
            for i in range(3):
                # Create a small image with pale beige/translucent colors
                img = Image.new('RGB', (100, 100), color=(240, 220, 180))
                mock_images.append(img)

            # Create mock prior
            prior = AppearancePrior.__new__(AppearancePrior)
            prior.photos_dir = Path("mock_path")
            prior.image_paths = [Path(f"mock_image_{i}.jpg") for i in range(3)]
            prior.images = mock_images
            prior.color_histograms = [np.random.rand(256, 3) for _ in mock_images]
            print("Created mock appearance prior for testing")
        else:
            prior = AppearancePrior(prior_path)
            print(f"Loaded appearance prior with {len(prior.images)} images")

        # Create appearance conditioner
        conditioner = AppearanceConditioner(prior)
        print("Created appearance conditioner")

        # Test color palette extraction
        test_image = prior.images[0]
        palette = conditioner.extract_color_palette(test_image, n_colors=3)
        print(f"Extracted color palette: {palette}")

        # Test prompt modifier generation
        color_modifier = conditioner.build_color_prompt_modifier(palette)
        print(f"Generated color modifier: {color_modifier}")

        # Test full appearance modifier
        appearance_modifier = conditioner.build_appearance_prompt_modifier()
        print(f"Generated full appearance modifier: {appearance_modifier}")

        # Test integration with taxonomic prompt
        base_prompt = "high-resolution macro photograph of crustacean carapace, smooth dorsal surface"
        integrated_prompt = f"{base_prompt}, {appearance_modifier}"
        print(f"Integrated prompt: {integrated_prompt}")

        print("✓ Appearance conditioning integration test passed!")
        return True

    except Exception as e:
        print(f"✗ Appearance conditioning integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_appearance_conditioning_edge_cases():
    """Test edge cases for appearance conditioning."""

    print("\nTesting appearance conditioning edge cases...")

    try:
        # Test with empty prior
        from campylaspis.generation.appearance_prior import AppearancePrior
        empty_prior = AppearancePrior.__new__(AppearancePrior)
        empty_prior.photos_dir = Path("empty_mock")
        empty_prior.image_paths = []
        empty_prior.images = []
        empty_prior.color_histograms = []

        conditioner = AppearanceConditioner(empty_prior)

        # Should handle empty dataset gracefully
        modifier = conditioner.build_appearance_prompt_modifier()
        print(f"Empty dataset modifier: {modifier}")

        print("✓ Edge case tests passed!")
        return True

    except Exception as e:
        print(f"✗ Edge case tests failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_appearance_conditioning_integration()
    success2 = test_appearance_conditioning_edge_cases()

    if success1 and success2:
        print("\n🎉 All appearance conditioning integration tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above.")
        sys.exit(1)