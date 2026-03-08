#!/usr/bin/env python3
"""
Test script for appearance conditioning metadata generation.

This script validates that the metadata.json includes all appearance conditioning
information for scientific reproducibility.
"""

import sys
import os
import json
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_metadata_structure():
    """Test that metadata includes all required appearance conditioning fields."""

    from campylaspis.generation import generate_dataset_from_taxonomy

    print("Testing metadata structure with appearance conditioning...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with appearance conditioning (will fail gracefully if dataset doesn't exist)
            result = generate_dataset_from_taxonomy(
                illustration_paths=None,
                description_texts=['Test crustacean with carapace and pereopods'],
                output_dir=tmpdir,
                num_images_per_specimen=1,
                seed=42,
                generation_mode='text_only',
                appearance_prior_dir='datasets/appearance_prior/cumacea_photos'
            )

            # Check returned metadata structure
            config = result['generation_config']

            required_fields = [
                'appearance_prior_used',
                'appearance_prior_dataset',
                'reference_images_sampled',
                'color_palette_used'
            ]

            for field in required_fields:
                if field not in config:
                    print(f"❌ Missing field: {field}")
                    return False
                print(f"✅ Field present: {field} = {config[field]}")

            # Validate field types
            assert isinstance(config['appearance_prior_used'], bool), "appearance_prior_used should be boolean"
            assert isinstance(config['reference_images_sampled'], int), "reference_images_sampled should be int"
            assert isinstance(config['color_palette_used'], list), "color_palette_used should be list"

            if config['appearance_prior_dataset'] is not None:
                assert isinstance(config['appearance_prior_dataset'], str), "appearance_prior_dataset should be string or None"

            print("✅ All metadata fields have correct types")

            # Check saved metadata.json
            metadata_path = os.path.join(tmpdir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    saved_metadata = json.load(f)

                saved_config = saved_metadata['generation_config']
                for field in required_fields:
                    if field not in saved_config:
                        print(f"❌ Missing field in saved metadata: {field}")
                        return False

                print("✅ metadata.json saved with all appearance conditioning fields")
            else:
                print("❌ metadata.json not found")
                return False

            return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that existing code without appearance conditioning still works."""

    from campylaspis.generation import generate_dataset_from_taxonomy

    print("\nTesting backward compatibility...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test without appearance conditioning
            result = generate_dataset_from_taxonomy(
                illustration_paths=None,
                description_texts=['Test crustacean description'],
                output_dir=tmpdir,
                num_images_per_specimen=1,
                seed=42,
                generation_mode='text_only'
            )

            config = result['generation_config']

            # Check that fields exist with default values
            assert config['appearance_prior_used'] == False
            assert config['appearance_prior_dataset'] is None
            assert config['reference_images_sampled'] == 0
            assert config['color_palette_used'] == []

            print("✅ Backward compatibility maintained")
            return True

    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_metadata_structure()
    success2 = test_backward_compatibility()

    if success1 and success2:
        print("\n🎉 All metadata tests passed!")
        print("Scientific reproducibility enabled with comprehensive appearance conditioning metadata.")
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)