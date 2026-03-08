#!/usr/bin/env python3
"""
Test script for Appearance Prior functionality.

This script demonstrates and tests the AppearancePrior class
for guiding realistic coloration in synthetic image generation.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def test_appearance_prior():
    """Test the AppearancePrior class functionality."""
    
    print("🖼️  Testing Appearance Prior Module")
    print("=" * 50)
    
    # Test with the cumacea photos directory
    photos_dir = "/home/maria-luiza-duda/datasets/appearance_prior/cumacea_photos"
    
    if not Path(photos_dir).exists():
        print(f"❌ Photos directory does not exist: {photos_dir}")
        print("Please ensure the appearance prior dataset is available at the expected location.")
        return
    
    try:
        # Import and create AppearancePrior instance
        from campylaspis.generation import AppearancePrior
        
        print(f"📁 Loading appearance prior from: {photos_dir}")
        prior = AppearancePrior(photos_dir)
        
        print(f"✅ Loaded {len(prior)} reference images")
        
        # Test sampling reference images
        print("\n🖼️  Testing reference image sampling...")
        sample_images = prior.sample_reference_images(n=3)
        print(f"✅ Sampled {len(sample_images)} reference images")
        
        for i, img in enumerate(sample_images):
            print(f"   Image {i+1}: {img.size} {img.mode}")
        
        # Test color histogram computation
        print("\n📊 Testing color histogram computation...")
        if sample_images:
            hist = prior.compute_color_histogram(sample_images[0])
            print(f"✅ Computed histogram with shape: {hist.shape}")
            print(f"   Histogram range: [{hist.min():.3f}, {hist.max():.3f}]")
        
        # Test dataset color statistics
        print("\n📈 Testing dataset color statistics...")
        stats = prior.compute_dataset_color_statistics()
        print(f"✅ Computed statistics for {stats['num_images']} images")
        print(f"   Mean histogram shape: {stats['mean_histogram'].shape}")
        print(f"   Mean histogram range: [{stats['mean_histogram'].min():.3f}, {stats['mean_histogram'].max():.3f}]")
        
        # Test color distribution sampling
        print("\n🎨 Testing color distribution sampling...")
        color_samples = prior.get_color_distribution_sample(n_samples=10)
        print(f"✅ Sampled {len(color_samples)} colors")
        print(f"   Color sample shape: {color_samples.shape}")
        print(f"   Color range: [{color_samples.min()}, {color_samples.max()}]")
        
        # Test texture references
        print("\n🖼️  Testing texture reference sampling...")
        texture_refs = prior.get_texture_references(n=2)
        print(f"✅ Sampled {len(texture_refs)} texture references")
        
        print("\n🎉 All AppearancePrior tests passed!")
        
        # Clean up
        for img in sample_images + texture_refs:
            img.close()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("The appearance_prior module could not be imported.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_appearance_prior()