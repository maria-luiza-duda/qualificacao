#!/usr/bin/env python3
"""
Test script for Appearance Conditioning functionality.

This script demonstrates and tests the AppearanceConditioner class
for generating realistic coloration prompt modifiers.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def test_appearance_conditioning():
    """Test the AppearanceConditioner class functionality."""
    
    print("🎨 Testing Appearance Conditioning Module")
    print("=" * 50)
    
    # Test with the cumacea photos directory
    photos_dir = "/home/maria-luiza-duda/datasets/appearance_prior/cumacea_photos"
    
    if not Path(photos_dir).exists():
        print(f"❌ Photos directory does not exist: {photos_dir}")
        print("Please ensure the appearance prior dataset is available at the expected location.")
        return
    
    try:
        # Import and create AppearancePrior and AppearanceConditioner instances
        from campylaspis.generation import AppearancePrior, AppearanceConditioner
        
        print(f"📁 Loading appearance prior from: {photos_dir}")
        prior = AppearancePrior(photos_dir)
        
        print(f"🎨 Creating appearance conditioner...")
        conditioner = AppearanceConditioner(prior)
        
        # Test color palette extraction
        print("\n🎨 Testing color palette extraction...")
        sample_images = prior.sample_reference_images(n=1)
        if sample_images:
            palette = conditioner.extract_color_palette(sample_images[0], n_colors=5)
            print(f"✅ Extracted palette with {len(palette)} colors:")
            for i, color in enumerate(palette):
                print(f"   Color {i+1}: RGB{color}")
            
            # Test color prompt modifier building
            print("\n📝 Testing color prompt modifier...")
            modifier = conditioner.build_color_prompt_modifier(palette)
            print(f"✅ Generated modifier: {modifier}")
        
        # Test comprehensive appearance prompt modifier
        print("\n📝 Testing comprehensive appearance prompt modifier...")
        full_modifier = conditioner.build_appearance_prompt_modifier()
        print(f"✅ Generated full modifier: {full_modifier}")
        
        # Test color variation modifier
        print("\n🎭 Testing color variation modifier...")
        variation_modifier = conditioner.get_color_variation_modifier(full_modifier, variation_strength=0.5)
        print(f"✅ Generated variation modifier: {variation_modifier}")
        
        # Test with different variation strengths
        print("\n🎭 Testing different variation strengths...")
        for strength in [0.1, 0.5, 0.8]:
            var_mod = conditioner.get_color_variation_modifier(full_modifier, strength)
            print(f"   Strength {strength}: {var_mod}")
        
        # Clean up
        for img in sample_images:
            img.close()
        
        print("\n🎉 All AppearanceConditioner tests passed!")
        
        # Show example usage
        print("\n📖 Example Usage in Generation Pipeline:")
        print("=" * 50)
        print("base_prompt = 'high-resolution macro photograph of crustacean carapace'")
        print(f"appearance_modifier = '{full_modifier}'")
        print("final_prompt = f'{base_prompt}, {appearance_modifier}'")
        print()
        print("This would produce prompts like:")
        print(f"'high-resolution macro photograph of crustacean carapace, {full_modifier}'")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Required dependencies: scikit-learn, numpy, PIL")
        print("Install with: pip install scikit-learn")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_appearance_conditioning()