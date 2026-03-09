#!/usr/bin/env python3
"""
Campylaspis Generation Package Integrity Check

Verifies that all morphology-guided generation components are correctly imported and connected.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def check_import(module_path: str, name: str) -> bool:
    """Check if a module/component can be imported."""
    try:
        __import__(module_path)
        print(f"✅ {name}: Available")
        return True
    except ImportError as e:
        print(f"❌ {name}: Not available - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {name}: Import error - {e}")
        return False

def check_dependency(package_name: str, import_name: str = None) -> bool:
    """Check if a Python package dependency is installed."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✅ {package_name}: Installed")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False
    except Exception as e:
        print(f"⚠️  {package_name}: Import error - {e}")
        return False

def check_generation_modes():
    """Check available generation modes."""
    try:
        from campylaspis.generation.pipeline import generate_dataset_from_taxonomy
        import inspect

        # Get the function signature to see supported modes
        sig = inspect.signature(generate_dataset_from_taxonomy)
        if 'generation_mode' in sig.parameters:
            param = sig.parameters['generation_mode']
            if hasattr(param, 'default') and param.default is not inspect.Parameter.empty:
                print(f"✅ Generation modes: Available (default: {param.default})")
                return True
            else:
                print(f"✅ Generation modes: Available (parameter exists)")
                return True
        else:
            print(f"❌ Generation modes: Parameter not found")
            return False
    except Exception as e:
        print(f"❌ Generation modes: Error checking - {e}")
        return False

def check_renderer_classes():
    """Check available renderer classes."""
    try:
        from campylaspis.generation import generator
        import inspect

        # Check for render methods in ScientificImageGenerator
        if hasattr(generator, 'ScientificImageGenerator'):
            cls = generator.ScientificImageGenerator
            render_methods = [method for method in dir(cls) if method.startswith('render_')]
            if render_methods:
                print(f"✅ Renderer classes: Available ({len(render_methods)} render methods: {', '.join(render_methods)})")
                return True
            else:
                print(f"❌ Renderer classes: No render methods found")
                return False
        else:
            print(f"❌ Renderer classes: ScientificImageGenerator not found")
            return False
    except Exception as e:
        print(f"❌ Renderer classes: Error checking - {e}")
        return False

def main():
    """Run the integrity check."""
    print("🔍 Campylaspis Generation Package Integrity Check")
    print("=" * 60)

    all_checks_passed = True

    # Check generation modes
    if not check_generation_modes():
        all_checks_passed = False

    # Check renderer classes
    if not check_renderer_classes():
        all_checks_passed = False

    # Check core components
    components = [
        ('campylaspis.generation.pipeline', 'generate_dataset_from_taxonomy'),
        ('campylaspis.generation.diffusion_renderer', 'MorphologyGuidedRenderer'),
        ('campylaspis.generation.prompts', 'ScientificPromptProcessor'),
        ('campylaspis.generation.conditioning', 'ScientificConditioningProcessor'),
        ('campylaspis.generation.appearance_prior', 'AppearancePrior'),
        ('campylaspis.generation.appearance_conditioning', 'AppearanceConditioner'),
    ]

    for module_path, component_name in components:
        if not check_import(module_path, component_name):
            all_checks_passed = False

    # Check required dependencies
    dependencies = [
        ('torch', 'torch'),
        ('diffusers', 'diffusers'),
        ('transformers', 'transformers'),
        ('numpy', 'numpy'),
        ('cv2', 'cv2'),
        ('PIL', 'PIL'),
    ]

    print("\n📦 Required Dependencies:")
    for package_name, import_name in dependencies:
        if not check_dependency(package_name, import_name):
            all_checks_passed = False

    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ INTEGRITY CHECK PASSED")
        return 0
    else:
        print("❌ INTEGRITY CHECK FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())