"""
Scientific Image Generation Module

Provides tools for generating realistic 2D organism images from
scientific illustrations and taxonomic descriptions.
"""

__all__ = []

try:
    from .generator import ScientificImageGenerator
    __all__.append('ScientificImageGenerator')
except ImportError:
    pass

try:
    from .prompts import ScientificPromptProcessor
    __all__.append('ScientificPromptProcessor')
except ImportError:
    pass

try:
    from .conditioning import ScientificConditioningProcessor
    __all__.append('ScientificConditioningProcessor')
except ImportError:
    pass

try:
    from .appearance_prior import AppearancePrior
    __all__.append('AppearancePrior')
except ImportError:
    pass

try:
    from .appearance_conditioning import AppearanceConditioner
    __all__.append('AppearanceConditioner')
except ImportError:
    pass

try:
    from .pipeline import generate_dataset_from_taxonomy
    __all__.append('generate_dataset_from_taxonomy')
except ImportError:
    pass

try:
    from .morphology_parser import MorphologyParser, build_morphology_prompt
    __all__.extend(['MorphologyParser', 'build_morphology_prompt'])
except ImportError:
    pass

try:
    from .diffusion_renderer import MorphologyGuidedRenderer
    __all__.append('MorphologyGuidedRenderer')
except ImportError:
    pass

if len(__all__) < 11:
    print(f"Warning: Only {len(__all__)} out of 11 generation module components could be imported due to missing dependencies")