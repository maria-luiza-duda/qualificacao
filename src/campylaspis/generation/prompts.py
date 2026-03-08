"""
Prompt Engineering for Scientific Image Generation

This module handles the creation and processing of prompts for generating
realistic organism images from taxonomic descriptions.
"""

from typing import List, Dict, Any
import re


class ScientificPromptProcessor:
    """
    Processes and generates prompts for scientific image generation.
    
    Handles taxonomic descriptions and converts them into effective
    prompts for image generation models.
    """
    
    def __init__(self):
        """Initialize the prompt processor."""
        self.taxonomic_keywords = {
            'morphology': ['scale', 'pattern', 'color', 'shape', 'size'],
            'habitat': ['forest', 'desert', 'aquatic', 'terrestrial'],
            'behavior': ['nocturnal', 'diurnal', 'arboreal', 'fossorial']
        }
    
    def process_taxonomic_description(self, description: str) -> Dict[str, Any]:
        """
        Process a taxonomic description into structured components.
        
        Args:
            description: Raw taxonomic description text
            
        Returns:
            Dictionary with processed description components
        """
        # Clean and normalize text
        clean_desc = self._clean_text(description)
        
        # Extract key morphological features
        features = self._extract_morphological_features(clean_desc)
        
        # Generate generation prompt
        prompt = self._create_generation_prompt(features)
        
        return {
            'original_description': description,
            'clean_description': clean_desc,
            'extracted_features': features,
            'generation_prompt': prompt
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize taxonomic description text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove citations and references
        text = re.sub(r'\[\d+\]|\(\d+\)', '', text)
        return text
    
    def _extract_morphological_features(self, text: str) -> Dict[str, List[str]]:
        """Extract morphological and other relevant features from text."""
        features = {
            'colors': [],
            'patterns': [],
            'shapes': [],
            'sizes': [],
            'habitats': []
        }
        
        # TODO: Implement feature extraction logic
        # This would use NLP techniques to identify morphological features
        
        return features
    
    def _create_generation_prompt(self, features: Dict[str, List[str]]) -> str:
        """Create an effective prompt for image generation."""
        # TODO: Implement prompt creation logic
        # Combine features into a coherent prompt for diffusion models
        
        base_prompt = "A realistic photograph of an organism"
        # Add features...
        
        return base_prompt
    
    def enhance_prompt(self, base_prompt: str, style: str = "photorealistic") -> str:
        """
        Enhance a base prompt with style and quality modifiers.
        
        Args:
            base_prompt: Base description prompt
            style: Desired style (photorealistic, illustration, etc.)
            
        Returns:
            Enhanced prompt
        """
        style_modifiers = {
            'photorealistic': ", highly detailed, professional photography, natural lighting",
            'scientific': ", scientific illustration style, detailed morphology, educational",
            'artistic': ", artistic rendering, vibrant colors, creative interpretation"
        }
        
        modifier = style_modifiers.get(style, style_modifiers['photorealistic'])
        return f"{base_prompt}{modifier}, high resolution, 8k"
    
    def build_taxonomic_prompt(self, species: str, description_dict: Dict[str, str]) -> str:
        """
        Build a taxonomic prompt for image generation from structured descriptions.
        
        Args:
            species: Species name (e.g., "Campylaspis sp.")
            description_dict: Dictionary with morphological descriptions
            
        Returns:
            Formatted prompt string for image generation
        """
        # Start with species identification
        prompt_parts = [f"A realistic biological specimen of {species}."]
        
        # Add general category (assuming marine crustacean for this context)
        prompt_parts.append("Marine crustacean.")
        
        # Build morphological description from dict
        morphological_parts = []
        for key, description in description_dict.items():
            if description.strip():  # Only include non-empty descriptions
                morphological_parts.append(description.strip())
        
        if morphological_parts:
            # Join with commas, but handle the last one appropriately
            if len(morphological_parts) == 1:
                prompt_parts.append(morphological_parts[0] + ".")
            else:
                prompt_parts.append(", ".join(morphological_parts[:-1]) + ", " + morphological_parts[-1] + ".")
        
        # Add style descriptors
        prompt_parts.extend([
            "Scientific illustration realism, natural lighting,",
            "neutral background, high biological fidelity."
        ])
        
        return "\n".join(prompt_parts)