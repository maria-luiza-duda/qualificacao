"""
Appearance Conditioning Module for Scientific Image Generation

This module provides functionality to condition synthetic image generation
with realistic coloration patterns derived from real crustacean photographs.
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
from PIL import Image

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    KMeans = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

try:
    from transformers import CLIPTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    CLIPTokenizer = None

from .appearance_prior import AppearancePrior


class AppearanceConditioner:
    """
    Conditions synthetic generation with realistic coloration from real photographs.
    
    Extracts color palettes from crustacean photographs and converts them into
    textual prompt modifiers for biologically plausible coloration.
    """
    
    def __init__(self, appearance_prior: AppearancePrior):
        """
        Initialize appearance conditioner with an appearance prior dataset.
        
        Args:
            appearance_prior: AppearancePrior instance with reference photographs
        """
        self.appearance_prior = appearance_prior
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for color palette extraction (pip install scikit-learn)")
        
        # Initialize CLIP tokenizer for token counting
        self.tokenizer = None
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            except Exception as e:
                print(f"⚠️  Could not load CLIP tokenizer: {e}")
        
        # Color name mappings for natural crustacean coloration
        self.color_mappings = {
            'white': ([240, 240, 240], [255, 255, 255]),
            'pale': ([220, 200, 180], [240, 220, 200]),
            'beige': ([200, 180, 150], [220, 200, 170]),
            'tan': ([180, 160, 120], [200, 180, 140]),
            'brown': ([120, 100, 80], [160, 140, 110]),
            'reddish': ([160, 100, 80], [200, 140, 120]),
            'gray': ([150, 150, 150], [180, 180, 180]),
            'translucent': ([220, 210, 200], [240, 230, 220])
        }
        
        self.texture_descriptors = [
            "natural surface texture",
            "marine specimen appearance",
            "biological realism",
            "authentic crustacean coloration",
            "realistic pigmentation patterns"
        ]
        
        # Store the last used color palette for metadata tracking
        self.last_color_palette = []
        self.color_names = {
            'base_tones': {
                'range': [(200, 255), (200, 255), (180, 255)],  # Light colors
                'names': ['pale', 'translucent', 'soft', 'light', 'creamy']
            },
            'marine_beige': {
                'range': [(180, 220), (160, 200), (120, 160)],
                'names': ['marine beige', 'sandy beige', 'oyster beige', 'shell beige']
            },
            'reddish_pigments': {
                'range': [(150, 200), (100, 150), (80, 130)],
                'names': ['reddish pigmentation', 'rosy tint', 'coral undertone', 'pinkish hue']
            },
            'brownish_tones': {
                'range': [(120, 170), (90, 140), (60, 110)],
                'names': ['brownish tone', 'earthy pigment', 'mud brown', 'sandy brown']
            },
            'greenish_hints': {
                'range': [(100, 150), (120, 170), (80, 130)],
                'names': ['greenish tint', 'olive undertone', 'seaweed hue', 'marine green']
            },
            'dark_accent': {
                'range': [(50, 100), (40, 90), (30, 80)],
                'names': ['dark accent', 'deep pigmentation', 'shadow detail', 'contrast marking']
            }
        }
        
        # Texture and pattern descriptors
        self.texture_descriptors = [
            'natural surface texture',
            'marine specimen appearance',
            'biological realism',
            'authentic crustacean coloration',
            'realistic pigmentation patterns'
        ]
    
    def extract_color_palette(self, image: Image.Image, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from an image using k-means clustering.
        
        Args:
            image: PIL Image to analyze
            n_colors: Number of dominant colors to extract
            
        Returns:
            List of RGB tuples representing dominant colors
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and reshape
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        
        # Remove any transparent or invalid pixels
        valid_pixels = pixels[~np.isnan(pixels).any(axis=1)]
        
        if len(valid_pixels) == 0:
            # Fallback to average color
            avg_color = np.mean(img_array.reshape(-1, 3), axis=0).astype(int)
            return [(avg_color[0], avg_color[1], avg_color[2])]
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=min(n_colors, len(valid_pixels)), random_state=42, n_init=10)
        kmeans.fit(valid_pixels)
        
        # Get cluster centers (dominant colors)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        # Sort by cluster size (most dominant first)
        labels = kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)  # Sort by count descending
        
        palette = []
        for idx in sorted_indices:
            color = dominant_colors[unique_labels[idx]]
            palette.append((int(color[0]), int(color[1]), int(color[2])))
        
        return palette
    
    def _classify_color(self, rgb: Tuple[int, int, int]) -> str:
        """
        Classify an RGB color into a natural crustacean color category.
        
        Args:
            rgb: RGB tuple
            
        Returns:
            Descriptive color name with crustacean-specific terminology
        """
        r, g, b = rgb
        
        # Calculate distances to color categories
        best_match = None
        best_distance = float('inf')
        
        for category, info in self.color_names.items():
            r_range, g_range, b_range = info['range']
            
            # Check if color falls within category range
            if (r_range[0] <= r <= r_range[1] and
                g_range[0] <= g <= g_range[1] and
                b_range[0] <= b <= b_range[1]):
                
                # Calculate distance from center of range
                r_center = (r_range[0] + r_range[1]) / 2
                g_center = (g_range[0] + g_range[1]) / 2
                b_center = (b_range[0] + b_range[1]) / 2
                
                distance = ((r - r_center) ** 2 + 
                           (g - g_center) ** 2 + 
                           (b - b_center) ** 2) ** 0.5
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = category
        
        if best_match:
            # Return a random name from the category with crustacean-specific terminology
            names = self.color_names[best_match]['names']
            base_name = random.choice(names)
            
            # Add crustacean-specific modifiers
            modifiers = [
                "exoskeleton", "shell", "carapace", "body", "appendages",
                "pigmentation", "coloration", "tone", "hue"
            ]
            
            # For certain categories, add specific modifiers
            if best_match == 'base_tones':
                modifier = random.choice(["exoskeleton", "shell", "body"])
            elif best_match == 'marine_beige':
                modifier = random.choice(["exoskeleton", "shell", "marine"])
            elif best_match == 'reddish_pigments':
                modifier = random.choice(["pigmentation", "coloration", "hue"])
            elif best_match == 'brownish_tones':
                modifier = random.choice(["pigmentation", "tone", "coloration"])
            elif best_match == 'greenish_hints':
                modifier = random.choice(["pigmentation", "hue", "tint"])
            elif best_match == 'dark_accent':
                modifier = random.choice(["accent", "marking", "pigmentation"])
            else:
                modifier = random.choice(modifiers)
            
            return f"{base_name} {modifier}"
        
        # Fallback: create a generic description with crustacean terminology
        intensity = (r + g + b) / 3
        if intensity > 200:
            return "pale exoskeleton"
        elif intensity > 150:
            return "medium shell tone"
        elif intensity > 100:
            return "dark pigmentation"
        else:
            return "deep marine coloration"
    
    def build_color_prompt_modifier(self, palette: List[Tuple[int, int, int]]) -> str:
        """
        Convert a color palette into a textual prompt modifier.
        
        Args:
            palette: List of RGB tuples from extract_color_palette
            
        Returns:
            Textual description of coloration for prompt modification
        """
        if not palette:
            return "natural crustacean coloration"
        
        # Classify each color in the palette
        color_descriptions = []
        for rgb in palette[:3]:  # Use top 3 colors
            description = self._classify_color(rgb)
            if description not in color_descriptions:  # Avoid duplicates
                color_descriptions.append(description)
        
        # Build the modifier string
        if not color_descriptions:
            return "natural crustacean coloration"
        
        # Start with base description
        modifier = "natural crustacean coloration"
        
        # Add color descriptions
        if color_descriptions:
            color_text = ", ".join(color_descriptions)
            modifier += f", {color_text}"
        
        # Add texture descriptor
        texture_desc = random.choice(self.texture_descriptors)
        modifier += f", {texture_desc}"
        
        return modifier
    
    def build_appearance_prompt_modifier(self, base_prompt: str = "") -> str:
        """
        Build a comprehensive appearance prompt modifier by sampling from the dataset.
        
        Samples multiple images from the appearance prior, extracts their color palettes,
        and combines them into a rich textual description of natural crustacean appearance.
        
        Args:
            base_prompt: Base prompt to enhance with appearance conditioning
            
        Returns:
            Enhanced prompt with appearance conditioning
        """
        # Get the appearance modifier
        appearance_modifier = self._build_appearance_modifier()
        
        # If no base prompt, return just the modifier
        if not base_prompt or base_prompt.strip() == "":
            return appearance_modifier
        
        # Combine with base prompt, ensuring token limits
        return self._ensure_token_limit(base_prompt, appearance_modifier, max_tokens=77)
    
    def _build_appearance_modifier(self) -> str:
        """
        Internal method to build the appearance modifier string.
        
        Returns:
            Appearance modifier string
        """
        # Sample multiple reference images
        n_samples = min(5, len(self.appearance_prior))  # Sample up to 5 images
        reference_images = self.appearance_prior.sample_reference_images(n=n_samples)
        
        if not reference_images:
            return "natural crustacean coloration, realistic marine specimen appearance"
        
        # Extract palettes from all sampled images
        all_palettes = []
        for img in reference_images:
            try:
                palette = self.extract_color_palette(img, n_colors=3)
                all_palettes.append(palette)
            except Exception as e:
                print(f"⚠️  Failed to extract palette from reference image: {e}")
                continue
        
        # Close the images to free memory
        for img in reference_images:
            img.close()
        
        if not all_palettes:
            return "natural crustacean coloration, realistic marine specimen appearance"
        
        # Collect all unique color descriptions
        all_color_descriptions = set()
        for palette in all_palettes:
            color_desc = self.build_color_prompt_modifier(palette)
            # Extract individual color terms
            parts = color_desc.split(", ")
            for part in parts[1:]:  # Skip "natural crustacean coloration"
                if part not in ["natural surface texture", "marine specimen appearance", 
                               "biological realism", "authentic crustacean coloration",
                               "realistic pigmentation patterns"]:
                    all_color_descriptions.add(part)
        
        # Build comprehensive modifier
        modifier = "natural crustacean coloration"
        
        # Add diverse color descriptions (limit to avoid overly long prompts)
        color_list = list(all_color_descriptions)[:4]  # Limit to 4 color descriptions
        if color_list:
            modifier += ", " + ", ".join(color_list)
        
        # Add texture and realism descriptors
        modifier += ", realistic marine specimen appearance, authentic pigmentation patterns"
        
        # Store the combined color palette for metadata (flatten and deduplicate)
        combined_palette = []
        for palette in all_palettes:
            for color in palette:
                if color not in combined_palette:
                    combined_palette.append(color)
        self.last_color_palette = combined_palette[:5]  # Store up to 5 representative colors
        
        return modifier
    
    def get_color_variation_modifier(self, base_modifier: str = "", variation_strength: float = 0.3) -> str:
        """
        Generate a color variation modifier for introducing natural variation.
        
        Args:
            base_modifier: Base appearance modifier to vary from
            variation_strength: How much variation to introduce (0.0-1.0)
            
        Returns:
            Modified prompt with natural color variation
        """
        if not base_modifier:
            base_modifier = self.build_appearance_prompt_modifier()
        
        # Add variation terms based on strength
        variation_terms = []
        
        if variation_strength > 0.7:
            variation_terms.extend([
                "with natural color variation",
                "subtle individual differences",
                "authentic specimen variation"
            ])
        elif variation_strength > 0.4:
            variation_terms.extend([
                "slight color variation",
                "natural pigmentation differences",
                "individual specimen characteristics"
            ])
        elif variation_strength > 0.1:
            variation_terms.extend([
                "subtle pigmentation variation",
                "natural color differences",
                "authentic coloration variation"
            ])
        
        if variation_terms:
            variation = random.choice(variation_terms)
            return f"{base_modifier}, {variation}"
        
        return base_modifier
    
    def _count_clip_tokens(self, text: str) -> int:
        """
        Count the number of CLIP tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of CLIP tokens
        """
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                print(f"⚠️  Token counting failed: {e}")
        
        # Fallback: rough estimate based on word count (CLIP tokens are usually 1-2 per word)
        words = text.split()
        return len(words) * 2  # Conservative estimate
    
    def _ensure_token_limit(self, base_prompt: str, appearance_modifier: str, max_tokens: int = 77) -> str:
        """
        Ensure the combined prompt stays within CLIP token limits.
        
        Args:
            base_prompt: Base morphological prompt
            appearance_modifier: Appearance conditioning text
            max_tokens: Maximum allowed tokens (default 77 for CLIP)
            
        Returns:
            Combined prompt within token limits
        """
        # Start with base prompt
        combined = base_prompt
        
        # Count base prompt tokens
        base_tokens = self._count_clip_tokens(base_prompt)
        remaining_tokens = max_tokens - base_tokens
        
        if remaining_tokens <= 0:
            print(f"⚠️  Base prompt already at token limit ({base_tokens}/{max_tokens})")
            return base_prompt
        
        # Split appearance modifier into components
        parts = appearance_modifier.split(", ")
        if not parts:
            return combined
        
        # Base appearance description (usually first part)
        base_appearance = parts[0]
        base_appearance_tokens = self._count_clip_tokens(base_appearance)
        
        if base_appearance_tokens > remaining_tokens:
            print(f"⚠️  Cannot add appearance conditioning - insufficient tokens ({remaining_tokens} remaining)")
            return combined
        
        # Add base appearance
        combined = f"{combined}, {base_appearance}"
        remaining_tokens -= base_appearance_tokens
        
        # Add additional color descriptors if space allows
        additional_parts = parts[1:]
        for part in additional_parts:
            part_tokens = self._count_clip_tokens(part)
            if part_tokens <= remaining_tokens:
                combined = f"{combined}, {part}"
                remaining_tokens -= part_tokens
            else:
                break
        
        final_tokens = self._count_clip_tokens(combined)
        print(f"📝 Final prompt tokens: {final_tokens}/{max_tokens}")
        
        return combined
    
    def get_last_color_palette(self) -> List[List[int]]:
        """
        Get the last color palette used for appearance conditioning.
        
        Returns:
            List of RGB color tuples from the last appearance modifier generation
        """
        return self.last_color_palette.copy()
    
    def __repr__(self) -> str:
        return f"AppearanceConditioner(appearance_prior={self.appearance_prior})"