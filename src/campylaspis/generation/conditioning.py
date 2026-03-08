"""
Conditioning Module for Scientific Image Generation

This module handles conditioning inputs for image generation,
including processing scientific illustrations and taxonomic data.
"""

from typing import Optional, Tuple, Dict, Any
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
from PIL import Image, ImageFilter
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def validate_illustration_file(image_path: str) -> Tuple[bool, str]:
    """
    Validate that an illustration file can be loaded by PIL.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (is_valid, reason)
    """
    import os
    from pathlib import Path
    
    path = Path(image_path)
    
    # Check 1: File exists
    if not path.exists():
        return False, f"File does not exist: {image_path}"
    
    # Check 2: File size > 0
    if path.stat().st_size == 0:
        return False, f"File is empty (0 bytes): {image_path}"
    
    # Check 3: PIL can open and verify it
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify the file is a valid image
        return True, "Valid image file"
    except Exception as e:
        return False, f"PIL cannot identify image file: {e}"


def detect_body_part_from_filename(filename: str) -> Tuple[str, str]:
    """
    Detect body part from filename patterns.
    
    Args:
        filename: Filename to analyze (e.g., "species_carapace.png", "species_p1.png")
        
    Returns:
        Tuple of (body_part, description) where body_part is a normalized identifier
        and description is a human-readable name
    """
    from pathlib import Path
    
    # Extract filename without extension
    stem = Path(filename).stem.lower()
    
    # Common body part patterns
    body_part_patterns = {
        # Pereopods (walking legs)
        'p1': ('pereopod_1', 'first pereopod'),
        'p2': ('pereopod_2', 'second pereopod'),
        'p3': ('pereopod_3', 'third pereopod'),
        'p4': ('pereopod_4', 'fourth pereopod'),
        'p5': ('pereopod_5', 'fifth pereopod'),
        'pereopod': ('pereopod', 'pereopod'),
        'walking_leg': ('pereopod', 'pereopod'),
        
        # Carapace and body parts
        'carapace': ('carapace', 'carapace'),
        'cephalothorax': ('cephalothorax', 'cephalothorax'),
        'abdomen': ('abdomen', 'abdomen'),
        'pleon': ('pleon', 'pleon'),
        
        # Appendages
        'uropod': ('uropod', 'uropod'),
        'uropods': ('uropod', 'uropod'),
        'telson': ('telson', 'telson'),
        'antenna': ('antenna', 'antenna'),
        'antennule': ('antennule', 'antennule'),
        'maxilliped': ('maxilliped', 'maxilliped'),
        
        # Body views
        'lateral': ('lateral_body', 'lateral body view'),
        'dorsal': ('dorsal_body', 'dorsal body view'),
        'ventral': ('ventral_body', 'ventral body view'),
        'body': ('body', 'body'),
        
        # Rostrum and head parts
        'rostrum': ('rostrum', 'rostrum'),
        'pseudorostrum': ('pseudorostrum', 'pseudorostrum'),
        'eye': ('eye', 'eye'),
        'compound_eye': ('compound_eye', 'compound eye'),
    }
    
    # Check for exact matches first
    for pattern, (body_part, description) in body_part_patterns.items():
        if pattern in stem:
            return body_part, description
    
    # Check for numbered patterns (p1, p2, etc.)
    import re
    numbered_match = re.search(r'p(\d+)', stem)
    if numbered_match:
        num = numbered_match.group(1)
        return f'pereopod_{num}', f'pereopod {num}'
    
    # Default fallback
    return 'unknown', 'unknown body part'


def build_body_part_prompt(body_part: str, species_name: str = "crustacean", 
                          base_description: str = "") -> str:
    """
    Build a specialized prompt for body part generation.
    
    Args:
        body_part: Body part identifier (e.g., 'pereopod_1', 'carapace')
        species_name: Species name for context
        base_description: Additional description from taxonomic data
        
    Returns:
        Specialized prompt for body part generation
    """
    # Body part specific prompts
    body_part_prompts = {
        'pereopod_1': "high-resolution macro photograph of crustacean first pereopod anatomy, marine specimen, scientific realism, detailed setae, joint articulation, exopod endopod structure",
        'pereopod_2': "high-resolution macro photograph of crustacean second pereopod anatomy, marine specimen, scientific realism, detailed setae, joint articulation, exopod endopod structure",
        'pereopod_3': "high-resolution macro photograph of crustacean third pereopod anatomy, marine specimen, scientific realism, detailed setae, joint articulation, exopod endopod structure",
        'pereopod_4': "high-resolution macro photograph of crustacean fourth pereopod anatomy, marine specimen, scientific realism, detailed setae, joint articulation, exopod endopod structure",
        'pereopod_5': "high-resolution macro photograph of crustacean fifth pereopod anatomy, marine specimen, scientific realism, detailed setae, joint articulation, exopod endopod structure",
        'pereopod': "high-resolution macro photograph of crustacean pereopod anatomy, marine specimen, scientific realism, detailed setae, joint articulation, exopod endopod structure",
        
        'carapace': "high-resolution macro photograph of crustacean carapace morphology, marine specimen, scientific realism, detailed surface texture, ridge patterns, scientific illustration style",
        'cephalothorax': "high-resolution macro photograph of crustacean cephalothorax anatomy, marine specimen, scientific realism, detailed segmentation, appendage attachment points",
        'abdomen': "high-resolution macro photograph of crustacean abdomen morphology, marine specimen, scientific realism, detailed pleura, tergites, scientific illustration",
        'pleon': "high-resolution macro photograph of crustacean pleon anatomy, marine specimen, scientific realism, detailed somites, uropod attachment",
        
        'uropod': "high-resolution macro photograph of crustacean uropod anatomy, marine specimen, scientific realism, detailed rami structure, setae distribution, biramous appendage",
        'telson': "high-resolution macro photograph of crustacean telson morphology, marine specimen, scientific realism, detailed posterior structure, scientific illustration",
        'antenna': "high-resolution macro photograph of crustacean antenna anatomy, marine specimen, scientific realism, detailed flagellum, peduncle, sensory setae",
        'antennule': "high-resolution macro photograph of crustacean antennule anatomy, marine specimen, scientific realism, detailed segments, aesthetascs",
        'maxilliped': "high-resolution macro photograph of crustacean maxilliped anatomy, marine specimen, scientific realism, detailed endites, palp structure",
        
        'rostrum': "high-resolution macro photograph of crustacean rostrum morphology, marine specimen, scientific realism, detailed anterior projection, scientific illustration",
        'pseudorostrum': "high-resolution macro photograph of crustacean pseudorostrum morphology, marine specimen, scientific realism, detailed anterior projection, scientific illustration",
        'eye': "high-resolution macro photograph of crustacean compound eye anatomy, marine specimen, scientific realism, detailed ommatidia, scientific illustration",
        'compound_eye': "high-resolution macro photograph of crustacean compound eye anatomy, marine specimen, scientific realism, detailed ommatidia, scientific illustration",
        
        'lateral_body': "high-resolution macro photograph of crustacean lateral body morphology, marine specimen, scientific realism, detailed body outline, appendage positioning",
        'dorsal_body': "high-resolution macro photograph of crustacean dorsal body morphology, marine specimen, scientific realism, detailed carapace structure, scientific illustration",
        'ventral_body': "high-resolution macro photograph of crustacean ventral body morphology, marine specimen, scientific realism, detailed sternites, appendage bases",
        'body': "high-resolution macro photograph of crustacean body morphology, marine specimen, scientific realism, detailed external anatomy, scientific illustration",
    }
    
    # Get base prompt for body part
    base_prompt = body_part_prompts.get(body_part, 
                                       f"high-resolution macro photograph of crustacean {body_part.replace('_', ' ')} anatomy, marine specimen, scientific realism")
    
    # Add species context if provided
    if species_name and species_name != "crustacean":
        base_prompt = base_prompt.replace("crustacean", f"{species_name} crustacean")
    
    # Add base description if provided
    if base_description:
        base_prompt += f", {base_description}"
    
    # Add consistent quality modifiers
    base_prompt += ", professional scientific photography, high detail, sharp focus, uniform lighting, taxonomic reference quality"
    
    return base_prompt


class ScientificConditioningProcessor:
    """
    Processes conditioning inputs for scientific image generation.
    
    Handles:
    - Scientific illustration preprocessing
    - Taxonomic feature extraction
    - Conditioning vector creation
    """
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the conditioning processor.
        
        Args:
            image_size: Target size for processed images
        """
        self.image_size = image_size
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
    
    def process_illustration(self, illustration: Image.Image):
        """
        Process a scientific illustration for conditioning.
        
        Args:
            illustration: Input scientific illustration
            
        Returns:
            Processed conditioning tensor
        """
        # Resize and normalize
        processed = self._preprocess_image(illustration)
        
        # Extract features for conditioning
        features = self._extract_visual_features(processed)
        
        return features
    
    def process_taxonomic_data(self, description: str, 
                             metadata: Optional[Dict[str, Any]] = None):
        """
        Process taxonomic description and metadata for conditioning.
        
        Args:
            description: Taxonomic description text
            metadata: Additional taxonomic metadata
            
        Returns:
            Conditioning vector from text
        """
        # TODO: Implement text encoding
        # This would use a text encoder (CLIP, BERT, etc.)
        
        # For now, return a placeholder
        return torch.randn(1, 768).to(self.device)  # CLIP-like embedding size
    
    def combine_conditionings(self, image_conditioning=None,
                            text_conditioning=None,
                            weight_image: float = 0.5):
        """
        Combine image and text conditionings.
        
        Args:
            image_conditioning: Conditioning from illustration
            text_conditioning: Conditioning from description
            weight_image: Weight for image conditioning (0-1)
            
        Returns:
            Combined conditioning tensor
        """
        if image_conditioning is None and text_conditioning is None:
            raise ValueError("At least one conditioning input required")
        
        if image_conditioning is not None and text_conditioning is not None:
            # Combine with weights
            combined = weight_image * image_conditioning + (1 - weight_image) * text_conditioning
        elif image_conditioning is not None:
            combined = image_conditioning
        else:
            combined = text_conditioning
            
        return combined
    
    def prepare_structure_condition(self, image_path: str):
        """
        Prepare structural conditioning from a scientific illustration.
        
        Steps:
        1. Validate illustration file
        2. Load illustration image
        3. Extract structural edges using Canny (or fallback)
        4. Generate control map
        5. Return tensor for diffusion generator
        
        Args:
            image_path: Path to the scientific illustration
            
        Returns:
            Control map (tensor or array) for conditioning
        """
        # Step 1: Validate illustration file
        is_valid, reason = validate_illustration_file(image_path)
        if not is_valid:
            print(f"⚠️  Illustration validation failed: {reason}")
            raise ValueError(f"Invalid illustration file: {reason}")
        
        # Step 2: Load image
        image = Image.open(image_path)
        
        # Convert to grayscale for edge detection
        if image.mode != 'L':
            image = image.convert('L')
        
        if not HAS_NUMPY:
            # Minimal fallback without numpy
            # Return the PIL image directly - not ideal but allows the pipeline to run
            return image
        
        # Step 2: Extract structural edges
        if HAS_CV2:
            # Use Canny if OpenCV is available
            image_array = np.array(image)
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image_array, (5, 5), 0)
            # Canny edge detection
            edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        else:
            # Fallback: Use PIL edge enhancement and find edges
            # Apply edge enhancement filter
            enhanced = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
            # Find edges using FIND_EDGES filter
            edges_image = enhanced.filter(ImageFilter.FIND_EDGES)
            edges = np.array(edges_image)
        
        # Step 3: Generate control map
        control_map = edges.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Optional: Dilate edges slightly for better conditioning (simple numpy version)
        if not HAS_CV2:
            # Simple dilation using max pooling
            from scipy.ndimage import maximum_filter
            try:
                control_map = maximum_filter(control_map, size=2)
            except ImportError:
                # If scipy not available, skip dilation
                pass
        
        # Step 4: Convert to tensor and resize if needed
        if HAS_TORCH:
            control_tensor = torch.from_numpy(control_map).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # Resize to target size if different
            if control_tensor.shape[-2:] != self.image_size:
                control_tensor = torch.nn.functional.interpolate(
                    control_tensor, 
                    size=self.image_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            
            return control_tensor.to(self.device)
        else:
            # Return numpy array if torch not available
            # Resize using PIL if needed
            if control_map.shape[:2] != self.image_size:
                control_pil = Image.fromarray((control_map * 255).astype(np.uint8))
                control_pil = control_pil.resize(self.image_size, Image.LANCZOS)
                control_map = np.array(control_pil).astype(np.float32) / 255.0
            
            return control_map
    
    def _preprocess_image(self, image: Image.Image):
        """Preprocess image for feature extraction."""
        # Resize
        image = image.resize(self.image_size, Image.LANCZOS)
        
        # Convert to tensor
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Normalize to [-1, 1] or [0, 1] depending on model
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def _extract_visual_features(self, image_tensor):
        """Extract visual features from preprocessed image."""
        # TODO: Implement feature extraction
        # This could use a pre-trained vision model (ResNet, ViT, CLIP vision encoder)
        
        # For now, return the image tensor as features
        return image_tensor.flatten(1)