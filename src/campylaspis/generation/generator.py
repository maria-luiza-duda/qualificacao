"""
Scientific Image Generation Module

This module provides functionality for generating realistic 2D organism images
from scientific illustrations and taxonomic descriptions.
"""

from typing import Optional, Union, List
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
from PIL import Image
from pathlib import Path
try:
    from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers import DPMSolverMultistepScheduler
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

# Define validation function at module level
def validate_illustration_file(image_path: str):
    """
    Validate that an illustration file can be loaded by PIL.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (is_valid, reason)
    """
    import os
    if not os.path.exists(image_path):
        return False, f"File does not exist: {image_path}"
    if os.path.getsize(image_path) == 0:
        return False, f"File is empty: {image_path}"
    try:
        Image.open(image_path).verify()
        return True, "Valid"
    except Exception as e:
        return False, f"Invalid image: {e}"

# Try to import the better version from conditioning module
try:
    from .conditioning import validate_illustration_file as conditioning_validate
    validate_illustration_file = conditioning_validate
except ImportError:
    try:
        from conditioning import validate_illustration_file as conditioning_validate
        validate_illustration_file = conditioning_validate
    except ImportError:
        # Use the fallback version defined above
        pass


class ScientificImageGenerator:
    """
    Generator for creating realistic 2D organism images.
    
    Supports generation from:
    - Scientific illustrations (image-to-image)
    - Taxonomic descriptions (text-to-image)
    """
    
    def __init__(self, model_path: Optional[str] = None, controlnet_path: Optional[str] = None, device: Optional[str] = None, appearance_prior_dir: Optional[str] = None):
        """
        Initialize the image generator.
        
        Args:
            model_path: Path or HuggingFace ID to pre-trained Stable Diffusion model
            controlnet_path: Path or HuggingFace ID to ControlNet model for structural conditioning
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
            appearance_prior_dir: Optional path to directory containing appearance prior photographs
        """
        self.model_path = model_path or "runwayml/stable-diffusion-v1-5"
        self.controlnet_path = controlnet_path or "lllyasviel/sd-controlnet-canny"
        
        if device:
            self.device = torch.device(device) if HAS_TORCH else None
        elif HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
            
        self.pipe = None
        self.controlnet = None
        self.control_pipe = None
        
        # Initialize appearance conditioning if provided
        self.appearance_prior = None
        self.appearance_conditioner = None
        if appearance_prior_dir is not None:
            try:
                from .appearance_prior import AppearancePrior
                from .appearance_conditioning import AppearanceConditioner
                self.appearance_prior = AppearancePrior(appearance_prior_dir)
                self.appearance_conditioner = AppearanceConditioner(self.appearance_prior)
                print(f"🎨 Initialized appearance conditioning with {len(self.appearance_prior)} reference images")
            except Exception as e:
                print(f"⚠️  Could not initialize appearance conditioning: {e}")
                self.appearance_prior = None
                self.appearance_conditioner = None
        
        if HAS_DIFFUSERS:
            self._load_models()
    
    def _apply_appearance_conditioning(self, base_prompt: str) -> tuple[str, bool]:
        """
        Apply appearance conditioning to a prompt if available.
        
        Args:
            base_prompt: The original prompt
            
        Returns:
            Tuple of (enhanced_prompt, appearance_used)
        """
        if self.appearance_conditioner is not None:
            try:
                appearance_modifier = self.appearance_conditioner.build_appearance_prompt_modifier()
                enhanced_prompt = f"{base_prompt}, {appearance_modifier}"
                return enhanced_prompt, True
            except Exception as e:
                print(f"⚠️  Failed to apply appearance conditioning: {e}")
                return base_prompt, False
        return base_prompt, False
    
    def _load_models(self):
        """Load the diffusion models."""
        try:
            # Load base model
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None,  # Disable safety checker for scientific content
                requires_safety_checker=False
            )
            
            # Use faster scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            if self.device:
                self.pipe = self.pipe.to(self.device)
            
            # Load ControlNet for structural conditioning
            self.controlnet = ControlNetModel.from_pretrained(
                self.controlnet_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            self.control_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_path,
                controlnet=self.controlnet,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.control_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.control_pipe.scheduler.config)
            
            if self.device:
                self.control_pipe = self.control_pipe.to(self.device)
                
            print(f"🔍 Debug: Loaded Stable Diffusion pipeline: {type(self.pipe).__name__}")
            print(f"🔍 Debug: Loaded ControlNet pipeline: {type(self.control_pipe).__name__}")
            print(f"🔍 Debug: ControlNet model loaded: {self.controlnet is not None}")
                
        except Exception as e:
            print(f"Warning: Could not load diffusion models: {e}")
            print(f"🔍 Debug: Stable Diffusion pipeline: None")
            print(f"🔍 Debug: ControlNet pipeline: None")
            print(f"🔍 Debug: ControlNet model loaded: False")
            self.pipe = None
            self.control_pipe = None
        
    def generate_from_illustration(self, illustration: Image.Image, 
                                 conditioning_text: Optional[str] = None) -> Image.Image:
        """
        Generate a realistic image from a scientific illustration.
        
        Args:
            illustration: Input scientific illustration
            conditioning_text: Optional taxonomic description for conditioning
            
        Returns:
            Generated realistic image
        """
        # TODO: Implement image-to-image generation
        raise NotImplementedError("Image-to-image generation not implemented yet")
    
    def generate_from_description(self, description: str, 
                                reference_image: Optional[Image.Image] = None) -> Image.Image:
        """
        Generate a realistic image from a taxonomic description.
        
        Args:
            description: Taxonomic description text
            reference_image: Optional reference illustration
            
        Returns:
            Generated realistic image
        """
        # TODO: Implement text-to-image generation
        raise NotImplementedError("Text-to-image generation not implemented yet")
    
    def save_model(self, path: str):
        """Save the trained model to disk."""
        # TODO: Implement model saving
        pass
    
    def load_model(self, path: str):
        """Load a trained model from disk."""
        # TODO: Implement model loading
        pass
    
    def generate_realistic_specimen(self, prompt: str, 
                                  structure_image: Optional[Union[str, Image.Image]] = None,
                                  num_images: int = 4, 
                                  seed: int = 42) -> List[Image.Image]:
        """
        Generate realistic specimen images using Stable Diffusion.
        
        Args:
            prompt: Text prompt describing the specimen
            structure_image: Optional path to or PIL Image for structural conditioning
            num_images: Number of images to generate
            seed: Random seed for reproducible generation
            
        Returns:
            List of generated PIL Images
        """
        if not HAS_DIFFUSERS or not self.pipe:
            raise RuntimeError("Diffusers library not available or models not loaded")
        
        # Set seed for reproducibility
        if HAS_TORCH:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Process structure image if provided
        if structure_image is not None:
            if isinstance(structure_image, str):
                # Validate and load image from path
                is_valid, reason = validate_illustration_file(structure_image)
                if not is_valid:
                    raise ValueError(f"Invalid structure image: {reason}")
                control_image = Image.open(structure_image)
            else:
                control_image = structure_image
            
            # Ensure it's RGB
            if control_image.mode != 'RGB':
                control_image = control_image.convert('RGB')
            
            # Apply appearance conditioning if available
            enhanced_prompt, appearance_used = self._apply_appearance_conditioning(prompt)
            if appearance_used:
                print(f"🎨 Applied appearance conditioning to realistic specimen prompt")
            
            # Use ControlNet pipeline
            images = self.control_pipe(
                prompt=[enhanced_prompt] * num_images,
                image=[control_image] * num_images,
                generator=generator,
                num_inference_steps=20,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0
            ).images
            
        else:
            # Apply appearance conditioning if available
            enhanced_prompt, appearance_used = self._apply_appearance_conditioning(prompt)
            if appearance_used:
                print(f"🎨 Applied appearance conditioning to realistic specimen prompt")
            
            # Use standard pipeline
            images = self.pipe(
                prompt=[enhanced_prompt] * num_images,
                generator=generator,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images
        
        return images
    
    def generate_structure_guided(self, prompt: str,
                                illustration_path: str,
                                structure_strength: float = 0.9,
                                num_images: int = 4,
                                seed: int = 42) -> List[Image.Image]:
        """
        Generate realistic specimen images using structure-guided ControlNet.
        
        Pipeline: illustration → edge detection → ControlNet → Stable Diffusion
        
        Args:
            prompt: Text prompt describing the specimen (secondary conditioning)
            illustration_path: Path to scientific illustration for edge extraction
            structure_strength: ControlNet conditioning strength (0.0-1.0, default 0.9)
            num_images: Number of images to generate
            seed: Random seed for reproducible generation
            
        Returns:
            List of generated PIL Images
        """
        if not HAS_DIFFUSERS:
            raise RuntimeError("Diffusers library not available - required for structure_guided generation")
        
        if not self.control_pipe:
            raise RuntimeError("ControlNet pipeline not loaded - structure_guided generation requires ControlNet. Check that diffusers and controlnet models are properly installed.")
        
        print(f"🔍 Debug: Using ControlNet pipeline: {type(self.control_pipe).__name__}")
        print(f"🔍 Debug: ControlNet conditioning scale: {structure_strength}")
        
        # Validate illustration file
        is_valid, reason = validate_illustration_file(illustration_path)
        if not is_valid:
            raise ValueError(f"Invalid illustration file: {reason}")
        
        print(f"🔍 Debug: Illustration file validated successfully: {illustration_path}")
        
        # Set seed for reproducibility
        if HAS_TORCH:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Step 1: Load illustration
        illustration = Image.open(illustration_path)
        
        # Debug: Save input illustration copy
        try:
            debug_dir = Path("outputs/smoke_test/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            illustration_copy = illustration.copy()
            illustration_copy.save(debug_dir / "input_illustration.png")
            print(f"🔍 Debug: Saved input illustration to {debug_dir / 'input_illustration.png'}")
        except Exception as e:
            print(f"⚠️  Debug: Could not save input illustration: {e}")
        
        # Step 2: Extract edges using conditioning processor
        try:
            from .conditioning import ScientificConditioningProcessor
            conditioning_processor = ScientificConditioningProcessor()
            edge_map = conditioning_processor.prepare_structure_condition(illustration_path)
        except ImportError:
            # Fallback edge detection if conditioning module not available
            try:
                from conditioning import ScientificConditioningProcessor
                conditioning_processor = ScientificConditioningProcessor()
                edge_map = conditioning_processor.prepare_structure_condition(illustration_path)
            except ImportError:
                # Manual edge detection fallback
                if illustration.mode != 'L':
                    gray = illustration.convert('L')
                else:
                    gray = illustration
                
                try:
                    import cv2
                    import numpy as np
                    gray_array = np.array(gray)
                    edges = cv2.Canny(gray_array, 50, 150)
                    edge_map = Image.fromarray(edges, mode='L')
                except ImportError:
                    # Ultimate fallback: use original image
                    edge_map = illustration.convert('L')
        
        # Convert edge map to RGB for ControlNet
        if isinstance(edge_map, Image.Image):
            if edge_map.mode != 'RGB':
                edge_map = edge_map.convert('RGB')
        else:
            # Handle numpy array case
            try:
                import numpy as np
                if hasattr(edge_map, 'shape'):  # numpy array or torch tensor
                    if hasattr(edge_map, 'cpu'):  # torch tensor
                        edge_map = edge_map.cpu().numpy()
                    edge_map = Image.fromarray((edge_map.squeeze() * 255).astype('uint8'), mode='L').convert('RGB')
            except ImportError:
                edge_map = illustration.convert('RGB')  # Fallback
        
        # Validate control image
        if edge_map is None:
            raise RuntimeError("Control image (edge map) is None - cannot proceed with structure-guided generation")
        
        print(f"🔍 Debug: Control image shape: {edge_map.size if hasattr(edge_map, 'size') else 'unknown'}")
        print(f"🔍 Debug: Control image mode: {edge_map.mode if hasattr(edge_map, 'mode') else 'unknown'}")
        
        # Debug: Save edge map
        try:
            debug_dir = Path("outputs/smoke_test/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            edge_map_copy = edge_map.copy()
            edge_map_copy.save(debug_dir / "edge_map.png")
            print(f"🔍 Debug: Saved edge map to {debug_dir / 'edge_map.png'}")
        except Exception as e:
            print(f"⚠️  Debug: Could not save edge map: {e}")
            raise RuntimeError(f"Failed to save debug edge map: {e}")
        
        # Step 3: Generate with ControlNet using edges as conditioning
        print(f"🔍 Debug: Calling ControlNet pipeline with {num_images} images, conditioning_scale={structure_strength}")
        
        # Apply appearance conditioning if available
        enhanced_prompt, appearance_used = self._apply_appearance_conditioning(prompt)
        if appearance_used:
            print(f"🎨 Applied appearance conditioning to prompt")
            print(f"🔍 Debug: Original prompt: {prompt}")
            print(f"🔍 Debug: Enhanced prompt: {enhanced_prompt}")
        
        try:
            images = self.control_pipe(
                prompt=[enhanced_prompt] * num_images,
                image=[edge_map] * num_images,
                generator=generator,
                num_inference_steps=20,
                guidance_scale=7.5,  # Text prompt guidance
                controlnet_conditioning_scale=structure_strength  # High structure conditioning
            ).images
            print(f"✅ ControlNet generation completed successfully")
        except Exception as e:
            print(f"❌ ControlNet generation failed: {e}")
            raise RuntimeError(f"ControlNet generation failed: {e}")
        
        return images
    
    def render_preserving_morphology(self, 
                                   illustration_path: str,
                                   prompt: Optional[str] = None,
                                   morphology_strength: float = 0.95,
                                   denoising_strength: float = 0.1,
                                   num_images: int = 1,
                                   seed: int = 42) -> List[Image.Image]:
        """
        Render scientific illustrations with morphology-preserving naturalistic enhancement.
        
        This mode preserves the exact anatomical structure while adding realistic rendering.
        Uses very low denoising to maintain original morphology.
        
        Pipeline: illustration → strong edge conditioning + low denoising → minimal enhancement
        
        Args:
            illustration_path: Path to scientific illustration
            prompt: Optional text prompt for subtle enhancement (can be None for pure img2img)
            morphology_strength: ControlNet conditioning strength (0.0-1.0, default 0.95 - very strong)
            denoising_strength: Image-to-image denoising strength (0.0-1.0, default 0.1 - minimal change)
            num_images: Number of images to generate
            seed: Random seed for reproducible generation
            
        Returns:
            List of rendered PIL Images preserving original morphology
        """
        if not HAS_DIFFUSERS:
            raise RuntimeError("Diffusers library not available - required for morphology-preserving rendering")
        
        if not self.control_pipe:
            raise RuntimeError("ControlNet pipeline not loaded - morphology-preserving rendering requires ControlNet")
        
        print(f"🔬 Morphology-preserving rendering mode")
        print(f"🔍 Parameters: morphology_strength={morphology_strength}, denoising_strength={denoising_strength}")
        
        # Validate illustration file
        is_valid, reason = validate_illustration_file(illustration_path)
        if not is_valid:
            raise ValueError(f"Invalid illustration file: {reason}")
        
        # Set seed for reproducibility
        if HAS_TORCH:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Step 1: Load and prepare illustration
        illustration = Image.open(illustration_path).convert('RGB')
        
        # Save original for comparison
        try:
            debug_dir = Path("outputs/smoke_test/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            illustration.save(debug_dir / "original_illustration.png")
            print(f"💾 Saved original illustration to {debug_dir / 'original_illustration.png'}")
        except Exception as e:
            print(f"⚠️  Could not save original illustration: {e}")
        
        # Step 2: Create strong edge conditioning for morphology preservation
        try:
            from .conditioning import ScientificConditioningProcessor
            conditioning_processor = ScientificConditioningProcessor()
            edge_map = conditioning_processor.prepare_structure_condition(illustration_path)
        except ImportError:
            try:
                from conditioning import ScientificConditioningProcessor
                conditioning_processor = ScientificConditioningProcessor()
                edge_map = conditioning_processor.prepare_structure_condition(illustration_path)
            except ImportError:
                # Manual edge detection with strong parameters
                if illustration.mode != 'L':
                    gray = illustration.convert('L')
                else:
                    gray = illustration
                
                try:
                    import cv2
                    import numpy as np
                    gray_array = np.array(gray)
                    # Strong edge detection for morphology preservation
                    edges = cv2.Canny(gray_array, 30, 100)  # Lower thresholds for more edges
                    edge_map = Image.fromarray(edges, mode='L')
                except ImportError:
                    # Ultimate fallback: use original image
                    edge_map = illustration.convert('L')
        
        # Convert edge map to RGB for ControlNet
        if isinstance(edge_map, Image.Image):
            if edge_map.mode != 'RGB':
                edge_map = edge_map.convert('RGB')
        else:
            # Handle numpy array case
            try:
                import numpy as np
                if hasattr(edge_map, 'shape'):
                    if hasattr(edge_map, 'cpu'):
                        edge_map = edge_map.cpu().numpy()
                    edge_map = Image.fromarray((edge_map.squeeze() * 255).astype('uint8'), mode='L').convert('RGB')
            except ImportError:
                edge_map = illustration.convert('RGB')
        
        # Save edge map for debugging
        try:
            debug_dir = Path("outputs/smoke_test/debug")
            edge_map.save(debug_dir / "morphology_edges.png")
            print(f"💾 Saved morphology edges to {debug_dir / 'morphology_edges.png'}")
        except Exception as e:
            print(f"⚠️  Could not save morphology edges: {e}")
        
        # Step 3: Prepare prompt for minimal enhancement
        if prompt is None or prompt.strip() == "":
            # Pure image-to-image with minimal text guidance
            final_prompt = "high resolution scientific illustration, detailed morphology, professional quality"
        else:
            # Apply appearance conditioning if available, but keep it subtle
            enhanced_prompt, appearance_used = self._apply_appearance_conditioning(prompt)
            if appearance_used:
                print(f"🎨 Applied subtle appearance conditioning")
            final_prompt = enhanced_prompt
        
        print(f"🔍 Using prompt: {final_prompt}")
        
        # Step 4: Generate with strong morphology preservation
        try:
            images = self.control_pipe(
                prompt=[final_prompt] * num_images,
                image=[illustration] * num_images,  # Use original illustration as base
                control_image=[edge_map] * num_images,  # Strong edge conditioning
                generator=generator,
                num_inference_steps=15,  # Fewer steps for morphology preservation
                guidance_scale=3.0,  # Lower guidance to reduce text influence
                controlnet_conditioning_scale=morphology_strength,  # Very strong structure control
                strength=denoising_strength  # Very low denoising to preserve morphology
            ).images
            
            print(f"✅ Morphology-preserving rendering completed successfully")
            
            # Save comparison outputs
            self._save_comparison_outputs(illustration, images[0], debug_dir)
            
        except Exception as e:
            print(f"❌ Morphology-preserving rendering failed: {e}")
            raise RuntimeError(f"Morphology-preserving rendering failed: {e}")
        
        return images
    
    def render_morphology_guided(self, 
                                illustration_path: str,
                                prompt: Optional[str] = None,
                                controlnet_conditioning_scale: float = 1.0,
                                strength: float = 0.4,
                                num_images: int = 1,
                                seed: int = 42) -> List[Image.Image]:
        """
        Render scientific illustrations with strict morphological fidelity using ControlNet + img2img.
        
        This mode uses StableDiffusionControlNetImg2ImgPipeline to ensure exact anatomical
        structure preservation while adding realistic rendering enhancements.
        
        Args:
            illustration_path: Path to scientific illustration
            prompt: Optional text prompt for enhancement
            controlnet_conditioning_scale: ControlNet conditioning strength (1.0 = strict preservation)
            strength: img2img denoising strength (0.4 = moderate enhancement)
            num_images: Number of images to generate
            seed: Random seed for reproducible generation
            
        Returns:
            List of rendered PIL Images with preserved morphology
        """
        try:
            from .diffusion_renderer import MorphologyGuidedRenderer
        except ImportError:
            try:
                from diffusion_renderer import MorphologyGuidedRenderer
            except ImportError:
                raise RuntimeError("MorphologyGuidedRenderer not available - required for morphology_guided mode")
        
        # Initialize renderer with same parameters as this generator
        renderer = MorphologyGuidedRenderer(
            model_path=self.model_path,
            controlnet_path=self.controlnet_path,
            device=self.device,
            appearance_prior_dir=self.appearance_prior_dir if hasattr(self, 'appearance_prior_dir') else None
        )
        
        # Render with strict morphological control
        images, used_prompt, metadata = renderer.render_morphology_preserved(
            illustration_path=illustration_path,
            prompt=prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,
            num_images=num_images,
            seed=seed
        )
        
        # Save comparison outputs for debugging
        try:
            debug_dir = Path("outputs/smoke_test/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            original = Image.open(illustration_path)
            renderer.save_comparison_outputs(original, images[0], debug_dir, "morphology_guided")
            
        except Exception as e:
            print(f"⚠️  Could not save morphology-guided comparison outputs: {e}")
        
        return images
    
    def _save_comparison_outputs(self, original: Image.Image, rendered: Image.Image, output_dir: Path):
        """
        Save side-by-side comparison of original and rendered images.
        
        Args:
            original: Original scientific illustration
            rendered: Morphology-preserving rendered result
            output_dir: Directory to save comparison outputs
        """
        try:
            # Create side-by-side comparison
            width1, height1 = original.size
            width2, height2 = rendered.size
            
            # Ensure same height for side-by-side
            max_height = max(height1, height2)
            new_width1 = int(width1 * max_height / height1)
            new_width2 = int(width2 * max_height / height2)
            
            # Resize images
            original_resized = original.resize((new_width1, max_height), Image.Resampling.LANCZOS)
            rendered_resized = rendered.resize((new_width2, max_height), Image.Resampling.LANCZOS)
            
            # Create combined image
            combined_width = new_width1 + new_width2
            combined = Image.new('RGB', (combined_width, max_height))
            combined.paste(original_resized, (0, 0))
            combined.paste(rendered_resized, (new_width1, 0))
            
            # Add labels
            try:
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(combined)
                # Try to use a font, fallback to default if not available
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                # Add labels
                draw.text((10, 10), "ORIGINAL", fill="red", font=font)
                draw.text((new_width1 + 10, 10), "RENDERED", fill="green", font=font)
                
            except ImportError:
                print("⚠️  PIL ImageDraw not available for labels")
            
            # Save comparison
            comparison_path = output_dir / "morphology_comparison.png"
            combined.save(comparison_path)
            print(f"💾 Saved morphology comparison to {comparison_path}")
            
            # Save rendered result separately
            rendered_path = output_dir / "rendered_result.png"
            rendered.save(rendered_path)
            print(f"💾 Saved rendered result to {rendered_path}")
            
        except Exception as e:
            print(f"⚠️  Could not create comparison outputs: {e}")
    
    def _apply_appearance_conditioning(self, prompt: str) -> tuple[str, bool]:
        """
        Apply appearance conditioning to prompt if available.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Tuple of (enhanced_prompt, was_enhanced)
        """
        if self.appearance_conditioner is not None:
            try:
                enhanced = self.appearance_conditioner.build_appearance_prompt_modifier(prompt)
                return enhanced, True
            except Exception as e:
                print(f"⚠️  Appearance conditioning failed: {e}")
                return prompt, False
        return prompt, False
    
    def generate_body_part(self, body_part: str,
                          species_name: str = "crustacean",
                          base_description: str = "",
                          illustration_path: Optional[str] = None,
                          num_images: int = 4,
                          seed: int = 42) -> List[Image.Image]:
        """
        Generate realistic body part images using specialized prompts.
        
        Pipeline: body part detection → specialized prompt → Stable Diffusion
        
        Args:
            body_part: Body part identifier (e.g., 'pereopod_1', 'carapace')
            species_name: Species name for context
            base_description: Additional taxonomic description
            illustration_path: Optional illustration for reference (structure_guided fallback)
            num_images: Number of images to generate
            seed: Random seed for reproducible generation
            
        Returns:
            List of generated PIL Images
        """
        if not HAS_DIFFUSERS or not self.pipe:
            raise RuntimeError("Diffusers library not available or models not loaded")
        
        # Set seed for reproducibility
        if HAS_TORCH:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Build specialized body part prompt
        try:
            from .conditioning import build_body_part_prompt
            prompt = build_body_part_prompt(body_part, species_name, base_description)
        except ImportError:
            try:
                from conditioning import build_body_part_prompt
                prompt = build_body_part_prompt(body_part, species_name, base_description)
            except ImportError:
                # Fallback prompt construction
                prompt = f"high-resolution macro photograph of {species_name} {body_part.replace('_', ' ')} anatomy, marine specimen, scientific realism, detailed structure, professional scientific photography"
        
        # If illustration provided, use structure-guided generation as fallback
        if illustration_path is not None:
            try:
                is_valid, reason = validate_illustration_file(illustration_path)
                if is_valid:
                    print(f"📸 Using illustration reference for {body_part} generation")
                    return self.generate_structure_guided(
                        prompt=prompt,
                        illustration_path=illustration_path,
                        structure_strength=0.8,  # Slightly lower for body parts
                        num_images=num_images,
                        seed=seed
                    )
            except Exception as e:
                print(f"⚠️  Illustration reference failed, falling back to text-only: {e}")
        
        # Apply appearance conditioning if available
        enhanced_prompt, appearance_used = self._apply_appearance_conditioning(prompt)
        if appearance_used:
            print(f"🎨 Applied appearance conditioning to body part prompt")
        
        # Generate with text-to-image
        images = self.pipe(
            prompt=[enhanced_prompt] * num_images,
            generator=generator,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images
        
        return images