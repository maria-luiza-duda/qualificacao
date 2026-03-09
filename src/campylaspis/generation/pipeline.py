"""
Scientific Image Generation Pipeline

High-level pipeline for generating datasets from taxonomic descriptions
and scientific illustrations.
"""

import os
import json
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from PIL import Image

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

try:
    from .generator import ScientificImageGenerator
    from .prompts import ScientificPromptProcessor
    from .morphology_parser import MorphologyParser, build_morphology_prompt
    from .conditioning import ScientificConditioningProcessor
except ImportError:
    try:
        # Fallback for direct import
        from generator import ScientificImageGenerator
        from prompts import ScientificPromptProcessor
        from morphology_parser import MorphologyParser, build_morphology_prompt
        from conditioning import ScientificConditioningProcessor
    except ImportError:
        # Last resort - define dummy classes if modules not available
        class ScientificImageGenerator:
            def __init__(self, model_path=None, controlnet_path=None):
                self.model_path = model_path or "runwayml/stable-diffusion-v1-5"
                self.controlnet_path = controlnet_path or "lllyasviel/sd-controlnet-canny"
            def generate_realistic_specimen(self, *args, **kwargs): raise NotImplementedError()
            def generate_structure_guided(self, *args, **kwargs): raise NotImplementedError()
        
        class ScientificPromptProcessor:
            def __init__(self): pass
            def process_taxonomic_description(self, *args, **kwargs): return {"extracted_features": {}}
            def build_taxonomic_prompt(self, *args, **kwargs): return "dummy prompt"
        
        class MorphologyParser:
            def __init__(self): pass
            def parse_taxonomic_description(self, *args, **kwargs): return {}
            def build_morphology_prompt(self, *args, **kwargs): return "dummy morphology prompt"
        
        def build_morphology_prompt(*args, **kwargs): return "dummy morphology prompt"
        
        class ScientificConditioningProcessor:
            def __init__(self): pass
            def prepare_structure_condition(self, *args, **kwargs): return None


def _ensure_image_extension(path: Path, default_ext: str = ".png") -> Path:
    """
    Ensure an image path has a valid extension.
    
    Args:
        path: The image path
        default_ext: Default extension to append if none exists
        
    Returns:
        Path with valid image extension
    """
    if not path.suffix:
        path = path.with_suffix(default_ext)
    return path


def generate_dataset_from_taxonomy(
    illustration_paths: Union[List[str], None],
    description_texts: List[str],
    output_dir: str,
    num_images_per_specimen: int = 4,
    seed: int = 42,
    device: Optional[str] = None,
    generation_mode: str = "text_only",
    structure_strength: float = 0.9,
    appearance_prior_dir: Optional[str] = None,
    annotations_csv: Optional[str] = None,
    species_list: Optional[List[str]] = None,
    morphology_strength: float = 0.95,
    denoising_strength: float = 0.1,
    controlnet_conditioning_scale: float = 1.0,
    img2img_strength: float = 0.4
) -> Dict[str, Any]:
    """
    Generate a dataset of realistic specimen images from taxonomic descriptions and illustrations.
    
    Pipeline steps:
    1. Parse description
    2. Build prompt
    3. Prepare structural conditioning
    4. Run diffusion generation
    5. Save images + metadata JSON
    
    Args:
        illustration_paths: List of paths to scientific illustrations (or None for text-only)
        description_texts: List of taxonomic description texts
        output_dir: Directory to save generated images and metadata
        num_images_per_specimen: Number of images to generate per specimen
        seed: int = 42,
        device: Device to run generation on ('cuda', 'cpu', or None for auto-detection)
        generation_mode: Generation mode - 'text_only', 'structure_guided', 'body_part_guided', 'render_preserving_morphology', 'morphology_guided'
        structure_strength: ControlNet conditioning strength for structure_guided mode (0.0-1.0)
        appearance_prior_dir: Optional path to directory containing appearance prior photographs
        annotations_csv: Path to CSV file with body part annotations (required for body_part_guided mode)
        species_list: List of species names (required for body_part_guided mode)
        morphology_strength: ControlNet conditioning strength for render_preserving_morphology (0.0-1.0, default 0.95)
        denoising_strength: Image-to-image denoising strength for render_preserving_morphology (0.0-1.0, default 0.1)
        controlnet_conditioning_scale: ControlNet conditioning strength for morphology_guided mode (0.0-1.0, default 1.0)
        img2img_strength: img2img denoising strength for morphology_guided mode (0.0-1.0, default 0.4)
        
    Returns:
        Dictionary with generation statistics and metadata
    """
    
    # Initialize components
    generator = ScientificImageGenerator(device=device, appearance_prior_dir=appearance_prior_dir)
    prompt_processor = ScientificPromptProcessor()
    conditioning_processor = ScientificConditioningProcessor()
    
    # Initialize body-part aware components if needed
    part_annotations = None
    description_parser = None
    if generation_mode == "body_part_guided":
        if annotations_csv is None or species_list is None:
            raise ValueError("body_part_guided mode requires annotations_csv and species_list parameters")
        
        try:
            from ..part_annotations import load_part_annotations
            from ..description_parser import DescriptionParser
            part_annotations = load_part_annotations(annotations_csv)
            description_parser = DescriptionParser()
            print(f"🔍 Loaded part annotations for {len(part_annotations)} species")
        except ImportError as e:
            raise ImportError(f"Required modules for body_part_guided mode not available: {e}")
    
    # Initialize appearance conditioning if provided
    appearance_prior = None
    appearance_conditioner = None
    if appearance_prior_dir is not None:
        try:
            from .appearance_prior import AppearancePrior
            from .appearance_conditioning import AppearanceConditioner
            appearance_prior = AppearancePrior(appearance_prior_dir)
            appearance_conditioner = AppearanceConditioner(appearance_prior)
            print(f"🎨 Initialized appearance conditioning in pipeline with {len(appearance_prior)} reference images")
        except Exception as e:
            print(f"⚠️  Could not initialize appearance conditioning in pipeline: {e}")
            appearance_prior = None
            appearance_conditioner = None
    
    # Pre-generate appearance modifier to populate color palette for metadata
    if generator.appearance_conditioner is not None:
        try:
            # Generate a dummy modifier to populate the color palette
            _ = generator.appearance_conditioner.build_appearance_prompt_modifier()
        except Exception as e:
            print(f"⚠️  Could not pre-generate appearance modifier: {e}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Handle illustration_paths being None (text-only generation)
    if illustration_paths is None:
        illustration_paths = [None] * len(description_texts)
    
    # Metadata for the dataset
    dataset_metadata = {
        "dataset_info": {
            "name": "scientific_specimen_generation_dataset",
            "description": "Generated realistic specimen images from taxonomic descriptions",
            "num_specimens": len(description_texts),
            "images_per_specimen": num_images_per_specimen,
            "total_images": len(description_texts) * num_images_per_specimen
        },
        "specimens": [],
        "generation_config": {
            "num_images_per_specimen": num_images_per_specimen,
            "base_seed": seed,
            "model": generator.model_path,
            "controlnet": generator.controlnet_path,
            "device": str(device) if device else "auto",
            "generation_mode": generation_mode,
            "structure_strength": structure_strength,
            "appearance_prior_used": generator.appearance_conditioner is not None,
            "appearance_prior_dataset": appearance_prior_dir,
            "reference_images_sampled": len(appearance_prior) if appearance_prior else 0,
            "color_palette_used": generator.appearance_conditioner.get_last_color_palette() if generator.appearance_conditioner else [],
            "morphology_strength": morphology_strength if generation_mode == "render_preserving_morphology" else None,
            "denoising_strength": denoising_strength if generation_mode == "render_preserving_morphology" else None,
            "controlnet_conditioning_scale": controlnet_conditioning_scale if generation_mode == "morphology_guided" else None,
            "img2img_strength": img2img_strength if generation_mode == "morphology_guided" else None
        }
    }
    
    image_counter = 0
    
    # Handle body_part_guided mode specially - it processes species and body parts differently
    if generation_mode == "body_part_guided":
        if part_annotations is None or description_parser is None or species_list is None:
            raise ValueError("Body-part guided mode requires annotations, description parser, and species list")
        
        print(f"🔍 Body-part guided generation: processing {len(species_list)} species")
        
        # Process each species
        for species_idx, species in enumerate(species_list):
            if species not in part_annotations:
                print(f"⚠️  Species '{species}' not found in annotations, skipping")
                continue
                
            print(f"Processing species {species_idx + 1}/{len(species_list)}: {species}")
            
            # Get structured descriptions for this species
            structured_descriptions = description_parser.parse_species_description(species)
            
            # Get available body parts for this species
            available_parts = list(part_annotations[species].keys())
            
            for part in available_parts:
                print(f"  Generating for body part: {part}")
                
                # Get illustration for this body part
                part_images = part_annotations[species][part]
                if not part_images:
                    print(f"    ⚠️  No illustrations found for {species} {part}, skipping")
                    continue
                
                # Use the first available illustration (could be randomized later)
                illustration_path = part_images[0]
                
                # Get description for this body part
                part_description = structured_descriptions.get(part, "")
                if not part_description:
                    print(f"    ⚠️  No description found for {species} {part}, using fallback")
                    # Try fallback descriptions from parser
                    part_description = description_parser._get_fallback_description(part)
                
                # Build part-specific prompt
                prompt = prompt_processor.build_part_prompt(species, part, part_description)
                
                try:
                    # Generate images using structure-guided mode with the part illustration
                    generated_images = generator.generate_structure_guided(
                        prompt=prompt,
                        illustration_path=illustration_path,
                        structure_strength=structure_strength,
                        num_images=num_images_per_specimen,
                        seed=seed + species_idx * 100 + hash(part) % 100  # Unique seed per part
                    )
                    
                    # Save images
                    specimen_images = []
                    for img_idx, image in enumerate(generated_images):
                        image_filename = f"{species}_{part}_{img_idx:03d}.png"
                        image_path = images_dir / image_filename
                        image_path = _ensure_image_extension(image_path, ".png")
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        image.save(image_path)
                        
                        specimen_images.append({
                            "filename": image_filename,
                            "path": str(image_path.relative_to(output_path)),
                            "index": image_counter
                        })
                        
                        image_counter += 1
                    
                    # Add specimen metadata for this body part
                    specimen_metadata = {
                        "id": image_counter // num_images_per_specimen,  # Approximate ID
                        "species": species,
                        "body_part": part,
                        "illustration_used": illustration_path,
                        "description_used": part_description,
                        "generation_prompt": prompt,
                        "images": specimen_images,
                        "seed_used": seed + species_idx * 100 + hash(part) % 100,
                        "appearance_prior_used": generator.appearance_conditioner is not None,
                        "conditioning_status": {
                            "illustration_provided": True,
                            "structure_conditioning_used": True,
                            "control_image_available": True,
                            "error": None
                        }
                    }
                    
                    dataset_metadata["specimens"].append(specimen_metadata)
                    
                except Exception as e:
                    print(f"    ❌ Error generating for {species} {part}: {e}")
                    # Add error metadata
                    dataset_metadata["specimens"].append({
                        "species": species,
                        "body_part": part,
                        "error": str(e),
                        "illustration_used": illustration_path,
                        "description_used": part_description
                    })
        
    else:
        # Original specimen processing loop for other modes
        # Process each specimen
        for idx, (illustration_path, description_text) in enumerate(zip(illustration_paths, description_texts)):
            
            specimen_name = f"Specimen_{idx:03d}" if illustration_path is None else Path(illustration_path).stem
            print(f"Processing specimen {idx + 1}/{len(description_texts)}: {specimen_name}")
            
            try:
                # Step 1: Parse description using morphology parser
                morphology_parser = MorphologyParser()
                morphological_constraints = morphology_parser.parse_taxonomic_description(description_text)
                
                # Step 2: Build prompt using structured morphological constraints
                species_name = f"Specimen_{idx:03d}"  # Default name if not extractable
                # Try to extract species name from description or filename
                if "campylaspis" in description_text.lower():
                    species_name = "Campylaspis_sp"
                
                # Use morphology parser to build structured prompt
                prompt = build_morphology_prompt(
                    species_name=species_name,
                    constraints=morphological_constraints,
                    illustration_path=illustration_path,
                )
                
                # Store morphological constraints in metadata
                generation_metadata = {
                    'morphological_constraints': morphological_constraints,
                    'species_name': species_name
                }
                
                # Step 3: Prepare structural conditioning
                control_image = None
                conditioning_error = None
                
                if generation_mode == "structure_guided":
                    # For structure_guided mode, validate illustration exists but don't prepare conditioning here
                    # The generator will handle edge extraction
                    if illustration_path is not None:
                        try:
                            is_valid, reason = validate_illustration_file(illustration_path)
                            if not is_valid:
                                conditioning_error = reason
                                print(f"⚠️  Illustration validation failed: {reason}")
                        except Exception as e:
                            conditioning_error = str(e)
                            print(f"⚠️  Illustration validation error: {e}")
                    else:
                        conditioning_error = "No illustration provided for structure_guided mode"
                        print(f"⚠️  {conditioning_error}")
                        
                elif illustration_path is not None:
                    # For other modes, prepare traditional conditioning if illustration available
                    try:
                        control_map = conditioning_processor.prepare_structure_condition(illustration_path)
                        
                        # Convert control map to PIL Image if it's a tensor/array
                        if hasattr(control_map, 'shape'):  # numpy array or torch tensor
                            if hasattr(control_map, 'cpu'):  # torch tensor
                                control_map = control_map.cpu().numpy()
                            # Convert to PIL Image
                            control_image = Image.fromarray((control_map.squeeze() * 255).astype('uint8'), mode='L')
                        else:
                            control_image = control_map
                    except ValueError as e:
                        # Validation error - log and continue without conditioning
                        conditioning_error = str(e)
                        print(f"⚠️  Structure conditioning unavailable: {conditioning_error}")
                        control_image = None
                    except Exception as e:
                        # Other error - log and continue without conditioning
                        conditioning_error = str(e)
                        print(f"⚠️  Structure conditioning failed: {conditioning_error}")
                        control_image = None
                
                # Step 4: Run diffusion generation
                structure_conditioning_used = False
                
                if generation_mode == "structure_guided":
                    # Structure-guided generation: use edges from illustration as primary conditioning
                    print(f"🔍 Structure-guided mode: checking illustration_path={illustration_path}")
                    
                    if illustration_path is not None:
                        # Validate illustration for structure_guided mode
                        try:
                            from .conditioning import validate_illustration_file as conditioning_validate
                            is_valid, reason = conditioning_validate(illustration_path)
                            print(f"🔍 Illustration validation: valid={is_valid}, reason='{reason}'")
                            
                            if is_valid:
                                print(f"✅ Using structure-guided generation with illustration: {illustration_path}")
                                try:
                                    generated_images = generator.generate_structure_guided(
                                        prompt=prompt,
                                        illustration_path=illustration_path,
                                        structure_strength=structure_strength,
                                        num_images=num_images_per_specimen,
                                        seed=seed + idx
                                    )
                                    structure_conditioning_used = True
                                    print(f"✅ Structure-guided generation completed successfully")
                                except Exception as e:
                                    print(f"❌ Structure-guided generation failed: {e}, falling back to text-only")
                                    generated_images = generator.generate_realistic_specimen(
                                        prompt=prompt,
                                        structure_image=None,
                                        num_images=num_images_per_specimen,
                                        seed=seed + idx
                                    )
                            else:
                                print(f"⚠️  Illustration validation failed: {reason}, falling back to text-only")
                                generated_images = generator.generate_realistic_specimen(
                                    prompt=prompt,
                                    structure_image=None,
                                    num_images=num_images_per_specimen,
                                    seed=seed + idx
                                )
                        except Exception as e:
                            print(f"⚠️  Illustration validation error: {e}, falling back to text-only")
                            generated_images = generator.generate_realistic_specimen(
                                prompt=prompt,
                                structure_image=None,
                                num_images=num_images_per_specimen,
                                seed=seed + idx
                            )
                    else:
                        print(f"⚠️  No illustration provided for structure_guided mode, falling back to text-only")
                        generated_images = generator.generate_realistic_specimen(
                            prompt=prompt,
                            structure_image=None,
                            num_images=num_images_per_specimen,
                            seed=seed + idx
                        )
                elif generation_mode == "body_part":
                    # Body part generation: detect body part from filename and generate specialized images
                    try:
                        from .conditioning import detect_body_part_from_filename
                        body_part, body_description = detect_body_part_from_filename(specimen_name)
                        print(f"🔍 Detected body part: {body_part} ({body_description})")
                        
                        generated_images = generator.generate_body_part(
                            body_part=body_part,
                            species_name=species_name,
                            base_description=desc_dict.get(body_part.split('_')[0], ""),  # Use relevant description
                            illustration_path=illustration_path,
                            num_images=num_images_per_specimen,
                            seed=seed + idx
                        )
                    except ImportError:
                        try:
                            from conditioning import detect_body_part_from_filename
                            body_part, body_description = detect_body_part_from_filename(specimen_name)
                            print(f"🔍 Detected body part: {body_part} ({body_description})")
                            
                            generated_images = generator.generate_body_part(
                                body_part=body_part,
                                species_name=species_name,
                                base_description=desc_dict.get(body_part.split('_')[0], ""),
                                illustration_path=illustration_path,
                                num_images=num_images_per_specimen,
                                seed=seed + idx
                            )
                        except ImportError:
                            # Fallback to text-only if body part detection fails
                            print(f"⚠️  Body part detection failed, falling back to text-only")
                            generated_images = generator.generate_realistic_specimen(
                                prompt=prompt,
                                structure_image=None,
                                num_images=num_images_per_specimen,
                                seed=seed + idx
                            )
                else:
                    # Original modes: text_only or combined
                    generated_images = generator.generate_realistic_specimen(
                        prompt=prompt,
                        structure_image=control_image,
                        num_images=num_images_per_specimen,
                        seed=seed + idx
                    )
                
                # Step 5: Save images and metadata
                specimen_images = []
                
                for img_idx, image in enumerate(generated_images):
                    image_filename = f"{idx:03d}_{img_idx:03d}.png"
                    image_path = images_dir / image_filename
                    
                    # Ensure image path has valid extension
                    image_path = _ensure_image_extension(image_path, ".png")
                    
                    # Ensure parent directories exist
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Debug logging
                    print(f"      💾 Saving image to: {image_path}")
                    print(f"      📎 Using suffix: {image_path.suffix}")
                    
                    image.save(image_path)
                    
                    specimen_images.append({
                        "filename": image_filename,
                        "path": str(image_path.relative_to(output_path)),
                        "index": image_counter
                    })
                    
                    image_counter += 1
                
                # Add specimen metadata
                specimen_metadata = {
                    "id": idx,
                    "species_name": species_name,
                    "illustration_source": str(Path(illustration_path).name) if illustration_path else None,
                    "description": description_text,
                    "parsed_description": parsed_description,
                    "generation_prompt": prompt,
                    "morphological_constraints": morphological_constraints,
                    "images": specimen_images,
                    "seed_used": seed + idx,
                    "conditioning_status": {
                        "illustration_provided": illustration_path is not None,
                        "structure_conditioning_used": structure_conditioning_used,
                        "control_image_available": control_image is not None,
                        "error": conditioning_error
                    }
                }
                
                # Add body part information for body_part generation mode
                if generation_mode == "body_part":
                    try:
                        from .conditioning import detect_body_part_from_filename
                        body_part, body_description = detect_body_part_from_filename(specimen_name)
                        specimen_metadata["body_part"] = {
                            "identifier": body_part,
                            "description": body_description,
                            "detected_from": specimen_name
                        }
                    except ImportError:
                        try:
                            from conditioning import detect_body_part_from_filename
                            body_part, body_description = detect_body_part_from_filename(specimen_name)
                            specimen_metadata["body_part"] = {
                                "identifier": body_part,
                                "description": body_description,
                                "detected_from": specimen_name
                            }
                        except ImportError:
                            specimen_metadata["body_part"] = {
                                "identifier": "unknown",
                                "description": "detection failed",
                                "detected_from": specimen_name
                            }
                
                dataset_metadata["specimens"].append(specimen_metadata)
                
            except Exception as e:
                print(f"Error processing specimen {idx}: {e}")
                # Add error metadata
                dataset_metadata["specimens"].append({
                    "id": idx,
                    "error": str(e),
                    "illustration_source": str(Path(illustration_path).name) if illustration_path else None,
                    "description": description_text
                })
        
        specimen_name = f"Specimen_{idx:03d}" if illustration_path is None else Path(illustration_path).stem
        print(f"Processing specimen {idx + 1}/{len(description_texts)}: {specimen_name}")
        
        try:
            # Step 1: Parse description using morphology parser
            morphology_parser = MorphologyParser()
            morphological_constraints = morphology_parser.parse_taxonomic_description(description_text)
            
            # Step 2: Build prompt using structured morphological constraints
            species_name = f"Specimen_{idx:03d}"  # Default name if not extractable
            # Try to extract species name from description or filename
            if "campylaspis" in description_text.lower():
                species_name = "Campylaspis_sp"
            
            # Use morphology parser to build structured prompt
            prompt = build_morphology_prompt(
                species_name=species_name,
                constraints=morphological_constraints,
                illustration_path=illustration_path,
            )
            
            # Store morphological constraints in metadata
            generation_metadata = {
                'morphological_constraints': morphological_constraints,
                'species_name': species_name
            }
            
            # Step 3: Prepare structural conditioning
            control_image = None
            conditioning_error = None
            
            if generation_mode == "structure_guided":
                # For structure_guided mode, validate illustration exists but don't prepare conditioning here
                # The generator will handle edge extraction
                if illustration_path is not None:
                    try:
                        is_valid, reason = validate_illustration_file(illustration_path)
                        if not is_valid:
                            conditioning_error = reason
                            print(f"⚠️  Illustration validation failed: {reason}")
                    except Exception as e:
                        conditioning_error = str(e)
                        print(f"⚠️  Illustration validation error: {e}")
                else:
                    conditioning_error = "No illustration provided for structure_guided mode"
                    print(f"⚠️  {conditioning_error}")
                    
            elif illustration_path is not None:
                # For other modes, prepare traditional conditioning if illustration available
                try:
                    control_map = conditioning_processor.prepare_structure_condition(illustration_path)
                    
                    # Convert control map to PIL Image if it's a tensor/array
                    if hasattr(control_map, 'shape'):  # numpy array or torch tensor
                        if hasattr(control_map, 'cpu'):  # torch tensor
                            control_map = control_map.cpu().numpy()
                        # Convert to PIL Image
                        control_image = Image.fromarray((control_map.squeeze() * 255).astype('uint8'), mode='L')
                    else:
                        control_image = control_map
                except ValueError as e:
                    # Validation error - log and continue without conditioning
                    conditioning_error = str(e)
                    print(f"⚠️  Structure conditioning unavailable: {conditioning_error}")
                    control_image = None
                except Exception as e:
                    # Other error - log and continue without conditioning
                    conditioning_error = str(e)
                    print(f"⚠️  Structure conditioning failed: {conditioning_error}")
                    control_image = None
            
            # Step 4: Run diffusion generation
            structure_conditioning_used = False
            
            if generation_mode == "structure_guided":
                # Structure-guided generation: use edges from illustration as primary conditioning
                print(f"🔍 Structure-guided mode: checking illustration_path={illustration_path}")
                
                if illustration_path is not None:
                    # Validate illustration for structure_guided mode
                    try:
                        from .conditioning import validate_illustration_file as conditioning_validate
                        is_valid, reason = conditioning_validate(illustration_path)
                        print(f"🔍 Illustration validation: valid={is_valid}, reason='{reason}'")
                        
                        if is_valid:
                            print(f"✅ Using structure-guided generation with illustration: {illustration_path}")
                            try:
                                generated_images = generator.generate_structure_guided(
                                    prompt=prompt,
                                    illustration_path=illustration_path,
                                    structure_strength=structure_strength,
                                    num_images=num_images_per_specimen,
                                    seed=seed + idx
                                )
                                structure_conditioning_used = True
                                print(f"✅ Structure-guided generation completed successfully")
                            except Exception as e:
                                print(f"❌ Structure-guided generation failed: {e}, falling back to text-only")
                                generated_images = generator.generate_realistic_specimen(
                                    prompt=prompt,
                                    structure_image=None,
                                    num_images=num_images_per_specimen,
                                    seed=seed + idx
                                )
                        else:
                            print(f"⚠️  Illustration validation failed: {reason}, falling back to text-only")
                            generated_images = generator.generate_realistic_specimen(
                                prompt=prompt,
                                structure_image=None,
                                num_images=num_images_per_specimen,
                                seed=seed + idx
                            )
                    except Exception as e:
                        print(f"⚠️  Illustration validation error: {e}, falling back to text-only")
            elif generation_mode == "render_preserving_morphology":
                # Morphology-preserving rendering: preserve anatomical structure with naturalistic enhancement
                if illustration_path is None:
                    raise ValueError("render_preserving_morphology mode requires an illustration_path")
                
                print(f"🔬 Morphology-preserving rendering mode")
                print(f"🔍 Parameters: morphology_strength={morphology_strength}, denoising_strength={denoising_strength}")
                
                try:
                    generated_images = generator.render_preserving_morphology(
                        illustration_path=illustration_path,
                        prompt=prompt,
                        morphology_strength=morphology_strength,
                        denoising_strength=denoising_strength,
                        num_images=num_images_per_specimen,
                        seed=seed + idx
                    )
                    structure_conditioning_used = True
                    print(f"✅ Morphology-preserving rendering completed successfully")
                except Exception as e:
                    print(f"❌ Morphology-preserving rendering failed: {e}, falling back to text-only")
                    generated_images = generator.generate_realistic_specimen(
                        prompt=prompt,
                        structure_image=None,
                        num_images=num_images_per_specimen,
                        seed=seed + idx
                    )
                    structure_conditioning_used = True
                    print(f"✅ Morphology-preserving rendering completed successfully")
                except Exception as e:
                    print(f"❌ Morphology-preserving rendering failed: {e}, falling back to text-only")
                    generated_images = generator.generate_realistic_specimen(
                        prompt=prompt,
                        structure_image=None,
                        num_images=num_images_per_specimen,
                        seed=seed + idx
                    )
            elif generation_mode == "morphology_guided":
                # Strict morphological fidelity using ControlNet + img2img - NO FALLBACK ALLOWED
                print(f"🔬 MORPHOLOGY_GUIDED MODE: Strict morphological fidelity required")
                print(f"🔍 Selected generation mode: {generation_mode}")

                # VALIDATION PHASE: Check all required components before proceeding
                validation_errors = []

                # 1. Check illustration
                if illustration_path is None:
                    validation_errors.append("morphology_guided mode requires an illustration_path")
                else:
                    is_valid, reason = validate_illustration_file(illustration_path)
                    if not is_valid:
                        validation_errors.append(f"Illustration validation failed: {reason}")
                    else:
                        print(f"✅ Illustration validated: {Path(illustration_path).name}")

                # 2. Check morphology parser
                try:
                    morphology_parser = MorphologyParser()
                    test_constraints = morphology_parser.parse_taxonomic_description(description_text)
                    if not test_constraints:
                        validation_errors.append("Morphology parser failed to extract any constraints from description")
                    else:
                        print(f"✅ Morphology parser working: extracted {len(test_constraints)} body parts")
                except Exception as e:
                    validation_errors.append(f"Morphology parser initialization failed: {e}")

                # 3. Check if MorphologyGuidedRenderer can be imported and initialized
                try:
                    from .diffusion_renderer import MorphologyGuidedRenderer
                    # Try to create a test renderer (this will fail if dependencies are missing)
                    test_renderer = MorphologyGuidedRenderer(device="cpu")
                    if test_renderer.pipe is None:
                        validation_errors.append("MorphologyGuidedRenderer pipeline not available (missing diffusers or CUDA)")
                    else:
                        print("✅ MorphologyGuidedRenderer available and pipeline initialized")
                except Exception as e:
                    validation_errors.append(f"MorphologyGuidedRenderer initialization failed: {e}")

                # FAIL LOUDLY if any validation fails
                if validation_errors:
                    error_msg = f"❌ MORPHOLOGY_GUIDED VALIDATION FAILED:\n" + "\n".join(f"  - {err}" for err in validation_errors)
                    print(error_msg)
                    raise RuntimeError(f"Morphology-guided mode requirements not met: {validation_errors}")

                # GENERATION PHASE: All validations passed
                print(f"🔬 Starting morphology-guided rendering with strict ControlNet + img2img fidelity")
                print(f"🔍 Parameters: controlnet_scale={controlnet_conditioning_scale}, img2img_strength={img2img_strength}")

                try:
                    generated_images = generator.render_morphology_guided(
                        illustration_path=illustration_path,
                        prompt=prompt,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        strength=img2img_strength,
                        num_images=num_images_per_specimen,
                        seed=seed + idx
                    )
                    structure_conditioning_used = True
                    print(f"✅ Morphology-guided rendering completed successfully")
                    print(f"📊 Generated {len(generated_images)} images")

                except Exception as e:
                    error_msg = f"❌ MORPHOLOGY_GUIDED GENERATION FAILED: {e}"
                    print(error_msg)
                    print("💥 NO FALLBACK ALLOWED in morphology_guided mode - failing loudly")
                    raise RuntimeError(f"Morphology-guided generation failed: {e}") from e
                else:
                    print(f"⚠️  No illustration provided for structure_guided mode, falling back to text-only")
                    generated_images = generator.generate_realistic_specimen(
                        prompt=prompt,
                        structure_image=None,
                        num_images=num_images_per_specimen,
                        seed=seed + idx
                    )
            elif generation_mode == "body_part":
                # Body part generation: detect body part from filename and generate specialized images
                try:
                    from .conditioning import detect_body_part_from_filename
                    body_part, body_description = detect_body_part_from_filename(specimen_name)
                    print(f"🔍 Detected body part: {body_part} ({body_description})")
                    
                    generated_images = generator.generate_body_part(
                        body_part=body_part,
                        species_name=species_name,
                        base_description=desc_dict.get(body_part.split('_')[0], ""),  # Use relevant description
                        illustration_path=illustration_path,
                        num_images=num_images_per_specimen,
                        seed=seed + idx
                    )
                except ImportError:
                    try:
                        from conditioning import detect_body_part_from_filename
                        body_part, body_description = detect_body_part_from_filename(specimen_name)
                        print(f"🔍 Detected body part: {body_part} ({body_description})")
                        
                        generated_images = generator.generate_body_part(
                            body_part=body_part,
                            species_name=species_name,
                            base_description=desc_dict.get(body_part.split('_')[0], ""),
                            illustration_path=illustration_path,
                            num_images=num_images_per_specimen,
                            seed=seed + idx
                        )
                    except ImportError:
                        # Fallback to text-only if body part detection fails
                        print(f"⚠️  Body part detection failed, falling back to text-only")
                        generated_images = generator.generate_realistic_specimen(
                            prompt=prompt,
                            structure_image=None,
                            num_images=num_images_per_specimen,
                            seed=seed + idx
                        )
            elif generation_mode == "body_part_guided":
                # Body-part guided generation: generate images for each body part of each species
                if part_annotations is None or description_parser is None:
                    raise ValueError("Body-part guided mode requires part annotations and description parser")
                
                print(f"🔍 Body-part guided mode: processing species from annotations")
                
                # This mode overrides the normal specimen loop - we'll generate for each species/body-part combination
                # For now, fall back to text-only to maintain compatibility
                print(f"⚠️  Body-part guided mode not yet fully implemented, falling back to text-only")
                generated_images = generator.generate_realistic_specimen(
                    prompt=prompt,
                    structure_image=None,
                    num_images=num_images_per_specimen,
                    seed=seed + idx
                )
            else:
                # Original modes: text_only or combined
                generated_images = generator.generate_realistic_specimen(
                    prompt=prompt,
                    structure_image=control_image,
                    num_images=num_images_per_specimen,
                    seed=seed + idx
                )
            
            # Step 5: Save images and metadata
            specimen_images = []
            
            for img_idx, image in enumerate(generated_images):
                image_filename = f"{idx:03d}_{img_idx:03d}.png"
                image_path = images_dir / image_filename
                
                # Ensure image path has valid extension
                image_path = _ensure_image_extension(image_path, ".png")
                
                # Ensure parent directories exist
                image_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Debug logging
                print(f"      💾 Saving image to: {image_path}")
                print(f"      📎 Using suffix: {image_path.suffix}")
                
                image.save(image_path)
                
                specimen_images.append({
                    "filename": image_filename,
                    "path": str(image_path.relative_to(output_path)),
                    "index": image_counter
                })
                
                image_counter += 1
            
            # Add specimen metadata
            specimen_metadata = {
                "id": idx,
                "species_name": species_name,
                "illustration_source": str(Path(illustration_path).name) if illustration_path else None,
                "description": description_text,
                "parsed_description": parsed_description,
                "morphological_constraints": morphological_constraints,
                "generation_prompt": prompt,
                "images": specimen_images,
                "seed_used": seed + idx,
                "conditioning_status": {
                    "illustration_provided": illustration_path is not None,
                    "structure_conditioning_used": structure_conditioning_used,
                    "control_image_available": control_image is not None,
                    "error": conditioning_error
                }
            }
            
            # Add body part information for body_part generation mode
            if generation_mode == "body_part":
                try:
                    from .conditioning import detect_body_part_from_filename
                    body_part, body_description = detect_body_part_from_filename(specimen_name)
                    specimen_metadata["body_part"] = {
                        "identifier": body_part,
                        "description": body_description,
                        "detected_from": specimen_name
                    }
                except ImportError:
                    try:
                        from conditioning import detect_body_part_from_filename
                        body_part, body_description = detect_body_part_from_filename(specimen_name)
                        specimen_metadata["body_part"] = {
                            "identifier": body_part,
                            "description": body_description,
                            "detected_from": specimen_name
                        }
                    except ImportError:
                        specimen_metadata["body_part"] = {
                            "identifier": "unknown",
                            "description": "detection failed",
                            "detected_from": specimen_name
                        }
            
            dataset_metadata["specimens"].append(specimen_metadata)
            
        except Exception as e:
            print(f"Error processing specimen {idx}: {e}")
            # For morphology_guided mode, validation failures should fail the entire pipeline
            if generation_mode == "morphology_guided":
                print("💥 MORPHOLOGY_GUIDED mode requires all components - re-raising error")
                raise RuntimeError(f"Morphology-guided specimen processing failed: {e}") from e
            # Add error metadata
            dataset_metadata["specimens"].append({
                "id": idx,
                "error": str(e),
                "illustration_source": str(Path(illustration_path).name) if illustration_path else None,
                "description": description_text
            })
    
    # Save metadata JSON
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary_path = output_path / "generation_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Scientific Specimen Generation Dataset Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total specimens processed: {len(illustration_paths)}\n")
        f.write(f"Images per specimen: {num_images_per_specimen}\n")
        f.write(f"Total images generated: {image_counter}\n")
        f.write(f"Output directory: {output_path}\n")
        f.write(f"Metadata saved to: {metadata_path}\n")
        
        successful_specimens = len([s for s in dataset_metadata["specimens"] if "error" not in s])
        f.write(f"Successfully processed specimens: {successful_specimens}\n")
        
        if successful_specimens < len(illustration_paths):
            f.write(f"Failed specimens: {len(illustration_paths) - successful_specimens}\n")
    
    print(f"Dataset generation complete! Generated {image_counter} images.")
    print(f"Results saved to: {output_path}")
    
    return dataset_metadata