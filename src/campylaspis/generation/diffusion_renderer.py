"""
Morphology-Guided Diffusion Renderer

This module provides strict morphological fidelity rendering using ControlNet + img2img diffusion.
Ensures that scientific illustrations maintain exact anatomical structure while gaining realistic rendering.
"""

from typing import Optional, List, Dict, Any, Tuple
from contextlib import nullcontext
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from PIL import Image
from pathlib import Path

try:
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
    from diffusers import StableDiffusionImg2ImgPipeline
    from diffusers import DPMSolverMultistepScheduler
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

# Import validation function
try:
    from .generator import validate_illustration_file
except ImportError:
    try:
        from generator import validate_illustration_file
    except ImportError:
        def validate_illustration_file(image_path: str):
            import os
            if not os.path.exists(image_path):
                return False, f"File does not exist: {image_path}"
            return True, "Valid"


class MorphologyGuidedRenderer:
    """
    Renderer that enforces strict morphological fidelity using ControlNet + img2img diffusion.

    This renderer uses the scientific illustration as both:
    - Initial image for img2img diffusion
    - Source for edge extraction (ControlNet conditioning)

    The result preserves exact anatomical structure while adding realistic rendering.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 controlnet_path: Optional[str] = None,
                 device: Optional[str] = None,
                 appearance_prior_dir: Optional[str] = None,
                 disable_safety_checker: bool = False,
                 stable_mode: bool = False):
        """
        Initialize the morphology-guided renderer.

        Args:
            model_path: Path or HuggingFace ID to pre-trained Stable Diffusion model
            controlnet_path: Path or HuggingFace ID to ControlNet model for structural conditioning
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
            appearance_prior_dir: Optional path to directory containing appearance prior photographs
            disable_safety_checker: Whether to disable diffusers safety checker
            stable_mode: Enable numerically stable inference (float32, no autocast, reduced parameters)
        """
        if not HAS_DIFFUSERS:
            raise RuntimeError("Diffusers library not available - required for morphology-guided rendering")

        self.model_path = model_path or "runwayml/stable-diffusion-v1-5"
        self.controlnet_path = controlnet_path or "lllyasviel/sd-controlnet-canny"
        self.disable_safety_checker = disable_safety_checker
        self.stable_mode = stable_mode

        if device:
            self.device = torch.device(device) if HAS_TORCH else None
        elif HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

        # Determine dtype based on stable_mode and device
        # stable_mode=True forces float32 for numerical stability
        # CPU requires float32, GPU can use float16 for efficiency (unless stable_mode=True)
        if self.stable_mode:
            self.dtype = torch.float32
            print(f"🔧 Stable mode enabled: forcing float32 for numerical stability")
        elif HAS_TORCH and self.device and self.device.type == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # Log device and dtype selection
        print(f"🔧 Selected device: {self.device}")
        print(f"🔧 Selected dtype: {self.dtype}")
        print(f"🔧 Stable mode: {self.stable_mode}")

        # Initialize ControlNet model
        print(f"🔧 Loading ControlNet model: {self.controlnet_path}")
        self.controlnet = ControlNetModel.from_pretrained(
            self.controlnet_path,
            torch_dtype=self.dtype
        )

        # Initialize ControlNet img2img pipeline
        print(f"🔧 Loading Stable Diffusion ControlNet img2img pipeline: {self.model_path}")
        pipeline_kwargs = {
            "controlnet": self.controlnet,
            "torch_dtype": self.dtype,
        }

        if self.disable_safety_checker:
            pipeline_kwargs["safety_checker"] = None
            pipeline_kwargs["requires_safety_checker"] = False

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.model_path,
            **pipeline_kwargs,
        )

        self.safety_checker_enabled = getattr(self.pipe, "safety_checker", None) is not None
        self.safety_checker_disabled = not self.safety_checker_enabled
        print(f"🔒 Safety checker enabled: {self.safety_checker_enabled}")
        print(f"🔓 Safety checker disabled: {self.safety_checker_disabled}")

        # Use faster scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        # Move to device
        if self.device:
            self.pipe = self.pipe.to(self.device)
            print(f"📍 Moved pipeline to device: {self.device}")

        # Enable memory optimizations for low-VRAM GPUs (unless stable_mode requires avoiding them)
        print(f"💾 Configuring memory optimizations...")
        
        # In stable_mode, only enable CPU offload if needed; avoid xformers and slicing
        if not self.stable_mode:
            try:
                # Enable xformers memory-efficient attention
                if HAS_TORCH:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print(f"  ✓ Enabled xformers memory-efficient attention")
            except Exception as e:
                print(f"  ⚠️  Could not enable xformers: {e}")

            # Enable attention slicing for very low VRAM
            try:
                self.pipe.enable_attention_slicing()
                print(f"  ✓ Enabled attention slicing")
            except Exception as e:
                print(f"  ⚠️  Could not enable attention slicing: {e}")

            # Enable VAE slicing for very low VRAM
            try:
                self.pipe.enable_vae_slicing()
                print(f"  ✓ Enabled VAE slicing")
            except Exception as e:
                print(f"  ⚠️  Could not enable VAE slicing: {e}")
        else:
            print(f"  ⚠️  Stable mode: disabled xformers, attention slicing, and VAE slicing")
        
        # Enable CPU offload for VRAM reduction only when running on CUDA.
        # On CPU-only runs this can unintentionally trigger CUDA usage via accelerate hooks.
        if self.device is not None and getattr(self.device, "type", None) == "cuda":
            try:
                self.pipe.enable_model_cpu_offload()
                print(f"  ✓ Enabled model CPU offload")
            except Exception as e:
                print(f"  ⚠️  Could not enable CPU offload: {e}")
        else:
            print(f"  ✓ Skipped CPU offload (device={self.device})")

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
            except ImportError as e:
                print(f"⚠️  Could not initialize appearance conditioning: {e}")

        # Initialize a separate img2img pipeline (no ControlNet) for diagnostic comparison
        self.pipe_no_controlnet = None
        try:
            print(f"🔧 Loading Stable Diffusion img2img pipeline (no ControlNet): {self.model_path}")
            pipeline_kwargs = {
                "torch_dtype": self.dtype,
            }

            if self.disable_safety_checker:
                pipeline_kwargs["safety_checker"] = None
                pipeline_kwargs["requires_safety_checker"] = False

            self.pipe_no_controlnet = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_path,
                **pipeline_kwargs,
            )

            # Use faster scheduler
            self.pipe_no_controlnet.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe_no_controlnet.scheduler.config)

            # Move to device
            if self.device:
                self.pipe_no_controlnet = self.pipe_no_controlnet.to(self.device)

            # Apply same memory optimizations as main pipeline
            if not self.stable_mode:
                try:
                    if HAS_TORCH:
                        self.pipe_no_controlnet.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
                try:
                    self.pipe_no_controlnet.enable_attention_slicing()
                except Exception:
                    pass
                try:
                    self.pipe_no_controlnet.enable_vae_slicing()
                except Exception:
                    pass
            
            if self.device is not None and getattr(self.device, "type", None) == "cuda":
                try:
                    self.pipe_no_controlnet.enable_model_cpu_offload()
                except Exception:
                    pass
            
            print("✅ No-ControlNet img2img pipeline initialized successfully")
        except Exception as e:
            print(f"⚠️  Could not initialize no-ControlNet pipeline: {e}")
            self.pipe_no_controlnet = None

        print("✅ Morphology-guided renderer initialized successfully")

    def _ensure_valid_img2img_strength(
        self,
        strength: float,
        num_inference_steps: int,
        mode_name: str,
        min_effective_steps: int = 2,
    ) -> float:
        """
        Ensure img2img schedule has enough effective denoising steps.

        Some diffusers versions can fail when int(num_inference_steps * strength) is 0,
        producing empty timestep tensors and downstream shape errors.
        """
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be > 0, got {num_inference_steps}")

        effective_steps = int(num_inference_steps * strength)
        if effective_steps >= min_effective_steps:
            return strength

        adjusted_strength = min(1.0, (min_effective_steps + 1e-3) / float(num_inference_steps))
        print(
            f"⚠️  Adjusting strength for {mode_name}: "
            f"{strength:.4f} -> {adjusted_strength:.4f} "
            f"(effective steps {effective_steps} -> {int(num_inference_steps * adjusted_strength)})"
        )
        return adjusted_strength

    def render_morphology_preserved(self,
                                  illustration_path: str,
                                  prompt: Optional[str] = None,
                                  controlnet_conditioning_scale: float = 1.0,
                                  strength: float = 0.25,
                                  guidance_scale: float = 4.0,
                                  num_inference_steps: int = 8,
                                  image_size: Optional[Tuple[int, int]] = None,
                                  num_images: int = 1,
                                  seed: int = 42,
                                  debug_capture_pre_post_stats: bool = False,
                                  debug_save_latent_stats_path: Optional[str] = None) -> Tuple[List[Image.Image], str, Dict[str, Any]]:
        """
        Render scientific illustration with strict morphological fidelity.

        Uses ControlNet + img2img to preserve exact anatomical structure while
        adding realistic rendering enhancements.

        Args:
            illustration_path: Path to scientific illustration
            prompt: Optional text prompt for subtle enhancement
            controlnet_conditioning_scale: ControlNet conditioning strength (1.0 = strict morphology preservation)
            strength: img2img denoising strength (0.25 = conservative enhancement, 0.15 in stable_mode)
            guidance_scale: Classifier-free guidance scale (4.0 default, 3.0 in stable_mode)
            num_inference_steps: Number of diffusion steps (8 default, 6 in stable_mode)
            image_size: Target image size (384x384 default, 320x320 in stable_mode)
            num_images: Number of images to generate
            seed: Random seed for reproducible generation
            debug_capture_pre_post_stats: Capture tensor stats before/after safety checker and postprocessing
            debug_save_latent_stats_path: Path to save per-step latent statistics JSON

        Returns:
            Tuple of (generated_images, used_prompt, metadata_dict)
        """
        # In stable_mode, override parameters for numerical stability
        if self.stable_mode:
            guidance_scale = 3.0
            strength = 0.15
            num_inference_steps = 6
            image_size = (320, 320)
            print(f"🔬 Stable mode: overriding parameters for numerical stability")
            print(f"   guidance_scale={guidance_scale}, strength={strength}, num_inference_steps={num_inference_steps}, image_size={image_size}")

        strength = self._ensure_valid_img2img_strength(
            strength=strength,
            num_inference_steps=num_inference_steps,
            mode_name="morphology_guided",
        )
        
        if image_size is None:
            image_size = (384, 384)
        
        print(f"🔬 Morphology-guided rendering with strict fidelity")
        print(f"🔍 Parameters: controlnet_scale={controlnet_conditioning_scale}, strength={strength}, guidance_scale={guidance_scale}, steps={num_inference_steps}")
        print(f"🎯 Selected generation mode: morphology_guided")
        print(f"🔧 Exact pipeline class used: {type(self.pipe).__name__}")

        # Validate illustration file
        is_valid, reason = validate_illustration_file(illustration_path)
        if not is_valid:
            raise ValueError(f"Invalid illustration file: {reason}")
        print(f"✅ Illustration file validated: {Path(illustration_path).name}")

        print(f"🔒 Safety checker enabled: {self.safety_checker_enabled}")
        print(f"🔓 Safety checker disabled: {self.safety_checker_disabled}")

        # Load and prepare illustration
        illustration = Image.open(illustration_path).convert('RGB')
        
        # Resize to target size
        if illustration.size != image_size:
            illustration = illustration.resize(image_size, Image.Resampling.LANCZOS)
            print(f"📖 Loaded and resized illustration: {illustration.size} {illustration.mode}")
        else:
            print(f"📖 Loaded illustration: {illustration.size} {illustration.mode}")
        
        print(f"🖼️  Init image will be passed to img2img: YES (illustration)")

        # Extract edges for ControlNet conditioning
        edge_map = self._extract_morphology_edges(illustration_path)
        
        # Resize edge map to match target size
        if edge_map.size != image_size:
            edge_map = edge_map.resize(image_size, Image.Resampling.LANCZOS)
        
        print(f"🔍 Extracted morphology edges for ControlNet conditioning")
        print(f"🎛️  Control image will be passed to ControlNet: YES (edge_map)")

        # Prepare prompt (allow appearance-prior photoreal conditioning)
        final_prompt, appearance_metadata = self._prepare_enhanced_prompt(
            prompt,
            strict_prompt_preservation=False,
        )
        print(f"📝 Final prompt: '{final_prompt}'")
        
        # Count tokens
        try:
            if self.appearance_conditioner and hasattr(self.appearance_conditioner, '_count_clip_tokens'):
                token_count = self.appearance_conditioner._count_clip_tokens(final_prompt)
                print(f"🔢 Final prompt length in tokens: {token_count}/77")
            else:
                print(f"🔢 Token counting unavailable")
        except Exception as e:
            print(f"🔢 Token counting failed: {e}")

        # Check negative prompt
        negative_prompt = "blurry, low quality, distorted, deformed, altered proportions, wrong segment ratios, modified silhouette, changed appendage lengths, anatomical deformation, watermark, text, lobster, shrimp, crab, prawn, scorpion, spider, insect, cephalopod, octopus tentacles, claws, extra legs, extra appendages, fantasy anatomy, monster, alien, vertebrate eyes, fish fins, illustration, drawing, sketch, pen and ink, painting, artwork, cartoon, not photographic"
        print(f"🚫 Negative prompt will be used: YES ('{negative_prompt}')")

        # Generate with strict morphological control
        print(f"🎨 Generating with ControlNet img2img (morphology preservation mode)...")

        # Log VRAM usage before generation
        if HAS_TORCH and self.device and self.device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                print(f"💾 VRAM before generation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            except Exception as e:
                print(f"💾 Could not log VRAM: {e}")

        output_tensor_stats_before_postprocess: Dict[str, Any] = {}
        output_tensor_stats_after_postprocess: Dict[str, Any] = {}
        latent_stats_per_step: List[Dict[str, Any]] = []
        sanitization_before_postprocess = False
        sanitization_after_postprocess = False
        nsfw_content_detected = None
        nan_detected_early = False
        nan_detected_at_step = None

        def _make_generator() -> Optional["torch.Generator"]:
            if HAS_TORCH:
                return torch.Generator(device=self.device).manual_seed(seed)
            return None

        # Per-step latent statistics callback
        def latent_callback(step: int, timestep: int, latents: "torch.Tensor") -> None:
            import numpy as np
            
            # Check dtype and device
            latent_dtype = str(latents.dtype)
            latent_device = str(latents.device)
            
            # Compute statistics
            latents_cpu = latents.detach().cpu().float().numpy()
            latent_min = float(np.min(latents_cpu))
            latent_max = float(np.max(latents_cpu))
            latent_mean = float(np.mean(latents_cpu))
            latent_std = float(np.std(latents_cpu))
            latent_contains_nan = bool(np.isnan(latents_cpu).any())
            latent_nan_count = int(np.isnan(latents_cpu).sum())
            
            stats = {
                "step": step,
                "timestep": int(timestep),
                "dtype": latent_dtype,
                "device": latent_device,
                "shape": list(latents_cpu.shape),
                "min": latent_min,
                "max": latent_max,
                "mean": latent_mean,
                "std": latent_std,
                "contains_nan": latent_contains_nan,
                "nan_count": latent_nan_count,
            }
            
            latent_stats_per_step.append(stats)
            
            print(
                f"📊 Step {step}/{num_inference_steps} | dtype={latent_dtype} device={latent_device} | "
                f"latent min={latent_min:.4f} max={latent_max:.4f} mean={latent_mean:.4f} std={latent_std:.4f} | "
                f"NaN={'YES' if latent_contains_nan else 'NO'} ({latent_nan_count})"
            )
            
            # Early abort if NaN detected
            nonlocal nan_detected_early, nan_detected_at_step
            if latent_contains_nan:
                nan_detected_early = True
                nan_detected_at_step = step
                print(f"❌ NaN detected at step {step}/{num_inference_steps}! Aborting early.")
                raise ValueError(f"NaN detected in latents at step {step}")

        try:
            generation_context = torch.no_grad() if HAS_TORCH else nullcontext()
            with generation_context:
                if debug_capture_pre_post_stats:
                    print("🔬 Debug mode: collecting tensor stats before and after safety checker/postprocessing")
                    print("🔬 Debug mode: collecting per-step latent statistics")

                    # Pass 1: latent output for pre-safety/pre-postprocess stats with per-step callback
                    try:
                        latent_output = self.pipe(
                            prompt=[final_prompt] * num_images,
                            image=[illustration] * num_images,
                            control_image=[edge_map] * num_images,
                            generator=_make_generator(),
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                            strength=strength,
                            output_type="latent",
                            return_dict=True,
                            callback=latent_callback,
                            callback_steps=1,
                        )
                    except ValueError as e:
                        if "NaN detected" in str(e):
                            print(f"❌ Generation aborted due to NaN in latents at step {nan_detected_at_step}")
                            # Save partial latent stats
                            if debug_save_latent_stats_path:
                                import json
                                with open(debug_save_latent_stats_path, 'w') as f:
                                    json.dump({
                                        "latent_stats_per_step": latent_stats_per_step,
                                        "nan_detected_early": True,
                                        "nan_detected_at_step": nan_detected_at_step,
                                        "generation_aborted": True,
                                    }, f, indent=2)
                                print(f"💾 Saved partial latent stats to: {debug_save_latent_stats_path}")
                            raise RuntimeError(f"Generation aborted: NaN detected in latents at step {nan_detected_at_step}")
                        raise

                    latents = latent_output.images
                    if hasattr(self.pipe, "decode_latents"):
                        decoded_before = self.pipe.decode_latents(latents)
                    else:
                        scaling_factor = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
                        decoded_tensor = self.pipe.vae.decode(latents / scaling_factor, return_dict=False)[0]
                        decoded_before = ((decoded_tensor / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).float().numpy())

                    output_tensor_stats_before_postprocess = self._collect_image_array_stats(
                        decoded_before,
                        stage="before_safety_checker_postprocess",
                    )
                    decoded_before, sanitization_before_postprocess = self._sanitize_image_array(
                        decoded_before,
                        stage="before_safety_checker_postprocess",
                    )

                    # Pass 2: normal output for post-safety/postprocess stats and nsfw flag
                    # Reset latent stats for second pass
                    latent_stats_per_step_pass2: List[Dict[str, Any]] = []
                    
                    def latent_callback_pass2(step: int, timestep: int, latents: "torch.Tensor") -> None:
                        import numpy as np
                        latents_cpu = latents.detach().cpu().float().numpy()
                        latent_stats_per_step_pass2.append({
                            "step": step,
                            "timestep": int(timestep),
                            "min": float(np.min(latents_cpu)),
                            "max": float(np.max(latents_cpu)),
                            "contains_nan": bool(np.isnan(latents_cpu).any()),
                            "nan_count": int(np.isnan(latents_cpu).sum()),
                        })
                    
                    final_output = self.pipe(
                        prompt=[final_prompt] * num_images,
                        image=[illustration] * num_images,
                        control_image=[edge_map] * num_images,
                        generator=_make_generator(),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        strength=strength,
                        output_type="np",
                        return_dict=True,
                        callback=latent_callback_pass2,
                        callback_steps=1,
                    )

                    postprocessed_images = final_output.images
                    nsfw_content_detected = final_output.nsfw_content_detected
                    print(f"🚨 nsfw_content_detected: {nsfw_content_detected}")

                    output_tensor_stats_after_postprocess = self._collect_image_array_stats(
                        postprocessed_images,
                        stage="after_safety_checker_postprocess",
                    )
                    postprocessed_images, sanitization_after_postprocess = self._sanitize_image_array(
                        postprocessed_images,
                        stage="after_safety_checker_postprocess",
                    )
                    generated_images = self.pipe.image_processor.numpy_to_pil(postprocessed_images)
                    
                    # Save latent stats if requested
                    if debug_save_latent_stats_path:
                        import json
                        with open(debug_save_latent_stats_path, 'w') as f:
                            json.dump({
                                "latent_stats_per_step": latent_stats_per_step,
                                "nan_detected_early": nan_detected_early,
                                "nan_detected_at_step": nan_detected_at_step,
                                "generation_aborted": False,
                                "stable_mode": self.stable_mode,
                                "dtype": str(self.dtype),
                                "device": str(self.device),
                                "guidance_scale": guidance_scale,
                                "strength": strength,
                                "num_inference_steps": num_inference_steps,
                            }, f, indent=2)
                        print(f"💾 Saved latent stats to: {debug_save_latent_stats_path}")
                else:
                    final_output = self.pipe(
                        prompt=[final_prompt] * num_images,
                        image=[illustration] * num_images,
                        control_image=[edge_map] * num_images,
                        generator=_make_generator(),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        strength=strength,
                        output_type="np",
                        return_dict=True,
                    )

                    postprocessed_images = final_output.images
                    nsfw_content_detected = final_output.nsfw_content_detected
                    print(f"🚨 nsfw_content_detected: {nsfw_content_detected}")

                    output_tensor_stats_after_postprocess = self._collect_image_array_stats(
                        postprocessed_images,
                        stage="after_safety_checker_postprocess",
                    )
                    postprocessed_images, sanitization_after_postprocess = self._sanitize_image_array(
                        postprocessed_images,
                        stage="after_safety_checker_postprocess",
                    )
                    generated_images = self.pipe.image_processor.numpy_to_pil(postprocessed_images)

            print(f"✅ Morphology-guided rendering completed successfully")

            # Log VRAM usage after generation
            if HAS_TORCH and self.device and self.device.type == "cuda":
                try:
                    allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                    reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                    print(f"💾 VRAM after generation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                except Exception as e:
                    print(f"💾 Could not log VRAM: {e}")

        except Exception as e:
            print(f"❌ Morphology-guided rendering failed: {e}")
            raise RuntimeError(f"Morphology-guided rendering failed: {e}")

        # Prepare metadata
        metadata = {
            "renderer": "MorphologyGuidedRenderer",
            "model": self.model_path,
            "controlnet": self.controlnet_path,
            "illustration_source": str(Path(illustration_path).name),
            "illustration_path": illustration_path,
            "illustration_size": illustration.size,
            "prompt_used": final_prompt,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "img2img_strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed_used": seed,
            "num_images_generated": len(generated_images),
            "appearance_conditioning": appearance_metadata,
            "morphology_preservation_mode": True,
            "pipeline_type": "StableDiffusionControlNetImg2ImgPipeline",
            "memory_optimizations_enabled": not self.stable_mode,
            "vram_optimization_mode": "disabled (stable_mode)" if self.stable_mode else "very-low-VRAM (xformers + CPU offload + attention slicing + VAE slicing)",
            "stable_mode": self.stable_mode,
            "dtype_used": str(self.dtype),
            "device_used": str(self.device),
            "safety_checker_enabled": self.safety_checker_enabled,
            "safety_checker_disabled": self.safety_checker_disabled,
            "nsfw_content_detected": nsfw_content_detected,
            "output_tensor_stats_before_postprocess": output_tensor_stats_before_postprocess,
            "output_tensor_stats_after_postprocess": output_tensor_stats_after_postprocess,
            "sanitization_before_postprocess": sanitization_before_postprocess,
            "sanitization_after_postprocess": sanitization_after_postprocess,
            "latent_stats_collected": len(latent_stats_per_step) > 0,
            "nan_detected_early": nan_detected_early,
            "nan_detected_at_step": nan_detected_at_step,
        }

        return generated_images, final_prompt, metadata

    def render_morphology_no_controlnet(self,
                                        illustration_path: str,
                                        prompt: Optional[str] = None,
                                        strength: float = 0.25,
                                        guidance_scale: float = 4.0,
                                        num_inference_steps: int = 8,
                                        image_size: Optional[Tuple[int, int]] = None,
                                        num_images: int = 1,
                                        seed: int = 42,
                                        debug_capture_pre_post_stats: bool = False,
                                        debug_save_latent_stats_path: Optional[str] = None) -> Tuple[List[Image.Image], str, Dict[str, Any]]:
        """
        Render scientific illustration WITHOUT ControlNet (diagnostic mode).

        Uses only img2img diffusion without structural conditioning.
        For diagnostic comparison with ControlNet-based rendering.

        Args:
            illustration_path: Path to scientific illustration
            prompt: Optional text prompt for enhancement
            strength: img2img denoising strength
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of diffusion steps
            image_size: Target image size
            num_images: Number of images to generate
            seed: Random seed for reproducible generation
            debug_capture_pre_post_stats: Capture tensor stats before/after safety checker and postprocessing
            debug_save_latent_stats_path: Path to save per-step latent statistics JSON

        Returns:
            Tuple of (generated_images, used_prompt, metadata_dict)
        """
        if self.pipe_no_controlnet is None:
            raise RuntimeError("No-ControlNet pipeline not available")
        
        # In stable_mode, override parameters for numerical stability
        if self.stable_mode:
            guidance_scale = 3.0
            strength = 0.15
            num_inference_steps = 6
            image_size = (320, 320)
            print(f"🔬 Stable mode: overriding parameters for numerical stability")
            print(f"   guidance_scale={guidance_scale}, strength={strength}, num_inference_steps={num_inference_steps}, image_size={image_size}")

        strength = self._ensure_valid_img2img_strength(
            strength=strength,
            num_inference_steps=num_inference_steps,
            mode_name="morphology_guided_no_controlnet",
        )
        
        if image_size is None:
            image_size = (384, 384)
        
        print(f"🔬 Morphology-guided rendering WITHOUT ControlNet (diagnostic mode)")
        print(f"🔍 Parameters: strength={strength}, guidance_scale={guidance_scale}, steps={num_inference_steps}")
        print(f"🎯 Selected generation mode: morphology_guided_no_controlnet")
        print(f"🔧 Exact pipeline class used: {type(self.pipe_no_controlnet).__name__}")

        # Validate illustration file
        is_valid, reason = validate_illustration_file(illustration_path)
        if not is_valid:
            raise ValueError(f"Invalid illustration file: {reason}")
        print(f"✅ Illustration file validated: {Path(illustration_path).name}")

        print(f"🔒 Safety checker enabled: {self.safety_checker_enabled}")
        print(f"🔓 Safety checker disabled: {self.safety_checker_disabled}")

        # Load and prepare illustration
        illustration = Image.open(illustration_path).convert('RGB')
        
        # Resize to target size
        if illustration.size != image_size:
            illustration = illustration.resize(image_size, Image.Resampling.LANCZOS)
            print(f"📖 Loaded and resized illustration: {illustration.size} {illustration.mode}")
        else:
            print(f"📖 Loaded illustration: {illustration.size} {illustration.mode}")
        
        print(f"🖼️  Init image will be passed to img2img: YES (illustration)")
        print(f"🎛️  ControlNet: DISABLED (diagnostic mode)")

        # Prepare prompt (allow appearance-prior photoreal conditioning)
        final_prompt, appearance_metadata = self._prepare_enhanced_prompt(
            prompt,
            strict_prompt_preservation=False,
        )
        print(f"📝 Final prompt: '{final_prompt}'")
        
        # Count tokens
        try:
            if self.appearance_conditioner and hasattr(self.appearance_conditioner, '_count_clip_tokens'):
                token_count = self.appearance_conditioner._count_clip_tokens(final_prompt)
                print(f"🔢 Final prompt length in tokens: {token_count}/77")
            else:
                print(f"🔢 Token counting unavailable")
        except Exception as e:
            print(f"🔢 Token counting failed: {e}")

        # Check negative prompt
        negative_prompt = "blurry, low quality, distorted, deformed, altered proportions, wrong segment ratios, modified silhouette, changed appendage lengths, anatomical deformation, watermark, text, lobster, shrimp, crab, prawn, scorpion, spider, insect, cephalopod, octopus tentacles, claws, extra legs, extra appendages, fantasy anatomy, monster, alien, vertebrate eyes, fish fins, illustration, drawing, sketch, pen and ink, painting, artwork, cartoon, not photographic"
        print(f"🚫 Negative prompt will be used: YES ('{negative_prompt}')")

        # Generate without ControlNet
        print(f"🎨 Generating with img2img only (NO ControlNet conditioning)...")

        # Log VRAM usage before generation
        if HAS_TORCH and self.device and self.device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                print(f"💾 VRAM before generation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            except Exception as e:
                print(f"💾 Could not log VRAM: {e}")

        output_tensor_stats_before_postprocess: Dict[str, Any] = {}
        output_tensor_stats_after_postprocess: Dict[str, Any] = {}
        latent_stats_per_step: List[Dict[str, Any]] = []
        sanitization_before_postprocess = False
        sanitization_after_postprocess = False
        nsfw_content_detected = None
        nan_detected_early = False
        nan_detected_at_step = None

        def _make_generator() -> Optional["torch.Generator"]:
            if HAS_TORCH:
                return torch.Generator(device=self.device).manual_seed(seed)
            return None

        # Per-step latent statistics callback
        def latent_callback(step: int, timestep: int, latents: "torch.Tensor") -> None:
            import numpy as np
            
            latent_dtype = str(latents.dtype)
            latent_device = str(latents.device)
            latents_cpu = latents.detach().cpu().float().numpy()
            latent_min = float(np.min(latents_cpu))
            latent_max = float(np.max(latents_cpu))
            latent_mean = float(np.mean(latents_cpu))
            latent_std = float(np.std(latents_cpu))
            latent_contains_nan = bool(np.isnan(latents_cpu).any())
            latent_nan_count = int(np.isnan(latents_cpu).sum())
            
            stats = {
                "step": step,
                "timestep": int(timestep),
                "dtype": latent_dtype,
                "device": latent_device,
                "shape": list(latents_cpu.shape),
                "min": latent_min,
                "max": latent_max,
                "mean": latent_mean,
                "std": latent_std,
                "contains_nan": latent_contains_nan,
                "nan_count": latent_nan_count,
            }
            
            latent_stats_per_step.append(stats)
            
            print(
                f"📊 Step {step}/{num_inference_steps} | dtype={latent_dtype} device={latent_device} | "
                f"latent min={latent_min:.4f} max={latent_max:.4f} mean={latent_mean:.4f} std={latent_std:.4f} | "
                f"NaN={'YES' if latent_contains_nan else 'NO'} ({latent_nan_count})"
            )
            
            # Early abort if NaN detected
            nonlocal nan_detected_early, nan_detected_at_step
            if latent_contains_nan:
                nan_detected_early = True
                nan_detected_at_step = step
                print(f"❌ NaN detected at step {step}/{num_inference_steps}! Aborting early.")
                raise ValueError(f"NaN detected in latents at step {step}")

        try:
            generation_context = torch.no_grad() if HAS_TORCH else nullcontext()
            with generation_context:
                if debug_capture_pre_post_stats:
                    print("🔬 Debug mode: collecting tensor stats before and after safety checker/postprocessing")
                    print("🔬 Debug mode: collecting per-step latent statistics")

                    # Pass 1: latent output with per-step callback
                    try:
                        latent_output = self.pipe_no_controlnet(
                            prompt=[final_prompt] * num_images,
                            image=[illustration] * num_images,
                            generator=_make_generator(),
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            strength=strength,
                            output_type="latent",
                            return_dict=True,
                            callback=latent_callback,
                            callback_steps=1,
                        )
                    except ValueError as e:
                        if "NaN detected" in str(e):
                            print(f"❌ Generation aborted due to NaN in latents at step {nan_detected_at_step}")
                            if debug_save_latent_stats_path:
                                import json
                                with open(debug_save_latent_stats_path, 'w') as f:
                                    json.dump({
                                        "latent_stats_per_step": latent_stats_per_step,
                                        "nan_detected_early": True,
                                        "nan_detected_at_step": nan_detected_at_step,
                                        "generation_aborted": True,
                                    }, f, indent=2)
                                print(f"💾 Saved partial latent stats to: {debug_save_latent_stats_path}")
                            raise RuntimeError(f"Generation aborted: NaN detected in latents at step {nan_detected_at_step}")
                        raise

                    latents = latent_output.images
                    if hasattr(self.pipe_no_controlnet, "decode_latents"):
                        decoded_before = self.pipe_no_controlnet.decode_latents(latents)
                    else:
                        scaling_factor = getattr(self.pipe_no_controlnet.vae.config, "scaling_factor", 0.18215)
                        decoded_tensor = self.pipe_no_controlnet.vae.decode(latents / scaling_factor, return_dict=False)[0]
                        decoded_before = ((decoded_tensor / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).float().numpy())

                    output_tensor_stats_before_postprocess = self._collect_image_array_stats(
                        decoded_before,
                        stage="before_safety_checker_postprocess",
                    )
                    decoded_before, sanitization_before_postprocess = self._sanitize_image_array(
                        decoded_before,
                        stage="before_safety_checker_postprocess",
                    )

                    # Pass 2: normal output
                    latent_stats_per_step_pass2: List[Dict[str, Any]] = []
                    
                    def latent_callback_pass2(step: int, timestep: int, latents: "torch.Tensor") -> None:
                        import numpy as np
                        latents_cpu = latents.detach().cpu().float().numpy()
                        latent_stats_per_step_pass2.append({
                            "step": step,
                            "timestep": int(timestep),
                            "min": float(np.min(latents_cpu)),
                            "max": float(np.max(latents_cpu)),
                            "contains_nan": bool(np.isnan(latents_cpu).any()),
                            "nan_count": int(np.isnan(latents_cpu).sum()),
                        })
                    
                    final_output = self.pipe_no_controlnet(
                        prompt=[final_prompt] * num_images,
                        image=[illustration] * num_images,
                        generator=_make_generator(),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        output_type="np",
                        return_dict=True,
                        callback=latent_callback_pass2,
                        callback_steps=1,
                    )

                    postprocessed_images = final_output.images
                    nsfw_content_detected = final_output.nsfw_content_detected
                    print(f"🚨 nsfw_content_detected: {nsfw_content_detected}")

                    output_tensor_stats_after_postprocess = self._collect_image_array_stats(
                        postprocessed_images,
                        stage="after_safety_checker_postprocess",
                    )
                    postprocessed_images, sanitization_after_postprocess = self._sanitize_image_array(
                        postprocessed_images,
                        stage="after_safety_checker_postprocess",
                    )
                    generated_images = self.pipe_no_controlnet.image_processor.numpy_to_pil(postprocessed_images)
                    
                    # Save latent stats if requested
                    if debug_save_latent_stats_path:
                        import json
                        with open(debug_save_latent_stats_path, 'w') as f:
                            json.dump({
                                "latent_stats_per_step": latent_stats_per_step,
                                "nan_detected_early": nan_detected_early,
                                "nan_detected_at_step": nan_detected_at_step,
                                "generation_aborted": False,
                                "stable_mode": self.stable_mode,
                                "dtype": str(self.dtype),
                                "device": str(self.device),
                                "guidance_scale": guidance_scale,
                                "strength": strength,
                                "num_inference_steps": num_inference_steps,
                                "controlnet_used": False,
                            }, f, indent=2)
                        print(f"💾 Saved latent stats to: {debug_save_latent_stats_path}")
                else:
                    final_output = self.pipe_no_controlnet(
                        prompt=[final_prompt] * num_images,
                        image=[illustration] * num_images,
                        generator=_make_generator(),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        output_type="np",
                        return_dict=True,
                    )

                    postprocessed_images = final_output.images
                    nsfw_content_detected = final_output.nsfw_content_detected
                    print(f"🚨 nsfw_content_detected: {nsfw_content_detected}")

                    output_tensor_stats_after_postprocess = self._collect_image_array_stats(
                        postprocessed_images,
                        stage="after_safety_checker_postprocess",
                    )
                    postprocessed_images, sanitization_after_postprocess = self._sanitize_image_array(
                        postprocessed_images,
                        stage="after_safety_checker_postprocess",
                    )
                    generated_images = self.pipe_no_controlnet.image_processor.numpy_to_pil(postprocessed_images)

            print(f"✅ No-ControlNet rendering completed successfully")

            # Log VRAM usage after generation
            if HAS_TORCH and self.device and self.device.type == "cuda":
                try:
                    allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                    reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                    print(f"💾 VRAM after generation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                except Exception as e:
                    print(f"💾 Could not log VRAM: {e}")

        except Exception as e:
            print(f"❌ No-ControlNet rendering failed: {e}")
            raise RuntimeError(f"No-ControlNet rendering failed: {e}")

        # Prepare metadata
        metadata = {
            "renderer": "MorphologyGuidedRenderer",
            "mode": "no_controlnet_diagnostic",
            "model": self.model_path,
            "controlnet": None,
            "illustration_source": str(Path(illustration_path).name),
            "illustration_path": illustration_path,
            "illustration_size": illustration.size,
            "prompt_used": final_prompt,
            "controlnet_conditioning_scale": None,
            "img2img_strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed_used": seed,
            "num_images_generated": len(generated_images),
            "appearance_conditioning": appearance_metadata,
            "morphology_preservation_mode": False,
            "pipeline_type": "StableDiffusionImg2ImgPipeline",
            "memory_optimizations_enabled": not self.stable_mode,
            "vram_optimization_mode": "disabled (stable_mode)" if self.stable_mode else "very-low-VRAM (xformers + CPU offload + attention slicing + VAE slicing)",
            "stable_mode": self.stable_mode,
            "dtype_used": str(self.dtype),
            "device_used": str(self.device),
            "safety_checker_enabled": self.safety_checker_enabled,
            "safety_checker_disabled": self.safety_checker_disabled,
            "nsfw_content_detected": nsfw_content_detected,
            "output_tensor_stats_before_postprocess": output_tensor_stats_before_postprocess,
            "output_tensor_stats_after_postprocess": output_tensor_stats_after_postprocess,
            "sanitization_before_postprocess": sanitization_before_postprocess,
            "sanitization_after_postprocess": sanitization_after_postprocess,
            "latent_stats_collected": len(latent_stats_per_step) > 0,
            "nan_detected_early": nan_detected_early,
            "nan_detected_at_step": nan_detected_at_step,
        }

        return generated_images, final_prompt, metadata

    def _collect_image_array_stats(self, image_array: Any, stage: str) -> Dict[str, Any]:
        """Collect numerical stats from an image tensor/array and log them."""
        import numpy as np

        arr = np.asarray(image_array, dtype=np.float32)
        contains_nan = bool(np.isnan(arr).any())
        contains_inf = bool(np.isinf(arr).any())
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        finite_mask = np.isfinite(arr)

        if finite_mask.any():
            finite_values = arr[finite_mask]
            min_value = float(finite_values.min())
            max_value = float(finite_values.max())
            mean_value = float(finite_values.mean())
            std_value = float(finite_values.std())
        else:
            min_value = None
            max_value = None
            mean_value = None
            std_value = None

        stats = {
            "stage": stage,
            "shape": list(arr.shape),
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "std": std_value,
            "contains_nan": contains_nan,
            "contains_inf": contains_inf,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }

        print(
            f"📊 Tensor stats [{stage}] min={stats['min']} max={stats['max']} "
            f"mean={stats['mean']} std={stats['std']} nan={stats['nan_count']} inf={stats['inf_count']}"
        )

        return stats

    def _sanitize_image_array(self, image_array: Any, stage: str) -> Tuple[Any, bool]:
        """Sanitize NaN/inf values in image arrays before conversion to output images."""
        import numpy as np

        arr = np.asarray(image_array, dtype=np.float32)
        contains_nan = bool(np.isnan(arr).any())
        contains_inf = bool(np.isinf(arr).any())

        if not contains_nan and not contains_inf:
            return arr, False

        print(f"⚠️  Sanitization triggered at [{stage}]")
        arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        print(f"🧹 Applied sanitization at [{stage}] (NaN->0.5, +inf->1.0, -inf->0.0)")
        return arr, True

    def _extract_morphology_edges(self, illustration_path: str) -> Image.Image:
        """
        Extract edges from illustration for ControlNet conditioning.

        Uses strong edge detection to preserve morphological details.

        Args:
            illustration_path: Path to illustration

        Returns:
            Edge map as PIL Image
        """
        # Load illustration
        illustration = Image.open(illustration_path)

        # Convert to grayscale for edge detection
        if illustration.mode != 'L':
            gray = illustration.convert('L')
        else:
            gray = illustration

        # Extract edges using OpenCV if available, otherwise fallback
        try:
            import cv2
            import numpy as np

            # Convert to numpy array
            gray_array = np.array(gray)

            # Strong edge detection for morphology preservation
            # Lower thresholds to capture more morphological details
            edges = cv2.Canny(gray_array, threshold1=50, threshold2=150)

            # Convert back to PIL Image
            edge_map = Image.fromarray(edges, mode='L')

        except ImportError:
            # Fallback: use simple edge detection or original image
            print("⚠️  OpenCV not available, using fallback edge detection")
            try:
                from PIL import ImageFilter
                # Simple edge enhancement
                edge_map = gray.filter(ImageFilter.FIND_EDGES)
            except:
                # Ultimate fallback
                edge_map = gray

        # Ensure RGB mode for ControlNet
        if edge_map.mode != 'RGB':
            edge_map = edge_map.convert('RGB')

        return edge_map

    def _prepare_enhanced_prompt(
        self,
        base_prompt: Optional[str],
        strict_prompt_preservation: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare enhanced prompt with optional appearance conditioning.

        Args:
            base_prompt: Base prompt text
            strict_prompt_preservation: When True, keep the prompt text unchanged
                and skip appearance prompt modification

        Returns:
            Tuple of (enhanced_prompt, appearance_metadata)
        """
        appearance_metadata = {
            "appearance_prior_used": False,
            "reference_images_sampled": 0,
            "color_palette_used": [],
            "strict_prompt_preservation": strict_prompt_preservation,
        }

        if base_prompt is None or base_prompt.strip() == "":
            # Default prompt for scientific rendering
            final_prompt = "high resolution scientific illustration, detailed morphology, professional quality, realistic biological specimen"
        else:
            final_prompt = base_prompt

        if strict_prompt_preservation:
            appearance_metadata["appearance_prior_skipped_for_prompt_integrity"] = True
            return final_prompt, appearance_metadata

        # Apply appearance conditioning if available
        if self.appearance_conditioner is not None:
            try:
                # Apply appearance conditioning to the base prompt
                enhanced_prompt = self.appearance_conditioner.build_appearance_prompt_modifier(final_prompt)
                
                appearance_metadata.update({
                    "appearance_prior_used": True,
                    "reference_images_sampled": len(self.appearance_prior) if self.appearance_prior else 0,
                    "color_palette_used": self.appearance_conditioner.get_last_color_palette()
                })

                print(f"🎨 Applied appearance conditioning to prompt")
                final_prompt = enhanced_prompt

            except Exception as e:
                print(f"⚠️  Appearance conditioning failed: {e}")

        return final_prompt, appearance_metadata

    def save_comparison_outputs(self,
                               original: Image.Image,
                               rendered: Image.Image,
                               output_dir: Path,
                               prefix: str = "morphology_guided") -> Dict[str, str]:
        """
        Save comparison outputs showing original vs rendered results.

        Args:
            original: Original scientific illustration
            rendered: Morphology-guided rendered result
            output_dir: Directory to save outputs
            prefix: Prefix for output filenames

        Returns:
            Dictionary with paths to saved files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        try:
            # Save original
            original_path = output_dir / f"{prefix}_original.png"
            original.save(original_path)
            saved_files["original"] = str(original_path)

            # Save rendered
            rendered_path = output_dir / f"{prefix}_rendered.png"
            rendered.save(rendered_path)
            saved_files["rendered"] = str(rendered_path)

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
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()

                # Add labels
                draw.text((10, 10), "ORIGINAL", fill="red", font=font)
                draw.text((new_width1 + 10, 10), "MORPHOLOGY-GUIDED", fill="green", font=font)

            except ImportError:
                print("⚠️  PIL ImageDraw not available for labels")

            # Save comparison
            comparison_path = output_dir / f"{prefix}_comparison.png"
            combined.save(comparison_path)
            saved_files["comparison"] = str(comparison_path)

            print(f"💾 Saved morphology-guided comparison outputs:")
            for key, path in saved_files.items():
                print(f"   {key}: {path}")

        except Exception as e:
            print(f"⚠️  Could not create comparison outputs: {e}")

        return saved_files


# Convenience function for easy usage
def render_morphology_guided(illustration_path: str,
                           prompt: Optional[str] = None,
                           model_path: Optional[str] = None,
                           controlnet_path: Optional[str] = None,
                           appearance_prior_dir: Optional[str] = None,
                           controlnet_conditioning_scale: float = 1.0,
                           strength: float = 0.4,
                           device: Optional[str] = None,
                           seed: int = 42) -> Tuple[List[Image.Image], str, Dict[str, Any]]:
    """
    Convenience function for morphology-guided rendering.

    Args:
        illustration_path: Path to scientific illustration
        prompt: Optional text prompt
        model_path: Stable Diffusion model path
        controlnet_path: ControlNet model path
        appearance_prior_dir: Optional appearance prior directory
        controlnet_conditioning_scale: ControlNet conditioning strength
        strength: img2img denoising strength
        device: Device to run on
        seed: Random seed

    Returns:
        Tuple of (generated_images, used_prompt, metadata_dict)
    """
    renderer = MorphologyGuidedRenderer(
        model_path=model_path,
        controlnet_path=controlnet_path,
        device=device,
        appearance_prior_dir=appearance_prior_dir
    )

    return renderer.render_morphology_preserved(
        illustration_path=illustration_path,
        prompt=prompt,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        strength=strength,
        num_images=1,
        seed=seed
    )


if __name__ == "__main__":
    # Example usage
    print("🔬 Morphology-Guided Renderer Example")

    # This would be used in practice:
    # images, prompt, metadata = render_morphology_guided(
    #     illustration_path="path/to/illustration.png",
    #     prompt="marine crustacean specimen",
    #     appearance_prior_dir="datasets/appearance_prior/cumacea_photos"
    # )

    print("✅ Module loaded successfully - use render_morphology_guided() function")