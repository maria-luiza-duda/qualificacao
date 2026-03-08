"""
Appearance Prior Module for Scientific Image Generation

This module provides functionality to incorporate appearance priors from real photographs
into the synthetic image generation process, ensuring biologically plausible coloration
and texture statistics for generated specimens.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import random
from pathlib import Path

from PIL import Image
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None


class AppearancePrior:
    """
    Appearance prior class for guiding realistic coloration and texture in synthetic generation.
    
    Loads a dataset of real photographs and provides methods to sample reference images
    and compute color statistics for biologically plausible color distribution.
    """
    
    def __init__(self, photos_dir: str):
        """
        Initialize appearance prior with a directory of reference photographs.
        
        Args:
            photos_dir: Path to directory containing reference photographs
        """
        self.photos_dir = Path(photos_dir)
        self.image_paths: List[Path] = []
        self.loaded_images: List[Image.Image] = []
        self.color_histograms: List[np.ndarray] = []
        
        if not self.photos_dir.exists():
            raise ValueError(f"Photos directory does not exist: {photos_dir}")
        
        if not self.photos_dir.is_dir():
            raise ValueError(f"Photos path is not a directory: {photos_dir}")
        
        # Load all image paths
        self._load_image_paths()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in directory: {photos_dir}")
        
        print(f"✅ Loaded {len(self.image_paths)} reference images from {photos_dir}")
        
        # Pre-compute color histograms for efficiency
        self._precompute_histograms()
    
    def _load_image_paths(self):
        """Load all valid image file paths from the directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for file_path in self.photos_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                self.image_paths.append(file_path)
        
        # Sort for consistent ordering
        self.image_paths.sort()
    
    def _precompute_histograms(self):
        """Pre-compute color histograms for all images."""
        print("🔄 Pre-computing color histograms...")
        
        for image_path in self.image_paths:
            try:
                with Image.open(image_path) as img:
                    histogram = self.compute_color_histogram(img)
                    self.color_histograms.append(histogram)
            except Exception as e:
                print(f"⚠️  Failed to process {image_path}: {e}")
                continue
        
        print(f"✅ Computed histograms for {len(self.color_histograms)} images")
    
    def sample_reference_images(self, n: int = 3) -> List[Image.Image]:
        """
        Randomly sample n reference images from the dataset.
        
        Args:
            n: Number of images to sample
            
        Returns:
            List of PIL Images
        """
        if n > len(self.image_paths):
            n = len(self.image_paths)
            print(f"⚠️  Requested {n} images but only {len(self.image_paths)} available")
        
        # Randomly sample image paths
        sampled_paths = random.sample(self.image_paths, n)
        
        images = []
        for path in sampled_paths:
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                print(f"⚠️  Failed to load {path}: {e}")
                continue
        
        return images
    
    def compute_color_histogram(self, image: Image.Image, bins: int = 32) -> np.ndarray:
        """
        Compute normalized RGB color histogram for an image.
        
        Args:
            image: PIL Image to analyze
            bins: Number of bins per color channel (default: 32)
            
        Returns:
            Normalized histogram as numpy array of shape (bins * 3,)
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Compute histograms for each channel
        histograms = []
        for channel in range(3):  # R, G, B
            hist, _ = np.histogram(img_array[:, :, channel], bins=bins, range=(0, 255), density=True)
            histograms.append(hist)
        
        # Concatenate into single array
        combined_hist = np.concatenate(histograms)
        
        return combined_hist
    
    def compute_dataset_color_statistics(self) -> Dict[str, np.ndarray]:
        """
        Compute aggregate color statistics across the entire dataset.
        
        Returns:
            Dictionary with 'mean_histogram' and 'std_histogram'
        """
        if len(self.color_histograms) == 0:
            raise ValueError("No color histograms available")
        
        # Stack all histograms
        hist_array = np.stack(self.color_histograms)
        
        # Compute mean and standard deviation
        mean_histogram = np.mean(hist_array, axis=0)
        std_histogram = np.std(hist_array, axis=0)
        
        return {
            'mean_histogram': mean_histogram,
            'std_histogram': std_histogram,
            'num_images': len(self.color_histograms)
        }
    
    def get_color_distribution_sample(self, n_samples: int = 1000) -> np.ndarray:
        """
        Sample from the color distribution of the dataset.
        
        This can be used to generate color palettes that match the biological
        coloration patterns observed in real specimens.
        
        Args:
            n_samples: Number of color samples to generate
            
        Returns:
            Array of RGB color values sampled from dataset distribution
        """
        if len(self.color_histograms) == 0:
            raise ValueError("No color histograms available")
        
        # For simplicity, sample from the mean histogram
        stats = self.compute_dataset_color_statistics()
        mean_hist = stats['mean_histogram']
        
        # Reconstruct approximate color distribution
        # This is a simplified approach - in practice, you might want more sophisticated sampling
        colors = []
        
        # Sample colors based on histogram probabilities
        for _ in range(n_samples):
            # Randomly select a bin from the histogram
            bin_idx = np.random.choice(len(mean_hist), p=mean_hist / np.sum(mean_hist))
            
            # Convert bin index back to RGB values
            channel = bin_idx // 32  # Which color channel (R=0, G=1, B=2)
            bin_value = (bin_idx % 32) * (256 // 32)  # Approximate color value
            
            if channel == 0:
                color = [bin_value, 128, 128]  # Red channel
            elif channel == 1:
                color = [128, bin_value, 128]  # Green channel
            else:
                color = [128, 128, bin_value]  # Blue channel
            
            colors.append(color)
        
        return np.array(colors)
    
    def get_texture_references(self, n: int = 5) -> List[Image.Image]:
        """
        Get reference images for texture analysis.
        
        These can be used for texture synthesis or as references for
        generating biologically plausible surface textures.
        
        Args:
            n: Number of reference images to return
            
        Returns:
            List of PIL Images suitable for texture analysis
        """
        # Sample images and return them (could be enhanced with texture filtering)
        return self.sample_reference_images(n)
    
    def __len__(self) -> int:
        """Return the number of reference images."""
        return len(self.image_paths)
    
    def __repr__(self) -> str:
        return f"AppearancePrior(photos_dir='{self.photos_dir}', num_images={len(self)})"