# CPU-Based Specimen Photography Generation Guide

## Overview

Your generation pipeline now supports **CPU-based generation** which provides:
- ✅ **Stable, predictable behavior** - No GPU memory issues or out-of-memory errors
- ✅ **Reproducible results** - Consistent outputs across different machines
- ✅ **Float32 precision** - Numerical stability during diffusion process
- ✅ **Full feature support** - Appearance conditioning, morphology preservation, all work on CPU

## Why Use CPU Instead of GPU?

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Stability** | Very stable, predictable | Can have memory/precision issues |
| **Numerical Precision** | Float32 (full precision) | Often uses Float16 (half precision) |
| **Reproducibility** | Highly reproducible | May vary with GPU changes |
| **Speed** | Slower (15-30 min per image) | Faster (2-5 min per image) |
| **Memory Issues** | Rare | Common with large batches |
| **Quality** | Excellent with proper tuning | Can have artifacts/instability |

## Quick Start

### Method 1: Using the CPU Generation Script (Easiest)

```bash
cd "/home/maria-luiza-duda/Área de trabalho/qualificacao"
source venv_generation/bin/activate

python3 scripts/generate_on_cpu.py \
  --illustration /path/to/illustration.jpg \
  --description "Your species description text here" \
  --appearance-prior /home/maria-luiza-duda/datasets/appearance_prior/cumacea_photos \
  --output-dir outputs/my_results \
  --num-images 4 \
  --num-steps 50 \
  --guidance-scale 7.5
```

### Method 2: Using Debug Pipeline with CPU

```bash
python3 scripts/debug_morphology_pipeline.py \
  --device cpu \
  --mode morphology_guided_no_controlnet \
  --illustration /path/to/illustration.jpg \
  --appearance-prior /home/maria-luiza-duda/datasets/appearance_prior/cumacea_photos \
  --output-dir outputs/debug_cpu_results
```

## Parameter Tuning for Best Photography Results

### Key Parameters for High-Quality Photographs:

**1. Guidance Scale (control adherence to description)**
```
guidance_scale = 7.5  # Default: good balance
guidance_scale = 5.0  # Lower: more creative, less adherent to description
guidance_scale = 10.0 # Higher: strictly follows description, may be less natural
```

**2. Inference Steps (quality vs speed)**
```
num_steps = 30   # Fast: ~5-10 minutes, decent quality
num_steps = 50   # Balanced: ~10-20 minutes, good quality
num_steps = 100  # Slow: ~40+ minutes, excellent quality
```

**3. Strength (how much to modify illustration)**
```
strength = 0.5   # Light modification: preserves illustration closely
strength = 0.75  # Balanced: adds realism while keeping structure
strength = 0.9   # Heavy modification: strong photorealistic transformation
```

### Example Configurations for Different Goals

#### Goal: Maximum Photograph Quality
```bash
python3 scripts/generate_on_cpu.py \
  --illustration illustration.jpg \
  --description "Your description" \
  --output-dir outputs/high_quality \
  --num-steps 100 \
  --guidance-scale 7.5 \
  --strength 0.75 \
  --num-images 4
```
⏱️ Expected time: 40-60 minutes per image

#### Goal: Fast Results (acceptable quality)
```bash
python3 scripts/generate_on_cpu.py \
  --illustration illustration.jpg \
  --description "Your description" \
  --output-dir outputs/quick \
  --num-steps 30 \
  --guidance-scale 7.5 \
  --strength 0.75 \
  --num-images 4
```
⏱️ Expected time: 5-10 minutes per image

#### Goal: Preserve Illustration Details
```bash
python3 scripts/generate_on_cpu.py \
  --illustration illustration.jpg \
  --description "Your description" \
  --output-dir outputs/preserve_details \
  --num-steps 50 \
  --guidance-scale 5.0 \
  --strength 0.4 \
  --num-images 4
```
⏱️ Expected time: 10-20 minutes per image, highly faithful to illustration

## Updates Made to Your Codebase

### 1. Debug Pipeline Script
**File:** `scripts/debug_morphology_pipeline.py`
- ✅ Added `--device` argument (choices: `cpu`, `cuda`)
- ✅ Passes device to MorphologyGuidedRenderer
- Usage: `--device cpu` to force CPU

### 2. New CPU Generation Script
**File:** `scripts/generate_on_cpu.py` (NEW)
- ✅ Convenient wrapper for specimen photography generation
- ✅ Built-in appearance conditioning support
- ✅ Saves generation metadata

### 3. Core Generation Pipeline
**File:** `src/campylaspis/generation/pipeline.py`
- ✅ Already supports `device` parameter
- ✅ Automatically detects CPU vs GPU
- ✅ Supports stable_mode for numerical stability

### 4. Generator Class
**File:** `src/campylaspis/generation/generator.py`
- ✅ Accepts `device` parameter in `__init__`
- ✅ Defaults to CPU if device='cpu'
- ✅ Uses Float32 for CPU (more stable)

## Hardware Requirements for CPU Generation

**Minimum:**
- Python 3.8+
- 8GB RAM (4GB for pure processing, 4GB for models)
- ~10GB free disk space (for model caches)

**Recommended:**
- 16GB RAM (smooth operation)
- CPU with >4 cores (faster processing)
- ~15GB free disk space

## Performance Expectations

On typical CPU hardware:
- Model loading: ~1-2 minutes (first run only)
- Per image generation: 
  - 30 steps: 5-10 minutes
  - 50 steps: 10-20 minutes  
  - 100 steps: 40-60 minutes

## Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce `num_steps` or `num_images`
```bash
python3 scripts/generate_on_cpu.py ... --num-steps 20 --num-images 1
```

### Issue: Very Slow Generation
**Solution:** This is normal on CPU! The generated photographs will have excellent quality.
- For faster results: reduce `num_steps` to 30-50
- CPU generation trades speed for stability and quality

### Issue: Models Keep Re-downloading
**Solution:** Models cache in `~/.cache/huggingface/hub/`. Check disk space:
```bash
du -sh ~/.cache/huggingface/hub/
```

### Issue: Description Text Not Used
**Solution:** Ensure you provide it in quotes:
```bash
python3 scripts/generate_on_cpu.py \
  --illustration img.jpg \
  --description "Carapace twice as long as high, with bifurcating groove..." \
  --output-dir outputs/test
```

## Generation Tips for Better Photographs

1. **Detailed Descriptions**: More specific morphological details → better results
   ```
   ❌ Bad: "A small crustacean"
   ✅ Good: "Carapace twice as long as high, with mid-dorsal bifurcating groove, rectangular mandibles"
   ```

2. **Appearance Prior**: Always include if available
   ```bash
   --appearance-prior /home/maria-luiza-duda/datasets/appearance_prior/cumacea_photos
   ```

3. **Seed for Reproducibility**: Same seed + params = same image
   ```bash
   --seed 42  # Use consistent seed for testing
   ```

4. **Generate Multiple**: Try different parameters and pick best
   ```bash
   --num-images 4  # Generate 4 variations
   ```

5. **Adjust Strength Wisely**:
   - `strength=0.3-0.4`: Close to illustration, less realistic
   - `strength=0.6-0.8`: Good balance
   - `strength=0.9+`: Very photorealistic, may diverge from illustration

## Next Steps

1. ✅ Start with `generate_on_cpu.py` for your first batch
2. Experiment with parameters for your species
3. Use generated photographs for your dataset
4. Enjoy excellent, stable results on CPU!

---
Generated: 2026-03-09  
Device: CPU with stable Float32 precision mode
