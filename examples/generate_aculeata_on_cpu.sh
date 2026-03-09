#!/bin/bash
# Example: Generate high-quality photographs of Campylaspis aculeata using CPU
# 
# This script demonstrates CPU-based generation for the aculeata species
# Expected time: ~15-30 minutes for 4 high-quality images on typical CPU

cd "$(dirname "$0")" || exit 1
source venv_generation/bin/activate

# Species configuration
SPECIES="aculeata"
ILLUSTRATION="/home/maria-luiza-duda/datasets/campylaspis/processed/aculeata/images/aculeata_lateral_body.jpg"
APPEARANCE_PRIOR="/home/maria-luiza-duda/datasets/appearance_prior/cumacea_photos"
OUTPUT_DIR="outputs/aculeata_cpu_results_$(date +%Y%m%d_%H%M%S)"

# Generation parameters (tuned for high-quality photorealistic results)
NUM_IMAGES=4
NUM_STEPS=50          # 50 steps = good quality + reasonable time
GUIDANCE_SCALE=4.5    # Moderate guidance: balance between ControlNet structure lock and realistic appearance
STRENGTH=0.35         # Allow texture transformation while maintaining anatomical proportions via ControlNet
CONTROLNET_SCALE=1.80 # Strong structure lock via edge detection / canny edges
MIN_EDGE_IOU=0.20     # Informational threshold: save all images regardless for visual review
SEED=42               # Fixed seed for reproducibility

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧬 CPU-Based Photograph Generation: Campylaspis aculeata"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  📸 Species:           $SPECIES"
echo "  🎨 Output directory:  $OUTPUT_DIR"
echo "  📊 Images to generate: $NUM_IMAGES"
echo "  ⚙️  Inference steps:    $NUM_STEPS (higher = better quality)"
echo "  🎯 Guidance scale:     $GUIDANCE_SCALE (moderate: balance between structure lock and realism)"
echo "  🔄 Strength:           $STRENGTH (allows texture transformation, structure locked by ControlNet)"
echo "  🧷 ControlNet scale:   $CONTROLNET_SCALE (strong edge-based structure locking)"
echo "  ✅ Min edge IoU:       $MIN_EDGE_IOU (all images saved for visual review)"
echo "  🖥️  Device:            CPU (stable, predictable)"
echo ""
echo "⏱️  Estimated time: 10-30 minutes per image"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run generation
python3 scripts/generate_on_cpu.py \
  --illustration "$ILLUSTRATION" \
  --appearance-prior "$APPEARANCE_PRIOR" \
  --output-dir "$OUTPUT_DIR" \
  --num-images "$NUM_IMAGES" \
  --num-steps "$NUM_STEPS" \
  --guidance-scale "$GUIDANCE_SCALE" \
  --strength "$STRENGTH" \
  --controlnet-scale "$CONTROLNET_SCALE" \
  --min-edge-iou "$MIN_EDGE_IOU" \
  --strict-morphology \
  --reject-low-structure \
  --seed "$SEED"

if [ $? -eq 0 ]; then
  echo ""
  echo "✨ Generation complete! Results saved to: $OUTPUT_DIR"
  echo ""
  echo "📁 Generated files:"
  ls -lh "$OUTPUT_DIR"/specimen_*.png
  echo ""
  echo "💡 Tips for next runs:"
  echo "   • Try different --num-steps values (30 for speed, 100 for quality)"
  echo "   • Adjust --guidance-scale (lower=more creative, higher=more adherent)"
  echo "   • Use --strength to control illustration vs realism balance"
  echo ""
else
  echo ""
  echo "❌ Generation failed. Check error messages above."
  exit 1
fi
