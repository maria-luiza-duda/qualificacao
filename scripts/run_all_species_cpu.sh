#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

source venv_generation/bin/activate

DATASET_ROOT="${DATASET_ROOT:-/home/maria-luiza-duda/datasets/campylaspis}"
APPEARANCE_PRIOR="${APPEARANCE_PRIOR:-/home/maria-luiza-duda/datasets/appearance_prior/cumacea_photos}"
OUT_ROOT="${OUT_ROOT:-outputs/dataset_cpu_$(date +%Y%m%d_%H%M%S)}"
NUM_IMAGES="${NUM_IMAGES:-1}"
NUM_STEPS="${NUM_STEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.5}"
STRENGTH="${STRENGTH:-0.35}"
CONTROLNET_SCALE="${CONTROLNET_SCALE:-1.80}"
MIN_EDGE_IOU="${MIN_EDGE_IOU:-0.20}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"

if [[ ! -d "$DATASET_ROOT/processed" ]]; then
  echo "❌ Dataset root inválido: $DATASET_ROOT/processed não existe"
  exit 1
fi

mkdir -p "$OUT_ROOT"

SUCCESS=0
FAILED=0
SKIPPED=0

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧬 Batch CPU generation por espécie"
echo "📂 Dataset: $DATASET_ROOT/processed"
echo "📁 Saída:   $OUT_ROOT"
echo "⚙️  Params: images=$NUM_IMAGES steps=$NUM_STEPS guidance=$GUIDANCE_SCALE strength=$STRENGTH controlnet=$CONTROLNET_SCALE min_iou=$MIN_EDGE_IOU size=$IMAGE_SIZE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for species_dir in "$DATASET_ROOT"/processed/*; do
  [[ -d "$species_dir" ]] || continue

  species="$(basename "$species_dir")"
  image_file="$(find "$species_dir/images" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) 2>/dev/null | sort | head -n 1 || true)"

  if [[ -z "$image_file" ]]; then
    echo "⏭️  [$species] sem imagem em images/, pulando"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  species_out="$OUT_ROOT/$species"
  mkdir -p "$species_out"

  echo "\n▶️  [$species] gerando..."
  set +e
  python3 scripts/generate_on_cpu.py \
    --illustration "$image_file" \
    --dataset-root "$DATASET_ROOT" \
    --appearance-prior "$APPEARANCE_PRIOR" \
    --output-dir "$species_out" \
    --num-images "$NUM_IMAGES" \
    --num-steps "$NUM_STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --strength "$STRENGTH" \
    --controlnet-scale "$CONTROLNET_SCALE" \
    --min-edge-iou "$MIN_EDGE_IOU" \
    --image-size "$IMAGE_SIZE" \
    --strict-morphology \
    --reject-low-structure \
    --disable-safety-checker
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "✅ [$species] ok"
    SUCCESS=$((SUCCESS + 1))
  else
    echo "❌ [$species] falhou (exit=$rc)"
    FAILED=$((FAILED + 1))
  fi
done

echo "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Resumo batch"
echo "✅ Sucesso: $SUCCESS"
echo "❌ Falhas:  $FAILED"
echo "⏭️  Pulados: $SKIPPED"
echo "📁 Saída:   $OUT_ROOT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
