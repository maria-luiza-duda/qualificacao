Running synthetic augmentation experiments

Use the new wrapper script to run the three conditions with reproducible outputs:

Example:

python3 scripts/run_synthetic_experiments.py \
  --train_jsonl train.jsonl \
  --val_jsonl val.jsonl \
  --manifest manifest.jsonl \
  --synth_jsonl synth.jsonl \
  --outdir runs/multimodal_by_image/ \
  --backbone_name resnet50 \
  --text_model_name bert-base-uncased \
  --model_type multimodal \
  --seed 42 \
  --epochs 10 \
  --batch_size 16 \
  --conditions real_only real_plus_synth synth_only

Outputs:
- A separate folder is created under the provided `--outdir` for each condition and seed, e.g. `runs/.../real_plus_synth_seed42/`.
- `args.json` is saved inside the run folder for reproducibility.
- Train logs and checkpoints go to the run folder.

Notes:
- Default training behavior is unchanged if you continue using `scripts/train.py` directly without `--data_condition` (it defaults to `real_only`).
- Synthetic usage is enabled via `--data_condition` and `--synth_jsonl`. Filtering is supported via `--synth_keep_frac`.
