# TTS-Implementation

Compact Grad-TTS-style text-to-mel project with training, inference, analysis plots, and report generation utilities.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

All commands below use `uv run`.

## Dataset Layout

The project expects the LJSpeech-1.1 dataset to be placed into the existing `src/datasets` structure.

Required layout:

```text
src/
  datasets/
    metadata.csv
    golden_set/
      LJ001-0001.wav
      LJ001-0002.wav
      ...
```

Notes:
- `metadata.csv` must keep the standard LJSpeech format with `|` separators.
- Audio files must be placed inside `src/datasets/golden_set`.
- The current code uses this structure by default, so keeping it unchanged is the simplest option.

## Main Entry Point

Training and inference are launched through:

```bash
uv run src/main.py ...
```

Available modes:
- `--mode train`: runs model training.
- `--mode inference`: runs synthesis and saves inference metrics/audio artifacts.

## Training

Basic training:

```bash
uv run src/main.py --mode train
```

Example with custom hyperparameters:

```bash
uv run src/main.py \
  --mode train \
  --epochs 80 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --min-learning-rate 2e-5 \
  --diffusion-steps 80
```

Training outputs are saved into `src/artifacts` by default:
- `training_history.csv`
- `compact_speech_best.pt`
- `compact_speech_last.pt`
- cached aligned data and generated artifacts

## Inference

Run inference from the best available checkpoint:

```bash
uv run src/main.py --mode inference
```

Run inference on a custom text:

```bash
uv run src/main.py \
  --mode inference \
  --custom-text "This is a test sentence." \
  --max-infer-items 1
```

Inference outputs are saved into `src/artifacts` by default:
- `inference_metrics.csv`
- generated `.wav` files
- prior `.wav` files
- reference `.wav` copies when available

## Analysis Plots

Generate plots from saved CSV metrics:

```bash
uv run src/analysis/plot_results.py \
  --training-csv src/artifacts/training_history.csv \
  --inference-csv src/artifacts/inference_metrics.csv \
  --output-dir src/analysis/plots
```

This creates:
- training line plots
- training summary bar chart
- inference histograms
- inference frame scatter plot
- inference top-item bar chart

## CLI Arguments

Path arguments:
- `--project-dir`: base project directory. Default: `.`
- `--metadata-path`: custom path to metadata file.
- `--wav-dir`: custom path to wav directory.
- `--artifacts-dir`: directory for checkpoints, CSV files, and generated outputs.
- `--warm-start-checkpoint`: checkpoint used for initialization before training or fallback inference.
- `--checkpoint-best`: path to best checkpoint.
- `--checkpoint-last`: path to last checkpoint.

Training and inference hyperparameters:
- `--epochs`: number of training epochs.
- `--batch-size`: dataloader batch size.
- `--learning-rate`: initial learning rate.
- `--min-learning-rate`: minimum learning rate for scheduler decay.
- `--diffusion-steps`: number of diffusion reverse steps.
- `--inference-temperature`: sampling temperature during synthesis.
- `--length-scale`: speaking-rate control through predicted durations.
- `--max-infer-items`: number of validation items to synthesize during inference.
- `--custom-text`: custom sentence for inference mode.

## Typical Workflow

1. Put LJSpeech-1.1 `metadata.csv` and `.wav` files into the existing `src/datasets` layout.
2. Install dependencies:

```bash
uv sync
```

3. Train:

```bash
uv run src/main.py --mode train --epochs 80
```

4. Run inference:

```bash
uv run src/main.py --mode inference --max-infer-items 12
```

5. Build plots:

```bash
uv run src/analysis/plot_results.py
```

## Additional Notes

- The project is an acoustic model: it predicts mel-spectrograms and reconstructs waveforms with Griffin-Lim for analysis.
- Training uses external duration supervision prepared from forced alignment.
