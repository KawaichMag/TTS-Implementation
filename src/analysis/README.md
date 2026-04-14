# Analysis Tools

Generate plots from training and inference metrics CSV files.

## Usage

```bash
python src/analysis/plot_results.py \
  --training-csv src/artifacts/training_history.csv \
  --inference-csv src/artifacts/inference_metrics.csv \
  --output-dir src/analysis/plots
```

## Produced figures

- `training_curves.png` (line plots)
- `training_bars_best_metrics.png` (bar chart)
- `inference_histograms.png` (histograms)
- `inference_frames_scatter.png` (scatter plot)
- `inference_top_items_bar.png` (bar chart)
