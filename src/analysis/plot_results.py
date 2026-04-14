import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    return pd.read_csv(path)


def _save(fig, output_path: Path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_curves(training_df: pd.DataFrame, output_path: Path):
    if training_df.empty:
        raise ValueError("Training dataframe is empty.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epoch_col = "epoch"

    curves = [
        ("train_total", "val_total", "Total Loss"),
        ("train_diffusion", "val_diffusion", "Diffusion Loss"),
        ("train_duration", "val_duration", "Duration Loss"),
    ]
    for ax, (train_col, val_col, title) in zip(axes, curves):
        if train_col in training_df.columns:
            ax.plot(training_df[epoch_col], training_df[train_col], marker="o", label="train")
        if val_col in training_df.columns:
            ax.plot(training_df[epoch_col], training_df[val_col], marker="o", label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    _save(fig, output_path)


def plot_training_bars(training_df: pd.DataFrame, output_path: Path):
    if training_df.empty:
        raise ValueError("Training dataframe is empty.")

    summary_pairs = [
        ("train_total", "Train Total"),
        ("val_total", "Val Total"),
        ("train_diffusion", "Train Diff"),
        ("val_diffusion", "Val Diff"),
        ("train_duration", "Train Dur"),
        ("val_duration", "Val Dur"),
    ]
    labels = []
    values = []
    for col, label in summary_pairs:
        if col in training_df.columns:
            labels.append(label)
            values.append(float(training_df[col].min()))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, values)
    ax.set_title("Best (Min) Training Metrics")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    _save(fig, output_path)


def plot_inference_histograms(infer_df: pd.DataFrame, output_path: Path):
    if infer_df.empty:
        raise ValueError("Inference dataframe is empty.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    cols = [
        ("mel_mae", "Mel MAE"),
        ("duration_mae", "Duration MAE"),
        ("rtf", "RTF"),
    ]
    for ax, (col, title) in zip(axes, cols):
        if col in infer_df.columns:
            ax.hist(infer_df[col].dropna(), bins=min(20, max(6, len(infer_df))), alpha=0.85)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    _save(fig, output_path)


def plot_inference_scatter(infer_df: pd.DataFrame, output_path: Path):
    if infer_df.empty:
        raise ValueError("Inference dataframe is empty.")
    if "ref_frames" not in infer_df.columns or "pred_frames" not in infer_df.columns:
        raise ValueError("Inference CSV must contain ref_frames and pred_frames columns.")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(infer_df["ref_frames"], infer_df["pred_frames"], alpha=0.8)
    lo = min(infer_df["ref_frames"].min(), infer_df["pred_frames"].min())
    hi = max(infer_df["ref_frames"].max(), infer_df["pred_frames"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    ax.set_title("Predicted vs Reference Frames")
    ax.set_xlabel("Reference Frames")
    ax.set_ylabel("Predicted Frames")
    ax.grid(alpha=0.3)
    _save(fig, output_path)


def plot_inference_bars(infer_df: pd.DataFrame, output_path: Path, top_k: int = 10):
    if infer_df.empty:
        raise ValueError("Inference dataframe is empty.")
    if "item_id" not in infer_df.columns or "mel_mae" not in infer_df.columns:
        raise ValueError("Inference CSV must contain item_id and mel_mae columns.")

    top = infer_df.nsmallest(min(top_k, len(infer_df)), "mel_mae")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(top["item_id"], top["mel_mae"])
    ax.set_title(f"Top-{len(top)} Items by Mel MAE (Lower is Better)")
    ax.set_ylabel("Mel MAE")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_path)


def save_all_plots(training_csv: Path, inference_csv: Path, output_dir: Path):
    _ensure_dir(output_dir)
    training_df = _read_csv(training_csv)
    infer_df = _read_csv(inference_csv)

    plot_training_curves(training_df, output_dir / "training_curves.png")
    plot_training_bars(training_df, output_dir / "training_bars_best_metrics.png")
    plot_inference_histograms(infer_df, output_dir / "inference_histograms.png")
    plot_inference_scatter(infer_df, output_dir / "inference_frames_scatter.png")
    plot_inference_bars(infer_df, output_dir / "inference_top_items_bar.png", top_k=10)
    print(f"Saved plots to: {output_dir}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot training and inference metrics.")
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=Path("src/artifacts/training_history.csv"),
        help="Path to training history CSV.",
    )
    parser.add_argument(
        "--inference-csv",
        type=Path,
        default=Path("src/artifacts/inference_metrics.csv"),
        help="Path to inference metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/analysis/plots"),
        help="Directory to save generated figures.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    save_all_plots(args.training_csv, args.inference_csv, args.output_dir)
