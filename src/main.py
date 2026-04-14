import argparse
from pathlib import Path

from inference import run_inference
from models import Config
from train import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Compact Grad-TTS training/inference launcher")
    parser.add_argument("--mode", choices=["train", "inference"], required=True)

    parser.add_argument("--project-dir", type=str, default=".")
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--wav-dir", type=str, default=None)
    parser.add_argument("--artifacts-dir", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--inference-temperature", type=float, default=None)
    parser.add_argument("--length-scale", type=float, default=None)
    parser.add_argument("--max-infer-items", type=int, default=None)

    parser.add_argument("--warm-start-checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-best", type=str, default=None)
    parser.add_argument("--checkpoint-last", type=str, default=None)
    parser.add_argument("--custom-text", type=str, default=None)
    return parser.parse_args()


def build_config_from_args(args) -> Config:
    cfg = Config(project_dir=Path(args.project_dir))
    if args.metadata_path:
        cfg.metadata_path = Path(args.metadata_path)
    if args.wav_dir:
        cfg.wav_dir = Path(args.wav_dir)
    if args.artifacts_dir:
        cfg.artifacts_dir = Path(args.artifacts_dir)
    if args.warm_start_checkpoint:
        cfg.warm_start_checkpoint_path = Path(args.warm_start_checkpoint)
    if args.checkpoint_best:
        cfg.checkpoint_best_path = Path(args.checkpoint_best)
    if args.checkpoint_last:
        cfg.checkpoint_last_path = Path(args.checkpoint_last)

    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "diffusion_steps": args.diffusion_steps,
        "inference_temperature": args.inference_temperature,
        "length_scale": args.length_scale,
        "max_infer_items": args.max_infer_items,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def main():
    args = parse_args()
    config = build_config_from_args(args)
    if args.mode == "train":
        run_training(config)
        return
    run_inference(config, custom_text=args.custom_text)


if __name__ == "__main__":
    main()
