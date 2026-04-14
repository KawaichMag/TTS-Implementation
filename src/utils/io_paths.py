from pathlib import Path

from models.config import Config


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("No valid path found. Tried: " + ", ".join(str(p) for p in paths))


def resolve_runtime_paths(config: Config) -> Config:
    project_dir = Path(config.project_dir).resolve()
    base_dirs = [project_dir, project_dir / "src", project_dir.parent, project_dir.parent / "src"]

    def candidate(*parts: str) -> list[Path]:
        return [base.joinpath(*parts) for base in base_dirs]

    if config.metadata_path is None:
        config.metadata_path = first_existing(
            candidate("datasets", "metadata.csv")
            + candidate("metadata2.csv")
            + candidate("src", "datasets", "metadata.csv")
        )
    else:
        config.metadata_path = Path(config.metadata_path).resolve()

    if config.wav_dir is None:
        config.wav_dir = first_existing(
            candidate("datasets", "golden_set")
            + candidate("golden_set_2")
            + candidate("src", "datasets", "golden_set")
        )
    else:
        config.wav_dir = Path(config.wav_dir).resolve()

    if config.artifacts_dir is None:
        config.artifacts_dir = project_dir / "src" / "artifacts"
    config.artifacts_dir = Path(config.artifacts_dir).resolve()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if config.aligned_cache_path is None:
        config.aligned_cache_path = config.artifacts_dir / "aligned_items_cache.pt"
    config.aligned_cache_path = Path(config.aligned_cache_path).resolve()

    if config.checkpoint_best_path is None:
        config.checkpoint_best_path = config.artifacts_dir / "compact_speech_best.pt"
    config.checkpoint_best_path = Path(config.checkpoint_best_path).resolve()

    if config.checkpoint_last_path is None:
        config.checkpoint_last_path = config.artifacts_dir / "compact_speech_last.pt"
    config.checkpoint_last_path = Path(config.checkpoint_last_path).resolve()

    if config.warm_start_checkpoint_path is None:
        warm_start_candidates = (
            candidate("compact_speech_pipeline_v2.pt")
            + candidate("compact_speech_pipeline_v2_last.pt")
            + candidate("compact_speech_pipeline.pt")
            + candidate("src", "compact_speech_pipeline.pt")
        )
        for ckpt in warm_start_candidates:
            if ckpt.exists():
                config.warm_start_checkpoint_path = ckpt.resolve()
                break

    return config
