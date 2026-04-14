from pathlib import Path

import torch

from models.config import Config


def load_checkpoint(path: Path | None):
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return torch.load(path, map_location="cpu", weights_only=False)


def apply_checkpoint_config(config: Config, checkpoint_config: dict | None):
    if not checkpoint_config:
        return config
    for key, value in checkpoint_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def build_checkpoint_payload(
    *,
    config: Config,
    epoch: int,
    best_loss: float,
    history: list[dict],
    model_state: dict,
    tokenizer_symbol_to_id: dict[str, int],
    mel_mean,
    mel_std,
    raw_model_state: dict | None = None,
    ema_model_state: dict | None = None,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    scaler_state: dict | None = None,
):
    payload = {
        "epoch": int(epoch),
        "model_state": model_state,
        "best_loss": float(best_loss),
        "history": history,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(config).items()},
        "mel_mean": mel_mean.cpu().numpy() if hasattr(mel_mean, "cpu") else mel_mean,
        "mel_std": mel_std.cpu().numpy() if hasattr(mel_std, "cpu") else mel_std,
        "tokenizer": tokenizer_symbol_to_id,
    }
    if raw_model_state is not None:
        payload["raw_model_state"] = raw_model_state
    if ema_model_state is not None:
        payload["ema_model_state"] = ema_model_state
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if scheduler_state is not None:
        payload["scheduler_state"] = scheduler_state
    if scaler_state is not None:
        payload["scaler_state"] = scaler_state
    return payload


def save_checkpoint(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
