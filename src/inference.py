import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models import CompactSpeechSynthModel, Config
from utils.audio import denormalize_mel, load_audio_mono, log_mel_to_audio, resample_audio
from utils.checkpointing import load_checkpoint
from utils.data_pipeline import build_loaders, prepare_items
from utils.io_paths import resolve_runtime_paths
from utils.logging_utils import set_seed


def _resize_mel_to_length(mel_t: torch.Tensor, target_length: int):
    if mel_t.shape[0] == target_length:
        return mel_t
    x = mel_t.transpose(0, 1).unsqueeze(0)
    y = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1)


@torch.no_grad()
def run_inference(config: Config, custom_text: str | None = None):
    set_seed(42)
    config = resolve_runtime_paths(config)

    checkpoint_path = config.checkpoint_best_path if Path(config.checkpoint_best_path).exists() else config.warm_start_checkpoint_path
    checkpoint = load_checkpoint(checkpoint_path)
    if not checkpoint:
        raise RuntimeError("No checkpoint found for inference. Train first or provide warm_start_checkpoint_path.")

    tokenizer_state = checkpoint.get("tokenizer")
    prepared = prepare_items(config, preset_symbol_to_id=tokenizer_state)
    _, _, _, val_items = build_loaders(config, prepared.items, prepared.tokenizer.pad_id)

    model = CompactSpeechSynthModel.build(prepared.tokenizer.vocab_size, config, device=config.device)
    loaded, key, error = model.load_compatible_state(checkpoint)
    if not loaded:
        raise RuntimeError(f"Failed to load checkpoint model state: {error}")
    print(f"Loaded model state for inference: {key}")
    model.eval()

    mel_mean = (
        torch.tensor(checkpoint.get("mel_mean"), dtype=torch.float32)
        if checkpoint.get("mel_mean") is not None
        else prepared.mel_mean
    )
    mel_std = (
        torch.tensor(checkpoint.get("mel_std"), dtype=torch.float32)
        if checkpoint.get("mel_std") is not None
        else prepared.mel_std
    )
    noise_levels = model.make_noise_levels(device=config.device)

    eval_items = val_items[: config.max_infer_items] if config.max_infer_items > 0 else val_items
    if custom_text:
        eval_items = [dict(eval_items[0], text=custom_text)]

    rows = []
    for item in tqdm(eval_items, total=len(eval_items), desc="Inference metrics"):
        start = time.perf_counter()
        generated_mel_norm, pred_durations, diag = model.synthesize(
            item["text"],
            prepared.tokenizer,
            noise_levels=noise_levels,
            temperature=config.inference_temperature,
            length_scale=config.length_scale,
        )
        elapsed = time.perf_counter() - start

        gen_denorm = denormalize_mel(generated_mel_norm, mel_mean, mel_std).cpu().float()
        ref_denorm = item["mel_raw"].cpu().float()
        gen_aligned = _resize_mel_to_length(gen_denorm, ref_denorm.shape[0])

        mel_mae = torch.mean(torch.abs(gen_aligned - ref_denorm)).item()
        mel_rmse = torch.sqrt(torch.mean((gen_aligned - ref_denorm) ** 2)).item()

        pred_durations = pred_durations.cpu().long()
        ref_durations = item["durations"].cpu().long()
        dur_len = min(len(pred_durations), len(ref_durations))
        dur_mae = torch.mean(torch.abs(pred_durations[:dur_len].float() - ref_durations[:dur_len].float())).item()

        pred_frames = int(pred_durations.sum().item())
        ref_frames = int(ref_durations.sum().item())
        audio_sec = max(pred_frames * config.hop_length / config.sample_rate, 1e-6)
        rtf = elapsed / audio_sec

        rows.append(
            {
                "item_id": item["item_id"],
                "mel_mae": mel_mae,
                "mel_rmse": mel_rmse,
                "duration_mae": dur_mae,
                "pred_frames": pred_frames,
                "ref_frames": ref_frames,
                "rtf": rtf,
            }
        )

        prior_mel = denormalize_mel(diag["prior_mel_norm"].float(), mel_mean, mel_std).cpu().numpy()
        gen_mel = gen_denorm.cpu().numpy()
        mel_min = prepared.mel_min
        mel_max = prepared.mel_max
        prior_audio = log_mel_to_audio(prior_mel, config, mel_min, mel_max, n_iter=config.griffin_lim_iters)
        gen_audio = log_mel_to_audio(gen_mel, config, mel_min, mel_max, n_iter=config.griffin_lim_iters)
        sf.write(Path(config.artifacts_dir) / f"{item['item_id']}_prior.wav", prior_audio, config.sample_rate)
        sf.write(Path(config.artifacts_dir) / f"{item['item_id']}_generated.wav", gen_audio, config.sample_rate)

        ref_path = Path(config.wav_dir) / f"{item['item_id']}.wav"
        if ref_path.exists():
            ref_audio, ref_sr = load_audio_mono(ref_path)
            ref_audio = resample_audio(ref_audio, ref_sr, config.sample_rate)
            sf.write(Path(config.artifacts_dir) / f"{item['item_id']}_reference.wav", ref_audio, config.sample_rate)

    metrics_df = pd.DataFrame(rows)
    metrics_path = Path(config.artifacts_dir) / "inference_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved inference metrics: {metrics_path}")
    return metrics_df
