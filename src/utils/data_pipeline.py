from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from models.config import Config
from utils.alignment import align_text_to_mel
from utils.audio import extract_log_mel_from_waveform, load_audio_mono, resample_audio
from utils.tokenizer import CharTokenizer, normalize_text


@dataclass
class PreparedData:
    items: list[dict]
    tokenizer: CharTokenizer
    mel_mean: torch.Tensor
    mel_std: torch.Tensor
    mel_min: float
    mel_max: float


class TTSDataset(Dataset):
    def __init__(self, subset: list[dict]):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]


def pad_1d(tensors, pad_value=0):
    max_len = max(t.shape[0] for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, tensor in enumerate(tensors):
        out[i, : tensor.shape[0]] = tensor
    return out


def pad_2d(tensors, pad_value=0.0):
    max_len = max(t.shape[0] for t in tensors)
    feat_dim = tensors[0].shape[1]
    out = torch.full((len(tensors), max_len, feat_dim), pad_value, dtype=tensors[0].dtype)
    for i, tensor in enumerate(tensors):
        out[i, : tensor.shape[0]] = tensor
    return out


def collate_batch(batch, pad_id: int):
    text_ids = [item["text_ids"] for item in batch]
    durations = [item["durations"] for item in batch]
    mels = [item["mel"] for item in batch]
    return {
        "item_id": [item["item_id"] for item in batch],
        "text": [item["text"] for item in batch],
        "text_ids": pad_1d(text_ids, pad_value=pad_id),
        "text_lengths": torch.tensor([len(x) for x in text_ids], dtype=torch.long),
        "durations": pad_1d(durations, pad_value=0),
        "mel": pad_2d(mels, pad_value=0.0),
        "mel_lengths": torch.tensor([len(x) for x in mels], dtype=torch.long),
    }


def prepare_items(config: Config, preset_symbol_to_id: dict[str, int] | None = None) -> PreparedData:
    metadata = pd.read_csv(
        config.metadata_path,
        sep="|",
        names=["item_id", "raw_text", "normalized_text"],
        quoting=3,
    )
    metadata["normalized_text"] = metadata["normalized_text"].fillna(metadata["raw_text"]).map(normalize_text)

    tokenizer = CharTokenizer(metadata["normalized_text"].tolist(), preset_symbol_to_id=preset_symbol_to_id)

    if config.aligned_cache_path.exists():
        payload = torch.load(config.aligned_cache_path, map_location="cpu", weights_only=False)
        items = payload["items"]
    else:
        align_bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        align_model = align_bundle.get_model().to(config.device)
        align_labels = align_bundle.get_labels()
        align_dict = {label: idx for idx, label in enumerate(align_labels)}

        rows = list(metadata.itertuples(index=False))
        if config.max_items > 0:
            rows = rows[: config.max_items]

        items = []
        for row in tqdm(rows, total=len(rows), desc="Preparing aligned data"):
            wav_path = config.wav_dir / f"{row.item_id}.wav"
            if not wav_path.exists():
                continue

            text = normalize_text(row.normalized_text)
            text_ids = np.asarray(tokenizer.encode(text), dtype=np.int64)
            if len(text_ids) == 0:
                continue

            wav, sr = load_audio_mono(wav_path)
            mel_wav = resample_audio(wav, sr, config.sample_rate)
            mel = extract_log_mel_from_waveform(mel_wav, config)

            try:
                durations = align_text_to_mel(
                    text,
                    wav,
                    sr,
                    mel.shape[0],
                    alignment_model=align_model,
                    align_bundle=align_bundle,
                    align_dict=align_dict,
                    device=config.device,
                )
            except Exception:
                continue

            if len(durations) != len(text_ids):
                continue

            items.append(
                {
                    "item_id": row.item_id,
                    "text": text,
                    "text_ids": torch.tensor(text_ids, dtype=torch.long),
                    "durations": torch.tensor(durations, dtype=torch.long),
                    "mel_raw": torch.tensor(mel, dtype=torch.float32),
                }
            )

        if config.device == "cuda":
            align_model = align_model.cpu()
            torch.cuda.empty_cache()
        del align_model
        torch.save({"items": items}, config.aligned_cache_path)

    if len(items) == 0:
        raise RuntimeError("No aligned items available. Check metadata/audio paths.")

    mel_sum = torch.zeros(config.n_mels)
    mel_sq_sum = torch.zeros(config.n_mels)
    mel_count = 0
    mel_min = float("inf")
    mel_max = float("-inf")
    for item in tqdm(items, total=len(items), desc="Computing mel stats"):
        mel = item["mel_raw"]
        mel_sum += mel.sum(dim=0)
        mel_sq_sum += (mel**2).sum(dim=0)
        mel_count += mel.shape[0]
        mel_min = min(mel_min, float(mel.min().item()))
        mel_max = max(mel_max, float(mel.max().item()))

    mel_mean = mel_sum / max(mel_count, 1)
    mel_var = mel_sq_sum / max(mel_count, 1) - mel_mean**2
    mel_std = torch.sqrt(torch.clamp(mel_var, min=1e-6))

    for item in tqdm(items, total=len(items), desc="Normalizing mels"):
        item["mel"] = (item["mel_raw"] - mel_mean) / mel_std

    return PreparedData(
        items=items,
        tokenizer=tokenizer,
        mel_mean=mel_mean,
        mel_std=mel_std,
        mel_min=mel_min,
        mel_max=mel_max,
    )


def build_loaders(config: Config, items: list[dict], pad_id: int):
    indices = np.arange(len(items))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    val_count = max(1, int(len(indices) * config.val_ratio))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    if len(train_idx) == 0:
        train_idx = val_idx

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]

    train_loader = DataLoader(
        TTSDataset(train_items),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.loader_workers,
        pin_memory=config.device == "cuda",
        collate_fn=lambda batch: collate_batch(batch, pad_id),
    )
    val_loader = DataLoader(
        TTSDataset(val_items),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.loader_workers,
        pin_memory=config.device == "cuda",
        collate_fn=lambda batch: collate_batch(batch, pad_id),
    )
    return train_loader, val_loader, train_items, val_items
