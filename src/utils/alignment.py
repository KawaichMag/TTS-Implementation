import numpy as np
import torch
import torchaudio

from utils.audio import resample_audio
from utils.tokenizer import normalize_text


def build_alignment_targets(text: str, align_dict: dict[str, int]):
    token_ids = []
    text_positions = []
    blank_id = align_dict["-"]
    word_sep_id = align_dict.get("|")

    for idx, ch in enumerate(normalize_text(text)):
        if ch in {" ", "-"} and word_sep_id is not None:
            token_ids.append(word_sep_id)
            text_positions.append(idx)
            continue

        token = ch.upper()
        token_id = align_dict.get(token)
        if token_id is not None and token_id != blank_id:
            token_ids.append(token_id)
            text_positions.append(idx)

    return token_ids, text_positions


def token_spans_to_mel_durations(spans, emission_frames: int, mel_frames: int) -> np.ndarray:
    if len(spans) == 0:
        return np.zeros(0, dtype=np.int64)

    boundaries = [0]
    for left, right in zip(spans[:-1], spans[1:]):
        boundaries.append(int(round((left.end + right.start) / 2)))
    boundaries.append(int(emission_frames))

    boundaries = np.asarray(boundaries, dtype=np.float64)
    mel_boundaries = np.rint(boundaries * mel_frames / emission_frames).astype(np.int64)
    mel_boundaries[0] = 0
    mel_boundaries[-1] = mel_frames
    mel_boundaries = np.clip(mel_boundaries, 0, mel_frames)
    mel_boundaries = np.maximum.accumulate(mel_boundaries)
    return np.diff(mel_boundaries)


def align_text_to_mel(
    text: str,
    wav: np.ndarray,
    sr: int,
    mel_frames: int,
    alignment_model,
    align_bundle,
    align_dict: dict[str, int],
    device: str,
) -> np.ndarray:
    normalized_text = normalize_text(text)
    token_ids, text_positions = build_alignment_targets(normalized_text, align_dict)
    if not token_ids:
        raise ValueError("Text has no symbols supported by the alignment model.")

    aligned_wav = resample_audio(wav, sr, align_bundle.sample_rate)
    waveform = torch.tensor(aligned_wav, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.inference_mode():
        emission, _ = alignment_model(waveform)
        log_probs = torch.log_softmax(emission, dim=-1).cpu()

    aligned_tokens, scores = torchaudio.functional.forced_align(
        log_probs,
        torch.tensor([token_ids], dtype=torch.int32),
        blank=align_dict["-"],
    )
    spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0].exp())
    if len(spans) != len(token_ids):
        raise RuntimeError("Alignment produced an unexpected number of token spans.")

    supported_durations = token_spans_to_mel_durations(spans, int(log_probs.shape[1]), mel_frames)
    durations = np.zeros(len(normalized_text), dtype=np.int64)
    for text_idx, duration in zip(text_positions, supported_durations):
        durations[text_idx] = int(duration)

    if durations.sum() != mel_frames:
        raise RuntimeError(f"Aligned durations sum to {durations.sum()} frames, expected {mel_frames}.")
    return durations
