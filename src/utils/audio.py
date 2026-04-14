import librosa
import numpy as np
import soundfile as sf
import torch

from models.config import Config


def load_audio_mono(wav_path):
    wav, sr = sf.read(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), sr


def resample_audio(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav.astype(np.float32)
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def extract_log_mel_from_waveform(wav: np.ndarray, config: Config) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
    )
    return np.log(np.clip(mel, 1e-5, None)).T.astype(np.float32)


def denormalize_mel(mel_tensor: torch.Tensor, mel_mean: torch.Tensor, mel_std: torch.Tensor) -> torch.Tensor:
    return mel_tensor * mel_std.to(mel_tensor.device) + mel_mean.to(mel_tensor.device)


def log_mel_to_audio(log_mel: np.ndarray, config: Config, mel_min: float, mel_max: float, n_iter: int = 48):
    safe_log_mel = np.asarray(log_mel, dtype=np.float32)
    safe_log_mel = np.nan_to_num(safe_log_mel, nan=mel_min, neginf=mel_min, posinf=mel_max)
    safe_log_mel = np.clip(safe_log_mel, mel_min, mel_max)
    mel_power = np.exp(safe_log_mel).T.astype(np.float32)
    mel_power = np.nan_to_num(mel_power, nan=1e-5, neginf=1e-5, posinf=float(np.exp(mel_max)))
    mel_power = np.clip(mel_power, 1e-5, float(np.exp(mel_max)))
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
        n_iter=n_iter,
    )
    audio = np.nan_to_num(audio, nan=0.0, neginf=0.0, posinf=0.0)
    if np.max(np.abs(audio)) > 0:
        audio = librosa.util.normalize(audio)
    return audio.astype(np.float32)
