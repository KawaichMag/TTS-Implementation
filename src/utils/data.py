from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np

from torch.utils.data import Dataset

from .text import symbols


def tokenize_text(text):
    return [symbols.index(i) for i in text if i in symbols]


def parse_texts(texts_path: Path):
    texts = {}
    print("Taking texts...")
    with open(texts_path, "r", encoding="utf-8") as f:
        for line in f:
            audio_id, transcript, norm_transcript = line.strip().split("|")
            texts[audio_id] = transcript
    return texts


def parse_mels(audios_path: Path):
    mels = {}
    print("Taking audio files...")

    for file in tqdm(audios_path.iterdir()):
        audio_id = str(file).split("\\")[-1][:-4]
        audio, sr = librosa.load(file)

        audio = audio / np.max(np.abs(audio))

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            power=1.0,
        )

        mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))

        mels[audio_id] = mel_spectrogram

    return mels


class TextMelDataset(Dataset):
    def __init__(self, mels: dict, texts: dict):
        self.mels = mels
        self.texts = texts
        self.idx_key_mapping = list(texts.keys())

    def __getitem__(self, index):
        return self.mels[self.idx_key_mapping[index]], self.texts[
            self.idx_key_mapping[index]
        ]

    def __len__(self):
        return len(self.mels)


def get_dataset(audios_path: Path, texts_path: Path):
    mels = parse_mels(audios_path)
    texts = parse_texts(texts_path)

    return TextMelDataset(mels, texts)


def log_mel_to_audio(log_mel_spec):
    mel_spec = np.exp(log_mel_spec)

    mel_inverter = librosa.feature.inverse.mel_to_stft(
        M=mel_spec, sr=22050, n_fft=1024, power=1.0
    )

    audio = librosa.griffinlim(mel_inverter, n_iter=32, hop_length=256, win_length=1024)

    return audio


if __name__ == "__main__":
    DATASETS_PATH = Path("./src/datasets")

    dataset = get_dataset(DATASETS_PATH / "golden_set", DATASETS_PATH / "metadata.csv")

    print(dataset)
