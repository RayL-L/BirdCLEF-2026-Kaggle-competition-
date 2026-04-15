import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 32000
CLIP_SECONDS = 5
TARGET_LEN = SAMPLE_RATE * CLIP_SECONDS


def load_audio_fixed_length(path, sr=SAMPLE_RATE, target_len=TARGET_LEN):
    y, _ = librosa.load(path, sr=sr, mono=True)

    if len(y) < target_len:
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width))
    else:
        y = y[:target_len]

    return y


def waveform_to_logmel(
    y,
    sr=SAMPLE_RATE,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    fmin=20,
    fmax=16000,
):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    mean = logmel.mean()
    std = logmel.std() + 1e-6
    logmel = (logmel - mean) / std

    return logmel


class BirdAudioDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = row["path"]
        target = int(row["target_idx"])

        y = load_audio_fixed_length(path)
        logmel = waveform_to_logmel(y)

        x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(target, dtype=torch.long)

        return x, y


def split_data(train_audio_metadata):
    species_counts = train_audio_metadata["target_idx"].value_counts()

    rare_classes = species_counts[species_counts < 2].index
    common_classes = species_counts[species_counts >= 2].index

    rare_df = train_audio_metadata[
        train_audio_metadata["target_idx"].isin(rare_classes)
    ].copy()

    common_df = train_audio_metadata[
        train_audio_metadata["target_idx"].isin(common_classes)
    ].copy()

    train_common_df, val_df = train_test_split(
        common_df,
        test_size=0.2,
        random_state=42,
        stratify=common_df["target_idx"]
    )

    train_df = pd.concat([train_common_df, rare_df], ignore_index=True)

    return train_df, val_df