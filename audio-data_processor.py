from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

SAMPLE_RATE = 32000
CLIP_SECONDS = 5
TARGET_LEN = SAMPLE_RATE * CLIP_SECONDS

train_audio_metadata = pd.read_csv("birdclef-2026 data/metadata/train_audio_metadata.csv")
sample_path = train_audio_metadata.iloc[0]["path"]

#dataset class
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













def load_audio(path, sr=SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y



# y = load_audio(sample_path)

# # print("Waveform shape:", y.shape)
# # print("First 10 values:", y[:10])
# # print("Duration in seconds:", len(y) / SAMPLE_RATE)





def load_audio_fixed_length(path, sr=SAMPLE_RATE, target_len=TARGET_LEN):
    y, _ = librosa.load(path, sr=sr, mono=True)

    if len(y) < target_len:
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width))
    else:
        y = y[:target_len]

    return y


# y_fixed = load_audio_fixed_length(sample_path)

# print("Fixed waveform shape:", y_fixed.shape)
# print("Expected length:", TARGET_LEN)
# print("Actual length:", len(y_fixed))




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


# logmel = waveform_to_logmel(y_fixed)


#test:

# print("Logmel shape:", logmel.shape)
# print("Min:", logmel.min())
# print("Max:", logmel.max())
# print("Mean:", logmel.mean())
# print("Std:", logmel.std())

# plt.figure(figsize=(10, 4))
# plt.imshow(logmel, origin="lower", aspect="auto")
# plt.colorbar()
# plt.title("Log-Mel Spectrogram")
# plt.xlabel("Time")
# plt.ylabel("Mel bins")
# plt.tight_layout()
# plt.show()



def split_data():
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










train_df, val_df = split_data()

train_dataset = BirdAudioDataset(train_df)
val_dataset = BirdAudioDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

batch_x, batch_y = next(iter(train_loader))