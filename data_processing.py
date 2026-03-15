from pathlib import Path
import pandas as pd
import json



ROOT = Path("~/Documents/ML/birdclef-2026").expanduser()
TRAIN_AUDIO = ROOT / "train_audio"
TRAIN_SOUNDSCAPE = ROOT / "train_soundscapes"
SOUNDSCAPE_LABELS_CSV = ROOT / "train_soundscapes_labels.csv"



#metadata process
def build_train_audio_metadata(train_audio_dir: Path) -> pd.DataFrame:
    rows = []

    for species_dir in sorted(train_audio_dir.iterdir()):
        if not species_dir.is_dir():
            continue

        species = species_dir.name

        for audio_path in species_dir.glob("*.ogg"):
            rows.append({
                "path": str(audio_path),
                "filename": audio_path.name,
                "species": species,
            })

    df = pd.DataFrame(rows)
    return df


def build_soundscape_metadata(labels_csv: Path, soundscape_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df = df.drop_duplicates().copy()
    df["path"] = df["filename"].apply(lambda x: str(soundscape_dir / x))
    return df


train_audio_metadata = build_train_audio_metadata(TRAIN_AUDIO)
soundscape_metadata = build_soundscape_metadata(SOUNDSCAPE_LABELS_CSV, TRAIN_SOUNDSCAPE)

METADATA_DIR = ROOT / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

train_audio_metadata.to_csv(METADATA_DIR / "train_audio_metadata.csv", index=False)
soundscape_metadata.to_csv(METADATA_DIR / "train_soundscape_metadata.csv", index=False)



#species vocab process
all_species = sorted(train_audio_metadata["species"].unique())

species_to_idx = {sp: i for i, sp in enumerate(all_species)}
idx_to_species = {i: sp for sp, i in species_to_idx.items()}

train_audio_metadata["target_idx"] = train_audio_metadata["species"].map(species_to_idx)

# print(train_audio_metadata.head())
# print("Number of species:", len(all_species))
# print("Example mapping:", list(species_to_idx.items())[:10])

with open(ROOT / "species_idx.json", "w") as f:
    json.dump(species_to_idx, f)

train_audio_metadata.to_csv(ROOT / "metadata" / "train_audio_metadata.csv", index=False)