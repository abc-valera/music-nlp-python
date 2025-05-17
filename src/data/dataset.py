import pandas as pd
import os
from .visualize import composer_song_counts, song_durations

DATA_FOLDER_PATH = "local/maestro-v3.0.0"


def new_dataset() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "maestro-v3.0.0.csv"))

    # Remove the extra fields
    df = df.astype(
        {
            "canonical_composer": pd.StringDtype(),
            "canonical_title": pd.StringDtype(),
            "duration": pd.Float64Dtype(),
            "midi_filename": pd.StringDtype(),
        }
    )

    return df


def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Remove songs with multiple composers
    df = df[~df["canonical_composer"].str.contains("/")]

    # Remove duplicate recordings
    df = (
        df.sort_values("duration")
        .groupby(["canonical_composer", "canonical_title"])
        .first()
        .reset_index()
    )[["canonical_composer", "canonical_title", "duration", "midi_filename"]]

    # Remove composers with few songs
    composer_counts = df["canonical_composer"].value_counts()
    df = df[df["canonical_composer"].isin(composer_counts[composer_counts >= 30].index)]

    composer_song_counts(df)
    song_durations(df)

    return df
