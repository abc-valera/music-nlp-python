import pandas as pd
import os
from . import visualize

DATA_FOLDER_PATH = "local/maestro-v3.0.0"


def new_dataset() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "maestro-v3.0.0.csv"))

    # Add an ID column to track the original index
    df["id"] = df.index

    # Remove the extra fields
    df = df.astype(
        {
            "id": pd.Int64Dtype(),
            "canonical_composer": pd.StringDtype(),
            "canonical_title": pd.StringDtype(),
            "duration": pd.Float64Dtype(),
            "midi_filename": pd.StringDtype(),
        }
    )

    visualize.composer_song_counts(df)
    visualize.song_durations(df)

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
    )[["id", "canonical_composer", "canonical_title", "duration", "midi_filename"]]

    composer_names = [
        "Frédéric Chopin",
        "Franz Schubert",
        "Ludwig van Beethoven",
        "Franz Liszt",
        "Johann Sebastian Bach",
    ]
    df = df[df["canonical_composer"].isin(composer_names)]

    return df
