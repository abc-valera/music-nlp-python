import os
import pandas as pd


DATA_FOLDER_PATH = "local/maestro-v3.0.0"


def new_dataset(csv_path: str) -> pd.DataFrame:
    dataset = pd.read_csv(csv_path)

    # Remove songs with multiple composers
    dataset = dataset[~dataset["canonical_composer"].str.contains("/")]

    # Remove duplicate recordings
    dataset = (
        dataset.sort_values("duration")
        .groupby(["canonical_composer", "canonical_title"])
        .first()
        .reset_index()
    )[["canonical_composer", "canonical_title", "midi_filename"]]

    # Remove the extra fields
    dataset = dataset.astype(
        {
            "canonical_composer": pd.StringDtype(),
            "canonical_title": pd.StringDtype(),
            "midi_filename": pd.StringDtype(),
        }
    )

    return dataset
