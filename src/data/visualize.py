import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import cache


def composer_song_counts(dataset: pd.DataFrame) -> None:
    composer_counts = dataset["canonical_composer"].value_counts().head(10).iloc[::-1]
    plt.figure(figsize=(10, max(6, len(composer_counts) * 0.4)))
    bars = plt.barh(composer_counts.index, composer_counts.values.astype(float), height=0.7)
    for bar in bars:
        plt.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())),
            va="center",
        )
    plt.title("Number of Pieces per Composer (Top 10)", fontsize=14)
    plt.xlabel("Number of Pieces")
    plt.tight_layout()
    plt.savefig(
        os.path.join(cache.FOLDER_PATH, "composer_song_counts.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def song_durations(dataset: pd.DataFrame) -> None:
    duration_minutes = dataset["duration"] / 60
    mean_duration = dataset["duration"].mean()
    median_duration = dataset["duration"].median()

    plt.figure(figsize=(10, 6))
    sns.histplot(duration_minutes.to_numpy(), bins=30, kde=True)
    plt.axvline(
        mean_duration / 60,
        color="r",
        linestyle="--",
        label=f"Mean: {mean_duration / 60:.1f} min",
    )
    plt.axvline(
        median_duration / 60,
        color="g",
        linestyle=":",
        label=f"Median: {median_duration / 60:.1f} min",
    )
    plt.title("Distribution of Piece Durations", fontsize=14)
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(cache.FOLDER_PATH, "song_durations.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
