import os
import numpy as np
import pandas as pd
import gensim
import pickle
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from song_encoding import write_corpus_to_file
from dataset import new_dataset, DATA_FOLDER_PATH
from processing import (
    tokenize,
    vectorize_avg,
    vectorize_cov,
    vectorize_avg_with_cov,
)
import visualize

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter


CACHE_FOLDER_PATH = "local/cache"
CORPUS_FILEPATH = os.path.join(CACHE_FOLDER_PATH, "corpus.txt")


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


logging.info("Loading dataset...")
dataset = new_dataset(os.path.join(DATA_FOLDER_PATH, "maestro-v3.0.0.csv"))
logging.info("Dataset loaded.")


logging.info("Writing corpus...")
write_corpus_to_file(dataset, CORPUS_FILEPATH)
logging.info("Corpus written.")


logging.info("Tokenizing songs...")
tokenized_songs = tokenize(
    corpus_filepath=CORPUS_FILEPATH,
    modelName=os.path.join(CACHE_FOLDER_PATH, "spm"),
    vocabSize=13_000,
    maxSenLength=100_000,
)
logging.info("Tokenization complete.")


logging.info("Training Word2Vec model...")
word2vec_path = os.path.join(CACHE_FOLDER_PATH, "word2vec.model")
if os.path.exists(word2vec_path):
    word2vec = gensim.models.Word2Vec.load(word2vec_path)
else:
    word2vec = gensim.models.Word2Vec(
        sentences=tokenized_songs,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
    )
    word2vec.save(word2vec_path)
logging.info("Word2Vec model trained.")


logging.info("Vectorizing songs...")

# songs_vectors_avg_path = os.path.join(CACHE_FOLDER_PATH, "songs_vectors_avg.pkl")
# if os.path.exists(songs_vectors_avg_path):
#     with open(songs_vectors_avg_path, "rb") as f:
#         songs_vectors = pickle.load(f)
# else:
#     songs_vectors = vectorize_avg(tokenized_songs, word2vec)
#     with open(songs_vectors_avg_path, "wb") as f:
#         pickle.dump(songs_vectors, f)


songs_vectors_cov_path = os.path.join(CACHE_FOLDER_PATH, "songs_vectors_cov.pkl")
if os.path.exists(songs_vectors_cov_path):
    with open(songs_vectors_cov_path, "rb") as f:
        songs_vectors = pickle.load(f)
else:
    songs_vectors = vectorize_cov(tokenized_songs, word2vec)
    with open(songs_vectors_cov_path, "wb") as f:
        pickle.dump(songs_vectors, f)

# songs_vectors_avg_cov_path = os.path.join(CACHE_FOLDER_PATH, "songs_vectors_avg_cov.pkl")
# if os.path.exists(songs_vectors_avg_cov_path):
#     with open(songs_vectors_avg_cov_path, "rb") as f:
#         songs_vectors = pickle.load(f)
# else:
#     songs_vectors = vectorize_avg_with_cov(tokenized_songs, word2vec)
#     with open(songs_vectors_avg_cov_path, "wb") as f:
#         pickle.dump(songs_vectors, f)

logging.info("Vectorization complete.")


PredictorScaler = StandardScaler()
PredictorScalerFit = PredictorScaler.fit(songs_vectors)

scaled_songs_vectors = PredictorScalerFit.transform(songs_vectors)

composers = dataset["canonical_composer"].to_list()


def tsne_visualize(
    data_vectors: np.ndarray,
    labels: list,
    title: str = "t-SNE Visualization of Music Data",
    save_path: str = None,
    perplexity: int = 30,
    random_state: int = 42,
    figsize: tuple = (12, 10),
):
    # Convert labels to unique numerical indices
    unique_composers = sorted(list(set(labels)))
    label_to_index = {composer: i for i, composer in enumerate(unique_composers)}
    numeric_labels = np.array([label_to_index[composer] for composer in labels])

    # Print dataset information
    print(f"Number of data points: {len(data_vectors)}")
    print(f"Number of unique composers: {len(unique_composers)}")
    print("Composers and counts:")
    composer_counts = Counter(labels)
    for composer, count in sorted(composer_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {composer}: {count}")

    # Perform t-SNE dimensionality reduction
    print(f"Performing t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embedded_vectors = tsne.fit_transform(data_vectors)

    # Create plot
    plt.figure(figsize=figsize)

    # Plot each composer with a different color
    for i, composer in enumerate(unique_composers):
        mask = numeric_labels == i
        plt.scatter(
            embedded_vectors[mask, 0],
            embedded_vectors[mask, 1],
            label=composer,
            alpha=0.7,
            s=50,  # Point size
        )

    # Add labels and formatting
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE dimension 1", fontsize=12)
    plt.ylabel("t-SNE dimension 2", fontsize=12)
    plt.grid(alpha=0.3)

    # Add legend with appropriate positioning
    if len(unique_composers) > 10:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    else:
        plt.legend(fontsize=10)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


tsne_visualize(
    data_vectors=scaled_songs_vectors,
    labels=composers,
    title="t-SNE Visualization of Composer Music Styles",
    save_path=os.path.join(CACHE_FOLDER_PATH, "tsne_visualization_cov.png"),
)
