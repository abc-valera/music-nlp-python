from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import matplotlib.colors as mcolors
import seaborn as sns


def visualize_vectors(
    data_vectors: np.ndarray,
    labels: list,
    title: str,
    save_path: str,
    perplexity: int = 30,
    random_state: int = 42,
    figsize: tuple = (16, 8),
):
    unique_composers = sorted(list(set(labels)))
    label_to_index = {composer: i for i, composer in enumerate(unique_composers)}
    numeric_labels = np.array([label_to_index[composer] for composer in labels])

    print(f"Performing t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embedded_vectors = tsne.fit_transform(data_vectors)

    plt.figure(figsize=figsize)
    # use default seaborn color palette
    palette = sns.color_palette("tab10")

    for i, composer in enumerate(unique_composers):
        mask = numeric_labels == i
        plt.scatter(
            embedded_vectors[mask, 0],
            embedded_vectors[mask, 1],
            label=composer,
            color=palette[i % len(palette)],
            alpha=0.7,
            s=50,  # Point size
        )

    plt.grid(alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
