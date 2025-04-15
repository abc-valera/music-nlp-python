from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def tsne(
    X: np.ndarray,
    y: np.ndarray,
    composers: list,
    title: str,
    save_path: str = None,
):
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for i, composer in enumerate(composers):
        plt.scatter(
            X_tsne[np.array(y) == i, 0], X_tsne[np.array(y) == i, 1], label=composer, alpha=0.7
        )
    plt.legend()
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
