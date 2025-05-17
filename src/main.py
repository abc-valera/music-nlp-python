import os
import gensim
import pickle
import logging


import data
import song_encoding
import processing
import cache


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


logging.info("The script has started...")


dataset = data.new_dataset()
dataset = data.filter_dataset(dataset)
logging.info("Dataset loaded")


CORPUS_FILEPATH = os.path.join(cache.FOLDER_PATH, "corpus.txt")
NOTE2CHAR_FILEPATH = os.path.join(cache.FOLDER_PATH, "note2char.pkl")
CHAR2NOTE_FILEPATH = os.path.join(cache.FOLDER_PATH, "char2note.pkl")
if (
    os.path.exists(CORPUS_FILEPATH)
    and os.path.exists(NOTE2CHAR_FILEPATH)
    and os.path.exists(NOTE2CHAR_FILEPATH)
):
    logging.info("Corpus already exists, skipping creation.")
else:
    corpus, note2char, char2note = song_encoding.create_corpus(dataset)

    with open(CORPUS_FILEPATH, "w", encoding="utf-8") as f:
        f.write(corpus)
    with open(NOTE2CHAR_FILEPATH, "wb") as f:
        pickle.dump(note2char, f)
    with open(CHAR2NOTE_FILEPATH, "wb") as f:
        pickle.dump(char2note, f)

    logging.info("Corpus created")


TOKENIZED_SONGS_FILEPATH = os.path.join(cache.FOLDER_PATH, "tokenized_songs.pkl")
if os.path.exists(TOKENIZED_SONGS_FILEPATH):
    with open(TOKENIZED_SONGS_FILEPATH, "rb") as f:
        tokenized_songs = pickle.load(f)

    logging.info("Tokenized songs already exist, skipping tokenization.")
else:
    tokenized_songs = processing.tokenize(
        corpus_filepath=CORPUS_FILEPATH,
        vocabSize=20_000,
        maxSenLength=100_000,
    )

    with open(TOKENIZED_SONGS_FILEPATH, "wb") as f:
        pickle.dump(tokenized_songs, f)

    logging.info("Tokenization complete")


WORD2VEC_FILEPATH = os.path.join(cache.FOLDER_PATH, "word2vec.model")
if os.path.exists(WORD2VEC_FILEPATH):
    word2vec = gensim.models.Word2Vec.load(WORD2VEC_FILEPATH)
    logging.info("Word2Vec model loaded.")
else:
    word2vec = gensim.models.Word2Vec(
        sentences=tokenized_songs,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
    )
    word2vec.save(WORD2VEC_FILEPATH)

    logging.info("Word2Vec model trained and saved.")


SONGS_VECTORS_AVG_FILEPATH = os.path.join(cache.FOLDER_PATH, "songs_vectors_avg.pkl")
if os.path.exists(SONGS_VECTORS_AVG_FILEPATH):
    with open(SONGS_VECTORS_AVG_FILEPATH, "rb") as f:
        vectors_avg = pickle.load(f)
else:
    vectors_avg = processing.vectorize_avg(tokenized_songs, word2vec)
    with open(SONGS_VECTORS_AVG_FILEPATH, "wb") as f:
        pickle.dump(vectors_avg, f)


SONGS_VECTORS_COV_FILEPATH = os.path.join(cache.FOLDER_PATH, "songs_vectors_cov.pkl")
if os.path.exists(SONGS_VECTORS_COV_FILEPATH):
    with open(SONGS_VECTORS_COV_FILEPATH, "rb") as f:
        vectors_cov = pickle.load(f)
else:
    vectors_cov = processing.vectorize_cov(tokenized_songs, word2vec)
    with open(SONGS_VECTORS_COV_FILEPATH, "wb") as f:
        pickle.dump(vectors_cov, f)


SONGS_VECTORS_AVG_COV_FILEPATH = os.path.join(cache.FOLDER_PATH, "songs_vectors_avg_cov.pkl")
if os.path.exists(SONGS_VECTORS_AVG_COV_FILEPATH):
    with open(SONGS_VECTORS_AVG_COV_FILEPATH, "rb") as f:
        vectors_avg_cov = pickle.load(f)
else:
    vectors_avg_cov = processing.vectorize_avg_cov(tokenized_songs, word2vec)
    with open(SONGS_VECTORS_AVG_COV_FILEPATH, "wb") as f:
        pickle.dump(vectors_avg_cov, f)


processing.visualize_vectors(
    data_vectors=processing.scale_vectors(vectors_avg),
    labels=dataset["canonical_composer"].to_list(),
    title="t-SNE Visualization of AVG vectors",
    save_path=os.path.join(cache.FOLDER_PATH, "tsne_avg.png"),
)

processing.visualize_vectors(
    data_vectors=processing.scale_vectors(vectors_cov),
    labels=dataset["canonical_composer"].to_list(),
    title="t-SNE Visualization of COV vectors",
    save_path=os.path.join(cache.FOLDER_PATH, "tsne_cov.png"),
)

processing.visualize_vectors(
    data_vectors=processing.scale_vectors(vectors_avg_cov),
    labels=dataset["canonical_composer"].to_list(),
    title="t-SNE Visualization of AVG+COV vectors",
    save_path=os.path.join(cache.FOLDER_PATH, "tsne_avg_cov.png"),
)


# from sklearn.cluster import KMeans
# import numpy as np

# n_clusters = len(dataset["canonical_composer"].unique())

# kmeans_avg = KMeans(n_clusters=n_clusters, random_state=42)
# clusters_avg = kmeans_avg.fit_predict(np.array(vectors_avg))
# logging.info("KMeans clustering on AVG vectors complete.")

# kmeans_cov = KMeans(n_clusters=n_clusters, random_state=42)
# clusters_cov = kmeans_cov.fit_predict(np.array(vectors_cov))
# logging.info("KMeans clustering on COV vectors complete.")

# kmeans_avg_cov = KMeans(n_clusters=n_clusters, random_state=42)
# clusters_avg_cov = kmeans_avg_cov.fit_predict(np.array(vectors_avg_cov))
# logging.info("KMeans clustering on AVG+COV vectors complete.")

# # Optionally, save clustering results
# CLUSTERS_AVG_FILEPATH = os.path.join(cache.FOLDER_PATH, "clusters_avg.pkl")
# CLUSTERS_COV_FILEPATH = os.path.join(cache.FOLDER_PATH, "clusters_cov.pkl")
# CLUSTERS_AVG_COV_FILEPATH = os.path.join(cache.FOLDER_PATH, "clusters_avg_cov.pkl")

# with open(CLUSTERS_AVG_FILEPATH, "wb") as f:
#     pickle.dump(clusters_avg, f)
# with open(CLUSTERS_COV_FILEPATH, "wb") as f:
#     pickle.dump(clusters_cov, f)
# with open(CLUSTERS_AVG_COV_FILEPATH, "wb") as f:
#     pickle.dump(clusters_avg_cov, f)

# logging.info("Clustering results saved.")


# processing.visualize_vectors(
#     data_vectors=processing.scale_vectors(vectors_avg),
#     labels=clusters_avg,
#     title="t-SNE Visualization of AVG vectors with KMeans Clustering",
#     save_path=os.path.join(cache.FOLDER_PATH, "tsne_avg_kmeans.png"),
# )
# processing.visualize_vectors(
#     data_vectors=processing.scale_vectors(vectors_cov),
#     labels=clusters_cov,
#     title="t-SNE Visualization of COV vectors with KMeans Clustering",
#     save_path=os.path.join(cache.FOLDER_PATH, "tsne_cov_kmeans.png"),
# )
# processing.visualize_vectors(
#     data_vectors=processing.scale_vectors(vectors_avg_cov),
#     labels=clusters_avg_cov,
#     title="t-SNE Visualization of AVG+COV vectors with KMeans Clustering",
#     save_path=os.path.join(cache.FOLDER_PATH, "tsne_avg_cov_kmeans.png"),
# )
