import os
import gensim
import pickle
import logging
from typing import Callable
import numpy as np
from gensim.models import Word2Vec


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
logging.info("Dataset loaded")


CORPUS_FILEPATH = os.path.join(cache.FOLDER_PATH, "1_corpus.txt")
NOTE2CHAR_FILEPATH = os.path.join(cache.FOLDER_PATH, "1_note2char.pkl")
CHAR2NOTE_FILEPATH = os.path.join(cache.FOLDER_PATH, "1_char2note.pkl")
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


TOKENIZED_SONGS_FILEPATH = os.path.join(cache.FOLDER_PATH, "2_tokenized_songs.pkl")
if os.path.exists(TOKENIZED_SONGS_FILEPATH):
    with open(TOKENIZED_SONGS_FILEPATH, "rb") as f:
        tokenized_songs = pickle.load(f)

    logging.info("Tokenized songs already exist, skipping tokenization.")
else:
    tokenized_songs = processing.tokenize(
        corpus_filepath=CORPUS_FILEPATH,
        vocabSize=25_000,
        maxSenLength=100_000,
    )

    with open(TOKENIZED_SONGS_FILEPATH, "wb") as f:
        pickle.dump(tokenized_songs, f)

    logging.info("Tokenization complete")


WORD2VEC_FILEPATH = os.path.join(cache.FOLDER_PATH, "3_word2vec.model")
if os.path.exists(WORD2VEC_FILEPATH):
    word2vec = gensim.models.Word2Vec.load(WORD2VEC_FILEPATH)
    logging.info("Word2Vec model loaded.")
else:
    word2vec = gensim.models.Word2Vec(
        sentences=tokenized_songs,
        window=10,
        min_count=1,
        workers=4,
        sg=1,
    )
    word2vec.save(WORD2VEC_FILEPATH)

    logging.info("Word2Vec model trained and saved.")


dataset = data.filter_dataset(dataset)
relevant_ids = dataset["id"].to_list()
tokenized_songs = [tokenized_songs[i] for i in relevant_ids]


def save_vectors(vectorisation_method, vectorize_func: Callable[[list, Word2Vec], list]):
    filepath = os.path.join(cache.FOLDER_PATH, f"4_songs_vectors_{vectorisation_method}.pkl")
    if not os.path.exists(filepath):
        vectors = vectorize_func(tokenized_songs, word2vec)

        with open(filepath, "wb") as f:
            pickle.dump(vectors, f)

        processing.visualize_vectors(
            data_vectors=np.array(processing.scale_vectors(vectors)),
            labels=dataset["canonical_composer"].to_list(),
            title=f"t-SNE Visualization of {vectorisation_method} vectors",
            save_path=os.path.join(cache.FOLDER_PATH, f"tsne_{vectorisation_method}.png"),
        )


vectors_avg = save_vectors("avg", processing.vectorize_avg)
vectors_std = save_vectors("std", processing.vectorize_std)
vectors_avg_std = save_vectors("avg_std", processing.vectorize_avg_std)
vectors_tfidf = save_vectors("tfidf", processing.vectorize_tfidf)
vectors_max_pool = save_vectors("max_pool", processing.vectorize_max_pool)
