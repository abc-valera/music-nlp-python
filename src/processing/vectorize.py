import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


def vectorize_avg(tokenized_sentences: list, model: Word2Vec) -> list:
    vector_size = model.wv.vector_size
    vectorized_sentences = []

    for sentence in tqdm(
        tokenized_sentences,
        desc="Vectorizing sentences with average",
        unit="sentence",
    ):
        temp = np.zeros(vector_size)
        for word in sentence:
            temp += model.wv[word]
        vectorized_sentences.append(temp / len(sentence))

    return vectorized_sentences


def vectorize_cov(tokenized_sentences: list, model: Word2Vec) -> list:
    vectorized_sentences = []

    for sentence in tqdm(
        tokenized_sentences,
        desc="Vectorizing sentences with covariance",
        unit="sentence",
    ):
        cov = []
        for word in sentence:
            cov.append(model.wv[word])
        data = np.array(cov)
        sd = np.std(data, axis=0)
        z = sd.tolist()
        z = np.array(z)
        vectorized_sentences.append(z)

    return vectorized_sentences


def vectorize_avg_cov(tokenized_sentences: list, model: Word2Vec) -> list:
    vector_size = model.wv.vector_size
    vectorized_sentences = []

    for sentence in tqdm(
        tokenized_sentences,
        desc="Vectorizing sentences with average and covariance",
        unit="sentence",
    ):
        temp = np.zeros(vector_size)
        cov = []
        for word in sentence:
            temp += model.wv[word]
            cov.append(model.wv[word])
        data = np.array(cov)
        sd = np.std(data, axis=0)
        z = temp / len(sentence)
        z = z.tolist()
        z += sd.tolist()
        z = np.array(z)
        vectorized_sentences.append(z)

    return vectorized_sentences


def vectorize_tfidf(tokenized_sentences: list, model: Word2Vec) -> list:
    # First create a corpus of detokenized sentences for TF-IDF
    detokenized = [" ".join(sentence) for sentence in tokenized_sentences]

    # Fit TF-IDF
    tfidf = TfidfVectorizer().fit(detokenized)
    word_to_tfidf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    vector_size = model.wv.vector_size
    vectorized_sentences = []

    for sentence in tqdm(
        tokenized_sentences, desc="Vectorizing with TF-IDF weights", unit="sentence"
    ):
        weights = np.zeros(len(sentence))
        vectors = np.zeros((len(sentence), vector_size))

        for i, word in enumerate(sentence):
            if word in word_to_tfidf and word in model.wv:
                weights[i] = word_to_tfidf[word]
                vectors[i] = model.wv[word]

        if weights.sum() > 0:
            weighted_avg = np.average(vectors, weights=weights, axis=0)
        else:
            weighted_avg = np.zeros(vector_size)

        vectorized_sentences.append(weighted_avg)

    return vectorized_sentences


def vectorize_max_pool(tokenized_sentences: list, model: Word2Vec) -> list:
    vector_size = model.wv.vector_size
    vectorized_sentences = []

    for sentence in tqdm(tokenized_sentences, desc="Vectorizing with max pooling", unit="sentence"):
        if not sentence:
            vectorized_sentences.append(np.zeros(vector_size))
            continue

        # Initialize with the first word's vector or zeros
        if sentence[0] in model.wv:
            max_vector = model.wv[sentence[0]].copy()
        else:
            max_vector = np.zeros(vector_size)

        # Element-wise maximum of all word vectors
        for word in sentence[1:]:
            if word in model.wv:
                max_vector = np.maximum(max_vector, model.wv[word])

        vectorized_sentences.append(max_vector)

    return vectorized_sentences
