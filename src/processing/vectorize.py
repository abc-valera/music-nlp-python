import numpy as np
from tqdm import tqdm


def vectorize_avg(tokenized_sentences, model):
    vectorized_sentences = []
    for sentence in tqdm(
        tokenized_sentences,
        desc="Vectorizing sentences with average",
        unit="sentence",
    ):
        temp = np.zeros(0)
        for word in sentence:
            temp = np.zeros(len(model.wv[word]))
            temp += model.wv[word]
        vectorized_sentences.append(temp / len(sentence))
    return vectorized_sentences


def vectorize_cov(tokenized_sentences, model):
    vectorized_sentences = []
    cov = []
    for sentence in tqdm(
        tokenized_sentences,
        desc="Vectorizing sentences with covariance",
        unit="sentence",
    ):
        for word in sentence:
            cov.append(model.wv[word])
        data = np.array(cov)
        sd = np.std(data, axis=0)
        z = sd.tolist()
        z = np.array(z)
        vectorized_sentences.append(z)
    return vectorized_sentences


def vectorize_avg_cov(tokenized_sentences, model):
    vectorized_sentences = []
    cov = []
    for sentence in tqdm(
        tokenized_sentences,
        desc="Vectorizing sentences with average and covariance",
        unit="sentence",
    ):
        temp = np.zeros(0)
        for word in sentence:
            temp = np.zeros(len(model.wv[word]))
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
