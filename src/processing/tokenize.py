import sentencepiece as spm
import os
import cache


MODEL_NAME = os.path.join(cache.FOLDER_PATH, "spm")


def tokenize(
    corpus_filepath,
    vocabSize,
    maxSenLength,
) -> list:
    if not os.path.exists(MODEL_NAME + ".model") or not os.path.exists(MODEL_NAME + ".vocab"):
        spm.SentencePieceTrainer.Train(
            input=corpus_filepath,
            model_prefix=MODEL_NAME,
            vocab_size=vocabSize,
            max_sentence_length=maxSenLength,
        )

    sp = spm.SentencePieceProcessor()
    sp.Load(MODEL_NAME + ".model")

    with open(corpus_filepath, "r") as f:
        corpus = f.read()

    return [sp.EncodeAsPieces(song) for song in corpus.split("\n")]
