import sentencepiece as spm
import os


def tokenize(corpus_filepath, modelName, vocabSize, maxSenLength):
    if not os.path.exists(modelName + ".model"):
        spm.SentencePieceTrainer.train(
            input=corpus_filepath,
            model_prefix=modelName,
            vocab_size=vocabSize,
            max_sentence_length=maxSenLength,
        )

    sp = spm.SentencePieceProcessor()
    sp.load(modelName + ".model")

    with open(corpus_filepath, "r") as f:
        corpus = f.read()

    return [sp.encode_as_pieces(song) for song in corpus.split("\n")]
