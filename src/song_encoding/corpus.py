import os
import pandas as pd
import pickle
from tqdm import tqdm
import pretty_midi
from dataset import DATA_FOLDER_PATH

from .note import Note
from .song import Song


def write_corpus_to_file(dataset: pd.DataFrame, filepath: str):
    if os.path.exists(filepath):
        print("Corpus file already exists. Skipping writing.")
        return

    songs = list[Song]()
    all_notes = list[Note]()
    for i in tqdm(range(dataset.shape[0]), desc="Processing MIDI files", unit="file"):
        suffix_path = dataset["midi_filename"][i]
        path = os.path.join(DATA_FOLDER_PATH, suffix_path)
        song = Song(pretty_midi.PrettyMIDI(path))

        songs.append(song)
        all_notes.extend(song)

    unique_sorted_notes = sorted(set(all_notes))

    print(len(unique_sorted_notes))

    note2ch = {note: encodeChinese(i) for i, note in enumerate(unique_sorted_notes)}
    ch2note = {encodeChinese(i): note for i, note in enumerate(unique_sorted_notes)}

    corpus = list[str]()
    for song in tqdm(songs, desc="Processing Songs"):
        encoded_song = list[str]()
        for note in song:
            encoded_song.append(note2ch[note])
        corpus.append("".join(encoded_song))

    f = open(filepath, "w")
    f.write("\n".join(corpus))
    f.close()


# Функція енкодидь число у китайський символ
def encodeChinese(index_number):
    val = index_number + 0x4E00
    return chr(val)
