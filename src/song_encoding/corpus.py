import os
import pandas as pd
from tqdm import tqdm
import pretty_midi
import data


from .note import Note
from .song import Song


def create_corpus(dataset: pd.DataFrame) -> tuple[str, dict[Note, str], dict[str, Note]]:
    songs = list[Song]()
    all_notes = list[Note]()
    for i in tqdm(range(dataset.shape[0]), desc="Processing MIDI files", unit="file"):
        suffix_path = dataset.iloc[i]["midi_filename"]
        path = os.path.join(data.DATA_FOLDER_PATH, suffix_path)
        song = Song(pretty_midi.PrettyMIDI(path))

        songs.append(song)
        all_notes.extend(song)

    unique_sorted_notes = sorted(set(all_notes))

    note2char = {note: encodeIntAsChinese(i) for i, note in enumerate(unique_sorted_notes)}
    char2note = {encodeIntAsChinese(i): note for i, note in enumerate(unique_sorted_notes)}

    corpus = list[str]()
    for song in tqdm(songs, desc="Processing Songs"):
        encoded_song = list[str]()
        for note in song:
            encoded_song.append(note2char[note])
        corpus.append("".join(encoded_song))

    return "\n".join(corpus), note2char, char2note


# encodeIntAsChar encodes an integer to a chinese character.
# Since all chinese characters are located in the range of 0x4E00 to 0x9FFF,
# the max value of the index number is 0x9FFF - 0x4E00 = 20,991.
def encodeIntAsChinese(index_number):
    val = index_number + 0x4E00
    return chr(val)
