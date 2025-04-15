import pretty_midi
from .note import Note


CHORD_TIMEOUT = 0.003


class Song(list[Note]):
    def __init__(self, midi: pretty_midi.PrettyMIDI):
        super().__init__()

        if not midi.instruments or not midi.instruments[0].notes:
            return

        sorted_midi_notes = sorted(
            midi.instruments[0].notes,
            key=lambda note: note.start,
        )

        chord = list[Note]()
        chord_start_time = sorted_midi_notes[0].start

        for i, midi_note in enumerate(sorted_midi_notes):
            note_gram = Note(
                pitch=midi_note.pitch,
                duration=midi_note.end - midi_note.start,
            )

            # Notes that are close in time are considered to be part of the same chord.
            # They are sorted in descending order by their pitch.

            is_note_part_of_chord = midi_note.start - chord_start_time <= CHORD_TIMEOUT

            is_note_last_in_song = i == len(sorted_midi_notes) - 1

            if not is_note_part_of_chord:
                self.extend(sorted(chord, key=lambda note: note.pitch, reverse=True))
                chord.clear()
                chord_start_time = midi_note.start

            chord.append(note_gram)

            if is_note_last_in_song:
                self.extend(sorted(chord, key=lambda note: note.pitch, reverse=True))
