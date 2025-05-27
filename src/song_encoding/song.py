import pretty_midi
from .note import Note

CHORD_TIMEOUT = 0.005


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
            note = Note(
                pitch=midi_note.pitch,
                velocity=midi_note.velocity,
                duration=midi_note.end - midi_note.start,
            )

            is_note_part_of_chord = midi_note.start - chord_start_time <= CHORD_TIMEOUT

            if not is_note_part_of_chord:
                self._standardize_and_add_chord(chord)
                chord.clear()
                chord_start_time = midi_note.start

            chord.append(note)

            is_note_last_in_song = i == len(sorted_midi_notes) - 1
            if is_note_last_in_song:
                self._standardize_and_add_chord(chord)

    def _standardize_and_add_chord(self, chord: list[Note]):
        if not chord:
            return

        # Calculate average velocity and max duration for the chord
        avg_velocity = sum(note.velocity for note in chord) // len(chord)
        max_duration = max(note.duration for note in chord)

        # Standardize velocity and duration for all notes in the chord
        for note in chord:
            note.velocity = avg_velocity
            note.duration = max_duration

        self.extend(sorted(chord, key=lambda note: note.pitch, reverse=True))
