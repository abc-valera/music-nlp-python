from functools import total_ordering


@total_ordering
class Note:
    pitch: int
    duration: float

    def __init__(self, pitch: int, duration: float):
        self.pitch = pitch
        # First round to 2 decimal places
        self.duration = round(duration, 2)
        # Then round to nearest 0.05
        self.duration = round(duration * 20) / 20

    def __eq__(self, other):
        return self.pitch == other.pitch and self.duration == other.duration

    def __lt__(self, other):
        if self.pitch == other.pitch:
            return self.duration < other.duration
        return self.pitch < other.pitch

    def __hash__(self):
        return hash((self.pitch, self.duration))
