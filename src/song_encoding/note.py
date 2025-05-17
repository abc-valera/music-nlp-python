from functools import total_ordering


@total_ordering
class Note:
    pitch: int
    velocity: int
    duration: float

    def __init__(self, pitch: int, velocity: int, duration: float):
        pitch = max(21, pitch)
        pitch = min(108, pitch)
        self.pitch = pitch

        # By default, MIDI velocity is in the range of 0-127.
        # Here it is being converted to a range of 0-4 for the internal representation.
        self.velocity = round(velocity * 5 / 127)

        duration = round(duration, 2)
        duration = max(0.0, duration)
        duration = min(15.0, duration)
        if duration < 0.5:
            # Round to 2 decimal places, then to nearest 0.05
            duration = round(duration * 20) / 20
        elif 0.5 <= duration < 2:
            # Round to 1 decimal place
            duration = round(duration, 1)
        elif 2 <= duration < 15:
            # Round to the nearest integer
            duration = round(duration)

        self.duration = duration

    def __eq__(self, other):
        return (
            self.pitch == other.pitch
            and self.velocity == other.velocity
            and self.duration == other.duration
        )

    def __lt__(self, other):
        if self.pitch == other.pitch:
            if self.velocity == other.velocity:
                return self.duration < other.duration
            return self.velocity < other.velocity
        return self.pitch < other.pitch

    def __hash__(self):
        return hash((self.pitch, self.velocity, self.duration))
