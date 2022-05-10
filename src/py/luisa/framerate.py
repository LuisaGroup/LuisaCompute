import time
from functools import reduce

class FrameRate:
    def __init__(self, history_size: int) -> None:
        self.history_size: int = history_size
        self._durations_and_frames: list[tuple[float, int]] = []
        self.last: float = time.time()
    
    def clear(self) -> None:
        self._durations_and_frames.clear()
        self.last = time.time()

    def duration(self) -> float:
        return time.time() - self.last

    def record(self, frame_count: int = 1) -> None:
        if len(self._durations_and_frames) == self.history_size:
            self._durations_and_frames.pop(0)
        self._durations_and_frames.append((self.duration(), frame_count))
        self.last = time.time()

    def report(self) -> float:
        if len(self._durations_and_frames) == 0:
            return 0
        tuple_add = lambda x, y: (x[0] + y[0], x[1] + y[1])
        total_duration, total_frame_count = reduce(tuple_add, self._durations_and_frames)
        return total_frame_count / total_duration
