"""Fixed P1 event algorithm with bounded native-PTS decoder threading."""
from .p1_entry import Solution as FixedP1
from .native_time_parallel import ParallelNativeTime


class Solution(FixedP1):
    def reset(self, video_meta):
        super().reset(video_meta)
        self.time_source = ParallelNativeTime(getattr(self, "context", None), video_meta)
