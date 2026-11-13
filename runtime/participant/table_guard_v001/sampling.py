"""Three bounded RGB samples from the parent's native PTS and RGB stream."""
from math import isfinite
from statistics import median
import time


TARGETS = (0.0, 0.48, 0.96)
TIE_EPSILON = 1e-12


def duration_seconds(pts):
    if len(pts) < 2:
        return 0.0
    intervals = [b-a for a, b in zip(pts, pts[1:]) if b > a]
    return pts[-1]-pts[0]+median(intervals) if intervals else 0.0


def nearest_frames(pts):
    result = []
    for target in TARGETS:
        errors = [abs(p-pts[0]-target) for p in pts]
        minimum = min(errors)
        result.append(next(f for f, err in enumerate(errors) if err <= minimum+TIE_EPSILON))
    return result


class Samples:
    def __init__(self, max_video_seconds=10.0):
        self.max_video_seconds = max_video_seconds
        self.slots = [None]*3
        self.closed = [False]*3
        self.mode = None
        self.indices = None
        self.first_pts = None
        self.last_pts = None
        self.frames_seen = 0
        self.long_video = False
        self.resize_calls = 0
        self.sampling_seconds = 0.0
        self.peak_cache_bytes = 0

    def consume(self, rgb, frame_id, pts, file_pts=None):
        started = time.perf_counter()
        try:
            if (frame_id != self.frames_seen or not isfinite(pts)
                    or self.last_pts is not None and pts <= self.last_pts):
                raise ValueError("Table sampling requires the parent's contiguous native PTS")
            if self.mode is None:
                self.first_pts = pts
                self.mode = "existing_native_index" if file_pts is not None else "online_native_pts"
                if file_pts is not None:
                    self.long_video = duration_seconds(file_pts) > self.max_video_seconds
                    self.indices = nearest_frames(file_pts) if not self.long_video else None
            self.last_pts = pts
            self.frames_seen += 1
            if pts-self.first_pts > self.max_video_seconds:
                self.long_video = True
            if self.long_video:
                self.clear()
                return
            if self.indices is not None:
                replace = [i for i, f in enumerate(self.indices) if f == frame_id]
            else:
                # The current sample is the adjacent frame before each not-yet-crossed
                # target. Compare the first frame after crossing before closing it.
                replace = []
                for i, target in enumerate(TARGETS):
                    if self.closed[i]:
                        continue
                    error = abs(pts-self.first_pts-target)
                    old = self.slots[i]
                    if old is None or error < old["error"]-TIE_EPSILON:
                        replace.append(i)
                    if pts-self.first_pts >= target:
                        self.closed[i] = True
            if replace:
                import cv2
                small = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_LINEAR)
                self.resize_calls += 1
                for i in replace:
                    self.slots[i] = dict(frame_id=frame_id, pts_seconds=pts,
                        target_relative_seconds=TARGETS[i], error=abs(pts-self.first_pts-TARGETS[i]), rgb=small)
                distinct = {id(s["rgb"]): s["rgb"] for s in self.slots if s is not None}
                self.peak_cache_bytes = max(self.peak_cache_bytes, sum(a.nbytes for a in distinct.values()))
        finally:
            self.sampling_seconds += time.perf_counter()-started

    def receipt(self):
        return dict(mode=self.mode, targets_seconds=list(TARGETS), tie_epsilon_seconds=TIE_EPSILON,
            tie_rule="earlier native frame", frames_seen=self.frames_seen,
            sample_frames=[None if s is None else s["frame_id"] for s in self.slots],
            sample_pts_seconds=[None if s is None else s["pts_seconds"] for s in self.slots],
            resize_calls=self.resize_calls, sampling_seconds=self.sampling_seconds,
            rgb_cache_peak_bytes=self.peak_cache_bytes, rgb_cache_limit_bytes=3*320*320*3,
            extra_rgb_decode=False, nominal_fps_used=False)

    def clear(self):
        self.slots = [None]*3
