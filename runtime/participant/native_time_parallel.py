"""Bounded decoder threading for full native PTS indexing; no packet shortcut."""
from math import isfinite
from pathlib import Path
import time

from .native_time import NativeTime

PTS_DECODE_THREADS = 2
FRAME_THREADS_CAPABILITY = 0x1000


class _UseOriginal(Exception):
    pass


class ParallelNativeTime(NativeTime):
    """Keep NativeTime's resolve contract and fully validate before returning PTS.

    Only the internal FFmpeg codec has up to two requested decoding threads.
    There is no Python worker, next-video prefetch, input-order assumption,
    packet timestamp sorting, frame skipping, FPS estimate or persistent cache.
    A container is closed before success, failure or fallback reaches the caller.
    """
    def __init__(self, context=None, video_meta=None):
        super().__init__(context, video_meta)
        self.parallel_attempt_seconds = 0.0
        self.parallel_used = False
        self.parallel_fallback = None

    def _index_actual_video(self):
        if self.context is None or self.video_meta is None:
            return super()._index_actual_video()
        path = Path(self.context.input_dir) / self.video_meta.path
        if not path.is_file():
            return super()._index_actual_video()
        import av
        start = time.perf_counter()
        try:
            pts = []
            signature = None
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                codec = stream.codec_context
                if not int(codec.codec.capabilities) & FRAME_THREADS_CAPABILITY:
                    raise _UseOriginal("codec lacks frame-thread capability")
                codec.thread_count = PTS_DECODE_THREADS
                codec.thread_type = "AUTO"
                for frame in container.decode(stream):
                    current = (frame.width, frame.height, frame.format.name)
                    if signature is None:
                        signature = current
                    if current != signature:
                        raise _UseOriginal("decoded frame dimensions or pixel format changed")
                    if frame.is_corrupt:
                        raise _UseOriginal("decoder reported a corrupt frame")
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError(f"Video frame {len(pts)} has no native timestamp: {path}")
                    value = float(frame.pts * frame.time_base)
                    if not isfinite(value) or (pts and value <= pts[-1]):
                        raise ValueError(f"Non-finite or non-increasing native PTS at frame {len(pts)}: {path}")
                    pts.append(value)
            if len(pts) != self.video_meta.n_frames:
                raise ValueError(f"PTS frame count {len(pts)} != manifest {self.video_meta.n_frames}: {path}")
            for fid, existing in enumerate(self.values):
                if abs(existing - pts[fid]) > 1e-9:
                    raise ValueError(f"Explicit PTS disagree with video PTS at frame {fid}")
        except Exception as error:
            # Do not turn an error into guessed timestamps. Re-run the original
            # decoder/validation, propagating its original failure if it fails.
            self.parallel_attempt_seconds = time.perf_counter() - start
            self.parallel_fallback = f"{type(error).__name__}: {error}"
            super()._index_actual_video()
            self.index_seconds += self.parallel_attempt_seconds
            return
        self.file_pts = pts
        self.parallel_used = True
        self.parallel_attempt_seconds = time.perf_counter() - start
        self.index_seconds = self.parallel_attempt_seconds

    def receipt(self):
        result = super().receipt()
        result.update(pts_parallel_used=self.parallel_used,
                      pts_parallel_threads_requested=PTS_DECODE_THREADS,
                      pts_parallel_attempt_seconds=self.parallel_attempt_seconds,
                      pts_parallel_fallback=self.parallel_fallback)
        return result
