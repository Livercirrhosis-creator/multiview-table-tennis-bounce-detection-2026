"""Native video timestamps for the official pts=None lifecycle; no fps guesses."""
from math import isfinite
from pathlib import Path
import time


class NativeTime:
    """Use explicit timestamps, or lazily index the actual input video's PTS.

    The video and manifest are normal runtime inputs. No development lookup
    table, fixed fps, or changes to the official core are needed. If true PTS
    cannot be read, fail with a concrete error rather than changing P1's 80ms.
    """
    def __init__(self, context=None, video_meta=None):
        self.context = context
        self.video_meta = video_meta
        self.values = []
        self.file_pts = None
        self.index_seconds = 0.0
        self.sources = set()

    def _index_actual_video(self):
        if self.context is None or self.video_meta is None:
            raise ValueError('Missing PTS requires reset(VideoMeta) and prepare(RunContext)')
        path = Path(self.context.input_dir) / self.video_meta.path
        if not path.is_file():
            raise FileNotFoundError(f'Native PTS source video not found: {path}')
        import av
        start = time.perf_counter()
        pts = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(f'Video frame {len(pts)} has no native timestamp: {path}')
                value = float(frame.pts * frame.time_base)
                if not isfinite(value) or (pts and value <= pts[-1]):
                    raise ValueError(f'Non-finite or non-increasing native PTS at frame {len(pts)}: {path}')
                pts.append(value)
        self.index_seconds = time.perf_counter() - start
        if len(pts) != self.video_meta.n_frames:
            raise ValueError(f'PTS frame count {len(pts)} != manifest {self.video_meta.n_frames}: {path}')
        self.file_pts = pts
        for fid, existing in enumerate(self.values):
            if abs(existing - pts[fid]) > 1e-9:
                raise ValueError(f'Explicit PTS disagree with video PTS at frame {fid}')

    def resolve(self, frame_meta):
        fid = frame_meta['frame_id']
        if isinstance(fid, bool) or not isinstance(fid, int) or fid != len(self.values):
            raise ValueError('Contiguous zero-based frame_id values are required')
        supplied = frame_meta.get('pts')
        if supplied is None:
            if self.file_pts is None:
                self._index_actual_video()
            if fid >= len(self.file_pts):
                raise ValueError('Input frame exceeds native PTS video length')
            value = self.file_pts[fid]
            self.sources.add('actual_video_pts_pyav')
        else:
            if isinstance(supplied, bool):
                raise ValueError('A Boolean is not a timestamp')
            value = float(supplied)
            self.sources.add('explicit_frame_pts')
            if self.file_pts is not None and (fid >= len(self.file_pts) or abs(value-self.file_pts[fid]) > 1e-9):
                raise ValueError(f'Explicit PTS disagree with video PTS at frame {fid}')
        if not isfinite(value) or (self.values and value <= self.values[-1]):
            raise ValueError('Native PTS must be finite and strictly increasing')
        self.values.append(value)
        return value

    def receipt(self):
        return {'sources': sorted(self.sources), 'native_pts_count':len(self.values),
                'video_pts_index_count':None if self.file_pts is None else len(self.file_pts),
                'pts_index_seconds':self.index_seconds, 'fps_estimation_used':False}
