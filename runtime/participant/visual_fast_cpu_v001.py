"""Keep the visual model inputs, reducing repeated CPU preprocessing work."""

import cv2
import numpy as np

from . import native_time_parallel
from .ball_detector import IN_H, IN_W, _MEAN, _STD
from .visual_calibration_v001.solution import Solution as VisualSolution


_PIXEL_VALUES = np.arange(256, dtype=np.float32).reshape(256, 1, 1)
_NORMALIZED_RGB = np.ascontiguousarray((_PIXEL_VALUES / 255.0 - _MEAN) / _STD)


def normalize_rgb(frame_rgb):
    resized = cv2.resize(frame_rgb, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
    normalized = cv2.LUT(resized, _NORMALIZED_RGB)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


class Solution(VisualSolution):
    def prepare(self, context):
        native_time_parallel.PTS_DECODE_THREADS = 4
        super().prepare(context)
        detector = self.channel_detector.detector
        values = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
        if detector._normalize(values).tobytes() != normalize_rgb(values).tobytes():
            raise RuntimeError('RGB normalization lookup differs from the original')
        detector._normalize = normalize_rgb
        context.info['cpu_fast_path'] = {'native_pts_threads': 4, 'rgb_normalization': 'float32_lookup'}
