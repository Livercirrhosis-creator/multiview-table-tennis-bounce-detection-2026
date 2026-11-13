"""Native-time RGB inputs identical to the fixed 96px visual training path."""
from bisect import bisect_left

import numpy as np
from PIL import Image


def frame_ids(pts, center):
    result = []
    for offset in (-0.08, 0.0, 0.08):
        if offset == 0:
            result.append(center)
            continue
        target = pts[center] + offset
        if target < pts[0] - 1e-8 or target > pts[-1] + 1e-8:
            result.append(None)
            continue
        lo, hi = (0, center) if offset < 0 else (center + 1, len(pts))
        index = bisect_left(pts, target, lo, hi)
        options = [i for i in (index - 1, index) if lo <= i < hi]
        selected = min(options, key=lambda i: (abs(pts[i] - target), i), default=None)
        if selected is not None and abs(pts[selected] - target) > 0.04 + 1e-8:
            selected = None
        result.append(selected)
    return result


def crop_rgb96(rgb, x, y, side):
    """Same physical crop/padding as training, with explicit 96px bilinear resize."""
    height, width = rgb.shape[:2]
    left, top = round(x - side / 2), round(y - side / 2)
    right, bottom = left + side, top + side
    crop = rgb[max(top, 0):min(bottom, height), max(left, 0):min(right, width)]
    if crop.shape[:2] != (side, side):
        crop = np.pad(crop, ((max(-top, 0), max(bottom-height, 0)),
                            (max(-left, 0), max(right-width, 0)), (0, 0)))
    if crop.shape != (side, side, 3):
        raise ValueError('Unexpected physical crop extent')
    return np.asarray(Image.fromarray(crop).resize((96, 96), Image.Resampling.BILINEAR)).transpose(2, 0, 1)


def inputs(frames, pts, candidates, height):
    sides = tuple(max(32, round(scale * height / 720)) for scale in (80, 192))
    pixels = np.zeros((len(candidates), 3, 2, 3, 96, 96), dtype=np.uint8)
    times = np.zeros((len(candidates), 3, 2), dtype=np.float32)
    for row, candidate in enumerate(candidates):
        center = candidate['native_frame']
        observation = candidate['center_observation']
        for slot, index in enumerate(frame_ids(pts, center)):
            if index is None:
                continue
            times[row, slot] = pts[index] - pts[center], 1
            for scale, side in enumerate(sides):
                pixels[row, slot, scale] = crop_rgb96(
                    frames[index], observation['x'], observation['y'], side)
    return pixels, times
