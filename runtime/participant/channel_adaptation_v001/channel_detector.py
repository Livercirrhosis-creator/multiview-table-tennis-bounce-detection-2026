"""Read alternate temporal heatmaps from the existing visual forward only.

The wrapped detector continues returning its unchanged middle-channel result to
P1. Alternate observations are consumed at EOF, never backdated into P1. Only
decoded points are retained; the inference tensor is released after each call.
"""
import numpy as np
from participant.ball_detector import IN_H, IN_W, SCORE_TH


def decode_reliable(heat, width, height):
    if float(heat.max()) < .75:
        return None
    mask = heat > SCORE_TH
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    weights = heat[ys, xs].astype(np.float64)
    total = weights.sum()
    return dict(x=float((xs*weights).sum()/total)*width/IN_W,
                y=float((ys*weights).sum()/total)*height/IN_H,
                conf=float(heat[mask].max()))


class ChannelDetector:
    def __init__(self, detector, owner):
        self.detector, self.owner = detector, owner
        self.original_infer = detector._infer
        detector._infer = self._capture_heat
        self.rows, self.last_heat = {}, None

    def _capture_heat(self, inputs):
        result = self.original_infer(inputs)
        self.last_heat = result
        return result

    def reset(self):
        self.rows.clear()
        self.last_heat = None
        self.detector.reset()

    def detect(self, image, width, height):
        value = self.detector.detect(image, width, height)
        fid, heat = self.owner.current_input_frame, self.last_heat
        if heat is None or heat.shape != (1, 3, IN_H, IN_W):
            raise ValueError('Expected one existing three-channel ball forward')
        self.rows.setdefault(fid, {})[2] = decode_reliable(heat[0, 2], width, height)
        if fid >= 1:
            self.rows.setdefault(fid-1, {})[1] = dict(value) if value is not None else None
        if fid >= 2:
            previous = self.owner.observations[fid-1]['detection']
            reliable = previous is not None and previous['conf'] >= .75
            self.rows.setdefault(fid-2, {})[0] = None if reliable else decode_reliable(heat[0, 0], width, height)
        self.last_heat = None
        return value

    def native_records(self, video):
        if set(self.rows) != set(range(video['frame_count'])):
            raise ValueError('Incomplete native temporal-channel capture')
        return [dict(native_frame=frame, native_pts_seconds=video['pts_seconds'][frame],
            samples=[dict(channel=channel, all_pixel=point) for channel,point in sorted(self.rows[frame].items())])
            for frame in range(video['frame_count'])]
