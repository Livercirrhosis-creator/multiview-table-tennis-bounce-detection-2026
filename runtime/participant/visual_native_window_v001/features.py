"""Restrict CHANNEL_TABLE feature extraction to observations that can fill a slot.

The caller is build_view_inputs, which has already validated both complete
observation streams and the native clock via _observation_index. Only the
numeric tensors are consumed; full-video extractor diagnostics are not used.
"""
from bisect import bisect_left, bisect_right

import numpy as np

from participant.event_decision_v001.native_features import (
    MAX_SLOT_DISTANCE_SECONDS,
    TARGET_OFFSETS_SECONDS,
    TIME_EPSILON_SECONDS,
    extract_candidate,
)
from participant.event_decision_v001.trajectory_features import augment


def features(video, observations, pool):
    if not pool:
        return np.empty((0, 15, 23)), np.empty((0, 44))

    times = [float(t) for t in video['pts_seconds']]
    points = [row['detection'] for row in observations[1:]]
    radius = max(map(abs, TARGET_OFFSETS_SECONDS)) + MAX_SLOT_DISTANCE_SECONDS + TIME_EPSILON_SECONDS
    sequence, summary = [], []
    for candidate in pool:
        center = candidate['native_frame']
        timestamp = times[center]
        # One extra clock entry on each side keeps rounded bounds conservative.
        first = max(0, bisect_left(times, timestamp - radius) - 1)
        stop = min(len(times), bisect_right(times, timestamp + radius) + 1)
        observed = [dict(frame_id=local_frame, pts=times[first + local_frame], **point)
                    for local_frame, point in enumerate(points[first:stop]) if point is not None]
        result = extract_candidate(observed, center - first, times[first:stop],
                                   video['width'], video['height'])
        if result['status_2d'] != 'available':
            raise ValueError('Candidate must have an exact observation in its own view')
        sequence.append(result['sequence_2d'])
        summary.append(result['summary_2d'])
    sequence = np.asarray(sequence, dtype=np.float64)
    return sequence, augment(sequence, np.asarray(summary, dtype=np.float64))
