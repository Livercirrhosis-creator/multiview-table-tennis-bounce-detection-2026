"""Use trajectory decisions to shortlist candidates for local RGB evidence."""
import numpy as np


def shortlist_ids(candidates, probabilities, dense_events, window_events, separation, *, include_boundary=False):
    if not candidates:
        return []
    times = np.asarray([c['native_pts_seconds'] for c in candidates])
    eligible = probabilities[:, 0] > probabilities[:, 1:].max(axis=1)
    event_times = [e['pts_seconds'] for events in (dense_events, window_events) for e in events]
    event_times = np.asarray(event_times)
    if include_boundary:
        starts = np.searchsorted(times, event_times - separation - 1e-12)
        stops = np.searchsorted(times, event_times + separation + 1e-12, side='right')
    else:
        # The event selector permits two points exactly one interval apart.
        starts = np.searchsorted(times, event_times - separation + 1e-12, side='right')
        stops = np.searchsorted(times, event_times + separation - 1e-12)
    for first, stop in zip(starts, stops):
        eligible[first:stop] = True
    return [candidate['candidate_id'] for candidate, keep in zip(candidates, eligible) if keep]
