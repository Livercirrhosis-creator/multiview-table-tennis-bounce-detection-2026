"""Label-free native-time windows on the complete original observation stream."""
from bisect import bisect_left, bisect_right
if "." in __package__:
    from ..event_decision_v001.candidates import _observation_index
else:
    from event_decision_v001.candidates import _observation_index

RADIUS_SECONDS = 0.08
CONFIDENCE_MINIMUM = 0.75
MAX_OBSERVED_GAP_SECONDS = 0.08
EPS = 1e-9


def build_windows(video, observations, candidates):
    """One window per candidate; all row indices refer to this candidate list.

    Connectivity is established from *all* raw observations, including missing
    and low-confidence detections. Candidate sparsity never bridges a gap.
    Low-confidence existing anchors remain available as singleton windows.
    No labels, source groups, model scores or files enter this function.
    """
    by_frame, _ = _observation_index(video, observations)
    candidates = list(candidates)
    segments, frame_segment = [], {}
    current = None
    previous_frame = None
    for frame, observation in by_frame.items():
        detection = observation["detection"]
        if detection is None or detection["conf"] < CONFIDENCE_MINIMUM:
            current, previous_frame = None, None
            continue
        t = observation["native_pts_seconds"]
        if (current is None or frame != previous_frame + 1 or
                t - by_frame[previous_frame]["native_pts_seconds"] > MAX_OBSERVED_GAP_SECONDS + EPS):
            segments.append([t, t])
            current = len(segments) - 1
        segments[current][1] = t
        frame_segment[frame] = current
        previous_frame = frame

    members_by_segment, candidate_segments = {}, []
    seen_frames = set()
    for i, candidate in enumerate(candidates):
        frame = candidate["native_frame"]
        observation = by_frame.get(frame)
        if (frame in seen_frames or observation is None or observation["detection"] is None or
                candidate["video_id"] != video["clip_id"] or
                candidate["candidate_id"] != f"{video['clip_id']}:{frame}" or
                candidate["native_pts_seconds"] != observation["native_pts_seconds"] or
                any(candidate["center_observation"][k] != observation["detection"][k]
                    for k in ("x", "y", "conf"))):
            raise ValueError("Window candidate must preserve one unique original observation")
        seen_frames.add(frame)
        segment = frame_segment.get(frame, -(frame + 1))
        candidate_segments.append(segment)
        members_by_segment.setdefault(segment, []).append((candidate["native_pts_seconds"], i))
    segment_candidates = {}
    for segment, rows in members_by_segment.items():
        ordered = sorted(rows)
        segment_candidates[segment] = ([t for t, _ in ordered], [i for _, i in ordered])

    windows = []
    for i, (candidate, segment) in enumerate(zip(candidates, candidate_segments)):
        t, frame = candidate["native_pts_seconds"], candidate["native_frame"]
        if segment < 0:
            start, end, members = t, t, [i]
        else:
            start = max(t - RADIUS_SECONDS, segments[segment][0])
            end = min(t + RADIUS_SECONDS, segments[segment][1])
            times, rows = segment_candidates[segment]
            members = rows[bisect_left(times, start - EPS):bisect_right(times, end + EPS)]
        if i not in members:
            raise ValueError("A window must include its original center")
        windows.append(dict(window_index=i, window_id=f"{video['clip_id']}:w{frame}",
            video_id=video["clip_id"], center_row_index=i, center_frame=frame,
            start_time=start, end_time=end, member_rows=members, segment_id=segment))
    return windows
