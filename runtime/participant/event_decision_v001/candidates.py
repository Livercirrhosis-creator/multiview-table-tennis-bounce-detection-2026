"""A label-free union of real native-time anchors and localization observations.

Inputs are complete single-video observations and the original unsuppressed P1
candidate stream. No file, label, model or dataset access occurs in this module.
The pool is built before event conflict handling; no minimum gap is imposed here.
"""
from copy import deepcopy
import math

from .e03_events import EventConfig, generate_events

E03_CONFIG = dict(variant="R", confidence_threshold=.75, velocity_reference_fps=25.,
                  change_threshold=12, max_neighbor_gap_sec=.08, min_event_gap_frames=10,
                  peak_relative_tolerance=1e-9, extra_side_points=2, max_side_span_sec=.16)
P1_SOURCE, E03_SOURCE, LOCAL_SOURCE = "p1_precooldown", "e03_R12", "localization_support"
LOCAL_WINDOW_SECONDS = .08
LOCAL_CONFIDENCE = .75
TIME_EPS = 1e-9
E03_REFERENCE_CANVAS = (1920, 1080)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _observation_index(video, observations):
    """Validate the existing ch1 mapping without applying a second frame shift."""
    video_id = video["clip_id"]
    width, height, count = (video[k] for k in ("width", "height", "frame_count"))
    pts = video["pts_seconds"]
    if (not isinstance(video_id, str) or not video_id or
            any(type(n) is not int or n <= 0 for n in (width, height, count)) or
            len(pts) != count or not all(_finite(t) for t in pts) or
            any(b <= a for a, b in zip(pts, pts[1:]))):
        raise ValueError("Invalid single-video identity/dimensions/native clock")
    rows = list(observations)
    if len(rows) != count:
        raise ValueError("Complete original input-frame observation sequence required")
    by_frame, e03_rows = {}, []
    for input_frame, row in enumerate(rows):
        if (row["video_id"] != video_id or type(row["input_frame"]) is not int or
                row["input_frame"] != input_frame or row["input_pts_seconds"] != pts[input_frame] or
                (row["width"], row["height"]) != (width, height)):
            raise ValueError("Observation input identity, clock or dimensions differ")
        if input_frame == 0:
            if (row["center_is_padding"] is not True or row["native_frame"] is not None or
                    row["native_pts_seconds"] is not None):
                raise ValueError("Initial padding cannot be a native candidate")
            continue
        native = row["native_frame"]
        if (type(native) is not int or native != input_frame - 1 or
                row["center_is_padding"] is not False or row["native_pts_seconds"] != pts[native]):
            raise ValueError("Native middle-channel mapping differs; do not shift twice")
        detection = row["detection"]
        if detection is not None and (not isinstance(detection, dict) or set(detection) != {"x", "y", "conf"} or
                not all(_finite(detection[k]) for k in detection) or
                not 0 <= detection["x"] < width or not 0 <= detection["y"] < height or
                not 0 <= detection["conf"] <= 1):
            raise ValueError("Invalid original detector observation")
        by_frame[native] = row
        point = dict(frame_id=native, timestamp_sec=row["native_pts_seconds"], visible=detection is not None)
        if detection is not None:
            point.update(x=detection["x"], y=detection["y"], confidence=detection["conf"])
        e03_rows.append(point)
    return by_frame, e03_rows


def build_pool(video, observations, original_candidates, external_candidates=()):
    """Return ``(candidates, diagnostics)`` for one complete native-time video.

    ``video`` and ``observations`` use reservoir runtime/videos.json and
    runtime/observations.jsonl schemas. ``original_candidates`` is this video's
    original P1 *pre-cooldown* subset (921 across the current development pool),
    not the 799 emitted outputs. The caller owns provenance and dataset access.

    Every output has one real native observation and a sorted source list.
    External candidates use the same identity/center fields plus a nonempty
    ``sources`` list and matching ``source_evidence`` dict. External geometry
    may nominate a native frame but may not replace its real detector center.
    Localization adds real observations within .08 seconds of an anchor in
    the same uninterrupted high-confidence segment. Missing/low-confidence
    observations and time gaps over .08 seconds break segments. No resampling
    or online availability is inferred. Source evidence is not a model input.
    The old P1
    availability is retained only inside its source evidence; computing this
    complete-trajectory pool does not establish online release timing.
    """
    by_frame, points = _observation_index(video, observations)
    video_id, width, height = video["clip_id"], video["width"], video["height"]
    original = list(original_candidates)
    pooled, original_frames = {}, set()

    def add(frame, source, evidence):
        row = by_frame.get(frame)
        if row is None or row["detection"] is None:
            raise ValueError("Candidate center must be an actual nonpadding observation")
        detection = row["detection"]
        if frame not in pooled:
            pooled[frame] = dict(candidate_id=f"{video_id}:{frame}", video_id=video_id,
                native_frame=frame, native_pts_seconds=row["native_pts_seconds"], width=width, height=height,
                center_observation=dict(input_frame=row["input_frame"], **deepcopy(detection)),
                sources=[], source_evidence={}, support_anchor_frames=[],
                availability=dict(phase="complete_trajectory_replay", online_release_established=False))
        item = pooled[frame]
        if source in item["source_evidence"]:
            raise ValueError("Duplicate source candidate at one native frame")
        item["sources"].append(source)
        item["source_evidence"][source] = deepcopy(evidence)

    def validate_anchor(candidate):
        frame = candidate["native_frame"]
        if (type(frame) is not int or candidate["video_id"] != video_id or
                candidate["candidate_id"] != f"{video_id}:{frame}" or
                (candidate["width"], candidate["height"]) != (width, height)):
            raise ValueError("Candidate anchor identity differs")
        row = by_frame.get(frame)
        center = candidate["center_observation"]
        if (row is None or row["detection"] is None or candidate["native_pts_seconds"] != row["native_pts_seconds"] or
                center["input_frame"] != row["input_frame"] or
                any(center[k] != row["detection"][k] for k in ("x", "y", "conf"))):
            raise ValueError("Candidate anchor must preserve exact raw center, time and coordinates")
        return frame

    for candidate in original:
        frame = validate_anchor(candidate)
        if frame in original_frames:
            raise ValueError("Original pre-cooldown candidate repeats")
        original_frames.add(frame)
        add(frame, P1_SOURCE, dict(availability=deepcopy(candidate.get("availability"))))

    # Only the scoring copy uses reference pixels; pool centers stay native.
    sx, sy = E03_REFERENCE_CANVAS[0] / width, E03_REFERENCE_CANVAS[1] / height
    reference_points = [dict(p, x=p["x"] * sx, y=p["y"] * sy) if p["visible"] else dict(p) for p in points]
    # The copied frozen generator also computes its historical gap-filtered
    # selected list. Deliberately consume only the pre-cooldown candidates.
    _, e03 = generate_events(reference_points, EventConfig(**E03_CONFIG))
    e03["coordinate_space"] = "1920x1080_reference_pixels"
    e03_frames = set()
    for candidate in e03["candidates"]:
        frame = candidate["frame_id"]
        row = by_frame.get(frame)
        if (row is None or row["detection"] is None or candidate["timestamp_sec"] != row["native_pts_seconds"] or
                candidate["x"] != row["detection"]["x"] * sx or candidate["y"] != row["detection"]["y"] * sy or
                candidate["confidence"] != row["detection"]["conf"]):
            raise ValueError("E03 reference candidate does not identify its real native observation")
        e03_frames.add(frame)
        add(frame, E03_SOURCE, dict(candidate, coordinate_space="1920x1080_reference_pixels"))

    external_frames = set()
    for candidate in external_candidates:
        frame = validate_anchor(candidate)
        sources, evidence = candidate["sources"], candidate["source_evidence"]
        if (not isinstance(sources, list) or not sources or
                any(not isinstance(s, str) or not s or s in (P1_SOURCE, E03_SOURCE, LOCAL_SOURCE) for s in sources) or
                len(set(sources)) != len(sources) or not isinstance(evidence, dict) or set(evidence) != set(sources)):
            raise ValueError("External source names/evidence must be unique and nonreserved")
        external_frames.add(frame)
        for source in sources:
            add(frame, source, evidence[source])

    anchor_frames = set(pooled)
    anchor_sources = {frame: sorted(row["sources"]) for frame, row in pooled.items()}
    segments, frame_segment, previous = [], {}, None
    for frame, row in by_frame.items():
        detection = row["detection"]
        if detection is None or detection["conf"] < LOCAL_CONFIDENCE:
            previous = None
            continue
        timestamp = row["native_pts_seconds"]
        if previous is None or timestamp - previous > LOCAL_WINDOW_SECONDS + TIME_EPS:
            segments.append([])
        segments[-1].append(frame)
        frame_segment[frame] = len(segments) - 1
        previous = timestamp
    supports = {}
    for anchor in sorted(anchor_frames):
        if anchor not in frame_segment:  # An existing low-confidence anchor remains unexpanded.
            continue
        timestamp = by_frame[anchor]["native_pts_seconds"]
        for frame in segments[frame_segment[anchor]]:
            if frame != anchor and abs(by_frame[frame]["native_pts_seconds"] - timestamp) <= LOCAL_WINDOW_SECONDS + TIME_EPS:
                supports.setdefault(frame, []).append(anchor)
    for frame, anchors in supports.items():
        add(frame, LOCAL_SOURCE, dict(anchor_sources={str(a): anchor_sources[a] for a in anchors},
            window_seconds=LOCAL_WINDOW_SECONDS, maximum_connected_gap_seconds=LOCAL_WINDOW_SECONDS,
            confidence_threshold=LOCAL_CONFIDENCE))
        pooled[frame]["support_anchor_frames"] = anchors
    result = [pooled[frame] for frame in sorted(pooled)]
    for row in result:
        row["sources"].sort()
    if not original_frames <= set(pooled) or set(pooled) != anchor_frames | set(supports):
        raise AssertionError("Candidate union removed or fabricated a center")
    diagnostics = dict(schema="event_decision_precooldown_pool.v1", video_id=video_id,
        frames=video["frame_count"], original_candidates=len(original), e03_candidates=len(e03_frames),
        overlap=len(original_frames & e03_frames), newly_added_e03=len(e03_frames - original_frames),
        external_anchor_frames=len(external_frames), anchor_frames=len(anchor_frames),
        localization_support_candidates=len(supports), newly_added_localization=len(set(supports) - anchor_frames),
        pooled_candidates=len(result), sources=sorted({s for row in result for s in row["sources"]}),
        localization=dict(window_seconds=LOCAL_WINDOW_SECONDS, confidence_threshold=LOCAL_CONFIDENCE,
            maximum_connected_gap_seconds=LOCAL_WINDOW_SECONDS, breaks_at_missing_or_low_confidence=True),
        all_original_candidates_preserved=True, native_frame_offset_applied=0,
        e03_reference_canvas=list(E03_REFERENCE_CANVAS), output_coordinate_space="native_pixels",
        raw_observation_coordinates_unchanged=True,
        pool_minimum_gap_applied=False, e03_selected_list_ignored=True,
        online_release_established=False, gt_used=False, model_calls=0, e03=e03)
    return result, diagnostics
