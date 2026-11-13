"""Pure native-time 2D evidence, with explicit missingness and geometry state.

Observation dictionaries contain frame_id (ALREADY native), x, y, conf and
optional pts. Padding is excluded by the caller; no frame shift occurs here.
The 15 physical slots encode input features, not resampled/modified videos.
Geometry state is metadata only, never a numeric feature or a deletion gate.
"""
from __future__ import annotations

from copy import deepcopy
import math

SCHEMA = "native_dual_2d_evidence.v1"
TARGET_OFFSETS_SECONDS = tuple(i / 100 for i in range(-28, 29, 4))
CENTER_SLOT = 7
MAX_SLOT_DISTANCE_SECONDS = 0.02
CONFIDENT_MINIMUM = 0.75
TIME_EPSILON_SECONDS = 1e-12
SLOT_CHANNELS = (
    "observed", "confident", "confidence", "actual_dt_seconds", "assignment_distance_seconds",
    "x_div_width", "y_div_height", "relative_position_known", "relative_x_div_width", "relative_y_div_height",
    "previous_observed", "next_observed", "previous_gap_seconds", "next_gap_seconds",
    "velocity_prev_known", "vx_prev", "vy_prev", "velocity_next_known", "vx_next", "vy_next",
    "turn_known", "delta_vx", "delta_vy",
)
SUMMARY_NAMES = tuple("center_" + name for name in SLOT_CHANNELS) + (
    "observed_fraction", "confident_fraction", "left_observed_fraction", "right_observed_fraction")
MASK_CHANNELS = ("observed", "confident", "relative_position_known", "velocity_prev_known",
                 "velocity_next_known", "turn_known")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _frame(value):
    return type(value) is int and value >= 0


def _geometry(value):
    if value is None:
        return {"status": "not_attempted", "source_ref": None, "reason": None}
    if not isinstance(value, dict) or set(value) - {"status", "source_ref", "reason"}:
        raise ValueError("Geometry API accepts only status/source_ref/reason metadata")
    status = value.get("status")
    if status not in ("not_attempted", "inferred", "unavailable"):
        raise ValueError("Geometry state must distinguish not_attempted/inferred/unavailable")
    source = value.get("source_ref")
    reason = value.get("reason")
    if source is not None and (not isinstance(source, dict) or not source):
        raise ValueError("Geometry source_ref must be a nonempty reference object or null")
    if status == "inferred" and source is None:
        raise ValueError("Inferred geometry requires a source evidence reference, not a reliability claim")
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ValueError("Geometry reason must be a nonempty string or null")
    if status == "unavailable" and reason is None:
        raise ValueError("Unavailable geometry requires an explicit reason")
    return deepcopy({"status": status, "source_ref": source, "reason": reason})


def _empty(center, geometry, reason, center_time=None, diagnostics=None):
    sequence = [[0.0] * len(SLOT_CHANNELS) for _ in TARGET_OFFSETS_SECONDS]
    slots = [{"slot_index": i, "target_offset_seconds": dt,
              "target_pts_seconds": None if center_time is None else center_time + dt,
              "native_frame": None, "pts_seconds": None, "actual_dt_seconds": None,
              "assignment_distance_seconds": None, "observed": False, "confident": False}
             for i, dt in enumerate(TARGET_OFFSETS_SECONDS)]
    return {"schema": SCHEMA, "status_2d": "unavailable", "reason": reason,
            "center_native_frame": center if _frame(center) else None,
            "center_pts_seconds": center_time, "sequence_2d": sequence,
            "summary_2d": [0.0] * len(SUMMARY_NAMES), "slots": slots,
            "masks": {name: [False] * 15 for name in MASK_CHANNELS},
            "geometry_evidence": geometry, "center_motion_reliable": False,
            "diagnostics": diagnostics or {}, "frame_offset_applied": 0,
            "training_eligibility_constructed": False}


def _nearest_slot(delta):
    best, distance = None, math.inf
    for i, target in enumerate(TARGET_OFFSETS_SECONDS):
        current = abs(delta - target)
        if current < distance - TIME_EPSILON_SECONDS:
            best, distance = i, current
    return best, distance


def extract_candidate(observations, center_native_frame, pts_seconds, width, height,
                      geometry_evidence=None):
    """Return 15x23 sequence, 27-vector summary and source/missingness metadata.

    Low-confidence but finite in-image detections remain observations. Motion
    requires confidence >= .75 at both immediately adjacent assigned slots;
    missing slots are never bridged. Turning is next minus previous velocity,
    not an acceleration or a physical contact decision. Velocities have units
    of image widths/heights per real second. Features are not fitted or clipped.

    Missing exact center explicitly yields unavailable 2D, retaining other slot
    observations for inspection with relative_position_known=False. A nearby
    frame is never substituted for the center. Invalid clocks/identity return
    unavailable evidence; malformed geometry metadata raises an API error.
    """
    geometry = _geometry(geometry_evidence)
    if not all(_finite(x) and x > 0 for x in (width, height)):
        return _empty(center_native_frame, geometry, "invalid_dimensions")
    if (not isinstance(pts_seconds, (list, tuple)) or not pts_seconds
            or not all(_finite(t) for t in pts_seconds)):
        return _empty(center_native_frame, geometry, "invalid_native_clock")
    times = [float(t) for t in pts_seconds]
    if any(b <= a for a, b in zip(times, times[1:])):
        return _empty(center_native_frame, geometry, "nonincreasing_native_clock")
    if not _frame(center_native_frame) or center_native_frame >= len(times):
        return _empty(center_native_frame, geometry, "invalid_center_native_frame")
    center_time = times[center_native_frame]
    stats = {"input_rows": 0, "invalid_frame_rows": 0, "invalid_value_rows": 0,
             "duplicate_rows": 0, "conflicting_native_frames": [], "outside_slots": 0,
             "noncenter_nearest_center_slot_excluded": 0}
    result = _empty(center_native_frame, geometry, "missing_exact_center", center_time, stats)
    values, conflicts = {}, set()
    for row in (() if observations is None else observations):
        stats["input_rows"] += 1
        if (not isinstance(row, dict) or not _frame(row.get("frame_id"))
                or row["frame_id"] >= len(times)):
            stats["invalid_frame_rows"] += 1
            continue
        fid = row["frame_id"]
        if "pts" in row and (not _finite(row["pts"]) or abs(row["pts"] - times[fid]) > TIME_EPSILON_SECONDS):
            return _empty(center_native_frame, geometry, "observation_pts_identity_mismatch", center_time, stats)
        value = None
        if (all(_finite(row.get(k)) for k in ("x", "y", "conf"))
                and 0 <= row["x"] < width and 0 <= row["y"] < height and 0 <= row["conf"] <= 1):
            value = tuple(float(row[k]) for k in ("x", "y", "conf"))
        else:
            stats["invalid_value_rows"] += 1
        if fid in values:
            stats["duplicate_rows"] += 1
            if values[fid] != value:
                conflicts.add(fid)
        else:
            values[fid] = value
    stats["conflicting_native_frames"] = sorted(conflicts)
    observed = {f: v for f, v in values.items() if v is not None and f not in conflicts}
    selected = {}
    for fid in sorted(observed):
        delta = times[fid] - center_time
        if not math.isfinite(delta):
            return _empty(center_native_frame, geometry, "nonfinite_relative_time", center_time, stats)
        slot, distance = _nearest_slot(delta)
        if distance > MAX_SLOT_DISTANCE_SECONDS + TIME_EPSILON_SECONDS:
            stats["outside_slots"] += 1
            continue
        if slot == CENTER_SLOT and fid != center_native_frame:
            stats["noncenter_nearest_center_slot_excluded"] += 1
            continue
        previous = selected.get(slot)
        if (previous is None or distance < previous[1] - TIME_EPSILON_SECONDS
                or abs(distance - previous[1]) <= TIME_EPSILON_SECONDS and fid < previous[0]):
            selected[slot] = (fid, distance)
    center = observed.get(center_native_frame)
    if center is not None:
        if selected.get(CENTER_SLOT, (None,))[0] != center_native_frame:
            raise AssertionError("Exact center assignment failed")
        result.update(status_2d="available", reason="ok")
    elif center_native_frame in conflicts:
        result["reason"] = "conflicting_exact_center"
    index = {name: i for i, name in enumerate(SLOT_CHANNELS)}
    for slot, (fid, distance) in selected.items():
        x, y, confidence = observed[fid]
        row = result["sequence_2d"][slot]
        data = {"observed": 1., "confident": float(confidence >= CONFIDENT_MINIMUM),
                "confidence": confidence, "actual_dt_seconds": times[fid] - center_time,
                "assignment_distance_seconds": distance, "x_div_width": x / width, "y_div_height": y / height,
                "relative_position_known": float(center is not None),
                "relative_x_div_width": (x - center[0]) / width if center is not None else 0.,
                "relative_y_div_height": (y - center[1]) / height if center is not None else 0.}
        for name, value in data.items():
            row[index[name]] = value
        result["slots"][slot].update(native_frame=fid, pts_seconds=times[fid],
            actual_dt_seconds=times[fid] - center_time, assignment_distance_seconds=distance,
            observed=True, confident=confidence >= CONFIDENT_MINIMUM)
    for slot, (fid, _) in selected.items():
        row = result["sequence_2d"][slot]
        x, y, confidence = observed[fid]
        for neighbor, direction in ((slot - 1, "prev"), (slot + 1, "next")):
            if neighbor not in selected:
                continue
            other = selected[neighbor][0]
            ox, oy, other_confidence = observed[other]
            gap = (times[fid] - times[other]) if direction == "prev" else (times[other] - times[fid])
            if not math.isfinite(gap) or gap <= 0:
                return _empty(center_native_frame, geometry, "invalid_motion_time_difference", center_time, stats)
            long_name = "previous" if direction == "prev" else "next"
            row[index[long_name + "_observed"]] = 1.
            row[index[long_name + "_gap_seconds"]] = gap
            if min(confidence, other_confidence) < CONFIDENT_MINIMUM:
                continue
            sign = 1 if direction == "prev" else -1
            row[index["velocity_" + direction + "_known"]] = 1.
            row[index["vx_" + direction]] = sign * (x - ox) / width / gap
            row[index["vy_" + direction]] = sign * (y - oy) / height / gap
        if row[index["velocity_prev_known"]] and row[index["velocity_next_known"]]:
            row[index["turn_known"]] = 1.
            row[index["delta_vx"]] = row[index["vx_next"]] - row[index["vx_prev"]]
            row[index["delta_vy"]] = row[index["vy_next"]] - row[index["vy_prev"]]
    if not all(math.isfinite(value) for row in result["sequence_2d"] for value in row):
        return _empty(center_native_frame, geometry, "nonfinite_feature_arithmetic", center_time, stats)
    for name in MASK_CHANNELS:
        result["masks"][name] = [bool(row[index[name]]) for row in result["sequence_2d"]]
    observed_mask, confident_mask = result["masks"]["observed"], result["masks"]["confident"]
    result["summary_2d"] = result["sequence_2d"][CENTER_SLOT].copy() + [
        sum(observed_mask) / 15, sum(confident_mask) / 15,
        sum(observed_mask[:CENTER_SLOT]) / 7, sum(observed_mask[CENTER_SLOT + 1:]) / 7]
    result["center_motion_reliable"] = result["masks"]["turn_known"][CENTER_SLOT]
    ids = [s["native_frame"] for s in result["slots"] if s["observed"]]
    if len(ids) != len(set(ids)):
        raise AssertionError("Observation reused across feature slots")
    stats.update(observed_slots=sum(observed_mask), confident_slots=sum(confident_mask),
                 missing_slots=15 - sum(observed_mask),
                 center_observed=center is not None, context_can_be_missing=True)
    return result
