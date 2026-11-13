#!/usr/bin/env python3
"""Independent public-data reproduction of the ping-pong contest score.

Uses only Python's standard library. This is not the hidden official scorer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

ANGLES = ("0度", "45度", "90度")
FRAME_TOLERANCE = 1
POSITION_TOLERANCE = 50.0


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def matches(pred, gt):
    return abs(pred["frame_id"] - gt["frame_id"]) <= FRAME_TOLERANCE and math.hypot(
        pred["x"] - gt["x"], pred["y"] - gt["y"]) <= POSITION_TOLERANCE


def maximum_matching(adjacency, n_gt, order=None):
    """Maximum-cardinality bipartite matching by iterative augmenting paths.

    Returns prediction-index -> GT-index; no greedy nearest-only matching.
    Optional left-vertex order preserves the maximum feasible match count of
    earlier priority groups when the groups are processed consecutively.
    """
    right = [-1] * n_gt
    left = {}
    for start in (range(len(adjacency)) if order is None else order):
        queue = deque([start])
        visited_left = {start}
        previous = {}
        endpoint = None
        while queue and endpoint is None:
            p = queue.popleft()
            for g in adjacency[p]:
                if g in previous:
                    continue
                previous[g] = p
                if right[g] == -1:
                    endpoint = g
                    break
                paired = right[g]
                if paired not in visited_left:
                    visited_left.add(paired)
                    queue.append(paired)
        if endpoint is None:
            continue
        g = endpoint
        while True:
            p = previous[g]
            old_g = left.get(p)
            right[g] = p
            left[p] = g
            if old_g is None:
                break
            g = old_g
    return left


def prf(tp, fp, fn):
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def score_events(predictions, gt):
    valid = [g for g in gt if g["valid"]]
    ignored = [g for g in gt if not g["valid"]]
    # Valid matching takes priority. Any remaining prediction close to an
    # ignored point is neutral under the explicitly documented local policy.
    near_ignore = [any(matches(p, g) for g in ignored) for p in predictions]
    adjacency = []
    for p in predictions:
        neighbors = [i for i, g in enumerate(valid) if matches(p, g)]
        neighbors.sort(key=lambda i: (abs(p["frame_id"] - valid[i]["frame_id"]),
                                     math.hypot(p["x"] - valid[i]["x"], p["y"] - valid[i]["y"]),
                                     valid[i]["frame_id"], valid[i]["x"], valid[i]["y"]))
        adjacency.append(neighbors)
    # Among maximum-TP matchings, prefer using predictions that cannot be
    # ignored. This makes ambiguous valid/ignore overlaps order independent.
    order = sorted(range(len(predictions)), key=lambda i: (
        near_ignore[i], predictions[i]["frame_id"], predictions[i]["x"], predictions[i]["y"]))
    paired = maximum_matching(adjacency, len(valid), order)
    ignored_predictions = [i for i in range(len(predictions)) if i not in paired and near_ignore[i]]
    fp_indices = [i for i in range(len(predictions)) if i not in paired and not near_ignore[i]]
    fn_indices = [i for i in range(len(valid)) if i not in paired.values()]
    tp = len(paired)
    result = prf(tp, len(fp_indices), len(fn_indices))
    result.update({
        "predictions": len(predictions), "valid_gt": len(valid), "ignored_gt": len(ignored),
        "ignored_predictions": len(ignored_predictions),
        "matches": [dict(
            prediction_line=predictions[p].get("_line", p + 1), gt_line=valid[g].get("_line", g + 1),
            prediction_frame=predictions[p]["frame_id"], gt_frame=valid[g]["frame_id"],
            frame_error=abs(predictions[p]["frame_id"] - valid[g]["frame_id"]),
            position_error_px=math.hypot(predictions[p]["x"] - valid[g]["x"], predictions[p]["y"] - valid[g]["y"]))
            for p, g in sorted(paired.items())],
        "fp_prediction_lines": [predictions[i].get("_line", i + 1) for i in fp_indices],
        "fn_gt_lines": [valid[i].get("_line", i + 1) for i in fn_indices],
        "ignored_prediction_lines": [predictions[i].get("_line", i + 1) for i in ignored_predictions],
    })
    return result


def performance_points(fps):
    if not finite_number(fps) or fps < 1:
        return None  # The supplied rules do not define points below 1 FPS.
    return min(25.0, 1.0 + (fps - 1.0) * 24.0 / 49.0)


def load_events(path, by_id, *, ground_truth=False):
    result = defaultdict(list)
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(rec, dict):
            raise ValueError(f"{path}:{lineno}: record must be an object")
        required = ("video_id", "frame_id", "x", "y", "valid") if ground_truth else ("video_id", "frame_id", "x", "y")
        if any(key not in rec for key in required):
            raise ValueError(f"{path}:{lineno}: missing required fields: {required}")
        vid = rec["video_id"]
        if not isinstance(vid, str) or vid not in by_id:
            raise ValueError(f"{path}:{lineno}: unknown video_id {vid!r}")
        meta = by_id[vid]
        fid = rec["frame_id"]
        if not isinstance(fid, int) or isinstance(fid, bool) or not 0 <= fid < meta["n_frames"]:
            raise ValueError(f"{path}:{lineno}: frame_id outside [0, {meta['n_frames']})")
        for axis, limit in (("x", meta["width"]), ("y", meta["height"])):
            if not finite_number(rec[axis]) or not 0 <= rec[axis] < limit:
                raise ValueError(f"{path}:{lineno}: {axis} must be finite and inside [0, {limit})")
        if ground_truth and not isinstance(rec["valid"], bool):
            raise ValueError(f"{path}:{lineno}: GT valid must be an explicit boolean")
        result[vid].append({**rec, "_line": lineno})
    return result


def near_duplicate_count(events):
    return sum(any(matches(pred, earlier) for earlier in events[:i]) for i, pred in enumerate(events))


def runtime_report(status, manifest, prediction_count):
    warnings = []
    videos = status.get("videos", [])
    by_id = {v["video_id"]: v for v in videos}
    expected_ids = {v["video_id"] for v in manifest}
    observed_ids = set(by_id)
    complete = status.get("status") == "ok"
    if len(by_id) != len(videos):
        warnings.append("Duplicate video IDs in run_status.")
        complete = False
    if expected_ids != observed_ids:
        warnings.append("run_status video IDs differ from the scoring manifest.")
        complete = False
    for meta in manifest:
        r = by_id.get(meta["video_id"], {})
        if r.get("status") != "ok" or r.get("frames") != meta["n_frames"]:
            warnings.append(f"{meta['video_id']}: status/frame count does not confirm a complete video.")
            complete = False
    expected_frames = sum(v["n_frames"] for v in manifest)
    processed_frames = sum(v.get("frames", 0) for v in videos if v.get("status") == "ok")
    if status.get("totals", {}).get("predictions") != prediction_count:
        warnings.append("run_status totals.predictions differs from the predictions file; confirm file provenance.")
        complete = False
    timing = status.get("timing", {})
    perf = status.get("performance", {})
    infer_fps = perf.get("infer_fps")
    infer_sec = timing.get("infer_sec")
    total_sec = timing.get("total_sec")
    if not finite_number(infer_fps) or infer_fps < 0:
        infer_fps = None
        warnings.append("Official-framework performance.infer_fps is missing/invalid; no primary performance points.")
    if not finite_number(total_sec) or total_sec <= 0:
        total_sec = None
        warnings.append("Official-framework timing.total_sec is missing/invalid; completion deadline is unverified.")
    deadline_pass = total_sec <= 7200 if total_sec is not None else None
    if deadline_pass is False:
        warnings.append("Total wall time exceeds the 7200-second task limit.")
    e2e_fps = processed_frames / total_sec if total_sec else None
    recomputed_infer_fps = processed_frames / infer_sec if finite_number(infer_sec) and infer_sec > 0 else None
    if infer_fps is not None and infer_fps < 1:
        warnings.append("Rules do not specify points below 1 FPS; performance/total points are null.")
    points = performance_points(infer_fps)
    return {
        "source_status": status.get("status"), "complete_run_verified": complete,
        "expected_frames": expected_frames, "processed_frames": processed_frames,
        "prepare_sec": timing.get("prepare_sec"), "engine_build_sec": timing.get("engine_build_sec"),
        "infer_sec": infer_sec, "total_sec": total_sec, "within_7200_seconds": deadline_pass,
        "primary_fps": infer_fps, "primary_fps_source": "run_status.performance.infer_fps",
        "primary_performance_points": points,
        "recomputed_infer_fps_from_rounded_seconds": recomputed_infer_fps,
        "end_to_end_fps_including_prepare": e2e_fps,
        "conservative_end_to_end_performance_points": performance_points(e2e_fps),
        "total_score_eligible": complete and deadline_pass is True and points is not None,
        "warnings": warnings,
    }


def file_provenance(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def evaluate(manifest_path, gt_path, predictions_path, run_status_path):
    doc = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest = doc["videos"]
    if not manifest:
        raise ValueError("manifest contains no videos")
    by_id = {v["video_id"]: v for v in manifest}
    if len(by_id) != len(manifest):
        raise ValueError("duplicate video IDs in manifest")
    if any(v["angle"] not in ANGLES for v in manifest):
        raise ValueError("manifest angles must be exactly 0度, 45度, or 90度")
    gt = load_events(gt_path, by_id, ground_truth=True)
    predictions = load_events(predictions_path, by_id)
    warnings = []
    rows = []
    for meta in manifest:
        vid = meta["video_id"]
        result = score_events(predictions[vid], gt[vid])
        expected = meta.get("n_valid_landings")
        if expected is not None and expected != result["valid_gt"]:
            raise ValueError(f"{vid}: manifest n_valid_landings differs from GT valid=true count")
        rows.append({"video_id": vid, "angle": meta["angle"], "n_frames": meta["n_frames"],
                     "near_duplicate_predictions": near_duplicate_count(predictions[vid]), **result})
    angles = []
    for angle in ANGLES:
        subset = [v for v in rows if v["angle"] == angle]
        counts = {key: sum(v[key] for v in subset) for key in (
            "tp", "fp", "fn", "predictions", "valid_gt", "ignored_gt", "ignored_predictions", "n_frames")}
        counts.update(prf(counts["tp"], counts["fp"], counts["fn"]))
        angles.append({"angle": angle, "n_videos": len(subset), **counts,
                       "accuracy_points": counts["f1"] * 25 if subset else None})
    full_angle_coverage = all(a["n_videos"] > 0 for a in angles)
    if not full_angle_coverage:
        warnings.append("Missing a required angle: no 75-point accuracy total is reported for a subset.")
    accuracy = sum(a["accuracy_points"] for a in angles) if full_angle_coverage else None
    aggregate = {key: sum(v[key] for v in rows) for key in (
        "tp", "fp", "fn", "predictions", "valid_gt", "ignored_gt", "ignored_predictions")}
    aggregate.update(prf(aggregate["tp"], aggregate["fp"], aggregate["fn"]))
    runtime = runtime_report(json.loads(Path(run_status_path).read_text(encoding="utf-8")), manifest, aggregate["predictions"])
    can_total = accuracy is not None and runtime["total_score_eligible"]
    conservative = runtime["conservative_end_to_end_performance_points"]
    return {
        "schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "Public validation set, independent local reproduction; not the hidden leaderboard.",
        "dataset_name": doc.get("name"), "n_videos": len(manifest),
        "rules": {
            "frame_tolerance_inclusive": FRAME_TOLERANCE, "pixel_tolerance_euclidean_inclusive": POSITION_TOLERANCE,
            "matching": "Maximum-cardinality one-to-one matching to valid=true GT, per video.",
            "angle_aggregation": "Pool TP/FP/FN within each angle, then sum 25 * F1_angle across 0/45/90.",
            "ignore_policy_local_assumption": "After valid-GT matching, any unmatched prediction within ±1 frame and 50 px of any valid=false GT is ignored. Ignore regions may absorb multiple predictions. Prefer non-ignore predictions among maximum-TP solutions.",
            "ignore_policy_limitation": "The public framework provides no official scorer or exact ignored-region matching implementation. This is a documented local assumption.",
            "performance_formula": "1 + (fps - 1) * 24 / 49 for 1 <= fps <= 50; 25 for fps > 50; undefined below 1.",
            "primary_timing": "Official framework performance.infer_fps; excludes prepare, includes decoding, solution, writing and runner overhead.",
            "deadline_timing": "Official framework timing.total_sec, including prepare and engine build; <= 7200 seconds.",
            "rounding": "Do not round F1 or intermediate scores; input FPS is already rounded by the official runner.",
        },
        "inputs": {name: file_provenance(path) for name, path in (
            ("manifest", manifest_path), ("ground_truth", gt_path), ("predictions", predictions_path), ("run_status", run_status_path))},
        "aggregate_event_metrics": aggregate, "per_angle": angles, "per_video": rows,
        "performance": runtime,
        "scores": {
            "accuracy_out_of_75": accuracy,
            "performance_out_of_25": runtime["primary_performance_points"] if runtime["total_score_eligible"] else None,
            "total_out_of_100": accuracy + runtime["primary_performance_points"] if can_total else None,
            "conservative_total_including_prepare": accuracy + conservative if can_total and conservative is not None else None,
        },
        "warnings": warnings + runtime["warnings"],
    }


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--run-status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = evaluate(args.manifest, args.gt, args.predictions, args.run_status)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Scoring input error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["angle", "n_videos", "n_frames", "valid_gt", "ignored_gt", "predictions", "ignored_predictions", "tp", "fp", "fn", "precision", "recall", "f1", "accuracy_points"]
    write_csv(args.output.with_name(args.output.stem + "_per_angle.csv"), result["per_angle"], fields)
    fields = ["video_id", "angle", "n_frames", "valid_gt", "ignored_gt", "predictions", "ignored_predictions", "near_duplicate_predictions", "tp", "fp", "fn", "precision", "recall", "f1"]
    write_csv(args.output.with_name(args.output.stem + "_per_video.csv"), result["per_video"], fields)
    print(json.dumps({"output": str(args.output), "scores": result["scores"], "warnings": result["warnings"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
