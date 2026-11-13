"""Build development-only natural and masked gap-contact classification inventory.

No model fitting, scoring, event admission, or visual forward pass occurs here.
Synthetic masks are supervised training views, never independent event samples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/20261107_masked_gap_decision"
INNER = EXP / "versions/v001/participant/pingpang/participant"
SOURCE = ROOT / "experiments/20261103_dual_path_rebuild/runs/reservoir_v001"
POLICY = ROOT / "experiments/20261102_research_foundation/data/development_videos.json"
NATURAL = ROOT / "experiments/20261107_gap_contact_prototype/runs/gap_prototype_v001/proposals.jsonl"
DENSE = ROOT / "experiments/20261105_coverage_aware_events/runs/dataset_v001/supervision.json"
HELPER = ROOT / "experiments/20261104_event_decision_rebuild/code/build_event_dataset.py"
SCORER = ROOT / "experiments/20261102_p1_time_selection/runs/coverage_mapping_v001/sealed_source/evaluation/evaluate_timeline.py"
OLD_PROPOSAL = ROOT / "experiments/20261107_gap_contact_prototype/versions/v001/participant/pingpang/participant/gap_contact_v001/proposal.py"
WIDTHS = (0.080, 0.160)
BG_CELL = 0.320
CONTEXT = 0.560
EPS = 1e-9
CLASS_NAMES = {-1: "excluded", 0: "table_contact", 1: "racket_hit", 2: "background"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grouped(path, allowed):
    result = defaultdict(list)
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["video_id"] not in allowed:
                    raise ValueError("Unexpected non-development video")
                result[row["video_id"]].append(row)
    return result


def reliable_frames(video, observations, config):
    result = []
    for row in observations:
        frame, detection = row["native_frame"], row["detection"]
        if row["center_is_padding"] or frame is None or detection is None:
            continue
        x, y, confidence = (float(detection[k]) for k in ("x", "y", "conf"))
        if (all(math.isfinite(v) for v in (x, y, confidence))
                and confidence >= config["reliable_confidence"]
                and 0 <= x < video["width"] and 0 <= y < video["height"]):
            result.append(frame)
    if len(result) != len(set(result)):
        raise ValueError("Duplicate reliable observation")
    return np.asarray(sorted(result), dtype=np.int64)


def mask_rejection(video, clip, low, high, seed_event_id, contacts):
    pts = np.asarray(video["pts_seconds"])
    if low < pts[0] - EPS or high > pts[-1] + EPS:
        return "mask_outside_video"
    first = int(np.searchsorted(pts, low - EPS))
    stop = int(np.searchsorted(pts, high + EPS, side="right"))
    if any(state != "evaluable" for state in clip["states"][first:stop]):
        return "mask_intersects_unknown_frame"
    for gap in clip.get("unknown_time_gaps", []):
        lo, hi = gap.get("pts_interval_seconds", gap.get("time_interval_seconds"))
        if low < hi - EPS and high > lo + EPS:
            return "mask_crosses_unknown_time_gap"
    for gap in video.get("source_pts_gaps", []):
        lo, hi = pts[gap["before_frame_id"]], pts[gap["after_frame_id"]]
        if low < hi - EPS and high > lo + EPS:
            return "mask_crosses_source_pts_gap"
    for event in contacts:
        if event["event_id"] == seed_event_id:
            continue
        lo, hi = event["pts_interval_seconds"]
        if low <= hi + EPS and high >= lo - EPS:
            return "mask_intersects_other_contact"
    return None


def affected_candidates(proposals, low, high, removed):
    selected, rejected = [], Counter()
    diagnostics = {(g["left_frame"], g["right_frame"]): g for g in proposals["gaps"]}
    for candidate in proposals["candidates"]:
        lf, rf = candidate["left_frame"], candidate["right_frame"]
        if not (lf < removed[0] <= removed[-1] < rf):
            continue
        if not (low - EPS <= candidate["tau_seconds"] <= high + EPS
                and low - EPS <= candidate["pts_seconds"] <= high + EPS):
            rejected["affected_contact_outside_mask"] += 1
            continue
        selected.append((candidate, diagnostics[(lf, rf)]))
    return selected, rejected


def summarize(rows):
    result = {}
    for label, name in CLASS_NAMES.items():
        selected = [r for r in rows if r["label"] == label]
        result[name] = dict(rows=len(selected),
                            independent_groups=len({r["group_id"] for r in selected}) if label >= 0 else 0,
                            event_ids=len({r["event_id"] for r in selected if r["event_id"] is not None}),
                            videos=len({r["video_id"] for r in selected}),
                            source_count=len({r["source_group"] for r in selected}))
    return result


def build(output):
    sys.path.insert(0, str(INNER))
    from gap_contact_v001.proposal import CONFIG, propose
    from masked_gap_v001.features import FEATURE_NAMES, extract

    helper = load_module("original_gap_supervision", HELPER)
    validator = load_module("gap_timeline_validation", SCORER)
    if helper.CLASSES != {"table_contact": 0, "racket_hit": 1, "background": 2}:
        raise ValueError("Original class mapping changed")
    if sha(INNER / "gap_contact_v001/proposal.py") != sha(OLD_PROPOSAL):
        raise ValueError("Gap proposal source differs from frozen prototype")
    paths = {name: SOURCE / name for name in ("runtime/videos.json", "runtime/observations.jsonl",
             "training/timeline.json", "metadata/video_groups.json", "splits/lofo_by_video.json")}
    allowed = set(read(POLICY)["video_ids"])
    videos = {v["clip_id"]: v for v in read(paths["runtime/videos.json"])["clips"]}
    groups = read(paths["metadata/video_groups.json"])
    timeline = read(paths["training/timeline.json"])
    if len(allowed) != 121 or set(videos) != allowed or set(groups) != allowed or {c["clip_id"] for c in timeline["clips"]} != allowed:
        raise ValueError("Data must equal development121")
    clips = validator.validate_timeline(timeline)
    observations = grouped(paths["runtime/observations.jsonl"], allowed)
    natural_by_video = grouped(NATURAL, allowed)
    if set(observations) != allowed or set(natural_by_video) != allowed or any(len(v) != 1 for v in natural_by_video.values()):
        raise ValueError("Missing or duplicated video")
    for vid in sorted(allowed):
        if any(videos[vid][k] != clips[vid][k] for k in ("frame_count", "pts_seconds", "media_sha256")):
            raise ValueError("Timeline and runtime media/PTS differ")
    dense = read(DENSE)
    legal_events = dense["bags"] + dense["unrepresented_events"]
    legal_by_video = defaultdict(list)
    for event in legal_events:
        if event["video_id"] not in allowed:
            raise ValueError("Dense event outside development121")
        legal_by_video[event["video_id"]].append(event)
    excluded_events = dense["excluded_events"]
    excluded_ids = {e["event_id"] for e in excluded_events}
    legal_ids = {e["event_id"] for e in legal_events}
    if len(legal_ids) != 1201 or len(excluded_ids) != 120 or legal_ids & excluded_ids:
        raise ValueError("Existing event eligibility differs")
    # Validate event eligibility through the unchanged helper, independently of candidates.
    for vid in sorted(allowed):
        bags, bg, ex, unrepresented, excluded = helper.supervision_for_video(clips[vid], [])
        if {e["event_id"] for e in unrepresented} != {e["event_id"] for e in legal_by_video[vid]}:
            raise ValueError("Dense event eligibility and original helper differ")
        if {e["event_id"] for e in excluded} != {e["event_id"] for e in excluded_events if e["video_id"] == vid}:
            raise ValueError("Original excluded event set changed")

    input_files = [POLICY, NATURAL, DENSE, *paths.values()]
    source_files = [Path(__file__), HELPER, SCORER, OLD_PROPOSAL,
                    INNER / "gap_contact_v001/proposal.py", INNER / "masked_gap_v001/features.py"]
    input_hashes = {rel(p): sha(p) for p in input_files}
    source_hashes = {rel(p): sha(p) for p in source_files}
    (output / "natural_proposals.jsonl").write_bytes(NATURAL.read_bytes())
    save(output / "build_plan.json", dict(
        schema="masked_gap_inventory.v1", mask_widths_seconds=WIDTHS,
        background_grid_seconds=BG_CELL, localized_context_seconds=CONTEXT,
        classes=helper.CLASSES, excluded_label=-1, feature_names=FEATURE_NAMES,
        event_center="midpoint of existing eligible event PTS interval; sampling only",
        background_center="video first PTS plus k*0.320 seconds",
        label_rule="Original supervision_for_video at actual inferred native frame",
        deduplication="Cache identical reliable deletions, then one retained row per exact local fit support",
        group_rule="Assigned event ID, or assigned frame time's video-local 320ms background cell",
        input_sha256=input_hashes, source_sha256=source_hashes,
        scope="development121 inventory only; held-out augmented samples never train or evaluate that fold"))

    index, features = [], []
    natural_gaps = {}
    for vid in sorted(allowed):
        record = natural_by_video[vid][0]
        diagnostics = {(g["left_frame"], g["right_frame"]): g for g in record["gaps"]}
        for candidate in record["candidates"]:
            gap = diagnostics[(candidate["left_frame"], candidate["right_frame"])]
            index.append(dict(row_index=len(index), video_id=vid, source_group=groups[vid],
                              origin="natural", candidate=candidate, gap=gap, mask=None))
            features.append(extract(videos[vid], observations[vid], candidate, gap))
            natural_gaps[(vid, candidate["left_frame"], candidate["right_frame"])] = gap
    if len(index) != 116:
        raise ValueError("Expected exact original 116 proposals")

    attempts, local_checks = [], []
    cache_hits, support_duplicates = 0, 0
    checked_strata = set()
    unique_support = set()
    for row in index:
        g = row["gap"]
        unique_support.add((row["video_id"], tuple(g["left_support_frames"]), tuple(g["right_support_frames"])))
    for number, vid in enumerate(sorted(allowed), 1):
        video, clip, obs = videos[vid], clips[vid], observations[vid]
        pts = np.asarray(video["pts_seconds"], dtype=np.float64)
        reliable = reliable_frames(video, obs, CONFIG)
        reliable_times = pts[reliable]
        real_obs = [r for r in obs if not r["center_is_padding"] and r["native_frame"] is not None]
        real_obs.sort(key=lambda r: r["native_frame"])
        obs_times = np.asarray([r["native_pts_seconds"] for r in real_obs])
        contacts = [e for e in clip["events"] if e["kind"] in helper.CONTACT_KINDS]
        seeds = [dict(seed_id=e["event_id"], seed_event_id=e["event_id"], seed_kind=e["kind"],
                      center_seconds=sum(e["pts_interval_seconds"]) / 2.)
                 for e in sorted(legal_by_video[vid], key=lambda e: (e["pts_interval_seconds"], e["event_id"]))]
        seeds += [dict(seed_id=f"{vid}:background_seed:{k}", seed_event_id=None,
                       seed_kind="background", center_seconds=float(pts[0] + k * BG_CELL))
                  for k in range(int(math.floor((pts[-1] - pts[0]) / BG_CELL + EPS)) + 1)]
        cached = {}
        for seed in seeds:
            for width in WIDTHS:
                low, high = seed["center_seconds"] - width / 2., seed["center_seconds"] + width / 2.
                attempt = dict(attempt_index=len(attempts), video_id=vid, source_group=groups[vid], **seed,
                               width_seconds=width, interval_seconds=[low, high])
                attempts.append(attempt)
                reason = mask_rejection(video, clip, low, high, seed["seed_event_id"], contacts)
                if reason:
                    attempt.update(status="rejected", reason=reason, retained_rows=[])
                    continue
                deleted = reliable[(reliable_times >= low - EPS) & (reliable_times <= high + EPS)].tolist()
                attempt["deleted_reliable_frames"] = deleted
                if not deleted:
                    attempt.update(status="rejected", reason="no_reliable_point_removed", retained_rows=[])
                    continue
                delete_key = tuple(deleted)
                if delete_key in cached:
                    cache_hits += 1
                    result, masked_obs = cached[delete_key]
                    attempt["reused_deletion_result"] = True
                else:
                    first = int(np.searchsorted(obs_times, low - CONTEXT - EPS))
                    stop = int(np.searchsorted(obs_times, high + CONTEXT + EPS, side="right"))
                    masked_obs = [r for r in real_obs[first:stop]
                                  if not low - EPS <= r["native_pts_seconds"] <= high + EPS]
                    result = propose(video, masked_obs)
                    cached[delete_key] = (result, masked_obs)
                    attempt["reused_deletion_result"] = False
                chosen, rejected = affected_candidates(result, low, high, deleted)
                affected_diagnostics = [g for g in result["gaps"]
                                        if g["left_frame"] < deleted[0] <= deleted[-1] < g["right_frame"]]
                attempt["affected_gap_reasons"] = dict(Counter(g["reason"] for g in affected_diagnostics))
                attempt["candidate_rejections"] = dict(rejected)
                # Check both successful and rejected proposals against the full observation list.
                stratum = (groups[vid], seed["seed_kind"], width, bool(chosen))
                if stratum not in checked_strata:
                    full_obs = [r for r in real_obs if not low - EPS <= r["native_pts_seconds"] <= high + EPS]
                    full_result = propose(video, full_obs)
                    full_chosen, full_rejected = affected_candidates(full_result, low, high, deleted)
                    if chosen != full_chosen or rejected != full_rejected:
                        raise ValueError(f"Localized and full proposal differ for {vid}/{seed['seed_id']}/{width}")
                    for (candidate, gap), (full_candidate, full_gap) in zip(chosen, full_chosen):
                        np.testing.assert_array_equal(extract(video, masked_obs, candidate, gap),
                                                      extract(video, full_obs, full_candidate, full_gap))
                    checked_strata.add(stratum)
                    local_checks.append(dict(attempt_index=attempt["attempt_index"], source_group=groups[vid],
                                             seed_kind=seed["seed_kind"], width_seconds=width,
                                             selected_candidates=len(chosen), exact_payload_and_features_equal=True))
                retained = []
                duplicates = 0
                for candidate, gap in chosen:
                    support_key = (vid, tuple(gap["left_support_frames"]), tuple(gap["right_support_frames"]))
                    if support_key in unique_support:
                        duplicates += 1
                        support_duplicates += 1
                        continue
                    unique_support.add(support_key)
                    mask = dict(attempt_index=attempt["attempt_index"], seed_id=seed["seed_id"],
                                seed_event_id=seed["seed_event_id"], seed_kind=seed["seed_kind"],
                                center_seconds=seed["center_seconds"], width_seconds=width,
                                interval_seconds=[low, high], deleted_reliable_frames=deleted)
                    row_index = len(index)
                    index.append(dict(row_index=row_index, video_id=vid, source_group=groups[vid],
                                      origin="masked", candidate=candidate, gap=gap, mask=mask))
                    features.append(extract(video, masked_obs, candidate, gap))
                    retained.append(row_index)
                attempt.update(retained_rows=retained, duplicate_support_candidates=duplicates)
                if retained:
                    attempt.update(status="success", reason="candidate_retained")
                elif duplicates:
                    attempt.update(status="deduplicated", reason="identical_effective_support")
                else:
                    attempt.update(status="rejected", reason="no_accepted_affected_contact_in_mask")
        if number % 20 == 0 or number == len(allowed):
            print(json.dumps(dict(processed_videos=number, attempts=len(attempts), rows=len(index)), ensure_ascii=False), flush=True)

    assigned_bags, background_rows, excluded_rows = [], [], []
    for vid in sorted(allowed):
        rows = [r for r in index if r["video_id"] == vid]
        helper_index = [dict(row_index=r["row_index"], native_frame=r["candidate"]["frame_id"]) for r in rows]
        bags, background, excluded, _, original_exclusions = helper.supervision_for_video(clips[vid], helper_index)
        for bag in bags:
            assigned_bags.append(bag)
            for row_index in bag["indices"]:
                index[row_index].update(label=bag["class_id"], event_id=bag["event_id"],
                                        group_id=f"{vid}:event:{bag['event_id']}")
        for row_index in background:
            row = index[row_index]
            cell = int(math.floor((row["candidate"]["pts_seconds"] - videos[vid]["pts_seconds"][0]) / BG_CELL + EPS))
            row.update(label=2, event_id=None, group_id=f"{vid}:background_cell:{cell}")
        for exclusion in excluded:
            row_index = exclusion["row_index"]
            index[row_index].update(label=-1, event_id=None, group_id=f"{vid}:excluded:{row_index}",
                                    exclusion_reason=exclusion["reason"], excluded_event_ids=exclusion["event_ids"])
        background_rows.extend(background)
        excluded_rows.extend(excluded)
    if Counter(r["label"] for r in index[:116]) != {-1: 38, 0: 13, 1: 41, 2: 24}:
        raise ValueError("Natural116 supervision unexpectedly changed")
    if any(r["event_id"] in excluded_ids for r in index if r["event_id"] is not None):
        raise ValueError("Excluded event was reinstated")
    for i, row in enumerate(index):
        if row["row_index"] != i or row["video_id"] not in allowed or row["label"] not in CLASS_NAMES:
            raise ValueError("Invalid inventory row")
    feature_matrix = np.asarray(features, dtype=np.float64)
    if feature_matrix.shape != (len(index), len(FEATURE_NAMES)) or not np.isfinite(feature_matrix).all():
        raise ValueError("Invalid feature inventory")
    np.save(output / "features.npy", feature_matrix, allow_pickle=False)
    for filename, rows in (("index.jsonl", index), ("augmentation_attempts.jsonl", attempts)):
        (output / filename).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    save(output / "supervision.json", dict(classes=helper.CLASSES, bags=assigned_bags,
         background_indices=background_rows, excluded_candidates=excluded_rows, excluded_events=excluded_events,
         eligible_seed_events=legal_events, label_assignment="Actual candidate native frame through unchanged helper",
         coverage_authority=timeline["coverage_authority"],
         semantics="Rows are augmented proposal-level class examples; group weight must total one per assigned event/background cell, not per row."))
    folds = read(paths["splits/lofo_by_video.json"])["folds"]
    held_seen = Counter()
    expanded = []
    for fold in folds:
        train, held = set(fold["train_video_ids"]), set(fold["held_out_video_ids"])
        family = fold["held_out_family"]
        if train & held or train | held != allowed or held != {v for v in allowed if groups[v] == family}:
            raise ValueError("Invalid source-separated fold")
        held_seen.update(held)
        expanded.append({**fold,
                         "train_rows": [r["row_index"] for r in index if r["video_id"] in train and r["label"] >= 0],
                         "held_out_natural_rows": [r["row_index"] for r in index[:116] if r["video_id"] in held]})
    if len(folds) != 10 or set(held_seen) != allowed or set(held_seen.values()) != {1}:
        raise ValueError("Expected one held-out occurrence per video")
    save(output / "folds.json", dict(folds=expanded, video_groups=groups,
         normalization="Fit only on eligible train_rows; no held-out natural or masked sample in any training statistic"))
    natural_rows, masked_rows = index[:116], index[116:]
    counts = dict(videos=121, source_groups=10, natural_rows=116, masked_rows=len(masked_rows), rows=len(index),
                  feature_dimensions=len(FEATURE_NAMES), classes=summarize(index), natural=summarize(natural_rows),
                  masked=summarize(masked_rows), attempts=len(attempts),
                  attempt_status=dict(Counter(a["status"] for a in attempts)),
                  attempt_reasons=dict(Counter(a["reason"] for a in attempts)),
                  deletion_result_cache_hits=cache_hits, duplicate_support_candidates=support_duplicates,
                  eligible_seed_events=len(legal_ids), original_excluded_events=len(excluded_ids),
                  local_vs_full_checks=len(local_checks), training_runs=0, model_inference_calls=0,
                  by_source={g: dict(all=summarize([r for r in index if r["source_group"] == g]),
                                     natural=summarize([r for r in natural_rows if r["source_group"] == g]),
                                     masked=summarize([r for r in masked_rows if r["source_group"] == g]))
                             for g in sorted(set(groups.values()))},
                  by_width={str(width): summarize([r for r in masked_rows if r["mask"]["width_seconds"] == width]) for width in WIDTHS},
                  seed_to_assigned_class={kind: dict(Counter(CLASS_NAMES[r["label"]] for r in masked_rows if r["mask"]["seed_kind"] == kind))
                                          for kind in ("table_contact", "racket_hit", "background")},
                  attempts_by_seed_kind_and_width={f"{kind}:{width}": dict(Counter(a["status"] for a in attempts if a["seed_kind"] == kind and a["width_seconds"] == width))
                                                   for kind in ("table_contact", "racket_hit", "background") for width in WIDTHS})
    save(output / "localization_checks.json", dict(checks=local_checks, exact_equal=True,
         selection="First mask per source/seed class/width/accepted-or-rejected affected proposal stratum; no class scores or TP status used"))
    report = dict(schema="masked_gap_dataset.v1", counts=counts, input_sha256=input_hashes,
                  source_sha256=source_hashes, feature_names=FEATURE_NAMES, protected33_used=False,
                  public15_used=False, mixed154_used=False, frozen_proposals_copy_sha256=sha(output / "natural_proposals.jsonl"),
                  limitations=["Training masks use known legal event midpoints and fixed background grid; this differs from naturally missing inference data.",
                               "Augmented copies and same-source videos are not additional independent events or sources.",
                               "Original source-declared annotation completeness is provisional, not independent expert certification.",
                               "Unknown natural candidates remain in inventory for honest held-out inference; label -1 is excluded from training.",
                               "No held-out augmentation may enter that fold's fitting, normalization, or evaluation."])
    report["output_sha256"] = {name: sha(output / name) for name in ("features.npy", "index.jsonl", "natural_proposals.jsonl", "supervision.json", "folds.json", "augmentation_attempts.jsonl", "localization_checks.json", "build_plan.json")}
    # Detect concurrent interface changes rather than silently mixing feature definitions.
    if input_hashes != {rel(p): sha(p) for p in input_files} or source_hashes != {rel(p): sha(p) for p in source_files}:
        raise ValueError("Input or source changed during inventory construction")
    save(output / "inventory.json", report)
    save(EXP / "analysis/dataset_v001.json", report)
    print(json.dumps({k: v for k, v in counts.items() if k not in ("by_source",)}, ensure_ascii=False), flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXP / "runs/dataset_v001")
    args = parser.parse_args()
    if args.output.resolve() != (EXP / "runs/dataset_v001").resolve():
        raise ValueError("This frozen builder owns only runs/dataset_v001")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
                  completed=False, training_runs=0, model_inference_calls=0)
    exit_code = 1
    with (args.output / "console.log").open("w", encoding="utf-8") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            try:
                counts = build(args.output)
                exit_code = 0
                status.update(completed=True, counts={k: counts[k] for k in ("rows", "natural_rows", "masked_rows", "attempts", "feature_dimensions")})
            except Exception:
                status["error"] = traceback.format_exc()
                traceback.print_exc()
            finally:
                status.update(exit_code=exit_code, finished_utc=datetime.now(timezone.utc).isoformat(),
                              wall_seconds=time.perf_counter() - started)
                save(args.output / "run_status.json", status)
                (args.output / "exit_code.txt").write_text(str(exit_code) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
