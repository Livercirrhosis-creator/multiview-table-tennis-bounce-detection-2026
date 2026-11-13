"""Extend the frozen gap inventory and run its original deployment fit."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

from check_inputs import CAPTURE, EXP, GAP, MANIFEST, NATIVE33, POLICY, ROOT, asset, ids, native, read


ACTIVATION = EXP / "runs/activation_v001.json"
DATASET = EXP / "runs/gap_inputs_v001"
DEPLOYMENT = EXP / "runs/gap_fit_v001"
ORIGINAL = GAP / "runs/dataset_v001"
CONFIRMATION = ROOT / "experiments/20261107_confirmation33_fixed_comparison"
TIMELINE_SHA = "daaacad9dde9ef56175794e901e1df06515df4bcfc83f8bdf0a3d01d51c30c6e"


sys.path.insert(0, str(GAP / "code"))
import build_dataset as original

sys.path.insert(0, str(original.INNER))
from gap_contact_v001.proposal import CONFIG, propose
from masked_gap_v001.features import FEATURE_NAMES, extract


def source_refs():
    inventory = read(ORIGINAL / "inventory.json")
    training = read(GAP / "protocol_v001.json")
    paths = [Path(original.__file__).resolve(), original.HELPER, original.SCORER, original.INNER / "gap_contact_v001/proposal.py",
             original.INNER / "masked_gap_v001/features.py"]
    fixed = [asset(path, inventory["source_sha256"][path.relative_to(ROOT).as_posix()]) for path in paths]
    trainers = [GAP / "code/train_evaluate.py", original.INNER / "masked_gap_v001/model.py"]
    return [asset(Path(__file__).resolve()), *fixed, *(asset(path, training["source_sha256"][path.relative_to(ROOT).as_posix()])
                                                      for path in trainers)]


def development_data():
    allowed = ids(POLICY / "development_videos.json", "development", 121)
    inventory = read(ORIGINAL / "inventory.json")
    for name in ("index.jsonl", "features.npy"):
        asset(ORIGINAL / name, inventory["output_sha256"][name])
    with (ORIGINAL / "index.jsonl").open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    features = np.load(ORIGINAL / "features.npy", allow_pickle=False)
    validate_rows(rows, features, allowed)
    return allowed, rows, features


def validate_rows(rows, features, allowed):
    if features.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError("Gap features differ from the frozen schema")
    for i, row in enumerate(rows):
        if row["row_index"] != i or row["video_id"] not in allowed or row["label"] not in original.CLASS_NAMES:
            raise ValueError("Unexpected gap row index, clip or class")


def shared_helpers():
    return original.load_module("refit_gap_supervision", original.HELPER), original.load_module(
        "refit_gap_timeline_validation", original.SCORER)


def candidates(video, clip, obs, group, helper, attempt_offset):
    """Apply the original natural/masked proposal policy to a single clip."""
    vid = video["clip_id"]
    pts = np.asarray(video["pts_seconds"], dtype=np.float64)
    reliable = original.reliable_frames(video, obs, CONFIG)
    reliable_times = pts[reliable]
    real_obs = sorted((row for row in obs if not row["center_is_padding"] and row["native_frame"] is not None),
                      key=lambda row: row["native_frame"])
    obs_times = np.asarray([row["native_pts_seconds"] for row in real_obs])
    contacts = [event for event in clip["events"] if event["kind"] in helper.CONTACT_KINDS]
    _, _, _, legal, _ = helper.supervision_for_video(clip, [])
    seeds = [dict(seed_id=e["event_id"], seed_event_id=e["event_id"], seed_kind=e["kind"],
                  center_seconds=sum(e["pts_interval_seconds"]) / 2.)
             for e in sorted(legal, key=lambda e: (e["pts_interval_seconds"], e["event_id"]))]
    seeds.extend(dict(seed_id=f"{vid}:background_seed:{k}", seed_event_id=None, seed_kind="background",
                      center_seconds=float(pts[0] + k * original.BG_CELL))
                 for k in range(int(math.floor((pts[-1] - pts[0]) / original.BG_CELL + original.EPS)) + 1))

    rows, features, supports = [], [], set()

    def append(candidate, gap, trajectory, mask):
        rows.append(dict(row_index=len(rows), video_id=vid, source_group=group,
                         origin="natural" if mask is None else "masked", candidate=candidate, gap=gap, mask=mask))
        features.append(extract(video, trajectory, candidate, gap))
        supports.add((tuple(gap["left_support_frames"]), tuple(gap["right_support_frames"])))

    natural = propose(video, obs)
    gaps = {(g["left_frame"], g["right_frame"]): g for g in natural["gaps"]}
    for candidate in natural["candidates"]:
        append(candidate, gaps[(candidate["left_frame"], candidate["right_frame"])], obs, None)

    cache = {}
    eps = original.EPS
    for seed in seeds:
        for width in original.WIDTHS:
            attempt = attempt_offset
            attempt_offset += 1
            low, high = seed["center_seconds"] - width / 2., seed["center_seconds"] + width / 2.
            if original.mask_rejection(video, clip, low, high, seed["seed_event_id"], contacts):
                continue
            deleted = reliable[(reliable_times >= low - eps) & (reliable_times <= high + eps)].tolist()
            if not deleted:
                continue
            key = tuple(deleted)
            if key not in cache:
                first = int(np.searchsorted(obs_times, low - original.CONTEXT - eps))
                stop = int(np.searchsorted(obs_times, high + original.CONTEXT + eps, side="right"))
                trajectory = [row for row in real_obs[first:stop] if not low - eps <= row["native_pts_seconds"] <= high + eps]
                cache[key] = (propose(video, trajectory), trajectory)
            result, trajectory = cache[key]
            chosen, _ = original.affected_candidates(result, low, high, deleted)
            for candidate, gap in chosen:
                support = (tuple(gap["left_support_frames"]), tuple(gap["right_support_frames"]))
                if support in supports:
                    continue
                append(candidate, gap, trajectory,
                       dict(attempt_index=attempt, **seed, width_seconds=width, interval_seconds=[low, high],
                            deleted_reliable_frames=deleted))

    index = [dict(row_index=row["row_index"], native_frame=row["candidate"]["frame_id"]) for row in rows]
    bags, background, excluded, _, _ = helper.supervision_for_video(clip, index)
    for bag in bags:
        for i in bag["indices"]:
            rows[i].update(label=bag["class_id"], event_id=bag["event_id"], group_id=f"{vid}:event:{bag['event_id']}")
    for i in background:
        cell = int(math.floor((rows[i]["candidate"]["pts_seconds"] - pts[0]) / original.BG_CELL + eps))
        rows[i].update(label=2, event_id=None, group_id=f"{vid}:background_cell:{cell}")
    for row in excluded:
        i = row["row_index"]
        rows[i].update(label=-1, event_id=None, group_id=f"{vid}:excluded:{i}",
                       exclusion_reason=row["reason"], excluded_event_ids=row["event_ids"])
    return rows, np.asarray(features, dtype=np.float64).reshape(-1, len(FEATURE_NAMES)), attempt_offset


def globalize(row, i):
    row["row_index"] = i
    if row["label"] == -1:
        row["group_id"] = f"{row['video_id']}:excluded:{i}"


def check():
    allowed, rows, x = development_data()
    helper, validator = shared_helpers()
    videos = {v["clip_id"]: v for v in read(original.SOURCE / "runtime/videos.json")["clips"]}
    groups = read(original.SOURCE / "metadata/video_groups.json")
    timeline = validator.validate_timeline(read(original.SOURCE / "training/timeline.json"))
    obs = original.grouped(original.SOURCE / "runtime/observations.jsonl", allowed)
    if any(set(v) != allowed for v in (videos, groups, timeline, obs)):
        raise ValueError("Cannot replay outside the existing development scope")
    expected = defaultdict(list)
    for row in rows:
        expected[row["video_id"]].append(row)
    attempt = 0
    for vid in sorted(allowed):
        made, features, attempt = candidates(videos[vid], timeline[vid], obs[vid], groups[vid], helper, attempt)
        if len(made) != len(expected[vid]):
            raise ValueError(f"Original proposal count changed for {vid}")
        indices = [row["row_index"] for row in expected[vid]]
        for row, i in zip(made, indices):
            globalize(row, i)
        if made != expected[vid]:
            raise ValueError(f"Original candidate or supervision changed for {vid}")
        np.testing.assert_array_equal(features, x[indices])
    if attempt != read(ORIGINAL / "inventory.json")["counts"]["attempts"]:
        raise ValueError("Mask attempt count changed")
    print(f"Verified 121-only replay: {len(rows)} candidate/label rows and feature values, {attempt} masks, no fitting")


def activation():
    gate = read(ACTIVATION)
    if (gate.get("schema") != "final154_activation.v1" or gate.get("scope") != "final154_training"
            or gate.get("activated") is not True or gate.get("training_authorized") is not True
            or gate.get("chain") not in {"CHANNEL_TABLE", "VISUAL_SHORTLIST"}
            or not re.fullmatch(r"[0-9a-f]{64}", gate.get("readiness_sha256", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", gate.get("external_gate_sha256", ""))):
        raise ValueError("Final154 activation must be recorded before accessing retained supervision")
    asset(MANIFEST, gate["readiness_sha256"])
    cohort = read(MANIFEST)["cohort"]
    for record in (*cohort["membership"], *cohort["native"]):
        asset(ROOT / record["path"], record["sha256"])
    return gate


def prepare():
    gate = activation()
    if DATASET.exists():
        raise FileExistsError(DATASET)
    sources = source_refs()
    allowed, base, x = development_data()
    held_path = POLICY / "protected_videos.json"
    held = ids(held_path, "evaluation_only", 33)
    if allowed & held:
        raise ValueError("Expected 154 disjoint source members")
    native33 = native(NATIVE33, held)
    helper, validator = shared_helpers()
    timeline_path = CONFIRMATION / "runs/data_v001/evaluation/timeline.json"
    timeline_asset = asset(timeline_path, TIMELINE_SHA)
    timeline = validator.validate_timeline(read(timeline_path))
    if set(timeline) != held:
        raise ValueError("Retained timeline differs from activated membership")
    seal_path = CAPTURE / "confirmation33_v001/prediction_seal.json"
    sealed = read(MANIFEST)["held_capture"]["seal"]
    if asset(seal_path, sealed["sha256"]) != sealed:
        raise ValueError("Original capture seal changed")
    hashes = read(seal_path)["captures_sha256"]
    if set(hashes) != held:
        raise ValueError("Retained capture seal differs from activated membership")
    start = time.perf_counter()
    attempt = read(ORIGINAL / "inventory.json")["counts"]["attempts"]
    augmented, pieces = [], [x]
    offset = len(base)
    for vid in sorted(held):
        path = CAPTURE / "capture33_v001_merged" / f"{vid}.json"
        asset(path, hashes[vid])
        data = read(path)
        video = data["video"]
        native_clip, clip = native33[vid], timeline[vid]
        for k in ("clip_id", "frame_count", "pts_seconds", "media_sha256"):
            if video[k] != native_clip[k] or video[k] != clip[k]:
                raise ValueError(f"Capture and frozen timeline identity differ for {vid}/{k}")
        if (video["width"], video["height"]) != (native_clip["width"], native_clip["height"]):
            raise ValueError(f"Capture dimensions differ from the native index for {vid}")
        if clip["camera_family"] != native_clip["camera_family"]:
            raise ValueError(f"Source-family mapping differs for {vid}")
        video = dict(video, source_pts_gaps=native_clip["source_pts_gaps"])
        rows, features, attempt = candidates(video, clip, data["observations"], clip["camera_family"], helper, attempt)
        for row in rows:
            globalize(row, offset)
            offset += 1
        augmented.extend(rows)
        pieces.append(features)
    combined, features = base + augmented, np.concatenate(pieces, axis=0)
    validate_rows(combined, features, allowed | held)
    DATASET.mkdir(parents=True)
    with (DATASET / "index.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    np.save(DATASET / "features.npy", features, allow_pickle=False)
    original.save(DATASET / "manifest.json", dict(schema="final154_gap_inputs.v1", chain=gate["chain"],
         activation=asset(ACTIVATION), timeline=timeline_asset, source_files=sources,
         development_videos=121, appended_videos=33, existing_rows=len(base), appended_rows=len(augmented),
         retained_classes=original.summarize(augmented), elapsed_seconds=time.perf_counter() - start,
         inputs=[asset(DATASET / name) for name in ("index.jsonl", "features.npy")]))
    print(f"Prepared gap inventory: {len(base)} frozen development rows + {len(augmented)} retained-scope rows")


def fit():
    gate = activation()
    manifest = read(DATASET / "manifest.json")
    if (manifest["chain"] != gate["chain"] or manifest["activation"] != asset(ACTIVATION)
            or manifest["source_files"] != source_refs()):
        raise ValueError("Prepared fit scope or frozen source changed")
    for item in manifest["inputs"]:
        asset(ROOT / item["path"], item["sha256"])
    from train_evaluate import CONFIG as TRAINING, lines, train_fold
    import torch

    rows = lines(DATASET / "index.jsonl")
    features = np.load(DATASET / "features.npy", allow_pickle=False)
    allowed = ids(POLICY / "development_videos.json", "development", 121)
    held = ids(POLICY / "protected_videos.json", "evaluation_only", 33)
    validate_rows(rows, features, allowed | held)
    if len(allowed | held) != 154 or len(rows) != manifest["existing_rows"] + manifest["appended_rows"]:
        raise ValueError("Prepared gap fit has inconsistent membership")
    DEPLOYMENT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(TRAINING["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    fold = dict(fold_id=0, held_out_family="deployment_all154_no_evaluation",
                train_video_ids=sorted(allowed | held), held_out_video_ids=[])
    predictions, result = train_fold(features, rows, fold, DEPLOYMENT)
    if predictions:
        raise ValueError("Final fit unexpectedly produced evaluation predictions")
    original.save(DEPLOYMENT / "report.json", dict(schema="final154_gap_fit.v1", chain=gate["chain"],
         fit=result, elapsed_seconds=time.perf_counter() - start, training_runs=1, evaluation_runs=0,
         weights=asset(ROOT / result["weights_path"], result["weights_sha256"])))
    print(f"Fitted one original-recipe 154 gap model: {result['weights_path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--fit", action="store_true")
    args = parser.parse_args()
    if args.check:
        source_refs()
        check()
    elif args.prepare:
        prepare()
    else:
        fit()


if __name__ == "__main__":
    main()
