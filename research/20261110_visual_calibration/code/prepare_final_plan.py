"""Prepare frozen final154 visual metadata and the missing33 crop requests; no fits or decoding."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np

from prepare_nested_plan import (EXP, ROOT, OLD, DATASET, ALLOW, MODEL, TRAINER,
    MODEL_SHA, TRAINER_SHA, CALIBRATION, read, lines, sha, save, describe, validate_probabilities)


FINAL = "experiments/20261109_final154_refit/runs/inputs_v001"
NATIVE33 = "experiments/20261107_confirmation33_fixed_comparison/runs/data_v001/input/native33/native_index.json"
PROTECTED = "experiments/20261102_research_foundation/data/protected_videos.json"
FRAME_HELPER = "experiments/20261109_contact_visual_evidence/code/build_rgb_inputs.py"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def artifact(path):
    return dict(name=path.name, sha256=sha(path), bytes=path.stat().st_size)


def prepare(args):
    root, output = args.assets_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh final154 metadata directory")
    dev, former = (set(read(root / p)["video_ids"]) for p in (ALLOW, PROTECTED))
    if len(dev) != 121 or len(former) != 33 or dev & former:
        raise ValueError("The original121/former33 membership differs")
    final_path = root / FINAL
    final_manifest = read(final_path / "manifest.json")
    dense_manifest = read(final_path / "dense/manifest.json")
    if (set(final_manifest["training_videos"]) != dev | former
            or dense_manifest["training_videos"] != 154
            or sha(final_path / "dense/manifest.json") != final_manifest["fits"]["dense"]["manifest_sha256"]):
        raise ValueError("The sealed final154 dense dataset differs")
    for name in ("index.jsonl", "supervision.json"):
        if sha(final_path / "dense" / name) != dense_manifest["output_sha256"][name]:
            raise ValueError("Final154 dense input changed: " + name)
    original_rows = lines(root / OLD / "runs/assets_v001/rows.jsonl")
    old_time = np.load(root / OLD / "runs/assets_v001/time_features.npy", allow_pickle=False)
    original_index = [r for r in lines(root / DATASET / "index.jsonl") if r["view"] == "augmented"]
    index = sorted((r for r in lines(final_path / "dense/index.jsonl") if r["view"] == "augmented"),
                   key=lambda r: r["row_index"])
    if (len(index) != 20641 or len(original_rows) != 17559 or index[:17559] != original_index
            or {r["video_id"] for r in index[17559:]} != former):
        raise ValueError("Expected an identical17559-row prefix and3082 former33 candidates")
    cache = read(args.cache_receipt)
    outer_status_path = args.outer_probabilities.parent / "run_status.json"
    outer_status = read(outer_status_path)
    if (not outer_status["completed"] or outer_status["training_runs"] != 10
            or outer_status["code_sha256"]["train_adapted_visual.py"] != TRAINER_SHA
            or outer_status["code_sha256"]["model.py"] != MODEL_SHA
            or outer_status["predictions_sha256"] != sha(args.outer_probabilities)
            or outer_status["input_sha256"]["receipt.json"] != sha(args.cache_receipt)
            or cache["model_source_sha256"] != MODEL_SHA or not cache["complete"]
            or any(outer_status[k] for k in ("protected33_used", "public15_used", "external130_used"))):
        raise ValueError("Original121 OOF or stem identity differs")
    for name in ("rows.jsonl", "time_features.npy"):
        if cache["files"][name]["sha256"] != sha(root / OLD / "runs/assets_v001" / name):
            raise ValueError("Original121 row/time metadata changed")
    validate_probabilities(read(args.outer_probabilities), original_rows, dev)
    helper = module("frozen_visual_frame_selection", root / FRAME_HELPER)
    clips = {c["clip_id"]: c for c in read(root / NATIVE33)["clips"]}
    if set(clips) != former:
        raise ValueError("Former33 native metadata differs")
    rows = list(original_rows)
    temporal = np.zeros((len(index), 3, 2), dtype=np.float32)
    temporal[:17559] = old_time
    requests = defaultdict(lambda: defaultdict(list))
    keys = ("row_index", "candidate_id", "physical_candidate_id", "video_id", "native_frame", "native_pts_seconds")
    for array_index, candidate in enumerate(index[17559:], 17559):
        clip = clips[candidate["video_id"]]
        pts, center = clip["pts_seconds"], candidate["native_frame"]
        x, y = (candidate["center_observation"][key] for key in ("x", "y"))
        if ((candidate["width"], candidate["height"]) != (clip["width"], clip["height"])
                or not 0 <= center < len(pts) or abs(pts[center] - candidate["native_pts_seconds"]) > helper.EPS
                or not np.isfinite([x, y]).all() or not 0 <= x < clip["width"] or not 0 <= y < clip["height"]):
            raise ValueError("Former33 candidate clock/center differs")
        frame_ids = helper.frame_ids(pts, center)
        rows.append(dict(array_index=array_index, frame_ids=frame_ids, **{key: candidate[key] for key in keys}))
        for slot, frame_id in enumerate(frame_ids):
            if frame_id is not None:
                temporal[array_index, slot] = (pts[frame_id] - pts[center], 1)
                requests[clip["clip_id"]][str(frame_id)].append(dict(array_index=array_index, slot=slot, x=x, y=y))
    if len({r["physical_candidate_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate final154 physical candidate")
    supervision = read(final_path / "dense/supervision.json")
    old_supervision = read(root / DATASET / "supervision.json")
    bags = [b for b in supervision["bags"] if b["view"] == "augmented"]
    # Adding former33 bags renumbers the storage-only bag_index. Event identity,
    # class, intervals and actual candidate members must remain identical.
    semantic = lambda bag: {key: value for key, value in bag.items() if key != "bag_index"}
    if ([semantic(b) for b in bags if b["video_id"] in dev]
            != [semantic(b) for b in old_supervision["bags"] if b["view"] == "augmented"]):
        raise ValueError("Original121 event supervision changed in final154")
    lookup = {r["row_index"]: r["array_index"] for r in rows}
    events = []
    for bag in bags:
        members = [lookup[i] for i in bag["indices"]]
        if not members or any(rows[i]["video_id"] != bag["video_id"] for i in members):
            raise ValueError("Final154 event crosses video boundary")
        events.append(dict(event_id=bag["event_id"], video_id=bag["video_id"], class_id=bag["class_id"], indices=members))
    backgrounds = [lookup[i] for i in supervision["background_indices"] if i in lookup]
    if (Counter(b["class_id"] for b in events) != {0: 729, 1: 642} or len(backgrounds) != 16275
            or set(backgrounds) & {i for e in events for i in e["indices"]}):
        raise ValueError("Final154 supervision counts or known background membership differs")
    old_bg = [lookup[i] for i in old_supervision["background_indices"] if i in lookup]
    if [i for i in backgrounds if rows[i]["video_id"] in dev] != old_bg:
        raise ValueError("Original121 legal background changed")
    output.mkdir(parents=True)
    with (output / "rows.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    np.save(output / "time_features.npy", temporal, allow_pickle=False)
    save(output / "supervision.json", dict(schema="final154_visual_supervision.v1",
        classes=supervision["classes"], events=events, background_indices=backgrounds))
    save(output / "additional33_requests.json", dict(schema="final154_additional33_requests.v1",
        clips=[dict(clips[vid], requests=dict(requests[vid])) for vid in sorted(former)]))
    dependency_paths = dict(old_trainer=TRAINER, old_model=MODEL,
        old_extractor=OLD + "/code/extract_stem_features.py", old_prepare=OLD + "/code/prepare_visual_assets.py",
        frame_helper=FRAME_HELPER, final_manifest=FINAL + "/manifest.json",
        final_dense_manifest=FINAL + "/dense/manifest.json", final_index=FINAL + "/dense/index.jsonl",
        final_supervision=FINAL + "/dense/supervision.json", native33=NATIVE33,
        development_allowlist=ALLOW, former33_allowlist=PROTECTED)
    sources = {key: describe(root, path) for key, path in dependency_paths.items()}
    if sources["old_trainer"]["sha256"] != TRAINER_SHA or sources["old_model"]["sha256"] != MODEL_SHA:
        raise ValueError("Original visual implementation changed")
    counts = dict(videos=154, rows=20641, base121_rows=17559, additional33_rows=3082,
        event_bags=1371, event_classes={"0": 729, "1": 642}, backgrounds=16275,
        videos_with_candidates=len({r["video_id"] for r in rows}),
        videos_without_candidates=sorted((dev | former) - {r["video_id"] for r in rows}),
        additional33_native_frames=sum(c["frame_count"] for c in clips.values()),
        additional33_distinct_requested_frames=sum(len(r) for r in requests.values()),
        additional33_valid_slots_by_time=temporal[17559:, :, 1].sum(0).astype(int).tolist(),
        visual_training_optimizer_steps=16 * ((1371 + 47) // 48))
    plan = dict(schema="final154_visual_plan.v1", created_utc=datetime.now(timezone.utc).isoformat(),
        inputs={key: artifact(output / name) for key, name in dict(rows="rows.jsonl", time_features="time_features.npy",
            supervision="supervision.json", additional33_requests="additional33_requests.json").items()},
        counts=counts, training_videos=sorted(dev | former), development_videos=sorted(dev), former33_videos=sorted(former),
        sources=sources, protocol_sha256=sha(EXP / "final_protocol_v001.json"),
        preparation_source_sha256=sha(__file__), training_source_sha256=sha(EXP / "code/train_final_visual.py"),
        nested_training_source_sha256=sha(EXP / "code/train_nested_visual.py"),
        nested_preparation_source_sha256=sha(EXP / "code/prepare_nested_plan.py"),
        base121_cache=dict(receipt_sha256=sha(args.cache_receipt), files=cache["files"], rows=17559,
            model_source_sha256=MODEL_SHA, pretrained_sha256=cache["pretrained_sha256"]),
        model=dict(source=sources["old_model"], pretrained_sha256=cache["pretrained_sha256"],
            RGB_SHAPE=[3, 2, 3, 96, 96], STEM_SHAPE=[3, 2, 96, 3, 3], training=outer_status["training"]),
        calibration=dict(configuration=CALIBRATION, fitting_videos=121, fitting_source_groups=10,
            input="original121 outer OOF raw probabilities only", outer_probabilities_sha256=sha(args.outer_probabilities),
            outer_run_status_sha256=sha(outer_status_path), event_bags=1168, backgrounds=13924,
            distribution_transfer="OOF models trained on9 development sources; applied to one full154 visual model; no assumed invariance"),
        scope=dict(visual_training_videos=154, calibration_training_videos=121,
            former33_visual_training=True, former33_calibration=False, public15_used=False,
            external130_used=False, actual_visual_training_runs=0, actual_calibration_fits=0))
    save(output / "plan.json", plan)
    receipt = dict(completed=True, plan_sha256=sha(output / "plan.json"), counts=counts,
        actual_visual_training_runs=0, actual_calibration_fits=0, new_video_decodes=0,
        original121_candidate_prefix_exact=True, original121_supervision_semantics_exact=True,
        storage_only_bag_index_renumbering_ignored=True,
        former33_used_for_metadata=True, former33_used_for_calibration=False)
    save(output / "run_status.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ROOT)
    parser.add_argument("--outer-probabilities", type=Path, required=True)
    parser.add_argument("--cache-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
