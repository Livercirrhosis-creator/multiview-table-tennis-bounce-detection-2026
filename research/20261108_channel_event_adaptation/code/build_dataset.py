"""Prepare paired original/channel-augmented views of the same development121.

This reuses the frozen interval-bag and complete-window label builders. It does
not fit a model, evaluate event predictions, read protected33, or run a detector.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
RUNTIME = EXP / "versions/v001/participant/pingpang"
VIEWS = ("original", "augmented")
PREVIOUS = ROOT / "experiments/20261107_pose_decode_channels"
DENSE = ROOT / "experiments/20261105_coverage_aware_events/runs/dataset_v001"
WINDOW = ROOT / "experiments/20261104_event_presence_location/runs/dataset_v001"
SPARSE = ROOT / "experiments/20261104_event_decision_rebuild/runs/dataset_v002"
RESERVOIR = ROOT / "experiments/20261103_dual_path_rebuild/runs/reservoir_v001"
POLICY = ROOT / "experiments/20261102_research_foundation/data/development_videos.json"
BAG_HELPER = ROOT / "experiments/20261104_event_decision_rebuild/code/build_event_dataset.py"
WINDOW_HELPER = ROOT / "experiments/20261104_event_presence_location/code/build_window_dataset.py"
REPLAY_HELPER = PREVIOUS / "code/channel_fallback_replay.py"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def event_view(record, view):
    result = dict(record, view=view)
    if "event_id" in result:
        result["physical_event_id"] = result["event_id"]
        result["event_id"] = f"{view}/{result['event_id']}"
    return result


def append_view(target, view, pool, sequence, summary, windows, video, family):
    """Translate a video's local runtime row indices into one paired dataset."""
    import numpy as np
    n, offset = len(pool), len(target["index"])
    assert np.asarray(sequence).shape == (n, 15, 23)
    assert np.asarray(summary).shape == (n, 44)
    assert np.isfinite(sequence).all() and np.isfinite(summary).all()
    assert len(windows) == n
    assert [w["center_row_index"] for w in windows] == list(range(n))
    local = []
    for number, candidate in enumerate(pool):
        frame = candidate["native_frame"]
        assert candidate["video_id"] == video["clip_id"]
        assert candidate["candidate_id"] == f"{video['clip_id']}:{frame}"
        assert candidate["native_pts_seconds"] == video["pts_seconds"][frame]
        row = dict(candidate, row_index=offset + number, camera_family=family,
                   physical_candidate_id=candidate["candidate_id"], view=view,
                   candidate_id=f"{view}/{candidate['candidate_id']}")
        local.append(row)
        target["index"].append(row)
    for w in windows:
        assert w["center_row_index"] in w["member_rows"]
        assert all(0 <= i < n for i in w["member_rows"])
        physical_id = f"{video['clip_id']}:w{w['center_frame']}"
        target["windows"].append(dict(w, window_index=offset + w["center_row_index"],
            window_id=f"{view}/{physical_id}", physical_window_id=physical_id, view=view,
            video_id=video["clip_id"], center_row_index=offset + w["center_row_index"],
            member_rows=[offset + i for i in w["member_rows"]]))
    target["sequence"].append(np.asarray(sequence, dtype=np.float64))
    target["summary"].append(np.asarray(summary, dtype=np.float64))
    return local


def dense_labels(target, clips, helper):
    supervision = dict(classes=helper.CLASSES, bags=[], background_indices=[],
        excluded_indices=[], excluded_candidates=[], unrepresented_events=[], excluded_events=[])
    grouped = defaultdict(list)
    for row in target["index"]:
        grouped[(row["video_id"], row["view"])].append(row)
    for vid in sorted(clips):
        for view in VIEWS:
            bags, background, excluded, unrepresented, excluded_events = helper.supervision_for_video(
                clips[vid], grouped[(vid, view)])
            for bag in bags:
                supervision["bags"].append(dict(event_view(bag, view), bag_index=len(supervision["bags"])))
            supervision["background_indices"].extend(background)
            supervision["excluded_candidates"].extend(dict(r, view=view, video_id=vid) for r in excluded)
            supervision["unrepresented_events"].extend(event_view(r, view) for r in unrepresented)
            supervision["excluded_events"].extend(event_view(r, view) for r in excluded_events)
    supervision["excluded_indices"] = [r["row_index"] for r in supervision["excluded_candidates"]]
    supervision["semantics"] = ("Unchanged interval-bag targets. Two input views share one physical event; "
        "training must average available view losses per physical_event_id. Views are not independent videos.")
    return supervision


def window_labels(target, clips, helper):
    positive, background, excluded = [], [], []
    for w in target["windows"]:
        state, detail = helper.classify_window(w, clips[w["video_id"]], target["index"])
        if state == "positive":
            positive.append(event_view(detail, w["view"]))
        elif state == "background":
            background.append(w["window_index"])
        else:
            excluded.append(dict(window_index=w["window_index"], reason=detail,
                                 video_id=w["video_id"], view=w["view"]))
    event_windows = defaultdict(list)
    for i, p in enumerate(positive):
        event_windows[(p["view"], p["physical_event_id"])].append(i)
    events, unrepresented = [], []
    for vid in sorted(clips):
        for view in VIEWS:
            for event in clips[vid]["events"]:
                if event["kind"] not in ("table_contact", "racket_hit"):
                    continue
                record = event_view(dict(event_id=event["event_id"], video_id=vid,
                    class_id=helper.CLASSES[event["kind"]]), view)
                members = event_windows[(view, event["event_id"])]
                if members:
                    events.append(dict(record, positive_window_indices=members))
                else:
                    unrepresented.append(dict(record, frame_interval=event["frame_interval"],
                                              reason="no_eligible_complete_window"))
    return dict(classes=helper.CLASSES, positive_windows=positive, events=events,
        background_windows=background, excluded_windows=excluded, unrepresented_events=unrepresented,
        semantics="Unchanged complete-window class/location supervision; two views of one physical event are averaged, not counted as independent events.")


def make_folds(target, supervision, frozen, groups, allowed, kind):
    folds, seen = [], Counter()
    index, windows = target["index"], target["windows"]
    for f in frozen:
        train, held = set(f["train_video_ids"]), set(f["held_out_video_ids"])
        assert not train & held and train | held == allowed
        assert held == {vid for vid in allowed if groups[vid] == f["held_out_family"]}
        seen.update(held)
        fold = dict(f,
            train_rows=[r["row_index"] for r in index if r["video_id"] in train],
            held_out_rows=[r["row_index"] for r in index if r["video_id"] in held],
            train_window_indices=[w["window_index"] for w in windows if w["video_id"] in train],
            held_out_window_indices=[w["window_index"] for w in windows if w["video_id"] in held])
        for prefix, vids in (("train", train), ("held_out", held)):
            fold[prefix + "_rows_by_view"] = {v:[r["row_index"] for r in index if r["video_id"] in vids and r["view"] == v] for v in VIEWS}
        if kind == "dense":
            bags = supervision["bags"]
            fold.update(train_bag_indices=[i for i, b in enumerate(bags) if b["video_id"] in train],
                held_out_bag_indices=[i for i, b in enumerate(bags) if b["video_id"] in held],
                train_background_indices=[i for i in supervision["background_indices"] if index[i]["video_id"] in train])
            norm = sorted(set(fold["train_background_indices"]) |
                          {i for b in fold["train_bag_indices"] for i in bags[b]["indices"]})
        else:
            events, positive = supervision["events"], supervision["positive_windows"]
            fold.update(train_event_indices=[i for i, e in enumerate(events) if e["video_id"] in train],
                held_out_event_indices=[i for i, e in enumerate(events) if e["video_id"] in held],
                train_background_windows=[i for i in supervision["background_windows"] if windows[i]["video_id"] in train])
            legal_windows = set(fold["train_background_windows"]) | {positive[pi]["window_index"]
                for ei in fold["train_event_indices"] for pi in events[ei]["positive_window_indices"]}
            norm = sorted({i for wi in legal_windows for i in windows[wi]["member_rows"]})
        assert set(norm) <= set(fold["train_rows"]) and not set(norm) & set(fold["held_out_rows"])
        fold["normalization_rows"] = norm
        folds.append(fold)
    assert len(folds) == 10 and set(seen) == allowed and all(n == 1 for n in seen.values())
    return dict(folds=folds, video_groups=groups,
        normalization="Legal training-side rows from both views only; original and augmented views of each video always remain in the same fold.")


def build(output, log):
    import numpy as np
    bag_helper = module("channel_bag_labels", BAG_HELPER)
    window_helper = module("channel_window_labels", WINDOW_HELPER)
    replay = module("channel_cache_locations", REPLAY_HELPER)
    selector = module("channel_fixed_selector", replay.SELECTOR).replace_observations
    sys.path.insert(0, str(replay.RUNTIME))
    runtime_inputs = RUNTIME / "participant/channel_adaptation_v001/inputs.py"
    build_dual_inputs = module("channel_adaptation_inputs", runtime_inputs).build_dual_inputs
    from participant.event_decision_v001.native_features import SLOT_CHANNELS, SUMMARY_NAMES
    from participant.event_decision_v001.trajectory_features import EXTRA_NAMES

    paths = [POLICY, RESERVOIR / "runtime/videos.json", RESERVOIR / "runtime/observations.jsonl",
        RESERVOIR / "runtime/candidates.jsonl", RESERVOIR / "training/timeline.json",
        RESERVOIR / "metadata/video_groups.json", RESERVOIR / "splits/lofo_by_video.json",
        replay.PROBE / "native_frame_channels.jsonl", replay.RUN / "replacements.json"]
    for dataset, files in ((DENSE, ("features.npz", "index.jsonl", "windows.json", "manifest.json")),
                           (WINDOW, ("windows.json", "manifest.json")),
                           (SPARSE, ("features.npz", "index.jsonl", "manifest.json"))):
        paths.extend(dataset / name for name in files)
    sources = [Path(__file__).resolve(), BAG_HELPER, WINDOW_HELPER, REPLAY_HELPER,
               replay.SELECTOR, bag_helper.SCORER, runtime_inputs]
    for package in ("dense_context_v001", "event_decision_v001", "event_window_v001"):
        sources.extend(sorted((replay.RUNTIME / "participant" / package).glob("*.py")))
    provenance = dict(input_sha256={relative(p):sha(p) for p in paths},
        source_sha256={relative(p):sha(p) for p in sources},
        observation_contract="Original anchors, parent membership and window support are preserved. Only augmented feature observations and new reliable dense-only singleton candidates change. Gap inference must still receive original observations.",
        original_gap_observations=dict(path=relative(RESERVOIR / "runtime/observations.jsonl"),
            sha256=sha(RESERVOIR / "runtime/observations.jsonl")),
        independent_videos=121, views_per_video=2, independent_videos_not_242=True,
        public15_used=False, protected33_used=False, training_runs=0, model_inference_calls=0, scoring_runs=0)
    save(output / "build_plan.json", dict(provenance, command=sys.argv, pid=os.getpid()))
    allowed = set(read(POLICY)["video_ids"])
    videos = {v["clip_id"]:v for v in read(RESERVOIR / "runtime/videos.json")["clips"]}
    groups = read(RESERVOIR / "metadata/video_groups.json")
    timeline = read(RESERVOIR / "training/timeline.json")
    assert len(allowed) == 121 and set(videos) == set(groups) == {c["clip_id"] for c in timeline["clips"]} == allowed
    clips = module("channel_timeline_validation", bag_helper.SCORER).validate_timeline(timeline)
    observations = bag_helper.grouped_rows(RESERVOIR / "runtime/observations.jsonl", allowed)
    anchors = bag_helper.grouped_rows(RESERVOIR / "runtime/candidates.jsonl", allowed)
    channels = bag_helper.grouped_rows(replay.PROBE / "native_frame_channels.jsonl", allowed)
    assert set(observations) == set(channels) == allowed and sum(map(len, anchors.values())) == 921
    saved_replacements = {r["video_id"]:r for r in read(replay.RUN / "replacements.json")}
    targets = {kind:dict(index=[], windows=[], sequence=[], summary=[]) for kind in ("dense", "window")}
    parents = {}
    for kind, dataset in (("dense", DENSE), ("window", SPARSE)):
        data = np.load(dataset / "features.npz", allow_pickle=False)
        parent_index = lines(dataset / "index.jsonl")
        parents[kind] = dict(index={r["candidate_id"]:r for r in parent_index}, sequence=data["sequence"], summary=data["summary"])
    parent_windows = {kind:{w["window_id"]:w for w in read(path / "windows.json")}
                      for kind, path in (("dense", DENSE), ("window", WINDOW))}
    receipts, checks = [], Counter()
    for number, vid in enumerate(sorted(allowed), 1):
        video, clip = videos[vid], clips[vid]
        assert all(video[k] == clip[k] for k in ("frame_count", "pts_seconds", "media_sha256"))
        augmented, replacement = selector(video, observations[vid], channels[vid])
        assert replacement == saved_replacements[vid], (vid, "Frozen selector replacement differs")
        inputs = build_dual_inputs(video, observations[vid], augmented, anchors[vid])
        receipts.append(dict(video_id=vid, replacement=replacement, runtime=inputs["receipt"]))
        for view in VIEWS:
            pool, sequence, summary, windows, _ = inputs[view]
            parent_rows = inputs["window_rows"][view]
            window_inputs = ([pool[i] for i in parent_rows], sequence[parent_rows], summary[parent_rows],
                             inputs["window_windows"][view])
            for kind, values in (("dense", (pool, sequence, summary, windows)), ("window", window_inputs)):
                ps, xs, zs, ws = values
                if view == "original":
                    for i, c in enumerate(ps):
                        old = parents[kind]["index"][c["candidate_id"]]
                        assert c == {k:v for k,v in old.items() if k not in ("row_index", "camera_family")}
                        assert np.array_equal(xs[i], parents[kind]["sequence"][old["row_index"]])
                        assert np.array_equal(zs[i], parents[kind]["summary"][old["row_index"]])
                        previous = parent_windows[kind][f"{vid}:w{c['native_frame']}"]
                        assert all(ws[i][key] == previous[key] for key in ("start_time", "end_time", "segment_id"))
                        old_members = [parents[kind]["index"][ps[j]["candidate_id"]]["row_index"]
                                       for j in ws[i]["member_rows"]]
                        assert old_members == previous["member_rows"]
                    checks[kind + "_original_rows_exact"] += len(ps)
                append_view(targets[kind], view, ps, xs, zs, ws, video, groups[vid])
        if number % 20 == 0 or number == 121:
            log(dict(videos_completed=number, dense_rows=len(targets["dense"]["index"])))
    assert checks == dict(dense_original_rows_exact=17028, window_original_rows_exact=8466)
    assert sum(r["replacement"]["replaced"] for r in receipts) == 531
    save(output / "view_provenance.json", dict(provenance, receipts=receipts,
        original_dataset_identity_checks=dict(checks)))
    all_counts = {}
    frozen = read(RESERVOIR / "splits/lofo_by_video.json")["folds"]
    for kind, target in targets.items():
        path = output / kind
        path.mkdir()
        index, windows = target["index"], target["windows"]
        assert [r["row_index"] for r in index] == [w["window_index"] for w in windows] == list(range(len(index)))
        assert len({r["candidate_id"] for r in index}) == len(index)
        assert all(index[i]["view"] == w["view"] and index[i]["video_id"] == w["video_id"] for w in windows for i in w["member_rows"])
        supervision = dense_labels(target, clips, bag_helper) if kind == "dense" else window_labels(target, clips, window_helper)
        folds = make_folds(target, supervision, frozen, groups, allowed, kind)
        np.savez_compressed(path / "features.npz", sequence=np.concatenate(target["sequence"]),
            summary=np.concatenate(target["summary"]), slot_channel_names=np.asarray(SLOT_CHANNELS),
            summary_names=np.asarray(tuple(SUMMARY_NAMES) + tuple(EXTRA_NAMES)))
        (path / "index.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in index), encoding="utf-8")
        save(path / "windows.json", windows)
        save(path / "supervision.json", supervision)
        save(path / "folds.json", folds)
        counts = dict(videos=121, source_families=10, views=2, candidates=len(index), windows=len(windows),
            candidates_by_view=dict(Counter(r["view"] for r in index)),
            physical_candidate_centers=len({r["physical_candidate_id"] for r in index}), folds=10)
        if kind == "dense":
            counts.update(bags=len(supervision["bags"]),
                represented_physical_events=len({b["physical_event_id"] for b in supervision["bags"]}),
                bags_by_view=dict(Counter(b["view"] for b in supervision["bags"])),
                background_rows=len(supervision["background_indices"]), excluded_rows=len(supervision["excluded_indices"]))
        else:
            counts.update(events=len(supervision["events"]),
                represented_physical_events=len({e["physical_event_id"] for e in supervision["events"]}),
                events_by_view=dict(Counter(e["view"] for e in supervision["events"])),
                positive_windows=len(supervision["positive_windows"]), background_windows=len(supervision["background_windows"]),
                excluded_windows=len(supervision["excluded_windows"]))
        manifest = dict(schema=f"channel_paired_{kind}_dataset.v1", counts=counts, **provenance,
            features=dict(path=relative(path / "features.npz"), sha256=sha(path / "features.npz")),
            index=dict(path=relative(path / "index.jsonl"), sha256=sha(path / "index.jsonl")),
            output_sha256={name:sha(path/name) for name in ("features.npz", "index.jsonl", "windows.json", "supervision.json", "folds.json")})
        save(path / "manifest.json", manifest)
        all_counts[kind] = counts
    assert all(sha(ROOT / name) == digest for name, digest in provenance["source_sha256"].items())
    save(output / "manifest.json", dict(schema="channel_paired_development121.v1", **provenance,
        counts=all_counts, original_dataset_identity_checks=dict(checks),
        output_sha256={f"{kind}/manifest.json":sha(output/kind/"manifest.json") for kind in targets},
        limitations="Two observation views of the same121 videos, not242 independent examples. Label completeness relies on the existing source declarations and frozen exclusions. No accuracy or FPS claim."))
    return all_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXP / "runs/dataset_v001")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                  pid=os.getpid(), completed=False, exit_code=None)
    def log(value):
        line = json.dumps(value, ensure_ascii=False)
        with (args.output / "console.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)
    save(args.output / "run_status.json", status)
    try:
        status["counts"] = build(args.output, log)
        status.update(completed=True, exit_code=0)
        log(status["counts"])
    except Exception:
        status.update(exit_code=1, error=traceback.format_exc())
        (args.output / "stderr.log").write_text(status["error"], encoding="utf-8")
        raise
    finally:
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-start)
        save(args.output / "run_status.json", status)
        (args.output / "exit_code.txt").write_text(str(status["exit_code"]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
