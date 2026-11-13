"""Freeze source-isolated nested folds using existing metadata; perform no fits."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
OLD = "experiments/20261110_adapted_visual_contact"
DATASET = "experiments/20261108_channel_event_adaptation/runs/dataset_v002/dense"
ALLOW = "experiments/20261102_research_foundation/data/development_videos.json"
LOSS = "experiments/20261105_coverage_aware_events/code/train_dense_context.py"
MODEL = OLD + "/versions/v001/participant/pingpang/participant/adapted_visual_v001/model.py"
TRAINER = OLD + "/code/train_adapted_visual.py"
TRAINER_SHA = "6a8dc9104fa8594f10f4f1b68a84b044df6d57bb074bf37b3ca62e14e57853c0"
MODEL_SHA = "45d7bb77eff31c92d9d9dd0e8671d81c9b295dc108c71527ee2b56615a02e7ee"
SCHEMA = "visual_calibration_nested_plan.v1"
CALIBRATION = dict(parameter="a", definition="softmax(a * log(max(raw_probability, 1e-30)))",
                   bounds=[0., 1.], probability_floor=1e-30, method="bounded",
                   xatol=1e-5, maxiter=100, compare_endpoints=[0., 1.],
                   event_weight=.5, background_weight=.5,
                   backgrounds="all legal backgrounds in outer training sources",
                   checkpoint="fixed numerical minimization, no outer-heldout selection",
                   global_optimum_guaranteed=False)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            result.update(part)
    return result.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes assets root")
    return path


def describe(root, relative):
    path = resolve(root, relative)
    return dict(path=relative, sha256=sha(path), bytes=path.stat().st_size)


def validate_probabilities(probabilities, rows, allowed):
    if set(probabilities) != set(allowed):
        raise ValueError("Probability video coverage differs")
    expected = defaultdict(set)
    for row in rows:
        expected[row["video_id"]].add(row["physical_candidate_id"])
    for vid in allowed:
        if set(probabilities[vid]) != expected[vid]:
            raise ValueError("Probability candidate coverage differs: " + vid)
        values = np.asarray(list(probabilities[vid].values()), dtype=np.float64)
        if not len(values):
            continue
        if (values.shape != (len(expected[vid]), 3) or not np.isfinite(values).all()
                or (values < 0).any() or (values > 1).any()
                or not np.allclose(values.sum(1), 1., atol=1e-6, rtol=0)):
            raise ValueError("Invalid probabilities: " + vid)


def source_bins(family_counts):
    if len(family_counts) != 9 or any(n < 0 for n in family_counts.values()):
        raise ValueError("Expected nine source candidate counts")
    families = sorted(family_counts, key=lambda name: (-family_counts[name], name))
    bins = [dict(bin_id=i, families=[], candidates=0) for i in range(3)]
    for family in families:
        target = min((b for b in bins if len(b["families"]) < 3),
                     key=lambda b: (b["candidates"], b["bin_id"]))
        target["families"].append(family)
        target["candidates"] += family_counts[family]
    return families, bins


def build_outer_folds(rows, supervision, split, allowed):
    """Bin nine outer-training sources using counts only, never outcomes."""
    groups = split["video_groups"]
    if set(groups) != allowed or len(set(groups.values())) != 10 or len(split["folds"]) != 10:
        raise ValueError("Expected the fixed ten-source development121 split")
    counts = Counter(groups[r["video_id"]] for r in rows)
    byrow = {r["row_index"]: r for r in rows}
    events = [b for b in supervision["bags"] if b["view"] == "augmented"]
    backgrounds = [i for i in supervision["background_indices"] if i in byrow]
    if Counter(b["class_id"] for b in events) != {0: 613, 1: 555}:
        raise ValueError("Expected the unchanged 613/555 physical event bags")
    for bag in events:
        if not bag["indices"] or any(byrow[i]["video_id"] != bag["video_id"] for i in bag["indices"]):
            raise ValueError("Cross-video or empty event bag")
    def statistics(videos):
        bags = [b for b in events if b["video_id"] in videos]
        return dict(videos=len(videos), candidates=sum(r["video_id"] in videos for r in rows),
                    events=len(bags), event_classes=dict(sorted(Counter(str(b["class_id"]) for b in bags).items())),
                    backgrounds=sum(byrow[i]["video_id"] in videos for i in backgrounds))
    outputs, outer_coverage = [], Counter()
    for fold in split["folds"]:
        train, held = set(fold["train_video_ids"]), set(fold["held_out_video_ids"])
        held_family = fold["held_out_family"]
        if train & held or train | held != allowed or {groups[v] for v in held} != {held_family}:
            raise ValueError("Outer fold is not the fixed source holdout")
        families, bins = source_bins({family: counts[family] for family in {groups[v] for v in train}})
        inner, coverage = [], Counter()
        for bucket in bins:
            calibration = {v for v in train if groups[v] in bucket["families"]}
            fitting = train - calibration
            if (len({groups[v] for v in fitting}) != 6 or len({groups[v] for v in calibration}) != 3
                    or fitting & calibration or (fitting | calibration) != train
                    or (fitting | calibration) & held):
                raise ValueError("Inner training/calibration escaped the outer training boundary")
            inner.append(dict(fold_id=bucket["bin_id"],
                held_out_family="|".join(sorted(bucket["families"])),
                held_out_families=sorted(bucket["families"]),
                train_video_ids=sorted(fitting), held_out_video_ids=sorted(calibration),
                train_counts=statistics(fitting), held_out_counts=statistics(calibration)))
            coverage.update(calibration)
        if set(coverage) != train or set(coverage.values()) != {1}:
            raise ValueError("Outer training videos must occur in exactly one inner holdout")
        outputs.append(dict(fold_id=fold["fold_id"], held_out_family=held_family,
            train_video_ids=sorted(train), held_out_video_ids=sorted(held),
            train_counts=statistics(train), held_out_counts=statistics(held),
            source_order=families, bins=bins, inner_folds=inner))
        outer_coverage.update(held)
    if set(outer_coverage) != allowed or set(outer_coverage.values()) != {1}:
        raise ValueError("Outer holdouts must cover development121 exactly once")
    return outputs


def metadata(root, files):
    paths = {name: resolve(root, item["path"]) for name, item in files.items()}
    for name, path in paths.items():
        if sha(path) != files[name]["sha256"] or path.stat().st_size != files[name]["bytes"]:
            raise ValueError("Frozen metadata/source differs: " + name)
    if files["old_trainer"]["sha256"] != TRAINER_SHA or files["old_model"]["sha256"] != MODEL_SHA:
        raise ValueError("Expected the exact previously run visual implementation")
    rows, supervision, split = lines(paths["rows"]), read(paths["supervision"]), read(paths["folds"])
    allowed = set(read(paths["allowlist"])["video_ids"])
    if len(allowed) != 121 or len(rows) != 17559 or {r["video_id"] for r in rows} - allowed:
        raise ValueError("Expected development121 augmented candidates")
    manifest = read(paths["manifest"])
    for name in ("index", "supervision", "folds"):
        if manifest["output_sha256"][paths[name].name] != files[name]["sha256"]:
            raise ValueError("Frozen dataset manifest differs")
    return rows, supervision, split, allowed


def prepare(args):
    output, root = args.output.resolve(), args.assets_root.resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh plan directory")
    sources = dict(old_trainer=TRAINER, old_model=MODEL, old_loss=LOSS, allowlist=ALLOW,
                   rows=OLD + "/runs/assets_v001/rows.jsonl",
                   time_features=OLD + "/runs/assets_v001/time_features.npy",
                   manifest=DATASET + "/manifest.json", index=DATASET + "/index.jsonl",
                   supervision=DATASET + "/supervision.json", folds=DATASET + "/folds.json")
    files = {name: describe(root, relative) for name, relative in sources.items()}
    rows, supervision, split, allowed = metadata(root, files)
    outer_status_path = args.outer_probabilities.parent / "run_status.json"
    outer_status, cache = read(outer_status_path), read(args.cache_receipt)
    if (not outer_status["completed"] or outer_status["training_runs"] != 10
            or any(outer_status[k] for k in ("protected33_used", "external130_used", "public15_used"))
            or sha(args.outer_probabilities) != outer_status["predictions_sha256"]
            or outer_status["code_sha256"]["train_adapted_visual.py"] != TRAINER_SHA
            or outer_status["code_sha256"]["model.py"] != MODEL_SHA):
        raise ValueError("Outer probabilities are not the completed frozen visual experiment")
    validate_probabilities(read(args.outer_probabilities), rows, allowed)
    if (not cache["complete"] or cache["schema"] != "adapted_visual_stem_cache.v1"
            or sha(args.cache_receipt) != outer_status["input_sha256"]["receipt.json"]
            or cache["model_source_sha256"] != MODEL_SHA):
        raise ValueError("Cache receipt differs from the original outer fit")
    for key in ("stem_features.npy", "rows.jsonl", "time_features.npy"):
        if cache["files"][key]["sha256"] != outer_status["input_sha256"][key]:
            raise ValueError("Outer/cache input identity differs: " + key)
    for name in ("index", "supervision", "folds", "rows", "time_features"):
        if files[name]["sha256"] != outer_status["input_sha256"][Path(sources[name]).name]:
            raise ValueError("Local data differs from actual outer fit inputs: " + name)
    protocol = EXP / "protocol_v001.json"
    plan = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(), files=files,
        protocol_sha256=sha(protocol), training_source_sha256=sha(EXP / "code/train_nested_visual.py"),
        preparation_source_sha256=sha(__file__), calibration=CALIBRATION,
        binning="descending candidate count, stable family-name tie; least-loaded bin with <3 sources, bin-ID tie",
        outer_folds=build_outer_folds(rows, supervision, split, allowed),
        original_visual_training=outer_status["training"],
        outer_reference=dict(probabilities_sha256=sha(args.outer_probabilities),
            run_status_sha256=sha(outer_status_path), candidates=17559, videos=121,
            model_fits=outer_status["folds"]),
        cache_reference=dict(receipt_sha256=sha(args.cache_receipt),
            files=cache["files"], pretrained_sha256=cache["pretrained_sha256"],
            rows=17559, videos=121, stem_shape=cache["stem_shape"]),
        expected=dict(visual_training_runs=30, calibration_fits=10, outer_models_reused=10,
            outer_heldout_videos=121, candidates=17559, epochs_per_visual_fit=16),
        data_scope=dict(protected33_used=False, external130_used=False, public15_used=False),
        fit_scope="All visual and scalar fits require --execute on the server; plan creation performs no fits")
    output.mkdir(parents=True)
    save(output / "plan.json", plan)
    shutil.copyfile(outer_status_path, output / "outer_reference_run_status.json")
    shutil.copyfile(args.cache_receipt, output / "cache_reference_receipt.json")
    receipt = dict(completed=True, plan_sha256=sha(output / "plan.json"), counts=plan["expected"],
        actual_visual_training_runs=0, actual_calibration_fits=0, media_decodes=0,
        protected33_used=False, public15_used=False, external130_used=False,
        inner_training_events_min=min(f["train_counts"]["events"] for o in plan["outer_folds"] for f in o["inner_folds"]),
        inner_training_events_max=max(f["train_counts"]["events"] for o in plan["outer_folds"] for f in o["inner_folds"]))
    save(output / "run_status.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ROOT)
    parser.add_argument("--outer-probabilities", type=Path, required=True)
    parser.add_argument("--cache-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
