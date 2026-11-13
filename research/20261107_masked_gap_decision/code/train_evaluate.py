"""One fixed ten-source experiment on natural gaps, with train-only masked inventory."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
INNER = EXP / "versions/v001/participant/pingpang/participant"
sys.path.insert(0, str(INNER))
from masked_gap_v001.features import FEATURE_NAMES
from masked_gap_v001.model import predict

RES = ROOT / "experiments/20261103_dual_path_rebuild/runs/reservoir_v001"
OLD = ROOT / "experiments/20261107_gap_contact_prototype"
DENSE = ROOT / "experiments/20261105_coverage_aware_events"
DATASET = EXP / "runs/dataset_v001"
PATHS = dict(
    features=DATASET / "features.npy", index=DATASET / "index.jsonl",
    policy=ROOT / "experiments/20261102_research_foundation/data/development_videos.json",
    folds=RES / "splits/lofo_by_video.json", timeline=RES / "training/timeline.json",
    groups=RES / "metadata/video_groups.json", proposals=OLD / "runs/gap_prototype_v001/proposals.jsonl",
    branches=DENSE / "runs/combined_runtime_v001/predictions.jsonl",
    baseline=DENSE / "runs/lofo_v001/evaluation/window_dense_union/strict.json",
    direct=OLD / "runs/gap_prototype_v001/strict.json",
    direct_report=OLD / "runs/gap_prototype_v001/report.json",
    direct_classification=OLD / "runs/gap_prototype_v001/new_candidate_classification.json",
    subsets=DENSE / "analysis/coverage_diagnosis_v001.json",
    scorer=ROOT / "experiments/20261102_p1_time_selection/runs/coverage_mapping_v001/sealed_source/evaluation/evaluate_timeline.py",
)
CONFIG = dict(epochs=200, seed=20261107, learning_rate=0.02, weight_decay=0.01,
              cpu_threads=2, checkpoint="last", dtype="float64", classes=["table_contact", "racket_hit", "background"],
              acceptance="argmax == table_contact", minimum_interval_seconds=0.08,
              normalization="training weighted mean/std; scale below 1e-6 becomes 1",
              weighting="equal class; equal available source within class; equal original group within source/class; split group mass across rows")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def row_weights(rows):
    grouped = defaultdict(list)
    identities = {}
    for i, row in enumerate(rows):
        key = (int(row["label"]), row["source_group"], row["group_id"])
        grouped[key].append(i)
        old = identities.setdefault(row["group_id"], key[:2])
        assert old == key[:2], "Group changed source or class"
    sources = defaultdict(set)
    counts = Counter((label, source) for label, source, _ in grouped)
    for label, source in counts:
        sources[label].add(source)
    assert set(sources) == {0, 1, 2}, "All three classes need training support"
    weights = np.zeros(len(rows), dtype=np.float64)
    for (label, source, _), ids in grouped.items():
        weights[ids] = 1. / (3 * len(sources[label]) * counts[label, source] * len(ids))
    assert np.isclose(weights.sum(), 1.)
    return weights, dict(rows=len(rows), unique_groups=len(grouped),
        groups_by_class=dict(Counter(str(k[0]) for k in grouped)),
        groups_by_source_class={f"{s}:{c}": n for (c, s), n in sorted(counts.items())},
        class_weight={str(c): float(weights[[i for i, r in enumerate(rows) if r["label"] == c]].sum()) for c in range(3)})


def prepare():
    rows = lines(PATHS["index"])
    x = np.load(PATHS["features"])
    allowed = set(read(PATHS["policy"])["video_ids"])
    assert len(allowed) == 121 and {r["video_id"] for r in rows} <= allowed
    assert x.shape == (len(rows), len(FEATURE_NAMES)) and np.isfinite(x).all()
    assert [r["row_index"] for r in rows] == list(range(len(rows)))
    natural = [r for r in rows if r["origin"] == "natural"]
    assert len(natural) == 116 and [r["row_index"] for r in natural] == list(range(116))
    folds = read(PATHS["folds"])["folds"]
    assert len(folds) == 10
    coverage, fitted = Counter(), []
    for fold in folds:
        train, held = set(fold["train_video_ids"]), set(fold["held_out_video_ids"])
        assert not train & held and train | held == allowed
        coverage.update(held)
        selected = [r for r in rows if r["video_id"] in train and r["label"] >= 0]
        _, inventory = row_weights(selected)
        fitted.append(dict(fold_id=fold["fold_id"], held_out_family=fold["held_out_family"], **inventory))
    assert coverage == Counter({vid: 1 for vid in allowed})
    sources = list(INNER.rglob("*.py")) + [Path(__file__), EXP / "code/build_dataset.py", PATHS["scorer"]]
    protocol = dict(schema="masked_gap_decision.v1", prepared_utc=datetime.now(timezone.utc).isoformat(),
        config=CONFIG, feature_names=FEATURE_NAMES, features=len(FEATURE_NAMES),
        trainable_parameters=3 * (len(FEATURE_NAMES) + 1), video_ids=sorted(allowed), folds=fitted,
        input_sha256={k: sha(p) for k, p in PATHS.items()},
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        target_semantics="Candidate classification within a complete legal contact interval, not a precise per-frame contact target; copies share original event weight",
        operations=dict(training_runs=10, epochs_per_run=200, scoring_runs=1,
                        public15_used=False, protected33_used=False, ssh=False, full_video_inference_runs=0),
        advance_rule="Inspect complete-chain net F1, recovered GT and source effects; do not select thresholds/epochs or claim independent validation")
    save(EXP / "protocol_v001.json", protocol)
    print(json.dumps(dict(prepared=True, rows=len(rows), natural=116, features=len(FEATURE_NAMES), parameters=protocol["trainable_parameters"])))


def compact(result):
    groups = result["per_camera_family"]
    return dict(metrics=result["metrics_on_evaluable_regions"], per_source=groups,
                macro_f1=sum(g["f1"] for g in groups) / len(groups))


def train_fold(x, rows, fold, out):
    train_ids = set(fold["train_video_ids"])
    selected = [r for r in rows if r["video_id"] in train_ids and r["label"] >= 0]
    weights, inventory = row_weights(selected)
    indices = [r["row_index"] for r in selected]
    values = x[indices]
    mean = (values * weights[:, None]).sum(axis=0)
    scale = np.sqrt(((values - mean) ** 2 * weights[:, None]).sum(axis=0))
    scale[scale < 1e-6] = 1.
    z = torch.as_tensor((values - mean) / scale, dtype=torch.float64)
    labels = torch.as_tensor([r["label"] for r in selected], dtype=torch.long)
    w = torch.as_tensor(weights, dtype=torch.float64)
    torch.manual_seed(CONFIG["seed"])
    model = torch.nn.Linear(len(FEATURE_NAMES), 3, dtype=torch.float64)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    losses = []
    for epoch in range(1, CONFIG["epochs"] + 1):
        optimizer.zero_grad()
        loss = (torch.nn.functional.cross_entropy(model(z), labels, reduction="none") * w).sum()
        loss.backward()
        optimizer.step()
        if epoch in (1, 50, 100, 150, 200):
            losses.append(dict(epoch=epoch, weighted_loss=float(loss.detach())))
    model.eval()
    result = dict(schema="masked_gap_linear.v1", feature_names=FEATURE_NAMES, classes=CONFIG["classes"],
        mean=mean.tolist(), scale=scale.tolist(), coefficients=model.weight.detach().numpy().tolist(),
        intercept=model.bias.detach().numpy().tolist(), config=CONFIG, fold_id=fold["fold_id"],
        held_out_family=fold["held_out_family"], train_video_ids=sorted(train_ids), inventory=inventory,
        training_losses=losses)
    held = [r for r in rows if r["origin"] == "natural" and r["video_id"] in fold["held_out_video_ids"]]
    values_held = x[[r["row_index"] for r in held]]
    logits, probs = predict(values_held, result)
    with torch.no_grad():
        expected = model(torch.as_tensor((values_held - mean) / scale, dtype=torch.float64)).numpy()
    difference = float(np.max(np.abs(logits - expected))) if len(held) else 0.
    assert difference < 1e-10
    result["numpy_export_max_logit_difference"] = difference
    weight_path = out / "weights" / f"fold_{fold['fold_id']:02d}.json"
    save(weight_path, result)
    outputs = [dict(row_index=r["row_index"], candidate_id=r["candidate"]["candidate_id"],
                    video_id=r["video_id"], fold_id=fold["fold_id"], logits=l.tolist(), probabilities=p.tolist(),
                    predicted_class=int(np.argmax(l)), margin=float(l[0] - max(l[1], l[2])))
               for r, l, p in zip(held, logits, probs)]
    return outputs, dict(fold_id=fold["fold_id"], held_out_family=fold["held_out_family"],
                        weights_path=weight_path.relative_to(ROOT).as_posix(), weights_sha256=sha(weight_path),
                        held_out_natural_rows=len(held), inventory=inventory, training_losses=losses,
                        numpy_export_max_logit_difference=difference)


def execute(out):
    plan = read(EXP / "protocol_v001.json")
    assert plan["config"] == CONFIG and plan["feature_names"] == FEATURE_NAMES
    assert all(sha(PATHS[k]) == value for k, value in plan["input_sha256"].items())
    assert all(sha(ROOT / k) == value for k, value in plan["source_sha256"].items())
    x, rows = np.load(PATHS["features"]), lines(PATHS["index"])
    torch.set_num_threads(CONFIG["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    outputs, fits = [], []
    start = time.perf_counter()
    for fold in read(PATHS["folds"])["folds"]:
        predicted, fit = train_fold(x, rows, fold, out)
        outputs.extend(predicted)
        fits.append(fit)
        save(out / "folds" / f"fold_{fold['fold_id']:02d}.json", dict(**fit, outputs=predicted))
        print(json.dumps(dict(fold_id=fold["fold_id"], held_out=fit["held_out_natural_rows"], final_loss=fit["training_losses"][-1])), flush=True)
    training_seconds = time.perf_counter() - start
    outputs.sort(key=lambda r: r["row_index"])
    assert [r["row_index"] for r in outputs] == list(range(116))
    save(out / "natural_oof.json", outputs)
    # All predictions are finalized before opening evaluation labels/results.
    proposals = lines(PATHS["proposals"])
    branches = {r["video_id"]: r for r in lines(PATHS["branches"])}
    by_candidate = {r["candidate_id"]: r for r in outputs}
    predictions, admission, per_video = [], [], []
    for proposal in proposals:
        vid = proposal["video_id"]
        current = [dict(e) for e in branches[vid]["union"]]
        original = {e["candidate_id"]: e for e in current}
        accepted, rejected = [], []
        for c in sorted(proposal["candidates"], key=lambda e: (-by_candidate[e["candidate_id"]]["margin"], e["pts_seconds"], e["frame_id"])):
            decision = by_candidate[c["candidate_id"]]
            if decision["predicted_class"] != 0:
                rejected.append(dict(candidate_id=c["candidate_id"], reason="classified_non_table", predicted_class=decision["predicted_class"]))
                continue
            conflicts = [e["candidate_id"] for e in current if abs(e["pts_seconds"] - c["pts_seconds"]) < CONFIG["minimum_interval_seconds"]]
            if conflicts:
                rejected.append(dict(candidate_id=c["candidate_id"], reason="minimum_interval_conflict", conflicts=conflicts))
            else:
                current.append(dict(c, selected_model="masked_gap_v001", probabilities=decision["probabilities"], margin=decision["margin"]))
                accepted.append(c["candidate_id"])
        assert all(next(e for e in current if e["candidate_id"] == key) == e for key, e in original.items())
        current.sort(key=lambda e: (e["pts_seconds"], e["frame_id"]))
        per_video.append(dict(video_id=vid, events=current))
        admission.append(dict(video_id=vid, accepted_ids=accepted, rejected=rejected))
        predictions.extend(dict(e, clip_id=vid, native_frame_id=e["frame_id"]) for e in current)
    save(out / "predictions_by_video.json", per_video)
    save(out / "predictions.json", predictions)
    save(out / "admission.json", admission)
    scorer = load("masked_gap_scorer", PATHS["scorer"])
    timeline = read(PATHS["timeline"])
    clips = scorer.validate_timeline(timeline)
    result = scorer.evaluate(timeline, predictions, 0.)
    assert result["eligible_positive_events"] == 680
    save(out / "strict.json", result)
    baseline, direct = read(PATHS["baseline"]), read(PATHS["direct"])
    old_hits = {p["event_id"] for p in baseline["matching"]["pairs"]}
    direct_hits = {p["event_id"] for p in direct["matching"]["pairs"]}
    hits = {p["event_id"] for p in result["matching"]["pairs"]}
    direct_recovered = direct_hits - old_hits
    accepted_ids = {c for v in admission for c in v["accepted_ids"]}
    previous_class = read(PATHS["direct_classification"])
    previous_racket_ids = {r["candidate_id"] for r in previous_class if r["kind"] == "labeled_racket_hit_interval"}
    natural_confusion = np.zeros((3, 3), dtype=int)
    for prediction in outputs:
        label = rows[prediction["row_index"]]["label"]
        if label >= 0:
            natural_confusion[label, prediction["predicted_class"]] += 1
    classifications = []
    for r in result["event_outcomes"]:
        if r["candidate_id"] not in accepted_ids:
            continue
        overlaps = [e for e in clips[r["clip_id"]]["events"] if e["pts_interval_seconds"][0] - 1e-9 <= r["pts_seconds"] <= e["pts_interval_seconds"][1] + 1e-9]
        if r["assessment"] == "tp": kind = "matched_table_contact"
        elif "excluded" in r["assessment"]: kind = r["assessment"]
        elif any(e["event_id"] in result["matching"]["adjacency"] for e in overlaps): kind = "duplicate_table_interval"
        elif any(e["kind"] == "racket_hit" for e in overlaps): kind = "labeled_racket_hit_interval"
        else: kind = "other_evaluable_background"
        classifications.append(dict(candidate_id=r["candidate_id"], video_id=r["clip_id"], frame_id=r["native_frame_id"], kind=kind, assessment=r["assessment"]))
    subset_results = {}
    for name, expected in (("no_candidate28", 28), ("no_positive_window133", 133)):
        ids = {e["event_id"] for e in read(PATHS["subsets"])["event_details"] if name in e["target_groups"]}
        assert len(ids) == expected
        subset_results[name] = dict(events=expected, baseline_hits=len(old_hits & ids), direct_hits=len(direct_hits & ids),
                                   current_hits=len(hits & ids), recovered=sorted((hits - old_hits) & ids))
    save(out / "new_candidate_classification.json", classifications)
    report = dict(completed=True, protocol_sha256=sha(EXP / "protocol_v001.json"), input_sha256=plan["input_sha256"],
        baseline=compact(baseline), direct_gap=compact(direct), classified_gap=compact(result), fits=fits,
        training_seconds=training_seconds, natural_candidates=116,
        predicted_class_counts=dict(Counter(CONFIG["classes"][r["predicted_class"]] for r in outputs)),
        admitted=len(accepted_ids), rejection_counts=dict(Counter(r["reason"] for v in admission for r in v["rejected"])),
        original_2d_outputs_preserved=True, recovered_vs_2d=sorted(hits - old_hits), lost_vs_2d=sorted(old_hits - hits),
        retained_previous_recoveries=sorted(direct_recovered & hits), rejected_previous_recoveries=sorted(direct_recovered - hits),
        previous_racket_fp_count=len(previous_racket_ids), previous_racket_fp_remaining=len(previous_racket_ids & accepted_ids),
        previous_racket_fp_removed=sorted(previous_racket_ids - accepted_ids),
        natural_labeled_confusion_rows_truth_columns_prediction=natural_confusion.tolist(),
        new_candidate_classification=dict(Counter(r["kind"] for r in classifications)), subsets=subset_results,
        predictions_sha256=sha(out / "predictions.json"), **plan["operations"],
        limitations=["Source-held-out development, repeatedly consulted; not blind generalization",
                     "Only naturally missing candidates evaluated; masked examples are dependent copies",
                     "Labels supervise contact class in a legal interval, not exact sub-frame time or XY",
                     "One combined prototype cannot isolate augmentation causal benefit",
                     "Cache CPU training/evaluation; no full inference FPS or public coordinate acceptance"])
    save(out / "report.json", report)
    save(EXP / "analysis/lofo_v001.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-name", default="lofo_v001")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if Path(args.run_name).name != args.run_name:
        raise ValueError("Expected a simple run name")
    out = EXP / "runs" / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), command=sys.argv, completed=False, exit_code=1)
    with (out / "console.log").open("x", encoding="utf-8") as console:
        try:
            with redirect_stdout(console), redirect_stderr(console):
                report = execute(out)
            status.update(completed=True, exit_code=0)
        except BaseException:
            status["error"] = traceback.format_exc()
            console.write(status["error"])
            raise
        finally:
            status.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - start)
            save(out / "run_status.json", status)
            (out / "exit_code.txt").write_text(str(status["exit_code"]) + "\n", encoding="utf-8")
    print(json.dumps(dict(completed=True, metrics=report["classified_gap"]["metrics"], admitted=report["admitted"], recovered=len(report["recovered_vs_2d"]))))


if __name__ == "__main__":
    main()
