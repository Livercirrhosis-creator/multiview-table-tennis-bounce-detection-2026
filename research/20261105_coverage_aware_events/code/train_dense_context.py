"""Fixed dense-center context training with interval and competing-contact loss."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
RUNTIME = EXP / "versions/v001/participant/pingpang"
DATASET = EXP / "runs/dataset_v001"
PROTOCOL = EXP / "protocol_v001.json"
ALLOWED = ROOT / "experiments/20261102_research_foundation/data/development_videos.json"
TIMELINE = ROOT / "experiments/20261103_dual_path_rebuild/runs/reservoir_v001/training/timeline.json"
SCORER = ROOT / "experiments/20261102_p1_time_selection/runs/coverage_mapping_v001/sealed_source/evaluation/evaluate_timeline.py"
PARENT = ROOT / "experiments/20261105_disagreement_fusion"
CONTROL_RUN = PARENT / "runs/fusion_lofo_v001/evaluation"
OLD_INDEX = ROOT / "experiments/20261104_event_decision_rebuild/runs/dataset_v002/index.jsonl"
WINDOW = ROOT / "experiments/20261104_event_presence_location"
LOSS_CHECK = EXP / "analysis/loss_semantics_v001.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def member_sha(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("ascii")).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def configuration():
    cfg = read(PROTOCOL)
    training = cfg["training"]
    assert training["fits"] == 10 and training["epochs"] == 80 and training["seed"] == 20261103
    assert training["optimizer"] == "AdamW" and training["learning_rate"] == .001 and training["weight_decay"] == .0001
    assert training["event_batch_size"] == 64 and training["threads"] == 2 and training["interop_threads"] == 1
    assert training["checkpoint"] == "Fixed last epoch" and training["variants"] == 1
    assert cfg["model"]["expected_parameters"] == 31651
    assert cfg["supervision"]["total_loss"] == "0.5event_loss+0.5background_loss"
    return cfg


def event_loss_terms(logits, bag_rows, classes):
    """One latent target position; all legal positions exclude only the other contact."""
    import torch
    assert len(bag_rows) == len(classes) and all(c in (0, 1) for c in classes)
    width = max(len(b) for b in bag_rows)
    padded = np.zeros((len(bag_rows), width), dtype=np.int64)
    mask = np.zeros_like(padded, dtype=bool)
    for i, rows in enumerate(bag_rows):
        assert rows
        padded[i, :len(rows)] = rows
        mask[i, :len(rows)] = True
    members = torch.from_numpy(padded)
    valid = torch.from_numpy(mask)
    kinds = torch.tensor(classes, dtype=torch.long)
    logp = torch.log_softmax(logits, dim=-1)
    target_logp = logp[members, kinds[:, None]].masked_fill(~valid, -torch.inf)
    target_mil = -target_logp.max(dim=1).values
    bag_logits = logits[members]
    # log(1-p_other) = logsumexp(target, background) - logsumexp(all).
    # This avoids subtracting p_other from one when the other class dominates.
    allowed_classes = torch.stack((kinds, torch.full_like(kinds, 2)), dim=1)
    allowed_logits = bag_logits.gather(2, allowed_classes[:, None, :].expand(-1, width, -1))
    forbidden_nll = torch.logsumexp(bag_logits, dim=-1)-torch.logsumexp(allowed_logits, dim=-1)
    other_contact = forbidden_nll.masked_fill(~valid, 0.).sum(dim=1)/valid.sum(dim=1)
    return target_mil, other_contact


def check_loss_semantics():
    import torch
    torch.set_num_threads(2)
    # Nonwinning positions may stay target or background; they are not hard background labels.
    original = torch.tensor([[.7, .1, .2], [.2, .1, .7]], dtype=torch.float64)
    shifted = torch.tensor([[.7, .1, .2], [.01, .1, .89]], dtype=torch.float64)
    contrary = torch.tensor([[.7, .1, .2], [.1, .8, .1]], dtype=torch.float64)
    def loss(p, rows, kinds):
        mil, other = event_loss_terms(p.log(), rows, kinds)
        return (mil+other).mean(), mil, other
    a, mil, other = loss(original, [[0, 1]], [0])
    expected = -math.log(.7)-math.log(.9)
    assert abs(float(a)-expected) < 1e-12
    b, _, _ = loss(shifted, [[0, 1]], [0])
    assert abs(float(a-b)) < 1e-12
    c, _, _ = loss(contrary, [[0, 1]], [0])
    assert c > a
    symmetric, _, _ = loss(original[:, [1, 0, 2]], [[0, 1]], [1])
    assert abs(float(symmetric-a)) < 1e-12
    both = torch.cat((original, torch.tensor([[.2, .6, .2]], dtype=torch.float64)))
    balanced, _, _ = loss(both, [[0, 1], [2]], [0, 1])
    expected_balanced = .5*(expected-math.log(.6)-math.log(.8))
    assert abs(float(balanced)-expected_balanced) < 1e-12
    extreme = torch.tensor([[1000., -1000., 0.], [-1000., 1000., 0.]], dtype=torch.float64, requires_grad=True)
    em, eo = event_loss_terms(extreme, [[0, 1]], [0])
    extreme_loss = (em+eo).mean()
    extreme_loss.backward()
    assert torch.isfinite(extreme_loss) and torch.isfinite(extreme.grad).all()
    return dict(passed=True, source_sha256=sha(Path(__file__)), protocol_sha256=sha(PROTOCOL),
        checks=dict(hand_calculated_max_mil_plus_other_contact=True,
            nonwinning_target_background_reallocation_not_penalized=True,
            other_contact_at_nonwinning_position_penalized=True,
            table_racket_symmetric=True, physical_events_equal_weight_despite_bag_length=True,
            extreme_logits_finite_loss_and_gradient=True),
        numerical_example=dict(mil=float(mil[0]), other_contact=float(other[0]), total=float(a),
            expected=expected, changed_other_contact_total=float(c), extreme_total=float(extreme_loss)),
        model_forwards=0, training_fits=0, scoring_runs=0)


def membership(supervision, fold):
    bags = [supervision["bags"][i] for i in fold["train_bag_indices"]]
    background = fold["train_background_indices"]
    legal = sorted({i for b in bags for i in b["indices"]} | set(background))
    assert legal == fold["normalization_rows"]
    assert set(legal) <= set(fold["train_rows"]) and not set(legal) & set(fold["held_out_rows"])
    assert bags and background
    return bags, background, legal


def load_dataset(dataset):
    manifest = read(dataset / "manifest.json")
    for name, digest in manifest["output_sha256"].items():
        assert sha(dataset / name) == digest, name
    index, windows = lines(dataset / "index.jsonl"), read(dataset / "windows.json")
    supervision, folds = read(dataset / "supervision.json"), read(dataset / "folds.json")["folds"]
    allowed = set(read(ALLOWED)["video_ids"])
    assert len(allowed) == 121 and len(index) == len(windows) == 17028 and len(folds) == 10
    assert {r["video_id"] for r in index} <= allowed
    assert [r["row_index"] for r in index] == list(range(len(index)))
    for i, w in enumerate(windows):
        assert w["window_index"] == w["center_row_index"] == i and i in w["member_rows"]
        assert w["video_id"] == index[i]["video_id"]
        assert all(index[j]["video_id"] == w["video_id"] for j in w["member_rows"])
    for fold in folds:
        assert set(fold["train_video_ids"]) | set(fold["held_out_video_ids"]) == allowed
        assert set(fold["train_video_ids"]).isdisjoint(fold["held_out_video_ids"])
        assert all(index[i]["camera_family"] == fold["held_out_family"] for i in fold["held_out_rows"])
        train, held = set(fold["train_rows"]), set(fold["held_out_rows"])
        assert train.isdisjoint(held) and train | held == set(range(len(index)))
        assert all(set(windows[i]["member_rows"]) <= train for i in train)
        assert all(set(windows[i]["member_rows"]) <= held for i in held)
        membership(supervision, fold)
    return manifest, index, windows, supervision, folds, allowed


def runtime_modules():
    sys.path.insert(0, str(RUNTIME))
    from participant.dense_context_v001.model import DenseContextModel
    from participant.event_decision_v001 import normalization, trajectory_features
    from participant.event_decision_v001.selection import select_events
    from participant.event_union_v001.runtime import combine_events
    return DenseContextModel, normalization, trajectory_features, select_events, combine_events


def source_paths(dataset):
    paths = [Path(__file__), PROTOCOL, ALLOWED, SCORER, dataset / "manifest.json", LOSS_CHECK,
        EXP / "analysis/coverage_diagnosis_v001.json", EXP / "analysis/added_fp_diagnosis_v001.json",
        WINDOW / "runs/dataset_v001/manifest.json", WINDOW / "runs/dataset_v001/windows.json", OLD_INDEX]
    for package in ("dense_context_v001", "event_decision_v001", "event_window_v001", "event_union_v001"):
        paths.extend(sorted((RUNTIME / "participant" / package).glob("*.py")))
    for arm in ("previous_complete_2d", "window", "union", "residual_fusion"):
        paths.extend(CONTROL_RUN / arm / name for name in ("predictions.json", "strict.json", "tolerance40ms.json"))
    return paths


def prepare_plan(dataset):
    cfg = configuration()
    assert read(LOSS_CHECK)["passed"]
    manifest, index, windows, supervision, folds, allowed = load_dataset(dataset)
    model_class, *_ = runtime_modules()
    model = model_class()
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == cfg["model"]["expected_parameters"]
    planned = []
    for fold in folds:
        bags, background, legal = membership(supervision, fold)
        planned.append(dict(fold_id=fold["fold_id"], held_out_family=fold["held_out_family"],
            training_events=len(bags), training_background_rows=len(background),
            normalization_rows=len(legal), normalization_membership_sha256=member_sha(legal),
            train_rows=len(fold["train_rows"]), held_out_rows=len(fold["held_out_rows"]),
            train_video_ids=fold["train_video_ids"], held_out_video_ids=fold["held_out_video_ids"],
            optimizer_steps=80*math.ceil(len(bags)/64)))
    return dict(prepared=True, protocol_sha256=sha(PROTOCOL), configuration=cfg,
        dataset_manifest_sha256=sha(dataset / "manifest.json"), dataset_counts=manifest.get("counts"),
        candidates=len(index), windows=len(windows), videos=len(allowed), parameters=parameters,
        folds=planned, source_sha256={relative(p):sha(p) for p in source_paths(dataset)},
        loss_semantics_sha256=sha(LOSS_CHECK), training_fits=0, optimizer_steps=0, scoring_runs=0,
        unknown_context="May be encoded as label-free context on training source; never fitted normalization or loss center",
        frozen_window="Original sparse8466 candidate/window membership and previously saved nine-source OOF outputs",
        public15_used=False, protected33_used=False, network_calls=0)


def batch_losses(model, x, z, windows, bags, background):
    import torch
    centers = sorted({i for b in bags for i in b["indices"]} | set(background))
    active = sorted({j for i in centers for j in windows[i]["member_rows"]})
    context_local = {i:j for j,i in enumerate(active)}
    center_local = {i:j for j,i in enumerate(centers)}
    embeddings = model.encode(x[active], z[active])
    logits = model.candidate_logits(embeddings,
        [[context_local[j] for j in windows[i]["member_rows"]] for i in centers],
        [context_local[i] for i in centers])
    mil, other = event_loss_terms(logits, [[center_local[i] for i in b["indices"]] for b in bags],
        [b["class_id"] for b in bags])
    event_loss = (mil+other).mean()
    logp = torch.log_softmax(logits, -1)
    background_loss = -logp[[center_local[i] for i in background], 2].mean() if background else logits.sum()*0
    loss = .5*(event_loss+background_loss)
    return loss, dict(event_mil=float(mil.mean().detach()), other_contact=float(other.mean().detach()),
        background_ce=float(background_loss.detach()), events=len(bags), background=len(background),
        encoded_rows=len(active), supervised_centers=len(centers))


def fit_fold(sequence, summary, windows, supervision, fold, cfg, out, runtime):
    import torch
    model_class, normalization, trajectory_features, _, _ = runtime
    out.mkdir(parents=True)
    bags, background, legal = membership(supervision, fold)
    norm = dict(sequence=normalization.fit_statistics(sequence[legal], "tcn"),
        summary=trajectory_features.fit_statistics(summary[legal]))
    x = torch.from_numpy(normalization.transform(sequence, norm["sequence"], "tcn"))
    z = torch.from_numpy(trajectory_features.transform(summary, norm["summary"]))
    save(out / "normalization.json", norm)
    save(out / "membership.json", dict(fold=fold, normalization_rows=legal,
        normalization_membership_sha256=member_sha(legal)))
    training = cfg["training"]
    torch.manual_seed(training["seed"])
    rng = np.random.default_rng(training["seed"])
    model = model_class().cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    batches, encoded = math.ceil(len(bags)/64), 0
    tick = time.perf_counter()
    with (out / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, 81):
            started = time.perf_counter()
            order = rng.permutation(len(bags))
            bg_batches = np.array_split(rng.permutation(background), batches)
            sums = defaultdict(float)
            for batch, start in enumerate(range(0, len(bags), 64)):
                selected_bags = [bags[i] for i in order[start:start+64]]
                optimizer.zero_grad(set_to_none=True)
                loss, m = batch_losses(model, x, z, windows, selected_bags, bg_batches[batch].tolist())
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                sums["event_mil"] += m["event_mil"]*m["events"]
                sums["other_contact"] += m["other_contact"]*m["events"]
                sums["background_ce"] += m["background_ce"]*m["background"]
                encoded += m["encoded_rows"]
            log.write(json.dumps(dict(epoch=epoch, event_mil=sums["event_mil"]/len(bags),
                other_contact=sums["other_contact"]/len(bags), background_ce=sums["background_ce"]/len(background),
                optimizer_steps=epoch*batches, epoch_seconds=time.perf_counter()-started))+"\n")
            log.flush()
            if epoch % 20 == 0:
                print(f"dense fold {fold['fold_id']} epoch {epoch}/80, elapsed {time.perf_counter()-tick:.1f}s", flush=True)
    fit_seconds = time.perf_counter()-tick
    model.eval()
    held = np.asarray(fold["held_out_rows"], dtype=np.int64)
    local = {int(i):j for j,i in enumerate(held)}
    with torch.inference_mode():
        started = time.perf_counter()
        embeddings = torch.cat([model.encode(x[held[start:start+512]], z[held[start:start+512]])
            for start in range(0, len(held), 512)])
        encode_seconds = time.perf_counter()-started
        started = time.perf_counter()
        logits = model.candidate_logits(embeddings,
            [[local[j] for j in windows[int(i)]["member_rows"]] for i in held], [local[int(i)] for i in held])
        probabilities = torch.softmax(logits, -1).numpy()
        logits = logits.numpy()
        head_seconds = time.perf_counter()-started
    torch.save(dict(model_state_dict=model.state_dict(), classes=["table_contact", "racket_hit", "background"],
        architecture="DenseContextModel", seed=training["seed"], epoch=80,
        held_out_family=fold["held_out_family"]), out / "model.pt")
    np.save(out / "held_rows.npy", held)
    np.save(out / "held_out_probabilities.npy", probabilities)
    result = dict(fold_id=fold["fold_id"], held_out_family=fold["held_out_family"], training_events=len(bags),
        training_background_rows=len(background), normalization_rows=len(legal),
        normalization_membership_sha256=member_sha(legal), held_out_rows=len(held), epochs=80, seed=training["seed"],
        optimizer_steps=80*batches, fit_seconds=fit_seconds, encoded_training_rows_across_batches=encoded,
        held_out_encode_seconds=encode_seconds, held_out_head_seconds=head_seconds,
        held_out_model_seconds=encode_seconds+head_seconds, parameters=sum(p.numel() for p in model.parameters()),
        model_sha256=sha(out / "model.pt"), normalization_sha256=sha(out / "normalization.json"),
        membership_sha256=sha(out / "membership.json"))
    save(out / "fit_result.json", result)
    return held, logits, probabilities, result


def compact(result):
    families = result["per_camera_family"]
    return dict(metrics=result["metrics_on_evaluable_regions"], total_predictions=result["total_predictions"],
        excluded=result["unknown_excluded_predictions"]+result["partial_positive_excluded_predictions"],
        source_macro_f1=sum(f["f1"] for f in families)/len(families), families=families)


def compare(after, before):
    a, b = compact(after), compact(before)
    groups = {f["camera_family"]:f for f in b["families"]}
    am, bm = set(after["missed_event_ids"]), set(before["missed_event_ids"])
    return dict(recovered_event_ids=sorted(bm-am), lost_event_ids=sorted(am-bm),
        aggregate_delta={k:a["metrics"][k]-b["metrics"][k] for k in ("tp", "fp", "fn", "f1")},
        source_macro_f1_delta=a["source_macro_f1"]-b["source_macro_f1"],
        sources=[dict(camera_family=f["camera_family"],
            delta={k:f[k]-groups[f["camera_family"]][k] for k in ("tp", "fp", "fn", "f1")}) for f in a["families"]])


def evaluate(out, index, probabilities, allowed, runtime):
    _, _, _, select_events, combine_events = runtime
    previous = read(PARENT / "runs/fusion_lofo_v001/report.json")["evaluation"]
    assert sha(SCORER) == previous["evaluator_sha256"] and sha(TIMELINE) == previous["timeline_sha256"]
    scorer = module("dense_frozen_event_evaluator", SCORER)
    timeline = read(TIMELINE)
    assert {c["clip_id"] for c in timeline["clips"]} == allowed
    controls = ("previous_complete_2d", "window", "union", "residual_fusion")
    details = {arm:{label:read(CONTROL_RUN / arm / (label+".json")) for label in ("strict", "tolerance40ms")} for arm in controls}
    hashes = {relative(CONTROL_RUN / arm / name):sha(CONTROL_RUN / arm / name)
        for arm in controls for name in ("predictions.json", "strict.json", "tolerance40ms.json")}
    fixed_window = read(CONTROL_RUN / "window/predictions.json")
    assert fixed_window == read(WINDOW / "runs/lofo_v001/evaluation/window_events/predictions.json")
    old_keys = {(r["video_id"], r["native_frame"]) for r in lines(OLD_INDEX)}
    assert len(old_keys) == 8466
    assert all((p["clip_id"], p["native_frame_id"]) in old_keys for p in fixed_window)
    rows_by_video, window_by_video = defaultdict(list), defaultdict(list)
    for r in index: rows_by_video[r["video_id"]].append(r)
    for p in fixed_window: window_by_video[p["clip_id"]].append(p)
    dense, union = [], []
    for vid in sorted(allowed):
        rows = rows_by_video[vid]
        events = select_events(rows, probabilities[[r["row_index"] for r in rows]], minimum_separation_seconds=.08)
        dense.extend(dict(clip_id=vid, native_frame_id=p["frame_id"], **p) for p in events)
        combined = combine_events(events, window_by_video[vid], minimum_separation_seconds=.08)
        for p in combined:
            p["selected_model"] = "dense_context" if p["selected_model"] == "old" else "original_window"
            p["available_models"] = sorted("dense_context" if s == "old" else "original_window" for s in p["available_models"])
            union.append(dict(clip_id=vid, native_frame_id=p["frame_id"], **p))
    for arm, predictions in (("dense_context", dense), ("window_dense_union", union)):
        save(out / arm / "predictions.json", predictions)
        details[arm] = {}
        for label, tolerance in (("strict", 0.), ("tolerance40ms", .04)):
            result = scorer.evaluate(timeline, predictions, tolerance)
            assert result["eligible_positive_events"] == 680
            save(out / arm / (label+".json"), result)
            details[arm][label] = result
    reports = {arm:{label:compact(r) for label,r in bylabel.items()} for arm,bylabel in details.items()}
    changes = {arm:{label:{control:compare(details[arm][label], details[control][label]) for control in controls}
        for label in ("strict", "tolerance40ms")} for arm in ("dense_context", "window_dense_union")}
    save(out / "matched_event_changes.json", changes)
    coverage = read(EXP / "analysis/coverage_diagnosis_v001.json")
    cohorts = {g:{r["event_id"] for r in coverage["event_details"] if g in r["target_groups"]}
        for g in ("no_candidate28", "no_positive_window133", "family10_old_lost16")}
    assert [len(cohorts[g]) for g in cohorts] == [28, 133, 16]
    fp_diagnosis = read(EXP / "analysis/added_fp_diagnosis_v001.json")
    prior_fp = {(r["video_id"], r["frame_id"]):r for r in fp_diagnosis["introduced_false_positives"]}
    assert len(prior_fp) == 70
    targeted = {}
    for arm, bylabel in details.items():
        result = bylabel["strict"]
        missed = set(result["missed_event_ids"])
        outcomes = {(p["clip_id"], p["native_frame_id"]):p for p in result["event_outcomes"]}
        targets = {g:dict(events=len(ids), matched=len(ids-missed), matched_event_ids=sorted(ids-missed),
            missed_event_ids=sorted(ids & missed)) for g,ids in cohorts.items()}
        if arm in ("dense_context", "window_dense_union"):
            exact_fp = [dict(video_id=k[0], frame_id=k[1], assessment=outcomes[k]["assessment"],
                previous_category=prior_fp[k]["category"]) for k in sorted(prior_fp.keys() & outcomes.keys())]
            added = [p for k,p in outcomes.items() if k not in old_keys]
            targets["prior70_added_fp_frames"] = dict(exact_frames_retained=len(exact_fp),
                retained_assessments=dict(Counter(p["assessment"] for p in exact_fp)), retained=exact_fp,
                caveat="Exact frame identity only; non-retention does not prove the erroneous event disappeared")
            targets["new_candidate_output_contribution"] = dict(outputs=len(added),
                assessments=dict(Counter(p["assessment"] for p in added)),
                by_family={f:dict(Counter(p["assessment"] for p in added if p["clip_id"] in {
                    c["clip_id"] for c in timeline["clips"] if c["camera_family"] == f}))
                    for f in sorted({c["camera_family"] for c in timeline["clips"]})},
                predictions=added)
        targeted[arm] = targets
    save(out / "targeted_diagnostics.json", targeted)
    return dict(arms=reports, comparisons=changes, targeted_diagnostics=targeted,
        reused_control_source_sha256=hashes, control_scoring_runs=0, new_scoring_runs=4,
        evaluator_sha256=sha(SCORER), timeline_sha256=sha(TIMELINE), full_video_inference_fps=None,
        official_score=None, independent_generalization=False)


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for s in self.streams: s.write(text)
        return len(text)
    def flush(self):
        for s in self.streams: s.flush()


def execute(dataset, out, prepared_plan):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = read(prepared_plan)
    for path,digest in plan["source_sha256"].items():
        assert sha(ROOT / path) == digest, "Prepared source changed: "+path
    cfg = configuration()
    manifest, index, windows, supervision, folds, allowed = load_dataset(dataset)
    runtime = runtime_modules()
    save(out / "training_plan.json", plan)
    save(out / "effective_configuration.json", cfg)
    save(out / "source_seal.json", dict(source_sha256=plan["source_sha256"],
        dataset_manifest_sha256=sha(dataset / "manifest.json"), torch_version=torch.__version__, python=sys.version))
    with np.load(dataset / "features.npz", allow_pickle=False) as archive:
        sequence, summary = archive["sequence"], archive["summary"]
    assert sequence.shape == (len(index), 15, 23) and summary.shape == (len(index), 44)
    scores = np.full((len(index), 3), np.nan, dtype=np.float32)
    logits = np.full_like(scores, np.nan)
    seen = np.zeros(len(index), dtype=np.int32)
    receipts = []
    for fold in folds:
        held, fold_logits, probabilities, receipt = fit_fold(sequence, summary, windows, supervision, fold, cfg,
            out / "folds" / f"{fold['fold_id']:02d}", runtime)
        scores[held], logits[held] = probabilities, fold_logits
        seen[held] += 1
        receipts.append(receipt)
        status = read(out / "run_status.json")
        status["fits_completed"] = len(receipts)
        save(out / "run_status.json", status)
        print(f"dense fold {fold['fold_id']} complete: {receipt['fit_seconds']:.1f}s", flush=True)
    assert np.all(seen == 1) and np.isfinite(scores).all() and np.isfinite(logits).all()
    np.save(out / "oof_probabilities.npy", scores)
    np.save(out / "oof_logits.npy", logits)
    evaluation = evaluate(out / "evaluation", index, scores, allowed, runtime)
    report = dict(protocol_sha256=sha(PROTOCOL), dataset_manifest_sha256=sha(dataset / "manifest.json"),
        prepared_plan_sha256=sha(prepared_plan), source_seal_sha256=sha(out / "source_seal.json"),
        fits=receipts, training_fits=10, final_full121_fit=False, evaluation=evaluation,
        public15_used=False, protected33_used=False, network_calls=0)
    save(out / "report.json", report)
    save(EXP / "analysis/comparison_v001.json", report)
    print(json.dumps({arm:r["strict"]["metrics"] for arm,r in evaluation["arms"].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-loss", action="store_true")
    action.add_argument("--prepare-plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-plan", type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve existing output; no automatic rerun")
    if args.check_loss:
        result = check_loss_semantics()
        save(args.output, result)
        print(json.dumps(result), flush=True)
        return 0
    if args.prepare_plan:
        result = prepare_plan(args.dataset)
        save(args.output, result)
        print(json.dumps(dict(prepared=True, candidates=result["candidates"], folds=10,
            parameters=result["parameters"], training_fits=0, plan_sha256=sha(args.output))), flush=True)
        return 0
    if args.prepared_plan is None: parser.error("--execute requires --prepared-plan")
    args.output.mkdir(parents=True)
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), command=sys.argv,
        completed=False, exit_code=None, fits_completed=0)
    save(args.output / "run_status.json", status)
    (args.output / "pid.txt").write_text(str(os.getpid())+"\n", encoding="ascii")
    tick, code = time.perf_counter(), 1
    with (args.output / "console.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                execute(args.dataset, args.output, args.prepared_plan)
                code = 0
            except BaseException:
                traceback.print_exc()
            finally:
                status = read(args.output / "run_status.json")
                status.update(completed=code == 0, exit_code=code, ended_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter()-tick)
                save(args.output / "run_status.json", status)
                (args.output / "exit_code.txt").write_text(str(code)+"\n", encoding="ascii")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
