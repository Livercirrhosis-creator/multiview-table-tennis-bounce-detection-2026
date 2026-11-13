"""Thirty source-isolated visual fits and ten scalar fits, on explicit server execution."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from prepare_nested_plan import (EXP, ROOT, SCHEMA, CALIBRATION, TRAINER_SHA, MODEL_SHA,
    read, sha, save, resolve, metadata, build_outer_folds, source_bins, validate_probabilities)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def calibrate(probabilities, a):
    values = np.asarray(probabilities, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()
            or (values < 0).any() or not np.allclose(values.sum(1), 1., atol=1e-6, rtol=0)
            or not np.isfinite(a) or not 0 <= a <= 1):
        raise ValueError("Invalid probabilities or fixed softening scale")
    # Preserve the serialized legacy probabilities exactly at the identity
    # endpoint. The fitting objective still uses normalized log probabilities.
    if a == 1:
        return values.copy()
    z = a * np.log(np.maximum(values, CALIBRATION["probability_floor"]))
    z -= z.max(axis=1, keepdims=True)
    output = np.exp(z)
    return output / output.sum(axis=1, keepdims=True)


class BagObjective:
    """Original event/background loss on OOF probabilities; no parameter fitting here."""
    def __init__(self, probabilities, bags, classes, backgrounds):
        values = np.asarray(probabilities, dtype=np.float64)
        calibrate(values, 1.)  # Validate without changing the stored raw probabilities.
        if (not bags or not backgrounds or len(bags) != len(classes)
                or not all(c in (0, 1) for c in classes) or any(not b for b in bags)):
            raise ValueError("Expected complete event bags and legal backgrounds")
        width = max(map(len, bags))
        self.members = np.zeros((len(bags), width), dtype=np.int64)
        self.valid = np.zeros_like(self.members, dtype=bool)
        for i, bag in enumerate(bags):
            self.members[i, :len(bag)], self.valid[i, :len(bag)] = bag, True
        self.classes = np.asarray(classes, dtype=np.int64)
        self.backgrounds = np.asarray(backgrounds, dtype=np.int64)
        all_indices = [i for bag in bags for i in bag] + list(backgrounds)
        if min(all_indices) < 0 or max(all_indices) >= len(values) or set(backgrounds) & {i for bag in bags for i in bag}:
            raise ValueError("Invalid or conflicting objective membership")
        self.logp = np.log(np.maximum(values, CALIBRATION["probability_floor"]))

    def terms(self, a):
        if not np.isfinite(a) or not 0 <= a <= 1:
            raise ValueError("Expected a in the frozen [0,1] interval")
        scaled = a * self.logp
        logq = scaled - np.logaddexp.reduce(scaled, axis=1, keepdims=True)
        target = logq[self.members, self.classes[:, None]]
        target_mil = -np.where(self.valid, target, -np.inf).max(axis=1)
        allowed_logq = np.logaddexp(target, logq[self.members, 2])
        competing = np.where(self.valid, -allowed_logq, 0.).sum(axis=1) / self.valid.sum(axis=1)
        event = float((target_mil + competing).mean())
        background = float(-logq[self.backgrounds, 2].mean())
        return dict(event_loss=event, background_loss=background, loss=.5 * (event + background))

    def __call__(self, a):
        result = self.terms(float(a))["loss"]
        if not np.isfinite(result):
            raise ValueError("Nonfinite scalar objective")
        return result


def objective_for_outer(oof, rows, events, backgrounds, outer):
    train = set(outer["train_video_ids"])
    indices = [i for i, r in enumerate(rows) if r["video_id"] in train]
    local = {index: i for i, index in enumerate(indices)}
    probabilities = [oof[rows[i]["video_id"]][rows[i]["physical_candidate_id"]] for i in indices]
    bags, classes = [], []
    for vid, kind, members in events:
        if vid in train:
            bags.append([local[i] for i in members])
            classes.append(kind)
    negatives = [local[i] for i in backgrounds if rows[i]["video_id"] in train]
    if len(bags) != outer["train_counts"]["events"] or len(negatives) != outer["train_counts"]["backgrounds"]:
        raise ValueError("Calibration supervision differs from the outer training side")
    return BagObjective(probabilities, bags, classes, negatives)


def fit_scalar(objective):
    # Imported only in the actual server fitting path. Local --check never calls this.
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(objective, bounds=(0., 1.), method="bounded",
                             options=dict(xatol=CALIBRATION["xatol"], maxiter=CALIBRATION["maxiter"]))
    if not np.isfinite(result.x) or not 0 <= result.x <= 1:
        raise ValueError("Scalar minimizer returned an invalid location")
    candidates = [dict(kind=kind, a=float(a), **objective.terms(float(a)))
                  for kind, a in (("lower_endpoint", 0.), ("upper_endpoint", 1.), ("bounded_solution", result.x))]
    if not all(np.isfinite(c["loss"]) for c in candidates):
        raise ValueError("Scalar candidate objective is nonfinite")
    chosen = min(candidates, key=lambda c: (c["loss"], c["a"]))
    a = chosen["a"]
    return dict(a=a, temperature=None if a == 0 else 1. / a,
                selected=chosen, candidates=candidates, tie_rule="minimum loss then smaller a",
                solver=dict(success=bool(result.success), message=str(result.message),
                            nfev=int(result.nfev), nit=int(result.nit), x=float(result.x), fun=float(result.fun)),
                numerical_convergence=bool(result.success), global_optimum_guaranteed=False)


def load_plan(args):
    plan = read(args.plan)
    if (plan["schema"] != SCHEMA or plan["calibration"] != CALIBRATION
            or plan["training_source_sha256"] != sha(__file__)
            or plan["preparation_source_sha256"] != sha(Path(__file__).with_name("prepare_nested_plan.py"))
            or plan["protocol_sha256"] != sha(EXP / "protocol_v001.json")):
        raise ValueError("Frozen nested plan, code or protocol differs")
    rows, supervision, split, allowed = metadata(args.assets_root, plan["files"])
    if build_outer_folds(rows, supervision, split, allowed) != plan["outer_folds"]:
        raise ValueError("Nested membership differs from deterministic source-count bins")
    outer_status = read(args.outer_probabilities.parent / "run_status.json")
    if (sha(args.outer_probabilities) != plan["outer_reference"]["probabilities_sha256"]
            or sha(args.outer_probabilities.parent / "run_status.json") != plan["outer_reference"]["run_status_sha256"]
            or not outer_status["completed"] or outer_status["training_runs"] != 10
            or any(outer_status[k] for k in ("protected33_used", "public15_used", "external130_used"))):
        raise ValueError("Frozen original outer probabilities or receipt differs")
    raw = read(args.outer_probabilities)
    validate_probabilities(raw, rows, allowed)
    if sha(args.cache / "receipt.json") != plan["cache_reference"]["receipt_sha256"]:
        raise ValueError("Frozen reusable stem cache receipt differs")
    if sha(args.pretrained) != plan["cache_reference"]["pretrained_sha256"]:
        raise ValueError("ImageNet initialization differs")
    return plan, raw, allowed


def execute(args):
    # Fail before creating any training output when the existing server lacks SciPy.
    import scipy
    from scipy.optimize import minimize_scalar  # noqa: F401
    plan, raw, allowed = load_plan(args)
    old = module("frozen_adapted_visual_training", resolve(args.assets_root, plan["files"]["old_trainer"]["path"]))
    if old.TRAINING != plan["original_visual_training"]:
        raise ValueError("Original visual training recipe differs")
    old.configure_torch(args.device)
    model_api = old.module("frozen_adapted_visual_model", old.MODEL_SOURCE)
    data = old.load_inputs(args.cache, args.assets_root, model_api, args.pretrained)
    stem, temporal, rows, events, backgrounds, _, input_hashes = data
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(schema="nested_visual_calibration_run.v1", completed=False, exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv, device=args.device,
        torch=torch.__version__, scipy=scipy.__version__, python=sys.executable,
        plan_sha256=sha(args.plan), protocol_sha256=plan["protocol_sha256"], input_sha256=input_hashes,
        training_source_sha256=sha(__file__), original_trainer_sha256=TRAINER_SHA,
        original_model_sha256=MODEL_SHA, visual_training_runs=0, visual_training_runs_started=0,
        calibration_fits=0, calibration_fits_started=0, outer_models_reused=10,
        protected33_used=False, public15_used=False, external130_used=False, outer_results=[])
    status_path = args.output / "run_status.json"
    save(status_path, status)
    started = time.perf_counter()
    try:
        original_copy = args.output / "raw_outer_probabilities_by_video.json"
        shutil.copyfile(args.outer_probabilities, original_copy)
        status["raw_outer_predictions_sha256"] = sha(original_copy)
        calibrated = {}
        for outer in plan["outer_folds"]:
            oid = outer["fold_id"]
            folder = args.output / f"outer_{oid:02d}"
            folder.mkdir()
            oof, inner_receipts = {}, []
            for inner in outer["inner_folds"]:
                iid = inner["fold_id"]
                status.update(current_outer_fold=oid, current_inner_fold=iid, current_epoch=0,
                              stage="visual_training")
                status["visual_training_runs_started"] += 1
                save(status_path, status)
                def progress(entry):
                    status.update(current_epoch=entry["epoch"], last_epoch_seconds=entry["epoch_seconds"],
                                  current_fit_elapsed_seconds=entry["fit_elapsed_seconds"])
                    save(status_path, status)
                destination = folder / f"inner_{iid:02d}"
                prediction, receipt = old.fit_fold(destination, inner,
                    (stem, temporal, rows, events, backgrounds), model_api, args.pretrained, args.device,
                    progress_callback=progress)
                if set(prediction) != set(inner["held_out_video_ids"]) or oof.keys() & prediction.keys():
                    raise ValueError("Inner holdout prediction coverage overlaps or differs")
                if (receipt["train_events"] != inner["train_counts"]["events"]
                        or receipt["heldout_candidates"] != inner["held_out_counts"]["candidates"]):
                    raise ValueError("Actual inner training/heldout counts differ")
                oof.update(prediction)
                inner_receipts.append(dict(inner_fold=iid, fit=receipt,
                    checkpoint=(destination / "model.pt").relative_to(args.output).as_posix(),
                    checkpoint_sha256=receipt["model_sha256"],
                    fit_receipt_sha256=sha(destination / "fit_receipt.json"),
                    train_video_ids=inner["train_video_ids"], held_out_video_ids=inner["held_out_video_ids"]))
                status["visual_training_runs"] += 1
                save(status_path, status)
            outer_train = set(outer["train_video_ids"])
            validate_probabilities(oof, [r for r in rows if r["video_id"] in outer_train], outer_train)
            oof_path = folder / "inner_oof_probabilities_by_video.json"
            save(oof_path, oof)
            status["stage"] = "scalar_calibration"
            status["calibration_fits_started"] += 1
            save(status_path, status)
            tick = time.perf_counter()
            objective = objective_for_outer(oof, rows, events, backgrounds, outer)
            fitted = fit_scalar(objective)
            calibration = dict(schema="nested_visual_scalar.v1", outer_fold=oid,
                outer_held_out_family=outer["held_out_family"], calibration_fit_seconds=time.perf_counter()-tick,
                train_video_ids=outer["train_video_ids"], held_out_video_ids=outer["held_out_video_ids"],
                calibration_counts=outer["train_counts"], configuration=CALIBRATION, **fitted,
                inner_oof_sha256=sha(oof_path), inner_fits=inner_receipts,
                outer_model_sha256=next(f["model_sha256"] for f in plan["outer_reference"]["model_fits"] if f["fold"] == oid),
                raw_outer_predictions_sha256=status["raw_outer_predictions_sha256"],
                plan_sha256=status["plan_sha256"], outer_heldout_used_for_fitting=False)
            save(folder / "calibration.json", calibration)
            status["calibration_fits"] += 1
            for vid in outer["held_out_video_ids"]:
                if vid in calibrated:
                    raise ValueError("Repeated outer heldout probability video")
                original = raw[vid]
                if original:
                    values = calibrate(list(original.values()), fitted["a"])
                    calibrated[vid] = {key: q.tolist() for key, q in zip(original, values)}
                else:
                    calibrated[vid] = {}
            status["outer_results"].append(dict(outer_fold=oid, a=fitted["a"],
                temperature=fitted["temperature"], calibration=(folder / "calibration.json").relative_to(args.output).as_posix(),
                calibration_sha256=sha(folder / "calibration.json"), inner_oof_sha256=sha(oof_path),
                inner_fits=inner_receipts, held_out_video_ids=outer["held_out_video_ids"]))
            save(status_path, status)
            print(json.dumps(dict(outer_fold=oid, a=fitted["a"], calibration_loss=fitted["selected"]["loss"],
                visual_training_runs=status["visual_training_runs"], calibration_fits=status["calibration_fits"])), flush=True)
        validate_probabilities(calibrated, rows, allowed)
        if status["visual_training_runs"] != 30 or status["calibration_fits"] != 10:
            raise ValueError("Incomplete nested fitting counts")
        prediction_path = args.output / "probabilities_by_video.json"
        save(prediction_path, calibrated)
        status.update(completed=True, exit_code=0, stage="complete", candidates=len(rows),
            outer_heldout_videos=len(calibrated), predictions_sha256=sha(prediction_path), total_fits=40)
    except BaseException:
        status["error"] = traceback.format_exc()
        raise
    finally:
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started)
        save(status_path, status)
        (args.output / "exit_code.txt").write_text(str(status["exit_code"]) + "\n", encoding="ascii")


def check(args):
    """Random algebra/gradient and synthetic grouping checks only; zero fits."""
    torch.set_num_threads(2)
    rng = np.random.default_rng(20261110)
    values = rng.dirichlet([.7, .9, .8], size=12)
    values[0] = [0., 1e-31, 1.]
    bags, classes, backgrounds = [[0, 1, 2], [3, 4], [5, 6]], [0, 1, 0], list(range(7, 12))
    objective = BagObjective(values, bags, classes, backgrounds)
    old = module("frozen_loss_algebra_check", args.assets_root / "experiments/20261105_coverage_aware_events/code/train_dense_context.py")
    maximum_difference = 0.
    for a in (0., .37, 1.):
        logits = torch.from_numpy(a * np.log(np.maximum(values, 1e-30)))
        target, other = old.event_loss_terms(logits, bags, classes)
        reference = .5 * ((target + other).mean() - torch.log_softmax(logits[backgrounds], -1)[:, 2].mean())
        maximum_difference = max(maximum_difference, abs(float(reference) - objective(a)))
        assert maximum_difference < 1e-12
    a = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
    logits = a * torch.from_numpy(np.log(np.maximum(values, 1e-30)))
    target, other = old.event_loss_terms(logits, bags, classes)
    reference = .5 * ((target + other).mean() - torch.log_softmax(logits[backgrounds], -1)[:, 2].mean())
    reference.backward()
    finite_difference = (objective(.370001) - objective(.369999)) / .000002
    assert abs(float(a.grad) - finite_difference) < 1e-7
    assert np.array_equal(calibrate(values, 0.), np.full_like(values, 1. / 3.))
    assert np.allclose(calibrate(values, 1.), values, atol=1e-15, rtol=0)
    helper_path = EXP / "versions/v001/participant/pingpang/participant/visual_calibration_v001/calibration.py"
    helper = module("runtime_calibration_algebra_check", helper_path)
    assert all(np.array_equal(calibrate(values, a), helper.calibrate(values, a)) for a in (0., .37, 1.))
    families = {f"family_{i:02d}": (1000 if i < 2 else 500-i*19) for i in range(9)}
    order, bins = source_bins(families)
    assert order[0:2] == ["family_00", "family_01"] and all(len(b["families"]) == 3 for b in bins)
    assert sorted(f for b in bins for f in b["families"]) == sorted(families)
    assert source_bins(dict(reversed(list(families.items())))) == (order, bins)
    report = dict(completed=True, scope="random algebra/gradient and synthetic grouping only",
        visual_training_runs=0, calibration_fits=0, server_connections=0,
        maximum_loss_difference=maximum_difference, gradient_absolute_difference=abs(float(a.grad)-finite_difference),
        checks=["original_bag_loss_equivalent", "analytic_vs_finite_difference_gradient",
                "a0_uniform", "a1_raw_up_to_fixed_floor", "runtime_calibration_exact", "stable_three_source_bins"],
        training_source_sha256=sha(__file__), preparation_source_sha256=sha(Path(__file__).with_name("prepare_nested_plan.py")),
        runtime_helper_sha256=sha(helper_path))
    if args.check_output:
        if args.check_output.exists():
            raise FileExistsError(args.check_output)
        args.check_output.parent.mkdir(parents=True, exist_ok=True)
        save(args.check_output, report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ROOT)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--outer-probabilities", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-output", type=Path)
    args = parser.parse_args()
    if args.check:
        if args.execute:
            parser.error("--check and --execute are exclusive")
        check(args)
        return
    if not args.execute or any(getattr(args, key) is None for key in ("cache", "pretrained", "outer_probabilities", "plan", "output")):
        parser.error("Real server fits require --execute --cache --pretrained --outer-probabilities --plan --output")
    if not args.device.startswith("cuda"):
        parser.error("This experiment requires the authorized server CUDA training path")
    execute(args)


if __name__ == "__main__":
    main()
