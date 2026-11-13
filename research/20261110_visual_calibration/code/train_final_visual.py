"""Fit the frozen final154 visual model and a121-only OOF scalar on the server."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from prepare_nested_plan import EXP, ROOT, MODEL_SHA, TRAINER_SHA, CALIBRATION, read, lines, sha, save, resolve, validate_probabilities
from train_nested_visual import module, BagObjective, fit_scalar


def verify_plan(args):
    plan = read(args.plan)
    if (plan["schema"] != "final154_visual_plan.v1"
            or plan["training_source_sha256"] != sha(__file__)
            or plan["preparation_source_sha256"] != sha(Path(__file__).with_name("prepare_final_plan.py"))
            or plan["nested_training_source_sha256"] != sha(Path(__file__).with_name("train_nested_visual.py"))
            or plan["nested_preparation_source_sha256"] != sha(Path(__file__).with_name("prepare_nested_plan.py"))
            or plan["protocol_sha256"] != sha(EXP / "final_protocol_v001.json")
            or plan["calibration"]["configuration"] != CALIBRATION):
        raise ValueError("Final154 plan, fixed implementation or protocol changed")
    for info in plan["inputs"].values():
        path = resolve(args.plan.parent, info["name"])
        if sha(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError("Prepared final154 metadata changed: " + info["name"])
    for name in ("old_trainer", "old_model"):
        info = plan["sources"][name]
        if sha(resolve(args.assets_root, info["path"])) != info["sha256"]:
            raise ValueError("Original visual source differs: " + name)
    if plan["sources"]["old_trainer"]["sha256"] != TRAINER_SHA or plan["sources"]["old_model"]["sha256"] != MODEL_SHA:
        raise ValueError("Expected frozen original visual implementation")
    if sha(args.pretrained) != plan["model"]["pretrained_sha256"]:
        raise ValueError("ImageNet initializer differs")
    if (sha(args.outer_probabilities) != plan["calibration"]["outer_probabilities_sha256"]
            or sha(args.outer_probabilities.parent / "run_status.json") != plan["calibration"]["outer_run_status_sha256"]):
        raise ValueError("Original121 OOF calibration evidence differs")
    return plan


def load_inputs(args, plan):
    rows = lines(args.plan.parent / plan["inputs"]["rows"]["name"])
    temporal = np.load(args.plan.parent / plan["inputs"]["time_features"]["name"], allow_pickle=False)
    supervision = read(args.plan.parent / plan["inputs"]["supervision"]["name"])
    stem = np.load(args.cache / "stem_features.npy", mmap_mode="r", allow_pickle=False)
    receipt = read(args.cache / "receipt.json")
    if (receipt["schema"] != "final154_visual_stem_cache.v1" or not receipt["completed"]
            or receipt["plan_sha256"] != sha(args.plan)
            or receipt["base121_stem_sha256"] != plan["base121_cache"]["files"]["stem_features.npy"]["sha256"]
            or receipt["model_source_sha256"] != MODEL_SHA
            or receipt["pretrained_sha256"] != plan["model"]["pretrained_sha256"]
            or receipt["rows"] != 20641 or receipt["development_candidates"] != 17559
            or receipt["former33_candidates"] != 3082 or receipt["dtype"] != "float32"):
        raise ValueError("Final154 stem extraction receipt differs")
    for name in ("stem_features.npy", "rows.jsonl", "time_features.npy"):
        if sha(args.cache / name) != receipt["files"][name]["sha256"]:
            raise ValueError("Final154 cache identity differs: " + name)
    for key in ("rows", "time_features"):
        info = plan["inputs"][key]
        if receipt["files"][info["name"]]["sha256"] != info["sha256"]:
            raise ValueError("Cache metadata differs from plan: " + key)
    if (len(rows) != 20641 or stem.shape != (20641, 3, 2, 96, 3, 3) or stem.dtype != np.float32
            or temporal.shape != (20641, 3, 2) or temporal.dtype != np.float32
            or not np.isfinite(temporal).all() or not np.isin(temporal[:, :, 1], [0, 1]).all()
            or not np.all(temporal[:, 1, 1] == 1) or not np.all(temporal[:, 1, 0] == 0)
            or [r["array_index"] for r in rows] != list(range(20641))):
        raise ValueError("Unexpected final154 feature/time/row layout")
    for start in range(0, len(rows), 256):
        part = stem[start:start+256]
        if not np.isfinite(part).all() or np.any(part[temporal[start:start+256, :, 1] == 0] != 0):
            raise ValueError("Nonfinite features or nonzero absent slots")
    dev, former, members = map(set, (plan["development_videos"], plan["former33_videos"], plan["training_videos"]))
    if (len(dev) != 121 or len(former) != 33 or dev & former or dev | former != members
            or {r["video_id"] for r in rows[:17559]} - dev or {r["video_id"] for r in rows[17559:]} != former):
        raise ValueError("Final154 membership differs")
    events = [(e["video_id"], e["class_id"], e["indices"]) for e in supervision["events"]]
    backgrounds = supervision["background_indices"]
    if len(events) != 1371 or len(backgrounds) != 16275:
        raise ValueError("Final154 supervision count differs")
    for vid, kind, indices in events:
        if kind not in (0, 1) or not indices or any(rows[i]["video_id"] != vid for i in indices):
            raise ValueError("Cross-video or invalid final154 event bag")
    if set(backgrounds) & {i for _, _, indices in events for i in indices}:
        raise ValueError("A known background belongs to an event")
    return stem, temporal, rows, events, backgrounds


def deployment_objective(probabilities, rows, events, backgrounds, development):
    development = set(development)
    original_rows = rows[:17559]
    validate_probabilities(probabilities, original_rows, development)
    values = [probabilities[r["video_id"]][r["physical_candidate_id"]] for r in original_rows]
    bags, classes = [], []
    for vid, kind, members in events:
        if vid in development:
            if any(i >= 17559 for i in members):
                raise ValueError("Former33 row entered121 calibration")
            bags.append(members)
            classes.append(kind)
    negatives = [i for i in backgrounds if rows[i]["video_id"] in development]
    if len(bags) != 1168 or len(negatives) != 13924 or any(i >= 17559 for i in negatives):
        raise ValueError("Calibration must use exactly the original121 supervision")
    return BagObjective(values, bags, classes, negatives)


def execute(args):
    import scipy
    from scipy.optimize import minimize_scalar  # noqa: F401
    plan = verify_plan(args)
    old = module("frozen_final_visual_trainer", resolve(args.assets_root, plan["sources"]["old_trainer"]["path"]))
    if old.TRAINING != plan["model"]["training"]:
        raise ValueError("Frozen16-epoch visual training recipe differs")
    old.configure_torch(args.device)
    model_api = old.module("frozen_final_visual_model", old.MODEL_SOURCE)
    data = load_inputs(args, plan)
    stem, temporal, rows, events, backgrounds = data
    raw = read(args.outer_probabilities)
    objective = deployment_objective(raw, rows, events, backgrounds, plan["development_videos"])
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(schema="final154_visual_fit.v1", completed=False, exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv, device=args.device,
        torch=torch.__version__, scipy=scipy.__version__, plan_sha256=sha(args.plan),
        protocol_sha256=plan["protocol_sha256"], cache_receipt_sha256=sha(args.cache / "receipt.json"),
        visual_training_runs=0, calibration_fits=0, visual_training_videos=154,
        calibration_training_videos=121, training_videos=154, calibration_videos=121,
        recipe_id="VISUAL_CALIBRATED_SCALE_AWARE_FINAL154_V001", former33_used_for_visual_training=True,
        former33_used_for_calibration=False, public15_used=False, external130_used=False,
        training_source_sha256=sha(__file__), model_source_sha256=MODEL_SHA,
        original_trainer_sha256=TRAINER_SHA)
    status_path = args.output / "run_status.json"
    save(status_path, status)
    started = time.perf_counter()
    try:
        status["stage"] = "calibration121_oof"
        status["calibration_fits_started"] = 1
        save(status_path, status)
        fitted = fit_scalar(objective)
        status["calibration_fits"] = 1
        status["calibration_numerical_convergence"] = fitted["numerical_convergence"]
        status["stage"] = "visual_final154"
        status["visual_training_runs_started"] = 1
        save(status_path, status)
        fold = dict(fold_id="final154", held_out_family=None,
                    train_video_ids=plan["training_videos"], held_out_video_ids=[])
        def progress(entry):
            status.update(current_epoch=entry["epoch"], last_epoch_seconds=entry["epoch_seconds"],
                          current_fit_elapsed_seconds=entry["fit_elapsed_seconds"])
            save(status_path, status)
        prediction, fit = old.fit_fold(args.output / "visual", fold, data, model_api, args.pretrained,
                                       args.device, progress_callback=progress)
        if prediction or fit["train_events"] != 1371 or fit["heldout_candidates"] != 0 or fit["optimizer_steps"] != 464:
            raise ValueError("Unexpected final154 training or evaluation count")
        status["visual_training_runs"] = 1
        calibration = dict(schema="final154_visual_calibration.v1", **fitted,
            model_sha256=fit["model_sha256"], classes=list(model_api.CLASSES), training=old.TRAINING,
            pretrained_sha256=plan["model"]["pretrained_sha256"],
            fit_scope="development121_outer_oof_only;visual_model_final154",
            recipe_id="VISUAL_CALIBRATED_SCALE_AWARE_FINAL154_V001", calibration_videos=121,
            visual_training_videos=154, calibration_sources=10, raw_oof_sha256=sha(args.outer_probabilities),
            plan_sha256=sha(args.plan), configuration=CALIBRATION,
            calibration_event_bags=1168, calibration_backgrounds=13924,
            former33_used_for_calibration=False, external130_used=False, public15_used=False,
            distribution_transfer=plan["calibration"]["distribution_transfer"])
        save(args.output / "calibration.json", calibration)
        status.update(completed=True, exit_code=0, stage="complete", model_sha256=fit["model_sha256"],
            checkpoint="visual/model.pt", calibration_sha256=sha(args.output / "calibration.json"),
            fit_receipt_sha256=sha(args.output / "visual/fit_receipt.json"),
            calibration="calibration.json", a=fitted["a"], fit=fit, evaluation_runs=0)
    except BaseException:
        status["error"] = traceback.format_exc()
        raise
    finally:
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started)
        save(status_path, status)
        (args.output / "exit_code.txt").write_text(str(status["exit_code"]) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ROOT)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--outer-probabilities", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute or not args.device.startswith("cuda"):
        parser.error("All actual visual/scalar fitting requires the authorized server CUDA --execute path")
    execute(args)


if __name__ == "__main__":
    main()
