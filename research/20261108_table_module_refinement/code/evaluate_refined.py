"""Seal two refined-table post-gate arms, then score only those two arms.

Reuses existing no-table and original table-LOFO scores. Uses development121
only, with no fitting, model inference, public videos or protected data access.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
import time
import traceback

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PREVIOUS = ROOT/"experiments/20261108_table_prior_potential"
sys.path.insert(0, str(PREVIOUS/"code"))
import evaluate_table_predictions as old

h = old.h
read, lines, save, sha, require = h.read, h.lines, h.save, h.sha, h.require
BUNDLE = EXP/"runs/packet_v001/bundle"
TRAINER = EXP/"code/train_table_refined.py"
trainer = h.module_from_path("table_refined_training_provenance", TRAINER)
OLD_RUN = PREVIOUS/"runs/table_prediction_evaluation_v001"


def validate_returned(run_root, evidence, inputs):
    """Verify the refinement contract and actual returned assets, without inference."""
    def tracked(path, expected=None):
        path = Path(path)
        digest = sha(path)
        require(expected is None or digest == expected, "File SHA mismatch: "+str(path))
        inputs[h.path_key(path)] = digest
        return path

    def referenced(folder, ref):
        return tracked(trainer.inside(folder, ref["path"]), ref["sha256"])

    allowed = set(read(tracked(h.WHITELIST))["video_ids"])
    preliminary_plan = read(tracked(BUNDLE/"training_plan.json"))
    preliminary_data = read(referenced(BUNDLE, preliminary_plan["data_manifest"]))
    require(len(allowed) == 121 and set(evidence) == allowed
            and len(preliminary_plan["expected_video_ids"]) == 121
            and set(preliminary_plan["expected_video_ids"]) == allowed,
            "Development121 membership mismatch")
    for role in ("training_rows", "anchor_rows"):
        require({r["video_id"] for r in preliminary_data[role]} == allowed,
                "Bundle contains videos outside the allowed development pool")
    # This checks all input/image hashes, source-isolated folds, the exact
    # augmentation/sampler settings and the original 7284-update budget.
    plan, data, weight, provenance = trainer.validate(BUNDLE.resolve())
    tracked(weight, trainer.COCO_SHA)
    tracked(TRAINER, provenance["script_sha256"])
    tracked(BUNDLE/"code/train_table_refined.py", provenance["script_sha256"])
    training, anchors = data["training_rows"], data["anchor_rows"]
    sample_index = {(v["video_id"], s["target_relative_seconds"]): s
                    for v in evidence.values() for s in v["samples"]}
    require(len(sample_index) == 363, "Expected three native anchors per video")
    for row in anchors:
        sample = sample_index[(row["video_id"], row["target_relative_seconds"])]
        require(row["frame_id"] == sample["frame_id"] and row["pts_seconds"] == sample["pts_seconds"]
                and row["boxes_original"] == sample["valid_boxes"], "Fixed anchor identity or diagnostic GT differs")
        require(all(row[k] == evidence[row["video_id"]][k] for k in ("width", "height")),
                "Anchor dimensions differ")
    batch = read(tracked(run_root/"batch_receipt.json"))
    require(batch == read(tracked(run_root/"run_status.json")), "Final batch/status receipts differ")
    require(tracked(run_root/"exit_code.txt").read_text().strip() == "0", "Training process exit code differs")
    require(batch["schema"] == "table_source_batch.v1" and batch["state"] == "complete"
            and batch["exit_code"] == 0 and batch["execute"] is True, "Training batch is incomplete")
    require(batch["completed_fits"] == batch["model_training_runs"] == 11
            and batch["completed_epochs"] == 132 and batch["optimizer_updates"] == 7284,
            "Unexpected fit, epoch or update count")
    require(batch["model_predictions"] == 363 and batch["baseline_coco_predictions"] == 0
            and batch["full_fit_evaluation"] is False and batch["anchor_iou_used_for_selection"] is False
            and batch["final_epoch"] == 12, "Unexpected prediction or selection scope")
    require("baseline_coco" not in batch and not (run_root/"baseline_coco").exists(),
            "Refinement must not rerun the COCO prediction control")
    require(batch["inputs"] == provenance, "Batch provenance differs from the frozen verified bundle")
    settings = dict(**trainer.FIXED, optimizer="AdamW", image_size=320, num_classes=91,
        table_label=67, trainable_backbone_layers=6, frozen_batchnorm_statistics=True,
        batchnorm_affine_trainable=True, augmentation=trainer.REFINEMENT["augmentation"],
        sampling=trainer.REFINEMENT["sampling"], scheduler=None, amp=False, data_loader_workers=0)
    require(batch["training_settings"] == settings and batch["strict_cpu_initialization_passed"] is True,
            "Actual refinement training settings differ")
    batch_fits = {f["fit_id"]: f for f in batch["fits"]}
    require(len(batch_fits) == len(batch["fits"]) == 11, "Duplicate or missing fit receipts")

    def prediction_file(folder, receipt, expected):
        rows = lines(referenced(folder, receipt))
        require(len(rows) == len(expected) == receipt["rows"], "Anchor prediction count differs")
        index = {(r["video_id"], r["target_relative_seconds"]): r for r in rows}
        require(len(index) == len(rows), "Duplicate predicted anchor")
        for anchor in expected:
            row = index[(anchor["video_id"], anchor["target_relative_seconds"])]
            require(all(row[k] == anchor[k] for k in ("video_id", "camera_family", "frame_id",
                        "pts_seconds", "target_relative_seconds", "width", "height")), "Anchor identity differs")
            require(row["image_sha256"] == trainer.image_ref(anchor)["sha256"]
                    and row["diagnostic_only"] is True
                    and row["gt_boxes_count"] == len(anchor["boxes_original"]), "Anchor provenance/scope differs")
            require(math.isfinite(row["score"]) and 0 <= row["score"] <= 1, "Invalid table score")
            if row["box"] is None:
                require(row["box_320"] is None and row["score"] == 0.0, "Missing box has a score")
            else:
                require(len(row["box"]) == len(row["box_320"]) == 4 and
                        all(math.isfinite(v) for v in row["box"]+row["box_320"]), "Invalid predicted box")
                expected_box = [row["box_320"][k]*(row["width"] if k % 2 == 0 else row["height"])/320
                                for k in range(4)]
                require(all(abs(a-b) < 1e-9 for a, b in zip(row["box"], expected_box)), "Inverse320 transform differs")
            iou = row["gt_best_iou"]
            require((iou is None and not anchor["boxes_original"]) or
                    (bool(anchor["boxes_original"]) and iou is not None and math.isfinite(iou) and 0 <= iou <= 1),
                    "Invalid final-anchor IoU diagnostic")
        require(receipt["diagnostic_only"] is True and receipt["used_for_selection"] is False,
                "Anchor diagnostics were used for selection")
        return rows

    fit_specs = [(f["fold_id"], f["held_out_family"], set(f["train_video_ids"]),
                  set(f["validation_video_ids"]), run_root/"folds"/f["fold_id"]) for f in plan["folds"]]
    fit_specs.append(("full121", None, allowed, set(), run_root/"full121"))
    predictions, models, total_updates = [], {}, 0
    for fit_id, family, train_ids, val_ids, folder in fit_specs:
        receipt = read(tracked(folder/"receipt.json"))
        require(receipt == batch_fits[fit_id] and receipt["state"] == "complete", "Fit receipt differs")
        require(receipt["seed"] == receipt["seed_per_fit"] == trainer.FIXED["seed"]
                and receipt["complete_epochs"] == receipt["epochs_per_fit"] == receipt["final_epoch"] == 12
                and receipt["fixed_final_epoch"] is True and receipt["anchor_rows_added_to_training"] is False,
                "Fit seed, epochs or checkpoint selection changed")
        require(all(receipt[k] == v for k, v in provenance.items()), "Fit provenance changed")
        rows = [r for r in training if r["video_id"] in train_ids]
        val_rows = [r for r in anchors if r["video_id"] in val_ids]
        source_counts = dict(Counter(r["camera_family"] for r in rows))
        require(receipt["training_source_image_counts"] == source_counts, "Sampler counts include another source")
        def members(items):
            return [dict(video_id=r["video_id"], frame_id=r["frame_id"],
                         image_sha256=trainer.image_ref(r)["sha256"]) for r in items]
        membership = read(tracked(folder/"membership.json", receipt["membership_sha256"]))
        require(membership == dict(fit_id=fit_id, held_out_family=family,
            train_video_ids=sorted(train_ids), validation_video_ids=sorted(val_ids),
            training_rows=members(rows), anchor_prediction_rows=members(val_rows)), "Fit membership differs")
        updates_per_epoch = math.ceil(len(rows)/trainer.FIXED["batch_size"])
        require(receipt["training_images"] == len(rows) and receipt["training_videos"] == len(train_ids)
                and receipt["anchor_predictions_planned"] == len(val_rows)
                and receipt["optimizer_updates"] == plan["planned_optimizer_steps"][fit_id] == 12*updates_per_epoch,
                "Fit draw/update counts differ")
        epochs = lines(tracked(folder/"epochs.jsonl", receipt["epochs_sha256"]))
        require(len(epochs) == 12, "Epoch log count differs")
        for epoch, item in enumerate(epochs, 1):
            require(item["epoch"] == epoch and item["batches"] == updates_per_epoch
                    and item["samples"] == len(rows) and item["optimizer_updates"] == epoch*updates_per_epoch
                    and math.isfinite(item["loss"]), "Epoch log differs")
            draws = item["source_draws"]
            require(set(draws) <= set(source_counts) and all(type(n) is int and n > 0 for n in draws.values())
                    and sum(draws.values()) == len(rows)
                    and 1 <= item["unique_training_images"] <= len(rows), "Weighted source draws differ")
        checkpoint = referenced(folder, receipt["model"])
        require(checkpoint.stat().st_size == receipt["model"]["bytes"], "Model file size differs")
        models[fit_id] = dict(path=h.path_key(checkpoint), **{k: receipt["model"][k] for k in ("sha256", "bytes")})
        total_updates += receipt["optimizer_updates"]
        if val_rows:
            predictions.extend(prediction_file(folder, receipt["anchor_predictions"], val_rows))
        else:
            require(receipt["anchor_predictions"] is None and not (folder/"anchor_predictions.jsonl").exists(),
                    "Full121 must not supply in-sample predictions")
    require(total_updates == 7284 and len(predictions) == 363, "Returned refinement totals differ")
    require(len({(r["video_id"], r["target_relative_seconds"]) for r in predictions}) == 363,
            "Held-out predictions contain duplicates")
    return predictions, dict(fits=11, complete_epochs=132, optimizer_updates=total_updates,
        held_out_fits=10, held_out_predictions=363, baseline_coco_predictions=0,
        full121_fit_evaluated=False, provenance=provenance, environment=batch["environment"], models=models,
        training_seconds=batch["elapsed_seconds"], training_settings=settings)


def validate_score(score, predictions, videos):
    require(score["eligible_positive_events"] == 680 and score["tolerance_seconds"] == 0.0,
            "Reused score conditions changed")
    scored = defaultdict(list)
    for event in score["event_outcomes"]:
        scored[event["clip_id"]].append(event)
    require(set(scored) <= set(videos), "Score includes videos outside development121")
    for row in predictions:
        require(h.identity(scored[row["video_id"]]) == h.identity(row["events"]), "Reused score/output mismatch")
    require(score["total_predictions"] == sum(len(r["events"]) for r in predictions), "Scored prediction count differs")


def comparison(before, after, rows):
    result = h.difference(before, after)
    result.update(h.fp_changes(before, after), per_source=h.source_changes(before, after),
                  removed_baseline_predictions=old.removed_assessments(before, rows))
    tp = before["metrics_on_evaluable_regions"]["tp"]
    result["baseline_tp_retention"] = 1-result["lost"]/tp if tp else None
    result["improved_source_groups"] = sum(r["delta"]["f1"] > 1e-12 for r in result["per_source"])
    result["declined_source_groups"] = sum(r["delta"]["f1"] < -1e-12 for r in result["per_source"])
    result["unchanged_source_groups"] = len(result["per_source"])-result["improved_source_groups"]-result["declined_source_groups"]
    return result


def prior_scope(values, families):
    return dict(usable_videos=sum(p["usable"] for p in values.values()), total_videos=len(values),
        usable_fraction=sum(p["usable"] for p in values.values())/len(values),
        reasons=dict(Counter(p["reason"] for p in values.values())),
        per_source={family: dict(videos=sum(f == family for f in families.values()),
            usable_videos=sum(values[v]["usable"] for v, f in families.items() if f == family))
                    for family in sorted(set(families.values()))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=EXP/"runs/evaluation_v001")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), command=sys.argv,
        completed=False, exit_code=1, scoring_runs=0, scoring_runs_completed=0, training_fits=0,
        image_model_forwards=0, ssh_connections=0, protected33_used=False, public15_used=False,
        phase="validating_returned_refinement")
    save(args.output/"run_status.json", status)

    def log(value):
        message = h.json.dumps(value, ensure_ascii=False, allow_nan=False)
        print(message, flush=True)
        with (args.output/"console.log").open("a", encoding="utf-8") as stream:
            stream.write(message+"\n")

    try:
        inputs = {}
        previous_seal = read(OLD_RUN/"prediction_seal.json")
        config = h.prior_api.Config()
        require(previous_seal["prior_config"] == config.__dict__, "Fixed spatial rule changed")
        evidence_path = PREVIOUS/"runs/table_evidence_v001/table_evidence.json"
        videos_path = h.RESERVOIR/"runtime/videos.json"
        timeline_path = h.RESERVOIR/"training/timeline.json"
        baseline_paths = dict(MASKED=h.GAP, CHANNEL=h.ADAPT/"runs/evaluation_v001/paired_enriched")
        fixed_inputs = [evidence_path, videos_path, timeline_path, h.WHITELIST]
        fixed_inputs += [folder/name for folder in baseline_paths.values()
                         for name in ("predictions_by_video.json", "strict.json")]
        for path in fixed_inputs:
            key = h.path_key(path)
            inputs[key] = sha(path)
            require(inputs[key] == previous_seal["input_sha256"][key], "Prior comparison input changed: "+key)
        # Only hash the event timeline here. Its first parse is after prediction sealing.
        evidence = {r["video_id"]: r for r in read(evidence_path)["videos"]}
        table_rows, training_receipt = validate_returned(args.run_root.resolve(), evidence, inputs)
        training_plan = read(BUNDLE/"training_plan.json")
        protocol_path = EXP/"protocol_v001.json"
        inputs[h.path_key(protocol_path)] = sha(protocol_path)
        require(inputs[h.path_key(protocol_path)] == training_plan["source_protocol_sha256"],
                "Training plan differs from its frozen source protocol")
        videos = {v["clip_id"]: v for v in read(videos_path)["clips"]}
        require(set(videos) == set(evidence) and len(videos) == 121, "Runtime video scope differs")
        priors = old.build_priors(table_rows, evidence, videos, config)
        families = {r["video_id"]: r["camera_family"] for r in table_rows}
        baselines = {chain: read(folder/"predictions_by_video.json") for chain, folder in baseline_paths.items()}
        arms, receipts = {}, {}
        for chain, rows in baselines.items():
            require(len(rows) == 121 and {r["video_id"] for r in rows} == set(videos), "Baseline scope differs")
            arm = "REFINED/"+chain
            arms[arm], receipts[arm] = [], []
            for row in sorted(rows, key=lambda r: r["video_id"]):
                vid = row["video_id"]
                selected, receipt = h.prior_api.select(row["events"], priors[vid], config)
                require(all(e in row["events"] for e in selected), "Post-gate changed retained event fields")
                if not priors[vid]["usable"]:
                    require(selected == row["events"], "Neutral fallback changed the baseline")
                arms[arm].append(dict(video_id=vid, events=selected))
                receipts[arm].append(dict(video_id=vid, **receipt))
            save(args.output/arm/"predictions_by_video.json", arms[arm])
            save(args.output/arm/"prior_receipts.json", receipts[arm])
            with (args.output/arm/"predictions.jsonl").open("x", encoding="utf-8") as stream:
                for row in arms[arm]:
                    for event in row["events"]:
                        stream.write(h.json.dumps(dict(event, video_id=row["video_id"]), allow_nan=False)+"\n")
        reuse_paths = [OLD_RUN/"prediction_seal.json", OLD_RUN/"report.json"]
        reuse_paths += [OLD_RUN/"LOFO"/chain/name for chain in baselines
                       for name in ("predictions_by_video.json", "strict.json", "prior_receipts.json")]
        inputs.update({h.path_key(p): sha(p) for p in reuse_paths})
        seal = dict(sealed_utc=datetime.now(timezone.utc).isoformat(),
            prediction_sha256={arm: sha(args.output/arm/"predictions_by_video.json") for arm in arms},
            prior_receipt_sha256={arm: sha(args.output/arm/"prior_receipts.json") for arm in arms},
            input_sha256=inputs, source_sha256={h.path_key(p): sha(p) for p in
                (Path(__file__).resolve(), Path(old.__file__).resolve(), Path(h.__file__).resolve(),
                 h.HELPERS, h.PRIOR, h.SCORER, TRAINER)},
            training_plan_sha256=sha(BUNDLE/"training_plan.json"),
            source_protocol_sha256=training_plan["source_protocol_sha256"], prior_config=config.__dict__,
            refinement=trainer.REFINEMENT, training_receipt=training_receipt,
            event_timeline_parsed_before_prediction_seal=False, scoring_runs_planned=2,
            baseline_scoring_runs_planned=0, full121_predictions_used=False)
        save(args.output/"prediction_seal.json", seal)
        status["phase"] = "predictions_sealed_before_scoring"
        save(args.output/"run_status.json", status)
        log(dict(phase=status["phase"], held_out_anchor_predictions=363, scoring_runs_planned=2))

        for path in fixed_inputs+reuse_paths:
            require(sha(path) == seal["input_sha256"][h.path_key(path)], "Sealed input changed")
        scorer = h.module_from_path("table_refined_fixed_temporal_evaluator", h.SCORER)
        timeline = read(timeline_path)
        require(set(scorer.validate_timeline(timeline)) == set(videos), "Timeline outside development121")
        old_report = read(OLD_RUN/"report.json")
        require(old_report["evaluator_sha256"] == sha(h.SCORER), "Reused scores used another evaluator")
        baseline_scores = {chain: read(folder/"strict.json") for chain, folder in baseline_paths.items()}
        old_scores, old_priors = {}, None
        for chain in baselines:
            validate_score(baseline_scores[chain], baselines[chain], videos)
            folder = OLD_RUN/"LOFO"/chain
            require(sha(folder/"predictions_by_video.json") == previous_seal["prediction_sha256"]["LOFO/"+chain],
                    "Original table predictions differ from their frozen seal")
            old_predictions = read(folder/"predictions_by_video.json")
            require(len(old_predictions) == 121 and {r["video_id"] for r in old_predictions} == set(videos),
                    "Original table result scope differs")
            old_scores[chain] = read(folder/"strict.json")
            validate_score(old_scores[chain], old_predictions, videos)
            require(h.compact(old_scores[chain]) == old_report["results"]["LOFO/"+chain], "Reused summary differs")
            values = {r["video_id"]: r["prior"] for r in read(folder/"prior_receipts.json")}
            require(set(values) == set(videos), "Original prior video scope differs")
            require(old_priors is None or old_priors == values, "Original chains used different table priors")
            old_priors = values
        details, versus_baseline, versus_old_table = {}, {}, {}
        for arm, rows in arms.items():
            require(sha(args.output/arm/"predictions_by_video.json") == seal["prediction_sha256"][arm],
                    "Sealed refined output changed")
            predictions = [dict(**{k: v for k, v in event.items() if k not in ("clip_id", "native_frame_id")},
                                clip_id=row["video_id"], native_frame_id=event["frame_id"])
                           for row in rows for event in row["events"]]
            status["scoring_runs"] += 1
            save(args.output/"run_status.json", status)
            result = scorer.evaluate(timeline, predictions, 0.0)
            status["scoring_runs_completed"] += 1
            require(result["eligible_positive_events"] == 680, "Eligible event count changed")
            save(args.output/arm/"strict.json", result)
            details[arm] = result
            chain = arm.split("/")[1]
            versus_baseline[arm] = comparison(baseline_scores[chain], result, rows)
            versus_old_table[arm] = comparison(old_scores[chain], result, rows)
            save(args.output/"run_status.json", status)
            log(dict(scored_arm=arm, metrics=result["metrics_on_evaluable_regions"]))
        require(status["scoring_runs"] == status["scoring_runs_completed"] == 2, "Expected exactly two new scores")
        report = dict(schema="table_refinement_post_gate.v1",
            baseline={chain: h.compact(score) for chain, score in baseline_scores.items()},
            original_table_lofo={chain: h.compact(score) for chain, score in old_scores.items()},
            results={arm: h.compact(score) for arm, score in details.items()},
            versus_no_table=versus_baseline, versus_original_table_lofo=versus_old_table,
            prior_scope=dict(ORIGINAL_LOFO=prior_scope(old_priors, families), REFINED=prior_scope(priors, families)),
            prior_availability_change=dict(
                newly_usable=sorted(v for v in videos if priors[v]["usable"] and not old_priors[v]["usable"]),
                newly_neutral=sorted(v for v in videos if old_priors[v]["usable"] and not priors[v]["usable"])),
            final_anchor_iou_diagnostic=dict(REFINED=old.box_diagnostic(table_rows),
                ORIGINAL_LOFO=old_report["final_anchor_iou_diagnostic"]["LOFO"]),
            training_receipt=training_receipt, scoring_runs=2, baseline_scoring_runs=0,
            protected33_used=False, public15_used=False, full121_predictions_used=False,
            new_training_fits=0, new_image_model_forwards=0, ssh_connections=0,
            scope="Repeatedly used development121 source-held-out temporal comparison; no independent, hidden, precise XY or full FPS claim",
            adoption_basis="Complete-chain event results, source changes and real runtime cost; anchor IoU is diagnostic only",
            cached_evaluation_seconds=time.perf_counter()-started,
            evaluator_sha256=sha(h.SCORER), prediction_seal_sha256=sha(args.output/"prediction_seal.json"))
        save(args.output/"report.json", report)
        status.update(completed=True, exit_code=0, phase="completed")
        log(dict(completed=True, scoring_runs=2))
    except Exception:
        status.update(phase="failed", error=traceback.format_exc(), exit_code=1)
        log(dict(error=status["error"]))
        raise
    finally:
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started)
        save(args.output/"run_status.json", status)
        (args.output/"exit_code.txt").write_text(str(status["exit_code"])+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
