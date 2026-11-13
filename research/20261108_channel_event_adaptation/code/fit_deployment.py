"""Two fixed full121 deployment fits using the already tested paired-view loss."""
import argparse
import contextlib
from datetime import datetime, timezone
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
DATASET = EXP / "runs/dataset_v002"
PLAN = EXP / "analysis/deployment_plan_v001.json"
OLD = dict(dense=("20261105_coverage_aware_events", "dense_context_v001"),
           window=("20261105_union_event_decision", "event_window_v001"))
CURRENT = ROOT / "experiments/20261107_masked_gap_decision/runs/packet_v001/bundle/version/participant/pingpang/participant/weights"


def trainer():
    spec = importlib.util.spec_from_file_location("fixed_paired_deployment_training", Path(__file__).with_name("train_adapted.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def full_fold(m, kind, index, supervision):
    fold = dict(fold_id="deployment", held_out_family=None,
        train_video_ids=sorted(m.read(m.ALLOW)["video_ids"]), held_out_video_ids=[],
        train_rows=list(range(len(index))), held_out_rows=[], held_out_window_indices=[])
    if kind == "dense":
        fold.update(train_bag_indices=list(range(len(supervision["bags"]))),
                    train_background_indices=supervision["background_indices"])
    else:
        fold.update(train_event_indices=list(range(len(supervision["events"]))),
                    train_background_windows=supervision["background_windows"])
    return fold


def old_assets(m, kind):
    topic, asset = OLD[kind]
    base = ROOT / "experiments" / topic
    norm = base / "weights" / asset / "normalization.json"
    membership = base / "runs/deployment_fit_v001/fit/membership.json"
    report = base / "analysis/deployment_fit_v001.json"
    previous, receipt = m.read(membership), m.read(report)
    assert m.sha(norm) == receipt["normalization_sha256"] == m.sha(CURRENT / asset / "normalization.json")
    members = previous if kind == "dense" else previous["fold"]
    ids = members["training_video_ids"] if kind == "dense" else members["train_video_ids"]
    assert set(ids) == set(m.read(m.ALLOW)["video_ids"]) and len(ids) == 121
    assert len(previous["normalization_rows"]) == receipt["normalization_rows"]
    assert m.digest_rows(previous["normalization_rows"]) == receipt["normalization_membership_sha256"]
    return norm, membership, report


def prepare(m):
    assert not PLAN.exists()
    sources = {m.relative(Path(__file__)):m.sha(Path(__file__))}
    fits = {}
    for kind in ("dense", "window"):
        research_plan = m.read(EXP / "analysis" / f"{kind}_training_plan_v001.json")
        for path, expected in research_plan["source_sha256"].items():
            assert m.sha(ROOT / path) == expected
        sources.update(research_plan["source_sha256"])
        path,_,index,windows,supervision,_,_,_ = m.load_dataset(DATASET,kind)
        fold = full_fold(m,kind,index,supervision)
        groups,bg,legal = m.membership(kind,index,windows,supervision,fold)
        norm,member,report = old_assets(m,kind)
        for p in (norm,member,report, CURRENT / OLD[kind][1] / "normalization.json"):
            sources[m.relative(p)] = m.sha(p)
        fits[kind] = dict(fold=fold, physical_training_events=len(groups),
            event_variants=sum(len(g["variants"]) for g in groups),
            background_by_view={view:len(rows) for view,rows in bg.items()},
            training_legal_rows=len(legal), training_legal_membership_sha256=m.digest_rows(legal),
            normalization_source=m.relative(norm), normalization_sha256=m.sha(norm),
            normalization_membership_source=m.relative(member),
            dataset_manifest_sha256=m.sha(path / "manifest.json"),
            planned_optimizer_steps=m.TRAINING["epochs"]*math.ceil(len(groups)/m.TRAINING["event_batch_size"]))
    value = dict(schema="paired_channel_deployment_plan.v1", prepared_utc=datetime.now(timezone.utc).isoformat(),
        configuration=m.TRAINING, planned_fits=2, fits=fits, source_sha256=sources,
        training_videos=121, views_are_independent_samples=False, checkpoint_selection="fixed epoch80 only",
        normalization="Reuse original full121 deployment statistics byte for byte; no new normalization fitting",
        source_functions="Unmodified train_adapted.fit_fold and batch losses; only normalization path and held-out forward hooks overridden",
        model_evaluation_runs=0, scoring_runs=0, public15_used=False, protected33_used=False,
        official_ball_pose_gap_models_refitted=False)
    m.save(PLAN,value)
    print({"plan":m.relative(PLAN),"sha256":m.sha(PLAN),"fits":2,
           "physical_events":{k:v["physical_training_events"] for k,v in fits.items()}})


def execute(m, kind, output):
    import torch
    plan = m.read(PLAN)
    assert plan["configuration"] == m.TRAINING
    for path, expected in plan["source_sha256"].items():
        assert m.sha(ROOT / path) == expected, "Frozen source changed: " + path
    _,_,index,windows,supervision,_,sequence,summary = m.load_dataset(DATASET,kind)
    fold = full_fold(m,kind,index,supervision)
    assert fold == plan["fits"][kind]["fold"]
    norm,member,_ = old_assets(m,kind)
    assert not output.exists()
    logdir = output.parent / "logs" / kind
    logdir.mkdir(parents=True,exist_ok=False)
    status = dict(state="running",pid=os.getpid(),kind=kind,command=sys.argv,
        started_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=m.sha(PLAN),
        planned_fits=1,completed_fits=0,model_evaluation_runs=0,scoring_runs=0)
    m.save(logdir / "run_status.json",status)
    (logdir / "pid.txt").write_text(str(os.getpid())+"\n",encoding="ascii")
    started = time.perf_counter()
    try:
        torch.set_num_threads(m.TRAINING["threads"])
        torch.set_num_interop_threads(m.TRAINING["interop_threads"])
        m.old_normalization = lambda requested_kind,unused_fold: (norm,member)
        def no_held_forward(model,requested_kind,x,z,windows,unused_fold,directory):
            assert requested_kind == kind and not unused_fold["held_out_rows"]
            # fit_fold hashes this hook's output. It is an explicit no-evaluation
            # receipt, renamed below; it contains no predictions or observations.
            m.save(directory / "held_out.npz",dict(evaluation_performed=False,reason="full121 deployment fit"))
            return {}
        m.predict_held = no_held_forward
        with (logdir / "console.log").open("x",encoding="utf-8",buffering=1) as log, contextlib.redirect_stdout(log):
            _,result = m.fit_fold(kind,sequence,summary,index,windows,supervision,fold,output,m.runtime_modules())
        (output / "held_out.npz").rename(output / "no_evaluation.json")
        checkpoint = torch.load(output / "model.pt",map_location="cpu",weights_only=True)
        checkpoint.pop("held_out_family")
        checkpoint.update(full_development_fit=True,training_videos=121,checkpoint_selection="fixed epoch80; no evaluation")
        torch.save(checkpoint,output / "model.pt")
        result.pop("fold_id"); result.pop("held_out_family")
        result["no_evaluation_receipt_sha256"] = result.pop("held_out_sha256")
        result["no_evaluation_hook_seconds"] = result.pop("held_out_forward_seconds")
        result.update(model_sha256=m.sha(output / "model.pt"),full_development_fit=True,training_videos=121,
            training_fits=1,model_evaluation_runs=0,scoring_runs=0,public15_used=False,protected33_used=False,
            plan_sha256=m.sha(PLAN),normalization_fits=0,source_functions_unchanged=True)
        m.save(output / "fit_result.json",result)
        m.save(output / "provenance.json",dict(result,normalization_source=m.relative(norm),
            normalization_membership_source=m.relative(member),training_source_sha256=plan["source_sha256"]))
        status.update(state="completed",completed_fits=1,exit_code=0)
    except Exception:
        status.update(state="failed",exit_code=1,error=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started)
        m.save(logdir / "run_status.json",status)
        (logdir / "exit_code.txt").write_text(str(status["exit_code"])+"\n",encoding="utf-8")
        if output.exists(): m.save(output / "run_status.json",status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-plan",action="store_true")
    action.add_argument("--execute",action="store_true")
    parser.add_argument("--kind",choices=("dense","window"))
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    m = trainer()
    if args.prepare_plan: prepare(m)
    else:
        if args.kind is None: parser.error("--execute requires --kind")
        execute(m,args.kind,args.output or EXP / "runs/deployment_v001" / args.kind)
