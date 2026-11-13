"""Append released official table-box samples to the frozen 121 fit."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ORIGINAL = ROOT / "experiments/20261108_table_prior_potential"
READY = EXP / "analysis/input_readiness_v001.json"
ACTIVATION = EXP / "runs/activation_v001.json"
SERVER_BASE = Path("/mnt/storage/experiments/20261108_table_prior_potential/server_batches/training_v001/bundle")
RUNTIME_ROOT = Path("/mnt/storage/participant")
FIT_ROOT = "runs/table_fit_v001"
TABLE_JOB = dict(
    server_base="/mnt/storage/experiments/20261109_final154_refit/server_batches/table_v001",
    verify_command=["python", "-B", "code/refit_table.py", "--bundle-root", "."],
    execute_command=["python", "-u", "-B", "bundle/code/refit_table.py", "--bundle-root", "bundle",
                     "--output", FIT_ROOT, "--execute"],
    status_paths=[FIT_ROOT + "/receipt.json"], timeout_seconds=1200,
)
BASE_PLAN_SHA = "5d31df15fa2789a66dc71d58f785a5dc448016820e91e46480e0ff2fd8e5ea35"
TRAINER_SHA = "35001548904e0bc067459f5ecfb68a9f96c84cb9e4fa1ff8ea153fb4aac76c4d"
LOCAL_CODE = ORIGINAL / "code"
TRAINER_DIR = Path(__file__).resolve().parent if Path(__file__).with_name("train_table_source.py").is_file() else LOCAL_CODE
sys.path.insert(0, str(TRAINER_DIR))
from train_table_source import COCO_SHA, FIXED, checked_ref, fit, image_ref, load, require, save, sha


def activated(path):
    record = load(path)
    require(record.get("schema") == "final154_activation.v1"
            and record.get("scope") == "final154_training"
            and record.get("activated") is True and record.get("training_authorized") is True
            and record.get("chain") in {"CHANNEL_TABLE", "VISUAL_SHORTLIST"}
            and all(re.fullmatch(r"[a-f0-9]{64}", record.get(key, ""))
                    for key in ("readiness_sha256", "external_gate_sha256")),
            "Missing final154 activation")
    return record


def base_inputs(bundle):
    plan = load(checked_ref(bundle, dict(path="training_plan.json", sha256=BASE_PLAN_SHA)))
    require(all(plan[k] == v for k, v in FIXED.items()) and plan["initialization"]["sha256"] == COCO_SHA,
            "Frozen 121 fit recipe changed")
    rows = load(checked_ref(bundle, plan["data_manifest"]))["training_rows"]
    ids = set(plan["expected_video_ids"])
    require(len(ids) == 121 and Counter(r["video_id"] for r in rows) == dict.fromkeys(ids, 4),
            "Frozen 121 table samples changed")
    for row in rows:
        row["_image_path"] = str(checked_ref(bundle, image_ref(row)))
    return rows, ids, checked_ref(bundle, plan["initialization"])


def prepare(out):
    require(out.is_relative_to(EXP / "runs") and not out.exists(), "Use a new run directory")
    gate = activated(ACTIVATION)
    require(sha(READY) == gate["readiness_sha256"], "Activation readiness hash changed")
    readiness = load(READY)
    sys.path.insert(0, str(LOCAL_CODE))
    from build_table_images import extract_video
    from extract_table_evidence import table_frame
    import cv2

    held_ref = readiness["cohort"]["membership"][1]
    native_ref = readiness["cohort"]["native"][1]
    held = load(checked_ref(ROOT, held_ref))["video_ids"]
    clips = {c["clip_id"]: c for c in load(checked_ref(ROOT, native_ref))["clips"]}
    require(len(held) == len(set(held)) == 33 and set(held) == set(clips), "Wrong released video membership")
    archive = checked_ref(ROOT, readiness["official_archive"])
    out.mkdir(parents=True)
    bundle = out / "bundle"
    (bundle / "temporary_media").mkdir(parents=True)
    cv2.setNumThreads(1)

    def extract(vid):
        clip = clips[vid]
        width, height, count = (clip[k] for k in ("width", "height", "frame_count"))
        stem = vid.replace("_", "/")
        annotation_path = f"pp_train_data/annotations/{stem}.json"
        video_path = f"rallies_videos/{stem}.mp4"
        with zipfile.ZipFile(archive) as source:
            raw = source.read(annotation_path)
        annotation = json.loads(raw)
        require((annotation["video_id"], annotation["video_path"], annotation["frame_width"],
                 annotation["frame_height"], annotation["num_frames"]) == (vid, video_path, width, height, count)
                and [frame["frame_id"] for frame in annotation["frames"]] == list(range(count)),
                f"Official annotation/native identity differs: {vid}")
        frames = []
        for frame, pts in zip(annotation["frames"], clip["pts_seconds"]):
            row, _ = table_frame(frame, pts, width, height)
            row["valid_boxes"] = [b for b in row["valid_boxes"]
                                  if max(0, b[0]) < min(width, b[2]) and max(0, b[1]) < min(height, b[3])]
            frames.append(row)
        if not any(row["valid_boxes"] for row in frames):
            return [], dict(video_id=vid, training_frame_ids=[], reason="no usable table boxes")
        from hashlib import sha256
        source = dict(video_member="pp_train_data/" + video_path, native_media_sha256=clip["media_sha256"],
                      annotation_member=annotation_path, annotation_sha256=sha256(raw).hexdigest())
        video = dict(video_id=vid, width=width, height=height, frame_count=count, frame_boxes=frames, samples=[])
        rows, _, receipt = extract_video(video, clip["camera_family"], source, bundle)
        for row in rows:
            row["image"]["path"] = (ROOT / row["image"]["path"]).relative_to(bundle).as_posix()
        return rows, receipt

    with ThreadPoolExecutor(max_workers=4) as workers:
        outputs = list(workers.map(extract, sorted(held)))
    rows = [row for samples, _ in outputs for row in samples]
    save(bundle / "samples.json", dict(training_rows=rows, released_video_ids=sorted(held),
                                       members=[receipt for _, receipt in outputs]))
    shutil.copyfile(ACTIVATION, bundle / "activation.json")
    code = bundle / "code"
    code.mkdir()
    trainer = checked_ref(LOCAL_CODE, dict(path="train_table_source.py", sha256=TRAINER_SHA))
    shutil.copyfile(trainer, code / trainer.name)
    shutil.copyfile(__file__, code / Path(__file__).name)
    save(bundle / "table_plan.json", dict(schema="table_full154.v1", samples_sha256=sha(bundle / "samples.json"),
         activation_sha256=sha(bundle / "activation.json"), source_archive_sha256=readiness["official_archive"]["sha256"],
         base_plan_sha256=BASE_PLAN_SHA, trainer_sha256=TRAINER_SHA))
    files = [p for p in sorted(bundle.rglob("*")) if p.is_file()]
    payload_manifest = bundle / "PAYLOAD_MANIFEST.json"
    save(payload_manifest, dict(files=[dict(path=p.relative_to(bundle).as_posix(), bytes=p.stat().st_size,
                                           sha256=sha(p)) for p in files]))
    archive = out / "table_extra33.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as payload:
        for p in [*files, payload_manifest]:
            payload.write(p, p.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive) as payload:
        require(payload.testzip() is None, "Prepared table package CRC failed")
    save(out / "job.json", dict(TABLE_JOB, archive_local_path=archive.relative_to(ROOT).as_posix(),
                                 archive_bytes=archive.stat().st_size, archive_sha256=sha(archive)))
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive),
                          extra_rows=len(rows), labelled_videos=len({r['video_id'] for r in rows}))), flush=True)


def run_fit(bundle, base, output, execute, runtime_root=RUNTIME_ROOT):
    plan = load(bundle / "table_plan.json")
    require(plan["schema"] == "table_full154.v1" and plan["base_plan_sha256"] == BASE_PLAN_SHA
            and plan["trainer_sha256"] == TRAINER_SHA, "Wrong table fit plan")
    gate = activated(checked_ref(bundle, dict(path="activation.json", sha256=plan["activation_sha256"])))
    data = load(checked_ref(bundle, dict(path="samples.json", sha256=plan["samples_sha256"])))
    rows, dev_ids, weight = base_inputs(base)
    extra, held_ids = data["training_rows"], set(data["released_video_ids"])
    require(len(held_ids) == 33 and not dev_ids & held_ids and len(data["members"]) == 33,
            "Released table scope overlaps 121 or lacks members")
    counts = Counter(row["video_id"] for row in extra)
    require(counts and set(counts) <= held_ids and all(1 <= count <= 4 for count in counts.values()),
            "Invalid released table-box coverage")
    for row in extra:
        boxes = row["boxes_320"]
        require(boxes and all(len(b) == 4 and 0 <= b[0] < b[2] <= 320 and 0 <= b[1] < b[3] <= 320 for b in boxes),
                "Invalid released table box")
        row["_image_path"] = str(checked_ref(bundle, image_ref(row)))
    report = dict(training_rows=len(rows) + len(extra), extra_rows=len(extra), activated_chain=gate["chain"],
                  eligible_videos=len(dev_ids) + len(held_ids), labelled_videos=len(dev_ids) + len(counts),
                  activation_sha256=plan["activation_sha256"], base_plan_sha256=BASE_PLAN_SHA,
                  extra_samples_sha256=plan["samples_sha256"], initialization_sha256=COCO_SHA,
                  frozen_recipe=FIXED, official_runtime_root=str(runtime_root))
    if not execute:
        print(json.dumps(report), flush=True)
        return
    require(output is not None and not output.exists(), "Use a new fit output directory")
    require(sys.platform == "linux", "Run the table fit in the prepared Linux GPU environment")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True, timeout=30)
    require(not gpu.stdout.strip(), "GPU is occupied")
    for task in ("pingpang", "basketball"):
        runtime = runtime_root / task
        subprocess.run([sys.executable, "-B", str(runtime / "integrity.py"), "--verify"], check=True, timeout=120)
        subprocess.run([sys.executable, "-B", str(runtime / "scripts/check_env.py")], check=True, timeout=120)
    import cv2
    import torch
    require(torch.cuda.is_available(), "GPU required for the original table fit")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    training = sorted(rows + extra, key=lambda r: (r["video_id"], r["frame_id"]))
    status = fit("full154", None, dev_ids | set(counts), set(),
                 dict(training_rows=training, anchor_rows=[]), weight, output, device,
                 time.perf_counter() + 1200, dict(refit=report))
    require(status["complete_epochs"] == 12 and status["optimizer_updates"] == 12 * math.ceil(len(training) / 8),
            "Table optimizer schedule did not complete")
    save(output / "metadata.json", dict(schema="table_guard_model.v1", architecture="ssdlite320_mobilenet_v3_large",
        num_classes=91, table_label=67, input_size=320, weights_sha256=status["model"]["sha256"],
        training_scope="final154_full_fit", checkpoint_selection="fixed_epoch_12"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", type=Path, help="Prepare only released extras after final activation")
    actions.add_argument("--check-base", action="store_true", help="Validate only existing 121 images and COCO")
    actions.add_argument("--bundle-root", type=Path, help="Check or fit from a prepared extra-image package")
    parser.add_argument("--base-bundle", type=Path, default=SERVER_BASE)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT,
                        help="Official participant root for integrity/environment checks; captain instance: /participant")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(sha(TRAINER_DIR / "train_table_source.py") == TRAINER_SHA, "Original table trainer changed")
    if args.prepare:
        prepare(args.prepare.resolve())
    elif args.check_base:
        rows, videos, weight = base_inputs(args.base_bundle.resolve())
        print(json.dumps(dict(training_rows=len(rows), videos=len(videos), initializer_sha256=sha(weight))), flush=True)
    else:
        run_fit(args.bundle_root.resolve(), args.base_bundle.resolve(), args.output, args.execute,
                runtime_root=args.runtime_root)


if __name__ == "__main__":
    main()
