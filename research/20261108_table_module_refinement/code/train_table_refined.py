"""One frozen source-held-out SSDLite refinement from a clean COCO state.

The bundle's frozen training_plan.json owns membership and relative paths.
Without --execute this validates inputs and a strict CPU load only. GPU and
official environment/integrity preflight belong to the caller's batch wrapper.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

COCO_SHA = "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"
FIXED = dict(seed=20261002, epochs=12, batch_size=8, lr=0.0003,
             weight_decay=0.0001, gradient_clip=10)
REFINEMENT = {
    "schema": "table_training_refinement.v1",
    "augmentation": {
        "original_probability": 0.25,
        "augmented_probability": 0.75,
        "order": ["horizontal_flip", "color_jitter", "gaussian_blur", "zoom_out"],
        "horizontal_flip_probability": 0.5,
        "brightness": [0.75, 1.25], "contrast": [0.75, 1.25],
        "saturation": [0.75, 1.25], "hue": [-0.03, 0.03],
        "color_jitter_random_order": True,
        "blur_probability": 0.2, "blur_kernel": 3, "blur_sigma": [0.1, 1.0],
        "zoom_out_probability": 0.5, "zoom_scale": [0.8, 1.0],
        "zoom_side_rounding": "Python round(scale * 320)",
        "zoom_canvas_rgb": [0.485, 0.456, 0.406],
        "zoom_interpolation": "bilinear", "zoom_antialias": True,
        "zoom_placement": "uniform integer top/left within canvas, inclusive bounds",
        "crop": False, "preserve_complete_table_boxes": True,
        "rgb_roundoff_tolerance": 0.000001, "rgb_roundoff_clamp": [0.0, 1.0],
    },
    "sampling": {
        "kind": "WeightedRandomSampler", "per_image_weight": "1/sqrt(training_images_in_camera_family)",
        "replacement": True, "draws_per_epoch": "number_of_training_images",
        "source_counts": "current fold training rows only",
        "generator_seed": 20261002,
        "generator_lifecycle": "one shared CPU torch.Generator for sampling and augmentation, continuously advanced across epochs",
    },
    "baseline_coco_predictions": False,
    "held_out_anchor_predictions": 363,
    "full121_anchor_predictions": 0,
    "augmentation_is_new_independent_data": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def inside(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(),
            "Bundle paths must be relative")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Bundle path escapes its root")
    return path


def checked_ref(root, ref):
    path = inside(root, ref["path"])
    require(path.is_file() and sha(path) == ref["sha256"], f"File SHA mismatch: {path}")
    return path


def image_ref(row):
    if isinstance(row["image"], dict):
        return row["image"]
    return dict(path=row["image"], sha256=row["image_sha256"])


def validate(bundle):
    """Hash every input and assert exact membership without opening other data."""
    from PIL import Image

    plan_path = bundle / "training_plan.json"
    plan = load(plan_path)
    require(plan["schema"] == "table_source_training.v1", "Unknown plan schema")
    require(plan.get("refinement") == REFINEMENT, "Frozen refinement differs")
    for key, expected in FIXED.items():
        require(plan[key] == expected, f"Frozen setting changed: {key}")
    require(plan["initialization"]["sha256"] == COCO_SHA, "Require clean COCO initialization")
    weight = checked_ref(bundle, plan["initialization"])
    data_path = checked_ref(bundle, plan["data_manifest"])
    data = load(data_path)
    require(data["schema"] == "table_image_samples.v1", "Unknown sample schema")
    require(plan["image_path_base"] in ("manifest_dir", "bundle_root"), "Unknown image path base")
    image_base = data_path.parent if plan["image_path_base"] == "manifest_dir" else bundle
    ids = plan["expected_video_ids"]
    require(len(ids) == len(set(ids)) == 121, "Expected exactly 121 distinct development videos")
    expected = set(ids)
    train_rows, anchor_rows = data["training_rows"], data["anchor_rows"]
    require(len(train_rows) == 484 and len(anchor_rows) == 363, "Frozen sample counts changed")
    families, assets = {}, {}
    for role, rows in (("training", train_rows), ("anchor", anchor_rows)):
        counts = Counter(row["video_id"] for row in rows)
        require(set(counts) == expected, f"{role} video membership differs")
        require(set(counts.values()) == ({4} if role == "training" else {3}),
                f"Unexpected {role} samples per video")
        seen = set()
        for row in rows:
            vid, frame = row["video_id"], row["frame_id"]
            require(isinstance(frame, int) and frame >= 0, "Invalid native frame")
            require(row.get("native_frame", frame) == frame, "Frame mapping mismatch")
            require((vid, frame) not in seen, f"Duplicate {role} video/frame")
            seen.add((vid, frame))
            family = row["camera_family"]
            require(families.setdefault(vid, family) == family, "Video has inconsistent source family")
            require(row["width"] > 0 and row["height"] > 0 and math.isfinite(row["pts_seconds"]),
                    "Invalid dimensions or native PTS")
            boxes = row["boxes_320"]
            require(len(boxes) == len(row["boxes_original"]), "Scaled box count differs")
            require(role != "training" or len(boxes) > 0, "Missing boxes cannot become training background")
            for box in boxes:
                require(len(box) == 4 and all(math.isfinite(x) for x in box)
                        and 0 <= box[0] < box[2] <= 320 and 0 <= box[1] < box[3] <= 320,
                        "Invalid training/evaluation box")
            if role == "anchor":
                require(row["target_relative_seconds"] in (0.0, 0.48, 0.96), "Changed anchor time")
            ref = image_ref(row)
            image = inside(image_base, ref["path"])
            require(image.is_relative_to(bundle), "Image escapes bundle")
            if str(image) in assets:
                require(assets[str(image)] == ref["sha256"], "Conflicting image SHA")
            else:
                require(image.is_file() and sha(image) == ref["sha256"], f"Image SHA mismatch: {image}")
                with Image.open(image) as png:
                    require(png.format == "PNG" and png.mode == "RGB" and png.size == (320, 320),
                            f"Invalid PNG format: {image}")
                    png.load()
                assets[str(image)] = ref["sha256"]
            row["_image_path"] = str(image)
    require(len(assets) == 726, "Unique image count changed")
    require(len(set(families.values())) == 10, "Expected ten source families")
    folds = plan["folds"]
    require(len(folds) == 10, "Expected ten LOFO fits")
    fold_ids, held_out, validation_counts = set(), set(), Counter()
    for fold in folds:
        name = fold["fold_id"]
        require(isinstance(name, str) and name and all(c.isalnum() or c in "_-" for c in name),
                "Unsafe fold identifier")
        require(name not in fold_ids, "Duplicate fold identifier")
        fold_ids.add(name)
        train, val = set(fold["train_video_ids"]), set(fold["validation_video_ids"])
        require(len(train) == len(fold["train_video_ids"]) and len(val) == len(fold["validation_video_ids"]),
                "Duplicate membership IDs")
        family = fold["held_out_family"]
        require(train and val and not train & val and train | val == expected, "Invalid fold membership")
        require(val == {vid for vid in expected if families[vid] == family}, "Hold out the entire source family")
        require(family not in held_out, "Duplicate held-out family")
        held_out.add(family)
        validation_counts.update(val)
    require(set(validation_counts) == expected and set(validation_counts.values()) == {1},
            "Every development video must be held out once")
    require(plan["full_fit_id"] == "full121", "Frozen full fit identifier changed")
    planned_steps = {f["fold_id"]: FIXED["epochs"] * math.ceil(
        sum(r["video_id"] in set(f["train_video_ids"]) for r in train_rows) / FIXED["batch_size"])
        for f in folds}
    planned_steps["full121"] = FIXED["epochs"] * math.ceil(len(train_rows) / FIXED["batch_size"])
    require(sum(planned_steps.values()) == 7284, "Frozen optimizer update budget differs")
    require(plan["planned_optimizer_steps"] == planned_steps and plan["planned_fits"] == 11
            and plan["planned_epochs"] == 132, "Frozen fit/epoch/update plan differs")
    receipt = dict(plan_sha256=sha(plan_path), data_manifest_sha256=sha(data_path),
                   initialization_sha256=COCO_SHA, script_sha256=sha(Path(__file__)),
                   videos=121, source_families=10, training_rows=484, anchor_rows=363,
                   unique_images=726, complete_fits_planned=11, epochs_per_fit=12,
                   seed_per_fit=20261002, fixed_final_epoch=True, anchor_rows_added_to_training=False,
                   refinement=REFINEMENT, baseline_coco_predictions=0,
                   held_out_anchor_predictions=363, full121_anchor_predictions=0)
    return plan, data, weight, receipt


def make_model(weight):
    import torch
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large

    # Both None arguments are necessary: the default ImageNet backbone has a
    # different reduced-tail construction and would also initiate a download.
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
    state = torch.load(weight, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    require(sum(p.numel() for p in model.parameters()) == 3440060, "Unexpected model structure")
    return model


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def horizontal_flip(image, boxes):
    from torchvision.transforms import functional as F

    result = boxes.clone()
    result[:, 0], result[:, 2] = image.shape[-1] - boxes[:, 2], image.shape[-1] - boxes[:, 0]
    return F.hflip(image), result


def zoom_out(image, boxes, scale, left, top):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as F

    require(tuple(image.shape) == (3, 320, 320), "Unexpected augmentation image size")
    require(0.8 <= scale <= 1, "Zoom scale outside frozen range")
    side = int(round(scale * 320))
    require(0 <= left <= 320 - side and 0 <= top <= 320 - side, "Zoom placement outside canvas")
    resized = F.resize(image, [side, side], interpolation=InterpolationMode.BILINEAR, antialias=True)
    canvas = image.new_tensor(REFINEMENT["augmentation"]["zoom_canvas_rgb"])[:, None, None].expand_as(image).clone()
    canvas[:, top:top + side, left:left + side] = resized
    result = boxes * (side / 320)
    result[:, [0, 2]] += left
    result[:, [1, 3]] += top
    return canvas, result


def augment_training(image, boxes, generator):
    import torch
    from torchvision.transforms import functional as F

    def uniform(low=0.0, high=1.0):
        return low + (high - low) * float(torch.rand((), generator=generator))

    settings = REFINEMENT["augmentation"]
    if uniform() < settings["original_probability"]:
        return image, boxes
    if uniform() < settings["horizontal_flip_probability"]:
        image, boxes = horizontal_flip(image, boxes)
    functions = (F.adjust_brightness, F.adjust_contrast, F.adjust_saturation, F.adjust_hue)
    factors = [uniform(*settings[key]) for key in ("brightness", "contrast", "saturation", "hue")]
    for index in torch.randperm(4, generator=generator).tolist():
        image = functions[index](image, factors[index])
    if uniform() < settings["blur_probability"]:
        sigma = uniform(*settings["blur_sigma"])
        image = F.gaussian_blur(image, [settings["blur_kernel"]] * 2, [sigma] * 2)
    if uniform() < settings["zoom_out_probability"]:
        scale = uniform(*settings["zoom_scale"])
        maximum = 320 - int(round(scale * 320))
        left = int(torch.randint(maximum + 1, (), generator=generator))
        top = int(torch.randint(maximum + 1, (), generator=generator))
        image, boxes = zoom_out(image, boxes, scale, left, top)
    tolerance = settings["rgb_roundoff_tolerance"]
    require(torch.isfinite(image).all().item() and image.min().item() >= -tolerance and image.max().item() <= 1 + tolerance,
            f"Augmented image is outside finite RGB range: {image.min().item()}, {image.max().item()}")
    image = image.clamp(0, 1)
    require(torch.isfinite(boxes).all().item() and (boxes >= 0).all().item() and (boxes <= 320).all().item()
            and (boxes[:, 2:] > boxes[:, :2]).all().item(), "Augmentation damaged a table box")
    return image, boxes


def source_sampler(rows, generator):
    from torch.utils.data import WeightedRandomSampler

    counts = Counter(row["camera_family"] for row in rows)
    require(bool(counts), "No training sources")
    weights = [1 / math.sqrt(counts[row["camera_family"]]) for row in rows]
    return WeightedRandomSampler(weights, len(rows), replacement=True, generator=generator), dict(counts)


class Images:
    def __init__(self, rows, generator=None):
        self.rows = rows
        self.generator = generator

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2
        import torch

        row = self.rows[index]
        bgr = cv2.imread(row["_image_path"], cv2.IMREAD_COLOR)
        require(bgr is not None and bgr.shape == (320, 320, 3), "Could not read the verified image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        boxes = torch.tensor(row["boxes_320"], dtype=torch.float32).reshape(-1, 4)
        if self.generator is not None:
            image, boxes = augment_training(image, boxes, self.generator)
        target = dict(boxes=boxes,
                      labels=torch.full((len(row["boxes_320"]),), 67, dtype=torch.int64))
        return image, target, row


def collate(batch):
    return tuple(zip(*batch))


def deadline_check(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("Frozen training batch time budget exceeded; no automatic retry")


def predict_anchors(model, rows, output, device, deadline):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.ops import box_iou

    model.eval()
    loader = DataLoader(Images(rows), batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)
    ious, missing = [], 0
    started = time.perf_counter()
    with output.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for images, targets, members in loader:
            deadline_check(deadline)
            predictions = model([image.to(device) for image in images])
            for pred, target, row in zip(predictions, targets, members):
                mask = pred["labels"] == 67
                boxes, scores = pred["boxes"][mask], pred["scores"][mask]
                box, score, iou = None, 0.0, None
                if len(boxes):
                    best = int(scores.argmax())
                    box, score = boxes[best].cpu(), float(scores[best])
                    require(torch.isfinite(box).all().item() and math.isfinite(score), "Nonfinite prediction")
                else:
                    missing += 1
                if len(target["boxes"]):
                    iou = float(box_iou(box[None], target["boxes"]).max()) if box is not None else 0.0
                    ious.append(iou)
                box320 = None if box is None else box.tolist()
                native = None if box is None else [float(box[k]) * (row["width"] if k % 2 == 0 else row["height"]) / 320
                                                   for k in range(4)]
                event = {key: row[key] for key in ("video_id", "camera_family", "frame_id", "pts_seconds",
                         "target_relative_seconds", "width", "height")}
                event.update(box=native, box_320=box320, score=score,
                             image_sha256=image_ref(row)["sha256"], gt_boxes_count=len(target["boxes"]),
                             gt_best_iou=iou, diagnostic_only=True)
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    torch.cuda.synchronize(device)
    return dict(path=output.name, sha256=sha(output), rows=len(rows),
                gt_annotated_rows=len(ious), missing_predictions=missing,
                mean_best_iou=None if not ious else sum(ious) / len(ious),
                diagnostic_only=True, used_for_selection=False,
                elapsed_seconds=time.perf_counter() - started)


def fit(fit_id, family, train_ids, val_ids, data, weight, out, device, deadline, receipt):
    import torch
    from torch.utils.data import DataLoader

    out.mkdir(parents=True, exist_ok=False)
    rows = [row for row in data["training_rows"] if row["video_id"] in train_ids]
    anchors = [row for row in data["anchor_rows"] if row["video_id"] in val_ids]
    membership = dict(fit_id=fit_id, held_out_family=family,
                      train_video_ids=sorted(train_ids), validation_video_ids=sorted(val_ids),
                      training_rows=[dict(video_id=r["video_id"], frame_id=r["frame_id"],
                                          image_sha256=image_ref(r)["sha256"]) for r in rows],
                      anchor_prediction_rows=[dict(video_id=r["video_id"], frame_id=r["frame_id"],
                                                   image_sha256=image_ref(r)["sha256"]) for r in anchors])
    save(out / "membership.json", membership)
    seed_all(FIXED["seed"])
    model = make_model(weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FIXED["lr"], weight_decay=FIXED["weight_decay"])
    generator = torch.Generator().manual_seed(FIXED["seed"])
    sampler, source_counts = source_sampler(rows, generator)
    loader = DataLoader(Images(rows, generator), batch_size=8, sampler=sampler, num_workers=0, collate_fn=collate)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    status = dict(fit_id=fit_id, state="running", started_utc=now(), complete_epochs=0,
                  optimizer_updates=0, training_images=len(rows), training_videos=len(train_ids),
                  anchor_predictions_planned=len(anchors), seed=FIXED["seed"],
                  training_source_image_counts=source_counts, **receipt)
    save(out / "receipt.json", status)
    try:
        with (out / "epochs.jsonl").open("x", encoding="utf-8") as epochs:
            for epoch in range(1, 13):
                deadline_check(deadline)
                tick = time.perf_counter()
                model.train()
                for layer in model.modules():
                    if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                        layer.eval()
                losses, components, samples = [], Counter(), 0
                source_draws, unique_images = Counter(), set()
                for images, targets, members in loader:
                    deadline_check(deadline)
                    loss_dict = model([image.to(device) for image in images],
                                      [{key: value.to(device) for key, value in target.items()} for target in targets])
                    loss = sum(loss_dict.values())
                    require(torch.isfinite(loss).item(), f"Nonfinite loss: {fit_id}/{epoch}")
                    optimizer.zero_grad()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), FIXED["gradient_clip"])
                    require(torch.isfinite(norm).item(), f"Nonfinite gradient: {fit_id}/{epoch}")
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    components.update({key: float(value.detach()) for key, value in loss_dict.items()})
                    samples += len(images)
                    source_draws.update(row["camera_family"] for row in members)
                    unique_images.update((row["video_id"], row["frame_id"]) for row in members)
                    status["optimizer_updates"] += 1
                torch.cuda.synchronize(device)
                row = dict(epoch=epoch, loss=sum(losses) / len(losses), batches=len(losses), samples=samples,
                           loss_components={key: value / len(losses) for key, value in components.items()},
                           optimizer_updates=status["optimizer_updates"], elapsed_seconds=time.perf_counter() - tick,
                           source_draws=dict(sorted(source_draws.items())), unique_training_images=len(unique_images))
                require(sum(source_draws.values()) == samples == len(rows), "Sampler changed epoch draw budget")
                epochs.write(json.dumps(row, allow_nan=False) + "\n")
                epochs.flush()
                status["complete_epochs"] = epoch
                save(out / "receipt.json", status)
                print(json.dumps(dict(fit_id=fit_id, **row)), flush=True)
        deadline_check(deadline)
        checkpoint = out / "model_final.pth"
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint)
        # Only final-epoch held-out anchors are inferred. The full fit is a
        # deployment asset; it receives no training-set anchor evaluation.
        prediction = predict_anchors(model, anchors, out / "anchor_predictions.jsonl", device, deadline) if anchors else None
        status.update(state="complete", finished_utc=now(), final_epoch=12,
                      model=dict(path=checkpoint.name, sha256=sha(checkpoint), bytes=checkpoint.stat().st_size),
                      membership_sha256=sha(out / "membership.json"), epochs_sha256=sha(out / "epochs.jsonl"),
                      anchor_predictions=prediction, elapsed_seconds=time.perf_counter() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
        save(out / "receipt.json", status)
        return status
    except BaseException as error:
        status.update(state="failed", finished_utc=now(), error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.perf_counter() - started)
        save(out / "receipt.json", status)
        raise
    finally:
        del optimizer, model
        torch.cuda.empty_cache()


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.original.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    bundle, output = args.bundle_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "pid.txt").write_text(str(os.getpid()) + "\n", encoding="ascii")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    status = dict(schema="table_source_batch.v1", state="validating", started_utc=now(),
                  execute=args.execute, pid=os.getpid(), fits=[], completed_fits=0, optimizer_updates=0)
    save(output / "run_status.json", status)
    code = 1
    with (output / "console.log").open("x", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(original_stdout, log), Tee(original_stderr, log)
        try:
            plan, data, weight, receipt = validate(bundle)
            import cv2
            import numpy as np
            import torch
            import torchvision

            torch.set_num_threads(4)
            cv2.setNumThreads(1)
            seed_all(FIXED["seed"])
            model = make_model(weight)
            status.update(inputs=receipt, environment=dict(python=sys.version, torch=torch.__version__,
                          torchvision=torchvision.__version__, numpy=np.__version__, opencv=cv2.__version__),
                          strict_cpu_initialization_passed=True,
                          training_settings=dict(**FIXED, optimizer="AdamW", image_size=320,
                          num_classes=91, table_label=67, trainable_backbone_layers=6,
                          frozen_batchnorm_statistics=True, batchnorm_affine_trainable=True,
                          augmentation=REFINEMENT["augmentation"], sampling=REFINEMENT["sampling"],
                          scheduler=None, amp=False, data_loader_workers=0))
            if not args.execute:
                status.update(state="validated_only", model_training_runs=0, model_predictions=0)
            else:
                require(torch.cuda.is_available(), "CUDA unavailable; CPU training is not authorized by this GPU batch")
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
                status.update(state="running", gpu=torch.cuda.get_device_name(device))
                save(output / "run_status.json", status)
                seconds = float(plan.get("max_seconds", 2400))
                require(math.isfinite(seconds) and 0 < seconds <= 2400, "Invalid GPU batch budget")
                deadline = time.perf_counter() + seconds
                del model
                torch.cuda.empty_cache()
                save(output / "batch_receipt.json", status)
                for fold in plan["folds"]:
                    result = fit(fold["fold_id"], fold["held_out_family"], set(fold["train_video_ids"]),
                                 set(fold["validation_video_ids"]), data, weight, output / "folds" / fold["fold_id"],
                                 device, deadline, receipt)
                    status["fits"].append(result)
                    status["completed_fits"] += 1
                    status["optimizer_updates"] += result["optimizer_updates"]
                    save(output / "batch_receipt.json", status)
                    save(output / "run_status.json", status)
                result = fit("full121", None, set(plan["expected_video_ids"]), set(), data, weight,
                             output / "full121", device, deadline, receipt)
                status["fits"].append(result)
                status["completed_fits"] += 1
                status["optimizer_updates"] += result["optimizer_updates"]
                status.update(state="complete", model_training_runs=11, completed_epochs=132,
                              model_predictions=363, baseline_coco_predictions=0, final_epoch=12,
                              anchor_iou_used_for_selection=False, full_fit_evaluation=False)
            status["finished_utc"] = now()
            code = 0
        except BaseException as error:
            status.update(state="failed", finished_utc=now(), error=f"{type(error).__name__}: {error}")
            # Preserve completed work and updates from an interrupted fit; never
            # describe a failed batch as if only the completed fits had run.
            paths = sorted((output / "folds").glob("*/receipt.json"))
            if (output / "full121/receipt.json").is_file():
                paths.append(output / "full121/receipt.json")
            status["fits"] = [load(path) for path in paths]
            status["completed_fits"] = sum(item["state"] == "complete" for item in status["fits"])
            status["optimizer_updates"] = sum(item["optimizer_updates"] for item in status["fits"])
            status["completed_epochs"] = sum(item["complete_epochs"] for item in status["fits"])
            traceback.print_exc()
        finally:
            status["exit_code"] = code
            status["elapsed_seconds"] = time.perf_counter() - started
            save(output / "run_status.json", status)
            save(output / "batch_receipt.json", status)
            (output / "exit_code.txt").write_text(str(code) + "\n", encoding="ascii")
            sys.stdout, sys.stderr = original_stdout, original_stderr
    return code


if __name__ == "__main__":
    raise SystemExit(main())
