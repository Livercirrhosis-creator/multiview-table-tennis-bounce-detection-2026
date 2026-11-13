"""Fixed ten-source visual adaptation on cached ImageNet stem features.

Only --execute starts a real fit. --check uses random tensors and three small
optimizer steps; it never reads candidate labels or writes a model checkpoint.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import torchvision


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
DATASET_REL = "experiments/20261108_channel_event_adaptation/runs/dataset_v002/dense"
ALLOW_REL = "experiments/20261102_research_foundation/data/development_videos.json"
LOSS_REL = "experiments/20261105_coverage_aware_events/code/train_dense_context.py"
MODEL_SOURCE = EXP / "versions/v001/participant/pingpang/participant/adapted_visual_v001/model.py"
TRAINING = dict(epochs=16, seed=20261109, event_batch_size=48, background_per_event=2,
                tail_learning_rate=1e-4, head_learning_rate=1e-3, weight_decay=1e-4,
                torch_threads=3, fold_workers=1, checkpoint="fixed_final_epoch",
                optimizer="AdamW", bn_statistics_and_affine="frozen",
                deterministic_algorithms=False, cudnn_benchmark=False, allow_tf32=False)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def configure_torch(device):
    torch.set_num_threads(TRAINING["torch_threads"])
    torch.set_num_interop_threads(1)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def event_loss_terms(logits, bag_rows, classes):
    """The frozen dense loss, with index tensors placed on logits.device.

    Each event gets one latent target maximum and the mean exclusion loss for
    the competing contact. Other locations remain allowed target/background.
    """
    if len(bag_rows) != len(classes) or not all(c in (0, 1) for c in classes):
        raise ValueError("Expected complete table/racket event bags")
    width = max(map(len, bag_rows))
    padded = np.zeros((len(bag_rows), width), dtype=np.int64)
    mask = np.zeros_like(padded, dtype=bool)
    for i, rows in enumerate(bag_rows):
        if not rows:
            raise ValueError("An event bag is empty")
        padded[i, :len(rows)], mask[i, :len(rows)] = rows, True
    members = torch.as_tensor(padded, device=logits.device)
    valid = torch.as_tensor(mask, device=logits.device)
    kinds = torch.tensor(classes, dtype=torch.long, device=logits.device)
    logp = torch.log_softmax(logits, dim=-1)
    target_logp = logp[members, kinds[:, None]].masked_fill(~valid, -torch.inf)
    target_mil = -target_logp.max(dim=1).values
    bag_logits = logits[members]
    allowed_classes = torch.stack((kinds, torch.full_like(kinds, 2)), dim=1)
    allowed_logits = bag_logits.gather(2, allowed_classes[:, None, :].expand(-1, width, -1))
    forbidden_nll = torch.logsumexp(bag_logits, dim=-1) - torch.logsumexp(allowed_logits, dim=-1)
    other_contact = forbidden_nll.masked_fill(~valid, 0.).sum(dim=1) / valid.sum(dim=1)
    return target_mil, other_contact


def load_inputs(input_dir, assets_root, model_api, pretrained):
    dataset, allow_path = assets_root / DATASET_REL, assets_root / ALLOW_REL
    manifest = read(dataset / "manifest.json")
    paths = [dataset / name for name in ("index.jsonl", "supervision.json", "folds.json")]
    hashes = {path.name: sha(path) for path in paths}
    if any(manifest["output_sha256"][name] != digest for name, digest in hashes.items()):
        raise ValueError("Frozen development labels or candidate indexes differ")
    allow = set(read(allow_path)["video_ids"])
    original = lines(paths[0])
    if len(allow) != 121 or {row["video_id"] for row in original} - allow:
        raise ValueError("Candidate pool outside development121")
    expected = sorted((r for r in original if r["view"] == "augmented"), key=lambda r: r["row_index"])
    rows = lines(input_dir / "rows.jsonl")
    keys = ("row_index", "candidate_id", "physical_candidate_id", "video_id", "native_frame", "native_pts_seconds")
    if len(rows) != 17559 or len(rows) != len(expected) or any(
            row["array_index"] != i or any(row[k] != ref[k] for k in keys)
            for i, (row, ref) in enumerate(zip(rows, expected))):
        raise ValueError("Stem cache rows differ from the frozen augmented candidate order")
    stem = np.load(input_dir / "stem_features.npy", mmap_mode="r", allow_pickle=False)
    temporal = np.load(input_dir / "time_features.npy", allow_pickle=False)
    if stem.shape != (len(rows), *model_api.STEM_SHAPE) or stem.dtype != np.float32:
        raise ValueError("Unexpected stem cache dimensions or dtype")
    if temporal.shape != (len(rows), 3, 2) or temporal.dtype != np.float32 or not np.isfinite(temporal).all():
        raise ValueError("Unexpected temporal array")
    mask = temporal[:, :, 1]
    if not np.isin(mask, [0, 1]).all() or not np.all(mask[:, 1] == 1) or not np.all(temporal[:, 1, 0] == 0):
        raise ValueError("Invalid temporal mask or center timestamp")
    for start in range(0, len(stem), 256):
        part = stem[start:start+256]
        if not np.isfinite(part).all() or np.any(part[mask[start:start+256] == 0] != 0):
            raise ValueError("Nonfinite features or nonzero missing slots")
    idx = {row["row_index"]: i for i, row in enumerate(rows)}
    if len(idx) != len(rows) or len({r["physical_candidate_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate augmented candidate")
    supervision, split = read(paths[1]), read(paths[2])
    if supervision["classes"] != {name: i for i, name in enumerate(model_api.CLASSES)} or set(split["video_groups"]) != allow:
        raise ValueError("Frozen classes or video groups differ")
    events = []
    for bag in supervision["bags"]:
        if bag["view"] != "augmented":
            continue
        indices = [idx[i] for i in bag["indices"]]
        if not indices or any(rows[i]["video_id"] != bag["video_id"] for i in indices):
            raise ValueError("Event interval includes an unrelated feature row")
        events.append((bag["video_id"], bag["class_id"], indices))
    if Counter(kind for _, kind, _ in events) != {0: 613, 1: 555}:
        raise ValueError("Expected the fixed 613 table and 555 racket event bags")
    backgrounds = [idx[i] for i in supervision["background_indices"] if i in idx]
    if set(backgrounds) & {i for _, _, event in events for i in event}:
        raise ValueError("An event row was also labeled background")
    fold_keys = ("fold_id", "held_out_family", "train_video_ids", "held_out_video_ids")
    folds = [{key: row[key] for key in fold_keys} for row in split["folds"]]
    heldout = Counter()
    for fold in folds:
        train, held = set(fold["train_video_ids"]), set(fold["held_out_video_ids"])
        if train & held or train | held != allow or {split["video_groups"][v] for v in held} != {fold["held_out_family"]}:
            raise ValueError("Split differs from the source-held-out design")
        heldout.update(held)
    if len(folds) != 10 or set(heldout) != allow or any(n != 1 for n in heldout.values()):
        raise ValueError("Every development video must be held out exactly once")
    receipt = read(input_dir / "receipt.json")
    if (receipt["schema"] != "adapted_visual_stem_cache.v1" or not receipt["complete"]
            or receipt["candidate_index_sha256"] != hashes["index.jsonl"]
            or receipt["allowlist_sha256"] != sha(allow_path) or receipt["videos"] != 121
            or receipt["rows"] != len(rows) or receipt["dtype"] != "float32"
            or receipt["stem_shape"] != list(stem.shape)
            or receipt["pretrained_sha256"] != model_api.WEIGHTS_SHA256
            or receipt["model_source_sha256"] != sha(MODEL_SOURCE)
            or sha(pretrained) != model_api.WEIGHTS_SHA256):
        raise ValueError("Stem cache receipt, model or pretrained identity differs")
    for name in ("stem_features.npy", "time_features.npy", "rows.jsonl"):
        hashes[name] = sha(input_dir / name)
        if hashes[name] != receipt["files"][name]["sha256"]:
            raise ValueError(f"Stem cache differs from its receipt: {name}")
    hashes["receipt.json"], hashes["allowlist.json"] = sha(input_dir / "receipt.json"), sha(allow_path)
    return stem, temporal, rows, events, backgrounds, folds, hashes


def train_batch(model, optimizer, stem, temporal, events, backgrounds, device):
    chosen, bags, classes = [], [], []
    for _, kind, indices in events:
        bags.append(list(range(len(chosen), len(chosen) + len(indices))))
        chosen.extend(indices)
        classes.append(kind)
    event_rows = len(chosen)
    chosen.extend(backgrounds)
    features = torch.from_numpy(stem[chosen]).to(device)
    times = torch.from_numpy(temporal[chosen]).to(device)
    logits = model.forward_stem(features, times)
    target, other_contact = event_loss_terms(logits[:event_rows], bags, classes)
    event_loss = (target + other_contact).mean()
    background_loss = -torch.log_softmax(logits[event_rows:], dim=-1)[:, 2].mean()
    loss = .5 * (event_loss + background_loss)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite adapted visual loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(event_loss.detach()), float(background_loss.detach())


def fit_fold(folder, fold, data, model_api, pretrained, device, progress_callback=None):
    stem, temporal, rows, all_events, all_backgrounds = data
    train, held = set(fold["train_video_ids"]), set(fold["held_out_video_ids"])
    events = [e for e in all_events if e[0] in train]
    backgrounds = [i for i in all_backgrounds if rows[i]["video_id"] in train]
    n_backgrounds = TRAINING["background_per_event"] * len(events)
    if not events or len(backgrounds) < n_backgrounds:
        raise ValueError("Insufficient legal training-side events or backgrounds")
    folder.mkdir(exist_ok=False)
    torch.manual_seed(TRAINING["seed"])
    rng = np.random.default_rng(TRAINING["seed"])
    model = model_api.ContactVisualModel(pretrained).to(device).train()
    frozen_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.AdamW(model.optimizer_groups(TRAINING["tail_learning_rate"], TRAINING["head_learning_rate"]),
                                  weight_decay=TRAINING["weight_decay"])
    batch_size = TRAINING["event_batch_size"]
    synchronize(device)
    started, steps = time.perf_counter(), 0
    with (folder / "epochs.jsonl").open("x", encoding="utf-8") as stream:
        for epoch in range(TRAINING["epochs"]):
            synchronize(device)
            epoch_started = time.perf_counter()
            order = rng.permutation(len(events))
            bg = rng.choice(backgrounds, size=n_backgrounds, replace=False)
            event_loss, background_loss = [], []
            for start in range(0, len(events), batch_size):
                batch = [events[i] for i in order[start:start+batch_size]]
                ratio = TRAINING["background_per_event"]
                negatives = bg[start*ratio:(start+len(batch))*ratio].tolist()
                el, bl = train_batch(model, optimizer, stem, temporal, batch, negatives, device)
                event_loss.append(el)
                background_loss.append(bl)
                steps += 1
            synchronize(device)
            entry = dict(epoch=epoch+1, event_loss=float(np.mean(event_loss)),
                         background_loss=float(np.mean(background_loss)), optimizer_steps=steps,
                         epoch_seconds=time.perf_counter()-epoch_started,
                         fit_elapsed_seconds=time.perf_counter()-started)
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(dict(fold=fold["fold_id"], **entry)), flush=True)
            if progress_callback is not None:
                progress_callback(entry)
    synchronize(device)
    fit_seconds = time.perf_counter() - started
    if any(not torch.equal(frozen_buffers[name], value) for name, value in model.named_buffers()):
        raise ValueError("A frozen pretrained normalization buffer changed")
    checkpoint = folder / "model.pt"
    torch.save(dict(model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    classes=model_api.CLASSES, training=TRAINING, pretrained_sha256=model_api.WEIGHTS_SHA256,
                    train_video_ids=sorted(train), held_out_video_ids=sorted(held)), checkpoint)
    model.eval()
    held_indices = [i for i, r in enumerate(rows) if r["video_id"] in held]
    predictions = {vid: {} for vid in sorted(held)}
    with torch.inference_mode():
        for start in range(0, len(held_indices), 256):
            indices = held_indices[start:start+256]
            logits = model.forward_stem(torch.from_numpy(stem[indices]).to(device),
                                        torch.from_numpy(temporal[indices]).to(device))
            values = torch.softmax(logits, -1).cpu().tolist()
            for i, probs in zip(indices, values):
                predictions[rows[i]["video_id"]][rows[i]["physical_candidate_id"]] = probs
    receipt = dict(fold=fold["fold_id"], heldout_family=fold["held_out_family"], train_events=len(events),
                   event_classes=dict(Counter(k for _, k, _ in events)), available_backgrounds=len(backgrounds),
                   backgrounds_per_epoch=n_backgrounds, heldout_candidates=len(held_indices),
                   fit_seconds=fit_seconds, optimizer_steps=steps, model_sha256=sha(checkpoint),
                   pretrained_bn_unchanged=True, device=str(device),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    save(folder / "fit_receipt.json", receipt)
    print(json.dumps(receipt, allow_nan=False), flush=True)
    return predictions, receipt


def check(model_api, pretrained, assets_root, device):
    torch.manual_seed(TRAINING["seed"])
    rng = np.random.default_rng(TRAINING["seed"])
    # The real fixed bags average 2.644 candidates/event, maximum four. This
    # fixture approximates its 223-row batch without reading any real labels.
    sizes = [3] * 31 + [2] * 17
    event_rows = sum(sizes)
    n = event_rows + 96
    stem = rng.normal(size=(n, *model_api.STEM_SHAPE)).astype(np.float32)
    temporal = np.broadcast_to(np.asarray([[-.08, 1.], [0., 1.], [.08, 1.]], np.float32), (n, 3, 2)).copy()
    model = model_api.ContactVisualModel(pretrained).to(device).train()
    rgb = torch.from_numpy(rng.integers(0, 256, (2, *model_api.RGB_SHAPE), dtype=np.uint8)).to(device)
    times = torch.from_numpy(temporal[:2]).to(device)
    times[0, 0, 1] = 0
    with torch.no_grad():
        cached = model.extract_stem(rgb)
        direct = model(rgb, times)
        separate = model.forward_stem(cached, times)
        altered = cached.clone()
        altered[0, 0] += 100
        masked = model.forward_stem(altered, times)
    assert cached.shape == (2, *model_api.STEM_SHAPE) and torch.equal(direct, separate)
    assert torch.equal(separate, masked) and direct.shape == (2, 3) and torch.isfinite(direct).all()
    old_loss = module("frozen_dense_loss_check", assets_root / LOSS_REL).event_loss_terms
    fixture = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    old = sum(old_loss(fixture, [[0, 1, 2], [3, 4], [5, 6]], [0, 1, 0]))
    new = sum(event_loss_terms(fixture.to(device), [[0, 1, 2], [3, 4], [5, 6]], [0, 1, 0])).cpu()
    assert torch.allclose(old, new, atol=1e-12, rtol=1e-12)
    old_grad = torch.autograd.grad(old.sum(), fixture, retain_graph=True)[0]
    new_grad = torch.autograd.grad(new.sum(), fixture)[0]
    assert torch.allclose(old_grad, new_grad, atol=1e-12, rtol=1e-12)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=TRAINING["weight_decay"])
    events, offset = [], 0
    for i, size in enumerate(sizes):
        events.append(("random_fixture", i % 2, list(range(offset, offset+size))))
        offset += size
    times_seconds = []
    for _ in range(3):
        synchronize(device)
        started = time.perf_counter()
        train_batch(model, optimizer, stem, temporal, events, list(range(event_rows, n)), device)
        synchronize(device)
        times_seconds.append(time.perf_counter()-started)
    frozen_names = {name for name, p in model.named_parameters() if not p.requires_grad}
    frozen_names.update(name for name, _ in model.named_buffers())
    current = model.state_dict()
    assert all(torch.equal(before[name], current[name]) for name in frozen_names)
    assert any(not torch.equal(before[name], current[name]) for name, p in model.named_parameters()
               if p.requires_grad and name.startswith("tail."))
    assert any(not torch.equal(before[name], current[name]) for name, p in model.named_parameters()
               if p.requires_grad and not name.startswith("tail."))
    assert all(not layer.training for layer in model.modules() if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm))
    return dict(completed=True, checked_utc=datetime.now(timezone.utc).isoformat(), device=str(device),
                torch=torch.__version__, torchvision=torchvision.__version__,
                model_source_sha256=sha(MODEL_SOURCE), training_source_sha256=sha(__file__),
                pretrained_sha256=sha(pretrained), frozen_loss_source_sha256=sha(assets_root / LOSS_REL),
                stem_shape=list(cached.shape), parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                trainable_tail_parameters=sum(p.numel() for p in model.tail.parameters() if p.requires_grad),
                synthetic_batch_rows=n, synthetic_event_bags=48, synthetic_background_rows=96,
                synthetic_optimizer_steps=3, synthetic_step_seconds=times_seconds,
                cache_float32_bytes=17559 * int(np.prod(model_api.STEM_SHAPE)) * 4,
                checks=["direct_and_cached_logits_exact", "missing_slot_invariant", "finite_class_logits",
                        "old_loss_value_and_gradient_equivalent", "stem_and_all_bn_unchanged",
                        "tail_and_head_updated"], real_training_runs=0, scoring_runs=0, ssh=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ROOT)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    configure_torch(args.device)
    model_api = module("adapted_visual_model", MODEL_SOURCE)
    if args.check:
        if args.execute:
            parser.error("--check and --execute are exclusive")
        if args.check_output and args.check_output.exists():
            raise FileExistsError(args.check_output)
        report = check(model_api, args.pretrained, args.assets_root, args.device)
        if args.check_output:
            args.check_output.parent.mkdir(parents=True, exist_ok=True)
            save(args.check_output, report)
        print(json.dumps(report, allow_nan=False), flush=True)
        return
    if not args.execute or args.cache is None or args.output is None:
        parser.error("A real fit requires --execute --cache and a new --output")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(exit_code=1, completed=False, started_utc=datetime.now(timezone.utc).isoformat(),
                  command=sys.argv, python=sys.executable, torch=torch.__version__, torchvision=torchvision.__version__,
                  training=TRAINING, protected33_used=False, public15_used=False, external130_used=False,
                  training_runs=0, training_runs_started=0, deployment_fit=False, device=args.device,
                  reproducibility="fixed seed and configuration; nondeterministic CUDA kernels allowed")
    save(output / "run_status.json", status)
    try:
        stem, temporal, rows, events, backgrounds, folds, hashes = load_inputs(
            args.cache.resolve(), args.assets_root.resolve(), model_api, args.pretrained)
        status["input_sha256"] = hashes
        status["code_sha256"] = {p.name: sha(p) for p in (Path(__file__), MODEL_SOURCE, args.assets_root / LOSS_REL)}
        status["pretrained_sha256"] = sha(args.pretrained)
        probabilities, receipts = {}, []
        for fold in folds:
            status["training_runs_started"] += 1
            status["current_fold"] = fold["fold_id"]
            status["current_epoch"] = 0
            save(output / "run_status.json", status)
            def progress(entry):
                status.update(current_epoch=entry["epoch"], last_epoch_seconds=entry["epoch_seconds"],
                              current_fold_fit_elapsed_seconds=entry["fit_elapsed_seconds"])
                save(output / "run_status.json", status)
            prediction, receipt = fit_fold(output / f"fold_{fold['fold_id']:02d}", fold,
                (stem, temporal, rows, events, backgrounds), model_api, args.pretrained, args.device,
                progress_callback=progress)
            if probabilities.keys() & prediction.keys():
                raise ValueError("Duplicate held-out video predictions")
            probabilities.update(prediction)
            receipts.append(receipt)
            status["training_runs"] += 1
            save(output / "run_status.json", status)
        if sum(map(len, probabilities.values())) != len(rows) or len(probabilities) != 121:
            raise ValueError("Incomplete source-held-out probability coverage")
        artifact = output / "probabilities_by_video.json"
        save(artifact, probabilities)
        status.update(exit_code=0, completed=True, candidates=len(rows),
                      predictions_sha256=sha(artifact), folds=receipts)
    except Exception:
        status["error"] = traceback.format_exc()
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "run_status.json", status)
        (output / "exit_code.txt").write_text(str(status["exit_code"]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
