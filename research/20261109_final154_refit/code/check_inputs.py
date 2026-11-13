"""Verify and inventory final-refit inputs without extracting protected labels."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
POLICY = ROOT / "experiments/20261102_research_foundation/data"
NATIVE121 = ROOT / "experiments/20261102_research_foundation/cache/native121/native_index.json"
NATIVE33 = ROOT / "experiments/20261107_confirmation33_fixed_comparison/runs/data_v001/input/native33/native_index.json"
ARCHIVE = ROOT / "官方乒乓球数据_20261011/01_官方原版训练集_154段.zip"
CAPTURE = ROOT / "experiments/20261109_contact_visual_shortlist/runs"
CHANNEL = ROOT / "experiments/20261108_channel_event_adaptation"
GAP = ROOT / "experiments/20261107_masked_gap_decision"
VISUAL = ROOT / "experiments/20261109_contact_visual_evidence"
TABLE = ROOT / "experiments/20261108_table_prior_potential"
RELEASE = ROOT / "experiments/20261109_deadline_visual_release/runs/visual_stage_v001"
MANIFEST = EXP / "analysis/input_readiness_v001.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def asset(path, expected=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    if expected is not None and value != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=value)


def ids(manifest, role, count):
    data = read(manifest)
    listed = data["video_ids"]
    if data["role"] != role or len(listed) != count or len(set(listed)) != count:
        raise ValueError(f"Unexpected scope: {manifest}")
    return set(listed)


def native(path, expected):
    clips = read(path)["clips"]
    indexed = {c["clip_id"]: c for c in clips}
    if len(indexed) != len(clips) or set(indexed) != expected:
        raise ValueError(f"Native index differs from the scope: {path}")
    return indexed


def archive_inventory(dev, held):
    checksum = "c2cb0443bf1c84d41d455dc569c6be8cc419a1127580db11291d21b45f3ffe17"
    result = asset(ARCHIVE, checksum)
    with zipfile.ZipFile(ARCHIVE) as archive:
        annotations = [r.filename for r in archive.infolist()
                       if r.filename.startswith("pp_train_data/annotations/") and r.filename.endswith(".json")]
        expected = {f"pp_train_data/annotations/{vid.replace('_', '/')}.json" for vid in dev.keys() | held.keys()}
        if len(annotations) != len(expected) or set(annotations) != expected:
            raise ValueError("Official archive annotation members differ from exact combined membership")
        for vid in sorted(dev.keys() | held.keys()):
            suffix = vid.replace("_", "/")
            annotation = archive.getinfo(f"pp_train_data/annotations/{suffix}.json")
            media = archive.getinfo(f"pp_train_data/rallies_videos/{suffix}.mp4")
            if not annotation.file_size or not media.file_size:
                raise ValueError(f"Empty official input for {vid}")
            if vid in held and media.file_size != held[vid]["bytes"]:
                raise ValueError(f"Official video size differs from frozen native index for {vid}")
    return dict(**result, exact_annotation_members=len(expected), matching_held_video_sizes=len(held),
                annotation_payloads_opened=0)


def captures(held):
    seal = CAPTURE / "confirmation33_v001/prediction_seal.json"
    checksums = read(seal)["captures_sha256"]
    folder = CAPTURE / "capture33_v001_merged"
    if set(checksums) != set(held) or {p.stem for p in folder.glob("*.json")} != set(held):
        raise ValueError("Sealed capture membership differs from exact held 33")
    receipts = [CAPTURE / f"capture33_v001_part{part}/receipt.json" for part in range(3)]
    observed = {}
    for receipt in receipts:
        data = read(receipt)
        if not data["completed"]:
            raise ValueError(f"Capture partition was not complete: {receipt}")
        for row in data["videos"]:
            vid = row["video_id"]
            if vid not in held or vid in observed or row["frames"] != held[vid]["frame_count"]:
                raise ValueError(f"Unexpected capture membership or frame count: {vid}")
            if row["capture_sha256"] != checksums[vid]:
                raise ValueError(f"Conflicting capture hashes for {vid}")
            observed[vid] = asset(folder / f"{vid}.json", checksums[vid])
    if set(observed) != set(held):
        raise ValueError("Capture partition receipts do not cover all held videos")
    return dict(directory=folder.relative_to(ROOT).as_posix(), count=len(observed),
                frame_count=sum(c["frame_count"] for c in held.values()),
                seal=asset(seal), receipts=[asset(p) for p in receipts],
                backend="Existing CPU ONNX Runtime capture with original ball weights")


def component_assets():
    weights = RELEASE / "participant/pingpang/participant/weights"
    channel_report = CHANNEL / "analysis/deployment_v001.json"
    channel_receipt = read(channel_report)
    heads = {}
    for name in ("dense", "window"):
        data = channel_receipt["models"][name]
        directory = weights / "channel_adaptation_v001" / name
        heads[name] = dict(model=asset(directory / "model.pt", data["model"]["sha256"]),
                           normalization=asset(directory / "normalization.json", data["normalization"]["sha256"]))
    gap_report = GAP / "analysis/deployment_v001.json"
    visual_report = CAPTURE / "deploy_fit_v001/run_status.json"
    table_report = TABLE / "analysis/training_result_v001.json"
    table = next(m for m in read(table_report)["models"] if m["fit_id"] == "full121")
    return dict(**heads,
        gap=asset(weights / "masked_gap_v001/model.json", read(gap_report)["weights_sha256"]),
        table=asset(weights / "table_guard_v001/model_final.pth", table["model"]["sha256"]),
        visual=asset(weights / "channel_visual_shortlist_v001/model.pt", read(visual_report)["fit"]["model_sha256"]),
        table_coco_initializer=asset(TABLE / "runs/pretrained_v001/ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
            "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"),
        ball_unchanged=asset(weights / "ball.onnx", asset(ROOT / "participant/pingpang/participant/weights/ball.onnx")["sha256"]),
        basketball_unchanged=asset(RELEASE / "participant/basketball/participant/weights/player.onnx",
            asset(ROOT / "participant/basketball/participant/weights/player.onnx")["sha256"]),
        receipts=[asset(p) for p in (channel_report, gap_report, visual_report, table_report)])


def cache_assets():
    result = {}
    for kind in ("dense", "window"):
        directory = CHANNEL / "runs/dataset_v002" / kind
        manifest = directory / "manifest.json"
        hashes = read(manifest)["output_sha256"]
        result[kind] = [asset(manifest), *(asset(directory / name, hashes[name]) for name in
                          ("features.npz", "index.jsonl", "supervision.json", "windows.json"))]
    gap_dir = GAP / "runs/dataset_v001"
    inventory = gap_dir / "inventory.json"
    result["gap"] = [asset(inventory), *(asset(gap_dir / name, read(inventory)["output_sha256"][name])
                      for name in ("features.npy", "index.jsonl", "supervision.json"))]
    rgb = VISUAL / "runs/rgb_inputs_v001"
    receipt = rgb / "receipt.json"
    result["rgb"] = [asset(receipt), *(asset(rgb / name, row["sha256"])
                      for name, row in read(receipt)["files"].items())]
    table_receipt = TABLE / "analysis/table_images_v001.json"
    result["table"] = [asset(table_receipt), *(asset(ROOT / row["path"], row["sha256"])
                        for row in read(table_receipt)["outputs"].values())]
    return result


def build():
    dev_manifest = POLICY / "development_videos.json"
    held_manifest = POLICY / "protected_videos.json"
    dev = native(NATIVE121, ids(dev_manifest, "development", 121))
    held = native(NATIVE33, ids(held_manifest, "evaluation_only", 33))
    if dev.keys() & held.keys():
        raise ValueError("Development and retained IDs overlap")
    return dict(schema="final154_input_readiness.v1", scope="preparation only", training_ready=False,
        training_started=False, held_annotation_payloads_opened=False,
        cohort=dict(dev=len(dev), held=len(held), distinct=len(dev.keys() | held.keys()),
                    membership=[asset(dev_manifest), asset(held_manifest)],
                    native=[asset(NATIVE121), asset(NATIVE33)], policy=asset(POLICY / "data_policy.json")),
        official_archive=archive_inventory(dev, held), held_capture=captures(held),
        baseline_assets=component_assets(), development_caches=cache_assets(),
        candidate_graphs=dict(CHANNEL_TABLE=["dense", "window", "gap", "table"],
                              VISUAL_SHORTLIST=["dense", "window", "gap", "table", "visual"]),
        pending=["final fit scope and winning graph frozen", "154 supervision and table-box coverage checks",
                 "154 trainer inputs and full-membership plans", "one fit per learned component"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write a new readiness manifest")
    args = parser.parse_args()
    result = build()
    if args.write:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    elif MANIFEST.exists() and read(MANIFEST) != result:
        raise ValueError("Input assets differ from the recorded readiness manifest")
    print("Verified 154 exact archive members, 33 sealed captures, baseline weights and development caches.")
    print("Preparation only. No protected annotation extracted and no training run.")


if __name__ == "__main__":
    main()
