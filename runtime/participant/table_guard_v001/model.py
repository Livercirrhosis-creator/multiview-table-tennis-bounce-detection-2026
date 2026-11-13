"""Resident offline SSDLite; no video decoder, fitting, or external code paths."""
import hashlib
import json
from math import isfinite
from pathlib import Path
import time


class TableDetector:
    def __init__(self, weights, metadata=None, device="cuda:0"):
        import torch
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large

        started = time.perf_counter()
        weights = Path(weights)
        if metadata is None:
            adjacent = weights.with_name("metadata.json")
            metadata = adjacent if adjacent.is_file() else Path(__file__).with_name("model_metadata.json")
        self.metadata = json.loads(Path(metadata).read_text(encoding="utf-8"))
        meta = self.metadata
        if (meta.get("schema") != "table_guard_model.v1"
                or meta.get("architecture") != "ssdlite320_mobilenet_v3_large"
                or meta.get("input_size") != 320):
            raise ValueError("Unsupported table model metadata")
        classes, label = meta.get("num_classes"), meta.get("table_label")
        if (type(classes) is not int or type(label) is not int
                or classes < 2 or not 0 < label < classes):
            raise ValueError("Invalid table model class metadata")
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest != meta.get("weights_sha256"):
            raise ValueError("Table model SHA256 differs from the fixed metadata")
        self.device = torch.device(device)
        # Both None values prevent downloads and match the trained backbone.
        self.model = ssdlite320_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=classes)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device).eval()
        self._synchronize()
        self.info = dict(metadata=meta, weights_sha256=digest, device=str(self.device),
                         parameters=sum(p.numel() for p in self.model.parameters()),
                         model_load_seconds=time.perf_counter()-started,
                         persistent_across_videos=True, warmup_forwards=0)

    def _synchronize(self):
        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def predict(self, samples, width, height):
        import torch

        if len(samples) != 3:
            raise ValueError("Exactly three fixed RGB samples are required")
        started = time.perf_counter()
        tensors = [torch.from_numpy(s["rgb"]).permute(2, 0, 1).float().div_(255).to(self.device)
                   for s in samples]
        self._synchronize()
        preprocessing = time.perf_counter()-started
        started = time.perf_counter()
        with torch.inference_mode():
            predictions = self.model(tensors)
        if len(predictions) != 3:
            raise ValueError("Table detector returned an incomplete batch")
        self._synchronize()
        forward_seconds = time.perf_counter()-started
        started = time.perf_counter()
        observations = []
        for sample, pred in zip(samples, predictions):
            mask = pred["labels"] == self.metadata["table_label"]
            boxes, scores = pred["boxes"][mask], pred["scores"][mask]
            box, score = [], 0.0
            if len(boxes):
                best = int(scores.argmax())
                raw = boxes[best].detach().cpu().tolist()
                box = [float(v)*(width if k % 2 == 0 else height)/320 for k, v in enumerate(raw)]
                score = float(scores[best])
                if not all(isfinite(v) for v in box) or not isfinite(score):
                    # The unchanged prior treats invalid observations neutrally;
                    # retain that behavior while keeping the receipt valid JSON.
                    box, score = [], 0.0
            observations.append(dict(frame_id=sample["frame_id"], pts_seconds=sample["pts_seconds"],
                target_relative_seconds=sample["target_relative_seconds"], box=box, score=score))
        return observations, dict(forward_calls=1, image_count=3,
            tensor_preprocessing_seconds=preprocessing,
            synchronized_forward_seconds=forward_seconds,
            prediction_to_cpu_seconds=time.perf_counter()-started)
