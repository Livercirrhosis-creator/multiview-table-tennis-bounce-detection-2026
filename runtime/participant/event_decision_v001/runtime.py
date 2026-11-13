"""Standalone event-decision block over original native-time observations.

Explicit weights are required. A LOFO checkpoint is a research artifact for its
held-out source, not a selected full-development deployment model. This block
does not run ball/table/3D perception or replace the official video entry point.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .candidates import build_pool
from .native_features import extract_candidate
from . import normalization, trajectory_features
from .model import EventDecisionModel, CLASSES
from .selection import select_events


def build_inputs(video, observations, original_candidates, external_candidates=()):
    pool, diagnostics = build_pool(video, observations, original_candidates, external_candidates)
    if not pool:
        return pool, np.empty((0, 15, 23)), np.empty((0, 44)), diagnostics
    observed = [{"frame_id": o["native_frame"], "pts": o["native_pts_seconds"], **o["detection"]}
                for o in observations if not o["center_is_padding"] and o["detection"] is not None]
    features = [extract_candidate(observed, c["native_frame"], video["pts_seconds"], video["width"], video["height"])
                for c in pool]
    if any(f["status_2d"] != "available" for f in features):
        raise ValueError("A proposal lacks its exact original observation")
    sequence = np.asarray([f["sequence_2d"] for f in features], dtype=np.float64)
    summary = np.asarray([f["summary_2d"] for f in features], dtype=np.float64)
    return pool, sequence, trajectory_features.augment(sequence, summary), diagnostics


class Runtime:
    def __init__(self, weights, statistics, *, minimum_separation_seconds=0.08):
        ck = torch.load(Path(weights), map_location="cpu", weights_only=True)
        if tuple(ck["classes"]) != CLASSES:
            raise ValueError("Checkpoint class order differs")
        self.model = EventDecisionModel().cpu().eval()
        self.model.load_state_dict(ck["model_state_dict"], strict=True)
        self.statistics = json.loads(Path(statistics).read_text(encoding="utf-8"))
        self.minimum_separation_seconds = minimum_separation_seconds

    def probabilities(self, sequence, summary):
        if not len(sequence):
            return np.empty((0, 3), dtype=np.float32)
        x = torch.from_numpy(normalization.transform(sequence, self.statistics["sequence"], "tcn"))
        z = torch.from_numpy(trajectory_features.transform(summary, self.statistics["summary"]))
        scores = []
        with torch.inference_mode():
            for start in range(0, len(x), 512):
                scores.append(torch.softmax(self.model(x[start:start+512], z[start:start+512]), -1).numpy())
        return np.concatenate(scores)

    def predict(self, video, observations, original_candidates, external_candidates=()):
        pool, sequence, summary, diagnostics = build_inputs(video, observations, original_candidates, external_candidates)
        probabilities = self.probabilities(sequence, summary)
        events = select_events(pool, probabilities, minimum_separation_seconds=self.minimum_separation_seconds)
        return dict(events=events, candidates=pool, probabilities=probabilities, candidate_diagnostics=diagnostics)
