"""Standalone complete2D decisions using explicit presence/location weights.

Training and evaluation code, labels and source identities are not runtime
dependencies. A source-held-out checkpoint remains a research checkpoint;
deploying this module requires separately fitted full-development weights.
"""
import json
from pathlib import Path
import numpy as np
import torch

if "." in __package__:
    from ..event_decision_v001.runtime import build_inputs
    from ..event_decision_v001 import normalization, trajectory_features
else:
    from event_decision_v001.runtime import build_inputs
    from event_decision_v001 import normalization, trajectory_features
from .model import WindowEventModel, CLASSES
from .selection import select_window_events
from .windows import build_windows


class Runtime:
    def __init__(self, weights, statistics, *, minimum_separation_seconds=.08):
        checkpoint = torch.load(Path(weights), map_location="cpu", weights_only=True)
        if tuple(checkpoint["classes"]) != CLASSES:
            raise ValueError("Checkpoint class order differs")
        self.model = WindowEventModel().cpu().eval()
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.statistics = json.loads(Path(statistics).read_text(encoding="utf-8"))
        self.minimum_separation_seconds = minimum_separation_seconds

    def scores(self, sequence, summary, windows):
        """Return window presence probabilities and candidate location logits."""
        if not len(sequence):
            if windows:
                raise ValueError("Windows cannot exist without candidate observations")
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        x = torch.from_numpy(normalization.transform(sequence, self.statistics["sequence"], "tcn"))
        z = torch.from_numpy(trajectory_features.transform(summary, self.statistics["summary"]))
        with torch.inference_mode():
            embeddings = torch.cat([self.model.encode(x[start:start + 512], z[start:start + 512])
                                    for start in range(0, len(x), 512)])
            location = self.model.location(embeddings).numpy()
            presence = [torch.softmax(self.model.window_logits(embeddings,
                [w["member_rows"] for w in windows[start:start + 512]]), dim=-1).numpy()
                for start in range(0, len(windows), 512)]
        return (np.concatenate(presence) if presence else np.empty((0, 3), dtype=np.float32)), location

    def predict(self, video, observations, original_candidates, external_candidates=()):
        pool, sequence, summary, diagnostics = build_inputs(
            video, observations, original_candidates, external_candidates)
        windows = build_windows(video, observations, pool)
        presence, location = self.scores(sequence, summary, windows)
        events = select_window_events(pool, windows, presence, location,
                                      minimum_separation_seconds=self.minimum_separation_seconds)
        # The compatibility key probabilities now describes windows, not a
        # conditional probability that every candidate is the exact event frame.
        return dict(events=events, candidates=pool, probabilities=presence,
                    presence_probabilities=presence, location_logits=location,
                    windows=windows, candidate_diagnostics=diagnostics)
