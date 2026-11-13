"""Dense label-free native inputs and one candidate-center event classifier."""
import json
from pathlib import Path
import time
import numpy as np
import torch

from ..event_decision_v001.native_features import extract_candidate
from ..event_decision_v001 import normalization, trajectory_features
from ..event_decision_v001.selection import select_events
from ..event_window_v001.windows import build_windows
from .candidates import build_pool
from .model import CLASSES, DenseContextModel


def build_inputs(video, observations, original_candidates, external_candidates=()):
    observations = list(observations)
    pool, diagnostics = build_pool(video, observations, original_candidates, external_candidates)
    windows = build_windows(video, observations, pool)
    if not pool:
        return pool, np.empty((0,15,23)), np.empty((0,44)), windows, diagnostics
    observed = [dict(frame_id=o["native_frame"],pts=o["native_pts_seconds"],**o["detection"])
                for o in observations if not o["center_is_padding"] and o["detection"] is not None]
    features = [extract_candidate(observed,c["native_frame"],video["pts_seconds"],video["width"],video["height"])
                for c in pool]
    if any(f["status_2d"] != "available" for f in features):
        raise ValueError("A dense candidate lacks its exact original observation")
    sequence = np.asarray([f["sequence_2d"] for f in features],dtype=np.float64)
    summary = np.asarray([f["summary_2d"] for f in features],dtype=np.float64)
    return pool, sequence, trajectory_features.augment(sequence,summary), windows, diagnostics


class Runtime:
    def __init__(self, weights, statistics, *, minimum_separation_seconds=.08):
        checkpoint = torch.load(Path(weights),map_location="cpu",weights_only=True)
        if tuple(checkpoint["classes"]) != CLASSES:
            raise ValueError("Dense checkpoint class order differs")
        self.model = DenseContextModel().cpu().eval()
        self.model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        self.statistics = json.loads(Path(statistics).read_text(encoding="utf-8"))
        self.minimum_separation_seconds = minimum_separation_seconds

    def probabilities(self, sequence, summary, windows):
        if not len(sequence):
            if windows:
                raise ValueError("Windows cannot exist without candidate observations")
            return np.empty((0,3),dtype=np.float32)
        if len(windows) != len(sequence) or [w["center_row_index"] for w in windows] != list(range(len(sequence))):
            raise ValueError("One ordered context is required for every candidate")
        x = torch.from_numpy(normalization.transform(sequence,self.statistics["sequence"],"tcn"))
        z = torch.from_numpy(trajectory_features.transform(summary,self.statistics["summary"]))
        with torch.inference_mode():
            embeddings = torch.cat([self.model.encode(x[start:start+512],z[start:start+512])
                                    for start in range(0,len(x),512)])
            logits = self.model.candidate_logits(embeddings,[w["member_rows"] for w in windows],
                                                 [w["center_row_index"] for w in windows])
            return torch.softmax(logits,-1).numpy()

    def predict(self, video, observations, original_candidates, external_candidates=()):
        started = time.perf_counter()
        pool,sequence,summary,windows,diagnostics = build_inputs(video,observations,original_candidates,external_candidates)
        probabilities = self.probabilities(sequence,summary,windows)
        events = select_events(pool,probabilities,minimum_separation_seconds=self.minimum_separation_seconds)
        return dict(events=events,candidates=pool,windows=windows,probabilities=probabilities,
            candidate_diagnostics=diagnostics,build_inputs_calls=1,decision_seconds=time.perf_counter()-started)
