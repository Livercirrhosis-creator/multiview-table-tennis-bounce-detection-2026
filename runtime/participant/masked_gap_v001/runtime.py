"""Infer missing-contact proposals without changing the existing 2D events."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
import time

import numpy as np

from ..gap_contact_v001.proposal import CONFIG, propose
from .features import extract
from .model import predict as classify


def diagnostic_native_pts_gaps(video):
    """Restore the exact cache diagnostic from actual PTS, not nominal FPS.

    A long interval is a diagnostic discontinuity, not proof of missing frames.
    This is the development cache rule; it is not an arbitrary-VFR guarantee.
    """
    pts = video["pts_seconds"]
    intervals = [b - a for a, b in zip(pts, pts[1:])]
    if not intervals:
        return []
    threshold = median(intervals) * 1.5 + 1e-8
    return [dict(before_frame_id=i, after_frame_id=i + 1,
                 before_pts_seconds=pts[i], after_pts_seconds=pts[i + 1],
                 delta_seconds=dt, at_final_transition=i == len(pts) - 2)
            for i, dt in enumerate(intervals) if dt > threshold]


class Runtime:
    def __init__(self, weights, *, base=None, receipt_dir=None):
        if isinstance(weights, (str, Path)):
            self.weights = json.loads(Path(weights).read_text(encoding="utf-8-sig"))
        else:
            self.weights = dict(weights)
        if self.weights["classes"] != ["table_contact", "racket_hit", "background"]:
            raise ValueError("Expected the frozen masked-gap class order")
        self.base = base
        self.receipt_dir = None if receipt_dir is None else Path(receipt_dir)

    def apply(self, video, observations, old_events):
        """Append classified proposals; old events retain every field unchanged."""
        started = time.perf_counter()
        observations = list(observations)
        diagnostic_gaps = diagnostic_native_pts_gaps(video)
        video = dict(video, source_pts_gaps=diagnostic_gaps)
        tick = time.perf_counter()
        proposal = propose(video, observations)
        proposal_seconds = time.perf_counter() - tick
        gaps = {(gap["left_frame"], gap["right_frame"]): gap for gap in proposal["gaps"]}
        tick = time.perf_counter()
        candidates = proposal["candidates"]
        by_frame = {row["native_frame"]: row for row in observations if not row["center_is_padding"]}
        features = []
        for candidate in candidates:
            gap = gaps[(candidate["left_frame"], candidate["right_frame"])]
            support = [by_frame[frame] for frame in gap["left_support_frames"] + gap["right_support_frames"]]
            features.append(extract(video, support, candidate, gap))
        feature_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        decisions = []
        if features:
            logits, probabilities = classify(np.asarray(features), self.weights)
            for candidate, scores, probability in zip(candidates, logits, probabilities):
                decisions.append(dict(candidate_id=candidate["candidate_id"],
                    predicted_class=int(np.argmax(scores)), logits=scores.tolist(),
                    probabilities=probability.tolist(), margin=float(scores[0] - max(scores[1], scores[2]))))
        classification_seconds = time.perf_counter() - tick
        by_id = {decision["candidate_id"]: decision for decision in decisions}
        tick = time.perf_counter()
        current = list(old_events)
        original_count = len(current)
        accepted, rejected = [], []
        for candidate in sorted(candidates, key=lambda c: (-by_id[c["candidate_id"]]["margin"],
                                                           c["pts_seconds"], c["frame_id"])):
            decision = by_id[candidate["candidate_id"]]
            if decision["predicted_class"] != 0:
                rejected.append(dict(candidate_id=candidate["candidate_id"],
                                     reason="classified_non_table", predicted_class=decision["predicted_class"]))
                continue
            conflicts = [event["candidate_id"] for event in current
                         if abs(event["pts_seconds"] - candidate["pts_seconds"]) < CONFIG["final_spacing_seconds"]]
            if conflicts:
                rejected.append(dict(candidate_id=candidate["candidate_id"],
                                     reason="minimum_interval_conflict", conflicts=conflicts))
            else:
                current.append(dict(candidate, selected_model="masked_gap_v001",
                                    probabilities=decision["probabilities"], margin=decision["margin"]))
                accepted.append(candidate["candidate_id"])
        current.sort(key=lambda event: (event["pts_seconds"], event["frame_id"]))
        admission_seconds = time.perf_counter() - tick
        receipt = dict(schema="masked_gap_runtime.v1", video_id=video["clip_id"],
            decision_mode="dense_context_union_masked_gap", source_gap_origin="diagnostic_native_pts_gap",
            source_gap_rule="delta_seconds > 1.5 * median(native_delta_seconds) + 1e-8",
            source_pts_gaps=diagnostic_gaps, proposal_counts=proposal["counts"],
            proposals=candidates, gaps=proposal["gaps"], decisions=decisions,
            accepted_ids=accepted, rejected=rejected, old_event_count=original_count,
            added_event_count=len(accepted), new_event_count=len(current),
            original_2d_events_preserved=True, minimum_interval_seconds=CONFIG["final_spacing_seconds"],
            timing_seconds=dict(proposal=proposal_seconds, features=feature_seconds,
                classification=classification_seconds, admission=admission_seconds,
                gap_total=time.perf_counter() - started),
            cost_scope="Extra cached-trajectory decision only; official runner determines full inference FPS")
        return dict(events=current, receipt=receipt)

    def predict(self, video, observations, original_candidates, external_candidates=()):
        """Wrap the base API and retain all its non-event result fields."""
        started = time.perf_counter()
        observations = list(observations)
        result = self.base.predict(video, observations, original_candidates, external_candidates)
        addition = self.apply(video, observations, result["events"])
        receipt = addition["receipt"]
        receipt["timing_seconds"].update(base_decision=result["decision_seconds"],
                                         wrapper_decision=time.perf_counter() - started)
        if self.receipt_dir is not None:
            self.receipt_dir.mkdir(parents=True, exist_ok=True)
            path = self.receipt_dir / (video["clip_id"] + ".json")
            path.write_text(json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        return dict(result, events=addition["events"])
