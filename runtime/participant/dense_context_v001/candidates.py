"""Retain the parent pool and add every reliable real native observation."""
from copy import deepcopy

from ..event_decision_v001.candidates import build_pool as build_parent_pool

DENSE_CONFIDENCE = .75
DENSE_SOURCE = "dense_native_confident"


def build_pool(video, observations, original_candidates, external_candidates=()):
    observations = list(observations)
    parent, parent_diagnostics = build_parent_pool(video, observations, original_candidates, external_candidates)
    pooled = {row["native_frame"]: row for row in parent}
    reliable_frames, added = [], []
    # The parent validates every observation's finite in-image center and clock.
    for observation in observations:
        detection = observation["detection"]
        if observation["center_is_padding"] or detection is None or detection["conf"] < DENSE_CONFIDENCE:
            continue
        frame = observation["native_frame"]
        reliable_frames.append(frame)
        if frame in pooled:
            continue
        pooled[frame] = dict(candidate_id=f"{video['clip_id']}:{frame}", video_id=video["clip_id"],
            native_frame=frame, native_pts_seconds=observation["native_pts_seconds"],
            width=video["width"], height=video["height"],
            center_observation=dict(input_frame=observation["input_frame"], **deepcopy(detection)),
            sources=[DENSE_SOURCE], source_evidence={DENSE_SOURCE:dict(confidence_minimum=DENSE_CONFIDENCE,
                nomination="Every reliable native observation; no anchor proximity requirement")},
            support_anchor_frames=[],
            availability=dict(phase="complete_trajectory_replay",online_release_established=False))
        added.append(frame)
    result = [pooled[frame] for frame in sorted(pooled)]
    diagnostics = dict(schema="dense_native_center_pool.v1",video_id=video["clip_id"],
        parent_candidates=len(parent),reliable_native_observations=len(reliable_frames),
        added_dense_candidates=len(added),pooled_candidates=len(result),
        inherited_low_confidence_candidates=sum(row["center_observation"]["conf"] < DENSE_CONFIDENCE for row in parent),
        parent_candidates_preserved=True,native_centers_unchanged=True,source_identity_is_model_feature=False,
        gt_used=False,parent=parent_diagnostics)
    return result, diagnostics
