"""Explicit full-video research entry; original framework and P1 remain intact.

The first integration re-decodes the original input once at EOF for clean spatial
evidence. Its decode, registration and final decision costs count as inference.
No development manifest, video-specific rule or label is accessed here.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from core.types import Landing
from ..p1_pts_parallel import Solution as ParentSolution
from ..official_reference import FALL_AMP, RISE_AMP, CONF_TH


def media_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded_frames(path):
    """CPU PyAV RGB24 policy used by the clean evidence collection run."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame_id, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                raise ValueError("Missing timestamp or corrupt native RGB frame")
            yield dict(frame_id=frame_id, pts_seconds=float(frame.pts * frame.time_base),
                       rgb=frame.to_ndarray(format="rgb24"))


class DetectorObserver:
    def __init__(self, detector, owner):
        self.detector, self.owner = detector, owner

    def reset(self):
        return self.detector.reset()

    def detect(self, image, width, height):
        value = self.detector.detect(image, width, height)
        owner = self.owner
        fid = owner.current_input_frame
        if fid != len(owner.observations) or (width, height) != (
                owner.video_meta.width, owner.video_meta.height):
            raise ValueError("Detector observation order or dimensions changed")
        # ch1 input f is native f-1. Input zero is padding, never a real center.
        native = fid - 1 if fid else None
        owner.observations.append(dict(video_id=owner.video_meta.video_id,
            input_frame=fid, input_pts_seconds=owner.native_pts[fid], native_frame=native,
            native_pts_seconds=owner.native_pts[native] if native is not None else None,
            center_is_padding=native is None, width=width, height=height,
            detection=dict(value) if value is not None else None))
        return value


class Solution(ParentSolution):
    """P1 capture, clean dense spatial perception, then spatial EOF decision."""
    mode = "spatial"

    def prepare(self, context):
        super().prepare(context)  # The original ball warmup is not an observation.
        from ..reference_native_evidence_v001.models import PublicModels
        weights = Path(__file__).resolve().parents[1] / "weights"
        heads = weights / "event_expert_final_v001"
        assets = [heads / "spatial.pt", heads / "spatial_normalization.json"]
        if self.mode == "spatial":
            from ..spatial_event_context_v001.runtime import Runtime
            self.event_runtime = Runtime(*assets)
        elif self.mode == "fusion":
            from .runtime import Runtime
            assets.extend([heads / "trajectory.pt", heads / "trajectory_normalization.json"])
            self.event_runtime = Runtime(trajectory_weights=assets[2],
                trajectory_statistics=assets[3], spatial_weights=assets[0],
                spatial_statistics=assets[1])
        else:
            raise ValueError("Unknown fixed event runtime mode")
        self.spatial_models = PublicModels(table_weights=weights / "public/table_keypoints_b2.pt",
            lifting_weights=weights / "public/uplifting_ours.pt", device="cuda:0", batch_size=16)
        self.det = DetectorObserver(self.det, self)
        context.info["event_expert_fusion"] = dict(
            version="v001", release="finish_video", decision_mode=self.mode,
            spatial_rgb="explicit original-input CPU re-decode at EOF",
            reference_policy="repair", minimum_separation_seconds=.08,
            head_assets={p.name: media_sha256(p) for p in assets},
            public_models=self.spatial_models.receipt())

    def reset(self, video_meta):
        super().reset(video_meta)
        self.video_meta = video_meta
        self.video_started = time.perf_counter()
        self.observations = []
        self.qualifying_candidates = {}
        self.parent_events = []
        self.diagnostic_sink = self._inspect_candidate

    def _emit(self, need_future):
        self._candidate_positions = {row[0]: j for j, row in enumerate(self.traj)}
        return super()._emit(need_future)

    def _inspect_candidate(self, record):
        """Same first-qualification rule as the existing 921-candidate audit.

        P1 skips fallback amplitudes after a gap rejection; inspect the amplitude
        window independently so suppressed anchors reach the final decision.
        """
        fid = record["candidate_input_frame"]
        if fid in self.qualifying_candidates or not record["local_maximum"]:
            return
        j = self._candidate_positions[fid]
        if not record["original_amplitudes_pass"]:
            before, after = self._fallback_window(j, -1), self._fallback_window(j, 1)
            if (not before or not after or
                    max(p["center_minus_y"] for p in before) < FALL_AMP or
                    max(p["center_minus_y"] for p in after) < RISE_AMP):
                return
        self.qualifying_candidates[fid] = dict(record)

    def process_frame(self, frame_rgb, frame_meta):
        self.parent_events.extend(super().process_frame(frame_rgb, frame_meta))
        return []

    def finish_capture(self, video_meta):
        """Complete the unchanged parent and expose only original observations."""
        self.parent_events.extend(super().finish_video(video_meta))
        if (len(self.observations) != video_meta.n_frames or
                len(self.native_pts) != video_meta.n_frames):
            raise ValueError("Incomplete original native observation stream")
        candidates = []
        for fid, record in sorted(self.qualifying_candidates.items()):
            row = self.observations[fid]
            native, at, phase = row["native_frame"], record["at_input_frame"], record["phase"]
            if native is None or row["detection"] is None:
                raise ValueError("Pre-cooldown center lacks its original observation")
            availability = dict(phase=phase, recorded_at_input_frame=at)
            if phase == "process_frame":
                availability.update(available_after_native_input_frame=at,
                    available_input_pts_seconds=self.native_pts[at],
                    basis="Recorded first qualification after processing this native input frame; media time, not wall-clock latency")
            else:
                availability.update(available_after_native_input_frame=None,
                    available_input_pts_seconds=None, requires_finish_video=phase == "finish_video",
                    basis="Finish-only or insufficient record: exact availability media time unknown; last-input value is retained but not substituted")
            candidates.append(dict(candidate_id=f"{video_meta.video_id}:{native}",
                video_id=video_meta.video_id, native_frame=native,
                native_pts_seconds=row["native_pts_seconds"], width=video_meta.width,
                height=video_meta.height, center_observation=dict(input_frame=fid,
                    **row["detection"]), availability=availability))
        return candidates

    def _extract_spatial_evidence(self, video, observations, centers, path):
        """One explicit decode; a narrow seam for the no-model integration test."""
        from ..reference_native_evidence_v001.evidence import extract_video
        frames = decoded_frames(path)
        try:
            return extract_video(meta=dict(video_id=video["clip_id"],
                **{k: video[k] for k in ("width", "height", "frame_count", "media_sha256")}),
                native_pts=video["pts_seconds"], observations=observations, candidates=centers,
                frames=frames, models=self.spatial_models, reference_policy="repair")
        finally:
            frames.close()

    def finish_video(self, video_meta):
        originals = self.finish_capture(video_meta)
        path = Path(self.context.input_dir) / video_meta.path
        pts = [self.native_pts[f] for f in range(video_meta.n_frames)]
        native_observations = [dict(frame_id=o["native_frame"],
            pts_seconds=o["native_pts_seconds"], **o["detection"])
            for o in self.observations if not o["center_is_padding"] and o["detection"] is not None]
        centers = [dict(candidate_id=f"{video_meta.video_id}:{o['frame_id']}", **o)
                   for o in native_observations if o["conf"] >= CONF_TH]
        video = dict(clip_id=video_meta.video_id, width=video_meta.width,
            height=video_meta.height, frame_count=video_meta.n_frames,
            media_sha256=media_sha256(path), pts_seconds=pts)
        try:
            evidence = self._extract_spatial_evidence(video, native_observations, centers, path)
            tick = time.perf_counter()
            result = self.event_runtime.predict(video, self.observations, originals, evidence,
                {p.frame_id for p in self.parent_events})
            decision_seconds = time.perf_counter() - tick
            events = [Landing(frame_id=p["frame_id"], x=p["x"], y=p["y"])
                      for p in result["events"]]
            receipt = dict(video_id=video_meta.video_id, decision_mode=self.mode,
                frame_count=video_meta.n_frames,
                media_sha256=video["media_sha256"], native_clock=self.time_source.receipt(),
                raw_observations=len(self.observations), precooldown_candidates=len(originals),
                parent_outputs=len(self.parent_events), dense_spatial_centers=len(centers),
                full_candidates=len(result["candidates"]), selected_events=len(events),
                reference=evidence["reference"],
                spatial_dispositions=dict(Counter(w["status"] for w in evidence["windows"])),
                spatial_timing_seconds=evidence["timing_seconds"],
                spatial_resources=evidence["resource_policy"],
                spatial_rgb_sequence_sha256=evidence["image_identity"]["rgb_sequence_sha256"],
                event_decision_seconds=decision_seconds,
                video_wall_seconds=time.perf_counter()-self.video_started,
                cost_scope="Video lifecycle including EOF spatial CPU re-decode; official runner determines inference FPS")
            folder = Path(self.context.output_dir) / "event_expert_fusion"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / (video_meta.video_id + ".json")).open("w", encoding="utf-8") as stream:
                json.dump(receipt, stream, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            return events
        finally:
            self.observations.clear()
            self.qualifying_candidates.clear()
            self.parent_events.clear()


class FusionSolution(Solution):
    """Optional research comparison; requires both frozen final expert weights."""
    mode = "fusion"
