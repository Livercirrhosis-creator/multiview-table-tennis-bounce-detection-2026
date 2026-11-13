"""Shared lifecycle wrapper; inherited complete event decisions run first."""
import json
from pathlib import Path
import time

from .model import TableDetector
from . import prior
from .sampling import Samples, duration_seconds


def retain_original_events(events, table_prior, config):
    # The old API consumes dictionaries. Use its keep decisions to return the
    # original objects: their field values, order and extra attributes survive.
    rows = [dict(frame_id=e.frame_id, x=e.x, y=e.y) for e in events]
    _, receipt = prior.select(rows, table_prior, config)
    return [event for event, decision in zip(events, receipt["decisions"]) if decision["kept"]], receipt


class TableGuardMixin:
    table_parent = None

    def prepare(self, context):
        super().prepare(context)
        started = time.perf_counter()
        extra = getattr(context, "extra", {})
        weights = extra.get("table_guard_weights", Path(__file__).resolve().parents[1]/
                            "weights/table_guard_v001/model_final.pth")
        self.table_detector = TableDetector(weights, extra.get("table_guard_metadata"),
                                            extra.get("table_guard_device", "cuda:0"))
        self.table_config = prior.Config()
        context.info["table_guard"] = dict(version="v001", parent=self.table_parent,
            model=self.table_detector.info, prior_config=self.table_config.__dict__,
            placement="after_complete_parent_finish_video", release="finish_video",
            existing_rgb_and_native_pts=True, maximum_forwards_per_video=1,
            maximum_images_per_video=3, prepare_seconds=time.perf_counter()-started)

    def reset(self, video_meta):
        old = getattr(self, "table_samples", None)
        if old is not None:
            old.clear()
        self.table_samples = Samples(self.table_config.max_video_seconds)
        self.table_finished = False
        super().reset(video_meta)

    def process_frame(self, frame_rgb, frame_meta):
        if self.table_finished:
            raise ValueError("A new video requires reset before process_frame")
        try:
            result = super().process_frame(frame_rgb, frame_meta)
            fid = frame_meta["frame_id"]
            self.table_samples.consume(frame_rgb, fid, self.native_pts[fid], self.time_source.file_pts)
            return result
        except Exception:
            self.table_samples.clear()
            raise

    def finish_video(self, video_meta):
        if self.table_finished:
            raise ValueError("finish_video was already called for this video")
        self.table_finished = True
        try:
            events = super().finish_video(video_meta)
            started = time.perf_counter()
            pts = [self.native_pts[f] for f in range(video_meta.n_frames)]
            duration = duration_seconds(pts)
            fps_adapter = len(pts)/duration if duration > 0 else 0.0
            samples = self.table_samples
            observations = []
            cost = dict(forward_calls=0, image_count=0, tensor_preprocessing_seconds=0.0,
                        synchronized_forward_seconds=0.0, prediction_to_cpu_seconds=0.0)
            if (0 < duration <= self.table_config.max_video_seconds
                    and all(s is not None for s in samples.slots)):
                observations, cost = self.table_detector.predict(samples.slots, video_meta.width, video_meta.height)
            tick = time.perf_counter()
            table_prior = prior.build_prior(observations, video_meta.width, video_meta.height,
                                           fps_adapter, len(pts), self.table_config)
            selected, decision = retain_original_events(events, table_prior, self.table_config)
            cost["post_gate_seconds"] = time.perf_counter()-tick
            receipt = dict(schema="table_guard_runtime.v1", video_id=video_meta.video_id,
                parent=self.table_parent, frame_count=len(pts), duration_seconds=duration,
                duration_rule="last PTS - first PTS + median positive native interval",
                fps_for_unchanged_prior_api_only=fps_adapter, sampling=samples.receipt(), cost=cost,
                decision=decision, table_eof_seconds_before_receipt_write=time.perf_counter()-started,
                model_weights_sha256=self.table_detector.info["weights_sha256"],
                video_wall_seconds_before_receipt_write=time.perf_counter()-self.video_started,
                cost_scope="Official runner includes parent chain, shared-frame sampling, table forward, final post gate and receipt write; preparation is separate")
            folder = Path(self.context.output_dir)/"table_guard_v001"/self.table_parent
            folder.mkdir(parents=True, exist_ok=True)
            (folder/(video_meta.video_id+".json")).write_text(
                json.dumps(receipt, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
            return selected
        finally:
            self.table_samples.clear()
