"""Explicit optional paired-observation candidate; original P1 capture retained."""
import json
from pathlib import Path
import time

from core.types import Landing
from participant.p1_pts_parallel import Solution as P1Solution
from participant.event_expert_fusion_v001.solution import Solution as CaptureSolution, DetectorObserver, media_sha256
from participant.masked_gap_v001.runtime import Runtime as GapRuntime
from .channel_detector import ChannelDetector
from .runtime import Runtime
from .selector import replace_observations


class Solution(CaptureSolution):
    def prepare(self, context):
        P1Solution.prepare(self, context)
        import torch
        torch.set_num_threads(2)
        weights = Path(__file__).resolve().parents[1]/'weights'
        names = ['channel_adaptation_v001/dense/model.pt', 'channel_adaptation_v001/dense/normalization.json',
                 'channel_adaptation_v001/window/model.pt', 'channel_adaptation_v001/window/normalization.json']
        assets = [weights/name for name in names]
        self.event_runtime = Runtime(*assets)
        self.gap_runtime = GapRuntime(weights/'masked_gap_v001/model.json')
        self.channel_detector = ChannelDetector(self.det, self)
        self.det = DetectorObserver(self.channel_detector, self)
        context.info['channel_adaptation'] = dict(version='v001', release='finish_video',
            original_p1_capture_preserved=True, original_window_support_preserved=True,
            gap_observation_basis='original', additional_visual_forwards=0,
            head_assets={name:media_sha256(path) for name,path in zip(names,assets)},
            alternate_rule='preserve ch1>=0.75; otherwise best same-frame ch0/ch2>=0.75, tie ch0',
            final_native_frame_excluded=True, nominal_fps_used=False,
            table_model_loaded=False, lifting_model_loaded=False)

    def finish_video(self, video_meta):
        anchors = self.finish_capture(video_meta)
        video = dict(clip_id=video_meta.video_id, width=video_meta.width, height=video_meta.height,
            frame_count=video_meta.n_frames, pts_seconds=[self.native_pts[f] for f in range(video_meta.n_frames)])
        try:
            tick = time.perf_counter()
            augmented, replacement = replace_observations(video, self.observations,
                                                          self.channel_detector.native_records(video))
            selection_seconds = time.perf_counter()-tick
            result = self.event_runtime.predict(video, self.observations, augmented, anchors)
            addition = self.gap_runtime.apply(video, self.observations, result['events'])
            events = [Landing(frame_id=e['frame_id'], x=e['x'], y=e['y']) for e in addition['events']]
            receipt = dict(video_id=video_meta.video_id, frame_count=video_meta.n_frames,
                native_clock=self.time_source.receipt(), original_observations=len(self.observations),
                original_precooldown_candidates=len(anchors), full_candidates=len(result['candidates']),
                selected_events=len(events), dense_events=len(result['dense_events']),
                window_events=len(result['window_events']), gap_events_added=addition['receipt']['added_event_count'],
                replacement=replacement, original_support=result['input_receipt'],
                decision_seconds=result['decision_seconds'], channel_selection_seconds=selection_seconds,
                gap_timing_seconds=addition['receipt']['timing_seconds'],
                video_wall_seconds=time.perf_counter()-self.video_started,
                cost_scope='Official runner inference includes visual capture, EOF adaptation and original-gap decisions')
            folder=Path(self.context.output_dir)/'channel_adaptation_v001'
            folder.mkdir(parents=True,exist_ok=True)
            (folder/(video_meta.video_id+'.json')).write_text(json.dumps(receipt,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
            return events
        finally:
            self.observations.clear()
            self.qualifying_candidates.clear()
            self.parent_events.clear()
            self.channel_detector.rows.clear()
            self.channel_detector.last_heat=None
