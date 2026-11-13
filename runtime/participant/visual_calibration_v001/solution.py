"""Explicit calibrated visual candidate over the unchanged CHANNEL_TABLE chain."""
from pathlib import Path

from participant.table_guard_v001.channel_solution import Solution as ChannelTable

from .runtime import BATCH_SIZE, Runtime


FRAME_MEMORY_LIMIT = 1536 * 1024 * 1024


class Solution(ChannelTable):
    def prepare(self, context):
        super().prepare(context)
        self.visual_frames = []
        weights = Path(__file__).resolve().parents[1] / 'weights/visual_calibration_v001'
        self.event_runtime = Runtime(self.event_runtime, self.visual_frames,
            weights / 'model.pt', weights / 'calibration.json', device='cuda:0', batch_size=BATCH_SIZE)
        context.info['visual_calibration'] = dict(self.event_runtime.metadata,
            frame_memory_limit_bytes=FRAME_MEMORY_LIMIT, retains_existing_decoder_rgb=True,
            second_decode=False, long_video_policy='unchanged CHANNEL_TABLE when retained RGB exceeds limit')

    def reset(self, video_meta):
        self.visual_frames.clear()
        self.retain_visual_frames = video_meta.width * video_meta.height * 3 * video_meta.n_frames <= FRAME_MEMORY_LIMIT
        super().reset(video_meta)

    def process_frame(self, frame_rgb, frame_meta):
        try:
            result = super().process_frame(frame_rgb, frame_meta)
            if self.retain_visual_frames:
                # Reuse the distinct array yielded by the existing decoder.
                self.visual_frames.append(frame_rgb)
            return result
        except Exception:
            self.visual_frames.clear()
            raise

    def finish_video(self, video_meta):
        try:
            return super().finish_video(video_meta)
        finally:
            self.visual_frames.clear()
