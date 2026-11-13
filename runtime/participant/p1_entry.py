"""P1: official B0 plus one fixed native-PTS bounded amplitude fallback.

Official-compatible participant implementation. Explicit PTS are preserved;
when core supplies pts=None, read the actual video PTS via NativeTime.
The fixed P1 event algorithm and 80ms window are unchanged.
"""

from core.types import Landing
from .native_time import NativeTime
from .runtime_backend import prepare_detector
from participant.official_reference import (
    Solution as OfficialSolution, MIN_GAP, CONF_TH, FALL_AMP, RISE_AMP,
)

MAX_SIDE_POINTS = 2
MAX_PTS_DISTANCE_SECONDS = 0.080
PTS_COMPARISON_EPSILON_SECONDS = 1e-9


class Solution(OfficialSolution):
    def _reset_time_state(self):
        self.native_pts = {}
        self.current_input_frame = None
        self.current_phase = None
        self.emission_history = []

    def prepare(self, context):
        self.context = context
        prepare_detector(self, context)
        self._reset_time_state()

    def reset(self, video_meta):
        super().reset(video_meta)
        self._reset_time_state()
        self.time_source = NativeTime(getattr(self, "context", None), video_meta)

    def process_frame(self, frame_rgb, frame_meta):
        fid = frame_meta['frame_id']
        pts = self.time_source.resolve(frame_meta)
        self.native_pts[fid] = pts
        self.current_input_frame = fid
        self.current_phase = 'process_frame'
        # This invokes the unchanged official detector/confidence/readiness path.
        return super().process_frame(frame_rgb, {**frame_meta, 'pts': pts})

    def finish_video(self, video_meta):
        self.current_phase = 'finish_video'
        return super().finish_video(video_meta)

    def _fallback_window(self, j, step):
        center_fid, _, center_y, _ = self.traj[j]
        center_source = center_fid-1
        if center_source < 0:
            return []
        center_pts = self.native_pts[center_source]
        records = []
        for distance in range(1, MAX_SIDE_POINTS+1):
            k = j+step*distance
            if not 0 <= k < len(self.traj):
                break
            fid, _, y, _ = self.traj[k]
            source = fid-1
            if source < 0:
                continue
            # source frames exist because each trajectory point came through
            # process_frame; missing metadata is deliberately not imputed.
            pts = self.native_pts[source]
            delta = abs(pts-center_pts)
            if delta <= MAX_PTS_DISTANCE_SECONDS + PTS_COMPARISON_EPSILON_SECONDS:
                records.append(dict(input_frame_id=fid, source_frame_id=source,
                    source_pts=pts, distance_seconds=delta, center_minus_y=center_y-y))
        return records

    def _emit(self, need_future):
        out = []
        for j in range(1, len(self.traj)-need_future):
            fid, x, y, conf = self.traj[j]
            prev_y, next_y = self.traj[j-1][2], self.traj[j+1][2]
            peak = y > prev_y and y >= next_y
            fall, rise = y-prev_y, y-next_y
            original_amplitudes = fall >= FALL_AMP and rise >= RISE_AMP
            gap = fid-self.last_emit
            before_last_emit = self.last_emit
            before_window = after_window = []
            fallback_fall = fallback_rise = None
            branch = None
            if gap < MIN_GAP:
                reason = 'minimum_gap'
            elif not peak:
                reason = 'not_image_y_local_maximum'
            elif original_amplitudes:
                reason, branch = 'emit', 'original'
            else:
                before_window = self._fallback_window(j, -1)
                after_window = self._fallback_window(j, 1)
                fallback_fall = max((p['center_minus_y'] for p in before_window), default=None)
                fallback_rise = max((p['center_minus_y'] for p in after_window), default=None)
                if not before_window or not after_window:
                    reason = 'fallback_missing_side_within_80ms'
                elif fallback_fall < FALL_AMP:
                    reason = 'fallback_fall_below_2_native_px'
                elif fallback_rise < RISE_AMP:
                    reason = 'fallback_rise_below_1_native_px'
                else:
                    reason, branch = 'emit', 'fallback'
            record = dict(at_input_frame=self.current_input_frame, phase=self.current_phase,
                need_future=need_future, candidate_input_frame=fid,
                candidate_native_frame=max(0, fid-1),
                candidate_native_pts=self.native_pts.get(fid-1),
                last_emit_input_before=before_last_emit, gap_input_frames=gap,
                local_maximum=peak, original_fall_native_px=fall,
                original_rise_native_px=rise, original_amplitudes_pass=original_amplitudes,
                previous_accepted_input=self.traj[j-1][0], next_accepted_input=self.traj[j+1][0],
                fallback_previous_window=before_window, fallback_next_window=after_window,
                fallback_fall_native_px=fallback_fall, fallback_rise_native_px=fallback_rise,
                reason=reason, branch=branch)
            if reason == 'emit':
                out.append(Landing(frame_id=max(0, fid-1), x=x, y=y))
                self.last_emit = fid
                self.emission_history.append(record)
            sink = getattr(self, 'diagnostic_sink', None)
            if sink is not None:
                sink(record)
        return out
