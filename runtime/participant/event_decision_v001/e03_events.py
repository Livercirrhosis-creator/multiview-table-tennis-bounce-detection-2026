"""E03-S/R event generators on unchanged official detector observations.

No labels, image-angle hints, interpolated points, scene/pose model, or cache
paths are used here. Frame IDs and timestamps belong to observed ball positions.
"""
from dataclasses import asdict, dataclass
import math

TIME_EPS = 1e-9


@dataclass(frozen=True)
class EventConfig:
    variant: str = 'S'
    confidence_threshold: float = 0.75
    velocity_reference_fps: float = 25.0
    change_threshold: float = 3.0
    max_neighbor_gap_sec: float = 0.08
    min_event_gap_frames: int = 10
    peak_relative_tolerance: float = 1e-9
    extra_side_points: int = 2
    max_side_span_sec: float = 0.16

    def __post_init__(self):
        if self.variant not in ('S','R'):
            raise ValueError('Only the implemented S/R variants are accepted')
        numbers = (self.confidence_threshold, self.velocity_reference_fps,
                   self.change_threshold, self.max_neighbor_gap_sec,
                   self.peak_relative_tolerance,self.max_side_span_sec)
        if any(not math.isfinite(x) for x in numbers):
            raise ValueError('Configuration must be finite')
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError('Confidence threshold outside [0,1]')
        if min(self.velocity_reference_fps, self.change_threshold,
               self.max_neighbor_gap_sec) <= 0 or self.peak_relative_tolerance < 0:
            raise ValueError('Invalid score/time configuration')
        if not isinstance(self.min_event_gap_frames, int) or self.min_event_gap_frames < 1:
            raise ValueError('Event gap must be a positive integer')
        if self.extra_side_points != 2 or self.max_side_span_sec <= 0:
            raise ValueError('R design fixes two real observations per side')

    def to_dict(self):
        return asdict(self)


def _point(row):
    fid = row['frame_id']
    if not isinstance(fid, int) or isinstance(fid, bool) or fid < 0:
        raise ValueError('Observed frame ID must be a nonnegative integer')
    t = float(row['timestamp_sec'])
    if not math.isfinite(t):
        raise ValueError('Observed timestamp must be finite')
    if not row.get('visible', True):
        return dict(frame_id=fid, timestamp_sec=t, visible=False)
    values = tuple(float(row[k]) for k in ('x', 'y', 'confidence'))
    if not all(math.isfinite(v) for v in values) or not 0 <= values[2] <= 1:
        raise ValueError('Visible observation must have finite position and valid confidence')
    return dict(frame_id=fid, timestamp_sec=t, visible=True,
                x=values[0], y=values[1], confidence=values[2])


def generate_events(observations, config=None):
    """Return selected events and full diagnostics; no frame offset is applied."""
    cfg = config or EventConfig()
    rows = [_point(p) for p in observations]
    for a, b in zip(rows, rows[1:]):
        if b['frame_id'] <= a['frame_id'] or b['timestamp_sec'] <= a['timestamp_sec']:
            raise ValueError('Observations must have strictly increasing IDs and times')
    points = [p for p in rows if p['visible'] and p['confidence'] >= cfg.confidence_threshold]
    centers = [None] * len(points)
    for j in range(1, len(points)-1):
        a, p, b = points[j-1:j+2]
        dt_before = p['timestamp_sec']-a['timestamp_sec']
        dt_after = b['timestamp_sec']-p['timestamp_sec']
        support = dict(before_sec=dt_before, after_sec=dt_after)
        if max(dt_before, dt_after) > cfg.max_neighbor_gap_sec+TIME_EPS:
            centers[j] = dict(frame_id=p['frame_id'], score=None, reason='gap_support_unavailable', **support)
            continue
        incoming = [(p[k]-a[k])/(dt_before*cfg.velocity_reference_fps) for k in ('x','y')]
        outgoing = [(b[k]-p[k])/(dt_after*cfg.velocity_reference_fps) for k in ('x','y')]
        delta = [v-u for u,v in zip(incoming,outgoing)]
        score = math.hypot(*delta)
        if not math.isfinite(score):
            raise ValueError('Velocity change is not finite')
        raw_score = score
        extra_support = None
        attenuation = None
        extra_delta = None
        if cfg.variant == 'R':
            extra_support = False
            if j >= 2 and j+2 < len(points):
                seq = points[j-2:j+3]
                span = max(p['timestamp_sec']-seq[0]['timestamp_sec'],seq[-1]['timestamp_sec']-p['timestamp_sec'])
                gaps = [v['timestamp_sec']-u['timestamp_sec'] for u,v in zip(seq,seq[1:])]
                if span <= cfg.max_side_span_sec+TIME_EPS and max(gaps) <= cfg.max_neighbor_gap_sec+TIME_EPS:
                    # Independent two-point side slopes: the center is absent.
                    before = [(seq[1][k]-seq[0][k])/((seq[1]['timestamp_sec']-seq[0]['timestamp_sec'])*cfg.velocity_reference_fps) for k in ('x','y')]
                    after = [(seq[4][k]-seq[3][k])/((seq[4]['timestamp_sec']-seq[3]['timestamp_sec'])*cfg.velocity_reference_fps) for k in ('x','y')]
                    extra_delta = [v-u for u,v in zip(before,after)]
                    attenuation = max(0.0,min(1.0,sum(u*v for u,v in zip(delta,extra_delta))/(raw_score*raw_score))) if raw_score else 0.0
                    score = raw_score*attenuation
                    extra_support = True
        centers[j] = dict(frame_id=p['frame_id'], timestamp_sec=p['timestamp_sec'],
            x=p['x'], y=p['y'], confidence=p['confidence'], score=score,
            support_frame_ids=[a['frame_id'],p['frame_id'],b['frame_id']],
            incoming=incoming,outgoing=outgoing,delta=delta,raw_score=raw_score,
            extra_support=extra_support,extra_delta=extra_delta,attenuation=attenuation,**support)
    candidates = []
    rejected_peak = 0
    for j, center in enumerate(centers):
        if center is None or center['score'] is None or center['score'] < cfg.change_threshold:
            continue
        scores = []
        for idx in (j-1,j+1):
            other = centers[idx] if 0 <= idx < len(centers) else None
            if other is None or other['score'] is None or abs(
                    points[idx]['timestamp_sec']-points[j]['timestamp_sec']) > cfg.max_neighbor_gap_sec+TIME_EPS:
                scores.append(None)
            else:
                scores.append(other['score'])
        prev_score, next_score = scores
        def tolerance(other):
            return cfg.peak_relative_tolerance*max(1.0,abs(center['score']),abs(other))
        if prev_score is not None and center['score'] <= prev_score+tolerance(prev_score):
            rejected_peak += 1
            continue
        if next_score is not None and center['score'] < next_score-tolerance(next_score):
            rejected_peak += 1
            continue
        candidates.append(dict(center,peak_context_incomplete=None in scores,
                               neighbor_scores=scores))
    selected = []
    suppressed = []
    last_frame = -10000
    for candidate in candidates:
        if candidate['frame_id']-last_frame < cfg.min_event_gap_frames:
            suppressed.append(candidate)
            continue
        selected.append(candidate)
        last_frame = candidate['frame_id']
    return selected, dict(config=cfg.to_dict(),input_rows=len(rows),valid_points=len(points),
        gap_support_unavailable=sum(c is not None and c['score'] is None for c in centers),
        below_threshold=sum(c is not None and c['score'] is not None and c['score'] < cfg.change_threshold for c in centers),
        rejected_peak=rejected_peak,centers=[c for c in centers if c is not None],
        candidates=candidates,suppressed_gap=suppressed,selected=selected,
        incomplete_context_selected=sum(c['peak_context_incomplete'] for c in selected))
