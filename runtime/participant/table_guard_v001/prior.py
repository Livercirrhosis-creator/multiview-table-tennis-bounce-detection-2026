"""Bounded negative spatial evidence; no GT, model imports or fitted probability."""
from dataclasses import dataclass
from math import hypot, isfinite
from statistics import median


@dataclass(frozen=True)
class Config:
    enabled: bool = True
    minimum_observations: int = 3
    confidence_threshold: float = .5
    agreement_iou: float = .5
    padding_at_1080: float = 20.
    distance_scale_at_1080: float = 40.
    dispersion_multiplier: float = 2.
    penalty_strength: float = .8
    minimum_spatial_score: float = .5
    max_video_seconds: float = 10.

    def __post_init__(self):
        if self.minimum_observations != 3:
            raise ValueError('v001 expects exactly three sample observations')
        if not all(isfinite(v) for v in self.__dict__.values()):
            raise ValueError('Non-finite configuration')
        if not (0 < self.confidence_threshold <= 1 and 0 < self.agreement_iou <= 1
                and 0 < self.penalty_strength <= 1 and 0 < self.minimum_spatial_score <= 1
                and self.padding_at_1080 >= 0 and self.distance_scale_at_1080 > 0
                and self.dispersion_multiplier >= 0 and self.max_video_seconds > 0):
            raise ValueError('Invalid spatial configuration')


def iou(a, b):
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/max(union,1e-8)


def build_prior(observations, width, height, fps, n_frames, config):
    neutral=dict(usable=False,reason='disabled',bounds=None,reliability=0.,distance_scale=None,
                 proves_whole_table=False,observations=observations)
    if not config.enabled:
        return neutral
    if min(width,height,fps,n_frames) <= 0:
        return dict(neutral,reason='invalid_video_metadata')
    if n_frames/fps > config.max_video_seconds:
        return dict(neutral,reason='long_video_outside_v001_scope')
    if len(observations) != config.minimum_observations:
        return dict(neutral,reason='missing_sample_observations')
    boxes=[]
    for o in observations:
        b=o.get('box',[]); score=o.get('score',0.)
        if (len(b)!=4 or not all(isfinite(v) for v in b) or not isfinite(score)
                or score < config.confidence_threshold or score > 1
                or not (0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height)):
            return dict(neutral,reason='invalid_or_weak_table_observation')
        boxes.append(b)
    pairs=[dict(i=i,j=j,iou=iou(a,b)) for i,a in enumerate(boxes) for j,b in enumerate(boxes) if j>i]
    reliability=min(p['iou'] for p in pairs)
    if reliability < config.agreement_iou:
        return dict(neutral,reason='table_observations_disagree',pairs=pairs)
    raw=[median(b[k] for b in boxes) for k in range(4)]
    pad=config.padding_at_1080*height/1080
    bounds=[max(0.,raw[0]-pad),max(0.,raw[1]-pad),min(width,raw[2]+pad),min(height,raw[3]+pad)]
    dispersion=max(max(b[k] for b in boxes)-min(b[k] for b in boxes) for k in range(4))
    scale=max(config.distance_scale_at_1080*height/1080,config.dispersion_multiplier*dispersion)
    return dict(neutral,usable=True,reason='three_consistent_short_clip_observations',
                raw_box=raw,bounds=bounds,reliability=reliability,distance_scale=scale,
                coordinate_dispersion=dispersion,pairs=pairs,
                reliability_meaning='minimum pairwise box IoU, not a calibrated probability')


def select(events, prior, config):
    out=[]; decisions=[]
    for event in events:
        if not all(isfinite(event[k]) for k in ('x','y')):
            raise ValueError('Invalid event coordinates')
        d=0.;penalty=0.
        if prior['usable']:
            b=prior['bounds']
            d=hypot(max(b[0]-event['x'],0.,event['x']-b[2]),
                    max(b[1]-event['y'],0.,event['y']-b[3]))
            penalty=config.penalty_strength*prior['reliability']*min(1.,d/prior['distance_scale'])
        score=1.-penalty
        keep=score >= config.minimum_spatial_score
        if keep:
            out.append(dict(event))
        decisions.append(dict(event=dict(event),outside_distance_px=d,penalty=penalty,
                              spatial_score=score,kept=keep,reason='neutral_fallback' if not prior['usable'] else
                              'spatially_retained' if keep else 'strong_outside_evidence'))
    return out,dict(prior=prior,decisions=decisions,baseline_events=len(events),selected_events=len(out),
                    event_probability=False,post_gate=True,coordinate_adjustment=False)
