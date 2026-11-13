"""Apply the frozen calibrated visual candidate to the existing CHANNEL runtime."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from participant.event_decision_v001.selection import select_events
from participant.event_union_v001.runtime import combine_events

from . import rgb
from .calibration import calibrate, fuse
from .model import CLASSES, WEIGHTS_SHA256, ContactVisualModel
from .shortlist import shortlist_ids


BATCH_SIZE = 32
RECIPE_ID = 'VISUAL_CALIBRATED_SCALE_AWARE_FINAL154_V001'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def score_candidates(model, device, frames, pts, candidates, height, *, batch_size=BATCH_SIZE):
    """Bound RGB/activation memory; each batch uses the unchanged full model."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Expected a positive candidate batch size')
    device = torch.device(device)
    if model.training:
        raise ValueError('Visual inference requires evaluation mode')
    scores = np.empty((len(candidates), 3), dtype=np.float32)
    costs = dict(rgb_seconds=0., head_seconds=0., batches=0, maximum_batch_candidates=0)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        # Match the original fit/extraction's FP32 CUDA policy, restoring the
        # surrounding application's settings after the visual calls.
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode(), torch.backends.cudnn.flags(
                benchmark=False, deterministic=False, allow_tf32=False):
            for first in range(0, len(candidates), batch_size):
                batch = candidates[first:first + batch_size]
                tick = perf_counter()
                pixels, times = rgb.inputs(frames, pts, batch, height)
                costs['rgb_seconds'] += perf_counter() - tick
                tick = perf_counter()
                logits = model(torch.from_numpy(pixels).to(device), torch.from_numpy(times).to(device))
                values = torch.softmax(logits, -1).cpu().numpy()
                if values.shape != (len(batch), 3) or not np.isfinite(values).all():
                    raise ValueError('Invalid visual model output')
                scores[first:first + len(batch)] = values
                costs['head_seconds'] += perf_counter() - tick
                costs['batches'] += 1
                costs['maximum_batch_candidates'] = max(costs['maximum_batch_candidates'], len(batch))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return scores, costs


class Runtime:
    def __init__(self, base, frames, checkpoint, calibration_path, *, device='cuda:0', batch_size=BATCH_SIZE):
        self.base, self.frames = base, frames
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('The frozen visual deployment requires CUDA')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Expected a positive candidate batch size')
        self.batch_size = batch_size
        configuration = json.loads(Path(calibration_path).read_text(encoding='utf-8-sig'))
        self.a = float(configuration['a'])
        checkpoint_sha = sha256(checkpoint)
        if (configuration['schema'] != 'final154_visual_calibration.v1'
                or configuration['recipe_id'] != RECIPE_ID
                or configuration['model_sha256'] != checkpoint_sha
                or tuple(configuration['classes']) != CLASSES
                or configuration['pretrained_sha256'] != WEIGHTS_SHA256
                or configuration['calibration_videos'] != 121
                or configuration['visual_training_videos'] != 154
                or configuration['calibration_sources'] != 10
                or configuration['fit_scope'] != 'development121_outer_oof_only;visual_model_final154'
                or not np.isfinite(self.a) or not 0 <= self.a <= 1):
            raise ValueError('Frozen final154 calibration identity or training scope differs')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (tuple(state['classes']) != CLASSES or state['pretrained_sha256'] != WEIGHTS_SHA256
                or state['training'] != configuration['training']
                or len(set(state['train_video_ids'])) != 154 or state['held_out_video_ids']):
            raise ValueError('Final visual checkpoint recipe or fitting scope differs')
        self.model = ContactVisualModel()
        self.model.load_state_dict(state['model_state_dict'], strict=True)
        self.model.requires_grad_(False).to(self.device).eval()
        self.metadata = dict(recipe_id=RECIPE_ID, a=self.a, model_sha256=checkpoint_sha,
            calibration_sha256=sha256(calibration_path), pretrained_sha256=WEIGHTS_SHA256,
            model_source_sha256=sha256(Path(__file__).with_name('model.py')),
            calibration_source_sha256=sha256(Path(__file__).with_name('calibration.py')),
            fit_scope=configuration['fit_scope'], calibration_videos=121, visual_training_videos=154,
            training_videos=154,
            device=str(self.device), candidate_batch_size=self.batch_size,
            crop_size=96, crop_interpolation='PIL BILINEAR', physical_minimum_side=32,
            shortlist_boundary='strict', fusion='calibrated_scale_aware')

    def predict(self, video, original_observations, augmented_observations, anchors):
        started = perf_counter()
        result = self.base.predict(video, original_observations, augmented_observations, anchors)
        pool = result['candidates']
        receipt = dict(frames_retained=len(self.frames), candidates=len(pool), shortlisted=0,
            rgb_seconds=0., head_seconds=0., batches=0, maximum_batch_candidates=0,
            a=self.a, device=str(self.device), candidate_batch_size=self.batch_size,
            fallback=None, second_decode=False)
        result['input_receipt']['visual_calibration'] = receipt
        try:
            if self.a == 0:
                receipt['fallback'] = 'visual_scale_zero'
                return result
            if not self.frames:
                receipt['fallback'] = 'raw_frame_memory_limit'
                return result
            pts = video['pts_seconds']
            if len(self.frames) != len(pts):
                raise ValueError('Visual frame retention does not match the native clock')
            separation = self.base.minimum_separation_seconds
            ids = set(shortlist_ids(pool, result['probabilities'], result['dense_events'],
                                    result['window_events'], separation, include_boundary=False))
            indices = [i for i, row in enumerate(pool) if row['candidate_id'] in ids]
            receipt['shortlisted'] = len(indices)
            if not indices:
                return result
            scores, costs = score_candidates(self.model, self.device, self.frames, pts,
                [pool[i] for i in indices], video['height'], batch_size=self.batch_size)
            receipt.update(costs)
            probabilities = result['probabilities'].copy()
            probabilities[indices] = fuse(probabilities[indices], calibrate(scores, self.a), self.a,
                                          scale_aware=True)
            dense = select_events(pool, probabilities, minimum_separation_seconds=separation)
            result.update(dense_events=dense, probabilities=probabilities,
                events=combine_events(dense, result['window_events'], minimum_separation_seconds=separation))
            return result
        finally:
            self.frames.clear()
            result['decision_seconds'] = perf_counter() - started
