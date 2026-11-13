"""Preserve original proposal/support identities while enriching observations.

No labels, folds, model scores, files or nominal FPS enter this module. Original
missingness controls proposal support; enriched observations supply trajectory
features. They are two views of the same video, not independent examples.
"""
from copy import deepcopy
import numpy as np

from participant.dense_context_v001.candidates import build_pool, DENSE_SOURCE
from participant.event_decision_v001.candidates import _observation_index
from participant.event_decision_v001.native_features import extract_candidate
from participant.event_decision_v001.trajectory_features import augment
from participant.event_window_v001.windows import build_windows


def _features(video, observations, pool):
    if not pool:
        return np.empty((0, 15, 23)), np.empty((0, 44))
    observed = [dict(frame_id=o['native_frame'], pts=o['native_pts_seconds'], **o['detection'])
                for o in observations if not o['center_is_padding'] and o['detection'] is not None]
    features = [extract_candidate(observed, c['native_frame'], video['pts_seconds'],
                                  video['width'], video['height']) for c in pool]
    if any(f['status_2d'] != 'available' for f in features):
        raise ValueError('Candidate must have an exact observation in its own view')
    sequence = np.asarray([f['sequence_2d'] for f in features], dtype=np.float64)
    summary = np.asarray([f['summary_2d'] for f in features], dtype=np.float64)
    return sequence, augment(sequence, summary)


def build_view_inputs(video, original_observations, augmented_observations, anchors, *, view):
    """Return dense inputs, sparse-window row map/support, and provenance.

    Existing dense and sparse windows retain their original members and temporal
    bounds. Added reliable centers are dense-only singleton contexts. Their
    sequence features still encode the normal real-time trajectory neighborhood.
    Old low-confidence centers may acquire new XY/confidence; their identity and
    old singleton support remain. Gap proposals must separately use originals.
    """
    if view not in ('original', 'augmented'):
        raise ValueError('Unknown observation view')
    original_observations, augmented_observations = list(original_observations), list(augmented_observations)
    old_rows, _ = _observation_index(video, original_observations)
    new_rows, _ = _observation_index(video, augmented_observations)
    for frame, row in old_rows.items():
        point = row['detection']
        if point is not None and point['conf'] >= .75 and new_rows[frame]['detection'] != point:
            raise ValueError('A reliable original observation was replaced')
    original_pool, diagnostics = build_pool(video, original_observations, anchors)
    original_windows = build_windows(video, original_observations, original_pool)
    original_parent = [c for c in original_pool if DENSE_SOURCE not in c['sources']]
    parent_windows = build_windows(video, original_observations, original_parent)
    observations = original_observations if view == 'original' else augmented_observations
    rows = old_rows if view == 'original' else new_rows
    pool = deepcopy(original_pool)
    original_frames = {c['native_frame'] for c in pool}
    changed_centers, added = [], []
    if view == 'augmented':
        for candidate in pool:
            frame = candidate['native_frame']
            point = rows[frame]['detection']
            if point is None:
                raise ValueError('Enrichment removed an original candidate observation')
            if point != old_rows[frame]['detection']:
                changed_centers.append(frame)
                candidate['center_observation'] = dict(input_frame=rows[frame]['input_frame'], **point)
        for frame, row in rows.items():
            point = row['detection']
            if frame in original_frames or point is None or point['conf'] < .75:
                continue
            added.append(frame)
            pool.append(dict(candidate_id=f"{video['clip_id']}:{frame}", video_id=video['clip_id'],
                native_frame=frame, native_pts_seconds=row['native_pts_seconds'],
                width=video['width'], height=video['height'],
                center_observation=dict(input_frame=row['input_frame'], **point),
                sources=[DENSE_SOURCE], source_evidence={DENSE_SOURCE:dict(
                    nomination='Reliable alternate temporal channel at an originally absent candidate',
                    confidence_minimum=.75, original_observation=deepcopy(old_rows[frame]['detection']))},
                support_anchor_frames=[], availability=dict(phase='complete_trajectory_replay',
                    online_release_established=False)))
        pool.sort(key=lambda c:c['native_frame'])
    by_frame = {c['native_frame']:i for i,c in enumerate(pool)}
    old_to_new = {i:by_frame[c['native_frame']] for i,c in enumerate(original_pool)}
    windows_by_frame = {w['center_frame']:dict(w,
        window_index=old_to_new[w['window_index']], center_row_index=old_to_new[w['center_row_index']],
        member_rows=[old_to_new[j] for j in w['member_rows']]) for w in original_windows}
    for frame in added:
        i, timestamp = by_frame[frame], rows[frame]['native_pts_seconds']
        windows_by_frame[frame] = dict(window_index=i, window_id=f"{video['clip_id']}:w{frame}",
            video_id=video['clip_id'], center_row_index=i, center_frame=frame,
            start_time=timestamp, end_time=timestamp, member_rows=[i], segment_id=-(frame+1))
    windows = [windows_by_frame[c['native_frame']] for c in pool]
    sequence, summary = _features(video, observations, pool)
    parent_rows = [by_frame[c['native_frame']] for c in original_parent]
    receipt = dict(view=view, original_candidates=len(original_pool), candidates=len(pool),
        added_candidate_frames=added, changed_original_center_frames=changed_centers,
        original_support_preserved=True, original_parent_candidates=len(original_parent),
        original_reliable_frames=[f for f,r in old_rows.items() if r['detection'] is not None and r['detection']['conf'] >= .75],
        originally_missing_frames=[f for f,r in old_rows.items() if r['detection'] is None],
        source_identity_is_model_feature=False, new_mask_feature_added=False,
        gap_observation_basis='original', gt_used=False)
    return dict(inputs=(pool, sequence, summary, windows, diagnostics),
                window_rows=parent_rows, window_windows=parent_windows, receipt=receipt)


def build_dual_inputs(video, original_observations, augmented_observations, anchors):
    views = {view:build_view_inputs(video, original_observations, augmented_observations, anchors, view=view)
             for view in ('original', 'augmented')}
    return dict(original=views['original']['inputs'], augmented=views['augmented']['inputs'],
                window_rows={v:r['window_rows'] for v,r in views.items()},
                window_windows={v:r['window_windows'] for v,r in views.items()},
                receipt={v:r['receipt'] for v,r in views.items()})
