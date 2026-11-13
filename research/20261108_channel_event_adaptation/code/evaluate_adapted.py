"""Frozen full-chain LOFO comparison, using original gaps for all arms."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
BASE = ROOT/'experiments/20261107_masked_gap_decision/runs/packet_v001/bundle/version/participant/pingpang'
NEW = EXP/'versions/v001/participant/pingpang/participant'
RESERVOIR = ROOT/'experiments/20261103_dual_path_rebuild/runs/reservoir_v001'
GAP = ROOT/'experiments/20261107_masked_gap_decision/runs/lofo_v001'
SCORER = ROOT/'experiments/20261102_p1_time_selection/runs/coverage_mapping_v001/sealed_source/evaluation/evaluate_timeline.py'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path):
    data = np.load(path/'features.npz', allow_pickle=False)
    index, windows = lines(path/'index.jsonl'), read(path/'windows.json')
    groups, wg = defaultdict(list), defaultdict(list)
    for r in index:
        groups[(r['video_id'], r['view'])].append(r)
    for w in windows:
        wg[(w['video_id'], w['view'])].append(w)
    return dict(sequence=data['sequence'], summary=data['summary'], groups=groups, windows=wg)


def physical_row(row, local):
    result = dict(row, row_index=local, candidate_id=row.get('physical_candidate_id', row['candidate_id']))
    return result


def prepare_inputs(dense, window, vid, view):
    rows, ws = dense['groups'][(vid, view)], dense['windows'][(vid, view)]
    dlocal = {r['row_index']:i for i,r in enumerate(rows)}
    frames = {r['native_frame']:i for i,r in enumerate(rows)}
    ids = [r['row_index'] for r in rows]
    pool = [physical_row(r, i) for i,r in enumerate(rows)]
    dw = [dict(w, window_index=i, center_row_index=dlocal[w['center_row_index']],
               member_rows=[dlocal[j] for j in w['member_rows']]) for i,w in enumerate(ws)]
    wr = window['groups'][(vid, view)]
    wlocal = {r['row_index']:i for i,r in enumerate(wr)}
    parent_rows = [frames[r['native_frame']] for r in wr]
    wi = [r['row_index'] for r in wr]
    if not np.array_equal(dense['sequence'][ids][parent_rows], window['sequence'][wi]):
        raise ValueError('Dense/window shared sequence mismatch')
    if not np.array_equal(dense['summary'][ids][parent_rows], window['summary'][wi]):
        raise ValueError('Dense/window shared summary mismatch')
    ww = [dict(w, window_index=i, window_id=w.get('physical_window_id', w['window_id']),
        center_row_index=wlocal[w['center_row_index']], member_rows=[wlocal[j] for j in w['member_rows']])
        for i,w in enumerate(window['windows'][(vid, view)])]
    return (pool, dense['sequence'][ids], dense['summary'][ids], dw, {}), parent_rows, ww


def identity(events):
    return [(r['candidate_id'], r['frame_id'], r['pts_seconds'], r['x'], r['y']) for r in events]


def compact(result):
    return dict(metrics=result['metrics_on_evaluable_regions'],
        source_macro_f1=float(np.mean([r['f1'] for r in result['per_camera_family']])),
        sources=[{k:r[k] for k in ('camera_family','tp','fp','fn','f1')} for r in result['per_camera_family']],
        total_predictions=result['total_predictions'])


def difference(before, after):
    old = {r['event_id'] for r in before['matching']['pairs']}
    new = {r['event_id'] for r in after['matching']['pairs']}
    bm, am = before['metrics_on_evaluable_regions'], after['metrics_on_evaluable_regions']
    return dict(recovered_event_ids=sorted(new-old), lost_event_ids=sorted(old-new),
        delta={k:am[k]-bm[k] for k in ('tp','fp','fn','f1')},
        recovered=len(new-old), lost=len(old-new))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=EXP/'runs/dataset_v002')
    p.add_argument('--dense-run', type=Path, default=EXP/'runs/lofo_parallel_v001/dense')
    p.add_argument('--window-run', type=Path, default=EXP/'runs/lofo_v001/window')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), command=sys.argv,
                  completed=False, protected33_used=False, public15_used=False)
    save(a.output/'run_status.json', status)
    try:
        sys.path.insert(0, str(BASE))
        import torch
        torch.set_num_threads(1)
        import participant
        participant.__path__.append(str(NEW))
        from participant.channel_adaptation_v001.runtime import Runtime
        from participant.masked_gap_v001.runtime import Runtime as GapRuntime
        dense, window = load_dataset(a.dataset/'dense'), load_dataset(a.dataset/'window')
        videos = {v['clip_id']:v for v in read(RESERVOIR/'runtime/videos.json')['clips']}
        allowed = set(read(ROOT/'experiments/20261102_research_foundation/data/development_videos.json')['video_ids'])
        assert set(videos) == allowed and len(allowed) == 121
        observations = defaultdict(list)
        for r in lines(RESERVOIR/'runtime/observations.jsonl'):
            assert r['video_id'] in allowed
            observations[r['video_id']].append(r)
        reference = {r['video_id']:r['events'] for r in read(GAP/'predictions_by_video.json')}
        arms = {k:[] for k in ('old_weights_retained_support','paired_enriched','paired_original_diagnostic')}
        timing, weights = [], {}
        for fold in read(RESERVOIR/'splits/lofo_by_video.json')['folds']:
            fid = fold['fold_id']
            od = ROOT/f'experiments/20261105_coverage_aware_events/runs/lofo_v001/folds/{fid:02d}'
            ow = ROOT/f'experiments/20261104_event_presence_location/runs/lofo_v001/folds/{fid:02d}'
            nd, nw = a.dense_run/f'folds/{fid:02d}', a.window_run/f'folds/{fid:02d}'
            old = Runtime(od/'model.pt', od/'normalization.json', ow/'model.pt', ow/'normalization.json')
            new = Runtime(nd/'model.pt', nd/'normalization.json', nw/'model.pt', nw/'normalization.json')
            gap_path = GAP/f'weights/fold_{fid:02d}.json'
            gap = GapRuntime(gap_path)
            for base in (od,ow,nd,nw):
                for name in ('model.pt','normalization.json'):
                    weights[(base/name).relative_to(ROOT).as_posix()] = sha(base/name)
            weights[gap_path.relative_to(ROOT).as_posix()] = sha(gap_path)
            for vid in sorted(fold['held_out_video_ids']):
                original, enriched = (prepare_inputs(dense, window, vid, v) for v in ('original','augmented'))
                replay = old.predict_prepared(*original)
                control = gap.apply(videos[vid], observations[vid], replay['events'])
                if identity(control['events']) != identity(reference[vid]):
                    raise ValueError('Original control does not reproduce prior output: '+vid)
                for arm, model, prepared in (
                    ('old_weights_retained_support', old, enriched),
                    ('paired_enriched', new, enriched),
                    ('paired_original_diagnostic', new, original)):
                    prediction = model.predict_prepared(*prepared)
                    added = gap.apply(videos[vid], observations[vid], prediction['events'])
                    arms[arm].append(dict(video_id=vid, events=added['events']))
                    timing.append(dict(arm=arm, video_id=vid, fold=fid,
                        dense_events=len(prediction['dense_events']), window_events=len(prediction['window_events']),
                        union_events=len(prediction['events']), final_events=len(added['events']),
                        decision_seconds=prediction['decision_seconds'],
                        gap_seconds=added['receipt']['timing_seconds']['gap_total']))
            print(json.dumps(dict(completed_fold=fid, completed_videos=len(arms['paired_enriched']))), flush=True)
        for arm, rows in arms.items():
            rows.sort(key=lambda r:r['video_id'])
            save(a.output/arm/'predictions_by_video.json', rows)
        save(a.output/'prediction_seal.json', dict(sealed_utc=datetime.now(timezone.utc).isoformat(),
            predictions={arm:sha(a.output/arm/'predictions_by_video.json') for arm in arms},
            weights_sha256=weights, original_control_frame_pts_xy_exact=True))
        spec = importlib.util.spec_from_file_location('adaptation_temporal_evaluator', SCORER)
        scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scorer)
        timeline = read(RESERVOIR/'training/timeline.json')
        baseline = read(GAP/'strict.json')
        details = {}
        for arm, rows in arms.items():
            predictions = [dict(**{k:v for k,v in event.items() if k not in ('clip_id','native_frame_id')},
                clip_id=row['video_id'], native_frame_id=event['frame_id']) for row in rows for event in row['events']]
            details[arm] = scorer.evaluate(timeline, predictions, 0.)
            assert details[arm]['eligible_positive_events'] == 680
            save(a.output/arm/'strict.json', details[arm])
        report = dict(baseline=compact(baseline), results={k:compact(v) for k,v in details.items()},
            versus_baseline={k:difference(baseline,v) for k,v in details.items()},
            trained_versus_old_weights=difference(details['old_weights_retained_support'],details['paired_enriched']),
            original_control_frame_pts_xy_exact=True, scoring_runs=3, protected33_used=False, public15_used=False,
            scope='Repeatedly used development121 source-held-out temporal evaluation; no precise XY or full FPS claim',
            cached_decision_seconds={arm:sum(r['decision_seconds']+r['gap_seconds'] for r in timing if r['arm']==arm) for arm in arms},
            protocol_sha256=sha(EXP/'protocol_v001.json'), evaluator_sha256=sha(SCORER),
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),NEW/'channel_adaptation_v001/inputs.py',NEW/'channel_adaptation_v001/runtime.py']})
        save(a.output/'report.json', report)
        save(a.output/'timing.json', timing)
        save(EXP/'analysis/result_summary_v001.json', report)
        status.update(completed=True, exit_code=0)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception:
        status.update(exit_code=1, error=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started)
        save(a.output/'run_status.json', status)
        (a.output/'exit_code.txt').write_text(str(status['exit_code'])+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
