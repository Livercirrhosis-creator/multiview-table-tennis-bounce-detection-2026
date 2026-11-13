"""Replay four predeclared visual evidence arms without changing the event chain."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import traceback

import numpy as np

EXP = Path(__file__).resolve().parents[1]
EXPERIMENTS = EXP.parent
PREVIOUS = EXPERIMENTS / '20261110_adapted_visual_contact'
sys.path.insert(0, str(EXPERIMENTS / '20261109_hierarchical_contact/code'))
from replay_hierarchy import adapt, fixed, score
sys.path.insert(0, str(EXPERIMENTS / '20261109_contact_visual_evidence/code'))
from replay_visual_pair import dense_with_visual

ARMS = ('CHANNEL_CONTROL', 'RAW_VISUAL', 'CALIBRATED_FIXED_MEAN', 'CALIBRATED_SCALE_AWARE')
CALIBRATION_SOURCE = EXP / 'versions/v001/participant/pingpang/participant/visual_calibration_v001/calibration.py'


def run(args):
    import torch
    torch.set_num_threads(1)
    allowed, videos, folds, priors, config, table, digests = fixed.load_fixed_inputs()
    runtime = EXPERIMENTS / ('20261108_branch_consolidation/runs/packet_v001/bundle/version/'
                            'participant/pingpang')
    sys.path.insert(0, str(runtime))
    from participant.channel_adaptation_v001.runtime import Runtime
    from participant.event_decision_v001.selection import select_events
    from participant.event_union_v001.runtime import combine_events
    from participant.masked_gap_v001.runtime import Runtime as GapRuntime
    api = fixed.h.load_module('frozen_visual_evidence_scale', CALIBRATION_SOURCE)
    old_visual = EXPERIMENTS / '20261109_contact_visual_evidence'
    shortlist_path = old_visual / 'versions/v001/participant/pingpang/participant/contact_visual_v001/shortlist.py'
    shortlist = fixed.h.load_module('calibration_original_shortlist', shortlist_path).shortlist_ids
    for p in (Path(__file__), EXP / 'protocol_v001.json', CALIBRATION_SOURCE, shortlist_path):
        fixed.tracked(p, digests)
    old_run = PREVIOUS / 'runs/server_full_runtime_v001/collection_20261110T071819061827Z/extracted/runs/research_v001/training'
    old_status = fixed.read(old_run / 'run_status.json')
    fixed.tracked(old_run / 'probabilities_by_video.json', digests, old_status['predictions_sha256'])
    raw = fixed.read(old_run / 'probabilities_by_video.json')
    scales = {fold['fold_id']: 1. for fold in folds}
    calibrated = raw
    if not args.check_historical:
        result = args.calibration_result
        status = fixed.read(result / 'run_status.json')
        fixed.require(status['completed'] and status['visual_training_runs'] == 30
                      and status['calibration_fits'] == 10 and status['outer_models_reused'] == 10
                      and not any(status[k] for k in ('public15_used', 'protected33_used', 'external130_used')),
                      'Expected a completed nested development-only fit')
        fixed.tracked(result / 'run_status.json', digests)
        fixed.tracked(result / 'raw_outer_probabilities_by_video.json', digests, old_status['predictions_sha256'])
        fixed.tracked(result / 'probabilities_by_video.json', digests, status['predictions_sha256'])
        calibrated = fixed.read(result / 'probabilities_by_video.json')
        for fold in folds:
            p = result / f'outer_{fold["fold_id"]:02d}/calibration.json'
            fixed.tracked(p, digests)
            scales[fold['fold_id']] = float(fixed.read(p)['a'])
        fixed.require(set(calibrated) == allowed, 'Calibrated video membership differs')
    fixed.require(set(raw) == allowed and sum(map(len, raw.values())) == 17559,
                  'Raw visual membership differs')
    previous_ref = {r['video_id']: r['events'] for r in fixed.read(PREVIOUS / 'runs/replay_v001/ADAPTED_VISUAL/predictions_by_video.json')}
    control_ref = {r['video_id']: r['events'] for r in fixed.read(fixed.TABLE_RUN / 'LOFO/CHANNEL/predictions_by_video.json')}
    for p in (PREVIOUS / 'runs/replay_v001/prediction_seal.json',
              PREVIOUS / 'runs/replay_v001/ADAPTED_VISUAL/predictions_by_video.json',
              fixed.TABLE_RUN / 'LOFO/CHANNEL/predictions_by_video.json'):
        fixed.tracked(p, digests)
    dense, window = (adapt.load_dataset(adapt.EXP / 'runs/dataset_v002' / k) for k in ('dense', 'window'))
    weights = fixed.read(adapt.EXP / 'runs/evaluation_v001/prediction_seal.json')['weights_sha256']
    h = fixed.h
    observations = h.grouped_jsonl(h.RESERVOIR / 'runtime/observations.jsonl', allowed)
    predictions, support = {arm: [] for arm in ARMS}, []
    for fold in folds:
        fid, a = fold['fold_id'], scales[fold['fold_id']]
        paths = [base / filename for base in (
            adapt.EXP / f'runs/lofo_parallel_v001/dense/folds/{fid:02d}',
            adapt.EXP / f'runs/lofo_v001/window/folds/{fid:02d}')
            for filename in ('model.pt', 'normalization.json')]
        gap_path = adapt.GAP / f'weights/fold_{fid:02d}.json'
        for p in [*paths, gap_path]:
            fixed.tracked(p, digests, weights[h.relative(p)])
        model, gap = Runtime(*paths), GapRuntime(gap_path)
        separation = model.minimum_separation_seconds
        for vid in sorted(fold['held_out_video_ids']):
            prepared = adapt.prepare_inputs(dense, window, vid, 'augmented')
            pool = prepared[0][0]
            original = model.predict_prepared(*prepared)
            fixed.require(set(raw[vid]) == set(calibrated[vid]) == {p['candidate_id'] for p in pool},
                          'Candidate membership differs: '+vid)
            keys = shortlist(pool, original['probabilities'], original['dense_events'],
                             original['window_events'], separation, include_boundary=False)
            positions = [i for i, p in enumerate(pool) if p['candidate_id'] in keys]
            raw_selected = {k: raw[vid][k] for k in keys}
            probabilities = {'CHANNEL_CONTROL': original['probabilities']}
            probabilities['RAW_VISUAL'], _ = dense_with_visual(pool, original['probabilities'], raw_selected, 1e-8)
            for arm in ARMS[2:]:
                probabilities[arm] = original['probabilities'].copy()
                if positions:
                    q = np.asarray([calibrated[vid][pool[i]['candidate_id']] for i in positions])
                    probabilities[arm][positions] = api.fuse(original['probabilities'][positions], q, a,
                        scale_aware=arm == 'CALIBRATED_SCALE_AWARE')
            stages = {stage: {} for stage in ('dense', 'union', 'gap', 'final')}
            for arm in ARMS:
                stages['dense'][arm] = select_events(pool, probabilities[arm], minimum_separation_seconds=separation)
                stages['union'][arm] = combine_events(stages['dense'][arm], original['window_events'],
                    minimum_separation_seconds=separation)
                stages['gap'][arm] = gap.apply(videos[vid], observations[vid], stages['union'][arm])['events']
                stages['final'][arm], _ = table(stages['gap'][arm], priors[vid], config)
                predictions[arm].append(dict(video_id=vid, events=stages['final'][arm]))
            fixed.require(adapt.identity(stages['final']['CHANNEL_CONTROL']) == adapt.identity(control_ref[vid]),
                          'Frozen CHANNEL output differs: '+vid)
            fixed.require(adapt.identity(stages['final']['RAW_VISUAL']) == adapt.identity(previous_ref[vid]),
                          'Frozen raw96 visual output differs: '+vid)
            if args.check_historical:
                fixed.require(all(adapt.identity(stages['final'][arm]) == adapt.identity(previous_ref[vid])
                                  for arm in ARMS[2:]), 'a=1 historical branch mismatch')
            before_class = probabilities['CHANNEL_CONTROL'][:, 0] > probabilities['CHANNEL_CONTROL'][:, 1:].max(axis=1)
            changes = {}
            for arm in ARMS[1:]:
                now = probabilities[arm][:, 0] > probabilities[arm][:, 1:].max(axis=1)
                changes[arm] = dict(contact_rejected=int((before_class & ~now).sum()),
                    contact_promoted=int((~before_class & now).sum()), **{
                    stage+'_changed': adapt.identity(values[arm]) != adapt.identity(values['CHANNEL_CONTROL'])
                    for stage, values in stages.items()})
            support.append(dict(video_id=vid, fold_id=fid, a=a, candidates=len(pool), applied=len(positions), changes=changes))
        print(f'fold {fid:02d}: CHANNEL and raw96 reference identities exact', flush=True)
    hashes = {}
    for arm, rows in predictions.items():
        (args.output / arm).mkdir()
        fixed.write_predictions(args.output / arm, rows)
        hashes[arm] = fixed.sha(args.output / arm / 'predictions_by_video.json')
    fixed.write(args.output / 'prediction_seal.json', dict(sealed_utc=datetime.now(timezone.utc).isoformat(),
        prediction_sha256=hashes, input_sha256=digests))
    fixed.write(args.output / 'support.json', support)
    summary = dict(control_exact=True, raw96_exact=True, videos=121, source_groups=10, scales=scales,
        applied_candidates=sum(r['applied'] for r in support), comparisons={}, historical_check=args.check_historical,
        adopted_candidate=False, complete_video_fps_measured=False)
    if not args.check_historical:
        timeline = fixed.read(h.RESERVOIR / 'training/timeline.json')
        scorer = h.load_module('nested_visual_temporal_score', h.SCORER)
        for metric, tolerance in (('strict', 0.), ('40ms', .04)):
            scores = {arm: score(scorer, timeline, rows, tolerance) for arm, rows in predictions.items()}
            for arm, result in scores.items():
                fixed.write(args.output / arm / (metric+'.json'), result)
            summary['comparisons'][metric] = {arm: dict(before=adapt.compact(scores[ARMS[0]]),
                after=adapt.compact(scores[arm]), difference=adapt.difference(scores[ARMS[0]], scores[arm]))
                for arm in ARMS[1:]}
    fixed.write(args.output / 'summary.json', summary)
    print(dict(videos=121, reference_exact=True, historical_check=args.check_historical), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-result', type=Path)
    parser.add_argument('--check-historical', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if bool(args.calibration_result) == args.check_historical:
        parser.error('Choose --calibration-result or --check-historical')
    if args.calibration_result:
        args.calibration_result = args.calibration_result.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    receipt = dict(completed=False, command=sys.argv, started_utc=datetime.now(timezone.utc).isoformat(),
        real_training_runs=0, public15_used=False, protected33_used=False, external130_used=False)
    try:
        run(args)
        receipt.update(completed=True, exit_code=0)
    except BaseException:
        receipt.update(exit_code=1, error=traceback.format_exc())
        raise
    finally:
        receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-start)
        fixed.write(args.output / 'run_status.json', receipt)
        (args.output / 'exit_code.txt').write_text(str(receipt['exit_code'])+'\n', encoding='ascii')


if __name__ == '__main__':
    main()
