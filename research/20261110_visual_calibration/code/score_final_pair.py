"""Use the unchanged external SAFE and public scorers for the frozen visual pair."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SOURCE = ROOT / 'experiments/20261110_probability_ensemble/code/score_pair.py'
spec = importlib.util.spec_from_file_location('existing154_pair_metrics', SOURCE)
pair = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pair
spec.loader.exec_module(pair)
require = pair.require
ENTRIES = dict(BASE154='participant.table_guard_v001.channel_solution:Solution',
               CALIBRATED154='participant.visual_calibration_v001.solution:Solution')
SCOPES = ('external130', 'public15')
RECIPE = 'VISUAL_CALIBRATED_SCALE_AWARE_FINAL154_V001'


def score(run, bundle):
    io = pair.external.load_scorer()
    receipt = io.read_json(run / 'batch_receipt.json')
    require(receipt.get('completed') is True and receipt.get('exit_code') == 0
            and receipt['engine_sha256'] == pair.external.ENGINE_SHA
            and set(receipt['arms']) == set(ENTRIES)
            and (run / 'batch.exit_code.txt').read_text().strip() == '0',
            'The fixed visual pair is incomplete')
    seal_path = run / 'deployment_seal.json'
    seal = io.read_json(seal_path)
    require(io.sha(seal_path) == receipt['deployment_seal_sha256']
            and seal['schema'] == 'visual_calibration_deployment_seal.v1'
            and seal['recipe_id'] == RECIPE
            and seal['training_videos'] == 154 and seal['calibration_videos'] == 121
            and seal['final_protocol_sha256'] == io.sha(EXP / 'final_protocol_v001.json'),
            'Deployment identity or data scope differs')
    fit_path, scalar_path = run / 'training/run_status.json', run / 'training/calibration.json'
    require(io.sha(run / 'training/visual/fit_receipt.json') == seal['fit_receipt_sha256']
            and io.sha(scalar_path) == seal['calibration_sha256']
            and io.sha(run / 'training/visual/model.pt') == seal['model_sha256'],
            'Returned training artifacts differ from the deployment seal')
    fit, scalar = io.read_json(fit_path), io.read_json(scalar_path)
    require(fit['completed'] is True and fit['exit_code'] == 0
            and fit['visual_training_runs'] == 1 and fit['calibration_fits'] == 1
            and fit['training_videos'] == 154 and fit['calibration_videos'] == 121
            and not fit['external130_used'] and not fit['public15_used']
            and scalar['a'] == seal['a'] and scalar['solver']['success'] is True,
            'The fixed deployment fit or calibration did not complete')
    manifest = io.read_json(bundle / 'PAYLOAD_MANIFEST.json')
    for row in manifest['files']:
        path = (bundle / row['path']).resolve()
        require(path.is_relative_to(bundle) and path.stat().st_size == row['bytes']
                and io.sha(path) == row['sha256'], 'Sealed payload changed: ' + row['path'])
    original, metadata, gap_basis = pair.original_assets(io)
    package, wrapper, engine, _ = io.verified_package()
    require(io.sha(pair.external.EXTERNAL / 'analysis/labels_v006_receipt.json')
            == pair.external.LABEL_RECEIPT_SHA, 'External label coverage receipt changed')
    scored, details, statuses, evidence = {}, {}, {}, {}
    for arm, entry in ENTRIES.items():
        row = receipt['arms'][arm]
        require(row['entrypoint'] == entry and row['weights_before'] == original
                and row['weights_after'] == original and set(row['scopes']) == set(SCOPES),
                'The original154 chain changed: ' + arm)
        scored[arm], statuses[arm], evidence[arm] = {}, {}, {}
        for scope in SCOPES:
            folder = run / arm / scope
            status, hashes = pair.check_run(io, folder, row['scopes'][scope], entry, scope)
            loaded = pair.check_loaded_weights(status, original, metadata, gap_basis, ensemble=False)
            visual = status['solution_info'].get('visual_calibration')
            if arm == 'CALIBRATED154':
                require(visual is not None and all(visual[k] == seal[k] for k in
                    ('model_sha256', 'calibration_sha256', 'a', 'training_videos', 'calibration_videos')),
                    'Actually loaded visual evidence differs from the fixed deployment')
            else:
                require(visual is None, 'Baseline enabled visual calibration')
            statuses[arm][scope] = status
            evidence[arm][scope] = dict(files_sha256=hashes, base_weights=loaded, visual=visual)
            if scope == 'external130':
                details[arm], breakdowns, completion = pair.score_external(
                    io, package, wrapper, engine, folder, status)
                scored[arm][scope] = dict(breakdowns=breakdowns, completion=completion)
            else:
                scored[arm][scope] = pair.score_public(io, folder, status)
            require(hashes == {name: io.sha(folder / name) for name in pair.external.RUN_FILES},
                    'Run outputs changed during scoring')
    base = scored['BASE154']['external130']['breakdowns']['safe']['all130_diagnostic']
    new = scored['CALIBRATED154']['external130']['breakdowns']['safe']['all130_diagnostic']
    comparison = pair.external.compare(new, base)
    pb, pn = (scored[a]['public15'] for a in ENTRIES)
    public_delta = pn['scores']['total_out_of_100'] - pb['scores']['total_out_of_100']
    fps = {a: {s: statuses[a][s]['performance']['infer_fps'] for s in SCOPES} for a in ENTRIES}
    conditions = dict(
        external_safe_spatiotemporal_improves=comparison['representative_spatiotemporal']['delta']['F1'] > 0,
        public_local_total_improves=public_delta > 0,
        candidate_external_fps_at_least55=fps['CALIBRATED154']['external130'] >= 55,
        candidate_public_fps_at_least55=fps['CALIBRATED154']['public15'] >= 55)
    aux = pair.fixed32_pair(io, package, details['BASE154'], details['CALIBRATED154'])
    aux['candidate_name'] = 'same-batch CALIBRATED154'
    old_angles = {r['angle']: r for r in pb['per_angle']}
    angles = [dict(angle=r['angle'], delta={k: r[k] - old_angles[r['angle']][k]
              for k in ('tp', 'fp', 'fn', 'f1', 'accuracy_points')}) for r in pn['per_angle']]
    return dict(schema='visual_calibration_same_batch154_pair.v1', baseline='BASE154',
        candidate='CALIBRATED154', recipe_id=RECIPE, deployment_seal=seal,
        batch_receipt_sha256=io.sha(run / 'batch_receipt.json'),
        inference_evidence=evidence,
        external130=dict(safe_all130=comparison, not_official_score=True,
            per_source_group=pair.source_comparison(new, base),
            baseline_breakdowns=pair.external.summarize(scored['BASE154']['external130']['breakdowns']),
            candidate_breakdowns=pair.external.summarize(scored['CALIBRATED154']['external130']['breakdowns'])),
        external32=aux, public15=dict(baseline=pb, candidate=pn, total_delta=public_delta,
                                     per_angle_delta=angles),
        speed=dict(official_inference_fps=fps, target=55, independent_repeats=1,
                   scope='Complete official inference timer; one run does not prove a stable speed ranking'),
        adoption_conditions=conditions, adoption_gate_pass=all(conditions.values()),
        boundaries=dict(former33='training-in', independent_blind_test=False,
            external130='Mixed reference annotations with prior selection exposure',
            public15='Previously exposed public regression and local official-format score'),
        scoring_sources_sha256={p.relative_to(ROOT).as_posix(): io.sha(p) for p in
            (Path(__file__), SOURCE, pair.FINAL / 'code/score_final154.py',
             pair.external.ORIGINAL_SCORER, pair.public.SCORER_PATH)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(EXP) and not output.exists(), 'Use a fresh output directory')
    result = score(args.run_root.resolve(), args.bundle_root.resolve())
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                   allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(conditions=result['adoption_conditions'],
                         public_delta=result['public15']['total_delta'],
                         external_delta=result['external130']['safe_all130'],
                         speed=result['speed']), ensure_ascii=False))


if __name__ == '__main__':
    main()
