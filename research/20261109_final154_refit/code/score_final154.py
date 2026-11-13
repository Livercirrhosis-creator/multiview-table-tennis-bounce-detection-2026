"""Compare a sealed final154 fit with its same-graph 121 deployment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
RUNS = EXP / 'runs'
ACTIVATION = RUNS / 'activation_v001.json'
VISUAL = EXP.parent / '20261109_visual_external_gate'
DEADLINE = EXP.parent / '20261109_deadline_visual_release/runs'
PUBLIC_BASE = (DEADLINE / 'server_full_runtime_v001/collection_20261109T140226225439Z'
               / 'extracted/runs/public15_v001')
PUBLIC_BASE_SHA = '350ce59cbf14e1ae1ec377d3c32f403fcf24b7a82a82088e2bbdea41d14d27f7'
PUBLIC_SCORE_SHA = '3a5da02209eae1daab43d3691618ee758b88809753c810dc2ee3842884af3ab7'
MANIFEST = ROOT / 'participant/pingpang/public_data/manifest.json'
GROUND_TRUTH = MANIFEST.with_name('gt_public.jsonl')

sys.path.insert(0, str(VISUAL / 'code'))
sys.path.insert(0, str(EXP.parent / '20261103_public15_baseline/code'))
import evaluate_public15 as public
import score_visual_external as external


BRANCHES = {'CHANNEL_TABLE': external.BASE_ENTRY, 'VISUAL_SHORTLIST': external.VISUAL_ENTRY}
require = external.require
EXTERNAL32 = EXP.parent / '20261109_channel_external_retrain'
EXTERNAL32_SPLIT = EXTERNAL32 / 'data/external_split_v001.json'
EXTERNAL32_SPLIT_SHA = '21c391aa4db5c11584793e48247012e3899649df34f0d2785a4a9aea81cdbcbf'
EXTERNAL32_BASELINE = EXTERNAL32 / 'runs/pair_scores_v001/comparison.json'
EXTERNAL32_BASELINE_SHA = 'de4c99c6bc1e113f49ae0b61b160c31b752d501dac7130d0830e13c09d466982'
EXTERNAL32_DETAILS = external.BASE_SEAL.with_name('all130_safe_details.json')
EXTERNAL32_DETAILS_SHA = '3ac372b6f7db39ef5e1262ec9399b1e3599bfef25b0a2460bd2325b323142c3c'


def final_authorization(io):
    activation = io.read_json(ACTIVATION)
    require(activation.get('schema') == 'final154_activation.v1'
            and activation.get('scope') == 'final154_training' and activation.get('activated') is True
            and activation.get('training_authorized') is True and activation.get('chain') in BRANCHES,
            'Missing approved final154 activation')
    require(activation['readiness_sha256'] == io.sha(EXP / 'analysis/input_readiness_v001.json'),
            'Final154 readiness changed since activation')
    return activation


def check_run(io, folder, record, entry, manifest_sha):
    require(record['entrypoint'] == entry and record['engine_sha256'] == external.ENGINE_SHA
            and record['input_manifest_sha256'] == manifest_sha, 'Run receipt identity differs')
    hashes = {name: io.sha(folder / name) for name in external.RUN_FILES}
    require(hashes == record['files'], 'Run outputs differ from their sealed hashes')
    execution = io.read_json(folder / 'execution.json')
    command = execution['command']
    require(all(execution[key] == record[key] for key in ('entrypoint', 'engine_sha256', 'input_manifest_sha256'))
            and command[command.index('--decoder') + 1] == 'gpu'
            and (folder / 'exit_code.txt').read_text(encoding='utf-8-sig').strip() == '0',
            'Run execution differs or did not complete successfully')
    status = io.read_json(folder / 'run_status.json')
    require(status['solution_info']['backend'] == 'tensorrt_cached' and status['timing']['engine_build_sec'] == 0
            and not any(row.get('cpu_fallback') for row in status['videos']), 'Unmatched backend or GPU fallback')
    return status, hashes


def check_weights(status, expected):
    info = status['solution_info']
    heads = info['channel_adaptation']['head_assets']
    require(all(heads[f'channel_adaptation_v001/{name}/model.pt'] == expected[name] for name in ('dense', 'window'))
            and info['table_guard']['model']['weights_sha256'] == expected['table'],
            'Loaded final154 dense, window or table weights differ from the receipt')
    if 'visual' in expected:
        require(info['channel_visual_shortlist']['model_sha256'] == expected['visual'],
                'Loaded final154 visual weights differ from the receipt')


def score_public(io, folder, status):
    public.require_hash(MANIFEST, public.MANIFEST_SHA, 'Public15 manifest')
    public.require_hash(GROUND_TRUTH, public.GT_SHA, 'Public15 labels')
    scorer = public.import_scorer()
    videos = public.check_manifest(io.read_json(MANIFEST))
    predictions = folder / 'predictions.jsonl'
    public.check_status(status, videos, scorer.load_events(predictions, videos))
    score = scorer.evaluate(MANIFEST, GROUND_TRUTH, predictions, folder / 'run_status.json')
    return dict(scores=score['scores'], events=score['aggregate_event_metrics'],
                infer_fps=status['performance']['infer_fps'], timing=status['timing'])


def baseline_public(io, chain):
    summary_path = DEADLINE / 'scores_v001/summary.json'
    receipt_path = PUBLIC_BASE / 'batch_receipt.json'
    require(io.sha(summary_path) == PUBLIC_SCORE_SHA and io.sha(receipt_path) == PUBLIC_BASE_SHA,
            'Frozen 121 public receipt or score differs')
    receipt = io.read_json(receipt_path)
    matches = [row for row in receipt['arms'] if row['arm'] == chain]
    require(receipt['completed'] is True and receipt['exit_code'] == 0
            and receipt['engine_sha256'] == external.ENGINE_SHA and len(matches) == 1
            and matches[0]['entrypoint'] == BRANCHES[chain] and matches[0]['exit_code'] == 0,
            'Expected one completed same-graph 121 public baseline')
    folder = PUBLIC_BASE / chain
    require(all(io.sha(folder / name) == matches[0]['files'][name] for name in ('predictions.jsonl', 'run_status.json')),
            'Frozen 121 public predictions changed')
    return io.read_json(summary_path)['arms'][chain], folder


def baseline_external(io, package, wrapper, engine, chain, activation, summary_path, visual_run):
    if chain == 'CHANNEL_TABLE':
        hashes, scores = external.verify_base(io)
        return external.evaluate(io, package, wrapper, engine, external.BASE, external.BASE_ENTRY, hashes, scores)
    require(io.sha(summary_path) == activation['external_gate_sha256'], 'Frozen visual external score changed')
    summary = io.read_json(summary_path)
    require(summary['external_gate_pass'] is True and summary['baseline_self_check'] is False
            and summary['labels_receipt_sha256'] == external.LABEL_RECEIPT_SHA,
            'Missing accepted 121 visual external score')
    receipt_sha = summary['candidate_batch_receipt_sha256']
    if visual_run is None:
        receipts = VISUAL.glob('runs/server_full_runtime_v001/collection_*/extracted/runs/external_v001/batch_receipt.json')
        matches = [path.parent for path in receipts if io.sha(path) == receipt_sha]
        require(len(matches) == 1, 'Pass --visual-baseline-root for the sealed 121 visual external run')
        visual_run = matches[0]
    arm = visual_run / 'VISUAL_SHORTLIST'
    _, actual_sha = external.verify_candidate(io, visual_run, arm)
    require(actual_sha == receipt_sha, 'Frozen 121 visual external batch differs')
    return external.evaluate(io, package, wrapper, engine, arm, external.VISUAL_ENTRY,
                             summary['candidate']['files_sha256'], summary['candidate_breakdowns'])


def fixed_external32(io, package, final_safe_details):
    """An auxiliary fixed source subset; reuse full130 event matching unchanged."""
    for path, digest in ((EXTERNAL32_SPLIT, EXTERNAL32_SPLIT_SHA),
                         (EXTERNAL32_BASELINE, EXTERNAL32_BASELINE_SHA),
                         (EXTERNAL32_DETAILS, EXTERNAL32_DETAILS_SHA)):
        require(io.sha(path) == digest, f'Frozen external32 input changed: {path.name}')
    split, prior = io.read_json(EXTERNAL32_SPLIT), io.read_json(EXTERNAL32_BASELINE)
    selected = split['heldout_clip_ids']
    require(split['schema'] == 'channel_external_source_holdout_v001'
            and len(selected) == len(set(selected)) == 32 and len(split['heldout_groups']) == 12
            and not set(selected) & set(split['train_clip_ids'])
            and prior['split_sha256'] == EXTERNAL32_SPLIT_SHA,
            'Frozen external32 membership differs')
    baseline_safe = io.read_json(EXTERNAL32_DETAILS)
    baseline, final = {}, {}
    for metric in external.METRICS:
        baseline[metric] = io.add_breakdowns(baseline_safe['variants'][metric], selected, package)
        expected = prior['external_holdout']['arms']['A'][metric]
        actual_macro, old_macro = baseline[metric]['source_group_macro'], expected['source_group_macro']
        # The older A/B helper used statistics.mean; add_breakdowns uses sum / n.
        macro_metrics = {'precision', 'recall', 'F1'}
        require({key: value for key, value in baseline[metric].items() if key != 'source_group_macro'}
                == {key: value for key, value in expected.items() if key != 'source_group_macro'}
                and actual_macro.keys() == old_macro.keys()
                and all(math.isclose(actual_macro[key], old_macro[key], rel_tol=0, abs_tol=1e-12)
                        if key in macro_metrics else actual_macro[key] == old_macro[key]
                        for key in actual_macro),
                f'External32 original CHANNEL baseline did not reproduce: {metric}')
        final[metric] = io.add_breakdowns(final_safe_details['variants'][metric], selected, package)
        require(len(final[metric]['per_video']) == 32, 'Final external32 coverage differs')
    require(baseline['representative_time']['micro']['scored_events']
            == split['counts']['heldout_safe_events'] == 204, 'External32 SAFE denominator changed')
    compact = lambda metrics: {name: {key: value[key] for key in
                              ('micro', 'source_group_macro', 'GT_provenance_recall_only')}
                               for name, value in metrics.items()}
    return dict(scope='fixed external32 source holdout from the earlier external-data training experiment',
                role='auxiliary comparison only; does not change final154 adoption gate',
                exposure='Previously evaluated in the external A/B experiment; not a fresh blind test',
                videos=32, source_groups=12, baseline_chain='CHANNEL_TABLE',
                split_sha256=EXTERNAL32_SPLIT_SHA, baseline_summary_sha256=EXTERNAL32_BASELINE_SHA,
                baseline_all130_details_sha256=EXTERNAL32_DETAILS_SHA,
                labels_receipt_sha256=external.LABEL_RECEIPT_SHA,
                baseline=compact(baseline), final=compact(final), comparison=external.compare(final, baseline))


def score_final(args):
    io = external.load_scorer()
    activation = final_authorization(io)
    chain = activation['chain']
    entry = BRANCHES[chain]
    receipt_path = args.receipt.resolve()
    receipt = io.read_json(receipt_path)
    require(receipt['completed'] is True and receipt['exit_code'] == 0 and receipt['chain'] == chain
            and receipt['activation_sha256'] == io.sha(ACTIVATION)
            and set(receipt['scopes']) == {'external130', 'public15'}, 'Incomplete or wrong final154 batch')
    weight_names = {'dense', 'window', 'gap', 'table', 'table_metadata'}
    if chain == 'VISUAL_SHORTLIST':
        weight_names.add('visual')
    weights = receipt['weight_sha256']
    require(set(weights) == weight_names, 'Final154 weights receipt does not cover the selected graph')
    require(receipt.get('runtime_weights_before') == weights
            and receipt.get('runtime_weights_after') == weights,
            'Final154 runtime assets differ before or after inference')
    require(io.sha(external.EXTERNAL / 'analysis/labels_v006_receipt.json') == external.LABEL_RECEIPT_SHA,
            'External130 label receipt changed')
    package, wrapper, engine, _ = io.verified_package()
    base_ext, base_info = baseline_external(io, package, wrapper, engine, chain, activation,
                                            args.visual_baseline_score, args.visual_baseline_root)
    base_pub, _ = baseline_public(io, chain)
    files = {}
    for scope, folder, manifest_sha in (('external130', args.external_run, external.MEDIA_SHA),
                                        ('public15', args.public_run, public.MANIFEST_SHA)):
        folder = folder.resolve()
        require(folder == receipt_path.parent / scope, f'{scope} arm directory differs from batch root')
        status, files[scope] = check_run(io, folder, receipt['scopes'][scope], entry, manifest_sha)
        check_weights(status, weights)
        if scope == 'external130':
            predictions = io.read_predictions(folder / 'predictions.jsonl')
            completion = io.check_complete_run(status, predictions, package['manifest'], 0)
            engine.validate_data(package['manifest'], package['annotations'], predictions)
            final_details, final_ext = io.evaluate_frozen(package, wrapper, predictions)
            external32 = fixed_external32(io, package, final_details['safe'])
        else:
            final_pub = score_public(io, folder, status)
        require(files[scope] == {name: io.sha(folder / name) for name in external.RUN_FILES},
                f'{scope} inputs changed during scoring')
    differences = external.compare(final_ext['safe']['all130_diagnostic'], base_ext['safe']['all130_diagnostic'])
    public_delta = final_pub['scores']['total_out_of_100'] - base_pub['scores']['total_out_of_100']
    return dict(chain=chain, final_batch_sha256=io.sha(receipt_path), final_weight_sha256=weights,
                external130=dict(safe_all130=differences, baseline_completion=base_info['completion'],
                                 final_completion=completion, baseline=external.summarize(base_ext),
                                 final=external.summarize(final_ext)),
                external32=external32,
                public15=dict(baseline=base_pub, final=final_pub, total_delta=public_delta),
                final154_gate_pass=public_delta > 0 and all(row['delta']['F1'] > 0 for row in differences.values()),
                training_data_scored=False, protected33_used_for_scoring=False)


def check121():
    io = external.load_scorer()
    result = {}
    for chain in BRANCHES:
        scores, folder = baseline_public(io, chain)
        rebuilt = score_public(io, folder, io.read_json(folder / 'run_status.json'))
        require(all(rebuilt[key] == scores[key] for key in rebuilt),
                f'Frozen public121 baseline for {chain} did not reproduce')
        result[chain] = scores['scores']['total_out_of_100']
    return result


def check_external32():
    io = external.load_scorer()
    package, _, _, _ = io.verified_package()
    result = fixed_external32(io, package, io.read_json(EXTERNAL32_DETAILS))
    require(result['baseline'] == result['final']
            and all(all(value == 0 for value in metric['delta'].values())
                    and metric['macro_F1_delta'] == 0 for metric in result['comparison'].values()),
            'External32 baseline-to-baseline comparison differs')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-run', type=Path)
    parser.add_argument('--public-run', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--visual-baseline-root', type=Path)
    parser.add_argument('--visual-baseline-score', type=Path, default=VISUAL / 'runs/scores_v001/summary.json')
    parser.add_argument('--check121', action='store_true')
    parser.add_argument('--check-external32', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.check121:
        print(json.dumps(check121()), flush=True)
        return
    if args.check_external32:
        print(json.dumps(check_external32()), flush=True)
        return
    if not all((args.external_run, args.public_run, args.receipt, args.output)):
        parser.error('Final154 scoring requires --external-run, --public-run, --receipt and --output')
    output = args.output.resolve()
    require(output.is_relative_to(RUNS) and not output.exists(), 'Use a new final154 scoring output directory')
    report = score_final(args)
    output.mkdir(parents=True, exist_ok=False)
    external.load_scorer().write_new(output / 'summary.json', report)
    print(json.dumps(dict(chain=report['chain'], gate=report['final154_gate_pass'],
                          public_delta=report['public15']['total_delta'],
                          external_delta=report['external130']['safe_all130'])), flush=True)


if __name__ == '__main__':
    main()
