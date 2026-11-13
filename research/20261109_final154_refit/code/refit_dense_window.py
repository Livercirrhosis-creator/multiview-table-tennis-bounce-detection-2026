"""Prepare and fit the original paired-view event heads on authorized final154 data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
RUNS = EXP / 'runs'
ADAPT = EXP.parent / '20261108_channel_event_adaptation'
READY = EXP / 'analysis/input_readiness_v001.json'
ACTIVATION = RUNS / 'activation_v001.json'
BASE_RUNTIME = (EXP.parent / '20261108_branch_consolidation/runs/packet_v001/bundle/version'
                / 'participant/pingpang')
HELD_PROTOCOL = EXP.parent / '20261108_final_pair_confirmation33/protocol_v001.json'
HELD_TIMELINE_SHA = 'daaacad9dde9ef56175794e901e1df06515df4bcfc83f8bdf0a3d01d51c30c6e'
KINDS = ('dense', 'window')
FILES = ('index.jsonl', 'windows.json', 'supervision.json', 'features.npz')

sys.path.insert(0, str(ADAPT / 'code'))
import build_dataset as builder
import fit_deployment as deployment


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(trainer, record):
    path = (ROOT / record['path']).resolve()
    require(path.is_relative_to(ROOT) and trainer.sha(path) == record['sha256'],
            f"Sealed input differs: {record['path']}")
    if 'bytes' in record:
        require(path.stat().st_size == record['bytes'], f"Sealed size differs: {record['path']}")
    return path


def authorization(trainer):
    require(ACTIVATION.is_file(), 'Final154 activation is absent; only --check121 is available')
    gate = trainer.read(ACTIVATION)
    external = gate.get('external_gate_sha256', '')
    require(gate.get('schema') == 'final154_activation.v1'
            and gate.get('scope') == 'final154_training' and gate.get('activated') is True
            and gate.get('training_authorized') is True and gate.get('readiness_sha256') == trainer.sha(READY)
            and gate.get('chain') in ('CHANNEL_TABLE', 'VISUAL_SHORTLIST')
            and isinstance(external, str) and len(external) == 64
            and all(c in '0123456789abcdef' for c in external),
            'Final154 fitting requires the parent-owned activation and readiness seal')
    return gate, trainer.read(READY)


def base_data(trainer):
    data = {}
    for kind in KINDS:
        _, manifest, index, windows, labels, _, sequence, summary = trainer.load_dataset(deployment.DATASET, kind)
        data[kind] = (manifest, labels, dict(index=index, windows=windows, sequence=[sequence], summary=[summary]))
    return data


def label_builders():
    dense = builder.module('refit154_dense_labels', builder.BAG_HELPER)
    window = builder.module('refit154_window_labels', builder.WINDOW_HELPER)
    validate = builder.module('refit154_validate_timeline', dense.SCORER).validate_timeline
    return dict(dense=lambda data, clips: builder.dense_labels(data, clips, dense),
                window=lambda data, clips: builder.window_labels(data, clips, window)), validate


def load_dev_timeline(trainer, validate, data):
    path = builder.RESERVOIR / 'training/timeline.json'
    relative = trainer.relative(path)
    digest = trainer.sha(path)
    require(all(manifest['input_sha256'][relative] == digest for manifest, _, _ in data.values()),
            'Frozen development timeline changed')
    return validate(trainer.read(path)), digest


def full_fold(trainer, kind, index, supervision, members):
    fold = deployment.full_fold(trainer, kind, index, supervision)
    return dict(fold, train_video_ids=sorted(members))


def fit_plan(trainer, kind, target, supervision, members):
    fold = full_fold(trainer, kind, target['index'], supervision, members)
    groups, background, legal = trainer.membership(kind, target['index'], target['windows'], supervision, fold)
    norm, norm_membership, _ = deployment.old_assets(trainer, kind)
    return dict(physical_events=len(groups), event_variants=sum(len(g['variants']) for g in groups),
                background_by_view={k: len(v) for k, v in background.items()},
                legal_rows=len(legal), legal_rows_sha256=trainer.digest_rows(legal),
                normalization=dict(path=trainer.relative(norm), sha256=trainer.sha(norm)),
                original_normalization_membership=dict(path=trainer.relative(norm_membership),
                                                       sha256=trainer.sha(norm_membership)),
                optimizer_steps=trainer.TRAINING['epochs'] * math.ceil(len(groups) / trainer.TRAINING['event_batch_size']))


def append_capture(targets, capture, native, family):
    from participant.channel_adaptation_v001.inputs import build_dual_inputs
    from participant.channel_adaptation_v001.selector import replace_observations

    video = capture['video']
    require(all(video[k] == native[k] for k in ('clip_id', 'width', 'height', 'frame_count', 'pts_seconds', 'media_sha256')),
            f"Native clock or media identity differs: {native['clip_id']}")
    original = capture['observations']
    augmented, _ = replace_observations(video, original, capture['channels'])
    dual = build_dual_inputs(video, original, augmented, capture['candidates'])
    for view in builder.VIEWS:
        pool, sequence, summary, windows, _ = dual[view]
        builder.append_view(targets['dense'], view, pool, sequence, summary, windows, video, family)
        rows = dual['window_rows'][view]
        builder.append_view(targets['window'], view, [pool[i] for i in rows], sequence[rows],
                            summary[rows], dual['window_windows'][view], video, family)


def check121(trainer):
    data = base_data(trainer)
    labels, validate = label_builders()
    clips, _ = load_dev_timeline(trainer, validate, data)
    expected = trainer.read(deployment.PLAN)['fits']
    report = {}
    for kind, (_, sealed, target) in data.items():
        rebuilt = labels[kind](target, clips)
        require(rebuilt == sealed, f'Unchanged {kind} label builder did not reproduce the frozen development labels')
        plan = fit_plan(trainer, kind, target, rebuilt, clips)
        require(plan['physical_events'] == expected[kind]['physical_training_events']
                and plan['optimizer_steps'] == expected[kind]['planned_optimizer_steps']
                and plan['legal_rows_sha256'] == expected[kind]['training_legal_membership_sha256'],
                f'{kind} full-deployment membership does not reproduce')
        report[kind] = plan
    print(json.dumps(report, ensure_ascii=False), flush=True)


def prepare(trainer, output):
    gate, ready = authorization(trainer)
    require(output.is_relative_to(RUNS) and not output.exists(), 'Use a new final154 runs directory')
    data = base_data(trainer)
    labels, validate = label_builders()
    dev, dev_timeline_sha = load_dev_timeline(trainer, validate, data)
    development, protected = (set(trainer.read(verify(trainer, row))['video_ids']) for row in ready['cohort']['membership'])
    require(len(development) == 121 and set(dev) == development and len(protected) == 33
            and not development.intersection(protected), 'The fixed 121 plus 33 membership differs')
    held_record = trainer.read(HELD_PROTOCOL)['fixed_files']['timeline']
    require(held_record['sha256'] == HELD_TIMELINE_SHA, 'Protected timeline identity differs')
    held = validate(trainer.read(verify(trainer, held_record)))
    native_path = verify(trainer, ready['cohort']['native'][1])
    native = {clip['clip_id']: clip for clip in trainer.read(native_path)['clips']}
    seal = trainer.read(verify(trainer, ready['held_capture']['seal']))['captures_sha256']
    capture_dir = (ROOT / ready['held_capture']['directory']).resolve()
    require(capture_dir.is_relative_to(ROOT) and set(native) == set(held) == set(seal) == protected,
            'Protected timeline, capture, or native index membership differs')
    sys.path.insert(0, str(BASE_RUNTIME))
    targets = {kind: data[kind][2] for kind in KINDS}
    for vid in sorted(protected):
        clip = native[vid]
        require(all(clip[k] == held[vid][k] for k in ('frame_count', 'pts_seconds', 'media_sha256')),
                f'Protected training timeline does not match native video: {vid}')
        path = capture_dir / f'{vid}.json'
        require(trainer.sha(path) == seal[vid], f'Sealed capture changed: {vid}')
        append_capture(targets, trainer.read(path), clip, held[vid]['camera_family'])
    clips = dev | held
    output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for kind, target in targets.items():
        index, windows = target['index'], target['windows']
        require([r['row_index'] for r in index] == [w['window_index'] for w in windows] == list(range(len(index)))
                and len({r['candidate_id'] for r in index}) == len(index), f'Invalid final154 {kind} row identity')
        supervision = labels[kind](target, clips)
        plan = fit_plan(trainer, kind, target, supervision, clips)
        folder = output / kind
        folder.mkdir()
        np.savez_compressed(folder / 'features.npz', sequence=np.concatenate(target['sequence']),
                            summary=np.concatenate(target['summary']))
        with (folder / 'index.jsonl').open('x', encoding='utf-8') as stream:
            for row in index:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        trainer.save(folder / 'windows.json', windows)
        trainer.save(folder / 'supervision.json', supervision)
        by_role = dict(development=sum(r['video_id'] in development for r in index),
                       former_confirmation=sum(r['video_id'] in protected for r in index))
        record = dict(kind=kind, candidates_by_role=by_role, training_videos=154, paired_views=True,
                      original_cache_files_sha256=data[kind][0]['output_sha256'],
                      output_sha256={name: trainer.sha(folder / name) for name in FILES}, fit_plan=plan)
        trainer.save(folder / 'manifest.json', record)
        summary[kind] = dict(record, manifest_sha256=trainer.sha(folder / 'manifest.json'))
    source = [Path(__file__), Path(trainer.__file__), Path(builder.__file__), Path(deployment.__file__),
              builder.BAG_HELPER, builder.WINDOW_HELPER, trainer.OLD['dense'] / 'code/train_dense_context.py']
    for name in ('channel_adaptation_v001', 'dense_context_v001', 'event_decision_v001', 'event_window_v001'):
        source.extend(sorted((BASE_RUNTIME / 'participant' / name).glob('*.py')))
        source.extend(sorted((trainer.RUNTIME / 'participant' / name).glob('*.py')))
    manifest = dict(schema='paired_channel_final154_inputs.v1', training_videos=sorted(clips),
                    architecture=gate['chain'], activation_sha256=trainer.sha(ACTIVATION),
                    timeline_sha256=dict(development=dev_timeline_sha, former_confirmation=HELD_TIMELINE_SHA),
                    former_confirmation_capture_sha256=seal, configuration=trainer.TRAINING,
                    source_sha256={trainer.relative(path): trainer.sha(path) for path in sorted(set(source))},
                    prepared_utc=datetime.now(timezone.utc).isoformat(), fits=summary)
    trainer.save(output / 'manifest.json', manifest)
    print(json.dumps({kind: row['candidates_by_role'] for kind, row in summary.items()}), flush=True)


def fit(trainer, kind, inputs, output):
    import torch

    authorization(trainer)
    require(inputs.is_relative_to(RUNS) and output.is_relative_to(RUNS) and not output.exists(),
            'Use final154 prepared data and a new final154 output')
    manifest = trainer.read(inputs / 'manifest.json')
    members = manifest['training_videos']
    require(manifest['schema'] == 'paired_channel_final154_inputs.v1'
            and len(set(members)) == len(members) == 154 and manifest['configuration'] == trainer.TRAINING
            and manifest['activation_sha256'] == trainer.sha(ACTIVATION), 'Final154 plan or scope changed')
    for name, digest in manifest['source_sha256'].items():
        verify(trainer, dict(path=name, sha256=digest))
    data = inputs / kind
    record = manifest['fits'][kind]
    require(trainer.sha(data / 'manifest.json') == record['manifest_sha256']
            and all(trainer.sha(data / name) == record['output_sha256'][name] for name in FILES),
            f'Sealed {kind} training inputs changed')
    index = trainer.lines(data / 'index.jsonl')
    windows, supervision = (trainer.read(data / name) for name in ('windows.json', 'supervision.json'))
    arrays = np.load(data / 'features.npz', allow_pickle=False)
    sequence, summary = arrays['sequence'], arrays['summary']
    require(sequence.shape == (len(index), 15, 23) and summary.shape == (len(index), 44)
            and {row['video_id'] for row in index} <= set(members), 'Final154 feature layout or scope differs')
    plan = fit_plan(trainer, kind, dict(index=index, windows=windows), supervision, members)
    require(plan == record['fit_plan'], f'Sealed {kind} training membership or normalization changed')
    norm = verify(trainer, plan['normalization'])
    norm_members = verify(trainer, plan['original_normalization_membership'])
    fold = full_fold(trainer, kind, index, supervision, members)
    trainer.old_normalization = lambda requested, unused: (norm, norm_members)

    def no_evaluation(model, requested, x, z, windows, unused, directory):
        require(requested == kind and not unused['held_out_rows'], 'Final154 fit must not score any training data')
        trainer.save(directory / 'held_out.npz', {'evaluation_performed': False, 'training_videos': 154})
        return {}

    trainer.predict_held = no_evaluation
    torch.set_num_threads(trainer.TRAINING['threads'])
    torch.set_num_interop_threads(trainer.TRAINING['interop_threads'])
    started = time.perf_counter()
    status_path = output.with_name(output.name + '_status.json')
    require(not status_path.exists(), 'Fit status path already exists')
    status = dict(kind=kind, started_utc=datetime.now(timezone.utc).isoformat(), completed=False,
                  exit_code=None, activation_sha256=trainer.sha(ACTIVATION), plan_sha256=trainer.sha(inputs / 'manifest.json'))
    trainer.save(status_path, status)
    try:
        _, result = trainer.fit_fold(kind, sequence, summary, index, windows, supervision, fold, output,
                                     trainer.runtime_modules())
        (output / 'held_out.npz').rename(output / 'no_evaluation.json')
        checkpoint = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
        checkpoint.pop('held_out_family')
        checkpoint.update(training_videos=154, full_final_fit=True, checkpoint_selection='fixed epoch80; no evaluation')
        torch.save(checkpoint, output / 'model.pt')
        result['no_evaluation_receipt_sha256'] = result.pop('held_out_sha256')
        result.update(training_videos=154, model_sha256=trainer.sha(output / 'model.pt'),
                      normalization_sha256=trainer.sha(output / 'normalization.json'), plan_sha256=status['plan_sha256'],
                      full_final_fit=True, training_fits=1, model_evaluation_runs=0, scoring_runs=0)
        trainer.save(output / 'fit_result.json', result)
        status.update(completed=True, exit_code=0)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException:
        status.update(exit_code=1, error=traceback.format_exc())
        raise
    finally:
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
        trainer.save(status_path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check121', action='store_true')
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--fit', choices=KINDS)
    parser.add_argument('--inputs', type=Path, default=RUNS / 'inputs_v001')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    trainer = deployment.trainer()
    if args.check121:
        check121(trainer)
    elif args.prepare:
        prepare(trainer, (args.output or args.inputs).resolve())
    else:
        fit(trainer, args.fit, args.inputs.resolve(), (args.output or RUNS / f'{args.fit}_v001').resolve())


if __name__ == '__main__':
    main()
