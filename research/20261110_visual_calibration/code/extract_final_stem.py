"""Reuse the exact121 stem prefix and append only the former33 crop features."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(path, info):
    require(Path(path).is_file() and Path(path).stat().st_size == info['bytes']
            and sha(path) == info['sha256'], 'Missing/changed frozen input: ' + str(path))


def contained(root, name):
    root = Path(root).resolve()
    target = (root / name).resolve()
    require(not Path(name).is_absolute() and target.is_relative_to(root), 'Unsafe input path')
    return target


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def execute(args):
    import numpy as np
    import torch
    import av
    started = time.perf_counter()
    plan = read(args.plan)
    require(plan['schema'] == 'final154_visual_plan.v1'
            and plan['counts']['rows'] == 20641 and plan['counts']['base121_rows'] == 17559
            and plan['counts']['additional33_rows'] == 3082, 'Unexpected final154 plan')
    for info in plan['inputs'].values():
        exact(contained(args.plan.parent, info['name']), info)
    for key in ('old_extractor', 'old_prepare', 'frame_helper'):
        exact(contained(args.assets_root, plan['sources'][key]['path']), plan['sources'][key])
    exact(contained(args.assets_root, plan['model']['source']['path']), plan['model']['source'])
    require(sha(args.base_cache / 'receipt.json') == plan['base121_cache']['receipt_sha256'],
            'Base121 cache receipt differs')
    for name, info in plan['base121_cache']['files'].items():
        exact(args.base_cache / name, info)
    require(sha(args.weights) == plan['model']['pretrained_sha256'], 'Pretrained weights differ')
    old_path = contained(args.assets_root, plan['sources']['old_extractor']['path'])
    sys.path.insert(0, str(old_path.parent))
    old = module('fixed_visual_crop_and_stem', old_path)
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    require(args.device.startswith('cuda') and torch.cuda.is_available(), 'Final extraction requires server CUDA')
    rows = [json.loads(line) for line in (args.plan.parent/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    previous_rows = [json.loads(line) for line in (args.base_cache/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    times = np.load(args.plan.parent/'time_features.npy', allow_pickle=False)
    previous_times = np.load(args.base_cache/'time_features.npy', allow_pickle=False)
    require(len(rows) == 20641 and rows[:17559] == previous_rows
            and times.shape == (20641, 3, 2) and times.dtype == np.float32
            and np.array_equal(times[:17559], previous_times), 'Prefix row/time identity differs')
    requests = read(args.plan.parent/'additional33_requests.json')
    require(requests['schema'] == 'final154_additional33_requests.v1' and len(requests['clips']) == 33,
            'Expected only additional33 media requests')
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(completed=False, exit_code=1, stage='prepare', started_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(args.plan), rows=20641, reused_rows=17559, newly_encoded_rows=3082,
        decoded_videos=0, model_forward_calls=0, training_runs=0, device=args.device)
    save(args.output/'run_status.json', state)
    try:
        shape = (20641, 3, 2, 96, 3, 3)
        previous = np.load(args.base_cache/'stem_features.npy', mmap_mode='r', allow_pickle=False)
        require(previous.shape == (17559, *shape[1:]) and previous.dtype == np.float32, 'Base121 stem shape differs')
        features = np.lib.format.open_memmap(args.output/'stem_features.npy', mode='w+', dtype=np.float32, shape=shape)
        features[:] = 0
        for start in range(0, 17559, 256):
            features[start:min(start+256, 17559)] = previous[start:min(start+256, 17559)]
        require(np.array_equal(features[:17559], previous), 'Copy changed existing stem prefix')
        del previous
        encoded = np.zeros(shape[:3], dtype=bool)
        encoded[:17559] = np.broadcast_to(times[:17559, :, 1:2].astype(bool), (17559, 3, 2))
        stem = old.FrozenStem(args.weights, args.device)
        pending, destinations = [], []
        calls = frames_total = rgb_total = 0

        def flush():
            nonlocal calls
            if not pending:
                return
            values = stem(torch.from_numpy(np.stack(pending))).cpu().numpy()
            for destination, value in zip(destinations, values):
                require(not encoded[destination], 'A former33 crop was encoded twice')
                features[destination] = value
                encoded[destination] = True
            calls += 1
            pending.clear()
            destinations.clear()

        per_video = []
        for clip in requests['clips']:
            path = contained(args.video_root, clip['path'])
            require(sha(path) == clip['media_sha256'], 'Former33 media SHA differs')
            expected = clip['pts_seconds']
            wanted = clip['requests']
            count = converted = 0
            tick = time.perf_counter()
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = 'FRAME'
                stream.codec_context.thread_count = 4
                for frame_index, frame in enumerate(container.decode(stream)):
                    require(frame_index < len(expected) and frame.pts is not None and frame.time_base is not None
                            and abs(float(frame.pts*frame.time_base)-expected[frame_index]) <= 1e-8
                            and (frame.width, frame.height) == (clip['width'], clip['height']),
                            'Native frame PTS/dimensions differ')
                    count += 1
                    if str(frame_index) not in wanted:
                        continue
                    rgb = frame.to_ndarray(format='rgb24')
                    converted += 1
                    for request in wanted[str(frame_index)]:
                        index, slot = request['array_index'], request['slot']
                        require(17559 <= index < 20641 and slot in (0, 1, 2)
                                and rows[index]['video_id'] == clip['clip_id']
                                and rows[index]['frame_ids'][slot] == frame_index and times[index, slot, 1] == 1,
                                'Crop request differs from original row/time mapping')
                        for scale, physical in enumerate((80, 192)):
                            side = max(32, round(physical*clip['height']/720))
                            pending.append(old.crop_rgb96(rgb, request['x'], request['y'], side))
                            destinations.append((index, slot, scale))
                            if len(pending) >= args.batch_size:
                                flush()
            require(count == clip['frame_count'] == len(expected), 'Former33 decoded count differs')
            frames_total += count
            rgb_total += converted
            per_video.append(dict(video_id=clip['clip_id'], media_sha256=clip['media_sha256'],
                native_frames=count, rgb_frames=converted, pts_and_dimensions_verified=True,
                wall_seconds=time.perf_counter()-tick))
            state.update(stage='extracting_additional33', decoded_videos=len(per_video),
                         decoded_frames=frames_total, model_forward_calls=calls)
            save(args.output/'run_status.json', state)
        flush()
        require(np.array_equal(encoded, np.broadcast_to(times[:, :, 1:2].astype(bool), encoded.shape)),
                'Final stem availability differs from fixed temporal mask')
        features.flush()
        del features
        for name in ('rows.jsonl', 'time_features.npy'):
            shutil.copyfile(args.plan.parent/name, args.output/name)
        files = {name:dict(bytes=(args.output/name).stat().st_size, sha256=sha(args.output/name))
                 for name in ('stem_features.npy', 'rows.jsonl', 'time_features.npy')}
        receipt = dict(schema='final154_visual_stem_cache.v1', completed=True, plan_sha256=sha(args.plan),
            base121_stem_sha256=plan['base121_cache']['files']['stem_features.npy']['sha256'],
            files=files, shape=list(shape), dtype='float32', rows=20641, development_candidates=17559,
            former33_candidates=3082, pretrained_sha256=sha(args.weights),
            model_source_sha256=plan['model']['source']['sha256'], extractor_source_sha256=sha(__file__),
            old_extractor_source_sha256=plan['sources']['old_extractor']['sha256'], prefix121_exact=True,
            decoded_videos=len(per_video), decoded_frames=frames_total, rgb_frames=rgb_total,
            valid_new_crops=int(encoded[17559:].sum()), model_forward_calls=calls, per_video=per_video,
            full_rgb_cache_written=False, training_runs=0, external130_used=False, public15_used=False,
            wall_seconds=time.perf_counter()-started)
        save(args.output/'receipt.json', receipt)
        state.update(completed=True, exit_code=0, stage='complete', model_forward_calls=calls,
                     receipt_sha256=sha(args.output/'receipt.json'))
        print(json.dumps(dict(completed=True, decoded_videos=33, new_candidates=3082,
                             model_forward_calls=calls, wall_seconds=time.perf_counter()-started)), flush=True)
    except BaseException:
        state.update(stage='failed', error=traceback.format_exc())
        raise
    finally:
        state.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started)
        save(args.output/'run_status.json', state)
        (args.output/'exit_code.txt').write_text(str(state['exit_code'])+'\n', encoding='ascii')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('assets-root', 'base-cache', 'weights', 'plan', 'output', 'video-root'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    require(args.execute and sys.platform == 'linux' and args.batch_size > 0,
            'Actual extraction requires explicit Linux server execution')
    execute(args)


if __name__ == '__main__':
    main()
