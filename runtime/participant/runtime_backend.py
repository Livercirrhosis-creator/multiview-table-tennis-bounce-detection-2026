"""Participant-only backend selection; original ball model and detector unchanged."""
import hashlib
import os
from pathlib import Path
import numpy as np
from .ball_detector import BallDetector

BALL_SHA256 = 'a5a467f134ae7e2fc5679e395dca9af12b6dfcc65cee51e633c465beb0c61543'


def prepare_detector(solution, context):
    path = Path(os.environ.get('PINGPANG_BALL_ONNX', str(Path(__file__).parent/'weights/ball.onnx')))
    if not path.is_file():
        raise FileNotFoundError(f'Official ball.onnx missing: {path}; set PINGPANG_BALL_ONNX or install participant/weights/ball.onnx')
    if hashlib.sha256(path.read_bytes()).hexdigest() != BALL_SHA256:
        raise ValueError('ball.onnx identity differs from the selected official original model')
    backend = os.environ.get('PINGPANG_BACKEND', 'tensorrt')
    if backend == 'tensorrt':
        from core.trt import load_or_build
        model, how = load_or_build(path, Path(context.output_dir)/'trt_cache', input_name='input', shape=(1,9,288,512), log=print)
        infer = model.infer
        context.info['backend'] = f'tensorrt_{how}'
    elif backend == 'ort_cpu':
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get('PINGPANG_ORT_THREADS', '4'))
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
        name = session.get_inputs()[0].name
        def infer(array):
            return session.run(None, {name:array})[0]
        context.info['backend'] = 'onnxruntime_cpu_explicit'
        context.info['ort_threads'] = opts.intra_op_num_threads
        context.info['ort_version'] = ort.__version__
    else:
        raise ValueError('PINGPANG_BACKEND must be tensorrt or ort_cpu; no silent backend fallback')
    solution.det = BallDetector(infer, path)
    solution.det.detect(np.zeros((288,512,3), np.uint8), 512, 288)
    solution.traj = []
    solution.last_emit = -10_000
    context.info['ball_sha256'] = BALL_SHA256
