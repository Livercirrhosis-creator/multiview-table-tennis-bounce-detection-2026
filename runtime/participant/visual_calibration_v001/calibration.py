"""Use a source-fitted visual evidence scale without changing neutral branch scores."""
import numpy as np


def probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()
            or (values < 0).any() or not np.allclose(values.sum(1), 1., atol=2e-6)):
        raise ValueError('Expected finite normalized three-class probabilities')
    return values


def calibrate(values, a):
    values = probabilities(values)
    if not np.isfinite(a) or not 0 <= a <= 1:
        raise ValueError('Expected the fixed softening range [0,1]')
    if a == 1:
        return values.copy()
    z = a * np.log(np.maximum(values, 1e-30))
    z -= z.max(axis=1, keepdims=True)
    output = np.exp(z)
    return output / output.sum(axis=1, keepdims=True)


def fuse(dense, calibrated_visual, a, *, scale_aware, floor=1e-8):
    dense, visual = probabilities(dense), probabilities(calibrated_visual)
    if dense.shape != visual.shape or not np.isfinite(a) or not 0 <= a <= 1:
        raise ValueError('Mismatched candidate classes or evidence scale')
    if scale_aware and a == 0:
        return dense.copy()
    result = np.sqrt(np.maximum(dense, floor) * np.maximum(visual, floor))
    result /= result.sum(axis=1, keepdims=True)
    if scale_aware and a != 1:
        result = np.exp(np.log(result) * (2. / (1. + a)))
        result /= result.sum(axis=1, keepdims=True)
    return result
