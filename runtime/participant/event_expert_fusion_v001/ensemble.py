"""One predetermined equal-probability mixture, without fitted parameters."""
import numpy as np


def equal_probabilities(trajectory, spatial):
    first, second = np.asarray(trajectory), np.asarray(spatial)
    if first.ndim != 2 or first.shape[1] != 3 or second.shape != first.shape:
        raise ValueError("Aligned N x 3 class probabilities are required")
    for values in (first, second):
        if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1) or
                not np.allclose(values.sum(axis=1), 1., rtol=0, atol=2e-6)):
            raise ValueError("Invalid expert class probability vector")
    return .5 * (first + second)
