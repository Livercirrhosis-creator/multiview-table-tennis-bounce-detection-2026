"""NumPy inference for the fixed three-class linear head."""
import numpy as np
from .features import FEATURE_NAMES


def predict(features, weights):
    if weights["feature_names"] != FEATURE_NAMES:
        raise ValueError("Feature schema mismatch")
    x = np.asarray(features, dtype=np.float64)
    z = (x - np.asarray(weights["mean"])) / np.asarray(weights["scale"])
    logits = z @ np.asarray(weights["coefficients"]).T + np.asarray(weights["intercept"])
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return logits, exp / exp.sum(axis=-1, keepdims=True)
