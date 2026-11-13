"""Fold-local observed-value statistics. Never accepts IDs, labels or holdout rows."""
import numpy as np

VALIDITY = {3: 0, 4: 0, 5: 0, 6: 0, 8: 7, 9: 7, 12: 10, 13: 11,
            15: 14, 16: 14, 18: 17, 19: 17, 21: 20, 22: 20}
BINARY = (0, 1, 7, 10, 11, 14, 17, 20)


def validate(x, kind):
    x = np.asarray(x)
    tail = (15, 23) if kind == "tcn" else (27,) if kind == "mlp" else None
    if tail is None or x.ndim != len(tail)+1 or x.shape[1:] != tail or x.shape[0] == 0:
        raise ValueError("Unexpected head feature shape/kind")
    if x.dtype.kind != "f" or not np.isfinite(x).all():
        raise ValueError("Features must be finite floating arrays")
    for mask in BINARY:
        if not np.isin(x[..., mask], [0., 1.]).all():
            raise ValueError("Observation-derived masks must be binary")
    return x


def fit_statistics(training_features, kind):
    x = validate(training_features, kind).astype(np.float64, copy=False)
    d = x.shape[-1]
    mean, scale, count = np.zeros(d), np.ones(d), np.zeros(d, dtype=np.int64)
    for ch, mask in VALIDITY.items():
        values = x[..., ch][x[..., mask] == 1]
        count[ch] = len(values)
        if len(values):
            mean[ch] = values.mean(dtype=np.float64)
            sd = values.std(ddof=0, dtype=np.float64)
            scale[ch] = sd if sd >= 1e-6 else 1.
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Nonfinite training statistics")
    return {"schema": "candidate_heads_normalization.v1", "kind": kind, "mean": mean.tolist(),
            "scale": scale.tolist(), "valid_counts": count.tolist(), "training_rows": len(x),
            "validity": {str(k): v for k, v in VALIDITY.items()}, "ddof": 0, "std_floor": 1e-6}


def transform(features, statistics, kind):
    x = validate(features, kind).astype(np.float64, copy=True)
    if statistics["kind"] != kind or statistics["validity"] != {str(k): v for k, v in VALIDITY.items()}:
        raise ValueError("Normalization kind/mask contract differs")
    mean = np.asarray(statistics["mean"], dtype=np.float64)
    scale = np.asarray(statistics["scale"], dtype=np.float64)
    if mean.shape != (x.shape[-1],) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid normalization statistics")
    for ch, mask in VALIDITY.items():
        x[..., ch] = np.where(x[..., mask] == 1, (x[..., ch]-mean[ch])/scale[ch], 0.)
    out = x.astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Float32 normalized features overflowed")
    return out
