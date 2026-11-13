"""Independent side fits; no center coordinate, GT, IDs or selection state in fits."""
import numpy as np
from . import normalization as original

WINDOW_SECONDS = 0.12
MAX_GAP_SECONDS = 0.08
EPS = 1e-6  # Existing cached features store actual seconds as float32.
SIDE_NAMES = ('known', 'support_fraction', 'span_seconds', 'vx', 'vy',
              'residual_known', 'residual_rms')
EXTRA_NAMES = tuple(f'{side}_{name}' for side in ('left', 'right') for name in SIDE_NAMES) + (
    'both_known', 'delta_vx', 'delta_vy')
VALIDITY = {2: 0, 3: 0, 4: 0, 6: 5, 9: 7, 10: 7, 11: 7, 13: 12, 15: 14, 16: 14}


def augment(sequence, summary):
    """Return N x 44; coordinates in image fractions, slopes per actual second.

    Each side excludes the center, fits confident support independently, and
    rejects support gaps above 80ms. Two points define velocity but provide no
    residual evidence. A zero residual with residual_known=0 is missing data.
    """
    sequence = original.validate(sequence, 'tcn').astype(np.float64, copy=False)
    summary = original.validate(summary, 'mlp')
    if len(sequence) != len(summary):
        raise ValueError('Sequence and summary row counts differ')
    extras = np.zeros((len(sequence), len(EXTRA_NAMES)), dtype=np.float64)
    for i, row in enumerate(sequence):
        for side, indices in enumerate((range(7), range(8, 15))):
            support = [row[j] for j in indices if row[j, 1] == 1 and
                       0 < abs(row[j, 3]) <= WINDOW_SECONDS + EPS and
                       (row[j, 3] < 0 if side == 0 else row[j, 3] > 0)]
            offset = side * 7
            extras[i, offset + 1] = len(support) / 3.0
            if len(support) < 2:
                continue
            support = np.asarray(sorted(support, key=lambda p: p[3]))
            t, xy = support[:, 3], support[:, 5:7]
            if (np.diff(t) <= 0).any() or np.diff(t).max() > MAX_GAP_SECONDS + EPS:
                continue
            if np.abs(t).min() > MAX_GAP_SECONDS + EPS:
                continue
            centered_t = t - t.mean()
            velocity = (centered_t[:, None] * (xy - xy.mean(0))).sum(0) / (centered_t @ centered_t)
            residual = xy - (xy.mean(0) + centered_t[:, None] * velocity)
            extras[i, offset] = 1.0
            extras[i, offset + 2:offset + 5] = (t[-1] - t[0], *velocity)
            if len(t) >= 3:
                extras[i, offset + 5] = 1.0
                extras[i, offset + 6] = np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
        if extras[i, 0] and extras[i, 7]:
            extras[i, 14] = 1.0
            extras[i, 15:17] = extras[i, 10:12] - extras[i, 3:5]
    result = np.concatenate((summary, extras), axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite augmented features')
    return result


def fit_statistics(training_features):
    base = original.fit_statistics(training_features[:, :27], 'mlp')
    x = training_features[:, 27:]
    mean, scale, counts = np.zeros(17), np.ones(17), np.zeros(17, dtype=np.int64)
    for channel, mask in VALIDITY.items():
        values = x[x[:, mask] == 1, channel]
        counts[channel] = len(values)
        if len(values):
            mean[channel] = values.mean()
            std = values.std(ddof=0)
            scale[channel] = std if std >= 1e-6 else 1.0
    return {'base': base, 'mean': mean.tolist(), 'scale': scale.tolist(),
            'valid_counts': counts.tolist(), 'extra_names': list(EXTRA_NAMES),
            'training_rows': len(training_features)}


def transform(features, statistics):
    base = original.transform(features[:, :27], statistics['base'], 'mlp')
    extra = features[:, 27:].astype(np.float64, copy=True)
    for channel, mask in VALIDITY.items():
        extra[:, channel] = np.where(extra[:, mask] == 1,
            (extra[:, channel] - statistics['mean'][channel]) / statistics['scale'][channel], 0.)
    out = np.concatenate((base, extra), axis=1).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite normalized features')
    return out
