"""Continuous local measurements; no labels, source IDs or expert predictions."""
from __future__ import annotations

import numpy as np
from ..gap_contact_v001.proposal import CONFIG, _fit


FEATURE_NAMES = [
    "gap_seconds", "contact_gap_fraction", "left_support_count", "right_support_count",
    "left_support_span_seconds", "right_support_span_seconds",
    "left_max_step_seconds", "right_max_step_seconds", "profile_width_seconds",
    "fit_rmse_image_heights", "log1p_bic_improvement", "log1p_design_condition",
    "contact_x_image_widths", "contact_y_image_heights",
    "left_velocity_x", "left_velocity_y", "right_velocity_x", "right_velocity_y",
    "acceleration_x", "acceleration_y", "log1p_left_speed", "log1p_right_speed",
    "log_speed_ratio", "turn_cosine", "turn_sine",
    "left_unit_x", "left_unit_y", "right_unit_x", "right_unit_y",
    "x_direction_product", "y_direction_product", "relative_velocity_jump",
    "fractional_speed_change",
]


def extract(video, observations, candidate, gap):
    """Reconstruct the frozen hinge fit and describe its two local velocities.

    Speeds use image heights/second, time uses native PTS, and all positions
    refer to the original image. The candidate's fitted time is distinct from
    the nearest frame timestamp. This function never uses supervision fields.
    """
    height, width = float(video["height"]), float(video["width"])
    pts = np.asarray(video["pts_seconds"], dtype=np.float64)
    left = np.asarray(gap["left_support_frames"], dtype=np.int64)
    right = np.asarray(gap["right_support_frames"], dtype=np.int64)
    support = np.concatenate([left, right])
    wanted = set(support.tolist())
    measured = {int(r["native_frame"]): r["detection"] for r in observations
                if not r["center_is_padding"] and r["native_frame"] in wanted
                and r["detection"] is not None}
    if set(measured) != wanted:
        raise ValueError("Missing fit support")
    if any(float(d["conf"]) < CONFIG["reliable_confidence"] for d in measured.values()):
        raise ValueError("Fit support is not reliable")
    lo, hi = pts[candidate["left_frame"]], pts[candidate["right_frame"]]
    origin, scale = (lo + hi) / 2., CONFIG["side_context_seconds"]
    tau = (candidate["tau_seconds"] - origin) / scale
    xy = np.asarray([[measured[int(f)]["x"], measured[int(f)]["y"]] for f in support]) / height
    fit = _fit((pts[support] - origin) / scale, xy, tau)
    if fit is None:
        raise ValueError("Accepted proposal has no valid hinge fit")
    coef = fit["coefficients"]
    vleft = (coef[1] + 2. * coef[2] * tau) / scale
    vright = vleft + coef[3] / scale
    acceleration = 2. * coef[2] / (scale * scale)
    sl, sr = float(np.linalg.norm(vleft)), float(np.linalg.norm(vright))
    # A numerical floor only: not an acceptance threshold or assumed frame rate.
    epsilon = 1e-6
    ul, ur = vleft / max(sl, epsilon), vright / max(sr, epsilon)
    profile = gap["profile_delta_bic2_seconds"]
    values = [
        hi - lo, (candidate["tau_seconds"] - lo) / (hi - lo), len(left), len(right),
        pts[left[-1]] - pts[left[0]], pts[right[-1]] - pts[right[0]],
        np.max(np.diff(pts[left])), np.max(np.diff(pts[right])), profile[1] - profile[0],
        np.sqrt(fit["sse"] / xy.size), np.log1p(gap["bic_improvement"]),
        np.log1p(fit["condition"]), candidate["x"] / width, candidate["y"] / height,
        *vleft, *vright, *acceleration, np.log1p(sl), np.log1p(sr),
        np.log((sr + epsilon) / (sl + epsilon)), float(ul @ ur),
        ul[0] * ur[1] - ul[1] * ur[0], *ul, *ur,
        ul[0] * ur[0], ul[1] * ur[1], np.linalg.norm(vright - vleft) / (sl + sr + epsilon),
        (sr - sl) / (sl + sr + epsilon),
    ]
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid feature vector")
    return result
