"""Compare a smooth arc with a continuous velocity change inside missing frames."""
from __future__ import annotations

from collections import Counter
import math
import numpy as np

CONFIG = dict(reliable_confidence=0.75, maximum_gap_seconds=0.240,
              side_context_seconds=0.320, minimum_points_per_side=5,
              maximum_points_per_side=8, bic_improvement_minimum=10.0,
              noise_floor_image_heights=0.002, maximum_rmse_image_heights=0.010,
              maximum_design_condition=1e8, grid_points=33, refinement_steps=24,
              h0_parameters=6, h1_parameters=9, final_spacing_seconds=0.080)


def _fit(u, xy, tau=None):
    design = np.column_stack([np.ones(len(u)), u, u * u])
    if tau is not None:
        design = np.column_stack([design, np.maximum(u - tau, 0.)])
    coefficients, _, rank, singular = np.linalg.lstsq(design, xy, rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] else math.inf
    if rank != design.shape[1] or condition > CONFIG["maximum_design_condition"]:
        return None
    residual = xy - design @ coefficients
    return dict(coefficients=coefficients, sse=float(np.square(residual).sum()), condition=condition)


def _bic(sse, count, parameters):
    variance = max(sse / count, CONFIG["noise_floor_image_heights"] ** 2)
    # Keep SSE/variance: below the variance floor it is no longer a constant.
    # The common noise-variance parameter and Gaussian constant cancel in H0-H1.
    return count * math.log(variance) + sse / variance + parameters * math.log(count)


def _best_change(u, xy, lower, upper):
    grid = np.linspace(lower, upper, CONFIG["grid_points"])
    fits = [_fit(u, xy, float(t)) for t in grid]
    values = [fit["sse"] if fit is not None else math.inf for fit in fits]
    winner = int(np.argmin(values))
    if not math.isfinite(values[winner]):
        return None
    best_tau, best_fit = float(grid[winner]), fits[winner]
    left, right = float(grid[max(0, winner - 1)]), float(grid[min(len(grid) - 1, winner + 1)])
    ratio = (math.sqrt(5.) - 1.) / 2.
    a, b = right - ratio * (right - left), left + ratio * (right - left)
    fa, fb = _fit(u, xy, a), _fit(u, xy, b)
    for _ in range(CONFIG["refinement_steps"]):
        va = fa["sse"] if fa is not None else math.inf
        vb = fb["sse"] if fb is not None else math.inf
        for tau, fit in ((a, fa), (b, fb)):
            if fit is not None and fit["sse"] < best_fit["sse"]:
                best_tau, best_fit = tau, fit
        if va < vb:
            right, b, fb = b, a, fa
            a = right - ratio * (right - left)
            fa = _fit(u, xy, a)
        else:
            left, a, fa = a, b, fb
            b = left + ratio * (right - left)
            fb = _fit(u, xy, b)
    best_bic = _bic(best_fit["sse"], xy.size, CONFIG["h1_parameters"])
    near = [float(t) for t, fit in zip(grid, fits)
            if fit is not None and _bic(fit["sse"], xy.size, CONFIG["h1_parameters"]) <= best_bic + 2.]
    return best_tau, best_fit, ([min(near), max(near)] if near else [best_tau, best_tau])


def propose(video, observations):
    """Consume only image metadata/ball measurements; return inferred contact proposals.

    Coordinates use the continuous fitted contact, while frame_id is the nearest
    real frame. No label, existing event, or protected evaluation input is needed.
    """
    vid = video["clip_id"]
    pts = np.asarray(video["pts_seconds"], dtype=np.float64)
    width, height = int(video["width"]), int(video["height"])
    if len(pts) != video["frame_count"] or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("Expected a finite strictly increasing native clock")
    if width <= 0 or height <= 0:
        raise ValueError("Invalid native dimensions")
    reliable = {}
    for row in observations:
        if row["video_id"] != vid:
            raise ValueError("Mixed video observations")
        if row["center_is_padding"]:
            continue
        frame, detection = row["native_frame"], row["detection"]
        if frame is None or not 0 <= frame < len(pts):
            raise ValueError("Invalid native frame")
        if abs(float(row["native_pts_seconds"]) - pts[frame]) > 1e-8:
            raise ValueError("Observation/native clock mismatch")
        if detection is None:
            continue
        x, y, confidence = (float(detection[k]) for k in ("x", "y", "conf"))
        if (not all(math.isfinite(v) for v in (x, y, confidence)) or
                confidence < CONFIG["reliable_confidence"] or not (0 <= x < width and 0 <= y < height)):
            continue
        if frame in reliable:
            raise ValueError("Duplicate reliable frame")
        reliable[frame] = (x / height, y / height)
    frames = np.asarray(sorted(reliable), dtype=np.int64)
    times = pts[frames]
    source_gaps = [(int(g["before_frame_id"]), int(g["after_frame_id"]))
                   for g in video.get("source_pts_gaps", [])]
    candidates, gaps, reasons = [], [], Counter()
    for index in range(len(frames) - 1):
        lf, rf = int(frames[index]), int(frames[index + 1])
        if rf == lf + 1:
            continue
        lo, hi = float(pts[lf]), float(pts[rf])
        item = dict(left_frame=lf, right_frame=rf, gap_seconds=hi - lo,
                    missing_native_frames=rf - lf - 1, accepted=False)
        gaps.append(item)
        if hi - lo > CONFIG["maximum_gap_seconds"] + 1e-9:
            item["reason"] = "gap_too_long"
            continue
        left_start = int(np.searchsorted(times, lo - CONFIG["side_context_seconds"] - 1e-9))
        right_stop = int(np.searchsorted(times, hi + CONFIG["side_context_seconds"] + 1e-9, side="right"))
        left = frames[max(left_start, index + 1 - CONFIG["maximum_points_per_side"]):index + 1]
        right = frames[index + 1:min(right_stop, index + 1 + CONFIG["maximum_points_per_side"])]
        item.update(left_support_frames=left.tolist(), right_support_frames=right.tolist())
        if min(len(left), len(right)) < CONFIG["minimum_points_per_side"]:
            item["reason"] = "insufficient_local_support"
            continue
        support = np.concatenate([left, right])
        if any(int(support[0]) <= a < b <= int(support[-1]) for a, b in source_gaps):
            item["reason"] = "source_time_discontinuity"
            continue
        origin, scale = (lo + hi) / 2., CONFIG["side_context_seconds"]
        u = (pts[support] - origin) / scale
        xy = np.asarray([reliable[int(frame)] for frame in support], dtype=np.float64)
        smooth = _fit(u, xy)
        changed = _best_change(u, xy, (lo - origin) / scale, (hi - origin) / scale)
        if smooth is None or changed is None:
            item["reason"] = "ill_conditioned_fit"
            continue
        tau, fit, profile = changed
        tau_seconds = origin + scale * tau
        coefficients = fit["coefficients"]
        point = np.array([1., tau, tau * tau, 0.]) @ coefficients
        bic0 = _bic(smooth["sse"], xy.size, CONFIG["h0_parameters"])
        bic1 = _bic(fit["sse"], xy.size, CONFIG["h1_parameters"])
        rmse = math.sqrt(fit["sse"] / xy.size)
        frame = lf + int(np.argmin(np.abs(pts[lf:rf + 1] - tau_seconds)))
        x, y = (float(value * height) for value in point)
        item.update(tau_seconds=tau_seconds, frame_id=frame, predicted_xy=[x, y],
                    h0_sse=float(smooth["sse"]), h1_sse=float(fit["sse"]),
                    h0_bic=bic0, h1_bic=bic1, bic_improvement=bic0 - bic1,
                    rmse_image_heights=rmse, design_condition=fit["condition"],
                    velocity_jump_image_heights_per_second=(coefficients[3] / scale).tolist(),
                    profile_delta_bic2_seconds=[origin + scale * v for v in profile],
                    profile_note="Coarse likelihood profile, not calibrated timing confidence")
        if tau_seconds <= lo + 1e-7 or tau_seconds >= hi - 1e-7 or frame in (lf, rf):
            item["reason"] = "contact_maps_to_observed_boundary"
        elif not (0 <= x < width and 0 <= y < height):
            item["reason"] = "contact_outside_image"
        elif rmse > CONFIG["maximum_rmse_image_heights"]:
            item["reason"] = "large_fit_residual"
        elif bic0 - bic1 < CONFIG["bic_improvement_minimum"]:
            item["reason"] = "smooth_arc_not_rejected"
        else:
            item.update(accepted=True, reason="contact_proposed")
            candidates.append(dict(candidate_id=f"{vid}:gap:{lf}:{rf}", frame_id=frame,
                pts_seconds=float(pts[frame]), tau_seconds=tau_seconds, x=x, y=y,
                score=bic0 - bic1, source="gap_contact_v001_inferred_contact",
                coordinate_source="continuous_shared_curvature_hinge_fit",
                left_frame=lf, right_frame=rf))
    reasons.update(item["reason"] for item in gaps)
    return dict(video_id=vid, candidates=candidates, gaps=gaps, config=dict(CONFIG),
                counts=dict(reliable_observations=len(frames), gaps=len(gaps),
                            proposals=len(candidates), reasons=dict(reasons)))
