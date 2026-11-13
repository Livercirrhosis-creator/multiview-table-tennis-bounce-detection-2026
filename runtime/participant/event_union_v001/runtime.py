"""Two fixed event heads share one native candidate and feature extraction.

The branch decisions and late union exactly follow their existing standalone
rules. Raw class margins rank disagreements; they are not calibrated joint
probabilities. No labels, video-source identities or evaluation code are used.
"""
import math
import time

from ..event_decision_v001.runtime import Runtime as TrajectoryRuntime, build_inputs
from ..event_decision_v001.selection import select_events
from ..event_window_v001.runtime import Runtime as WindowRuntime
from ..event_window_v001.selection import select_window_events
from ..event_window_v001.windows import build_windows


def combine_events(old_events, window_events, *, minimum_separation_seconds=.08):
    """Union selected branch events, resolve same-frame ties, select in time."""
    winners = {}
    for origin, events in (("old", old_events), ("window", window_events)):
        for event in events:
            p = event["probabilities"]
            margin = math.log(max(p[0], 1e-30)) - math.log(max(p[1], p[2], 1e-30))
            frame = event["frame_id"]
            rank = (margin, origin == "window")
            previous = winners.get(frame)
            available = [] if previous is None else previous["available_models"]
            available = sorted(set(available + [origin]))
            if previous is None or rank > previous["rank"]:
                winners[frame] = dict(event=event, rank=rank, selected_model=origin,
                                      available_models=available)
            else:
                previous["available_models"] = available
    ordered = [winners[f] for f in sorted(winners)]
    candidates = [dict(native_frame=w["event"]["frame_id"],
        native_pts_seconds=w["event"]["pts_seconds"],
        center_observation=dict(x=w["event"]["x"], y=w["event"]["y"]),
        candidate_id=w["event"]["candidate_id"], sources=w["event"]["sources"])
        for w in ordered]
    selected = select_events(candidates, [w["event"]["probabilities"] for w in ordered],
                             minimum_separation_seconds=minimum_separation_seconds)
    for event in selected:
        winner = winners[event["frame_id"]]
        event.update(selected_model=winner["selected_model"], available_models=winner["available_models"])
    return selected


class Runtime:
    def __init__(self, old_weights, old_statistics, window_weights, window_statistics,
                 *, minimum_separation_seconds=.08):
        self.old = TrajectoryRuntime(old_weights, old_statistics,
                                    minimum_separation_seconds=minimum_separation_seconds)
        self.window = WindowRuntime(window_weights, window_statistics,
                                    minimum_separation_seconds=minimum_separation_seconds)
        self.minimum_separation_seconds = minimum_separation_seconds

    def predict(self, video, observations, original_candidates, external_candidates=()):
        started = time.perf_counter()
        pool, sequence, summary, diagnostics = build_inputs(
            video, observations, original_candidates, external_candidates)
        built = time.perf_counter()
        windows = build_windows(video, observations, pool)
        windowed = time.perf_counter()
        old_probabilities = self.old.probabilities(sequence, summary)
        old_scored = time.perf_counter()
        presence, location = self.window.scores(sequence, summary, windows)
        window_scored = time.perf_counter()
        old_events = select_events(pool, old_probabilities,
                                   minimum_separation_seconds=self.minimum_separation_seconds)
        window_events = select_window_events(pool, windows, presence, location,
                                             minimum_separation_seconds=self.minimum_separation_seconds)
        branch_selected = time.perf_counter()
        events = combine_events(old_events, window_events,
                                minimum_separation_seconds=self.minimum_separation_seconds)
        finished = time.perf_counter()
        return dict(events=events, old_events=old_events, window_events=window_events,
            candidates=pool, windows=windows, candidate_diagnostics=diagnostics,
            old_probabilities=old_probabilities, presence_probabilities=presence, location_logits=location,
            build_inputs_calls=1, costs_seconds=dict(
                candidates_and_features=built-started, windows=windowed-built,
                old_model=old_scored-windowed, window_model=window_scored-old_scored,
                branch_selection=branch_selected-window_scored, union_selection=finished-branch_selected,
                total=finished-started))
