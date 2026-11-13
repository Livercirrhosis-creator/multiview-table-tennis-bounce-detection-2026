"""One-pass adapted event inference; no extra fusion network or image model."""
import time

from participant.dense_context_v001.union_runtime import Runtime as BaseRuntime
from participant.event_decision_v001.selection import select_events
from participant.event_window_v001.selection import select_window_events
from participant.event_union_v001.runtime import combine_events
from .inputs import build_view_inputs


class Runtime(BaseRuntime):
    def predict_prepared(self, inputs, parent_rows, parent_windows):
        started = time.perf_counter()
        pool, sequence, summary, windows, diagnostics = inputs
        probabilities = self.dense.probabilities(sequence, summary, windows)
        dense_events = select_events(pool, probabilities,
                                    minimum_separation_seconds=self.minimum_separation_seconds)
        parent_pool = [pool[i] for i in parent_rows]
        presence, location = self.window.scores(sequence[parent_rows], summary[parent_rows], parent_windows)
        window_events = select_window_events(parent_pool, parent_windows, presence, location,
                                             minimum_separation_seconds=self.minimum_separation_seconds)
        events = combine_events(dense_events, window_events,
                                minimum_separation_seconds=self.minimum_separation_seconds)
        return dict(events=events, dense_events=dense_events, window_events=window_events,
                    candidates=pool, probabilities=probabilities, presence=presence,
                    location_logits=location, candidate_diagnostics=diagnostics,
                    decision_seconds=time.perf_counter()-started)

    def predict(self, video, original_observations, augmented_observations, anchors, *, view='augmented'):
        started = time.perf_counter()
        prepared = build_view_inputs(video, original_observations, augmented_observations, anchors, view=view)
        result = self.predict_prepared(prepared['inputs'], prepared['window_rows'], prepared['window_windows'])
        result.update(input_receipt=prepared['receipt'], decision_seconds=time.perf_counter()-started)
        return result
