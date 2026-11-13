"""Share dense native features while preserving the fixed sparse window expert."""
import time

from ..event_decision_v001.selection import select_events
from ..event_window_v001.runtime import Runtime as WindowRuntime
from ..event_window_v001.selection import select_window_events
from ..event_window_v001.windows import build_windows
from ..event_union_v001.runtime import combine_events
from .candidates import DENSE_SOURCE
from .runtime import Runtime as DenseRuntime, build_inputs


class Runtime:
    def __init__(self, dense_weights, dense_statistics, window_weights, window_statistics,
                 *, minimum_separation_seconds=.08):
        self.dense = DenseRuntime(dense_weights,dense_statistics,
                                 minimum_separation_seconds=minimum_separation_seconds)
        self.window = WindowRuntime(window_weights,window_statistics,
                                    minimum_separation_seconds=minimum_separation_seconds)
        self.minimum_separation_seconds = minimum_separation_seconds

    def predict_from_inputs(self, video, observations, pool, sequence, summary, windows, diagnostics):
        """Consume one existing dense build; the old expert sees only parent rows."""
        started = time.perf_counter()
        dense_probabilities = self.dense.probabilities(sequence,summary,windows)
        dense_events = select_events(pool,dense_probabilities,
                                     minimum_separation_seconds=self.minimum_separation_seconds)
        parent_rows = [i for i,candidate in enumerate(pool) if DENSE_SOURCE not in candidate["sources"]]
        parent_pool = [pool[i] for i in parent_rows]
        parent_windows = build_windows(video,observations,parent_pool)
        presence,location = self.window.scores(sequence[parent_rows],summary[parent_rows],parent_windows)
        window_events = select_window_events(parent_pool,parent_windows,presence,location,
                                             minimum_separation_seconds=self.minimum_separation_seconds)
        events = combine_events(dense_events,window_events,
                                minimum_separation_seconds=self.minimum_separation_seconds)
        return dict(events=events,dense_events=dense_events,window_events=window_events,
            candidates=pool,windows=windows,probabilities=dense_probabilities,
            parent_rows=parent_rows,old_window_candidates=parent_pool,old_windows=parent_windows,
            presence_probabilities=presence,location_logits=location,candidate_diagnostics=diagnostics,
            build_inputs_calls=0,decision_seconds=time.perf_counter()-started)

    def predict(self, video, observations, original_candidates, external_candidates=()):
        started = time.perf_counter()
        observations = list(observations)
        inputs = build_inputs(video,observations,original_candidates,external_candidates)
        result = self.predict_from_inputs(video,observations,*inputs)
        result.update(build_inputs_calls=1,decision_seconds=time.perf_counter()-started)
        return result
