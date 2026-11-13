"""Opt-in visual154 with the previously verified local-PTS feature window."""
from types import FunctionType, MethodType
from time import perf_counter

from participant.channel_adaptation_v001 import inputs as original_inputs
from participant.visual_calibration_v001.solution import Solution as VisualSolution
from .features import features


# Give the unchanged builder its own feature function; never mutate the original
# module or another Solution's runtime. Proposal/support/receipt code is shared.
_original_build = original_inputs.build_view_inputs
build_view_inputs = FunctionType(_original_build.__code__,
    dict(_original_build.__globals__, _features=features), _original_build.__name__,
    _original_build.__defaults__, _original_build.__closure__)
build_view_inputs.__kwdefaults__ = _original_build.__kwdefaults__


def _predict(self, video, original_observations, augmented_observations, anchors, *, view='augmented'):
    started = perf_counter()
    prepared = build_view_inputs(video, original_observations, augmented_observations, anchors, view=view)
    result = self.predict_prepared(prepared['inputs'], prepared['window_rows'], prepared['window_windows'])
    result.update(input_receipt=prepared['receipt'], decision_seconds=perf_counter()-started)
    return result


class Solution(VisualSolution):
    def prepare(self, context):
        super().prepare(context)
        base = self.event_runtime.base
        if base.predict.__func__.__globals__.get('build_view_inputs') is not _original_build:
            raise ValueError('Expected the unchanged CHANNEL input builder')
        base.predict = MethodType(_predict, base)
        context.info['native_window_features'] = dict(version='v001',
            source_experiment='20261109_channel_speed_equivalent',
            scope='this_runtime_only', original_candidate_support_preserved=True)
