"""Complete CHANNEL chain followed by the unchanged E06 post gate."""
from ..channel_adaptation_v001.solution import Solution as ParentSolution
from .runtime import TableGuardMixin


class Solution(TableGuardMixin, ParentSolution):
    table_parent = "channel_adaptation_v001"
