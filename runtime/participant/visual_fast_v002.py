"""Compose the independent CPU and native-window fast paths."""

from .visual_fast_cpu_v001 import Solution as CpuFastSolution
from .visual_native_window_v001.solution import Solution as NativeWindowSolution


class Solution(CpuFastSolution, NativeWindowSolution):
    """Both setup methods run once before the same visual model starts."""
