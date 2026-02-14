"""
Utility functions and helpers for the LuisaCompute Python DSL v2.
"""

from __future__ import annotations
from typing import Optional


class UnrolledRange:
    """
    Marker class for unrolled loops.
    
    Usage:
        for i in unrolled(range(4)):
            ...  # This loop is unrolled at compile time
    
    The loop body will be replicated for each iteration.
    Use only for small iteration counts to avoid code bloat!
    """
    
    def __init__(self, start: int, stop: Optional[int] = None, step: int = 1):
        if stop is None:
            start, stop = 0, start
        self.start = start
        self.stop = stop
        self.step = step
    
    def __iter__(self):
        """Python-side iteration (for reference)."""
        return iter(range(self.start, self.stop, self.step))
    
    def __len__(self) -> int:
        """Return the number of iterations."""
        return max(0, (self.stop - self.start + self.step - 1) // self.step)


def unrolled(r: range) -> UnrolledRange:
    """
    Mark a range for compile-time unrolling.
    
    Usage:
        for i in unrolled(range(4)):      # Unrolled: 0, 1, 2, 3
        for i in unrolled(range(1, 5)):   # Unrolled: 1, 2, 3, 4
        for i in unrolled(range(0, 8, 2)):# Unrolled: 0, 2, 4, 6
    
    The loop body will be replicated for each iteration at compile time.
    This eliminates loop overhead but increases code size.
    
    Only use for small iteration counts (typically < 16) to avoid:
    - Excessive compilation times
    - Code bloat
    - Instruction cache pressure
    
    For larger loops, use regular range() which generates device-side loops.
    """
    return UnrolledRange(r.start, r.stop, r.step)
