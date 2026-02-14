"""
Ray tracing builtin functions for the LuisaCompute Python DSL v2.

Ray tracing and acceleration structure operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ast import Value, InstructionValue

from ..ast import IROp
from ..types import bool_, uint, float3, float4
from .math import _get_builder


# ============================================================================
# Ray Types
# ============================================================================

class Ray:
    """Ray structure for ray tracing."""
    
    def __init__(self, origin: float3, direction: float3, t_min: float = 0.0, t_max: float = 1e30):
        self.origin = origin
        self.direction = direction
        self.t_min = t_min
        self.t_max = t_max
    
    def at(self, t: float) -> float3:
        """Get point at distance t along the ray."""
        # This would need proper DSL implementation
        pass


class TriangleHit:
    """Hit result for triangle intersection."""
    
    def __init__(self):
        self.inst: uint = 0
        self.prim: uint = 0
        self.bary: float3 = float3(0.0, 0.0, 0.0)
        self.hit: bool = False


class ProceduralHit:
    """Hit result for procedural primitive."""
    
    def __init__(self):
        self.inst: uint = 0
        self.prim: uint = 0


class CommittedHit:
    """Committed hit from ray query."""
    
    def __init__(self):
        self.inst: uint = 0
        self.prim: uint = 0
        self.bary: float3 = float3(0.0, 0.0, 0.0)
        self.t: float = 0.0
        self.hit: bool = False


# ============================================================================
# Ray Tracing Queries
# ============================================================================

def trace_closest(accel: Value, ray: Ray, mask: uint = 0xFF) -> InstructionValue:
    """
    Trace a ray and return the closest hit.
    
    Args:
        accel: Acceleration structure handle
        ray: Ray to trace
        mask: Instance mask (default: 0xFF)
    
    Returns:
        TriangleHit result
    """
    # In real implementation, would construct ray from components
    return _get_builder()._emit(IROp.TRACE_CLOSEST, TriangleHit, [accel, mask])


def trace_any(accel: Value, ray: Ray, mask: uint = 0xFF) -> InstructionValue:
    """
    Trace a ray and return True if any hit is found (occlusion test).
    
    Args:
        accel: Acceleration structure handle
        ray: Ray to trace
        mask: Instance mask (default: 0xFF)
    
    Returns:
        True if any hit found
    """
    return _get_builder()._emit(IROp.TRACE_ANY, bool_, [accel, mask])


def ray_query_all(accel: Value, ray: Ray, mask: uint = 0xFF) -> InstructionValue:
    """
    Create a ray query for all potential hits (inline traversal).
    
    Args:
        accel: Acceleration structure handle
        ray: Ray to trace
        mask: Instance mask (default: 0xFF)
    
    Returns:
        RayQuery object for iterative traversal
    """
    from ..types import RayQuery
    return _get_builder()._emit(IROp.RAY_QUERY_ALL, RayQuery(query_any=False), [accel, mask])


def ray_query_any(accel: Value, ray: Ray, mask: uint = 0xFF) -> InstructionValue:
    """
    Create a ray query for any hit (inline traversal).
    
    Args:
        accel: Acceleration structure handle
        ray: Ray to trace
        mask: Instance mask (default: 0xFF)
    
    Returns:
        RayQuery object for iterative traversal
    """
    from ..types import RayQuery
    return _get_builder()._emit(IROp.RAY_QUERY_ANY, RayQuery(query_any=True), [accel, mask])


# ============================================================================
# Ray Query Operations
# ============================================================================

def ray_query_world_space_ray(query: Value) -> InstructionValue:
    """Get the world-space ray from a ray query."""
    return _get_builder()._emit(IROp.RAY_QUERY_WORLD_RAY, Ray, [query])


def ray_query_proceed(query: Value) -> InstructionValue:
    """
    Proceed to the next candidate hit in a ray query.
    
    Returns True if there are more candidates.
    """
    return _get_builder()._emit(IROp.RAY_QUERY_PROCEED, bool_, [query])


def ray_query_committed_hit(query: Value) -> InstructionValue:
    """Get the committed (closest) hit from a ray query."""
    return _get_builder()._emit(IROp.RAY_QUERY_COMMITTED_HIT, CommittedHit, [query])


def ray_query_candidate_triangle_hit(query: Value) -> InstructionValue:
    """Get the current candidate triangle hit from a ray query."""
    return _get_builder()._emit(IROp.RAY_QUERY_CANDIDATE_TRIANGLE_HIT, TriangleHit, [query])


def ray_query_candidate_procedural_hit(query: Value) -> InstructionValue:
    """Get the current candidate procedural hit from a ray query."""
    return _get_builder()._emit(IROp.RAY_QUERY_CANDIDATE_PROCEDURAL_HIT, ProceduralHit, [query])


def ray_query_commit_triangle(query: Value) -> InstructionValue:
    """Commit the current triangle candidate as the closest hit."""
    from ..types import Void
    return _get_builder()._emit(IROp.RAY_QUERY_COMMIT_TRIANGLE, Void(), [query])


def ray_query_commit_procedural(query: Value, t: Value) -> InstructionValue:
    """
    Commit a procedural primitive hit.
    
    Args:
        query: Ray query
        t: Hit distance along the ray
    """
    from ..types import Void
    return _get_builder()._emit(IROp.RAY_QUERY_COMMIT_PROCEDURAL, Void(), [query, t])


def ray_query_terminate(query: Value) -> InstructionValue:
    """Terminate the ray query early."""
    from ..types import Void
    return _get_builder()._emit(IROp.RAY_QUERY_TERMINATE, Void(), [query])


# ============================================================================
# Acceleration Structure Operations
# ============================================================================

def accel_instance_transform(accel: Value, instance_id: Value) -> InstructionValue:
    """
    Get the transformation matrix of an instance.
    
    Args:
        accel: Acceleration structure
        instance_id: Instance index
    
    Returns:
        4x4 transformation matrix (float4x4)
    """
    from ..types import float4x4
    return _get_builder()._emit(IROp.ACCEL_INSTANCE_TRANSFORM, float4x4, [accel, instance_id])


def accel_instance_user_id(accel: Value, instance_id: Value) -> InstructionValue:
    """Get the user-defined ID of an instance."""
    return _get_builder()._emit(IROp.ACCEL_INSTANCE_USER_ID, uint, [accel, instance_id])


def accel_instance_visibility_mask(accel: Value, instance_id: Value) -> InstructionValue:
    """Get the visibility mask of an instance."""
    return _get_builder()._emit(IROp.ACCEL_INSTANCE_VISIBILITY_MASK, uint, [accel, instance_id])


def make_ray(origin: float3, direction: float3, t_min: float = 0.0, t_max: float = 1e30) -> Ray:
    """Construct a ray."""
    return Ray(origin, direction, t_min, t_max)
