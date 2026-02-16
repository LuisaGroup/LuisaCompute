"""
Ray tracing builtin functions for the LuisaCompute Python DSL v2.

Ray tracing and acceleration structure operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...transform.ir import Value, InstructionValue

from ...transform.op import Op
from ..types import Ray, TriangleHit, ProceduralHit, CommittedHit, struct, Accel, RayQuery
from ..jit import callable as dsl_callable
from ...transform.builder import get_current_builder


# ============================================================================
# Ray Tracing Queries
# ============================================================================

def trace_closest(accel: Value, ray: Ray, mask: Value = 0xFF) -> InstructionValue:
    """
    Trace a ray and return the closest hit.
    
    Args:
        accel: Acceleration structure handle
        ray: Ray to trace
        mask: Instance mask (default: 0xFF)
    
    Returns:
        TriangleHit result
    """
    return get_current_builder()._emit(Op.TRACE_CLOSEST, TriangleHit, [accel, ray, mask])


def trace_any(accel: Value, ray: Ray, mask: Value = 0xFF) -> InstructionValue:
    """
    Trace a ray and return True if any hit is found (occlusion test).
    
    Args:
        accel: Acceleration structure handle
        ray: Ray to trace
        mask: Instance mask (default: 0xFF)
    
    Returns:
        True if any hit found
    """
    from ..types import Bool
    return get_current_builder()._emit(Op.TRACE_ANY, Bool, [accel, ray, mask])


def ray_query_all(accel: Value, ray: Ray, mask: Value = 0xFF) -> InstructionValue:
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
    return get_current_builder()._emit(Op.RAY_QUERY_ALL, RayQuery(query_any=False), [accel, ray, mask])


def ray_query_any(accel: Value, ray: Ray, mask: Value = 0xFF) -> InstructionValue:
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
    return get_current_builder()._emit(Op.RAY_QUERY_ANY, RayQuery(query_any=True), [accel, ray, mask])


# ============================================================================
# Ray Query Operations
# ============================================================================

def ray_query_world_space_ray(query: Value) -> InstructionValue:
    """Get the world-space ray from a ray query."""
    return get_current_builder()._emit(Op.RAY_QUERY_WORLD_RAY, Ray, [query])


def ray_query_proceed(query: Value) -> InstructionValue:
    """
    Proceed to the next candidate hit in a ray query.
    
    Returns True if there are more candidates.
    """
    from ..types import Bool
    return get_current_builder()._emit(Op.RAY_QUERY_PROCEED, Bool, [query])


def ray_query_committed_hit(query: Value) -> InstructionValue:
    """Get the committed (closest) hit from a ray query."""
    return get_current_builder()._emit(Op.RAY_QUERY_COMMITTED_HIT, CommittedHit, [query])


def ray_query_candidate_triangle_hit(query: Value) -> InstructionValue:
    """Get the current candidate triangle hit from a ray query."""
    return get_current_builder()._emit(Op.RAY_QUERY_CANDIDATE_TRIANGLE_HIT, TriangleHit, [query])


def ray_query_candidate_procedural_hit(query: Value) -> InstructionValue:
    """Get the current candidate procedural hit from a ray query."""
    return get_current_builder()._emit(Op.RAY_QUERY_CANDIDATE_PROCEDURAL_HIT, ProceduralHit, [query])


def ray_query_commit_triangle(query: Value) -> InstructionValue:
    """Commit the current triangle candidate as the closest hit."""
    from ..types import Void
    return get_current_builder()._emit(Op.RAY_QUERY_COMMIT_TRIANGLE, Void(), [query])


def ray_query_commit_procedural(query: Value, t: Value) -> InstructionValue:
    """
    Commit a procedural primitive hit.
    
    Args:
        query: Ray query
        t: Hit distance along the ray
    """
    from ..types import Void
    return get_current_builder()._emit(Op.RAY_QUERY_COMMIT_PROCEDURAL, Void(), [query, t])


def ray_query_terminate(query: Value) -> InstructionValue:
    """Terminate the ray query early."""
    from ..types import Void
    return get_current_builder()._emit(Op.RAY_QUERY_TERMINATE, Void(), [query])


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
        4x4 transformation matrix (Float4x4)
    """
    from ..types import Float4x4
    return get_current_builder()._emit(Op.ACCEL_INSTANCE_TRANSFORM, Float4x4, [accel, instance_id])


def accel_instance_user_id(accel: Value, instance_id: Value) -> InstructionValue:
    """Get the user-defined ID of an instance."""
    from ..types import UInt
    return get_current_builder()._emit(Op.ACCEL_INSTANCE_USER_ID, UInt, [accel, instance_id])


def accel_instance_visibility_mask(accel: Value, instance_id: Value) -> InstructionValue:
    """Get the visibility mask of an instance."""
    from ..types import UInt
    return get_current_builder()._emit(Op.ACCEL_INSTANCE_VISIBILITY_MASK, UInt, [accel, instance_id])


@dsl_callable
def make_ray(origin: 'Float3', direction: 'Float3', t_min: 'Float', t_max: 'Float') -> Ray:
    """Construct a ray."""
    return Ray(origin, t_min, direction, t_max)
