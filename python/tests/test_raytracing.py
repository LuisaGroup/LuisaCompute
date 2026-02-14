"""Tests for ray tracing operations."""

import pytest
from luisa import (
    float32, float3,
    Accel,
    # Ray types
    Ray, TriangleHit, ProceduralHit, CommittedHit,
    # Tracing
    trace_closest, trace_any, ray_query_all, ray_query_any,
    # Ray query operations
    ray_query_world_space_ray, ray_query_proceed,
    ray_query_committed_hit, ray_query_candidate_triangle_hit, ray_query_candidate_procedural_hit,
    ray_query_commit_triangle, ray_query_commit_procedural, ray_query_terminate,
    # Accel operations
    accel_instance_transform, accel_instance_user_id, accel_instance_visibility_mask,
    make_ray,
)


def test_ray_type_exists():
    """Test that Ray type is defined."""
    print("Testing Ray type...")
    
    assert Ray is not None
    
    print("  ✓ Ray type OK")


def test_hit_types_exist():
    """Test that hit types are defined."""
    print("Testing hit types...")
    
    assert TriangleHit is not None
    assert ProceduralHit is not None
    assert CommittedHit is not None
    
    print("  ✓ Hit types OK")


def test_trace_functions_exist():
    """Test that trace functions are defined."""
    print("Testing trace functions...")
    
    assert callable(trace_closest)
    assert callable(trace_any)
    
    print("  ✓ Trace functions OK")


def test_ray_query_functions_exist():
    """Test that ray query functions are defined."""
    print("Testing ray query functions...")
    
    assert callable(ray_query_all)
    assert callable(ray_query_any)
    assert callable(ray_query_proceed)
    assert callable(ray_query_committed_hit)
    assert callable(ray_query_terminate)
    
    print("  ✓ Ray query functions OK")


def test_make_ray_exists():
    """Test that make_ray function is defined."""
    print("Testing make_ray...")
    
    assert callable(make_ray)
    
    print("  ✓ make_ray OK")


def test_accel_instance_functions_exist():
    """Test that accel instance functions are defined."""
    print("Testing accel instance functions...")
    
    assert callable(accel_instance_transform)
    assert callable(accel_instance_user_id)
    assert callable(accel_instance_visibility_mask)
    
    print("  ✓ Accel instance functions OK")
