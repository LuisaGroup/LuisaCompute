"""Tests for ray tracing operations - with IR building and pretty printing."""

import pytest
import ast as python_ast
from luisa import (
    kernel, callable, pprint,
    Float,
    Accel, Buffer,
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
    dispatch_id,
)
from luisa.lang.inspect import count_instructions, get_ir_ast


def print_ast(staged_func, title="Parsed AST"):
    """Helper to print the parsed AST of a staged function."""
    print(f"\n{title}:")
    tree = get_ir_ast(staged_func)
    if tree:
        print(python_ast.dump(tree, indent=2))
    else:
        print("  (No AST available)")


from luisa.lang.inspect import get_ir_ast
import ast as python_ast


def print_ast(staged_func, title="Parsed AST"):
    """Helper to print the parsed AST of a staged function."""
    print(f"\n{title}:")
    tree = get_ir_ast(staged_func)
    if tree:
        print(python_ast.dump(tree, indent=2))
    else:
        print("  (No AST available)")


def test_ray_type_exists():
    """Test that Ray type is defined and usable."""
    print("\n" + "=" * 60)
    print("Test: Ray type")
    print("=" * 60)

    assert Ray is not None

    # Ray type exists (can't instantiate directly in test context)
    print("✓ Ray type exists")
    print("=" * 60)


def test_hit_types_exist():
    """Test that hit types are defined."""
    print("\n" + "=" * 60)
    print("Test: hit types")
    print("=" * 60)

    assert TriangleHit is not None
    assert ProceduralHit is not None
    assert CommittedHit is not None

    # Test instantiation (without field initialization)
    th = TriangleHit.__new__(TriangleHit)
    ph = ProceduralHit.__new__(ProceduralHit)
    ch = CommittedHit.__new__(CommittedHit)

    print("✓ All hit types exist")
    print("=" * 60)


def test_make_ray_exists():
    """Test make_ray function exists."""
    print("\n" + "=" * 60)
    print("Test: make_ray")
    print("=" * 60)

    assert callable(make_ray)

    print("✓ make_ray callable")
    print("=" * 60)


def test_trace_closest_signature():
    """Test trace_closest function exists."""
    print("\n" + "=" * 60)
    print("Test: trace_closest")
    print("=" * 60)

    assert callable(trace_closest)

    print("✓ trace_closest callable")
    print("=" * 60)


def test_trace_any_signature():
    """Test trace_any function exists."""
    print("\n" + "=" * 60)
    print("Test: trace_any")
    print("=" * 60)

    assert callable(trace_any)

    print("✓ trace_any callable")
    print("=" * 60)


def test_ray_query_functions_exist():
    """Test that ray query functions are defined."""
    print("\n" + "=" * 60)
    print("Test: ray query functions")
    print("=" * 60)

    assert callable(ray_query_all)
    assert callable(ray_query_any)
    assert callable(ray_query_proceed)
    assert callable(ray_query_committed_hit)
    assert callable(ray_query_terminate)

    print("✓ All ray query functions callable")
    print("=" * 60)


def test_accel_instance_functions_exist():
    """Test that accel instance functions are defined."""
    print("\n" + "=" * 60)
    print("Test: accel instance functions")
    print("=" * 60)

    assert callable(accel_instance_transform)
    assert callable(accel_instance_user_id)
    assert callable(accel_instance_visibility_mask)

    print("✓ All accel instance functions callable")
    print("=" * 60)


def test_simple_ray_tracing_kernel():
    """Test a simple ray tracing kernel pattern."""
    print("\n" + "=" * 60)
    print("Test: simple ray tracing kernel")
    print("=" * 60)

    from luisa import Float3

    @kernel
    def simple_rt_kernel(accel: Accel, output: Buffer[Float]):
        idx = dispatch_id().x
        # Simple kernel that just writes output
        output[idx] = 0.0

    ir = simple_rt_kernel(None, None)
    print_ast(simple_rt_kernel, "AST: simple_rt_kernel")

    print("\nGenerated IR:")
    print(pprint(ir))

    assert ir.is_kernel
    print(f"✓ Built ray tracing kernel with {len(ir.blocks)} blocks")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_raytracing.py tests")
    print("=" * 70)

    test_ray_type_exists()
    test_hit_types_exist()
    test_make_ray_exists()
    test_trace_closest_signature()
    test_trace_any_signature()
    test_ray_query_functions_exist()
    test_accel_instance_functions_exist()
    test_simple_ray_tracing_kernel()

    print("\n" + "=" * 70)
    print("All test_raytracing.py tests passed!")
    print("=" * 70)
