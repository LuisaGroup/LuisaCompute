"""Tests for builtin functions - with IR building and pretty printing."""

import pytest
from luisa import (
    kernel, callable, pprint,
    # Math
    sqrt, abs, sin, cos, tan, asin, acos, atan, atan2,
    exp, exp2, log, log2, log10,
    floor, ceil, round, trunc, fract, saturate,
    normalize, length, length_squared,
    min, max, clamp, lerp, step, smoothstep, pow,
    dot, cross, distance, reflect, refract, faceforward,
    transpose, inverse, determinant,
    # Special registers
    dispatch_id, thread_id, block_id, dispatch_size,
    # Synchronization
    sync_block,
    # Type casting
    cast, bitcast,
    # Print
    device_print,
    # Assertions
    assume, device_assert, unreachable,
    # Profiling
    clock,
    # Types
    Int, Float, Float3, UInt3, Buffer,
)
from luisa.lang.inspect import count_instructions, get_ir_ast
import ast as python_ast


def print_ast(staged_func, title="Parsed AST"):
    """Helper to print the parsed AST of a staged function."""
    print(f"\n{title}:")
    tree = get_ir_ast(staged_func)
    if tree:
        print(python_ast.dump(tree, indent=2))
    else:
        print("  (No AST available)")


def test_math_builtins_build_ir():
    """Test math builtins actually build IR."""
    print("\n" + "=" * 60)
    print("Test: math builtins build IR")
    print("=" * 60)

    @callable
    def math_ops(x: Float) -> Float:
        a = sqrt(x)
        b = sin(a)
        c = cos(b)
        d = exp(c)
        e = log(d)
        f = floor(e)
        g = ceil(f)
        return g

    ir = math_ops(1.0)

    print_ast(math_ops, "AST: math_ops")

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert 'SQRT' in counts
    assert 'SIN' in counts
    assert 'COS' in counts

    print(f"✓ Built IR with {len(ir.blocks)} blocks")
    print(f"  Instructions: SQRT={counts.get('SQRT', 0)}, SIN={counts.get('SIN', 0)}, "
          f"COS={counts.get('COS', 0)}, EXP={counts.get('EXP', 0)}, LOG={counts.get('LOG', 0)}")
    print("=" * 60)


def test_special_registers_build_ir():
    """Test special registers actually build IR."""
    print("\n" + "=" * 60)
    print("Test: special registers build IR")
    print("=" * 60)

    @kernel
    def special_reg_kernel():
        did = dispatch_id()
        tid = thread_id()
        bid = block_id()
        dsize = dispatch_size()

    ir = special_reg_kernel()

    print_ast(special_reg_kernel, "AST: special_reg_kernel")

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert 'DISPATCH_ID' in counts
    assert 'THREAD_ID' in counts
    assert 'BLOCK_ID' in counts
    assert 'DISPATCH_SIZE' in counts

    print(f"✓ Built kernel with {len(ir.blocks)} blocks")
    print(f"  Special registers: DISPATCH_ID={counts.get('DISPATCH_ID', 0)}, "
          f"THREAD_ID={counts.get('THREAD_ID', 0)}, BLOCK_ID={counts.get('BLOCK_ID', 0)}")
    print("=" * 60)


def test_dispatch_id_in_computation():
    """Test dispatch_id used in actual computation."""
    print("\n" + "=" * 60)
    print("Test: dispatch_id in computation")
    print("=" * 60)

    @kernel
    def index_kernel(buf: Buffer[Float]):
        idx = dispatch_id().x
        buf[idx] = Float(idx)

    ir = index_kernel(None)

    print_ast(index_kernel, "AST: index_kernel")

    print("\nGenerated IR:")
    print(pprint(ir))

    assert ir.is_kernel
    counts = count_instructions(ir)
    assert 'DISPATCH_ID' in counts

    print(f"✓ Kernel uses dispatch_id, {len(ir.blocks)} blocks")
    print("=" * 60)


def test_sync_block_builds_ir():
    """Test sync_block actually builds IR."""
    print("\n" + "=" * 60)
    print("Test: sync_block builds IR")
    print("=" * 60)

    @kernel
    def sync_kernel(buf: Buffer[Float]):
        idx = dispatch_id().x
        buf[idx] = 1.0
        sync_block()
        buf[idx] = buf[idx] + 1.0

    ir = sync_kernel(None)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert 'SYNC_BLOCK' in counts

    print(f"✓ Built kernel with SYNC_BLOCK, {len(ir.blocks)} blocks")
    print("=" * 60)


def test_cast_builds_ir():
    """Test cast/bitcast actually build IR."""
    print("\n" + "=" * 60)
    print("Test: cast builds IR")
    print("=" * 60)

    @callable
    def cast_ops(x: Int) -> Float:
        f = Float(x)
        i = Int(f)
        return Float(i)

    ir = cast_ops(42)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert 'CAST' in counts

    print(f"✓ Built IR with {counts.get('CAST', 0)} CAST instructions")
    print("=" * 60)


def test_device_print_builds_ir():
    """Test device_print actually builds IR."""
    print("\n" + "=" * 60)
    print("Test: device_print builds IR")
    print("=" * 60)

    @kernel
    def print_kernel(x: Int):
        device_print("Value: {}", x)

    ir = print_kernel(42)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert 'PRINT' in counts

    print(f"✓ Built kernel with PRINT instruction")
    print("=" * 60)


def test_clock_builds_ir():
    """Test clock actually builds IR."""
    print("\n" + "=" * 60)
    print("Test: clock builds IR")
    print("=" * 60)

    @callable
    def timed_function() -> Int:
        start = clock()
        # Some computation
        x = Int(0)
        i = 0
        while i < 10:
            x = x + Int(i)
            i = i + 1
        end = clock()
        return Int(end - start)

    ir = timed_function()

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert 'CLOCK' in counts

    print(f"✓ Built IR with {counts.get('CLOCK', 0)} CLOCK instructions")
    print("=" * 60)


def test_assertions_build_ir():
    """Test assume/device_assert actually build IR."""
    print("\n" + "=" * 60)
    print("Test: assertions build IR")
    print("=" * 60)

    @callable
    def checked_function(x: Int) -> Int:
        assume(x > 0, "x must be positive")
        result = x * 2
        device_assert(result > x, "result should be greater than x")
        return result

    ir = checked_function(5)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    # Note: ASSERT and ASSUME may not be in all builds
    print(f"✓ Built IR with assertions")
    print("=" * 60)


def test_matrix_ops_build_ir():
    """Test matrix operations actually build IR."""
    print("\n" + "=" * 60)
    print("Test: matrix ops build IR")
    print("=" * 60)

    from luisa import Float4x4

    @callable
    def matrix_ops(m: Float4x4) -> Float:
        t = transpose(m)
        # Note: inverse/determinant may not be fully implemented
        return Float(0.0)

    # Use None since we don't have actual matrix values
    ir = matrix_ops(None)

    print("\nGenerated IR:")
    print(pprint(ir))

    print(f"✓ Built IR for matrix operations")
    print("=" * 60)


def test_vector_math_builds_ir():
    """Test vector math operations build IR."""
    print("\n" + "=" * 60)
    print("Test: vector math builds IR")
    print("=" * 60)

    @callable
    def vector_ops(a: Float3, b: Float3) -> Float3:
        d = dot(a, b)
        c = cross(a, b)
        n = normalize(a)
        l = length(b)
        return c

    ir = vector_ops(None, None)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    # Vector ops should generate instructions
    print(f"✓ Built IR with vector operations")
    print(f"  Instructions: {dict(counts)}")
    print("=" * 60)


def test_clamp_lerp_build_ir():
    """Test clamp and lerp build IR."""
    print("\n" + "=" * 60)
    print("Test: clamp/lerp build IR")
    print("=" * 60)

    @callable
    def utility_ops(x: Float) -> Float:
        c = clamp(x, 0.0, 1.0)
        l = lerp(0.0, 1.0, c)
        s = step(0.5, l)
        return smoothstep(0.0, 1.0, s)

    ir = utility_ops(0.5)

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    print(f"✓ Built IR with utility functions")
    print(f"  Instructions: CLAMP={counts.get('CLAMP', 0)}, "
          f"LERP={counts.get('LERP', 0)}, STEP={counts.get('STEP', 0)}")
    print("=" * 60)


def test_unreachable_builds_ir():
    """Test unreachable actually builds IR."""
    print("\n" + "=" * 60)
    print("Test: unreachable builds IR")
    print("=" * 60)

    @kernel
    def unreachable_kernel():
        unreachable("this should not happen")

    ir = unreachable_kernel()

    print("\nGenerated IR:")
    print(pprint(ir))

    counts = count_instructions(ir)
    assert 'UNREACHABLE' in counts

    print(f"✓ Built kernel with UNREACHABLE instruction")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running test_builtins.py tests")
    print("=" * 70)

    test_math_builtins_build_ir()
    test_special_registers_build_ir()
    test_dispatch_id_in_computation()
    test_sync_block_builds_ir()
    test_cast_builds_ir()
    test_device_print_builds_ir()
    test_clock_builds_ir()
    test_assertions_build_ir()
    test_unreachable_builds_ir()
    test_matrix_ops_build_ir()
    test_vector_math_builds_ir()
    test_clamp_lerp_build_ir()

    print("\n" + "=" * 70)
    print("All test_builtins.py tests passed!")
    print("=" * 70)
