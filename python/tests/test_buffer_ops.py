"""Tests for buffer operations - with IR building and pretty printing."""

import pytest
from luisa import kernel, callable, float32, int32, Buffer, dispatch_id, pprint
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


def test_buffer_write():
    """Test buffer write operation - builds and prints IR."""
    print("\n" + "="*60)
    print("Test: buffer_write")
    print("="*60)
    
    @callable
    def write_to_buffer(buf: Buffer[float32]) -> None:
        buf[0] = 1.0
    
    ir = write_to_buffer(0)
    
    print_ast(write_to_buffer, "AST: write_to_buffer")
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    # Should have one BUFFER_WRITE
    counts = count_instructions(ir)
    assert 'BUFFER_WRITE' in counts
    assert counts['BUFFER_WRITE'] == 1
    
    assert ir is not None
    assert len(ir.blocks) > 0
    print(f"✓ Generated {len(ir.blocks)} blocks, BUFFER_WRITE count: {counts['BUFFER_WRITE']}")
    print("="*60)


def test_buffer_read():
    """Test buffer read operation - builds and prints IR."""
    print("\n" + "="*60)
    print("Test: buffer_read")
    print("="*60)
    
    @callable
    def read_from_buffer(buf: Buffer[float32]) -> float32:
        return buf[0]
    
    ir = read_from_buffer(0)
    
    print_ast(read_from_buffer, "AST: read_from_buffer")
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    counts = count_instructions(ir)
    assert 'BUFFER_READ' in counts
    assert counts['BUFFER_READ'] == 1
    
    print(f"✓ Generated {len(ir.blocks)} blocks, BUFFER_READ count: {counts['BUFFER_READ']}")
    print("="*60)


def test_buffer_read_write():
    """Test buffer read and write in same function."""
    print("\n" + "="*60)
    print("Test: buffer_read_write")
    print("="*60)
    
    @callable
    def copy_buffer(src: Buffer[float32], dst: Buffer[float32]) -> None:
        dst[0] = src[0]
    
    ir = copy_buffer(0, 0)
    
    print_ast(copy_buffer, "AST: copy_buffer")
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    counts = count_instructions(ir)
    assert 'BUFFER_READ' in counts
    assert 'BUFFER_WRITE' in counts
    
    print(f"✓ BUFFER_READ: {counts['BUFFER_READ']}, BUFFER_WRITE: {counts['BUFFER_WRITE']}")
    print("="*60)


def test_saxpy_kernel():
    """Test SAXPY kernel pattern - Single-precision A*X Plus Y."""
    print("\n" + "="*60)
    print("Test: SAXPY kernel")
    print("="*60)
    
    @kernel
    def saxpy(result: Buffer[float32], a: float32, x: Buffer[float32], y: Buffer[float32]) -> None:
        idx = dispatch_id().x
        result[idx] = a * x[idx] + y[idx]
    
    ir = saxpy(0, 2.0, 0, 0)
    
    print_ast(saxpy, "AST: saxpy")
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    assert ir.is_kernel
    
    counts = count_instructions(ir)
    assert 'BUFFER_READ' in counts
    assert 'BUFFER_WRITE' in counts
    assert 'MUL' in counts
    assert 'ADD' in counts
    assert 'DISPATCH_ID' in counts
    
    print(f"✓ Kernel with {len(ir.blocks)} blocks")
    print(f"  Instructions: BUFFER_READ={counts.get('BUFFER_READ', 0)}, "
          f"BUFFER_WRITE={counts.get('BUFFER_WRITE', 0)}, "
          f"MUL={counts.get('MUL', 0)}, ADD={counts.get('ADD', 0)}")
    print("="*60)


def test_buffer_with_dynamic_index():
    """Test buffer access with dynamic index."""
    print("\n" + "="*60)
    print("Test: buffer dynamic index")
    print("="*60)
    
    @callable
    def dynamic_access(buf: Buffer[float32], idx: int32) -> float32:
        return buf[idx]
    
    ir = dynamic_access(0, 5)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    counts = count_instructions(ir)
    assert 'BUFFER_READ' in counts
    
    print(f"✓ Generated {len(ir.blocks)} blocks with BUFFER_READ")
    print("="*60)


def test_buffer_multiple_writes():
    """Test multiple buffer writes."""
    print("\n" + "="*60)
    print("Test: buffer multiple writes")
    print("="*60)
    
    @callable
    def fill_buffer(buf: Buffer[float32]) -> None:
        buf[0] = 0.0
        buf[1] = 1.0
        buf[2] = 2.0
    
    ir = fill_buffer(0)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    counts = count_instructions(ir)
    assert counts['BUFFER_WRITE'] == 3
    
    print(f"✓ Generated {len(ir.blocks)} blocks with {counts['BUFFER_WRITE']} BUFFER_WRITEs")
    print("="*60)


def test_buffer_2d_kernel():
    """Test 2D buffer access pattern."""
    print("\n" + "="*60)
    print("Test: 2D buffer kernel")
    print("="*60)
    
    @kernel
    def matrix_transpose(out: Buffer[float32], inp: Buffer[float32], width: int32, height: int32):
        x = dispatch_id().x
        y = dispatch_id().y
        if x < width and y < height:
            out[y * width + x] = inp[x * height + y]
    
    ir = matrix_transpose(None, None, 64, 64)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    assert ir.is_kernel
    assert len(ir.blocks) >= 2  # Should have condition block
    
    counts = count_instructions(ir)
    total = sum(counts.values())
    print(f"✓ 2D kernel with {len(ir.blocks)} blocks, {total} instructions")
    print(f"  DISPATCH_ID: {counts.get('DISPATCH_ID', 0)}")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running test_buffer_ops.py tests")
    print("="*70)
    
    test_buffer_write()
    test_buffer_read()
    test_buffer_read_write()
    test_saxpy_kernel()
    test_buffer_with_dynamic_index()
    test_buffer_multiple_writes()
    test_buffer_2d_kernel()
    
    print("\n" + "="*70)
    print("All test_buffer_ops.py tests passed!")
    print("="*70)
