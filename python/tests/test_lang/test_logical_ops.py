"""Tests for logical operations and chained comparisons."""

import pytest
from luisa import (
    kernel, callable, pprint,
    Int, Float, Bool, Buffer
)
from luisa.lang.inspect import analyze_control_flow, count_instructions


def test_logical_and_short_circuit():
    """Test logical AND with short-circuiting."""
    @callable
    def logic_and(a: Bool, b: Bool) -> Bool:
        return a and b

    ir = logic_and(True, True)
    
    # Short-circuiting AND is implemented with an IF
    cf = analyze_control_flow(ir)
    assert cf['ifs'] == 1
    
    # Should have a LOAD/STORE because we use a temporary variable for short-circuiting
    counts = count_instructions(ir)
    assert 'ALLOCA' in counts
    assert 'STORE' in counts
    assert 'LOAD' in counts


def test_logical_or_short_circuit():
    """Test logical OR with short-circuiting."""
    @callable
    def logic_or(a: Bool, b: Bool) -> Bool:
        return a or b

    ir = logic_or(True, True)
    
    # Short-circuiting OR is implemented with an IF
    cf = analyze_control_flow(ir)
    assert cf['ifs'] == 1


def test_chained_comparisons():
    """Test chained comparisons like x < a < y."""
    @callable
    def chain_comp(x: Int, a: Int, y: Int) -> Bool:
        return x < a < y

    ir = chain_comp(0, 5, 10)
    
    # x < a < y is rewritten to (x < a) and (a < y)
    # With short-circuiting, this should have an IF
    cf = analyze_control_flow(ir)
    assert cf['ifs'] == 1
    
    counts = count_instructions(ir)
    assert counts.get('LT', 0) >= 2


def test_complex_logic():
    """Test complex logical expressions."""
    @callable
    def complex_logic(a: Bool, b: Bool, c: Bool) -> Bool:
        return (a and b) or (not a and c)

    ir = complex_logic(True, False, True)
    
    # (a and b) -> 1 IF
    # (not a and c) -> 1 IF
    # (...) or (...) -> 1 IF
    # Total 3 IFs for short-circuiting
    cf = analyze_control_flow(ir)
    assert cf['ifs'] >= 3


def test_logic_with_side_effects():
    """Test logical ops where the RHS has 'side effects' (buffer write)."""
    # In DSL, everything is expressions, but we can have nested callables
    
    @callable
    def effect(buf: Buffer[Int], idx: Int) -> Bool:
        buf[idx] = 1
        return True

    @kernel
    def logic_kernel(a: Bool, buf: Buffer[Int]):
        if a and effect(buf, 0):
            pass

    ir = logic_kernel(True, None)
    
    # The CALL to 'effect' should only happen if 'a' is true
    # Pretty print to visually verify structure if needed
    # print(pprint(ir, recursive=True))
    
    cf = analyze_control_flow(ir)
    # 1 for 'if a and effect', 1 for short-circuiting AND
    assert cf['ifs'] >= 2


def test_chained_comparison_mixed():
    """Test chained comparisons with different operators."""
    @callable
    def mixed_chain(x: Int, a: Int, y: Int) -> Bool:
        return x <= a < y != 10

    ir = mixed_chain(0, 5, 10)
    
    counts = count_instructions(ir)
    assert counts.get('LE', 0) >= 1
    assert counts.get('LT', 0) >= 1
    assert counts.get('NE', 0) >= 1
    
    cf = analyze_control_flow(ir)
    # (x <= a) and (a < y) and (y != 10) -> 2 IFs for short-circuiting
    assert cf['ifs'] >= 2
