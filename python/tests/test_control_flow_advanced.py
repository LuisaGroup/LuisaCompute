"""Advanced control flow tests for the LuisaCompute Python DSL - with IR building and pretty printing."""

import pytest
import ast as python_ast
from luisa import kernel, callable, int32, float32, bool_, pprint
from luisa.lang.ir import IROp
from luisa.lang.inspect import find_operations, analyze_control_flow, count_instructions, get_ir_ast


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



def test_nested_if_statements():
    """Test nested if statements."""
    print("\n" + "="*60)
    print("Test: nested if statements")
    print("="*60)
    
    @callable
    def nested_if(x: int32, y: int32) -> int32:
        if x > 0:
            if y > 0:
                return x + y
            return x - y
        return 0
    
    ir = nested_if(1, 2)
    print_ast(nested_if, "AST: nested_if")
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    cf = analyze_control_flow(ir)
    assert cf['conditional_branches'] >= 2
    
    print(f"✓ Nested ifs: {len(ir.blocks)} blocks, {cf['conditional_branches']} conditionals")
    print("="*60)


def test_if_elif_else_chain():
    """Test if-elif-else chain (using nested ifs)."""
    print("\n" + "="*60)
    print("Test: if-elif-else chain")
    print("="*60)
    
    @callable
    def if_chain(x: int32) -> int32:
        if x < 0:
            return -1
        else:
            if x == 0:
                return 0
            else:
                return 1
    
    ir = if_chain(0)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    cf = analyze_control_flow(ir)
    assert cf['conditional_branches'] >= 2
    
    print(f"✓ If chain: {len(ir.blocks)} blocks, {cf['conditional_branches']} conditionals")
    print("="*60)


def test_while_loop_with_break():
    """Test while loop with break."""
    print("\n" + "="*60)
    print("Test: while loop with break")
    print("="*60)
    
    @callable
    def while_with_break(x: int32) -> int32:
        i = 0
        while i < 100:
            if i == x:
                break
            i = i + 1
        return i
    
    ir = while_with_break(10)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    cf = analyze_control_flow(ir)
    assert cf['has_loops']
    
    print(f"✓ While with break: {len(ir.blocks)} blocks, has_loops={cf['has_loops']}")
    print("="*60)


def test_while_loop_with_continue():
    """Test while loop with continue."""
    print("\n" + "="*60)
    print("Test: while loop with continue")
    print("="*60)
    
    @callable
    def while_with_continue(x: int32) -> int32:
        i = 0
        sum = 0
        while i < 10:
            i = i + 1
            if i == x:
                continue
            sum = sum + i
        return sum
    
    ir = while_with_continue(5)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    cf = analyze_control_flow(ir)
    assert cf['has_loops']
    
    print(f"✓ While with continue: {len(ir.blocks)} blocks")
    print("="*60)


def test_for_range_loop():
    """Test for-range loop."""
    print("\n" + "="*60)
    print("Test: for-range loop")
    print("="*60)
    
    @callable
    def for_range_sum(n: int32) -> int32:
        total = 0
        for i in range(n):
            total = total + i
        return total
    
    ir = for_range_sum(10)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    cf = analyze_control_flow(ir)
    assert cf['has_loops']
    
    print(f"✓ For-range: {len(ir.blocks)} blocks, has_loops={cf['has_loops']}")
    print("="*60)


def test_for_range_with_step():
    """Test for-range loop with step."""
    print("\n" + "="*60)
    print("Test: for-range with step")
    print("="*60)
    
    @callable
    def for_range_step(n: int32) -> int32:
        total = 0
        for i in range(0, n, 2):
            total = total + i
        return total
    
    ir = for_range_step(10)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    cf = analyze_control_flow(ir)
    assert cf['has_loops']
    
    print(f"✓ For-range with step: {len(ir.blocks)} blocks")
    print("="*60)


def test_early_return():
    """Test function with early return."""
    print("\n" + "="*60)
    print("Test: early return")
    print("="*60)
    
    @callable
    def early_return(x: int32) -> int32:
        if x < 0:
            return 0
        if x > 100:
            return 100
        return x
    
    ir = early_return(50)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    cf = analyze_control_flow(ir)
    assert cf['returns'] >= 1
    
    print(f"✓ Early return: {len(ir.blocks)} blocks, {cf['returns']} returns")
    print("="*60)


def test_complex_boolean_expression():
    """Test complex boolean expressions in conditions."""
    print("\n" + "="*60)
    print("Test: complex boolean expression")
    print("="*60)
    
    @callable
    def complex_bool(x: int32, y: int32) -> int32:
        if x > 0 and y > 0:
            return x + y
        return 0
    
    ir = complex_bool(1, 2)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    counts = count_instructions(ir)
    
    print(f"✓ Boolean expression: {len(ir.blocks)} blocks")
    print(f"  Instructions: {dict(counts)}")
    print("="*60)


def test_multiple_returns_in_branches():
    """Test function with multiple returns in different branches."""
    print("\n" + "="*60)
    print("Test: multiple returns in branches")
    print("="*60)
    
    @callable
    def multi_return(x: int32) -> int32:
        if x < 0:
            return -1
        elif x == 0:
            return 0
        else:
            return 1
    
    ir = multi_return(0)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    cf = analyze_control_flow(ir)
    assert cf['returns'] >= 1
    
    print(f"✓ Multiple returns: {len(ir.blocks)} blocks, {cf['returns']} returns")
    print("="*60)


def test_deeply_nested_control_flow():
    """Test deeply nested control flow."""
    print("\n" + "="*60)
    print("Test: deeply nested control flow")
    print("="*60)
    
    @callable
    def nested_deep(x: int32) -> int32:
        if x > 0:
            if x > 10:
                if x > 100:
                    return 1000
                return 100
            return 10
        return 0
    
    ir = nested_deep(50)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    assert len(ir.blocks) >= 4  # Should have multiple branches
    
    print(f"✓ Deep nesting: {len(ir.blocks)} blocks")
    print("="*60)


def test_loop_with_multiple_exits():
    """Test loop with multiple exit conditions."""
    print("\n" + "="*60)
    print("Test: loop with multiple exits")
    print("="*60)
    
    @callable
    def multi_exit_loop(x: int32) -> int32:
        i = 0
        while i < 100:
            if i == x:
                break
            if i > x * 2:
                break
            i = i + 1
        return i
    
    ir = multi_exit_loop(25)
    
    print("\nGenerated IR:")
    print(pprint(ir))
    
    cf = analyze_control_flow(ir)
    assert cf['has_loops']
    
    print(f"✓ Multi-exit loop: {len(ir.blocks)} blocks")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Running test_control_flow_advanced.py tests")
    print("="*70)
    
    test_nested_if_statements()
    test_if_elif_else_chain()
    test_while_loop_with_break()
    test_while_loop_with_continue()
    test_for_range_loop()
    test_for_range_with_step()
    test_early_return()
    test_complex_boolean_expression()
    test_multiple_returns_in_branches()
    test_deeply_nested_control_flow()
    test_loop_with_multiple_exits()
    
    print("\n" + "="*70)
    print("All test_control_flow_advanced.py tests passed!")
    print("="*70)
