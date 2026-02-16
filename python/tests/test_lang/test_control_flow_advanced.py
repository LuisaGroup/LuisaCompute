"""Advanced control flow tests for the LuisaCompute Python DSL - with IR building and pretty printing."""

from luisa import kernel, callable, Int, Float


def test_nested_if_statements(print_ir, verify_ir):
    """Test nested if statements."""
    @callable
    def nested_if(x: Int, y: Int) -> Int:
        if x > 0:
            if y > 0:
                return x + y
            return x - y
        return 0

    print_ir(nested_if, "nested_if")

    expected = """
i32 nested_if(i32 arg0, i32 arg1) {
  i1 v0 = gt(arg0, 0);
  if (v0) { 
    i1 v2 = gt(arg1, 0);
    if (v2) { 
      i32 v4 = add(arg0, arg1);
      return v4;
    } else {
      (empty)
    }
    i32 v6 = sub(arg0, arg1);
    return v6;
  } else {
    (empty)
  }
  return 0;
}
"""
    verify_ir(nested_if, expected)


def test_if_elif_else_chain(print_ir, verify_ir):
    """Test if-elif-else chain (using nested ifs)."""
    @callable
    def if_chain(x: Int) -> Int:
        if x < 0:
            return -1
        elif x == 0:
            return 0
        else:
            return 1

    print_ir(if_chain, "if_chain")

    expected = """
i32 if_chain(i32 arg0) {
  i1 v0 = lt(arg0, 0);
  if (v0) { 
    return -1;
  } else {
    i1 v3 = eq(arg0, 0);
    if (v3) { 
      return 0;
    } else {
      return 1;
    }
  }
}
"""
    verify_ir(if_chain, expected)


def test_while_loop_with_break(print_ir, verify_ir):
    """Test while loop with break."""
    @callable
    def while_with_break(x: Int) -> Int:
        i = Int(0)
        while i < 100:
            if i == x:
                break
            i = i + 1
        return i

    print_ir(while_with_break, "while_with_break")

    expected = """
i32 while_with_break(i32 arg0) {
  i32 vi = alloca();
  store(vi, 0);
  i32 v2 = load(vi);
  i1 v3 = lt(v2, 100);
  while (true) { 
    i1 v5 = logical_not(v3);
    if (v5) { 
      break;
    } else {
      (empty)
    }
    i32 v8 = load(vi);
    i1 v9 = eq(v8, arg0);
    if (v9) { 
      (empty)
    } else {
      (empty)
    }
  }
  i32 v11 = load(vi);
  return v11;
}
"""
    verify_ir(while_with_break, expected)


def test_while_loop_with_continue(print_ir, verify_ir):
    """Test while loop with continue."""
    @callable
    def while_with_continue(x: Int) -> Int:
        i = Int(0)
        s = Int(0)
        while i < 10:
            i = i + 1
            if i == x:
                continue
            s = s + i
        return s

    print_ir(while_with_continue, "while_with_continue")

    expected = """
i32 while_with_continue(i32 arg0) {
  i32 vi = alloca();
  store(vi, 0);
  i32 vs = alloca();
  store(vs, 0);
  i32 v4 = load(vi);
  i1 v5 = lt(v4, 10);
  while (true) { 
    i1 v7 = logical_not(v5);
    if (v7) { 
      break;
    } else {
      (empty)
    }
    i32 v10 = load(vi);
    i32 v11 = add(v10, 1);
    store(vi, v11);
    i32 v13 = load(vi);
    i1 v14 = eq(v13, arg0);
    if (v14) { 
      (empty)
    } else {
      (empty)
    }
  }
  i32 v16 = load(vs);
  return v16;
}
"""
    verify_ir(while_with_continue, expected)


def test_for_range_loop(print_ir, verify_ir):
    """Test for-range loop."""
    @callable
    def for_range_sum(n: Int) -> Int:
        total = Int(0)
        for i in range(n):
            total = total + i
        return total

    print_ir(for_range_sum, "for_range_sum")

    expected = """
i32 for_range_sum(i32 arg0) {
  i32 vtotal = alloca();
  store(vtotal, 0);
  i32 vi = alloca();
  store(vi, 0);
  while (true) { 
    i32 v5 = load(vi);
    i1 v6 = lt(v5, arg0);
    i1 v7 = logical_not(v6);
    if (v7) { 
      break;
    } else {
      (empty)
    }
    i32 v10 = load(vtotal);
    i32 v11 = add(v10, v5);
    store(vtotal, v11);
    i32 v13 = add(v5, 1);
    store(vi, v13);
  }
  i32 v15 = load(vtotal);
  return v15;
}
"""
    verify_ir(for_range_sum, expected)


def test_for_range_with_step(print_ir, verify_ir):
    """Test for-range loop with step."""
    @callable
    def for_range_step(n: Int) -> Int:
        total = Int(0)
        for i in range(0, n, 2):
            total = total + i
        return total

    print_ir(for_range_step, "for_range_step")

    expected = """
i32 for_range_step(i32 arg0) {
  i32 vtotal = alloca();
  store(vtotal, 0);
  i32 vi = alloca();
  store(vi, 0);
  while (true) { 
    i32 v5 = load(vi);
    i1 v6 = lt(v5, arg0);
    i1 v7 = logical_not(v6);
    if (v7) { 
      break;
    } else {
      (empty)
    }
    i32 v10 = load(vtotal);
    i32 v11 = add(v10, v5);
    store(vtotal, v11);
    i32 v13 = add(v5, 2);
    store(vi, v13);
  }
  i32 v15 = load(vtotal);
  return v15;
}
"""
    verify_ir(for_range_step, expected)


def test_early_return(print_ir, verify_ir):
    """Test function with early return."""
    @callable
    def early_return(x: Int) -> Int:
        if x < 0:
            return 0
        if x > 100:
            return 100
        return x

    print_ir(early_return, "early_return")

    expected = """
i32 early_return(i32 arg0) {
  i1 v0 = lt(arg0, 0);
  if (v0) { 
    return 0;
  } else {
    (empty)
  }
  i1 v3 = gt(arg0, 100);
  if (v3) { 
    return 100;
  } else {
    (empty)
  }
  return arg0;
}
"""
    verify_ir(early_return, expected)


def test_complex_boolean_expression(print_ir, verify_ir):
    """Test complex boolean expressions in conditions."""
    @callable
    def complex_bool(x: Int, y: Int) -> Int:
        if x > 0 and y > 0:
            return x + y
        return 0

    print_ir(complex_bool, "complex_bool")

    expected = """
i32 complex_bool(i32 arg0, i32 arg1) {
  i1 v0 = gt(arg0, 0);
  i1 v1 = alloca();
  store(v1, v0);
  if (v0) { 
    i1 v4 = gt(arg1, 0);
    store(v1, v4);
  } else {
    (empty)
  }
  i1 v6 = load(v1);
  if (v6) { 
    i32 v8 = add(arg0, arg1);
    return v8;
  } else {
    (empty)
  }
  return 0;
}
"""
    verify_ir(complex_bool, expected)


def test_multiple_returns_in_branches(print_ir, verify_ir):
    """Test function with multiple returns in different branches."""
    @callable
    def multi_return(x: Int) -> Int:
        if x < 0:
            return -1
        elif x == 0:
            return 0
        else:
            return 1

    print_ir(multi_return, "multi_return")

    expected = """
i32 multi_return(i32 arg0) {
  i1 v0 = lt(arg0, 0);
  if (v0) { 
    return -1;
  } else {
    i1 v3 = eq(arg0, 0);
    if (v3) { 
      return 0;
    } else {
      return 1;
    }
  }
}
"""
    verify_ir(multi_return, expected)


def test_deeply_nested_control_flow(print_ir, verify_ir):
    """Test deeply nested control flow."""
    @callable
    def nested_deep(x: Int) -> Int:
        if x > 0:
            if x > 10:
                if x > 100:
                    return 1000
                return 100
            return 10
        return 0

    print_ir(nested_deep, "nested_deep")

    expected = """
i32 nested_deep(i32 arg0) {
  i1 v0 = gt(arg0, 0);
  if (v0) { 
    i1 v2 = gt(arg0, 10);
    if (v2) { 
      i1 v4 = gt(arg0, 100);
      if (v4) { 
        return 1000;
      } else {
        (empty)
      }
      return 100;
    } else {
      (empty)
    }
    return 10;
  } else {
    (empty)
  }
  return 0;
}
"""
    verify_ir(nested_deep, expected)


def test_loop_with_multiple_exits(print_ir, verify_ir):
    """Test loop with multiple exit conditions."""
    @callable
    def multi_exit_loop(x: Int) -> Int:
        i = Int(0)
        while i < 100:
            if i == x:
                break
            if i > x * 2:
                break
            i = i + 1
        return i

    print_ir(multi_exit_loop, "multi_exit_loop")

    expected = """
i32 multi_exit_loop(i32 arg0) {
  i32 vi = alloca();
  store(vi, 0);
  i32 v2 = load(vi);
  i1 v3 = lt(v2, 100);
  while (true) { 
    i1 v5 = logical_not(v3);
    if (v5) { 
      break;
    } else {
      (empty)
    }
    i32 v8 = load(vi);
    i1 v9 = eq(v8, arg0);
    if (v9) { 
      (empty)
    } else {
      (empty)
    }
  }
  i32 v11 = load(vi);
  return v11;
}
"""
    verify_ir(multi_exit_loop, expected)


def test_python_match_to_switch(print_ir, verify_ir):
    """Test that Python's match statement is translated to IR SWITCH."""
    @callable
    def match_test(tag: Int) -> Int:
        res = Int(0)
        match tag:
            case 0:
                res = Int(10)
            case 1:
                res = Int(20)
            case 2:
                res = Int(30)
            case _:
                res = Int(-1)
        return res

    print_ir(match_test, "match_test")

    expected = """
i32 match_test(i32 arg0) {
  i32 vres = alloca();
  store(vres, 0);
  switch (arg0) { 
    case 0: {
      store(vres, 10);
    }
    case 1: {
      store(vres, 20);
    }
    case 2: {
      store(vres, 30);
    }
    default: {
      store(vres, -1);
    }
  }
  i32 v7 = load(vres);
  return v7;
}
"""
    verify_ir(match_test, expected)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
