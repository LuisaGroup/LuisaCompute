# AI Agent Guide: LuisaCompute Python DSL v2

This document provides a condensed technical map for AI agents to understand and modify the LuisaCompute Python DSL v2.

## 🚀 Core Architecture: Multistage Programming

The DSL uses a four-stage pipeline to transform Python into optimized GPU IR:

1.  **Parsing (Decoration Time)**: `@kernel` or `@callable` extracts the AST via `inspect.getsource()`.
2.  **Transformation (Rewrite Time)**: `ASTRewriter` replaces Python ops with `__luisa_rt` calls and injects template parameters.
3.  **Generation (Execution Time)**: The rewritten "Builder Function" runs. DSL operations build a Structured IR tree, while host-side Python (e.g., `static_range`) is expanded.
4.  **Lowering**: The Structured IR is sent to the LuisaCompute backend.

## 📁 Key File Map

| Path | Purpose |
| :--- | :--- |
| `luisa/lang/jit.py` | Implementation of `@kernel`/`@callable`, `TemplatedFunction`, and `StagedFunction`. |
| `luisa/lang/ops.py` | The `__luisa_rt` runtime router. Handles host/device dispatch and constant folding. |
| `luisa/lang/types.py` | DSL type system (Scalars, Vectors, Matrices, Buffers, Structs). |
| `luisa/transform/rewriter.py` | `ast.NodeTransformer` that turns Python syntax into IR builder calls. |
| `luisa/transform/builder.py` | Fluent API for constructing IR nodes and managing global builder context. |
| `luisa/transform/ir.py` | Definition of the Structured IR nodes (`If`, `Loop`, `Switch`, `Call`, etc.). |
| `luisa/transform/inspect.py` | Function parsing and analysis utilities. |
| `luisa/lang/router.py` | Host/device routing decisions and constant folding utilities. |

## 🧬 Template System (Critical)

Templates use **AST Injection** instead of string templates.
- **Explicit**: `func[Int]` - specialized via `__getitem__`.
- **Implicit**: Unannotated args are treated as `__impl_<name>` template params.
- **Partial**: Supports chaining specializations (e.g., `func[Int][Float]`).
- **Implementation**: Look at `TemplatedFunction._inject_template_params` in `jit.py`. It prepends `T = __luisa_spec.get('T')` to the rewritten AST.

## 🔍 The Linecache Strategy for Nested Functions

**Problem**: `inspect.getsourcelines()` fails for functions defined inside dynamically executed code (via `exec()`).

**Solution**: Populate Python's `linecache` with the rewritten source before execution:

```python
# In TemplatedFunction._do_compile():
rewritten_source = ast.unparse(module)

# Populate linecache so inspect.getsourcelines() works
linecache.cache[filename] = (
    len(rewritten_source),
    None,  # mtime
    rewritten_source.splitlines(keepends=True),
    filename
)

# Now exec() - nested functions can be inspected!
exec(compile(rewritten_source, filename, 'exec'), namespace)
```

This eliminates the need for special `source=` parameter handling in nested staged functions.

## 🛠 Debugging & Verification

### Environment Variables
- **AST Dumps**: `LUISA_DUMP_REWRITTEN_AST=1` - See how Python is transformed
- **IR Dumps**: `LUISA_DUMP_IR=1` - See the generated IR

### IR Inspection
```python
from luisa import pprint

# For kernels/callables
print(pprint(my_kernel.ir, recursive=True, show_location=False))
```

### Testing
- Run all tests: `python -m pytest python/tests/` (from repo root)
- JIT tests: `python -m pytest python/tests/test_jit/`
- Transformation tests: `python -m pytest python/tests/test_transform/`
- Language tests: `python -m pytest python/tests/test_lang/`

### Test Patterns
```python
def test_my_feature(verify_ir):
    @callable
    def my_func(x: Float) -> Float:
        return x + 1.0
    
    expected = """
f32 my_func(f32 arg0) {
  f32 v0 = add(arg0, 1.0);
  return v0;
}
"""
    verify_ir(my_func, expected)
```

## ⚠️ Important Implementation Details

### 1. Variable Semantics
`ASTRewriter` converts almost all assignments into DSL variables (`alloca` + `store`) to support reassignment in loops. Only these remain Python constants:
- `static()` calls
- `Const[Type]()` calls  
- Collection literals (list, tuple, dict, set)

### 2. The Runtime Router (`__luisa_rt`)
All DSL operations go through `ops.py`:
- **Host execution**: If all args are constants, compute in Python
- **IR emission**: If any arg is a DSL value, emit IR instruction
- **Mixed**: Automatically promote Python values to DSL constants

### 3. Nested Functions
Nested `@callable`/`@kernel` functions work naturally thanks to the linecache strategy. The outer function rewrite preserves nested function definitions, and when they execute, `inspect.getsourcelines()` finds them in linecache.

### 4. Short-Circuiting
`and` and `or` are rewritten to `and_` and `or_` functions that take lambdas to implement lazy evaluation in the IR:
```python
# Python: a and b
# Rewritten: and_(lambda: a, lambda: b)
```

### 5. Structured IR
Unlike flat basic-block IR, this DSL preserves high-level control flow:
- `If` nodes have `true` and `false` blocks
- `Loop` nodes have a body block with explicit `break`/`continue`
- `Switch` nodes have case blocks

### 6. Builder Context
A global stack manages the current IR builder:
```python
with set_current_builder(builder):
    # All DSL operations append to this builder
    result = some_callable(x, y)
```

### 7. Constant Folding
The `@router` decorator enables automatic constant folding:
```python
# Computed at host time
x = sin(1.0) + cos(2.0)  # ConstantValue

# Emitted as IR
y = sin(x)  # DSL instruction if x is DSL value
```

## 📝 Common Patterns

### Adding New Syntax Support
1. **Rewriter**: Add `visit_X` method in `rewriter.py`
2. **Runtime**: Add handler in `ops.py` 
3. **Builder**: Add IR construction method in `builder.py` (if needed)
4. **Test**: Add test in `tests/test_jit/` or `tests/test_lang/`

### Adding New Built-in Functions
1. Add to `luisa/lang/builtins/` or `luisa/lang/builtins.py`
2. Decorate with `@router` for automatic host/device routing
3. For device-only functions, use `primitive` decorator

### Template Parameter Resolution
Template params are resolved in this order:
1. Explicit specialization: `func[Int]`
2. Partial specialization chain: `func[Int][Float]`
3. Implicit deduction from arguments
4. Runtime error if undeducible

## 🐛 Common Pitfalls

1. **Don't use `print()` in DSL code** - Use `device_print()` instead
2. **All DSL variables are SSA** - Use `store`/`load` for mutable variables
3. **Template params must be available at parse time** - Don't compute type names dynamically
4. **Nested functions need linecache** - Always populate linecache before exec
5. **Be careful with Python closures** - Captured vars are analyzed at parse time

## 🧪 Testing Best Practices

- Use `verify_ir` fixture to check generated IR matches expectations
- Use `print_ir` fixture to debug IR generation
- Test both host-side and device-side behavior
- Test with constants and DSL values
- Use `pytest.mark.xfail` for known limitations
