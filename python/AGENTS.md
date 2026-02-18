# AI Agent Guide: LuisaCompute Python DSL v2

This document provides a condensed technical map for AI agents to understand and modify the LuisaCompute Python DSL v2.

## 🚀 Core Architecture: Multistage Programming

The DSL uses a four-stage pipeline to transform Python into optimized GPU IR:

1.  **Parsing (Decoration Time)**: `@kernel` or `@callable` extracts the AST via `inspect.getsourcel()`.
2.  **Transformation (Rewrite Time)**: `ASTRewriter` replaces Python ops with `__luisa_rt` calls and injects template parameters.
3.  **Generation (Execution Time)**: The rewritten "Builder Function" runs. DSL operations build a Structured IR tree, while host-side Python (e.g., `static_range`) is expanded.
4.  **Lowering**: The Structured IR is sent to the LuisaCompute backend.

## 📁 Directory Structure

```
python/
├── AGENTS.md              # This file - AI agent guide
├── DESIGN.md              # Detailed design document
├── README.md              # User-facing documentation
├── pyproject.toml         # Package configuration
├── luisa/                 # Main package
│   ├── __init__.py        # Public API exports
│   ├── version.py         # Version info
│   ├── printer.py         # IR pretty printer (LLVM-style)
│   ├── serialize.py       # IR serialization
│   ├── lang/              # Language implementation
│   │   ├── __init__.py    # Language exports
│   │   ├── jit.py         # @kernel, @callable, JIT compilation
│   │   ├── ops.py         # __luisa_rt runtime operations
│   │   ├── types.py       # Type system (Vector, Matrix, Buffer, etc.)
│   │   ├── router.py      # Host/device routing
│   │   ├── control_flow.py # Static control flow utilities
│   │   ├── inspect.py     # IR inspection utilities
│   │   └── builtins/      # Built-in functions
│   │       ├── __init__.py
│   │       ├── core.py    # Core builtins (dispatch_id, etc.)
│   │       ├── math.py    # Math functions (sin, cos, etc.)
│   │       ├── atomic.py  # Atomic operations
│   │       ├── warp.py    # Warp operations
│   │       ├── rtx.py     # Ray tracing builtins
│   │       └── resource.py # Buffer/Texture operations
│   └── transform/         # AST transformation and IR
│       ├── __init__.py
│       ├── rewriter.py    # AST -> IR builder calls
│       ├── builder.py     # IR builder API
│       ├── ir.py          # IR node definitions
│       ├── inspect.py     # AST analysis utilities
│       └── op.py          # Operation type definitions
└── tests/                 # Test suite
    ├── conftest.py        # Pytest fixtures (verify_ir, print_ir)
    ├── test_jit/          # JIT and staging tests
    ├── test_lang/         # Language feature tests
    ├── test_transform/    # AST transformation tests
    └── test_types/        # Type system tests
```

## 📁 Key File Map

| Path | Purpose |
| :--- | :--- |
| `luisa/lang/jit.py` | Implementation of `@kernel`/`@callable`, `TemplatedFunction`, and `StagedFunction`. |
| `luisa/lang/ops.py` | The `__luisa_rt` runtime router. Handles host/device dispatch and constant folding. |
| `luisa/lang/types.py` | DSL type system (Scalars, Vectors, Matrices, Buffers, Structs). |
| `luisa/lang/router.py` | `@router` decorator for automatic host/device routing. |
| `luisa/transform/rewriter.py` | `ast.NodeTransformer` that turns Python syntax into IR builder calls. |
| `luisa/transform/builder.py` | Fluent API for constructing IR nodes and managing global builder context. |
| `luisa/transform/ir.py` | Definition of the Structured IR nodes (`If`, `Loop`, `Switch`, `Call`, etc.). |
| `luisa/transform/inspect.py` | Function parsing and analysis utilities. |
| `luisa/printer.py` | LLVM-style IR pretty printer. |
| `tests/conftest.py` | Pytest fixtures including `verify_ir` and `print_ir`. |

## 🧬 Template System (Critical)

Templates use **AST Injection** instead of string templates.
- **Explicit**: `func[Int]` - specialized via `__getitem__`.
- **Implicit**: Unannotated args are treated as `__impl_<name>` template params.
- **Partial**: Supports chaining specializations (e.g., `func[Int][Float]`).
- **Implementation**: Look at `TemplatedFunction._inject_template_params` in `jit.py`. It prepends `T = __luisa_spec.get('T')` to the rewritten AST.

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `TemplatedFunction` | `jit.py` | Factory for creating specialized functions |
| `StagedFunction` | `jit.py` | Fully specialized function with concrete IR |
| `StagedFunctionDecorator` | `jit.py` | Wrapper for `@kernel`/`@callable` decorators |

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

### Testing Commands
```bash
# Run all tests
python -m pytest python/tests/

# Run specific test categories
python -m pytest python/tests/test_jit/      # JIT and staging
python -m pytest python/tests/test_lang/     # Language features
python -m pytest python/tests/test_transform/ # AST transformation
python -m pytest python/tests/test_types/    # Type system

# Run with verbose output
python -m pytest python/tests/ -v

# Run with IR debugging
LUISA_DUMP_IR=1 python -m pytest python/tests/test_jit/test_staged.py -v
```

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

### Fixtures (from conftest.py)

| Fixture | Purpose |
|---------|---------|
| `verify_ir` | Compare generated IR against expected string |
| `print_ir` | Print IR for debugging |
| `verify_execution` | Verify kernel compiles (placeholder for future execution tests) |

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

### 8. Type Promotion
Types are automatically promoted:
```python
Int(1) + Float(2.0)  # Promotes to Float
Vector(Float, 3) + Vector(Int, 3)  # Promotes element-wise
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

### Working with Types
```python
# Checking types
from luisa import is_vector_type, is_scalar_type, get_element_type

if is_vector_type(T):
    element = get_element_type(T)  # Get Float from Float3
```

## 🐛 Common Pitfalls

1. **Don't use `print()` in DSL code** - Use `device_print()` instead
2. **All DSL variables are SSA** - Use `store`/`load` for mutable variables
3. **Template params must be available at parse time** - Don't compute type names dynamically
4. **Nested functions need linecache** - Always populate linecache before exec
5. **Be careful with Python closures** - Captured vars are analyzed at parse time
6. **Ref[Type] arguments** - Automatically handled via `load`/`store` injection
7. **Static loops** - Use `static_range()` for compile-time unrolling

## 🧪 Testing Best Practices

- Use `verify_ir` fixture to check generated IR matches expectations
- Use `print_ir` fixture to debug IR generation
- Test both host-side and device-side behavior
- Test with constants and DSL values
- Use `pytest.mark.xfail` for known limitations
- Write tests that verify the IR structure, not just that it compiles

## 📊 Code Quality

- **Formatter**: `black` (line length 100)
- **Linter**: `ruff` (E, F, I, N, W rules)
- **Type Checker**: `mypy` (strict mode)
- **Test Runner**: `pytest`

### Running Code Quality Tools
```bash
cd python
black luisa/ tests/
ruff check luisa/ tests/
mypy luisa/
pytest tests/
```

## 🔗 Related Documents

- `DESIGN.md` - Detailed design document with examples
- `README.md` - User-facing documentation with quick start
- `pyproject.toml` - Package configuration and tool settings
