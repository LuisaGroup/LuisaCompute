# LuisaCompute Python DSL v2 Design Document

## 🏗 Architecture Overview

The LuisaCompute Python DSL v2 is built on a **Multistage Programming** architecture. Instead of interpreting Python at runtime, we utilize a tiered compilation pipeline that transforms high-level Python syntax into an optimized, structured Intermediate Representation (IR).

### The Compilation Pipeline

1.  **Stage 1: Parsing (Decoration Time)**
    *   Triggered by `@kernel` or `@callable`.
    *   The function's AST is extracted via `inspect.getsource()`.
    *   Type annotations are parsed and stored as strings (to handle forward references and template params).
    *   Captured variables from the outer scope are analyzed.
    *   **Result**: A `TemplatedFunction` or `StagedFunction` is created with the parsed metadata.

2.  **Stage 2: Transformation (Rewrite Time)**
    *   Triggered when a `TemplatedFunction` is called or specialized with concrete types.
    *   The AST is passed through `ASTRewriter`.
    *   Every DSL-relevant operation (arithmetic, control flow, built-in calls) is replaced with a direct function call (e.g., `add(a, b)`, `load(x)`, `if_(cond, ...)`).
    *   For templated functions: template parameter assignments are injected at the function start.
    *   **Result**: A **Builder Function** that, when executed, will generate the equivalent Luisa IR.

3.  **Stage 3: Generation (StagedFunction Creation)**
    *   Triggered when a `TemplatedFunction` becomes fully specialized (all template params resolved).
    *   A `StagedFunction` is created with concrete argument types.
    *   The Builder Function is executed immediately to build the IR.
    *   **Host-side logic** (standard Python `if`, `for`, list comprehensions) is expanded normally by Python.
    *   **DSL operations** call into the `Builder` to record instructions into a **Structured IR Tree**.
    *   The resulting IR is stored in the `StagedFunction` and cached for subsequent calls.

4.  **Stage 4: Lowering (Backend)**
    *   The Structured IR is lowered to the LuisaCompute C++ AST or XIR.
    *   Structured nodes like `IF`, `LOOP`, and `SWITCH` are preserved, allowing the backend to perform higher-level optimizations before final machine code generation.

---

## 🎯 Template Specialization System

The DSL features a powerful C++-style template system that enables generic programming while maintaining zero-cost abstraction.

### Syntax

```python
from luisa import callable, kernel, Float, Int, Buffer

# Template callable with single parameter
@callable['T']
def identity(x: T) -> T:
    return x

# Template with multiple parameters
@callable['T', 'U']
def cast_and_scale(x: T, scale: U) -> U:
    return U(x) * scale

# Kernel template with buffer element type
@kernel['T']
def fill_buffer(buf: Buffer[T], value: T):
    buf[0] = value
```

### Specialization Modes

#### 1. Explicit Specialization
Use `[]` to explicitly provide type arguments:

```python
int_identity = identity[Int]        # Returns StagedFunction
float_identity = identity[Float]    # Different specialization
```

#### 2. Implicit Specialization
Call with typed arguments, types inferred automatically:

```python
# T inferred as Int from argument type
result = identity(Int(42))  # Creates/uses Int specialization
```

#### 3. Partial Specialization
Provide some template arguments, leave others for later:

```python
@callable['T', 'U']
def combine(a: T, b: U) -> T:
    return a + T(b)

partial = combine[Int]      # T=Int, U still polymorphic
full = partial[Float]       # Now T=Int, U=Float
```

### Multistage Template Resolution Strategy

The key insight is using **AST injection** instead of string manipulation:

```python
# Original template:
@callable['T']
def func(x: Buffer[T]) -> T:
    return x

# Rewritten AST becomes:
def __luisa_built_func(arg0):
    T = __luisa_spec.get("T")  # <-- Injected at function start
    set_location(...)
    return arg0
```

**Benefits:**
- Template params become local variables in function scope
- Annotations like `Buffer[T]` resolve naturally during function definition
- Nested functions capture template params via Python's closure mechanism
- Supports arbitrarily complex nested generics: `Buffer[Vector[T, 3]]`

### Implementation Classes

- **`TemplatedFunction`**: Factory for creating specializations, manages template params and partial specialization state
- **`StagedFunction`**: Fully specialized function with concrete types, holds cached IR
- **`KernelInvoke`**: Records kernel + arguments for device dispatch

### Implicit Template Parameters

Arguments without type annotations are automatically treated as **implicit template parameters**. They cannot be explicitly specialized but are always deduced from call arguments.

```python
# Pure implicit templates
@callable
def identity(x):  # x is implicit template param
    return x

# Mixed explicit and implicit
@callable['T']
def scale(a: T, b):  # T explicit, b implicit
    return a * T(b)

# Usage
@kernel
def test():
    result1 = identity(Int(42))      # Implicit param = Int
    result2 = scale(Float(2.0), 3)  # T=Float, implicit = Int
```

**Key differences from explicit templates:**
- No `[]` syntax for specialization: `identity[Int]` raises `TypeError`
- Always deduced from argument types at call site
- Named internally as `__impl_<arg_name>` to avoid collisions

---

## ⚡ Constant Folding & Routing

A core innovation in v2 is the **Host/Device Routing** system, facilitated by the `@router` decorator.

### The `@router` Mechanism
The router intelligently decides whether an operation should be evaluated immediately on the host (Constant Folding) or emitted as a device instruction.

*   **Rule**: If all arguments to a routed function are compile-time constants (Python primitives or `ConstantValue`), the host-side implementation is executed.
*   **Optimization**: This allows complex expressions like `sin(1.0) + cos(2.0)` to be folded into a single numeric literal in the final IR, reducing GPU execution overhead.

### Constant Value Arithmetic
The `ConstantValue` class overrides standard Python operators. This enables expressions involving both Python literals and DSL constants to be folded seamlessly:
```python
a = sin(1.0) # Evaluated on host -> ConstantValue
b = a * 2.0  # Also evaluated on host -> ConstantValue
```

---

## 🌳 Structured IR

Unlike traditional compiler IRs that rely on flat basic blocks and explicit jumps (`GOTO`), this DSL produces a **Structured IR**.

### Core Nodes
*   **`IF`**: Encapsulates a condition and two distinct sub-blocks (`true` and `false`).
*   **`LOOP`**: Encapsulates a body block. Control flow is managed via explicit `BREAK` and `CONTINUE` instructions.
*   **`SWITCH`**: Maps a value to multiple case blocks, including an optional `DEFAULT`.

### Type System
Types are represented using an LLVM-inspired naming convention for clarity during debugging and serialization:
*   **Scalars**: `i1` (bool), `i32` (int), `f32` (float).
*   **Vectors**: `<N x T>` (e.g., `<4 x f32>`).
*   **Matrices**: `[N x <N x T>]` (column-major arrays of vectors).
*   **Resources**: `buffer<T>`, `texture2d<T>`, `accel`.

### Type Promotion
Arithmetic operations automatically promote types following standard rules:
```python
Int(1) + Float(2.0)  # Promotes to Float
Vector(Float, 3) + Vector(Int, 3)  # Promotes element-wise
```

---

## 🔗 Variable Management

### DSL Variables vs. Host Constants
The `ASTRewriter` distinguishes between standard Python variables and DSL variables.

**All variable assignments create DSL variables** (using `alloca`, `load`, and `store`) to ensure correct reassignment semantics:
```python
@kernel
def example(buf: Buffer[Float]):
    a = 1.0           # Creates DSL variable (alloca + store)
    a = a + 1.0       # Properly reassigns (load + add + store)
    buf[0] = a
```

This ensures correct behavior for patterns like:
```python
@kernel
def sum_loop(buf: Buffer[Float]):
    a = 1.0
    b = 0.0
    for i in range(10):
        b += a        # Works correctly: b = b + a
        a += 1.0      # Works correctly: a = a + 1
    buf[0] = b        # Result: 55 (1+2+3+...+10)
```

**Exceptions** (remain as Python constants):
*   `static()` calls: `a = static(sin(1.0))` - Python constant for host-side computation
*   `Const[Type]()` calls: `c = Const[Float](1.5)` - DSL constant value
*   List/tuple/dict/set literals: `impls = [func1, func2]` - Python values for iteration

### Augmented Assignments
The DSL supports Python's augmented assignment operators (`+=`, `-=`, `*=`, `/=`, etc.) which are automatically rewritten into standard assignments with binary operations.

### Reference Arguments (`Ref[T]`)
References are implemented as a property of function arguments. When a variable is identified as a `Ref`, the rewriter automatically injects the necessary pointer-logic (`load`/`store`) to ensure that in-place modifications within a `@callable` are reflected in the caller's scope.

---

## 🔄 Control Flow

### Native Python Constructs
The DSL supports idiomatic Python control flow:

```python
@kernel
def example(buf: Buffer[Float]):
    # If-elif-else
    if x > 0:
        buf[0] = 1.0
    elif x < 0:
        buf[0] = -1.0
    else:
        buf[0] = 0.0
    
    # While loops with break/continue
    i = 0
    while i < 10:
        if i == 5:
            break
        i = i + 1
    
    # Range-based for loops
    for j in range(10):
        buf[j] = Float(j)
    
    # Unrolled loops (compile-time iteration)
    for k in static_range(4):
        buf[k] = Float(k)  # Unrolled 4 times
```

### Match/Switch Statements
Python's `match` statement is directly translated:

```python
@callable
def classify(tag: Int) -> Int:
    match tag:
        case 0: return 10
        case 1: return 20
        case _: return -1
```

---

## 🛠 Key Components

| Component | Responsibility |
|-----------|----------------|
| `jit.py` | Manages the staging process, template specialization, and JIT caching |
| `rewriter.py` | Implements `ast.NodeTransformer` for stage 2 transformation |
| `builder.py` | Fluent API for manual IR construction and global context management |
| `ir.py` | Defines data structures for the IR tree and instructions |
| `router.py` | Handles constant folding and host/device dispatch |
| `printer.py` | Generates LLVM-style human-readable IR representation |
| `types.py` | Type system: scalars, vectors, matrices, buffers, textures |

---

## 🔧 Implementation Tricks & Strategies

### 1. AST Rewriting Strategy
The `ASTRewriter` transforms Python syntax into IR builder calls:
- `a + b` → `add(a, b)` (direct function call)
- `x[i] = y` → `subscript_assign(x, i, y)`
- `if cond:` → `if_(cond, ...)`

This allows Python's execution engine to handle control flow while DSL operations build IR.

### 2. Builder Context Management
A global builder stack enables nested function calls to contribute to the same IR:
```python
with set_current_builder(builder):
    # All DSL operations append to this builder
    result = some_callable(x, y)
```

### 3. Type Inference
Argument types are detected via:
1. Explicit annotations (if provided)
2. Runtime type inspection (`value_to_type()`)
3. IR value type attributes (`arg.type`)

### 4. Template Param Injection
Instead of complex string parsing, template params are injected as local variables:
```python
# Before execution, inject:
T = __luisa_spec.get("T")  # Where __luisa_spec = {'T': Int}

# Now annotations resolve naturally:
def func(x: Buffer[T]) -> T:  # Buffer[T] becomes Buffer[Int]
```

### 5. Caching Strategy
- **TemplatedFunction._cache**: Maps `(arg_types,) -> StagedFunction`
- **Implicit specialization**: Same argument types reuse cached IR
- **No device needed**: IR generation happens purely in Python

### 6. Nested Function Support
Inner callables capture template params from outer scopes:
```python
@callable['T']
def outer(x: T):
    @callable
    def inner(y: T):  # Captures T from outer
        return y * 2
    return inner(x)
```

---

## 🔍 Nested Staged Functions and the Linecache Solution

### The Problem: Dynamic Function Definition

When a nested `@callable` or `@kernel` is defined inside a dynamically executed function, Python's `inspect.getsourcelines()` cannot retrieve its source code:

```python
@kernel
def my_kernel(buf: Buffer[Float]):
    @callable
    def inner(x: Float) -> Float:  # Defined during kernel execution
        return x + 1.0
    buf[0] = inner(1.0)
```

**Why it fails**: 
- `inspect.getsourcelines()` relies on the function having an associated source file
- Functions defined inside `exec()` (which the DSL uses internally) have no source file
- Result: `OSError: could not get source code`

### The Solution: Linecache Population

Instead of passing source explicitly, we populate Python's `linecache` module with the rewritten source before execution:

```python
# 1. Rewrite the outer function AST
rewritten_ast = rewriter.rewrite(original_ast)

# 2. Unparse to string
rewritten_source = ast.unparse(rewritten_ast)

# 3. Populate linecache BEFORE exec()
linecache.cache[filename] = (
    len(rewritten_source),
    None,  # mtime
    rewritten_source.splitlines(keepends=True),
    filename
)

# 4. Compile and execute
compiled = compile(rewritten_source, filename, 'exec')
exec(compiled, namespace)
```

Now when `@callable` decorator runs on the nested function, `inspect.getsourcelines()` finds the source in linecache and succeeds!

### Why This Works

The key insight is that `inspect.getsourcelines()` checks multiple sources:
1. Actual source files (`.py` files)
2. `linecache` module (used for tracebacks and debugging)

By populating `linecache` with our rewritten source, we make dynamically defined functions appear to have source available.

### Benefits Over `source=` Parameter

The previous approach required:
- Special detection of nested staged functions
- Capturing source via `ast.unparse(node)`
- Passing `source=` keyword to decorator
- Complex AST transformation for decorator application

The linecache approach:
- **No special handling** for nested staged functions
- **Uniform treatment** of all nested functions
- **Natural multistaging** - nested `@callable` just works
- **Simpler code** - removed `_is_staged_decorator()` and `_rewrite_nested_staged_function()`

### The Multistage Data Flow

```
Decoration Time              Rewrite Time                 Execution Time
     |                            |                              |
     v                            v                              v
+-------------+            +--------------+             +----------------+
| @callable   |   AST      | ASTRewriter  |  Unparse   | populate       |
| def inner() | ---------> | rewrites     | ---------> | linecache      |
|             |   saved    | AST          |  to string | + exec()       |
+-------------+            +--------------+             +----------------+
                                                                  |
                                                                  v
                                                           +----------------+
                                                           | @callable runs |
                                                           | on inner()     |
                                                           | inspect finds  |
                                                           | source!        |
                                                           +----------------+
                                                                  |
                                                                  v
                                                           +----------------+
                                                           | TemplatedFunction|
                                                           | created for    |
                                                           | inner()        |
                                                           +----------------+
```

---

## ⚠️ Limitations & Edge Cases

### Variable Assignment Semantics
All variable assignments create DSL variables to ensure correct reassignment. This has some implications:

1. **More alloca instructions**: The IR will have more `alloca`/`store`/`load` operations compared to fully optimized constant folding.
2. **No Python variable reuse**: Once a variable is assigned in DSL context, it becomes a DSL variable:
   ```python
   @kernel
   def example():
       a = 1.0           # DSL variable (not Python constant)
       b = a + 1.0       # DSL operation
   ```

### Constant Folding Limitations
While basic constant folding works at the Python level, **complex nested struct constant folding in DSL context** requires the callable to be specialized from a kernel:
```python
# This works at Python level
@struct
class Point:
    x: Float
    y: Float
c = Const[Point](1.0, 2.0)  # Python ConstantValue

# DSL-level folding of nested structs needs kernel context
@callable
def use_point() -> Float:
    p = Const[Point](1.0, 2.0)
    return p.x + p.y  # Requires specialization to fold
```

### TemplatedFunction Printing
Unspecialized templated functions cannot be printed as IR:
```python
@callable
def func(x: Float) -> Float:
    return x

# This is a TemplatedFunction, not a StagedFunction
print(func.ir)  # Error: no 'ir' attribute until specialized

# Must specialize first
staged = func[Float]  # Now has .ir
```

### List/Tuple Literals in DSL
List/tuple/dict/set literals remain as Python values for host-side iteration:
```python
@kernel
def example():
    # This works - Python list for host iteration
    impls = [func1, func2, func3]
    for impl in impls:
        result = impl(val)
    
    # This also works - indexed access
    impls[0](val)
```

---

## 📊 Debugging Features

### Environment Variables
- `LUISA_DUMP_REWRITTEN_AST=1`: Print transformed AST before execution
- `LUISA_DUMP_IR=1`: Print generated IR

### Utilities
```python
from luisa import pprint
from luisa.lang.inspect import format_ir_summary

# Pretty-print IR
ir = my_kernel.ir
print(pprint(ir))

# Get statistics
summary = format_ir_summary(ir)
print(summary)
```

---

## 🎯 Best Practices

1. **Use Templates for Generic Code**: Write one function, specialize for multiple types
2. **Leverage Constant Folding**: Complex math on literals is evaluated at compile time
3. **Prefer Structured Control Flow**: Use `if/else`, `while`, `for` over manual jumps
4. **Type Annotations**: Help the type system with explicit annotations on function boundaries
5. **Static Loops for Unrolling**: Use `static_range()` for compile-time loop unrolling
