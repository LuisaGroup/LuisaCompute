# New Python DSL Design Document

## Table of Contents

1. [Overview](#overview)
2. [Goals](#goals)
3. [Architecture Overview](#architecture-overview)
4. [Directory Structure](#directory-structure)
5. [Type System](#type-system)
6. [Expression System](#expression-system)
7. [Intermediate Representation (IR)](#intermediate-representation-ir)
8. [Control Flow with Context Managers](#control-flow-with-context-managers)
9. [Multistage Programming](#multistage-programming)
10. [User-Facing API](#user-facing-api)
11. [Builtin Functions](#builtin-functions)
12. [Type Checking](#type-checking)
13. [Implementation Roadmap](#implementation-roadmap)
14. [Example Usage](#example-usage)
15. [Benefits of This Design](#benefits-of-this-design)

## Overview

This document describes the design of a new Python DSL (v2) for LuisaCompute with complete type hinting support and multistage programming capabilities. The new DSL translates Python AST to an Intermediate Representation (IR) that is JSON-serializable for easy data exchange with the C++ backend.

## Goals

1. **Complete Type Hinting**: Full support for Python type hints with static type checking
2. **Multistage Programming**: Separate compilation stages for better performance and flexibility
3. **JSON-Serializable IR**: Easy data exchange between Python frontend and C++ backend
4. **Clean Architecture**: Modular design that's easy to extend and maintain
5. **Type Safety**: Catch type errors at compile time, not runtime

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Python Source Code                              │
│  @kernel                                                                     │
│  def compute_shader(a: Buffer[float], b: float3) -> None:                    │
│      idx = dispatch_id().x                                                   │
│      a[idx] = a[idx] + b                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Stage 1: AST Parser & Analyzer                       │
│  • Parse Python AST                                                          │
│  • Type inference and checking                                               │
│  • Semantic analysis                                                         │
│  • Captured variable analysis                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Stage 2: IR Builder Function Generation                 │
│  • Generate a Python function that builds the IR when executed               │
│  • This is the "staged" code - it captures all compile-time information      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Stage 3: IR Builder Execution                           │
│  • Execute the builder function to construct the actual IR                   │
│  • Type-specialized based on actual argument types                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Stage 4: IR (Intermediate Representation)            │
│  • JSON-serializable data structures                                         │
│  • Can be serialized to JSON for C++ backend                                 │
│  • Can be directly converted to XIR in Python                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Stage 5: Backend Code Generation                        │
│  • Convert IR to Luisa AST/XIR                                               │
│  • Compile to GPU/CPU machine code                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/py/luisa/v2/
├── __init__.py              # Public API exports
├── types.py                 # Type system definitions
├── expr.py                  # Expression representations
├── ir.py                    # IR node definitions
├── builder.py               # IR builder interface
├── parser.py                # Python AST parser
├── typechecker.py           # Type inference and checking
├── codegen/                 # Code generation backends
│   ├── __init__.py
│   ├── xir.py               # XIR code generation
│   └── json_serializer.py   # JSON serialization
└── builtins/                # Builtin functions
    ├── __init__.py
    ├── math.py              # Math operations
    ├── memory.py            # Buffer/texture operations
    └── rtx.py               # Ray tracing operations
```

## Type System

### Core Types

```python
from __future__ import annotations
from typing import TypeVar, Generic, Union, Optional
from dataclasses import dataclass
from enum import Enum, auto

class ScalarType(Enum):
    """Scalar data types"""
    BOOL = auto()
    INT8 = auto()
    UINT8 = auto()
    INT16 = auto()
    UINT16 = auto()
    INT32 = auto()
    UINT32 = auto()
    INT64 = auto()
    UINT64 = auto()
    FLOAT16 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()

@dataclass(frozen=True)
class Type:
    """Base class for all types in the DSL"""
    pass

@dataclass(frozen=True)
class Scalar(Type):
    """Scalar type"""
    dtype: ScalarType
    
    # Predefined scalar types
    @classmethod
    def bool(cls) -> Scalar: return cls(ScalarType.BOOL)
    @classmethod
    def int32(cls) -> Scalar: return cls(ScalarType.INT32)
    @classmethod
    def uint32(cls) -> Scalar: return cls(ScalarType.UINT32)
    @classmethod
    def float32(cls) -> Scalar: return cls(ScalarType.FLOAT32)
    # ... etc

@dataclass(frozen=True)
class Vector(Type):
    """Vector type (e.g., float3, int4)"""
    element: Scalar
    size: int  # 2, 3, or 4

@dataclass(frozen=True)
class Matrix(Type):
    """Matrix type (e.g., float3x3)"""
    element: Scalar  # typically float32
    size: int  # 2, 3, or 4

@dataclass(frozen=True)
class Array(Type):
    """Fixed-size array type"""
    element: Type
    size: int

@dataclass(frozen=True)
class Struct(Type):
    """Struct type"""
    name: str
    fields: tuple[tuple[str, Type], ...]
    alignment: int = 4

@dataclass(frozen=True)
class Buffer(Type):
    """Buffer type (GPU memory)"""
    element: Type

@dataclass(frozen=True)
class Texture2D(Type):
    """2D texture type"""
    element: Scalar

@dataclass(frozen=True)
class Texture3D(Type):
    """3D texture type"""
    element: Scalar

@dataclass(frozen=True)
class BindlessArray(Type):
    """Bindless array type"""
    pass

@dataclass(frozen=True)
class Accel(Type):
    """Acceleration structure type for ray tracing"""
    pass

@dataclass(frozen=True)
class RayQuery(Type):
    """Ray query type"""
    query_any: bool  # True for RayQueryAny, False for RayQueryAll

@dataclass(frozen=True)
class Callable(Type):
    """Callable function type"""
    arg_types: tuple[Type, ...]
    ret_type: Optional[Type]

@dataclass(frozen=True)
class Void(Type):
    """Void type"""
    pass
```

### Type Aliases for Convenience

```python
# Scalar aliases
bool_ = Scalar.bool()
int8 = Scalar(ScalarType.INT8)
uint8 = Scalar(ScalarType.UINT8)
int16 = Scalar(ScalarType.INT16)
uint16 = Scalar(ScalarType.UINT16)
int32 = Scalar(ScalarType.INT32)
uint32 = Scalar(ScalarType.UINT32)
int64 = Scalar(ScalarType.INT64)
uint64 = Scalar(ScalarType.UINT64)
float16 = Scalar(ScalarType.FLOAT16)
float32 = Scalar(ScalarType.FLOAT32)
float64 = Scalar(ScalarType.FLOAT64)

# Common vector types
int2 = Vector(int32, 2)
int3 = Vector(int32, 3)
int4 = Vector(int32, 4)
uint2 = Vector(uint32, 2)
uint3 = Vector(uint32, 3)
uint4 = Vector(uint32, 4)
float2 = Vector(float32, 2)
float3 = Vector(float32, 3)
float4 = Vector(float32, 4)
bool2 = Vector(bool_, 2)
bool3 = Vector(bool_, 3)
bool4 = Vector(bool_, 4)

# Matrix types
float2x2 = Matrix(float32, 2)
float3x3 = Matrix(float32, 3)
float4x4 = Matrix(float32, 4)
```

## Expression System

Expressions represent values in the DSL. They are used during parsing and type checking.

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Expr:
    """Base class for expressions"""
    type: Type
    
@dataclass
class ConstExpr(Expr):
    """Constant expression"""
    value: Any  # Python literal value

@dataclass
class VarExpr(Expr):
    """Variable reference"""
    name: str
    is_lvalue: bool = False

@dataclass
class BinaryOpExpr(Expr):
    """Binary operation"""
    op: str  # '+', '-', '*', '/', etc.
    left: Expr
    right: Expr

@dataclass
class UnaryOpExpr(Expr):
    """Unary operation"""
    op: str  # '-', '~', 'not'
    operand: Expr

@dataclass
class CallExpr(Expr):
    """Function call"""
    func: Expr
    args: list[Expr]
    kwargs: dict[str, Expr]

@dataclass
class SubscriptExpr(Expr):
    """Subscript operation (a[i])"""
    value: Expr
    index: Expr

@dataclass
class AttributeExpr(Expr):
    """Attribute access (a.b)"""
    value: Expr
    attr: str

@dataclass
class SwizzleExpr(Expr):
    """Vector swizzle (v.xyz)"""
    value: Expr
    pattern: str  # e.g., "xyz", "w"

@dataclass
class CastExpr(Expr):
    """Type cast"""
    target_type: Type
    value: Expr

@dataclass
class TernaryExpr(Expr):
    """Conditional expression (a if cond else b)"""
    condition: Expr
    true_val: Expr
    false_val: Expr
```

## Intermediate Representation (IR)

The IR is designed to be JSON-serializable and closely maps to Luisa's XIR.

### IR Nodes

```python
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum, auto

class IROp(Enum):
    """IR operation types"""
    # Literals and constants
    CONST = auto()
    
    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()
    
    # Bitwise
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_NOT = auto()
    SHL = auto()
    SHR = auto()
    
    # Comparison
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Math functions
    SQRT = auto()
    POW = auto()
    EXP = auto()
    LOG = auto()
    SIN = auto()
    COS = auto()
    TAN = auto()
    ABS = auto()
    MIN = auto()
    MAX = auto()
    CLAMP = auto()
    LERP = auto()
    DOT = auto()
    CROSS = auto()
    NORMALIZE = auto()
    LENGTH = auto()
    
    # Memory
    ALLOCA = auto()
    LOAD = auto()
    STORE = auto()
    GEP = auto()  # Get element pointer
    
    # Resources
    BUFFER_READ = auto()
    BUFFER_WRITE = auto()
    TEXTURE_READ = auto()
    TEXTURE_WRITE = auto()
    TEXTURE_SAMPLE = auto()
    
    # Control flow
    BR = auto()
    COND_BR = auto()
    PHI = auto()
    RETURN = auto()
    
    # Function calls
    CALL = auto()
    
    # Cast
    CAST = auto()
    BITCAST = auto()
    
    # Special registers
    THREAD_ID = auto()
    BLOCK_ID = auto()
    DISPATCH_ID = auto()
    DISPATCH_SIZE = auto()
    
    # Ray tracing
    TRACE_CLOSEST = auto()
    TRACE_ANY = auto()
    RAY_QUERY_ALL = auto()
    RAY_QUERY_ANY = auto()
    
    # Atomic
    ATOMIC_EXCHANGE = auto()
    ATOMIC_ADD = auto()
    ATOMIC_SUB = auto()
    ATOMIC_AND = auto()
    ATOMIC_OR = auto()
    ATOMIC_XOR = auto()
    ATOMIC_MIN = auto()
    ATOMIC_MAX = auto()
    ATOMIC_CMP_EXCH = auto()
    
    # Warp operations
    WARP_SUM = auto()
    WARP_PRODUCT = auto()
    WARP_MIN = auto()
    WARP_MAX = auto()
    WARP_ALL = auto()
    WARP_ANY = auto()
    WARP_BROADCAST = auto()
    
    # Synchronization
    SYNC_BLOCK = auto()
    
    # Print
    PRINT = auto()

@dataclass
class IRValue:
    """Base class for IR values"""
    type: Type
    name: Optional[str] = None

@dataclass
class IRConstant(IRValue):
    """Constant value in IR"""
    value: Any

@dataclass
class IRInstruction:
    """IR instruction"""
    op: IROp
    type: Type
    args: list[Union[int, str, IRValue]]  # Can reference other instructions by index
    result: Optional[str] = None

@dataclass
class IRBasicBlock:
    """Basic block in IR"""
    name: str
    instructions: list[IRInstruction]
    terminator: Optional[IRInstruction] = None

@dataclass
class IRFunction:
    """IR function"""
    name: str
    arg_types: list[Type]
    ret_type: Optional[Type]
    blocks: list[IRBasicBlock]
    is_kernel: bool = False
    block_size: Optional[tuple[int, int, int]] = None

@dataclass
class IRModule:
    """IR module containing functions"""
    functions: list[IRFunction]
    constants: list[IRConstant] = field(default_factory=list)
```

### JSON Serialization Example

```python
# Example IR for: def add(a: float, b: float) -> float: return a + b
{
    "module": {
        "functions": [
            {
                "name": "add",
                "arg_types": [
                    {"kind": "scalar", "dtype": "float32"},
                    {"kind": "scalar", "dtype": "float32"}
                ],
                "ret_type": {"kind": "scalar", "dtype": "float32"},
                "is_kernel": False,
                "blocks": [
                    {
                        "name": "entry",
                        "instructions": [
                            {
                                "op": "ADD",
                                "type": {"kind": "scalar", "dtype": "float32"},
                                "args": [0, 1],  # References to arguments
                                "result": "add_result"
                            }
                        ],
                        "terminator": {
                            "op": "RETURN",
                            "type": {"kind": "void"},
                            "args": ["add_result"],
                            "result": null
                        }
                    }
                ]
            }
        ]
    }
}
```

## Control Flow with Context Managers

The control flow design uses Python context managers for clean, readable code with automatic constant folding support.

### IfScope - Conditional Branching

```python
from contextlib import contextmanager
from typing import Optional, Callable

class IfScope:
    """
    Context manager for if-then-else control flow.
    
    Automatically folds constant conditions:
    - If cond is known True: only executes true branch
    - If cond is known False: only executes false branch (if exists)
    - Otherwise: generates conditional branch IR
    """
    
    def __init__(self, builder: 'IRBuilder', condition: 'Value'):
        self.builder = builder
        self.condition = condition
        self.true_block: Optional[IRBasicBlock] = None
        self.false_block: Optional[IRBasicBlock] = None
        self.merge_block: Optional[IRBasicBlock] = None
        self.has_false_branch = False
        
    def __enter__(self):
        # Check for constant folding
        if isinstance(self.condition, ConstantValue):
            if self.condition.value == True:
                # Constant True - no branching needed
                self._folded = True
                self._fold_result = True
                return self
            elif self.condition.value == False:
                # Constant False - skip true branch
                self._folded = True
                self._fold_result = False
                # Return a no-op context
                return NoOpScope(self.builder)
        
        self._folded = False
        
        # Create blocks
        self.true_block = self.builder.create_block("if_true")
        self.merge_block = self.builder.create_block("if_merge")
        
        # Emit conditional branch
        self.builder.branch_conditional(
            self.condition,
            self.true_block,
            self.merge_block  # Fall-through to merge if no else
        )
        
        # Start true block
        self.builder.set_insert_point(self.true_block)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        if self._folded:
            return True
            
        # Branch to merge if not terminated
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.merge_block)
        
        # If no else branch, set insert point to merge
        if not self.has_false_branch:
            self.builder.set_insert_point(self.merge_block)
        
        return True
    
    def otherwise(self):
        """Start the else branch"""
        if self._folded:
            # Return appropriate scope based on constant folding
            if self._fold_result:
                # True was folded, so else branch never executes
                return NoOpScope(self.builder)
            else:
                # False was folded, so else branch always executes
                return DirectScope(self.builder)
        
        self.has_false_branch = True
        
        # Create false block
        self.false_block = self.builder.create_block("if_false")
        
        # Update the conditional branch to point to false block
        self.builder.update_conditional_branch(
            self.condition,
            self.true_block,
            self.false_block
        )
        
        # Start false block
        self.builder.set_insert_point(self.false_block)
        
        return ElseScope(self.builder, self.merge_block)


class ElseScope:
    """Context manager for else branch"""
    
    def __init__(self, builder: 'IRBuilder', merge_block: IRBasicBlock):
        self.builder = builder
        self.merge_block = merge_block
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        # Branch to merge
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.merge_block)
        
        # Set insert point to merge block
        self.builder.set_insert_point(self.merge_block)
        return True


class NoOpScope:
    """No-op scope for constant-folded branches that don't execute"""
    
    def __init__(self, builder: 'IRBuilder'):
        self.builder = builder
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True
    
    def otherwise(self):
        # Return a scope that will execute
        return DirectScope(self.builder)


class DirectScope:
    """Direct execution scope for constant-folded branches that always execute"""
    
    def __init__(self, builder: 'IRBuilder'):
        self.builder = builder
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True
    
    def otherwise(self):
        # Return no-op for else branch
        return NoOpScope(self.builder)
```

### Loop Scopes - While and For Loops

```python
class WhileScope:
    """
    Context manager for while loops.
    
    Structure:
        header_block:
            if condition:
                goto body_block
            else:
                goto exit_block
        body_block:
            ... body ...
            goto header_block
        exit_block:
    """
    
    def __init__(self, builder: 'IRBuilder', condition: 'Value'):
        self.builder = builder
        self.condition = condition
        self.header_block: Optional[IRBasicBlock] = None
        self.body_block: Optional[IRBasicBlock] = None
        self.exit_block: Optional[IRBasicBlock] = None
        
    def __enter__(self):
        # Check for constant folding - if False, skip entire loop
        if isinstance(self.condition, ConstantValue) and self.condition.value == False:
            self._folded = True
            return NoOpScope(self.builder)
        
        self._folded = False
        
        # Create blocks
        self.header_block = self.builder.create_block("while_header")
        self.body_block = self.builder.create_block("while_body")
        self.exit_block = self.builder.create_block("while_exit")
        
        # Branch to header
        self.builder.branch(self.header_block)
        
        # Emit condition check in header
        self.builder.set_insert_point(self.header_block)
        self.builder.branch_conditional(
            self.condition,
            self.body_block,
            self.exit_block
        )
        
        # Push loop context for break/continue
        self.builder.push_loop(self.header_block, self.exit_block)
        
        # Start body
        self.builder.set_insert_point(self.body_block)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        if self._folded:
            return True
            
        # Loop back to header
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.header_block)
        
        # Pop loop context
        self.builder.pop_loop()
        
        # Continue in exit block
        self.builder.set_insert_point(self.exit_block)
        return True


class ForRangeScope:
    """
    Context manager for for-loops over range().
    
    This generates a dynamic loop on the device-side, even if the
    range bounds are known at compile time. This is the default
    behavior to avoid code bloat from unrolling.
    """
    
    def __init__(self, builder: 'IRBuilder', 
                 start: 'Value', stop: 'Value', step: 'Value',
                 loop_var_name: str):
        self.builder = builder
        self.start = start
        self.stop = stop
        self.step = step
        self.loop_var_name = loop_var_name
        self.loop_var: Optional['Value'] = None
        
    def __enter__(self):
        # Create blocks
        self.header_block = self.builder.create_block("for_header")
        self.body_block = self.builder.create_block("for_body")
        self.exit_block = self.builder.create_block("for_exit")
        
        # Allocate loop variable
        self.loop_var = self.builder.alloca(self.start.type)
        self.builder.store(self.loop_var, self.start)
        
        # Branch to header
        self.builder.branch(self.header_block)
        
        # Header: check condition
        self.builder.set_insert_point(self.header_block)
        current_val = self.builder.load(self.loop_var)
        cond = self.builder.compare_lt(current_val, self.stop)  # i < stop
        self.builder.branch_conditional(cond, self.body_block, self.exit_block)
        
        # Push loop context
        self.builder.push_loop(self.header_block, self.exit_block)
        
        # Start body
        self.builder.set_insert_point(self.body_block)
        
        # Bind loop variable name to current value
        current_val = self.builder.load(self.loop_var)
        self.builder.bind_local(self.loop_var_name, current_val)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        # Increment loop variable
        current_val = self.builder.load(self.loop_var)
        next_val = self.builder.add(current_val, self.step)
        self.builder.store(self.loop_var, next_val)
        
        # Loop back
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.header_block)
        
        # Pop loop context
        self.builder.pop_loop()
        
        # Continue in exit block
        self.builder.set_insert_point(self.exit_block)
        return True


class UnrolledForScope:
    """
    Context manager for explicitly unrolled for-loops.
    
    Usage: for i in unrolled(range(n)): ...
    
    This fully unrolls the loop at compile time. Only use for small
    iteration counts to avoid code bloat.
    """
    
    def __init__(self, builder: 'IRBuilder', 
                 start: int, stop: int, step: int,
                 loop_var_name: str):
        self.builder = builder
        self.start = start
        self.stop = stop
        self.step = step
        self.loop_var_name = loop_var_name
        self.iteration = 0
        
    def __enter__(self):
        # Start first iteration
        self._start_iteration()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        return True
    
    def _start_iteration(self):
        """Start a new unrolled iteration"""
        val = self.start + self.iteration * self.step
        if val >= self.stop:
            return False
            
        # Create block for this iteration
        iter_block = self.builder.create_block(f"unroll_{self.iteration}")
        self.builder.set_insert_point(iter_block)
        
        # Bind loop variable to constant value
        const_val = self.builder.constant(IntType.int32(), val)
        self.builder.bind_local(self.loop_var_name, const_val)
        
        return True
    
    def next_iteration(self):
        """Called at the end of each iteration body"""
        self.iteration += 1
        return self._start_iteration()
```

### Switch/Match Scope

```python
class SwitchScope:
    """
    Context manager for switch statements (match in Python).
    
    Supports constant folding when the switch value is known.
    """
    
    def __init__(self, builder: 'IRBuilder', value: 'Value'):
        self.builder = builder
        self.value = value
        self.cases: list[tuple[list[int], IRBasicBlock]] = []
        self.default_block: Optional[IRBasicBlock] = None
        self.exit_block: Optional[IRBasicBlock] = None
        self.active_case_values: Optional[list[int]] = None
        
    def __enter__(self):
        # Check for constant folding
        if isinstance(self.value, ConstantValue):
            self._constant_value = self.value.value
            self._folded = True
        else:
            self._constant_value = None
            self._folded = False
            self.exit_block = self.builder.create_block("switch_exit")
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        if self._folded:
            return True
            
        # Emit switch instruction
        self.builder.set_insert_point(self._switch_block)
        self.builder.switch(self.value, self.default_block, self.cases)
        
        # Continue in exit block
        self.builder.set_insert_point(self.exit_block)
        return True
    
    def case(self, *values: int):
        """Start a case block"""
        if self._folded:
            # Check if this case matches the constant value
            if self._constant_value in values:
                # This case executes
                return DirectCaseScope(self.builder)
            else:
                # This case is skipped
                return NoOpCaseScope(self.builder)
        
        # Create case block
        case_block = self.builder.create_block(f"case_{values[0]}")
        self.cases.append((list(values), case_block))
        
        return CaseScope(self.builder, case_block, self.exit_block)
    
    def default(self):
        """Start the default block"""
        if self._folded:
            # Check if default should execute
            all_case_values = []
            for vals, _ in self.cases:
                all_case_values.extend(vals)
            if self._constant_value not in all_case_values:
                return DirectCaseScope(self.builder)
            else:
                return NoOpCaseScope(self.builder)
        
        self.default_block = self.builder.create_block("case_default")
        return CaseScope(self.builder, self.default_block, self.exit_block)


class CaseScope:
    """Context manager for a case block"""
    
    def __init__(self, builder: 'IRBuilder', 
                 case_block: IRBasicBlock, exit_block: IRBasicBlock):
        self.builder = builder
        self.case_block = case_block
        self.exit_block = exit_block
        
    def __enter__(self):
        self.builder.set_insert_point(self.case_block)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
            
        # Break to exit
        if not self.builder.current_block.is_terminated():
            self.builder.branch(self.exit_block)
            
        return True


class DirectCaseScope:
    """Case scope that always executes (for constant folding)"""
    
    def __init__(self, builder: 'IRBuilder'):
        self.builder = builder
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class NoOpCaseScope:
    """Case scope that never executes (for constant folding)"""
    
    def __init__(self, builder: 'IRBuilder'):
        self.builder = builder
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True
```

### Builder API for Control Flow

```python
class IRBuilder:
    """IR Builder with control flow support"""
    
    # ... previous methods ...
    
    def if_(self, condition: 'Value') -> IfScope:
        """Create an if scope"""
        return IfScope(self, condition)
    
    def while_(self, condition: 'Value') -> WhileScope:
        """Create a while loop scope"""
        return WhileScope(self, condition)
    
    def for_range(self, start: 'Value', stop: 'Value', 
                  step: 'Value', loop_var: str) -> ForRangeScope:
        """Create a for-range loop scope (dynamic)"""
        return ForRangeScope(self, start, stop, step, loop_var)
    
    def for_unrolled(self, start: int, stop: int, 
                     step: int, loop_var: str) -> UnrolledForScope:
        """Create an unrolled for-range loop scope"""
        return UnrolledForScope(self, start, stop, step, loop_var)
    
    def switch(self, value: 'Value') -> SwitchScope:
        """Create a switch scope"""
        return SwitchScope(self, value)
    
    def break_(self):
        """Emit a break instruction"""
        loop_info = self._loop_stack[-1]
        self.branch(loop_info.exit_block)
    
    def continue_(self):
        """Emit a continue instruction"""
        loop_info = self._loop_stack[-1]
        self.branch(loop_info.header_block)
```

### Usage in Builder Executor

```python
class BuilderExecutor:
    """Execute Python AST using the IR builder"""
    
    def visit_If(self, node):
        """Visit if statement using IfScope"""
        cond = self.execute(node.test)
        
        with self.builder.if_(cond) as if_scope:
            # True branch
            for stmt in node.body:
                self.execute(stmt)
            
            # False branch (if exists)
            if node.orelse:
                with if_scope.otherwise():
                    for stmt in node.orelse:
                        self.execute(stmt)
    
    def visit_While(self, node):
        """Visit while loop using WhileScope"""
        # Build the condition
        cond = self.execute(node.test)
        
        with self.builder.while_(cond):
            for stmt in node.body:
                self.execute(stmt)
    
    def visit_For(self, node):
        """Visit for loop - determine if unrolled or dynamic"""
        # Check if this is an unrolled loop
        is_unrolled = self._is_unrolled_iterator(node.iter)
        
        if is_unrolled:
            # Get unroll parameters
            start, stop, step = self._get_unroll_range(node.iter)
            loop_var = node.target.id
            
            with self.builder.for_unrolled(start, stop, step, loop_var):
                for stmt in node.body:
                    self.execute(stmt)
        else:
            # Dynamic loop
            start, stop, step = self._get_range_values(node.iter)
            loop_var = node.target.id
            
            with self.builder.for_range(start, stop, step, loop_var):
                for stmt in node.body:
                    self.execute(stmt)
    
    def visit_Match(self, node):
        """Visit match statement using SwitchScope"""
        value = self.execute(node.subject)
        
        with self.builder.switch(value) as switch:
            for case in node.cases:
                if isinstance(case.pattern, ast.MatchValue):
                    case_values = [case.pattern.value.value]
                    with switch.case(*case_values):
                        for stmt in case.body:
                            self.execute(stmt)
                elif case.pattern is None:  # default case
                    with switch.default():
                        for stmt in case.body:
                            self.execute(stmt)
```

### Constant Folding Example

```python
# Example 1: Constant condition folding
@kernel
def example1(x: buffer[float]):
    if True:  # Folded - always executes
        x[0] = 1.0
    else:     # Folded - never executes (dead code eliminated)
        x[0] = 2.0

# Example 2: Variable condition - generates branch
@kernel
def example2(x: buffer[float], cond: bool):
    if cond:  # Generates conditional branch
        x[0] = 1.0
    else:
        x[0] = 2.0

# Example 3: Constant switch folding
@kernel
def example3(x: buffer[float]):
    match 2:  # Constant value
        case 1:  # Skipped
            x[0] = 1.0
        case 2:  # Executed
            x[0] = 2.0
        case _:  # Skipped
            x[0] = 0.0

# Example 4: Dynamic vs unrolled loops
@kernel
def example4(x: buffer[float]):
    # Dynamic loop - runs on device
    for i in range(100):  # Generates loop instructions
        x[i] = float(i)
    
    # Unrolled loop - unrolled at compile time
    for i in unrolled(range(4)):  # Generates 4 stores
        x[i] = float(i)
```

## Multistage Programming

The multistage approach separates compile-time and runtime concerns:

### Stage 1: Parsing (Compile Time)

```python
from ast import parse, FunctionDef
from typing import Callable

class Parser:
    """Parse Python AST and create a staged builder function"""
    
    def parse_function(self, func: Callable) -> StagedFunction:
        """
        Parse a Python function and return a staged function.
        
        The staged function, when called with actual arguments,
        will execute the builder to generate IR.
        """
        source = getsource(func)
        tree = parse(source)
        func_def = tree.body[0]
        
        # Analyze captured variables
        captured = self._analyze_captured_vars(func)
        
        # Analyze argument types from annotations
        arg_types = self._extract_arg_types(func_def)
        
        # Create the staged function
        return StagedFunction(
            name=func.__name__,
            ast=func_def,
            captured_vars=captured,
            arg_types=arg_types,
            ret_type=self._extract_ret_type(func_def)
        )
```

### Stage 2: Builder Function Generation

```python
class StagedFunction:
    """
    A function that generates IR when called.
    
    This is the core of multistage programming - the actual IR generation
    is deferred until the function is called with specific argument types.
    """
    
    def __init__(self, name: str, ast: FunctionDef, 
                 captured_vars: dict, arg_types: list[Type],
                 ret_type: Optional[Type]):
        self.name = name
        self.ast = ast
        self.captured_vars = captured_vars
        self.arg_types = arg_types
        self.ret_type = ret_type
        self._cache: dict[tuple[Type, ...], IRFunction] = {}
    
    def __call__(self, *args, **kwargs):
        """
        When called, this function:
        1. Determines the actual argument types
        2. Creates an IR builder
        3. Executes the builder to generate IR
        4. Returns/caches the compiled function
        """
        # Get actual argument types
        actual_types = tuple(get_runtime_type(arg) for arg in args)
        
        # Check cache
        if actual_types in self._cache:
            return self._cache[actual_types]
        
        # Create builder context
        builder = IRBuilder(self.name, actual_types, self.ret_type)
        
        # Execute the builder (this is where the magic happens)
        # This will recursively walk the AST and call builder methods
        executor = BuilderExecutor(builder, self.captured_vars)
        executor.execute(self.ast)
        
        # Get the generated IR
        ir_func = builder.build()
        
        # Cache and return
        self._cache[actual_types] = ir_func
        return ir_func
```

### Stage 3: Builder Execution

```python
class IRBuilder:
    """
    Builder for constructing IR.
    
    This class provides methods that directly correspond to IR operations.
    When executed, it constructs the IR data structures.
    """
    
    def __init__(self, name: str, arg_types: tuple[Type, ...], 
                 ret_type: Optional[Type]):
        self.name = name
        self.arg_types = arg_types
        self.ret_type = ret_type
        self.blocks: list[IRBasicBlock] = []
        self.current_block: Optional[IRBasicBlock] = None
        self.instruction_counter = 0
        self.local_vars: dict[str, int] = {}  # name -> instruction index
    
    def _emit(self, op: IROp, type: Type, args: list) -> str:
        """Emit an instruction and return its result name"""
        result_name = f"t{self.instruction_counter}"
        self.instruction_counter += 1
        
        instr = IRInstruction(
            op=op,
            type=type,
            args=args,
            result=result_name
        )
        self.current_block.instructions.append(instr)
        return result_name
    
    def add(self, left: str, right: str, type: Type) -> str:
        """Emit an add instruction"""
        return self._emit(IROp.ADD, type, [left, right])
    
    def load(self, ptr: str, type: Type) -> str:
        """Emit a load instruction"""
        return self._emit(IROp.LOAD, type, [ptr])
    
    def store(self, ptr: str, value: str) -> None:
        """Emit a store instruction"""
        self._emit(IROp.STORE, Void(), [ptr, value])
    
    def if_then_else(self, cond: str, 
                     then_builder: Callable[[], str],
                     else_builder: Optional[Callable[[], str]]) -> str:
        """Emit an if-then-else control flow"""
        # Create basic blocks
        then_block = self._create_block("then")
        else_block = self._create_block("else") if else_builder else None
        merge_block = self._create_block("merge")
        
        # Emit conditional branch
        self._emit(IROp.COND_BR, Void(), 
                   [cond, then_block.name, 
                    else_block.name if else_block else merge_block.name])
        
        # Build then block
        self.current_block = then_block
        then_result = then_builder()
        self._emit(IROp.BR, Void(), [merge_block.name])
        
        # Build else block
        if else_block:
            self.current_block = else_block
            else_result = else_builder()
            self._emit(IROp.BR, Void(), [merge_block.name])
        
        # Continue in merge block
        self.current_block = merge_block
        
        # Return phi if there's a result
        if then_result:
            return self._emit(IROp.PHI, type, 
                            [then_result, then_block.name,
                             else_result, else_block.name])
        return None
    
    def build(self) -> IRFunction:
        """Finalize and return the IR function"""
        return IRFunction(
            name=self.name,
            arg_types=list(self.arg_types),
            ret_type=self.ret_type,
            blocks=self.blocks
        )
```

### Stage 4: Builder Executor

```python
class BuilderExecutor:
    """
    Executes the Python AST using the IR builder.
    
    This is essentially an interpreter that interprets the Python AST
    and calls the appropriate builder methods.
    """
    
    def __init__(self, builder: IRBuilder, captured_vars: dict):
        self.builder = builder
        self.captured_vars = captured_vars
        self.local_vars: dict[str, str] = {}  # name -> value reference
    
    def execute(self, node: ast.AST):
        """Execute an AST node"""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def visit_FunctionDef(self, node: FunctionDef):
        """Visit a function definition"""
        # Create entry block
        entry = self.builder._create_block("entry")
        self.builder.current_block = entry
        
        # Initialize arguments as local variables
        for i, arg in enumerate(node.args.args):
            self.local_vars[arg.arg] = f"arg{i}"
        
        # Visit function body
        for stmt in node.body:
            self.execute(stmt)
    
    def visit_Return(self, node):
        """Visit a return statement"""
        value = self.execute(node.value) if node.value else None
        self.builder._emit(IROp.RETURN, 
                          self.builder.ret_type or Void(), 
                          [value] if value else [])
    
    def visit_BinOp(self, node):
        """Visit a binary operation"""
        left = self.execute(node.left)
        right = self.execute(node.right)
        type = self._get_expr_type(node)  # Infer type from AST
        
        op_map = {
            ast.Add: IROp.ADD,
            ast.Sub: IROp.SUB,
            ast.Mult: IROp.MUL,
            ast.Div: IROp.DIV,
        }
        
        return self.builder._emit(op_map[type(node.op)], type, [left, right])
    
    def visit_Subscript(self, node):
        """Visit a subscript operation (array indexing)"""
        value = self.execute(node.value)
        index = self.execute(node.slice)
        elem_type = self._get_element_type(node.value)
        
        # Compute element pointer
        gep = self.builder._emit(IROp.GEP, elem_type, [value, index])
        return self.builder.load(gep, elem_type)
```

## User-Facing API

### Kernel Definition

```python
from luisa.v2 import kernel, buffer, float3, dispatch_id

@kernel
def saxpy(result: buffer[float], a: float, x: buffer[float], y: buffer[float]) -> None:
    """
    Single-precision A*X Plus Y kernel.
    
    Type annotations are mandatory and checked at compile time.
    """
    idx = dispatch_id().x
    result[idx] = a * x[idx] + y[idx]

# Usage
result = buffer.zeros(1024, dtype=float)
x = buffer.from_numpy(np.random.randn(1024).astype(np.float32))
y = buffer.from_numpy(np.random.randn(1024).astype(np.float32))

# The kernel is compiled lazily when first called
# The actual compilation happens in multiple stages:
# 1. Parse the Python AST (done once when @kernel decorator is applied)
# 2. When called, generate IR specialized for the argument types
# 3. Compile IR to GPU/CPU code
saxpy(result, 2.0, x, y, dispatch_size=1024)
```

### Callable Functions

```python
from luisa.v2 import callable, float3

@callable
def lerp(a: float3, b: float3, t: float) -> float3:
    """Linear interpolation between two vectors"""
    return a * (1.0 - t) + b * t

@kernel
def blend_images(out: buffer[float3], img1: buffer[float3], 
                 img2: buffer[float3], alpha: float) -> None:
    idx = dispatch_id().x
    out[idx] = lerp(img1[idx], img2[idx], alpha)
```

### Struct Definition

```python
from luisa.v2 import struct

@struct
class Particle:
    position: float3
    velocity: float3
    mass: float
    
@kernel
def update_particles(particles: buffer[Particle], dt: float) -> None:
    idx = dispatch_id().x
    p = particles[idx]
    p.position = p.position + p.velocity * dt
    particles[idx] = p
```

### Type-Safe Operations

```python
from luisa.v2 import kernel, buffer, float2, length, normalize

@kernel
def normalize_vectors(vectors: buffer[float2], 
                      result: buffer[float2]) -> None:
    idx = dispatch_id().x
    v = vectors[idx]
    
    # Type checking ensures v is float2
    # length(v) returns float
    # normalize(v) returns float2
    if length(v) > 0.0:
        result[idx] = normalize(v)
    else:
        result[idx] = float2(0.0, 0.0)
```

### Loop Control - Dynamic vs Unrolled

```python
from luisa.v2 import kernel, buffer, unrolled

@kernel
def loop_examples(x: buffer[float], n: int) -> None:
    # Dynamic loop - generates device-side loop instructions
    # Runs on GPU/CPU at runtime, even if n is known at compile time
    for i in range(n):
        x[i] = float(i)
    
    # Unrolled loop - fully unrolled at compile time
    # Use only for small iteration counts to avoid code bloat
    for i in unrolled(range(4)):
        x[i] = float(i * i)
    
    # With step
    for i in unrolled(range(0, 8, 2)):  # 0, 2, 4, 6
        x[i] = float(i)
```

**Why separate `unrolled` from normal loops?**

1. **Compilation Time**: Automatically unrolling all constant loops could lead to excessive compilation times when the iteration count is large
2. **Code Bloat**: Unrolling large loops generates massive code that may not fit in instruction cache
3. **User Intent**: Explicit `unrolled()` makes the programmer's intent clear - they want loop unrolling for optimization
4. **Dynamic Loops**: Device-side loops are often desired for register pressure reduction and cache behavior, even when bounds are known

```python
# Implementation of unrolled helper
class UnrolledRange:
    """Marker class for unrolled loops"""
    def __init__(self, start: int, stop: Optional[int] = None, step: int = 1):
        if stop is None:
            start, stop = 0, start
        self.start = start
        self.stop = stop
        self.step = step
    
    def __iter__(self):
        """Python-side iteration for the parser to analyze"""
        return iter(range(self.start, self.stop, self.step))

def unrolled(r: range) -> UnrolledRange:
    """
    Mark a range for compile-time unrolling.
    
    The loop body will be replicated for each iteration.
    Use only for small iteration counts!
    """
    return UnrolledRange(r.start, r.stop, r.step)
```

## Builtin Functions

Builtin functions are defined with type signatures:

```python
from typing import overload
from luisa.v2 import Expr, floatN, intN

@overload
def length(v: float2) -> float: ...
@overload
def length(v: float3) -> float: ...
@overload
def length(v: float4) -> float: ...

def length(v: Expr) -> Expr:
    """Compute the length of a vector"""
    return Expr(
        type=Scalar.float32(),
        ir_ref=builder.call(IROp.LENGTH, v.ir_ref)
    )

@overload
def dot(a: float2, b: float2) -> float: ...
@overload
def dot(a: float3, b: float3) -> float: ...
@overload
def dot(a: float4, b: float4) -> float: ...

def dot(a: Expr, b: Expr) -> Expr:
    """Compute the dot product of two vectors"""
    return Expr(
        type=Scalar.float32(),
        ir_ref=builder.call(IROp.DOT, a.ir_ref, b.ir_ref)
    )

# Dispatch IDs
def dispatch_id() -> uint3:
    """Get the dispatch ID"""
    return Expr(
        type=Vector(Scalar.uint32(), 3),
        ir_ref=builder.call(IROp.DISPATCH_ID)
    )

def thread_id() -> uint3:
    """Get the thread ID within a block"""
    return Expr(
        type=Vector(Scalar.uint32(), 3),
        ir_ref=builder.call(IROp.THREAD_ID)
    )
```

## Type Checking

The type checker runs during parsing to catch errors early:

```python
class TypeChecker:
    """Type checker for the DSL"""
    
    def check_binop(self, left: Type, op: ast.operator, right: Type) -> Type:
        """Check and return the result type of a binary operation"""
        if isinstance(op, (ast.Add, ast.Sub, ast.Mult)):
            # Scalar arithmetic
            if left == right and isinstance(left, Scalar):
                return left
            # Vector arithmetic (with broadcasting)
            if isinstance(left, Vector) and isinstance(right, Vector):
                if left.element == right.element and left.size == right.size:
                    return left
            raise TypeError(
                f"Cannot apply {op.__class__.__name__} to {left} and {right}"
            )
        # ... more checks
    
    def check_call(self, func: Type, args: list[Type]) -> Type:
        """Check a function call"""
        if isinstance(func, Callable):
            if len(args) != len(func.arg_types):
                raise TypeError(
                    f"Expected {len(func.arg_types)} arguments, got {len(args)}"
                )
            for i, (expected, actual) in enumerate(zip(func.arg_types, args)):
                if expected != actual:
                    raise TypeError(
                        f"Argument {i}: expected {expected}, got {actual}"
                    )
            return func.ret_type
        raise TypeError(f"Cannot call non-callable type {func}")
```

## Implementation Roadmap

### Phase 1: Core Infrastructure

1. Type system definitions (`types.py`)
2. IR data structures (`ir.py`)
3. JSON serializer (`codegen/json_serializer.py`)

### Phase 2: Parser and Type Checker

1. AST parser (`parser.py`)
2. Type inference (`typechecker.py`)
3. Expression representations (`expr.py`)

### Phase 3: Builder System

1. IR builder (`builder.py`)
2. Builder executor
3. Staged function wrapper

### Phase 4: Public API

1. `@kernel` decorator
2. `@callable` decorator
3. `@struct` decorator
4. Builtin functions

### Phase 5: Backend Integration

1. IR to XIR conversion
2. Compilation pipeline
3. Execution interface

### Phase 6: Advanced Features

1. Ray tracing support
2. Autodiff support
3. Rasterization support
4. Cooperative matrices

## Example Usage

```python
import numpy as np
from luisa.v2 import *

# Initialize device
init(backend="cuda")

# Define a kernel with full type annotations
@kernel
def matmul_kernel(
    C: buffer[float], 
    A: buffer[float], 
    B: buffer[float],
    M: int, N: int, K: int
) -> None:
    """Simple matrix multiplication kernel"""
    row = dispatch_id().y
    col = dispatch_id().x
    
    if row < M and col < N:
        sum = 0.0
        for k in range(K):
            sum += A[row * K + k] * B[k * N + col]
        C[row * N + col] = sum

# Create buffers
M, N, K = 1024, 1024, 1024
A = buffer.from_numpy(np.random.randn(M, K).astype(np.float32))
B = buffer.from_numpy(np.random.randn(K, N).astype(np.float32))
C = buffer.zeros((M, N), dtype=float)

# Launch kernel
matmul_kernel(C, A, B, M, N, K, dispatch_size=(N, M, 1))

# Get result
result = C.to_numpy()
```

## Benefits of This Design

1. **Type Safety**: Full type checking at compile time with Python type hints
2. **Performance**: Multistage compilation allows aggressive optimization
3. **Flexibility**: JSON-serializable IR enables easy integration with other tools
4. **Maintainability**: Clean separation between parsing, IR generation, and code generation
5. **Extensibility**: Easy to add new operations and types
6. **Debugging**: IR can be inspected, visualized, and debugged independently
