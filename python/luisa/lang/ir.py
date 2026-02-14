"""
Intermediate Representation (IR) for the LuisaCompute Python DSL v2.

This module defines the IR data structures that are used to represent
the compiled kernel code. The IR is designed to be JSON-serializable
for easy exchange with the C++ backend.
"""

from __future__ import annotations
from typing import Optional, Union, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum, auto

if TYPE_CHECKING:
    from .types import Type


# ============================================================================
# Source Location
# ============================================================================

@dataclass(frozen=True)
class SourceLocation:
    """Source code location for debugging."""
    file: str
    line: int

    def __repr__(self) -> str:
        return f"{self.file}:{self.line}"


# ============================================================================
# IR Operations
# ============================================================================

class IROp(Enum):
    """IR operation types."""
    
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
    SHL = auto()  # Shift left
    SHR = auto()  # Shift right
    
    # Comparison
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    
    # Logical
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    LOGICAL_NOT = auto()
    
    # Math functions
    SQRT = auto()
    POW = auto()
    EXP = auto()
    EXP2 = auto()
    LOG = auto()
    LOG2 = auto()
    LOG10 = auto()
    SIN = auto()
    COS = auto()
    TAN = auto()
    ASIN = auto()
    ACOS = auto()
    ATAN = auto()
    ATAN2 = auto()
    SINH = auto()
    COSH = auto()
    TANH = auto()
    ABS = auto()
    FLOOR = auto()
    CEIL = auto()
    ROUND = auto()
    TRUNC = auto()
    FRACT = auto()
    MIN = auto()
    MAX = auto()
    CLAMP = auto()
    LERP = auto()
    SATURATE = auto()
    STEP = auto()
    SMOOTHSTEP = auto()
    DOT = auto()
    CROSS = auto()
    NORMALIZE = auto()
    LENGTH = auto()
    LENGTH_SQUARED = auto()
    DISTANCE = auto()
    REFLECT = auto()
    REFRACT = auto()
    FACEFORWARD = auto()
    
    # Matrix operations
    MATRIX_DETERMINANT = auto()
    MATRIX_TRANSPOSE = auto()
    MATRIX_INVERSE = auto()
    
    # Memory
    ALLOCA = auto()
    LOAD = auto()
    STORE = auto()
    GEP = auto()  # Get element pointer
    MEMBER_ACCESS = auto()  # Struct member access
    
    # Resources
    BUFFER_READ = auto()
    BUFFER_WRITE = auto()
    BUFFER_SIZE = auto()
    TEXTURE2D_READ = auto()
    TEXTURE2D_WRITE = auto()
    TEXTURE2D_SAMPLE = auto()
    TEXTURE2D_SAMPLE_LEVEL = auto()
    TEXTURE3D_READ = auto()
    TEXTURE3D_WRITE = auto()
    TEXTURE3D_SAMPLE = auto()
    
    # Control flow
    PHI = auto()  # Phi node for SSA
    RETURN = auto()
    
    # Structured Control Flow
    IF = auto()
    LOOP = auto()
    BREAK = auto()
    CONTINUE = auto()
    SWITCH = auto()
    
    # Function calls
    CALL = auto()
    CALL_BUILTIN = auto()
    
    # Cast
    CAST = auto()  # Static cast
    BITCAST = auto()  # Bitwise cast
    
    # Special registers
    THREAD_ID = auto()
    BLOCK_ID = auto()
    DISPATCH_ID = auto()
    DISPATCH_SIZE = auto()
    KERNEL_ID = auto()
    OBJECT_ID = auto()
    
    # Ray tracing
    TRACE_CLOSEST = auto()
    TRACE_ANY = auto()
    RAY_QUERY_ALL = auto()
    RAY_QUERY_ANY = auto()
    
    # Atomic operations
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
    WARP_IS_FIRST_ACTIVE_LANE = auto()
    WARP_FIRST_ACTIVE_LANE = auto()
    WARP_ACTIVE_COUNT_BITS = auto()
    WARP_SUM = auto()
    WARP_PRODUCT = auto()
    WARP_MIN = auto()
    WARP_MAX = auto()
    WARP_ALL = auto()
    WARP_ANY = auto()
    WARP_ACTIVE_ALL_EQUAL = auto()
    WARP_ACTIVE_BIT_AND = auto()
    WARP_ACTIVE_BIT_OR = auto()
    WARP_ACTIVE_BIT_XOR = auto()
    WARP_ACTIVE_BIT_MASK = auto()
    WARP_BROADCAST = auto()
    WARP_PREFIX_SUM = auto()
    WARP_PREFIX_PRODUCT = auto()
    WARP_PREFIX_COUNT_BITS = auto()
    WARP_READ_LANE = auto()
    WARP_READ_FIRST_ACTIVE_LANE = auto()
    
    # Synchronization
    SYNC_BLOCK = auto()
    
    # Print
    PRINT = auto()
    
    # Swizzle
    SWIZZLE = auto()
    
    # Additional resource operations
    TEXTURE2D_SIZE = auto()
    TEXTURE3D_SIZE = auto()
    BUFFER_DEVICE_ADDRESS = auto()
    DEVICE_ADDRESS_READ = auto()
    DEVICE_ADDRESS_WRITE = auto()
    
    # Additional ray tracing
    RAY_QUERY_WORLD_RAY = auto()
    RAY_QUERY_PROCEED = auto()
    RAY_QUERY_COMMITTED_HIT = auto()
    RAY_QUERY_CANDIDATE_TRIANGLE_HIT = auto()
    RAY_QUERY_CANDIDATE_PROCEDURAL_HIT = auto()
    RAY_QUERY_COMMIT_TRIANGLE = auto()
    RAY_QUERY_COMMIT_PROCEDURAL = auto()
    RAY_QUERY_TERMINATE = auto()
    ACCEL_INSTANCE_TRANSFORM = auto()
    ACCEL_INSTANCE_USER_ID = auto()
    ACCEL_INSTANCE_VISIBILITY_MASK = auto()
    
    # Additional operations
    ASSERT = auto()
    ASSUME = auto()
    CLOCK = auto()


# ============================================================================
# IR Values
# ============================================================================

@dataclass
class Value:
    """Base class for IR values."""
    typ: Type  # Renamed from 'type' to avoid builtin shadowing
    name: Optional[str] = None

    def __repr__(self) -> str:
        if self.name:
            return f"%{self.name}"
        return "%<unnamed>"

    @property
    def type(self) -> Type:
        """Backward compatibility accessor for type."""
        return self.typ


@dataclass
class ConstantValue(Value):
    """Constant value in IR."""
    value: Any = None

    def __post_init__(self):
        if self.name is None:
            self.name = str(self.value)

    def __repr__(self) -> str:
        return f"const {self.value}"


@dataclass
class ArgumentValue(Value):
    """Function argument value."""
    index: int = 0

    def __post_init__(self):
        if self.name is None:
            self.name = f"arg{self.index}"


@dataclass
class InstructionValue(Value):
    """Value produced by an instruction."""
    instruction: Optional[IRInstruction] = None

    def __repr__(self) -> str:
        if self.name:
            return f"%{self.name}"
        return f"%<inst>"


# Forward reference for IRInstruction
@dataclass
class IRInstruction:
    """IR instruction."""
    op: IROp
    typ: Type  # Renamed from 'type' to avoid builtin shadowing
    args: list[Union[int, str, Value, 'IRBasicBlock']] = field(default_factory=list)
    result: Optional[str] = None
    loc: Optional[SourceLocation] = None

    def __repr__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        if self.result:
            return f"%{self.result} = {self.op.name}({args_str})"
        return f"{self.op.name}({args_str})"

    @property
    def type(self) -> Type:
        """Backward compatibility accessor for type."""
        return self.typ


# Update the forward reference
InstructionValue.__annotations__['instruction'] = Optional[IRInstruction]


# ============================================================================
# IR Basic Blocks and Functions
# ============================================================================

@dataclass
class IRBasicBlock:
    """Basic block in IR."""
    name: str
    instructions: list[IRInstruction] = field(default_factory=list)
    loc: Optional[SourceLocation] = None
    
    def __repr__(self) -> str:
        lines = [f"{self.name}:"]
        for inst in self.instructions:
            lines.append(f"  {inst}")
        return "\n".join(lines)
    
    def is_terminated(self) -> bool:
        """Check if the block has a terminator instruction."""
        if not self.instructions:
            return False
        last = self.instructions[-1]
        return last.op in (
            IROp.RETURN, IROp.BREAK, IROp.CONTINUE
        )
    
    def add_instruction(self, inst: IRInstruction) -> None:
        """Add an instruction to the block."""
        if self.is_terminated():
            raise RuntimeError(f"Cannot add instruction to terminated block {self.name}")
        self.instructions.append(inst)


@dataclass
class IRFunction:
    """IR function."""
    name: str
    arg_types: list[Type]
    ret_type: Optional[Type]
    blocks: list[IRBasicBlock] = field(default_factory=list)
    is_kernel: bool = False
    block_size: Optional[tuple[int, int, int]] = None
    loc: Optional[SourceLocation] = None
    
    def __repr__(self) -> str:
        args_str = ", ".join(str(t) for t in self.arg_types)
        ret_str = str(self.ret_type) if self.ret_type else "void"
        kind = "kernel" if self.is_kernel else "callable"
        lines = [f"{kind} @{self.name}({args_str}) -> {ret_str} {{"]
        for block in self.blocks:
            lines.append(str(block))
        lines.append("}")
        return "\n".join(lines)
    
    def get_block(self, name: str) -> Optional[IRBasicBlock]:
        """Get a block by name."""
        for block in self.blocks:
            if block.name == name:
                return block
        return None


@dataclass
class IRModule:
    """IR module containing functions."""
    functions: list[IRFunction] = field(default_factory=list)
    constants: list[ConstantValue] = field(default_factory=list)
    
    def __repr__(self) -> str:
        lines = ["module {"]
        for func in self.functions:
            lines.append(str(func))
        lines.append("}")
        return "\n".join(lines)
    
    def add_function(self, func: IRFunction) -> None:
        """Add a function to the module."""
        self.functions.append(func)
    
    def get_function(self, name: str) -> Optional[IRFunction]:
        """Get a function by name."""
        for func in self.functions:
            if func.name == name:
                return func
        return None


# ============================================================================
# Utility Functions
# ============================================================================

def get_op_name(op: IROp) -> str:
    """Get the name of an IR operation."""
    return op.name


def is_arithmetic_op(op: IROp) -> bool:
    """Check if an operation is arithmetic."""
    return op in (
        IROp.ADD, IROp.SUB, IROp.MUL, IROp.DIV, IROp.MOD, IROp.NEG
    )


def is_comparison_op(op: IROp) -> bool:
    """Check if an operation is a comparison."""
    return op in (
        IROp.EQ, IROp.NE, IROp.LT, IROp.LE, IROp.GT, IROp.GE
    )


def is_logical_op(op: IROp) -> bool:
    """Check if an operation is logical."""
    return op in (
        IROp.LOGICAL_AND, IROp.LOGICAL_OR, IROp.LOGICAL_NOT
    )


def is_terminator_op(op: IROp) -> bool:
    """Check if an operation is a terminator."""
    return op in (IROp.RETURN, IROp.BREAK, IROp.CONTINUE)


def is_memory_op(op: IROp) -> bool:
    """Check if an operation is a memory operation."""
    return op in (
        IROp.ALLOCA, IROp.LOAD, IROp.STORE, IROp.GEP, IROp.MEMBER_ACCESS
    )


def is_resource_op(op: IROp) -> bool:
    """Check if an operation is a resource operation."""
    return op in (
        IROp.BUFFER_READ, IROp.BUFFER_WRITE, IROp.BUFFER_SIZE,
        IROp.TEXTURE2D_READ, IROp.TEXTURE2D_WRITE, 
        IROp.TEXTURE2D_SAMPLE, IROp.TEXTURE2D_SAMPLE_LEVEL,
        IROp.TEXTURE3D_READ, IROp.TEXTURE3D_WRITE, IROp.TEXTURE3D_SAMPLE,
    )
