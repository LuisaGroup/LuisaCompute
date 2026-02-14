"""
Staged function wrapper for the LuisaCompute Python DSL v2.

This module implements the multistage programming system:
- Stage 1: Parse the Python function
- Stage 2: When called, execute the builder to generate IR
"""

from __future__ import annotations
import ast
from typing import Callable, Optional, Any, TYPE_CHECKING


if TYPE_CHECKING:
    from .types import Type
    from .ir import IRFunction, Value
    from .parser import ParsedFunction

from .types import Type, value_to_type, int32
from .ir import IRFunction
from .builder import IRBuilder
from .parser import parse_function, CapturedVar
from .builder_executor import BuilderExecutor

from .builtins.math import set_builder as set_math_builder


# ============================================================================
# Staged Function
# ============================================================================

class StagedFunction:
    """
    A staged function that generates IR when called.
    
    This is the core of multistage programming - the actual IR generation
    is deferred until the function is called with specific argument types.
    
    When called, it:
    1. Determines actual argument types
    2. Creates an IR builder
    3. Executes the builder to generate IR (Stage 3)
    4. Returns/caches the compiled function
    """
    
    def __init__(self, func: Callable, is_kernel: bool = False, parsed: Optional[ParsedFunction] = None):
        self.pyfunc = func
        self.is_kernel = is_kernel
        
        # Stage 1: Parse the function
        if parsed is not None:
            self.parsed = parsed
        else:
            self.parsed = parse_function(func)
        
        # Cache for compiled versions (keyed by argument types)
        self._cache: dict[tuple[Type, ...], IRFunction] = {}
    
    @property
    def name(self) -> str:
        return self.parsed.name
    
    def __call__(self, *args, parent_executor: Optional[BuilderExecutor] = None, **kwargs) -> IRFunction:
        """
        Execute the staged function.
        
        This performs Stage 3 of multistage programming:
        - Create builder
        - Execute builder to generate IR
        - Return the generated IR function
        """
        # Get argument types - prefer annotations, fall back to runtime types
        arg_types = []
        for i, arg in enumerate(args):
            # First check if we have a type annotation
            if i < len(self.parsed.arg_annotations) and self.parsed.arg_annotations[i] is not None:
                arg_types.append(self.parsed.arg_annotations[i])
            else:
                # Fall back to inferring from runtime value
                arg_types.append(self._get_arg_type(arg))
        arg_types = tuple(arg_types)
        
        # Check cache
        if arg_types in self._cache:
            return self._cache[arg_types]
        
        # Create builder context (Stage 3 starts here)
        builder = IRBuilder(
            name=self.parsed.name,
            arg_types=arg_types,
            ret_type=self.parsed.ret_annotation
        )
        
        # Execute the builder (this is where the magic happens)
        # Set the builder context for builtins
        set_math_builder(builder)
        try:
            executor = BuilderExecutor(
                builder=builder,
                parsed=self.parsed,
                captured_vars=self.parsed.captured_vars,
                arg_values=args,
                parent=parent_executor
            )
            executor.execute()
        finally:
            # Clear the builder context
            set_math_builder(None)
        
        # Get the generated IR
        ir_func = builder.build()
        ir_func.is_kernel = self.is_kernel
        
        # Cache and return
        self._cache[arg_types] = ir_func
        return ir_func
    
    def _get_arg_type(self, arg: Any) -> Type:
        """Get the DSL type of a runtime argument."""
        # Try to infer type from value
        inferred = value_to_type(arg)
        if inferred is not None:
            return inferred
        
        # Default to int32 for unknown types
        # In a full implementation, we'd have type annotations on the values
        return arg.type if hasattr(arg, 'type') else arg.__class__.__name__


# ============================================================================
# Decorators
# ============================================================================

def kernel(func: Callable) -> StagedFunction:
    """Decorator to mark a function as a kernel."""
    return StagedFunction(func, is_kernel=True)


def callable(func: Callable) -> StagedFunction:
    """Decorator to mark a function as a callable device function."""
    return StagedFunction(func, is_kernel=False)
