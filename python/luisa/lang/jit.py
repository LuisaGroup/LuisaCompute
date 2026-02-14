"""
Staged function wrapper for the LuisaCompute Python DSL v2.

This module implements the multistage programming system:
- Stage 1: Parse the Python function
- Stage 2: When called, execute the builder to generate IR
"""

from __future__ import annotations
import ast
from typing import Callable, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .types import Type
    from .ast import IRFunction, Value
    from .parser import ParsedFunction

from .types import Type, value_to_type, int32
from .ast import IRFunction
from .builder import IRBuilder
from .parser import parse_function, CapturedVar
from ..util import UnrolledRange


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
    
    def __init__(self, func: Callable, is_kernel: bool = False):
        self.pyfunc = func
        self.is_kernel = is_kernel
        
        # Stage 1: Parse the function
        self.parsed = parse_function(func)
        
        # Cache for compiled versions (keyed by argument types)
        self._cache: dict[tuple[Type, ...], IRFunction] = {}
    
    @property
    def name(self) -> str:
        return self.parsed.name
    
    def __call__(self, *args, **kwargs) -> IRFunction:
        """
        Execute the staged function.
        
        This performs Stage 3 of multistage programming:
        - Create builder
        - Execute builder to generate IR
        - Return the generated IR function
        """
        # Get actual argument types from runtime values
        arg_types = tuple(self._get_arg_type(arg) for arg in args)
        
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
        executor = BuilderExecutor(
            builder=builder,
            parsed=self.parsed,
            captured_vars=self.parsed.captured_vars,
            arg_values=args
        )
        executor.execute()
        
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
# Builder Executor
# ============================================================================

class BuilderExecutor:
    """
    Execute the Python AST using the IR builder.
    
    This interprets the Python AST and calls the appropriate builder methods.
    It's essentially a Python interpreter that generates IR instead of executing.
    """
    
    def __init__(self, builder: IRBuilder, 
                 parsed: ParsedFunction,
                 captured_vars: dict[str, CapturedVar],
                 arg_values: tuple):
        self.builder = builder
        self.parsed = parsed
        self.captured_vars = captured_vars
        self.arg_values = arg_values
        self.local_vars: dict[str, Value] = {}
    
    def execute(self) -> None:
        """Execute the parsed function body."""
        func_def = self.parsed.ast_node
        
        # Create entry block
        entry = self.builder.create_block("entry")
        self.builder.set_insert_point(entry)
        
        # Bind arguments
        for i, (name, value) in enumerate(zip(self.parsed.arg_names, self.arg_values)):
            arg_val = self.builder.get_argument(i)
            self.local_vars[name] = arg_val
        
        # Bind captured variables
        for name, captured in self.captured_vars.items():
            const_val = self.builder.constant(captured.type, captured.value)
            self.local_vars[name] = const_val
        
        # Visit function body
        for stmt in func_def.body:
            self.visit(stmt)
    
    def visit(self, node: ast.AST) -> Optional[Any]:
        """Visit an AST node."""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node: ast.AST) -> None:
        """Default visitor for unhandled nodes."""
        raise NotImplementedError(f"Unsupported AST node: {node.__class__.__name__}")
    
    # ========================================================================
    # Statements
    # ========================================================================
    
    def visit_Expr(self, node: ast.Expr) -> None:
        """Visit an expression statement."""
        self.visit(node.value)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit an assignment."""
        value = self.visit(node.value)
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_vars[target.id] = value
            else:
                raise NotImplementedError(f"Unsupported assignment target: {target}")
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit an annotated assignment."""
        value = self.visit(node.value)
        
        if isinstance(node.target, ast.Name):
            self.local_vars[node.target.id] = value
        else:
            raise NotImplementedError(f"Unsupported annotated assignment target: {node.target}")
    
    def visit_Return(self, node: ast.Return) -> None:
        """Visit a return statement."""
        if node.value is None:
            self.builder.return_(None)
        else:
            value = self.visit(node.value)
            self.builder.return_(value)
    
    def visit_If(self, node: ast.If) -> None:
        """Visit an if statement using structured IfStmt."""
        cond = self.visit(node.test)
        
        if_ = self.builder.if_(cond)
        
        # True branch
        with if_.true_scope():
            for stmt in node.body:
                self.visit(stmt)
        
        # False branch (if exists)
        if node.orelse:
            with if_.false_scope():
                for stmt in node.orelse:
                    self.visit(stmt)
    
    def visit_While(self, node: ast.While) -> None:
        """Visit a while loop using structured WhileStmt."""
        cond = self.visit(node.test)
        
        while_ = self.builder.while_(cond)
        with while_.body_scope():
            for stmt in node.body:
                self.visit(stmt)
    
    def visit_For(self, node: ast.For) -> None:
        """Visit a for loop."""
        # Check if this is an unrolled loop
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "unrolled":
                self._visit_for_unrolled(node)
                return
        
        # Check if this is a range() loop
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                self._visit_for_range(node)
                return
        
        raise NotImplementedError(f"Unsupported for loop iterator: {node.iter}")
    
    def _visit_for_range(self, node: ast.For) -> None:
        """Visit a for-range loop (dynamic device-side loop)."""
        # Get range arguments
        call = node.iter
        args = [self.visit(arg) for arg in call.args]
        
        # Determine start, stop, step
        if len(args) == 1:
            start = self.builder.constant(int32, 0)
            stop = args[0]
            step = self.builder.constant(int32, 1)
        elif len(args) == 2:
            start = args[0]
            stop = args[1]
            step = self.builder.constant(int32, 1)
        elif len(args) == 3:
            start = args[0]
            stop = args[1]
            step = args[2]
        else:
            raise ValueError("range() takes 1-3 arguments")
        
        # Get loop variable name
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Only simple loop variables supported")
        loop_var = node.target.id
        
        # Use dynamic for-range scope
        for_ = self.builder.for_range(start, stop, step, loop_var)
        with for_.body_scope():
            for stmt in node.body:
                self.visit(stmt)
    
    def _visit_for_unrolled(self, node: ast.For) -> None:
        """Visit an unrolled for loop (compile-time unrolling)."""
        # Get unrolled arguments - unrolled(range(...))
        unrolled_call = node.iter
        if not unrolled_call.args:
            raise ValueError("unrolled() requires a range argument")
        
        range_call = unrolled_call.args[0]
        if not isinstance(range_call, ast.Call):
            raise ValueError("unrolled() argument must be a range() call")
        
        # Get range arguments - these must be constants for unrolling
        range_args = range_call.args
        
        # Evaluate constant range arguments
        def eval_const(node):
            if isinstance(node, ast.Constant):
                return node.value
            raise ValueError(f"unrolled() requires constant arguments, got {ast.dump(node)}")
        
        # Determine start, stop, step
        if len(range_args) == 1:
            start_val = 0
            stop_val = eval_const(range_args[0])
            step_val = 1
        elif len(range_args) == 2:
            start_val = eval_const(range_args[0])
            stop_val = eval_const(range_args[1])
            step_val = 1
        elif len(range_args) == 3:
            start_val = eval_const(range_args[0])
            stop_val = eval_const(range_args[1])
            step_val = eval_const(range_args[2])
        else:
            raise ValueError("range() takes 1-3 arguments")
        
        # Get loop variable name
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Only simple loop variables supported")
        loop_var = node.target.id
        
        # Use unrolled for-range scope
        for_ = self.builder.for_unrolled(start_val, stop_val, step_val, loop_var)
        for _ in for_.body_scope():
            # Execute body for each unrolled iteration
            for stmt in node.body:
                self.visit(stmt)
    
    def visit_Pass(self, node: ast.Pass) -> None:
        """Visit a pass statement (no-op)."""
        pass
    
    def visit_Break(self, node: ast.Break) -> None:
        """Visit a break statement."""
        self.builder.break_()
    
    def visit_Continue(self, node: ast.Continue) -> None:
        """Visit a continue statement."""
        self.builder.continue_()
    
    def visit_Match(self, node: ast.Match) -> None:
        """Visit a match statement (Python 3.10+)."""
        # Evaluate the subject
        subject = self.visit(node.subject)
        
        # Use structured switch
        switch = self.builder.switch(subject)
        
        for case in node.cases:
            # Handle case pattern
            if isinstance(case.pattern, ast.MatchValue):
                # Single value case
                case_value = case.pattern.value.value
                with switch.case_scope(case_value):
                    for stmt in case.body:
                        self.visit(stmt)
            
            elif isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None:
                # Default case (case _)
                with switch.default_scope():
                    for stmt in case.body:
                        self.visit(stmt)
            
            else:
                raise NotImplementedError(f"Unsupported case pattern: {type(case.pattern)}")
    
    # ========================================================================
    # Expressions
    # ========================================================================
    
    def visit_Constant(self, node: ast.Constant) -> Any:
        """Visit a constant."""
        value = node.value
        typ = value_to_type(value)
        if typ is None:
            raise ValueError(f"Unsupported constant type: {type(value)}")
        return self.builder.constant(typ, value)
    
    def visit_Name(self, node: ast.Name) -> Any:
        """Visit a name reference."""
        name = node.id
        
        if name in self.local_vars:
            return self.local_vars[name]
        
        raise NameError(f"Undefined variable: {name}")
    
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Visit a binary operation."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        if isinstance(node.op, ast.Add):
            return self.builder.add(left, right)
        elif isinstance(node.op, ast.Sub):
            return self.builder.sub(left, right)
        elif isinstance(node.op, ast.Mult):
            return self.builder.mul(left, right)
        elif isinstance(node.op, ast.Div):
            return self.builder.div(left, right)
        elif isinstance(node.op, ast.Mod):
            return self.builder.mod(left, right)
        elif isinstance(node.op, ast.Pow):
            return self.builder.pow(left, right)
        elif isinstance(node.op, ast.FloorDiv):
            # Floor division: convert to floor(a / b)
            div_result = self.builder.div(left, right)
            return self.builder.floor(div_result)
        elif isinstance(node.op, ast.BitAnd):
            return self.builder.bit_and(left, right)
        elif isinstance(node.op, ast.BitOr):
            return self.builder.bit_or(left, right)
        elif isinstance(node.op, ast.BitXor):
            return self.builder.bit_xor(left, right)
        elif isinstance(node.op, ast.LShift):
            return self.builder.shl(left, right)
        elif isinstance(node.op, ast.RShift):
            return self.builder.shr(left, right)
        elif isinstance(node.op, ast.Eq):
            return self.builder.eq(left, right)
        elif isinstance(node.op, ast.NotEq):
            return self.builder.ne(left, right)
        elif isinstance(node.op, ast.Lt):
            return self.builder.lt(left, right)
        elif isinstance(node.op, ast.LtE):
            return self.builder.le(left, right)
        elif isinstance(node.op, ast.Gt):
            return self.builder.gt(left, right)
        elif isinstance(node.op, ast.GtE):
            return self.builder.ge(left, right)
        else:
            raise NotImplementedError(f"Unsupported binary operator: {node.op}")
    
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Visit a unary operation."""
        operand = self.visit(node.operand)
        
        if isinstance(node.op, ast.USub):
            return self.builder.neg(operand)
        elif isinstance(node.op, ast.Not):
            return self.builder.logical_not(operand)
        else:
            raise NotImplementedError(f"Unsupported unary operator: {node.op}")
    
    def visit_Compare(self, node: ast.Compare) -> Any:
        """Visit a comparison."""
        left = self.visit(node.left)
        
        # Handle single comparison
        if len(node.ops) == 1:
            right = self.visit(node.comparators[0])
            op = node.ops[0]
            
            if isinstance(op, ast.Eq):
                return self.builder.eq(left, right)
            elif isinstance(op, ast.NotEq):
                return self.builder.ne(left, right)
            elif isinstance(op, ast.Lt):
                return self.builder.lt(left, right)
            elif isinstance(op, ast.LtE):
                return self.builder.le(left, right)
            elif isinstance(op, ast.Gt):
                return self.builder.gt(left, right)
            elif isinstance(op, ast.GtE):
                return self.builder.ge(left, right)
            else:
                raise NotImplementedError(f"Unsupported comparison: {op}")
        
        # Handle chained comparisons (a < b < c)
        # For simplicity, just do the first one
        right = self.visit(node.comparators[0])
        op = node.ops[0]
        
        if isinstance(op, ast.Lt):
            return self.builder.lt(left, right)
        elif isinstance(op, ast.LtE):
            return self.builder.le(left, right)
        elif isinstance(op, ast.Gt):
            return self.builder.gt(left, right)
        elif isinstance(op, ast.GtE):
            return self.builder.ge(left, right)
        elif isinstance(op, ast.Eq):
            return self.builder.eq(left, right)
        elif isinstance(op, ast.NotEq):
            return self.builder.ne(left, right)
        else:
            raise NotImplementedError(f"Unsupported comparison: {op}")
    
    def visit_Call(self, node: ast.Call) -> Any:
        """Visit a function call."""
        # Get function
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Handle built-in functions - they need to be imported
            # For now, raise a more informative error
            raise NotImplementedError(
                f"Direct function calls not yet implemented: {func_name}. "
                f"Use 'from luisa import {func_name}' and ensure it's a staged builtin."
            )
        
        raise NotImplementedError(f"Unsupported function call: {node.func}")
    
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Visit an attribute access (e.g., obj.attr)."""
        value = self.visit(node.value)
        attr = node.attr
        
        # Check for vector swizzle pattern
        from .types import Vector
        if isinstance(value.type, Vector):
            # Check if attr is a valid swizzle
            valid_swizzle_chars = set('xyzwrgba0123')
            if all(c in valid_swizzle_chars for c in attr):
                # Return a swizzle operation
                return self.builder.swizzle(value, attr)
        
        # Regular attribute access (for struct fields, etc.)
        raise NotImplementedError(f"Attribute access not yet implemented: {attr}")
    
    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Visit a subscript operation (e.g., a[i])."""
        value = self.visit(node.value)
        index = self.visit(node.slice)
        
        # Handle buffer/array indexing
        from .types import Buffer, Array
        if isinstance(value.type, (Buffer, Array)):
            # Emit a buffer read
            return self.builder.buffer_read(value, index, value.type.element)
        
        raise NotImplementedError(f"Subscript not yet implemented for type: {value.type}")


# ============================================================================
# Decorators
# ============================================================================

def kernel(func: Callable) -> StagedFunction:
    """Decorator to mark a function as a kernel."""
    return StagedFunction(func, is_kernel=True)


def callable(func: Callable) -> StagedFunction:
    """Decorator to mark a function as a callable device function."""
    return StagedFunction(func, is_kernel=False)
