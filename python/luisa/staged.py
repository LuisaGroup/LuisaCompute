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
    from .dsl_types import Type
    from .ir import IRFunction, Value
    from .parser import ParsedFunction

from .dsl_types import Type, value_to_type, int32
from .ir import IRFunction
from .builder import IRBuilder
from .parser import parse_function, CapturedVar


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
        """Visit an if statement using IfScope."""
        cond = self.visit(node.test)
        
        with self.builder.if_(cond) as if_scope:
            # True branch
            for stmt in node.body:
                self.visit(stmt)
            
            # False branch (if exists)
            if node.orelse:
                with if_scope.otherwise():
                    for stmt in node.orelse:
                        self.visit(stmt)
    
    def visit_While(self, node: ast.While) -> None:
        """Visit a while loop using WhileScope."""
        # Note: For proper while loops, we need to handle the condition
        # evaluation at the header block. This is a simplified version.
        
        # For now, only handle constant conditions
        cond = self.visit(node.test)
        
        with self.builder.while_(cond):
            for stmt in node.body:
                self.visit(stmt)
    
    def visit_For(self, node: ast.For) -> None:
        """Visit a for loop."""
        # Check if this is a range() loop
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                self._visit_for_range(node)
                return
        
        raise NotImplementedError(f"Unsupported for loop iterator: {node.iter}")
    
    def _visit_for_range(self, node: ast.For) -> None:
        """Visit a for-range loop."""
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
        with self.builder.for_range(start, stop, step, loop_var):
            for stmt in node.body:
                self.visit(stmt)
    
    def visit_Pass(self, node: ast.Pass) -> None:
        """Visit a pass statement (no-op)."""
        pass
    
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
            
            # Handle built-in functions
            # This would be extended with a registry of built-in functions
            raise NotImplementedError(f"Function calls not yet implemented: {func_name}")
        
        raise NotImplementedError(f"Unsupported function call: {node.func}")
    
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Visit an attribute access (e.g., obj.attr)."""
        value = self.visit(node.value)
        attr = node.attr
        
        # Handle vector swizzles
        # This would check if value is a vector and attr is a swizzle pattern
        raise NotImplementedError(f"Attribute access not yet implemented: {attr}")
    
    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Visit a subscript operation (e.g., a[i])."""
        value = self.visit(node.value)
        index = self.visit(node.slice)
        
        # This would emit a load instruction
        raise NotImplementedError("Subscript not yet implemented")


# ============================================================================
# Decorators
# ============================================================================

def kernel(func: Callable) -> StagedFunction:
    """Decorator to mark a function as a kernel."""
    return StagedFunction(func, is_kernel=True)


def callable(func: Callable) -> StagedFunction:
    """Decorator to mark a function as a callable device function."""
    return StagedFunction(func, is_kernel=False)
