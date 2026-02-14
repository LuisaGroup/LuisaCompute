import ast
import inspect
import copy
import sys
import builtins
from typing import Optional, Any, TYPE_CHECKING
from .ir import Value
from .parser import CapturedVar, ParsedFunction, annotation_to_type
from .builder import IRBuilder
from .types import Type, value_to_type, int32, Scalar, Vector, Buffer, Array

if TYPE_CHECKING:
    from .ir import IRFunction
    from .jit import StagedFunction

# ============================================================================
# Helper to find names in AST
# ============================================================================

class GlobalNameFinder(ast.NodeVisitor):
    """Find all names used in an AST that are not defined locally."""
    def __init__(self):
        self.used_names = set()
        self.defined_names = set()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.defined_names.add(node.id)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg):
        self.defined_names.add(node.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old_defined = self.defined_names.copy()
        self.defined_names.add(node.name)
        for arg in node.args.args:
            self.defined_names.add(arg.arg)
        if node.args.vararg:
            self.defined_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.defined_names.add(node.args.kwarg.arg)
        
        for stmt in node.body:
            self.visit(stmt)
            
        self.defined_names = old_defined

    def get_global_names(self):
        return self.used_names - self.defined_names


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
                 arg_values: tuple,
                 parent: Optional['BuilderExecutor'] = None):
        self.builder = builder
        self.parsed = parsed
        self.captured_vars = captured_vars
        self.arg_values = arg_values
        self.local_vars: dict[str, Any] = {}
        self.parent = parent
    
    def execute(self) -> None:
        """Execute the parsed function body."""
        from .jit import StagedFunction
        func_def = self.parsed.ast_node
        
        # Create entry block
        entry = self.builder.create_block("entry")
        self.builder.set_insert_point(entry)
        
        # Bind arguments
        for i, name in enumerate(self.parsed.arg_names):
            arg_val = self.builder.get_argument(i)
            self.local_vars[name] = arg_val
        
        # Bind captured variables
        for name, captured in self.captured_vars.items():
            val = captured.value
            # Don't turn Types, StagedFunctions, Modules, or Values into IR constants here.
            # We want to keep the raw Python value for constant folding/unrolling.
            self.local_vars[name] = val
        
        # Visit function body
        for stmt in func_def.body:
            if self.builder.current_block.is_terminated():
                break
            self.visit(stmt)
    
    def visit(self, node: ast.AST) -> Optional[Any]:
        """Visit an AST node."""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        """Visit a boolean operation (and/or)."""
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            result = values[0]
            for v in values[1:]:
                result = self.builder.logical_and(result, v)
            return result
        elif isinstance(node.op, ast.Or):
            result = values[0]
            for v in values[1:]:
                result = self.builder.logical_or(result, v)
            return result
        else:
            raise NotImplementedError(f"Unsupported boolean operator: {node.op}")

    def generic_visit(self, node: ast.AST) -> None:
        """Default visitor for unhandled nodes."""
        raise NotImplementedError(f"Unsupported AST node: {node.__class__.__name__}")
    
    def visit_Expr(self, node: ast.Expr) -> None:
        """Visit an expression statement."""
        self.visit(node.value)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit an assignment."""
        value = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_vars[target.id] = value
            elif isinstance(target, ast.Subscript):
                target_value = self.visit(target.value)
                index = self.visit(target.slice)
                if isinstance(target_value.type, (Buffer, Array)):
                    self.builder.buffer_write(target_value, index, value)
                else:
                    raise NotImplementedError(f"Subscript assignment not implemented for type: {target_value.type}")
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
        with if_.true_scope():
            for stmt in node.body:
                if self.builder.current_block.is_terminated():
                    break
                self.visit(stmt)
        if node.orelse:
            with if_.false_scope():
                for stmt in node.orelse:
                    if self.builder.current_block.is_terminated():
                        break
                    self.visit(stmt)
    
    def visit_While(self, node: ast.While) -> None:
        """Visit a while loop using structured WhileStmt."""
        cond = self.visit(node.test)
        while_ = self.builder.while_(cond)
        with while_.body_scope():
            for stmt in node.body:
                if self.builder.current_block.is_terminated():
                    break
                self.visit(stmt)
    
    def visit_For(self, node: ast.For) -> None:
        """Visit a for loop."""
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "unrolled":
                self._visit_for_unrolled(node)
                return
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                self._visit_for_range(node)
                return
        raise NotImplementedError(f"Unsupported for loop iterator: {node.iter}")
    
    def _visit_for_range(self, node: ast.For) -> None:
        """Visit a for-range loop (dynamic device-side loop)."""
        call = node.iter
        args = [self.visit(arg) for arg in call.args]
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
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Only simple loop variables supported")
        loop_var = node.target.id
        for_ = self.builder.for_range(start, stop, step, loop_var)
        with for_.body_scope():
            for stmt in node.body:
                self.visit(stmt)
    
    def _visit_for_unrolled(self, node: ast.For) -> None:
        """Visit an unrolled for loop (compile-time unrolling)."""
        unrolled_call = node.iter
        if not unrolled_call.args:
            raise ValueError("unrolled() requires a range argument")
        range_call = unrolled_call.args[0]
        if not isinstance(range_call, ast.Call):
            raise ValueError("unrolled() argument must be a range() call")
        range_args = range_call.args
        def eval_const(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                val = self._lookup_name(node.id)
                # For unrolling, we need the raw Python constant, not an IR Value
                if val is not None and not isinstance(val, Value):
                    return val
            raise ValueError(f"unrolled() requires constant arguments, got {ast.dump(node)}")
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
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("Only simple loop variables supported")
        loop_var = node.target.id
        for_ = self.builder.for_unrolled(start_val, stop_val, step_val, loop_var)
        for _ in for_.body_scope():
            for stmt in node.body:
                self.visit(stmt)
    
    def visit_Pass(self, _node: ast.Pass) -> None:
        """Visit a pass statement (no-op)."""
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition (nested callable)."""
        from .jit import callable as luisa_callable, StagedFunction
        import luisa
        is_callable = False
        callable_decorator_node = None
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == 'callable':
                is_callable = True
                callable_decorator_node = dec
                break
        if not is_callable:
            raise NotImplementedError("Nested functions must be decorated with @callable")
        node_no_dec = copy.deepcopy(node)
        node_no_dec.decorator_list = [d for d in node_no_dec.decorator_list 
                                     if not (isinstance(d, ast.Name) and d.id == 'callable')]
        try:
            func_source = ast.unparse(node_no_dec)
        except AttributeError:
            raise RuntimeError("ast.unparse required for nested callables")
        namespace = {}
        # Add captured vars from the current function
        namespace.update({name: var.value for name, var in self.captured_vars.items() 
                         if not isinstance(var.value, Value)})
        # Add local vars (like other nested callables)
        for name, val in self.local_vars.items():
            if not isinstance(val, Value):
                namespace[name] = val
        if self.parsed.pyfunc and hasattr(self.parsed.pyfunc, '__globals__'):
            for name, val in self.parsed.pyfunc.__globals__.items():
                if name not in namespace:
                    namespace[name] = val
        exec(func_source, namespace)
        pyfunc = namespace[node.name]
        arg_names = []
        arg_annotations = []
        sig = inspect.signature(pyfunc)
        for name, param in sig.parameters.items():
            arg_names.append(name)
            arg_annotations.append(annotation_to_type(param.annotation))
        ret_annotation = annotation_to_type(sig.return_annotation)
        finder = GlobalNameFinder()
        finder.visit(node)
        used_names = finder.get_global_names()
        captured_for_nested = {}
        for name in used_names:
            val = self._lookup_name(name)
            if val is not None:
                if isinstance(val, Value):
                    captured_for_nested[name] = CapturedVar(name=name, value=val, type=val.type)
                else:
                    captured_for_nested[name] = CapturedVar(name=name, value=val)
            elif name in namespace:
                val = namespace[name]
                captured_for_nested[name] = CapturedVar(name=name, value=val)
            elif name in self.captured_vars:
                captured_for_nested[name] = self.captured_vars[name]
        if 'callable' not in captured_for_nested:
            captured_for_nested['callable'] = CapturedVar(name='callable', value=luisa_callable)
        parsed_nested = ParsedFunction(
            name=node.name,
            ast_node=node,
            arg_names=arg_names,
            arg_annotations=arg_annotations,
            ret_annotation=ret_annotation,
            captured_vars=captured_for_nested,
            source=func_source,
            pyfunc=pyfunc
        )
        staged = StagedFunction(pyfunc, is_kernel=False, parsed=parsed_nested)
        self.local_vars[node.name] = staged

    def _lookup_name(self, name: str) -> Any:
        """Look up a name in local vars, captured vars, and parents."""
        if name in self.local_vars:
            return self.local_vars[name]
        if name in self.captured_vars:
            return self.captured_vars[name].value
        if self.parent:
            return self.parent._lookup_name(name)
        return None

    def visit_Break(self, _node: ast.Break) -> None:
        """Visit a break statement."""
        self.builder.break_()
    
    def visit_Continue(self, _node: ast.Continue) -> None:
        """Visit a continue statement."""
        self.builder.continue_()
    
    def visit_Match(self, node: ast.Match) -> None:
        """Visit a match statement (Python 3.10+)."""
        subject = self.visit(node.subject)
        switch = self.builder.switch(subject)
        for case in node.cases:
            if isinstance(case.pattern, ast.MatchValue):
                case_value = case.pattern.value.value
                with switch.case_scope(case_value):
                    for stmt in case.body:
                        self.visit(stmt)
            elif isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None:
                with switch.default_scope():
                    for stmt in case.body:
                        self.visit(stmt)
            else:
                raise NotImplementedError(f"Unsupported case pattern: {type(case.pattern)}")
    
    def visit_Constant(self, node: ast.Constant) -> Any:
        """Visit a constant."""
        value = node.value
        if value is None or isinstance(value, str):
            return None
        typ = value_to_type(value)
        if typ is None:
            raise ValueError(f"Unsupported constant type: {type(value)}")
        return self.builder.constant(typ, value)
    
    def visit_Name(self, node: ast.Name) -> Any:
        """Visit a name reference."""
        name = node.id
        val = self._lookup_name(name)
        if val is not None:
            if isinstance(val, (Value, Type, type(lambda:0), type(sys), type(inspect))):
                return val
            # Python literals (int, float, bool) should be turned into IR constants
            # ONLY when they are being used as expressions in the DSL.
            typ = value_to_type(val)
            if typ is not None:
                return self.builder.constant(typ, val)
            return val
        from .jit import callable as luisa_callable, kernel as luisa_kernel
        if name == "callable": return luisa_callable
        if name == "kernel": return luisa_kernel
        builder_val = self.builder.lookup_local(name)
        if builder_val is not None: return builder_val
        if self.parsed.pyfunc and hasattr(self.parsed.pyfunc, '__globals__'):
            if name in self.parsed.pyfunc.__globals__:
                val = self.parsed.pyfunc.__globals__[name]
                if isinstance(val, (Type, type(lambda:0), type(sys))): return val
                typ = value_to_type(val)
                if typ is not None: return self.builder.constant(typ, val)
                return val
        if 'luisa' in sys.modules:
            luisa = sys.modules['luisa']
            if hasattr(luisa, name): return getattr(luisa, name)
        raise NameError(f"Undefined variable: {name}")
    
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Visit a binary operation."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add): return self.builder.add(left, right)
        if isinstance(node.op, ast.Sub): return self.builder.sub(left, right)
        if isinstance(node.op, ast.Mult): return self.builder.mul(left, right)
        if isinstance(node.op, ast.Div): return self.builder.div(left, right)
        if isinstance(node.op, ast.Mod): return self.builder.mod(left, right)
        if isinstance(node.op, ast.Pow): return self.builder.pow(left, right)
        if isinstance(node.op, ast.FloorDiv): return self.builder.floor(self.builder.div(left, right))
        if isinstance(node.op, ast.BitAnd): return self.builder.bit_and(left, right)
        if isinstance(node.op, ast.BitOr): return self.builder.bit_or(left, right)
        if isinstance(node.op, ast.BitXor): return self.builder.bit_xor(left, right)
        if isinstance(node.op, ast.LShift): return self.builder.shl(left, right)
        if isinstance(node.op, ast.RShift): return self.builder.shr(left, right)
        if isinstance(node.op, ast.Eq): return self.builder.eq(left, right)
        if isinstance(node.op, ast.NotEq): return self.builder.ne(left, right)
        if isinstance(node.op, ast.Lt): return self.builder.lt(left, right)
        if isinstance(node.op, ast.LtE): return self.builder.le(left, right)
        if isinstance(node.op, ast.Gt): return self.builder.gt(left, right)
        if isinstance(node.op, ast.GtE): return self.builder.ge(left, right)
        raise NotImplementedError(f"Unsupported binary operator: {node.op}")
    
    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Visit a unary operation."""
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub): return self.builder.neg(operand)
        if isinstance(node.op, ast.Not): return self.builder.logical_not(operand)
        raise NotImplementedError(f"Unsupported unary operator: {node.op}")
    
    def visit_Compare(self, node: ast.Compare) -> Any:
        """Visit a comparison."""
        left = self.visit(node.left)
        if len(node.ops) == 1:
            right = self.visit(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Eq): return self.builder.eq(left, right)
            if isinstance(op, ast.NotEq): return self.builder.ne(left, right)
            if isinstance(op, ast.Lt): return self.builder.lt(left, right)
            if isinstance(op, ast.LtE): return self.builder.le(left, right)
            if isinstance(op, ast.Gt): return self.builder.gt(left, right)
            if isinstance(op, ast.GtE): return self.builder.ge(left, right)
        raise NotImplementedError(f"Chained comparisons not fully supported")
    
    def visit_Call(self, node: ast.Call) -> Any:
        """Visit a function call."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            try:
                func_val = self.visit_Name(node.func)
            except NameError:
                func_val = None
            if func_val is not None:
                if isinstance(func_val, Type):
                    args = [self.visit(arg) for arg in node.args]
                    if len(args) == 1: return self.builder.cast(args[0], func_val)
                    raise ValueError(f"Type cast takes exactly one argument, got {len(args)}")
                from .jit import StagedFunction
                if isinstance(func_val, StagedFunction):
                    args = [self.visit(arg) for arg in node.args]
                    arg_types = tuple(a.type for a in args)
                    if arg_types not in func_val._cache:
                        from .builtins.math import set_builder as set_math_builder
                        set_math_builder(self.builder)
                        try:
                            callee_ir = func_val(*args, parent_executor=self)
                        finally:
                            set_math_builder(None)
                        func_val._cache[arg_types] = callee_ir
                    return self.builder.call(func_val._cache[arg_types], args)
                if builtins.callable(func_val) and not isinstance(func_val, Value):
                    args = [self.visit(arg) for arg in node.args]
                    return func_val(*args)
            raise NotImplementedError(f"Unknown function call: {func_name}. Make sure to import it from luisa.")
        raise NotImplementedError(f"Unsupported function call: {node.func}")
    
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Visit an attribute access (e.g., obj.attr)."""
        value = self.visit(node.value)
        attr = node.attr
        from .types import Vector
        if isinstance(value.type, Vector):
            valid_swizzle_chars = set('xyzwrgba0123')
            if all(c in valid_swizzle_chars for c in attr):
                return self.builder.swizzle(value, attr)
        raise NotImplementedError(f"Attribute access not yet implemented: {attr}")
    
    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Visit a subscript operation (e.g., a[i])."""
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if isinstance(value.type, (Buffer, Array)):
            return self.builder.buffer_read(value, index, value.type.element)
        raise NotImplementedError(f"Subscript not yet implemented for type: {value.type}")
