"""
AST Rewriter for the LuisaCompute Python DSL v2.

This module provides AST transformation and metadata extraction logic
to turn Python functions into IR-building functions.
"""

from __future__ import annotations

import ast
from typing import Any, Optional

# ============================================================================
# AST Transformation
# ============================================================================

class ASTRewriter(ast.NodeTransformer):
    """
    Transforms Python AST into IR-building code.

    Example:
        a + b  =>  binop(ast.Add(), a, b)
    """

    def __init__(self, file: str = "<unknown>"):
        self.file = file
        self._in_loop = 0
        self.rt_alias = "__luisa_rt"
        self.ref_vars = set()  # Variables that are of Ref type
        self._is_top_level = True
        self.const_vars = set()  # Variables marked with @const or const()
        self.dsl_vars = set()    # Variables that should be DSL variables (alloca)
        self.arg_names = set()   # Function argument names

    def rewrite(self, node: ast.AST) -> ast.AST:
        """Entry point for rewriting."""
        self._is_top_level = True
        return self.visit(node)

    def _set_loc(self, node: ast.AST) -> Optional[ast.Expr]:
        """Create a call to set_location based on node lineno."""
        if hasattr(node, 'lineno'):
            return ast.Expr(value=self._rt_call(
                "set_location",
                ast.Constant(value=self.file),
                ast.Constant(value=node.lineno)
            ))
        return None

    def visit_Name(self, node: ast.Name) -> Any:
        """Handle names, handling Ref loads and DSL variable loads."""
        if isinstance(node.ctx, ast.Load):
            if node.id in self.ref_vars:
                # Automatic load for Ref types: load(name)
                return self._rt_call("load", node)
            elif node.id in self.dsl_vars and node.id not in self.const_vars:
                # Load from DSL variable: load(name)
                # Use maybe_load which handles the case where the variable
                # might still be a Python value (not yet converted to DSL)
                return self._rt_call("maybe_load", ast.Name(id=node.id, ctx=ast.Load()))

        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Rewrite function definition."""
        is_top = self._is_top_level
        self._is_top_level = False

        # Nested functions: rewrite body for DSL
        # Note: nested @callable/@kernel functions will have their source available
        # via inspect.getsourcelines() because we populate linecache with the
        # rewritten source before exec()
        if not is_top:
            return self._rewrite_nested_function(node)

        # Top-level function: mangle for IR building
        # Detect Ref arguments
        for arg in node.args.args:
            self.arg_names.add(arg.arg)
            # Check if annotation is Ref[...]
            ann = arg.annotation
            if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name) and ann.value.id == 'Ref':
                self.ref_vars.add(arg.arg)
            # Handle from luisa import Ref; a: Ref[Int]
            elif isinstance(ann, ast.Name) and ann.id == 'Ref':
                self.ref_vars.add(arg.arg)

        # New arguments: original arguments (builder is now global context)
        new_args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=a.arg, annotation=None) for a in node.args.args],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        )

        # Rewrite body
        old_ref_vars = self.ref_vars.copy()
        old_const_vars = self.const_vars.copy()
        old_dsl_vars = self.dsl_vars.copy()
        new_body = []
        for stmt in node.body:
            loc_call = self._set_loc(stmt)
            if loc_call:
                new_body.append(loc_call)

            rewritten = self.visit(stmt)
            if isinstance(rewritten, list):
                new_body.extend(rewritten)
            else:
                new_body.append(rewritten)
        self.ref_vars = old_ref_vars
        self.const_vars = old_const_vars
        self.dsl_vars = old_dsl_vars

        return ast.FunctionDef(
            name=f"{node.name}_",  # Add underscore suffix to avoid shadowing injected functions
            args=new_args,
            body=new_body,
            decorator_list=[],  # Remove decorators
            returns=None
        )

    def _rt_call(self, name: str, *args: ast.expr) -> ast.Call:
        """Helper to create a call to a runtime function.
        
        The function is referenced directly by name (e.g., 'add', 'store').
        The actual functions are injected into the execution namespace.
        """
        return ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()),
            args=list(args),
            keywords=[]
        )

    def _rewrite_nested_function(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Rewrite regular nested function body for DSL operations."""
        old_ref_vars = self.ref_vars.copy()
        new_node = self.generic_visit(node)
        self.ref_vars = old_ref_vars
        return new_node

    # Map AST operator types to direct function names
    _BINOP_MAP = {
        'Add': 'add',
        'Sub': 'sub',
        'Mult': 'mul',
        'Div': 'div',
        'Mod': 'mod',
        'Pow': 'pow',
        'FloorDiv': 'floordiv',
        'BitAnd': 'bitand',
        'BitOr': 'bitor',
        'BitXor': 'bitxor',
        'LShift': 'lshift',
        'RShift': 'rshift',
        'MatMult': 'matmul',
    }

    def visit_BinOp(self, node: ast.BinOp) -> ast.Call:
        """Rewrite binary operations to direct function calls."""
        op_name = node.op.__class__.__name__
        func_name = self._BINOP_MAP.get(op_name, 'binop')
        
        # For unknown operators, fall back to binop with ast operator
        if func_name == 'binop':
            return self._rt_call(
                "binop",
                ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()),
                         args=[], keywords=[]),
                self.visit(node.left),
                self.visit(node.right)
            )
        
        # Use direct function call: add(a, b) instead of binop(ast.Add(), a, b)
        return self._rt_call(
            func_name,
            self.visit(node.left),
            self.visit(node.right)
        )

    # Map AST unary operator types to direct function names
    _UNARYOP_MAP = {
        'USub': 'neg',
        'Not': 'logical_not',
        'Invert': 'bit_not',
    }

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.Call:
        """Rewrite unary operations to direct function calls."""
        op_name = node.op.__class__.__name__
        func_name = self._UNARYOP_MAP.get(op_name, 'unaryop')
        
        # For unknown operators, fall back to unaryop with ast operator
        if func_name == 'unaryop':
            return self._rt_call(
                "unaryop",
                ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()),
                         args=[], keywords=[]),
                self.visit(node.operand)
            )
        
        # Use direct function call: neg(a) instead of unaryop(ast.USub(), a)
        return self._rt_call(
            func_name,
            self.visit(node.operand)
        )

    # Map AST comparison operator types to direct function names
    _COMPARE_MAP = {
        'Eq': 'eq',
        'NotEq': 'ne',
        'Lt': 'lt',
        'LtE': 'le',
        'Gt': 'gt',
        'GtE': 'ge',
    }

    def visit_Compare(self, node: ast.Compare) -> Any:
        """Rewrite comparison operations, including chained comparisons."""
        if len(node.ops) == 1:
            op_name = node.ops[0].__class__.__name__
            func_name = self._COMPARE_MAP.get(op_name, 'compare')
            
            # For unknown operators, fall back to compare with ast operator
            if func_name == 'compare':
                return self._rt_call(
                    "compare",
                    ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()),
                             args=[], keywords=[]),
                    self.visit(node.left),
                    self.visit(node.comparators[0])
                )
            
            # Use direct function call: eq(a, b) instead of compare(ast.Eq(), a, b)
            return self._rt_call(
                func_name,
                self.visit(node.left),
                self.visit(node.comparators[0])
            )

        # Chained comparisons: a < b < c => (a < b) and (b < c)
        comparisons = []
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            # Create a simple comparison node
            comp = ast.Compare(
                left=left,
                ops=[op],
                comparators=[right]
            )
            comparisons.append(comp)
            left = right

        # Use the same logic as visit_BoolOp but without double-visiting
        return self._make_short_circuit_op("and_", comparisons)

    def _make_short_circuit_op(self, op_name: str, exprs: list[ast.expr]) -> ast.expr:
        """Helper to create nested short-circuiting calls."""
        if len(exprs) == 1:
            return self.visit(exprs[0])

        return self._rt_call(
            op_name,
            ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                       body=self.visit(exprs[0])),
            ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                       body=self._make_short_circuit_op(op_name, exprs[1:]))
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.Call:
        """Rewrite boolean operations with short-circuiting."""
        op_name = "and_" if isinstance(node.op, ast.And) else "or_"
        return self._make_short_circuit_op(op_name, node.values)

    def visit_If(self, node: ast.If) -> list[ast.stmt]:
        """Rewrite if statements."""
        if_var = "__luisa_if"

        def visit_body(body):
            visited = []
            for s in body:
                loc_call = self._set_loc(s)
                if loc_call:
                    visited.append(loc_call)
                rewritten = self.visit(s)
                if isinstance(rewritten, list):
                    visited.extend(rewritten)
                else:
                    visited.append(rewritten)
            return visited or [ast.Pass()]

        # _if = if_(lambda: test)
        assign_stmt = ast.Assign(
            targets=[ast.Name(id=if_var, ctx=ast.Store())],
            value=self._rt_call("if_",
                                ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                                              kw_defaults=[], defaults=[]),
                                           body=self.visit(node.test))
                                )
        )

        # if _if.should_run_true():
        #     with _if.true_scope():
        #         body
        true_branch = ast.If(
            test=ast.Call(func=ast.Attribute(value=ast.Name(id=if_var, ctx=ast.Load()), attr="should_run_true",
                                             ctx=ast.Load()), args=[], keywords=[]),
            body=[
                ast.With(
                    items=[ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Attribute(value=ast.Name(id=if_var, ctx=ast.Load()), attr="true_scope",
                                               ctx=ast.Load()), args=[], keywords=[]),
                    )],
                    body=visit_body(node.body)
                )
            ],
            orelse=[]
        )

        # if _if.should_run_false():
        #     with _if.false_scope():
        #         orelse
        false_branch_stmts = []
        if node.orelse:
            false_branch_stmts = [
                ast.If(
                    test=ast.Call(func=ast.Attribute(value=ast.Name(id=if_var, ctx=ast.Load()), attr="should_run_false",
                                                     ctx=ast.Load()), args=[], keywords=[]),
                    body=[
                        ast.With(
                            items=[ast.withitem(
                                context_expr=ast.Call(
                                    func=ast.Attribute(value=ast.Name(id=if_var, ctx=ast.Load()), attr="false_scope",
                                                       ctx=ast.Load()), args=[], keywords=[]),
                            )],
                            body=visit_body(node.orelse)
                        )
                    ],
                    orelse=[]
                )
            ]

        return [assign_stmt, true_branch] + false_branch_stmts

    def visit_Return(self, node: ast.Return) -> ast.Expr:
        """Rewrite return statements."""
        val = self.visit(node.value) if node.value else ast.Constant(value=None)
        return ast.Expr(value=self._rt_call("return_", val))

    def visit_Call(self, node: ast.Call) -> Any:
        """Rewrite function calls."""
        return self._rt_call(
            "call",
            self.visit(node.func),
            *([self.visit(a) for a in node.args]),
            # TODO: handle keywords
        )

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Rewrite subscript access."""
        if isinstance(node.ctx, ast.Load):
            return self._rt_call("subscript", self.visit(node.value), self.visit(node.slice))
        return node  # Store handled in visit_Assign

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Rewrite attribute access."""
        if isinstance(node.ctx, ast.Load):
            return self._rt_call("attribute", self.visit(node.value), ast.Constant(value=node.attr))
        return node  # Store handled in visit_Assign

    def _is_const_call(self, node: ast.expr) -> bool:
        """Check if a node is a call to static() or Const[Type](value) or Const(value)."""
        if isinstance(node, ast.Call):
            # Direct static() call
            if isinstance(node.func, ast.Name) and node.func.id == 'static':
                return True
            # Const(value) call
            if isinstance(node.func, ast.Name) and node.func.id == 'Const':
                return True
            # Const[Type](value) call
            if isinstance(node.func, ast.Subscript):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'Const':
                    return True
        return False

    def _extract_const_value(self, node: ast.expr) -> ast.expr:
        """Extract the value from static(x) or Const(x) -> x."""
        if isinstance(node, ast.Call):
            # For Const[Type](...) with explicit type, keep the call as-is
            # so the runtime can properly construct the typed constant
            if isinstance(node.func, ast.Subscript):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'Const':
                    # Keep the full Const[Type](args) call - don't extract
                    return self.visit(node)
            if len(node.args) == 1:
                return self.visit(node.args[0])
            elif len(node.args) > 1:
                # Multiple args - return tuple
                return ast.Tuple(elts=[self.visit(arg) for arg in node.args], ctx=ast.Load())
        return self.visit(node)

    # Set of builtin functions that are known to return DSL values
    DSL_PRODUCING_BUILTINS = {
        'sin', 'cos', 'tan', 'sqrt', 'exp', 'log', 'abs', 'floor', 'ceil',
        'round', 'min', 'max', 'clamp', 'lerp', 'pow', 'atan2', 'clock',
        'dispatch_id', 'thread_id', 'block_id', 'dispatch_size', 'kernel_id', 'object_id',
        'normalize', 'length', 'length_squared', 'dot', 'cross', 'distance', 'reflect',
        'rsqrt', 'exp2', 'exp10', 'log2', 'log10', 'sinh', 'cosh', 'tanh',
        'asinh', 'acosh', 'atanh', 'isinf', 'isnan', 'copysign', 'fma',
        'clz', 'ctz', 'popcount', 'reverse', 'transpose', 'inverse', 'determinant',
        'cast', 'bitcast', 'device_print'
    }

    # Set of resource-related methods that return DSL values
    DSL_PRODUCING_METHODS = {
        "buffer_read", "texture2d_read", "texture2d_sample", "texture2d_sample_level",
        "texture3d_read", "texture3d_sample", "buffer_size", "texture2d_size", "texture3d_size",
        "buffer_device_address", "device_address_load"
    }

    def _is_dsl_value(self, node: ast.expr) -> bool:
        """
        Check if an AST node represents a DSL value that should trigger an alloca.

        A node is considered a DSL value if it:
        1. Refers to an existing DSL variable or argument.
        2. Is a call to a recognized DSL builtin or type constructor.
        3. Is an operation (BinOp, etc.) involving at least one DSL value.
        """
        if isinstance(node, ast.Name):
            return node.id in self.dsl_vars or node.id in self.arg_names

        if isinstance(node, ast.Call):
            func = node.func
            # Direct function calls: sin(x), Float(x), etc.
            if isinstance(func, ast.Name):
                # Explicitly exclude compile-time markers
                if func.id in ("Const", "static", "Shared"):
                    return False

                # Type constructors (starting with uppercase)
                if func.id[0].isupper():
                    return True

            # Method calls: buf.read(idx), etc.
            if isinstance(func, ast.Attribute):
                if func.attr in self.DSL_PRODUCING_METHODS:
                    return True

        # Recursive checks for operations
        if isinstance(node, ast.BinOp):
            return self._is_dsl_value(node.left) or self._is_dsl_value(node.right)

        if isinstance(node, ast.UnaryOp):
            return self._is_dsl_value(node.operand)

        if isinstance(node, ast.Subscript):
            return self._is_dsl_value(node.value) or self._is_dsl_value(node.slice)

        if isinstance(node, ast.Attribute):
            return self._is_dsl_value(node.value)

        return False

    def visit_Assign(self, node: ast.Assign) -> Any:
        """Rewrite assignments."""
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript):
                return ast.Expr(
                    value=self._rt_call("subscript_assign", self.visit(target.value), self.visit(target.slice),
                                        self.visit(node.value)))

            if isinstance(target, ast.Name):
                var_name = target.id

                if target.id in self.ref_vars:
                    # Automatic store for Ref types: store(name, value)
                    return ast.Expr(value=self._rt_call("store", ast.Name(id=target.id, ctx=ast.Load()), self.visit(node.value)))

                # Check if this is a const assignment
                if self._is_const_call(node.value):
                    # Const variables are kept as Python values (not DSL variables)
                    # but we still need to track them for proper handling
                    self.const_vars.add(var_name)
                    self.dsl_vars.discard(var_name)  # Remove from dsl_vars if present
                    # Return the value directly - it's a Python constant
                    return ast.Assign(
                        targets=[target],
                        value=self._extract_const_value(node.value)
                    )
                elif var_name in self.dsl_vars:
                    # Variable was previously a DSL variable - store to it
                    return ast.Expr(value=self._rt_call("store", ast.Name(id=var_name, ctx=ast.Load()), self.visit(node.value)))
                elif self._is_dsl_value(node.value):
                    # This looks like a DSL value - create a DSL variable
                    self.dsl_vars.add(var_name)
                    # Emit: target = local_var_assign(name, value)
                    return ast.Assign(
                        targets=[target],
                        value=self._rt_call("local_var_assign", ast.Constant(value=var_name), self.visit(node.value))
                    )
                else:
                    # Regular Python value - create a DSL variable to enable reassignment
                    # This ensures that even constants can be reassigned later (e.g., in loops)
                    # EXCEPTION: Don't convert list/tuple literals as they're often used
                    # for Python-level iteration (e.g., impls = [func1, func2])
                    if isinstance(node.value, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
                        # Keep list/tuple/dict/set literals as Python values
                        return ast.Assign(
                            targets=[target],
                            value=self.visit(node.value)
                        )
                    self.dsl_vars.add(var_name)
                    return ast.Assign(
                        targets=[target],
                        value=self._rt_call("local_var_assign", ast.Constant(value=var_name), self.visit(node.value))
                    )

        # Standard assignment is fine
        return ast.Assign(
            targets=[self.visit(t) for t in node.targets],
            value=self.visit(node.value)
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        """Rewrite augmented assignment (e.g., a += b) to regular assignment (a = a + b)."""
        # Convert augassign to regular assign: target = target op value
        # Create a new Name node for the load context (to avoid Store context issues)
        if isinstance(node.target, ast.Name):
            left = ast.Name(id=node.target.id, ctx=ast.Load())
        else:
            left = node.target
        bin_op = ast.BinOp(left=left, op=node.op, right=node.value)
        assign = ast.Assign(targets=[node.target], value=bin_op)
        return self.visit_Assign(assign)

    def visit_For(self, node: ast.For) -> Any:
        """Rewrite for loops."""
        # Handle static_range
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            if node.iter.func.id == "static_range":
                return self._rewrite_for_static_range(node)
            if node.iter.func.id == "unrolled":
                if node.iter.args and isinstance(node.iter.args[0], ast.Call) and \
                        isinstance(node.iter.args[0].func, ast.Name) and \
                        node.iter.args[0].func.id == "range":
                    node.iter = ast.Call(
                        func=ast.Name(id="static_range", ctx=ast.Load()),
                        args=node.iter.args[0].args,
                        keywords=[]
                    )
                    return self._rewrite_for_static_range(node)

        # Generic for loop (handles both IR and Host via for_)
        loop_var = "__luisa_loop"
        target_name = node.target.id if isinstance(node.target, ast.Name) else "__loop_var"

        def visit_body(body):
            visited = []
            for s in body:
                loc_call = self._set_loc(s)
                if loc_call:
                    visited.append(loc_call)
                rewritten = self.visit(s)
                if isinstance(rewritten, list):
                    visited.extend(rewritten)
                else:
                    visited.append(rewritten)
            return visited or [ast.Pass()]

        body = visit_body(node.body)

        return ast.For(
            target=ast.Name(id=loop_var, ctx=ast.Store()),
            iter=self._rt_call("for_", self.visit(node.iter), ast.Constant(value=target_name)),
            body=[
                ast.With(
                    items=[ast.withitem(
                        context_expr=self._rt_call("loop_scope", ast.Name(id=loop_var, ctx=ast.Load())),
                        optional_vars=node.target
                    )],
                    body=body
                )
            ],
            orelse=[]
        )

    def _rewrite_for_static_range(self, node: ast.For) -> Any:
        """Rewrite static_range loop."""

        def visit_body(body):
            visited = []
            for s in body:
                loc_call = self._set_loc(s)
                if loc_call:
                    visited.append(loc_call)
                rewritten = self.visit(s)
                if isinstance(rewritten, list):
                    visited.extend(rewritten)
                else:
                    visited.append(rewritten)
            return visited or [ast.Pass()]

        return ast.For(
            target=node.target,
            iter=self._rt_call("call", self.visit(node.iter.func), *[self.visit(a) for a in node.iter.args]),
            body=visit_body(node.body),
            orelse=[]
        )

    def visit_While(self, node: ast.While) -> Any:
        """Rewrite while loops."""
        loop_var = "__luisa_while"

        def visit_body(body):
            visited = []
            for s in body:
                loc_call = self._set_loc(s)
                if loc_call:
                    visited.append(loc_call)
                rewritten = self.visit(s)
                if isinstance(rewritten, list):
                    visited.extend(rewritten)
                else:
                    visited.append(rewritten)
            return visited or [ast.Pass()]

        body = visit_body(node.body)

        return ast.For(
            target=ast.Name(id=loop_var, ctx=ast.Store()),
            iter=self._rt_call("while_", ast.Lambda(
                args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=self.visit(node.test))),
            body=[
                ast.With(
                    items=[ast.withitem(
                        context_expr=self._rt_call("while_scope", ast.Name(id=loop_var, ctx=ast.Load())),
                        optional_vars=None
                    )],
                    body=body
                )
            ],
            orelse=[]
        )

    def visit_Match(self, node: ast.Match) -> ast.With:
        """Rewrite match statements to SWITCH."""
        switch_var = "__luisa_switch"

        def visit_body(body):
            visited = []
            for s in body:
                loc_call = self._set_loc(s)
                if loc_call:
                    visited.append(loc_call)
                rewritten = self.visit(s)
                if isinstance(rewritten, list):
                    visited.extend(rewritten)
                else:
                    visited.append(rewritten)
            return visited or [ast.Pass()]

        cases = []
        for case in node.cases:
            if isinstance(case.pattern, ast.MatchValue):
                cases.append(ast.With(
                    items=[ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Attribute(value=ast.Name(id=switch_var, ctx=ast.Load()), attr="case_scope",
                                               ctx=ast.Load()),
                            args=[self.visit(case.pattern.value)],
                            keywords=[]
                        )
                    )],
                    body=visit_body(case.body)
                ))
            elif isinstance(case.pattern, ast.MatchAs) and case.pattern.name is None:
                cases.append(ast.With(
                    items=[ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Attribute(value=ast.Name(id=switch_var, ctx=ast.Load()), attr="default_scope",
                                               ctx=ast.Load()),
                            args=[],
                            keywords=[]
                        )
                    )],
                    body=visit_body(case.body)
                ))
            else:
                raise NotImplementedError(f"Unsupported match pattern: {type(case.pattern)}")

        return ast.With(
            items=[ast.withitem(
                context_expr=self._rt_call("switch", self.visit(node.subject)),
                optional_vars=ast.Name(id=switch_var, ctx=ast.Store())
            )],
            body=cases
        )
