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

    def __init__(self, file: str = "<unknown>",
                 template_params: Optional[tuple[str, ...]] = None):
        self.file = file
        self.template_params = set(template_params or [])
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
        """Handle names, preserving template parameters and handling Ref loads."""
        if node.id in self.template_params:
            return node

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

        # Save and reset DSL variable tracking for nested functions
        saved_const_vars = self.const_vars.copy()
        saved_dsl_vars = self.dsl_vars.copy()
        self.const_vars = set()
        self.dsl_vars = set()

        # If it's a nested function, check if it has Luisa decorators
        if not is_top:
            is_staged = False
            for deco in node.decorator_list:
                # Simple check for 'callable' or 'kernel' decorators
                if isinstance(deco, ast.Name) and deco.id in ('callable', 'kernel'):
                    is_staged = True
                    break
                if isinstance(deco, ast.Attribute) and deco.attr in ('callable', 'kernel'):
                    is_staged = True
                    break
                # Handle indexed decorators like callable[Int]
                if isinstance(deco, ast.Subscript):
                    if isinstance(deco.value, ast.Name) and deco.value.id in ('callable', 'kernel'):
                        is_staged = True
                        break
                    if isinstance(deco.value, ast.Attribute) and deco.value.attr in ('callable', 'kernel'):
                        is_staged = True
                        break

            if is_staged:
                # Staged functions are plain Python code that defines DSL functions.
                # They will be processed by their own StagedFunction instance.
                # To handle the 'inspect' failure in 'exec', we can pass the source code.

                source = ast.unparse(node)

                # Visit decorators
                original_decorators = node.decorator_list
                node.decorator_list = []  # Remove decorators from the def statement

                # We return:
                # def f(...): ...
                # f = deco1(deco2(f), source=source) -- Wait, decorators return StagedFunctionDecorators

                definition = node
                value_to_assign = ast.Name(id=node.name, ctx=ast.Load())

                # We want to apply decorators and then pass source to the final StagedFunctionDecorator.__call__
                # Actually, @callable returns a StagedFunction.
                # If we have @callable \n def f(): ...
                # It becomes f = callable(f, source=source)

                # For multiple decorators, it's more complex, but usually it's just one.
                # Let's assume one decorator for now or handle the last one.

                if original_decorators:
                    last_deco = original_decorators[0]  # The one closest to 'def'
                    value_to_assign = ast.Call(
                        func=self.visit(last_deco),
                        args=[value_to_assign],
                        keywords=[ast.keyword(arg="source", value=ast.Constant(value=source))]
                    )
                    # Apply other decorators if any
                    for deco in reversed(original_decorators[1:]):
                        value_to_assign = ast.Call(
                            func=self.visit(deco),
                            args=[value_to_assign],
                            keywords=[]
                        )

                res = [
                    definition,
                    ast.Assign(
                        targets=[ast.Name(id=node.name, ctx=ast.Store())],
                        value=value_to_assign
                    )
                ]
                # Restore state
                self.const_vars = saved_const_vars
                self.dsl_vars = saved_dsl_vars
                return res

            # For non-staged nested functions, we treat them as local DSL helpers.
            # They capture the builder from the parent scope.
            # We don't change the signature, but we rewrite the body.
            old_ref_vars = self.ref_vars.copy()
            new_node = self.generic_visit(node)
            self.ref_vars = old_ref_vars
            self.const_vars = saved_const_vars
            self.dsl_vars = saved_dsl_vars
            return new_node

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
            name=f"__luisa_built_{node.name}",
            args=new_args,
            body=new_body,
            decorator_list=[],  # Remove decorators
            returns=None
        )

    def _rt_call(self, name: str, *args: ast.expr) -> ast.Call:
        """Helper to create a call to a runtime function."""
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.rt_alias, ctx=ast.Load()),
                attr=name,
                ctx=ast.Load()
            ),
            args=list(args),
            keywords=[]
        )

    def visit_BinOp(self, node: ast.BinOp) -> ast.Call:
        """Rewrite binary operations."""
        op_name = node.op.__class__.__name__
        return self._rt_call(
            "binop",
            ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()),
                     args=[], keywords=[]),
            self.visit(node.left),
            self.visit(node.right)
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.Call:
        """Rewrite unary operations."""
        op_name = node.op.__class__.__name__
        return self._rt_call(
            "unaryop",
            ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()),
                     args=[], keywords=[]),
            self.visit(node.operand)
        )

    def visit_Compare(self, node: ast.Compare) -> Any:
        """Rewrite comparison operations, including chained comparisons."""
        if len(node.ops) == 1:
            op_name = node.ops[0].__class__.__name__
            return self._rt_call(
                "compare",
                ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()),
                         args=[], keywords=[]),
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

    def visit_If(self, node: ast.If) -> ast.With:
        """Rewrite if statements."""
        if_var = "__luisa_if"

        def visit_body(body):
            visited = []
            for s in body:
                loc_call = self._set_loc(s)
                if loc_call:
                    visited.append(loc_call)
                visited.append(self.visit(s))
            return visited or [ast.Pass()]

        return ast.With(
            items=[ast.withitem(
                context_expr=self._rt_call("if_",
                                           ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                                                         kw_defaults=[], defaults=[]),
                                                      body=self.visit(node.test))
                                           ),
                optional_vars=ast.Name(id=if_var, ctx=ast.Store())
            )],
            body=[
                ast.With(
                    items=[ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Attribute(value=ast.Name(id=if_var, ctx=ast.Load()), attr="true_scope",
                                               ctx=ast.Load()), args=[], keywords=[]),
                    )],
                    body=visit_body(node.body)
                ),
                ast.With(
                    items=[ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Attribute(value=ast.Name(id=if_var, ctx=ast.Load()), attr="false_scope",
                                               ctx=ast.Load()), args=[], keywords=[]),
                    )],
                    body=visit_body(node.orelse)
                ) if node.orelse else ast.Pass()
            ]
        )

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
            if len(node.args) == 1:
                return node.args[0]
            elif len(node.args) > 1:
                # Multiple args - return tuple
                return ast.Tuple(elts=list(node.args), ctx=ast.Load())
        return node

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
                    # Mark as const variable
                    self.const_vars.add(var_name)
                    self.dsl_vars.discard(var_name)  # Remove from dsl_vars if present
                    # Standard assignment with the constructor call (preserved)
                    return ast.Assign(
                        targets=[target],
                        value=self._rt_call("local_assign", ast.Constant(value=var_name), self.visit(node.value))
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
                    # Regular Python variable - keep as-is
                    return ast.Assign(
                        targets=[target],
                        value=self.visit(node.value)
                    )

        # Standard assignment is fine
        return ast.Assign(
            targets=[self.visit(t) for t in node.targets],
            value=self.visit(node.value)
        )

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
                visited.append(self.visit(s))
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
                visited.append(self.visit(s))
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
                visited.append(self.visit(s))
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
                visited.append(self.visit(s))
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
