"""
AST Rewriter for the LuisaCompute Python DSL v2.

This module transforms the Python AST of a DSL function into a 
builder function that generates the equivalent IR.
"""

from __future__ import annotations
import ast
import copy
from typing import Any, Optional


class ASTRewriter(ast.NodeTransformer):
    """
    Transforms Python AST into IR-building code.
    
    Example:
        a + b  =>  l_binop(builder, ast.Add(), a, b)
    """
    
    def __init__(self, file: str = "<unknown>", 
                 builder_name: str = "__luisa_builder",
                 template_params: Optional[tuple[str, ...]] = None):
        self.file = file
        self.builder_name = builder_name
        self.template_params = set(template_params or [])
        self._in_loop = 0
        self.rt_alias = "__luisa_rt"
        self.ref_vars = set() # Variables that are of Ref type
        self._is_top_level = True

    def rewrite(self, node: ast.AST) -> ast.AST:
        """Entry point for rewriting."""
        self._is_top_level = True
        return self.visit(node)

    def _set_loc(self, node: ast.AST) -> Optional[ast.Expr]:
        """Create a call to builder.set_location based on node lineno."""
        if hasattr(node, 'lineno'):
            return ast.Expr(value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.builder_name, ctx=ast.Load()),
                    attr="set_location",
                    ctx=ast.Load()
                ),
                args=[ast.Constant(value=self.file), ast.Constant(value=node.lineno)],
                keywords=[]
            ))
        return None

    def visit_Name(self, node: ast.Name) -> Any:
        """Handle names, preserving template parameters and handling Ref loads."""
        if node.id in self.template_params:
            return node
        
        if isinstance(node.ctx, ast.Load) and node.id in self.ref_vars:
            # Automatic load for Ref types: builder.load(name)
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.builder_name, ctx=ast.Load()),
                    attr="load",
                    ctx=ast.Load()
                ),
                args=[node],
                keywords=[]
            )
            
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Rewrite function definition."""
        is_top = self._is_top_level
        self._is_top_level = False
        
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
                # Handle indexed decorators like callable[int32]
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
                node.decorator_list = [] # Remove decorators from the def statement
                
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
                    last_deco = original_decorators[0] # The one closest to 'def'
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

                return [
                    definition,
                    ast.Assign(
                        targets=[ast.Name(id=node.name, ctx=ast.Store())],
                        value=value_to_assign
                    )
                ]
            
            # For non-staged nested functions, we treat them as local DSL helpers.
            # They capture the builder from the parent scope.
            # We don't change the signature, but we rewrite the body.
            old_ref_vars = self.ref_vars.copy()
            new_node = self.generic_visit(node)
            self.ref_vars = old_ref_vars
            return new_node

        # Top-level function: mangle for IR building
        # Detect Ref arguments
        for arg in node.args.args:
            # Check if annotation is Ref[...]
            ann = arg.annotation
            if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name) and ann.value.id == 'Ref':
                self.ref_vars.add(arg.arg)
            # Handle from luisa import Ref; a: Ref[i32]
            elif isinstance(ann, ast.Name) and ann.id == 'Ref':
                self.ref_vars.add(arg.arg)

        # Create a new argument for the builder
        builder_arg = ast.arg(arg=self.builder_name, annotation=None)
        
        # New arguments: builder, then original arguments
        new_args = ast.arguments(
            posonlyargs=[],
            args=[builder_arg] + [ast.arg(arg=a.arg, annotation=None) for a in node.args.args],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        )
        
        # Rewrite body
        old_ref_vars = self.ref_vars.copy()
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
            args=[ast.Name(id=self.builder_name, ctx=ast.Load())] + list(args),
            keywords=[]
        )

    def visit_BinOp(self, node: ast.BinOp) -> ast.Call:
        """Rewrite binary operations."""
        op_name = node.op.__class__.__name__
        return self._rt_call(
            "l_binop",
            ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()), args=[], keywords=[]),
            self.visit(node.left),
            self.visit(node.right)
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.Call:
        """Rewrite unary operations."""
        op_name = node.op.__class__.__name__
        return self._rt_call(
            "l_unaryop",
            ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()), args=[], keywords=[]),
            self.visit(node.operand)
        )

    def visit_Compare(self, node: ast.Compare) -> Any:
        """Rewrite comparison operations."""
        if len(node.ops) != 1:
            # TODO: handle chained comparisons
            raise NotImplementedError("Chained comparisons not yet supported in rewriter")
        
        op_name = node.ops[0].__class__.__name__
        return self._rt_call(
            "l_compare",
            ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()), args=[], keywords=[]),
            self.visit(node.left),
            self.visit(node.comparators[0])
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.Call:
        """Rewrite boolean operations."""
        op_name = node.op.__class__.__name__
        return self._rt_call(
            "l_boolop",
            ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()), args=[], keywords=[]),
            ast.List(elts=[self.visit(v) for v in node.values], ctx=ast.Load())
        )

    def visit_If(self, node: ast.If) -> ast.With:
        """Rewrite if statements."""
        if_var = "__luisa_if"
        
        def visit_body(body):
            visited = []
            for s in body:
                loc_call = self._set_loc(s)
                if loc_call: visited.append(loc_call)
                visited.append(self.visit(s))
            return visited or [ast.Pass()]

        return ast.With(
            items=[ast.withitem(
                context_expr=self._rt_call("l_if", 
                    ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), 
                               body=self.visit(node.test))
                ), 
                optional_vars=ast.Name(id=if_var, ctx=ast.Store())
            )],
            body=[
                ast.With(
                    items=[ast.withitem(
                        context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id=if_var, ctx=ast.Load()), attr="true_scope", ctx=ast.Load()), args=[], keywords=[]),
                    )],
                    body=visit_body(node.body)
                ),
                ast.With(
                    items=[ast.withitem(
                        context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id=if_var, ctx=ast.Load()), attr="false_scope", ctx=ast.Load()), args=[], keywords=[]),
                    )],
                    body=visit_body(node.orelse)
                ) if node.orelse else ast.Pass()
            ]
        )

    def visit_Return(self, node: ast.Return) -> ast.Expr:
        """Rewrite return statements."""
        val = self.visit(node.value) if node.value else ast.Constant(value=None)
        return ast.Expr(value=self._rt_call("l_return", val))

    def visit_Call(self, node: ast.Call) -> Any:
        """Rewrite function calls."""
        return self._rt_call(
            "l_call",
            self.visit(node.func),
            *([self.visit(a) for a in node.args]),
            # TODO: handle keywords
        )

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Rewrite subscript access."""
        if isinstance(node.ctx, ast.Load):
            return self._rt_call("l_subscript", self.visit(node.value), self.visit(node.slice))
        return node # Store handled in visit_Assign

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Rewrite attribute access."""
        if isinstance(node.ctx, ast.Load):
            return self._rt_call("l_attribute", self.visit(node.value), ast.Constant(value=node.attr))
        return node # Store handled in visit_Assign

    def visit_Assign(self, node: ast.Assign) -> Any:
        """Rewrite assignments."""
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript):
                return ast.Expr(value=self._rt_call("l_subscript_assign", self.visit(target.value), self.visit(target.slice), self.visit(node.value)))
            
            if isinstance(target, ast.Name):
                if target.id in self.ref_vars:
                    # Automatic store for Ref types: builder.store(name, value)
                    return ast.Expr(value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=self.builder_name, ctx=ast.Load()),
                            attr="store",
                            ctx=ast.Load()
                        ),
                        args=[ast.Name(id=target.id, ctx=ast.Load()), self.visit(node.value)],
                        keywords=[]
                    ))
                else:
                    # Standard assignment wrapped in l_local_assign
                    return ast.Assign(
                        targets=[target],
                        value=self._rt_call("l_local_assign", ast.Constant(value=target.id), self.visit(node.value))
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
        
        # Generic for loop (handles both IR and Host via l_for)
        loop_var = "__luisa_loop"
        target_name = node.target.id if isinstance(node.target, ast.Name) else "__loop_var"
        
        def visit_body(body):
            visited = []
            for s in body:
                loc_call = self._set_loc(s)
                if loc_call: visited.append(loc_call)
                visited.append(self.visit(s))
            return visited or [ast.Pass()]

        body = visit_body(node.body)
        
        return ast.For(
            target=ast.Name(id=loop_var, ctx=ast.Store()),
            iter=self._rt_call("l_for", self.visit(node.iter), ast.Constant(value=target_name)),
            body=[
                ast.With(
                    items=[ast.withitem(
                        context_expr=self._rt_call("l_loop_scope", ast.Name(id=loop_var, ctx=ast.Load())),
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
                if loc_call: visited.append(loc_call)
                visited.append(self.visit(s))
            return visited or [ast.Pass()]

        return ast.For(
            target=node.target,
            iter=self._rt_call("l_call", self.visit(node.iter.func), *[self.visit(a) for a in node.iter.args]),
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
                if loc_call: visited.append(loc_call)
                visited.append(self.visit(s))
            return visited or [ast.Pass()]

        body = visit_body(node.body)
        
        return ast.For(
            target=ast.Name(id=loop_var, ctx=ast.Store()),
            iter=self._rt_call("l_while", ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=self.visit(node.test))),
            body=[
                ast.With(
                    items=[ast.withitem(
                        context_expr=self._rt_call("l_while_scope", ast.Name(id=loop_var, ctx=ast.Load())),
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
                if loc_call: visited.append(loc_call)
                visited.append(self.visit(s))
            return visited or [ast.Pass()]

        cases = []
        for case in node.cases:
            if isinstance(case.pattern, ast.MatchValue):
                cases.append(ast.With(
                    items=[ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Attribute(value=ast.Name(id=switch_var, ctx=ast.Load()), attr="case_scope", ctx=ast.Load()),
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
                            func=ast.Attribute(value=ast.Name(id=switch_var, ctx=ast.Load()), attr="default_scope", ctx=ast.Load()),
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
                context_expr=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=self.builder_name, ctx=ast.Load()), attr="switch", ctx=ast.Load()),
                    args=[self.visit(node.subject)],
                    keywords=[]
                ),
                optional_vars=ast.Name(id=switch_var, ctx=ast.Store())
            )],
            body=cases
        )
