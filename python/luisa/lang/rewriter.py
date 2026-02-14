"""
AST Rewriter for the LuisaCompute Python DSL v2.

This module transforms the Python AST of a DSL function into a 
builder function that generates the equivalent IR.
"""

from __future__ import annotations
import ast
from typing import Any


class ASTRewriter(ast.NodeTransformer):
    """
    Transforms Python AST into IR-building code.
    
    Example:
        a + b  =>  l_binop(builder, ast.Add(), a, b)
    """
    
    def __init__(self, builder_name: str = "__luisa_builder"):
        self.builder_name = builder_name
        self.rt_alias = "__luisa_rt"

    def rewrite(self, node: ast.AST) -> ast.AST:
        """Entry point for rewriting."""
        return self.visit(node)

    def _termination_check(self) -> ast.If:
        """Create a check to return if the current block is terminated."""
        return ast.If(
            test=ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id=self.builder_name, ctx=ast.Load()),
                        attr="current_block",
                        ctx=ast.Load()
                    ),
                    attr="is_terminated",
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            ),
            body=[ast.Return(value=None)],
            orelse=[]
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Rewrite function definition."""
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
        new_body = []
        for stmt in node.body:
            rewritten = self.visit(stmt)
            if isinstance(rewritten, list):
                new_body.extend(rewritten)
            else:
                new_body.append(rewritten)

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
                    body=[self.visit(s) for s in node.body] or [ast.Pass()]
                ),
                ast.With(
                    items=[ast.withitem(
                        context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id=if_var, ctx=ast.Load()), attr="false_scope", ctx=ast.Load()), args=[], keywords=[]),
                    )],
                    body=[self.visit(s) for s in node.orelse] or [ast.Pass()]
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
        
        def process_body(body):
            new_body = []
            for stmt in body:
                rewritten = self.visit(stmt)
                if isinstance(rewritten, list):
                    new_body.extend(rewritten)
                else:
                    new_body.append(rewritten)
            return new_body

        body = process_body(node.body)
        
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
        return ast.For(
            target=node.target,
            iter=self._rt_call("l_call", self.visit(node.iter.func), *[self.visit(a) for a in node.iter.args]),
            body=[self.visit(s) for s in node.body],
            orelse=[]
        )

    def visit_While(self, node: ast.While) -> Any:
        """Rewrite while loops."""
        loop_var = "__luisa_while"
        
        def process_body(body):
            new_body = []
            for stmt in body:
                rewritten = self.visit(stmt)
                if isinstance(rewritten, list):
                    new_body.extend(rewritten)
                else:
                    new_body.append(rewritten)
            return new_body

        body = process_body(node.body)
        
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
                    body=[self.visit(s) for s in case.body] or [ast.Pass()]
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
                    body=[self.visit(s) for s in case.body] or [ast.Pass()]
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
