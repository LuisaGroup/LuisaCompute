"""
AST Compiler for the LuisaCompute Python DSL v2.

This module provides AST transformation and metadata extraction logic
to turn Python functions into IR-building functions.
"""

from __future__ import annotations
import ast
import inspect
import textwrap
from typing import Any, Optional, Callable
from dataclasses import dataclass

# Runtime imports
from .type import (
    Type, value_to_type, annotation_to_type
)


# ============================================================================
# Compilation Metadata
# ============================================================================

@dataclass
class CapturedVar:
    """Information about a captured variable."""
    name: str
    value: Any
    type: Optional[Type] = None

    def __post_init__(self):
        if self.type is None:
            self.type = value_to_type(self.value)


@dataclass
class ParsedFunction:
    """A Python function parsed into an AST with metadata."""
    name: str
    ast_node: ast.FunctionDef
    arg_names: list[str]
    arg_annotations: list[Optional[Type]]
    arg_is_reference: list[bool]
    ret_annotation: Optional[Type]
    captured_vars: dict[str, CapturedVar]
    source: str
    pyfunc: Optional[Callable] = None

    def get_arg_type(self, index: int) -> Optional[Type]:
        """Get the type annotation for an argument."""
        if index < len(self.arg_annotations):
            return self.arg_annotations[index]
        return None


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
            old_const_vars = self.const_vars.copy()
            old_dsl_vars = self.dsl_vars.copy()
            new_node = self.generic_visit(node)
            self.ref_vars = old_ref_vars
            self.const_vars = old_const_vars
            self.dsl_vars = old_dsl_vars
            return new_node

        # Top-level function: mangle for IR building
        # Detect Ref arguments
        for arg in node.args.args:
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
        """Rewrite comparison operations."""
        if len(node.ops) != 1:
            # TODO: handle chained comparisons
            raise NotImplementedError("Chained comparisons not yet supported in rewriter")

        op_name = node.ops[0].__class__.__name__
        return self._rt_call(
            "compare",
            ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()),
                     args=[], keywords=[]),
            self.visit(node.left),
            self.visit(node.comparators[0])
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.Call:
        """Rewrite boolean operations."""
        op_name = node.op.__class__.__name__
        return self._rt_call(
            "boolop",
            ast.Call(func=ast.Attribute(value=ast.Name(id="ast", ctx=ast.Load()), attr=op_name, ctx=ast.Load()),
                     args=[], keywords=[]),
            ast.List(elts=[self.visit(v) for v in node.values], ctx=ast.Load())
        )

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
        """Check if a node is a call to const()."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'const':
                return True
        return False

    def _extract_const_value(self, node: ast.expr) -> ast.expr:
        """Extract the value from const(x) -> x."""
        if isinstance(node, ast.Call) and len(node.args) == 1:
            return node.args[0]
        return node

    def _is_dsl_value(self, node: ast.expr) -> bool:
        """
        Check if a node likely represents a DSL value.
        
        This is a heuristic to determine if we should create a DSL variable
        for the assignment or keep it as a Python variable.
        """
        # Function calls that return DSL values
        if isinstance(node, ast.Call):
            func = node.func
            # Direct function calls like sin(), cos(), etc.
            if isinstance(func, ast.Name):
                # Math builtins that return DSL values
                dsl_builtins = {'sin', 'cos', 'tan', 'sqrt', 'exp', 'log', 
                               'abs', 'floor', 'ceil', 'round', 'min', 'max',
                               'clamp', 'lerp', 'pow', 'atan2'}
                if func.id in dsl_builtins:
                    return True
                # Type casts like Float(x) return DSL values
                if func.id[0].isupper():
                    return True
            # Method calls on builder that return values
            if isinstance(func, ast.Attribute):
                # builder.switch(), builder.if_(), etc. return Python objects
                # builder.create_block(), etc. also return Python objects
                if func.attr in ('switch', 'if_', 'while_', 'for_range', 
                                'create_block', 'call'):
                    return False
                # Other method calls likely return values
                return True
        
        # Binary operations on DSL values
        if isinstance(node, ast.BinOp):
            return True
        
        # Unary operations on DSL values
        if isinstance(node, ast.UnaryOp):
            return True
        
        # Subscript access like buf[idx]
        if isinstance(node, ast.Subscript):
            return True
        
        # Attributes like dispatch_id().x
        if isinstance(node, ast.Attribute):
            return True
        
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
                    # Extract the inner value from const(x)
                    inner_value = self._extract_const_value(node.value)
                    # Standard assignment with the inner value
                    return ast.Assign(
                        targets=[target],
                        value=self._rt_call("local_assign", ast.Constant(value=var_name), self.visit(inner_value))
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


# ============================================================================
# Entry Points
# ============================================================================

def parse_function(func: Callable, source: Optional[str] = None) -> ParsedFunction:
    """Parse a Python function and return its metadata."""
    # Logic moved from parser.py
    # If we already have the AST
    if hasattr(func, '_luisa_ast'):
        func_def = func._luisa_ast
        source = ast.unparse(func_def)
    elif source is not None:
        # Parse AST from provided source
        try:
            tree = ast.parse(source)
            func_def = tree.body[0]
        except Exception as e:
            raise RuntimeError(f"Error parsing provided source for {func}: {e}") from e
    else:
        # Get source code
        try:
            lines, start_line = inspect.getsourcelines(func)
            source = "".join(lines)

            # Dedent source to handle nested function definitions
            source = textwrap.dedent(source)

            # Parse AST
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                raise RuntimeError(f"Syntax error in {func}: {e}") from e

            # Get function definition
            if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
                raise RuntimeError(f"Expected function definition, got {type(tree.body[0])}")

            func_def = tree.body[0]

            # Adjust line numbers to be global
            ast.increment_lineno(func_def, start_line - 1)
        except (OSError, TypeError) as e:
            raise RuntimeError(f"Cannot get source for {func}: {e}") from e

    # Get signature
    try:
        sig = inspect.signature(func)

        # Extract argument names and annotations
        arg_names = []
        arg_annotations = []
        arg_is_reference = []

        for name, param in sig.parameters.items():
            arg_names.append(name)
            ann, is_ref = annotation_to_type(param.annotation)
            arg_annotations.append(ann)
            arg_is_reference.append(is_ref)

        # Extract return annotation
        ret_annotation, _ = annotation_to_type(sig.return_annotation)
    except (NameError, TypeError):
        # Fallback for specialized functions where types are not yet defined
        arg_names = [arg.arg for arg in func_def.args.args]
        arg_annotations = [None] * len(arg_names)
        arg_is_reference = [False] * len(arg_names)
        ret_annotation = None

    # Analyze captured variables
    captured_vars = _analyze_captured_vars(func)

    return ParsedFunction(
        name=func.__name__,
        ast_node=func_def,
        arg_names=arg_names,
        arg_annotations=arg_annotations,
        arg_is_reference=arg_is_reference,
        ret_annotation=ret_annotation,
        captured_vars=captured_vars,
        source=source,
        pyfunc=func
    )


def _analyze_captured_vars(func: Callable) -> dict[str, CapturedVar]:
    """Analyze captured (closure) variables."""
    captured = {}

    try:
        closure = inspect.getclosurevars(func)

        # Non-local variables
        for name, value in closure.nonlocals.items():
            captured[name] = CapturedVar(name=name, value=value)

        # Global variables
        for name, value in closure.globals.items():
            captured[name] = CapturedVar(name=name, value=value)

    except (TypeError, ValueError):
        # Function has no closure
        pass

    return captured
