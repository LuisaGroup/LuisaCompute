import ast
import inspect
import sys
from types import SimpleNamespace, ModuleType
from .types import length_of, element_of, vector, uint, implicit_convertible, short, ushort, long, ulong
from . import globalvars
from .dylibs import lcapi
from .builtin import builtin_func_names, builtin_func, builtin_bin_op, builtin_type_cast, \
    builtin_unary_op, callable_call, wrap_with_tmp_var
from .types import dtype_of, to_lctype, CallableType, vector_dtypes, matrix_dtypes
from .types import BuiltinFuncType, BuiltinFuncBuilder
from .vector import is_swizzle_name, get_swizzle_code, get_swizzle_resulttype
from .struct import StructType


def ctx():
    return globalvars.current_context


class VariableInfo:
    def __init__(self, dtype, expr, is_arg=False):
        self.dtype = dtype
        self.expr = expr
        self.is_arg = is_arg


# ---------------------------------------------------------------------------
# Host-side (python-level) constant evaluation.
#
# The native C++ DSL resolves ``if`` / ``while`` / ``switch`` conditions that are
# known on the host and erases the unselected branch during AST construction
# (see lc_api skill: "Native `if` on a host variable -- the unselected branch is
# erased during AST construction").  A previously-removed C++ ast-evaluator used
# to provide this; we restore an equivalent, deliberately narrow python version
# so that e.g. ``if ConstantFlag: <illegal code>`` is eliminated before the body
# is ever name-checked.
#
# We only fold pure python ``int``/``bool`` literals combined with basic operators.
# Anything involving a device value (a buffer read, dispatch_id, a function
# argument, ...) evaluates to ``_HOST_UNSET`` and is left as a real GPU branch.
# ---------------------------------------------------------------------------
_HOST_UNSET = object()
_INT32_LO = -(1 << 31)
_INT32_HI = (1 << 31) - 1


def _as_int(v):
    return int(v) if isinstance(v, bool) else v


def _int_ok(v):
    # accept python bool/int whose integer value fits int32 (device `int`); reject
    # floats and out-of-range ints so we never disagree with the GPU codegen.
    if isinstance(v, bool) or isinstance(v, int):
        return _INT32_LO <= _as_int(v) <= _INT32_HI
    return False


def _host_binop(op, a, b):
    a, b = _as_int(a), _as_int(b)
    try:
        if op is ast.Add:
            r = a + b
        elif op is ast.Sub:
            r = a - b
        elif op is ast.Mult:
            r = a * b
        elif op is ast.FloorDiv:
            if b == 0:
                return _HOST_UNSET
            r = a // b
        elif op is ast.Mod:
            if b == 0:
                return _HOST_UNSET
            r = a % b
        elif op is ast.LShift:
            r = a << b
        elif op is ast.RShift:
            r = a >> b
        elif op is ast.BitAnd:
            r = a & b
        elif op is ast.BitOr:
            r = a | b
        elif op is ast.BitXor:
            r = a ^ b
        elif op is ast.Pow:
            if b < 0:
                return _HOST_UNSET
            r = a ** b
        else:
            return _HOST_UNSET
    except Exception:
        return _HOST_UNSET
    return r if _int_ok(r) else _HOST_UNSET


def _host_cmp(op, a, b):
    if op is ast.Eq:
        return a == b
    if op is ast.NotEq:
        return a != b
    if op is ast.Lt:
        return a < b
    if op is ast.LtE:
        return a <= b
    if op is ast.Gt:
        return a > b
    if op is ast.GtE:
        return a >= b
    return _HOST_UNSET


def host_eval(node):
    """Return a python int/bool constant for a pure-constant expression, or
    ``_HOST_UNSET`` if it cannot be resolved on the host."""
    t = type(node)
    if t is ast.Constant:
        return node.value if _int_ok(node.value) else _HOST_UNSET
    if t is ast.Name:
        if node.id in ctx().host_constants:
            return ctx().host_constants[node.id]
        cv = ctx().closure_variable.get(node.id, _HOST_UNSET)
        if cv is not _HOST_UNSET and _int_ok(cv):
            return cv
        return _HOST_UNSET
    if t is ast.UnaryOp:
        v = host_eval(node.operand)
        if v is _HOST_UNSET:
            return _HOST_UNSET
        if type(node.op) is ast.UAdd:
            return v
        if type(node.op) is ast.USub:
            r = -_as_int(v)
            return r if _int_ok(r) else _HOST_UNSET
        if type(node.op) is ast.Not:
            return not v
        if type(node.op) is ast.Invert:
            r = ~_as_int(v)
            return r if _int_ok(r) else _HOST_UNSET
        return _HOST_UNSET
    if t is ast.BinOp:
        a = host_eval(node.left)
        b = host_eval(node.right)
        if a is _HOST_UNSET or b is _HOST_UNSET:
            return _HOST_UNSET
        return _host_binop(type(node.op), a, b)
    if t is ast.Compare:
        if len(node.ops) != 1:
            return _HOST_UNSET
        a = host_eval(node.left)
        b = host_eval(node.comparators[0])
        if a is _HOST_UNSET or b is _HOST_UNSET:
            return _HOST_UNSET
        return _host_cmp(type(node.ops[0]), a, b)
    if t is ast.BoolOp:
        vals = [host_eval(x) for x in node.values]
        if any(v is _HOST_UNSET for v in vals):
            return _HOST_UNSET
        if type(node.op) is ast.And:
            r = True
            for v in vals:
                r = r and bool(v)
            return r
        if type(node.op) is ast.Or:
            r = False
            for v in vals:
                r = r or bool(v)
            return r
        return _HOST_UNSET
    return _HOST_UNSET



# Visit AST of the python functions; computes the following for each expression node:
# node.dtype: type of the expression
# node.expr: the expression as defined in the function builder
# node.lr: "l" or "r", representing l-value or r-value

class ASTVisitor:
    def __call__(self, node):
        method = getattr(self, 'build_' + node.__class__.__name__, None)
        if method is None:
            raise NotImplementedError(f'Unsupported syntax node {node}')
        try:
            self.comment_source(node)
            return method(node)
        except Exception as e:
            if not hasattr(e, "already_printed"):
                self.print_error(node, e)
                e.already_printed = True
            raise

    @staticmethod
    def comment_source(node):
        n = node.lineno - 1
        if getattr(ctx(), "_last_comment_source_lineno", None) != n:
            ctx()._last_comment_source_lineno = n
            # lcapi.builder().comment_(str(n) + "  " + ctx().sourcelines[n].strip())

    @staticmethod
    def print_error(node, e):
        # print exception, line number and corresponding source code
        def support_color():
            try:
                shell = get_ipython().__class__.__name__
                return shell in ('ZMQInteractiveShell', 'TerminalInteractiveShell')
            except NameError:
                return sys.stdout.isatty()

        if support_color():
            red, green, bold, clr = "\x1b[31;1m", "\x1b[32;1m", "\x1b[1m", "\x1b[0m"
        else:
            red = green = bold = clr = ""
        prefix = f"{bold}{ctx().func.filename.split('/')[-1]}:{ctx().func.lineno - 1 + node.lineno} {clr}"
        if type(e).__name__ == "CompileError":
            print(
                f"{prefix}{red}Error:{clr}{bold} The above error occured during compilation of '{e.func.__name__}'{clr}")
        else:
            print(f"{prefix}{red}Error:{clr}{bold} {type(e).__name__}: {e}{clr}")
        source = ctx().sourcelines[node.lineno - 1: node.end_lineno]
        for idx, line in enumerate(source):
            print(line.rstrip('\n'))
            startcol = node.col_offset if idx == 0 else 0
            endcol = node.end_col_offset if idx == len(source) - 1 else len(line)
            print(green + ' ' * startcol + '~' * (endcol - startcol) + clr)
        print(f"in luisa.func '{ctx().func.__name__}' in {ctx().func.filename}")
        if type(e).__name__ != "CompileError":
            import traceback
            traceback.print_exc(limit=-2)
            # print("Traceback (most recent call last):")
            # _, _, tb = sys.exc_info()
            # traceback.print_tb(tb,limit=-1) # Fixed format
            print()  # blank line

    @staticmethod
    def build_FunctionDef(node):
        for x in node.body:
            build(x)

    @staticmethod
    def build_Expr(node):
        if isinstance(node.value, ast.Call):
            build(node.value)
            if node.value.dtype is not None:
                raise TypeError("Discarding non-void return value")
        else:
            if not isinstance(node.value, ast.Constant):
                raise TypeError("Dangling expression")

    @staticmethod
    def build_Return(node):
        if node.value is not None:
            build(node.value)
        # deduce & check type of return value
        return_type = None if node.value is None else node.value.dtype
        if hasattr(ctx(), 'return_type'):
            if ctx().return_type != return_type:
                raise TypeError("inconsistent return type in multiple return statements")
        else:
            ctx().return_type = return_type
            if ctx().call_from_host and return_type is not None:
                raise TypeError("luisa func called on host can't return value")
        # build return statement
        lcapi.builder().return_(getattr(node.value, 'expr', None))

    @staticmethod
    def build_Call(node):
        build(node.func)
        for x in node.args:
            build(x)
        for x in node.keywords:
            build(x.value)
        # if it's called as method, call with self (the object)
        if type(node.func) is ast.Attribute and getattr(node.func, 'calling_method', False):
            args = [node.func.value] + node.args
        else:
            args = node.args
        kwargs = {x.arg: x.value for x in node.keywords}
        # custom function
        if node.func.dtype is CallableType:
            node.dtype, node.expr = callable_call(node.func.expr, *args, **kwargs)
        # builtin function (called by name string, as defined in builtin_func_names)
        elif node.func.dtype is BuiltinFuncType:
            node.dtype, node.expr = builtin_func(node.func.expr, *args, **kwargs)
        elif node.func.dtype is BuiltinFuncBuilder:
            node.dtype, node.expr = node.func.expr.builder(*args, **kwargs)
        # type: cast / construct
        elif node.func.dtype is type:
            dtype = node.func.expr
            node.dtype, node.expr = builtin_type_cast(dtype, *args, **kwargs)
        else:
            raise TypeError(f"{node.func.dtype} is not callble")
        node.lr = 'r'

    @staticmethod
    def build_Attribute(node):
        build(node.value)
        # vector swizzle
        if node.value.dtype in vector_dtypes:
            if is_swizzle_name(node.attr):
                original_size = to_lctype(node.value.dtype).dimension()
                swizzle_size = len(node.attr)
                swizzle_code = get_swizzle_code(node.attr, original_size)
                node.dtype = get_swizzle_resulttype(node.value.dtype, swizzle_size)
                node.expr = lcapi.builder().swizzle(to_lctype(node.dtype), node.value.expr, swizzle_size, swizzle_code)
                node.lr = 'l' if swizzle_size == 1 else 'r'
            else:
                raise AttributeError(f"vector has no attribute '{node.attr}'")
        # struct member
        elif type(node.value.dtype) is StructType:
            if node.attr in node.value.dtype.idx_dict:  # data member
                idx = node.value.dtype.idx_dict[node.attr]
                node.dtype = node.value.dtype.membertype[idx]
                node.expr = lcapi.builder().member(to_lctype(node.dtype), node.value.expr, idx)
                node.lr = node.value.lr
            elif node.attr in node.value.dtype.method_dict:  # struct method
                node.dtype = CallableType
                node.calling_method = True
                node.expr = node.value.dtype.method_dict[node.attr]
            else:
                raise AttributeError(f"struct {node.value.dtype} has no attribute '{node.attr}'")
        elif node.value.dtype is ModuleType:
            node.dtype, node.expr, node.lr = build.captured_expr(getattr(node.value.expr, node.attr))
        elif hasattr(node.value.dtype, node.attr):
            entry = getattr(node.value.dtype, node.attr)
            if type(entry).__name__ == "func":
                node.dtype, node.expr = CallableType, entry
                node.calling_method = True
            elif type(entry) is BuiltinFuncBuilder:
                node.dtype, node.expr = BuiltinFuncBuilder, entry
                node.calling_method = True
            else:
                raise TypeError(f"Can't access attribute {node.attr} ({entry}) in luisa func")
        else:
            raise AttributeError(f"type {node.value.dtype} has no attribute '{node.attr}'")

    @staticmethod
    def build_Subscript(node):
        build(node.value)
        node.lr = node.value.lr
        build(node.slice)
        if type(node.value.dtype).__name__ == "ArrayType":
            node.dtype = node.value.dtype.dtype
        elif type(node.value.dtype).__name__ == "SharedArrayType":
            node.dtype = node.value.dtype.dtype
        elif node.value.dtype in vector_dtypes:
            node.dtype = element_of(node.value.dtype)
        elif node.value.dtype in matrix_dtypes:
            # matrix indexed is a column vector
            node.dtype = vector(float, length_of(node.value.dtype))
        else:
            raise TypeError(f"{node.value.dtype} object is not subscriptable")
        node.expr = lcapi.builder().access(to_lctype(node.dtype), node.value.expr, node.slice.expr)

    # external variable captured in kernel -> (dtype, expr, lr)
    @staticmethod
    def captured_expr(val):
        dtype = dtype_of(val)
        if dtype == ModuleType:
            return dtype, val, None
        if dtype == type:
            return dtype, val, None
        if dtype == CallableType:
            return dtype, val, None
        if dtype == BuiltinFuncBuilder:
            return dtype, val, None
        if dtype == str:
            return dtype, val, 'r'
        if dtype is int and isinstance(val, int) and not isinstance(val, bool):
            dtype, lctype, expr = _build_int_literal(_int_literal_dtype(val), val)
            return dtype, expr, 'r'
        lctype = to_lctype(dtype)
        if lctype.is_basic():
            return dtype, lcapi.builder().literal(lctype, val), 'r'
        if lctype.is_buffer() or lctype.is_custom_buffer():
            return dtype, lcapi.builder().buffer_binding(lctype, val.handle, 0,
                                                         val.bytesize), 'l'  # offset defaults to 0
        if lctype.is_texture():
            return dtype, lcapi.builder().texture_binding(lctype, val.handle, 0), 'l'  # miplevel defaults to 0
        if lctype.is_bindless_array():
            return dtype, lcapi.builder().bindless_array_binding(val.handle), 'l'
        if lctype.is_accel():
            return dtype, lcapi.builder().accel_binding(val.handle), 'l'
        if lctype.is_array():
            # create array and assign each element
            expr = lcapi.builder().local(lctype)
            for idx, x in enumerate(val.values):
                sliceexpr = lcapi.builder().literal(to_lctype(int),idx)
                lhs = lcapi.builder().access(lctype.element(), expr, sliceexpr)
                rhs_dtype, rhs_expr, _ = build.captured_expr(x)
                assert implicit_convertible(rhs_dtype, dtype.dtype)
                lcapi.builder().assign(lhs, rhs_expr)
            return dtype, expr, 'r'
        if lctype.is_structure():
            # create struct and assign each element
            expr = lcapi.builder().local(lctype)
            for idx, x in enumerate(val.values):
                lhs = lcapi.builder().member(to_lctype(dtype.membertype[idx]), expr, idx)
                rhs_dtype, rhs_expr, rhs_lr = build.captured_expr(x)
                assert implicit_convertible(rhs_dtype, dtype.membertype[idx])
                lcapi.builder().assign(lhs, rhs_expr)
            return dtype, expr, 'r'
        raise TypeError("unrecognized closure var type:", type(val))

    @staticmethod
    def build_Name(node, allow_none=False):
        # Note: in Python all local variables are function-scoped
        if node.id in builtin_func_names:
            node.dtype, node.expr = BuiltinFuncType, node.id
        elif node.id in ctx().local_variable:
            varinfo = ctx().local_variable[node.id]
            node.dtype = varinfo.dtype
            node.expr = varinfo.expr
            node.is_arg = varinfo.is_arg
            node.lr = 'l'
        elif node.id in ctx().default_arg_values:
            default_val = ctx().default_arg_values[node.id]
            node.dtype = dtype_of(default_val)
            if node.dtype is int and isinstance(default_val, int) and not isinstance(default_val, bool):
                node.dtype, _, node.expr = _build_int_literal(_int_literal_dtype(default_val), default_val)
            else:
                node.expr = lcapi.builder().literal(to_lctype(node.dtype), default_val)
            node.lr = 'r'
        else:
            val = ctx().closure_variable.get(node.id)
            if val is None:
                if not allow_none:
                    raise NameError(f"undeclared identifier '{node.id}'")
                node.dtype = None
                return
            node.dtype, node.expr, node.lr = build.captured_expr(val)

    @staticmethod
    def build_Constant(node):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            # widen/ synthesise int literals that overflow or would be rounded by
            # the C++ float-greedy literal binding (see _build_int_literal)
            node.dtype, _, node.expr = _build_int_literal(_int_literal_dtype(node.value), node.value)
        else:
            node.dtype = dtype_of(node.value)
            if node.dtype is str:
                node.expr = node.value
            else:
                node.expr = lcapi.builder().literal(to_lctype(node.dtype), node.value)
        node.lr = 'r'

    @staticmethod
    def build_tuple_assign(lhs, rhs):
        # tuple isn't supported in general;
        # here we support a special case: tuple assignment
        if type(rhs) is not ast.Tuple:
            raise TypeError(f"can only unpack from tuple (got {rhs.dtype})")
        if len(lhs.elts) < len(rhs.elts):
            raise ValueError("too many values to unpack (expected {len(lhs.elts)}, got {len(rhs.elts)})")
        if len(lhs.elts) > len(rhs.elts):
            raise ValueError("not enough values to unpack (expected {len(lhs.elts)}, got {len(rhs.elts)})")
        tmps = []
        for r in rhs.elts:
            if r.dtype is None:
                raise TypeError("Can't assign None to variable")
            tmpexpr = lcapi.builder().local(to_lctype(r.dtype))
            lcapi.builder().assign(tmpexpr, r.expr)
            tmps.append(SimpleNamespace(dtype=r.dtype, expr=tmpexpr))
        for i in range(len(lhs.elts)):
            build.build_assign_pair(lhs.elts[i], tmps[i])

    @staticmethod
    def build_assign_pair(lhs, rhs):
        if rhs.dtype is None:
            raise TypeError("Can't assign None to variable")
        if type(lhs) is ast.Tuple:
            return build.build_tuple_assign(lhs, rhs)
        # allows left hand side to be undefined
        if type(lhs) is ast.Name:
            build.build_Name(lhs, allow_none=True)
        else:
            build(lhs)
        # create local variable if it doesn't exist yet
        if lhs.dtype is None:
            dtype = rhs.dtype  # craete variable with same type as rhs
            # store type & ptr info into name
            # ref type
            
            if type(rhs.dtype).__name__ == "SharedArrayType":
                ctx().local_variable[lhs.id] = VariableInfo(dtype, rhs.expr)
                lhs.expr = rhs.expr
                lhs.dtype = dtype
                return
            else:
                lhs.expr = lcapi.builder().local(to_lctype(dtype))
                ctx().local_variable[lhs.id] = VariableInfo(dtype, lhs.expr)
            # Note: all local variables are function scope
        else:
            # must assign with same type; no implicit casting is allowed.
            if not implicit_convertible(lhs.dtype, rhs.dtype):
                lhs_type = to_lctype(lhs.dtype)
                rhs_type = to_lctype(rhs.dtype)
                if (not (lhs_type.is_vector() or lhs_type.is_matrix())) or not implicit_convertible(lhs_type.element(),
                                                                                                   rhs_type):
                    raise TypeError(f"Can't assign to {lhs.dtype} with {rhs.dtype} ")
            if lhs.lr == "r":
                raise TypeError("Can't assign to read-only value")
        lcapi.builder().assign(lhs.expr, rhs.expr)

    @staticmethod
    def build_AnnAssign(node):
        build(node.annotation)
        if node.annotation.dtype is not type:
            raise TypeError("invalid assign annotation")
        build(node.value)
        build.build_assign_pair(node.target, node.value)
        if not implicit_convertible(node.target.dtype, node.annotation.expr):
            raise TypeError(f"assign annotation is {node.annotation.expr}, got {node.target.dtype}")

    @staticmethod
    def build_Assign(node):
        build(node.value)
        if len(node.targets) == 1:
            build.build_assign_pair(node.targets[0], node.value)
            # Track python-level constants of simple `name = <const>` assignments
            # so that `if name:` can be folded as a macro branch later.
            if type(node.targets[0]) is ast.Name:
                hv = host_eval(node.value)
                if hv is _HOST_UNSET:
                    ctx().host_constants.pop(node.targets[0].id, None)
                else:
                    ctx().host_constants[node.targets[0].id] = hv
        else:  # chained assignment
            wrap_with_tmp_var(node.value)
            for targ in node.targets:
                build.build_assign_pair(targ, node.value)
                if type(targ) is ast.Name:
                    ctx().host_constants.pop(targ.id, None)

    @staticmethod
    def build_AugAssign(node):
        build(node.target)
        build(node.value)
        dtype, expr = builtin_bin_op(type(node.op), node.target, node.value)
        lcapi.builder().assign(node.target.expr, expr)
        # an augmented assignment produces a device value; drop any host constant
        if type(node.target) is ast.Name:
            ctx().host_constants.pop(node.target.id, None)

    @staticmethod
    def build_UnaryOp(node):
        build(node.operand)
        node.dtype, node.expr = builtin_unary_op(type(node.op), node.operand)
        node.lr = 'r'

    @staticmethod
    def build_BinOp(node):
        build(node.left)
        build(node.right)
        node.dtype, node.expr = builtin_bin_op(type(node.op), node.left, node.right)
        node.lr = 'r'

    @staticmethod
    def build_Compare(node):
        build(node.left)
        for x in node.comparators:
            build(x)
        node.dtype, node.expr = builtin_bin_op(type(node.ops[0]), node.left, node.comparators[0])
        # compare from left to right
        for idx in range(1, len(node.comparators)):
            obj = SimpleNamespace()
            obj.dtype, obj.expr = builtin_bin_op(type(node.ops[idx]), node.comparators[idx - 1], node.comparators[idx])
            node.dtype, node.expr = builtin_bin_op(ast.And, node, obj)
        node.lr = 'r'

    @staticmethod
    def build_BoolOp(node):
        for x in node.values:
            build(x)
        node.dtype, node.expr = builtin_bin_op(type(node.op), node.values[0], node.values[1])
        # bool operators of same type are left-associative
        for idx in range(2, len(node.values)):
            node.dtype, node.expr = builtin_bin_op(type(node.op), node, node.values[idx])
        node.lr = 'r'

    @staticmethod
    def build_With(node):
        for i in node.items:
            func_name = i.context_expr.func.id
            if func_name == "autodiff":
                ad = lcapi.builder().autodiff_()
                with ad.body():
                    for x in node.body:
                        build(x)
            else:
                raise SyntaxError("Unknown with.")

    @staticmethod
    def build_If(node):
        # Macro branch: if the test is a pure host (python) constant, resolve it
        # here and build ONLY the taken branch -- the unselected branch is erased
        # before codegen, so it may even contain illegal / undeclared names.
        cond = host_eval(node.test)
        if cond is not _HOST_UNSET:
            taken = node.body if cond else node.orelse
            for x in taken:
                build(x)
            return
        # condition
        build(node.test)
        if node.test.dtype != bool:
            raise TypeError(f"If condition must be bool, got {node.test.dtype}")
        ifstmt = lcapi.builder().if_(node.test.expr)
        # branches
        with ifstmt.true_branch():
            for x in node.body:
                build(x)
        with ifstmt.false_branch():
            for x in node.orelse:
                build(x)

    @staticmethod
    def build_Match(node):
        # condition
        build(node.subject)
        case_map = {}
        if type(node.subject.dtype).__name__ in {"RayQueryAllType", "RayQueryAnyType"}:
            # rayquery expr
            query_stmt = lcapi.builder().ray_query_(node.subject.expr)
            for c in node.cases:
                if type(c.pattern) != ast.MatchClass or (
                        (c.pattern.cls.id) != "is_triangle" and (c.pattern.cls.id) != "is_procedural"):
                    raise TypeError("Rayquery condition must be \"case is_triangle():\" or \"case is_procedural():\"")
                if case_map.get(c.pattern) == True:
                    raise SyntaxError("Case value can only have one.")
                case_map[c.pattern] = True
                if (c.pattern.cls.id) == "is_triangle":
                    with query_stmt.on_triangle_candidate():
                        for x in c.body:
                            build(x)
                else:
                    with query_stmt.on_procedural_candidate():
                        for x in c.body:
                            build(x)
            return
        if not node.subject.dtype in {int, uint, short, ushort, long, ulong}:
            raise TypeError(f"Match condition must be int or uint, got {node.subject.dtype}")
        switch_stmt = lcapi.builder().switch_(node.subject.expr)
        with switch_stmt.body():
            for c in node.cases:
                if type(c.pattern) == ast.MatchValue:
                    if type(c.pattern.value.value) != int:
                        raise TypeError(f"Match case condition must be int or uint, got {type(c.pattern.value.value)}")
                    if case_map.get(c.pattern.value.value) == True:
                        raise SyntaxError("Case value can only have one.")
                    case_map[c.pattern.value.value] = True
                    build(c.pattern.value)
                    case_stmt = lcapi.builder().case_(c.pattern.value.expr)
                    with case_stmt.body():
                        for x in c.body:
                            build(x)
                        lcapi.builder().break_()
                else:
                    if case_map.get("default") == True:
                        raise SyntaxError("Case value can only have one.")
                    case_map["default"] = True
                    default_stmt = lcapi.builder().default_()
                    with default_stmt.body():
                        for x in c.body:
                            build(x)
                        lcapi.builder().break_()

    @staticmethod
    def build_IfExp(node):
        build(node.body)
        build(node.test)
        build(node.orelse)
        from .dylibs.lcapi import bool2, bool3, bool4
        if node.test.dtype not in {bool, bool2, bool3, bool4}:
            raise TypeError(f"IfExp condition must be bool or bool vector, got {node.test.dtype}")
        if node.body.dtype != node.orelse.dtype:
            raise TypeError(
                f"Both result expressions of IfExp must be of same type. ({node.body.dtype} vs {node.orelse.dtype})")
        if node.test.dtype != bool and length_of(node.test.dtype) != length_of(node.body.dtype):
            raise TypeError(
                f"IfExp condition must be either bool or vector of same length ({length_of(node.test.dtype)} != {length_of(node.body.dtype)})")
        node.dtype = node.body.dtype
        node.expr = lcapi.builder().call(to_lctype(node.dtype), lcapi.CallOp.SELECT,
                                         [node.orelse.expr, node.body.expr, node.test.expr])
        node.lr = 'r'

    @staticmethod
    def build_range_for(node):
        if len(node.iter.args) not in {1, 2, 3}:
            raise TypeError(f"'range' expects 1/2/3 arguments, got {len(node.iter.args)}")
        for x in node.iter.args:
            build(x)
            assert x.dtype in {int, uint, short, ushort, long, ulong}
        if len(node.iter.args) == 1:
            range_start = lcapi.builder().literal(to_lctype(int),0)
            range_stop = node.iter.args[0].expr
            range_step = lcapi.builder().literal(to_lctype(int),1)
        if len(node.iter.args) == 2:
            range_start, range_stop = [x.expr for x in node.iter.args]
            range_step = lcapi.builder().literal(to_lctype(int),1)
        if len(node.iter.args) == 3:
            range_start, range_stop, range_step = [x.expr for x in node.iter.args]
        # loop variable
        varexpr = lcapi.builder().local(to_lctype(int))
        lcapi.builder().assign(varexpr, range_start)
        ctx().local_variable[node.target.id] = VariableInfo(int, varexpr)
        # build for statement
        condition = lcapi.builder().binary(to_lctype(bool), lcapi.BinaryOp.LESS, varexpr, range_stop)
        forstmt = lcapi.builder().for_(varexpr, condition, range_step)
        with forstmt.body():
            for x in node.body:
                build(x)

    @staticmethod
    def build_container_for(node):
        range_start = lcapi.builder().literal(to_lctype(int),0)
        range_stop = lcapi.builder().literal(to_lctype(int),length_of(node.iter.dtype))
        range_step = lcapi.builder().literal(to_lctype(int),1)
        # loop variable
        idxexpr = lcapi.builder().local(to_lctype(int))
        lcapi.builder().assign(idxexpr, range_start)
        eltype = element_of(node.iter.dtype) if node.iter.dtype not in matrix_dtypes else vector(float, length_of(
            node.iter.dtype))  # iterating through matrix yields vectors
        varexpr = lcapi.builder().access(to_lctype(eltype), node.iter.expr, idxexpr)  # loop variable (element)
        ctx().local_variable[node.target.id] = VariableInfo(eltype, varexpr)
        # build for statement
        condition = lcapi.builder().binary(to_lctype(bool), lcapi.BinaryOp.LESS, idxexpr, range_stop)
        forstmt = lcapi.builder().for_(idxexpr, condition, range_step)
        with forstmt.body():
            for x in node.body:
                build(x)

    @staticmethod
    def build_For(node):
        # currently only supports for x in range(...)
        assert type(node.target) is ast.Name
        if type(node.iter) is ast.Call and type(node.iter.func) is ast.Name and node.iter.func.id == "range":
            v = build.build_range_for(node)
            return v
        build(node.iter)
        if type(node.iter.dtype).__name__ == "ArrayType" or node.iter.dtype in {*vector_dtypes, *matrix_dtypes}:
            v = build.build_container_for(node)
            return v
        else:
            raise TypeError(f"{node.iter.dtype} object is not iterable")

    @staticmethod
    def build_While(node):
        loopstmt = lcapi.builder().loop_()
        with loopstmt.body():
            # condition
            build(node.test)
            if node.test.dtype != bool:
                raise TypeError(f"While condition must be bool, got {node.test.dtype}")
            ifstmt = lcapi.builder().if_(node.test.expr)
            with ifstmt.true_branch():
                for x in node.body:
                    build(x)
            with ifstmt.false_branch():
                lcapi.builder().break_()
                # body

    @staticmethod
    def build_Break(node):
        lcapi.builder().break_()

    @staticmethod
    def build_Continue(node):
        lcapi.builder().continue_()

    @staticmethod
    def build_JoinedStr(node):
        node.joined = []
        for x in node.values:
            if isinstance(x, ast.FormattedValue):
                build(x.value)
                if hasattr(x.value, 'joined'):
                    node.joined += x.value.joined
                else:
                    node.joined.append(x.value)
            elif isinstance(x, ast.Constant):
                build(x)
                node.joined.append(x)
            else:
                assert False

    @staticmethod
    def build_List(node):
        node.dtype = list
        for x in node.elts:
            build(x)
        node.expr = None

    @staticmethod
    def build_Tuple(node):
        node.dtype = list
        for x in node.elts:
            build(x)
        node.expr = None

    @staticmethod
    def build_Pass(node):
        pass


build = ASTVisitor()


def _int_literal_dtype(v):
    """Pick the luisa scalar dtype for a python ``int`` literal, mirroring C++
    integer-literal typing.  The value keeps dtype ``int`` whenever it fits
    int32 (so all existing programs are unchanged); only literals that would
    overflow int32 -- and therefore abort in the C++ literal binding's
    range check (src/py/export_runtime.cpp:789) -- are widened to ``uint`` /
    ``long`` / ``ulong``.  ``bool`` is handled by the caller."""
    if -(1 << 31) <= v <= (1 << 31) - 1:
        return int
    if 0 <= v <= (1 << 32) - 1:
        return uint
    if -(1 << 63) <= v <= (1 << 63) - 1:
        return long
    if 0 <= v <= (1 << 64) - 1:
        return ulong
    raise TypeError(f"integer literal {v} is out of range for any supported scalar type")


def _build_int_literal(dtype, value):
    """Build a luisa scalar integer literal of ``dtype`` for a python int.

    The C++ ``FunctionBuilder::literal`` binding takes a ``std::variant`` whose
    *float* alternative greedily captures python ``int`` objects (pybind converts
    via ``PyFloat_AsDouble``).  Any integer that is not exactly representable as
    an IEEE-754 binary32 mantissa is therefore silently rounded (e.g.
    2654435800 -> 2654435840) and ``0xFFFFFFFF`` overflows and aborts in the
    binding's round-trip check.  We keep the direct ``literal`` call for values
    that survive the float round-trip unchanged (this covers every small literal
    and is behaviour-identical to before), and otherwise synthesise the exact
    value from base-2**16 device-integer digits (each < 2**24, hence float-exact,
    combined with exact device ``ulong`` arithmetic, constant-folded by codegen
    so the generated value is exact and costs nothing at runtime).
    """
    lctype = to_lctype(dtype)
    import struct
    try:
        # the C++ variant greedily packs python ints through its `float`
        # (binary32) alternative; the literal is faithful only when value is
        # exactly representable as float32 (== ~24-bit mantissa).
        f32 = struct.unpack('f', struct.pack('f', float(value)))[0]
        faithful = (int(f32) == value)
    except (OverflowError, ValueError):
        faithful = False
    if faithful:
        return dtype, lctype, lcapi.builder().literal(lctype, value)
    # two's-complement 64-bit bit pattern, built in ulong then cast to dtype
    bits = value & ((1 << 64) - 1)
    ulong_t = to_lctype(ulong)
    base = 1 << 16
    expr = lcapi.builder().literal(ulong_t, 0)
    for shift in (48, 32, 16, 0):
        digit = (bits >> shift) & 0xFFFF  # <= 0xFFFF: float-exact literal
        mul = lcapi.builder().binary(ulong_t, lcapi.BinaryOp.MUL, expr,
                                     lcapi.builder().literal(ulong_t, base))
        expr = lcapi.builder().binary(ulong_t, lcapi.BinaryOp.ADD, mul,
                                      lcapi.builder().literal(ulong_t, digit))
    if dtype is not ulong:
        expr = lcapi.builder().cast(lctype, lcapi.CastOp.STATIC, expr)
    return dtype, lctype, expr
