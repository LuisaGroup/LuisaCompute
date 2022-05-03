import ast
import astpretty
import inspect
import sys
import traceback
from .types import from_lctype
from . import globalvars
import lcapi
from .builtin import builtin_func_names, builtin_func, builtin_bin_op, builtin_type_cast, \
    builtin_unary_op, callable_call
from .types import dtype_of, to_lctype, CallableType, is_vector_type
from .types import BuiltinFuncType, BuiltinFuncEntry, BuiltinFuncBuilder
from .vector import is_swizzle_name, get_swizzle_code, get_swizzle_resulttype
from .arraytype import ArrayType
from .structtype import StructType

def ctx():
    return globalvars.current_context


class VariableInfo:
    def __init__(self, dtype, expr, is_arg = False):
        self.dtype = dtype
        self.expr = expr
        self.is_arg = is_arg


class ASTVisitor:
    def __call__(self, node):
        method = getattr(self, 'build_' + node.__class__.__name__, None)
        if method is None:
            raise Exception(f'Unsupported node {node}:\n{astpretty.pformat(node)}')
        try:
            # print(astpretty.pformat(node))
            self.comment_source(node)
            return method(node)
        except Exception as e:
            final_message = "error when building AST"
            if str(e) != final_message:
                self.print_error(node, e)
            raise Exception(final_message)

    @staticmethod
    def comment_source(node):
        n = node.lineno-1
        if getattr(ctx(), "_last_comment_source_lineno", None) != n:
            ctx()._last_comment_source_lineno = n
            lcapi.builder().comment_(str(n) + "  " + ctx().sourcelines[n].strip())

    @staticmethod
    def print_error(node, e):
        if sys.stdout.isatty():
            red = "\x1b[31;1m"
            green = "\x1b[32;1m"
            bold = "\x1b[1m"
            clr = "\x1b[0m"
        else:
            red = ""
            green = ""
            bold = ""
            clr = ""
        print(f"{bold}({ctx().kernel.__class__.__name__}){ctx().__name__}:{node.lineno}:{node.col_offset}: {clr}{red}Error:{clr}{bold} {type(e).__name__}: {e}{clr}")
        source = ctx().sourcelines[node.lineno-1: node.end_lineno]
        for idx,line in enumerate(source):
            print(line.rstrip('\n'))
            startcol = node.col_offset if idx==0 else 0
            endcol = node.end_col_offset if idx==len(source)-1 else len(line)
            print(green + ' ' * startcol + '~' * (endcol - startcol) + clr)
        print("Traceback:")
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        print('An error occurred on line {} in statement {}'.format(line, text))

    @staticmethod
    def build_FunctionDef(node):
        for x in node.body: # build over a list
            build(x)

    @staticmethod
    def build_Expr(node):
        if isinstance(node.value, ast.Call):
            build(node.value)
            if node.value.dtype != None:
                raise TypeError("Expr: Discarding return value")
        else:
            raise TypeError("Dangling expression.")

    @staticmethod
    def build_Return(node):
        if node.value != None:
            build(node.value)
        # deduce & check type of return value
        return_type = None if node.value == None else node.value.dtype
        if hasattr(ctx(), 'return_type'):
            if ctx().return_type != return_type:
                raise TypeError("inconsistent return type in multiple return statements")
        else:
            ctx().return_type = return_type
            if not ctx().is_device_callable and return_type != None:
                raise TypeError("only callable can return value")
        # build return statement
        lcapi.builder().return_(node.value.expr)

    @staticmethod
    def build_Call(node):
        for x in node.args:
            build(x)
        # static function
        if type(node.func) is ast.Name:
            build(node.func)
            # custom callable
            if node.func.dtype is CallableType:
                node.dtype, node.expr = callable_call(node.func.expr, node.args)
            # funciton name undefined: look into builtin functions
            elif node.func.dtype is BuiltinFuncType:
                node.dtype, node.expr = builtin_func(node.func.expr, node.args)
            elif node.func.dtype is BuiltinFuncBuilder:
                node.dtype, node.expr = node.func.expr.builder(node.args)
            # type: cast / construct
            elif node.func.dtype is type:
                dtype = node.func.expr
                node.dtype, node.expr = builtin_type_cast(dtype, node.args)
            else:
                raise TypeError(f"calling non-callable variable: {node.func.id} ({node.func.dtype})")
        # class method
        elif type(node.func) is ast.Attribute: # class method
            build(node.func)
            if node.func.dtype is CallableType:
                node.dtype, node.expr = callable_call(node.func.expr, [node.func.value] + node.args)
            elif node.func.dtype is BuiltinFuncType:
                node.dtype, node.expr = builtin_func(node.func.expr, [node.func.value] + node.args)
            elif node.func.dtype is BuiltinFuncBuilder:
                node.dtype, node.expr = node.func.expr.builder([node.func.value] + node.args)
            else:
                raise TypeError(f'calling non-callable member ({node.func.dtype}).')
        else:
            raise Exception('unrecognized call func type')

    @staticmethod
    def build_Attribute(node):
        build(node.value)
        # vector swizzle
        if is_vector_type(node.value.dtype):
            if is_swizzle_name(node.attr):
                original_size = to_lctype(node.value.dtype).dimension()
                swizzle_size = len(node.attr)
                swizzle_code = get_swizzle_code(node.attr, original_size)
                node.dtype = get_swizzle_resulttype(node.value.dtype, swizzle_size)
                node.expr = lcapi.builder().swizzle(to_lctype(node.dtype), node.value.expr, swizzle_size, swizzle_code)
            else:
                raise AttributeError(f"vector has no attribute '{node.attr}'")
        # struct member
        elif type(node.value.dtype) is StructType:
            idx = node.value.dtype.idx_dict[node.attr]
            node.dtype = node.value.dtype.membertype[idx]
            if node.dtype == CallableType: # method
                node.expr = node.value.dtype.method_dict[node.attr]
            else: # data member
                node.expr = lcapi.builder().member(to_lctype(node.dtype), node.value.expr, idx)
        elif hasattr(node.value.dtype, node.attr):
            entry = getattr(node.value.dtype, node.attr)
            if type(entry).__name__ == "kernel":
                if not entry.is_device_callable:
                    raise TypeError("can't call kernel in kernel/callable")
                node.dtype, node.expr = CallableType, entry
            elif type(entry) is BuiltinFuncEntry:
                node.dtype, node.expr = BuiltinFuncType, entry.name
            elif type(entry) is BuiltinFuncBuilder:
                node.dtype, node.expr = BuiltinFuncBuilder, entry
            else:
                raise TypeError(f"Can't access member {entry} in kernel/callable")
        else:
            raise AttributeError(f"type {node.value.dtype} has no attribute '{node.attr}'")

    @staticmethod
    def build_Subscript(node):
        build(node.value)
        build(node.slice)
        if type(node.value.dtype) is ArrayType:
            node.dtype = node.value.dtype.dtype
        elif is_vector_type(node.value.dtype):
            node.dtype = from_lctype(to_lctype(node.value.dtype).element())
        elif node.value.dtype in {lcapi.float2x2, lcapi.float3x3, lcapi.float4x4}:
            element_dtypename = to_lctype(node.value.dtype).element().description()
            dim = to_lctype(node.value.dtype).dimension()
            node.dtype = getattr(lcapi, element_dtypename + str(dim))
        else:
            raise TypeError(f"{node.value.dtype} can't be subscripted")
        node.expr = lcapi.builder().access(to_lctype(node.dtype), node.value.expr, node.slice.expr)

    # external variable captured in kernel -> (dtype, expr)
    @staticmethod
    def captured_expr(val):
        dtype = dtype_of(val)
        if dtype == type:
            return dtype, val
        if dtype == CallableType:
            return dtype, val
        if dtype == BuiltinFuncBuilder:
            return dtype, val
        lctype = to_lctype(dtype)
        if lctype.is_basic():
            return dtype, lcapi.builder().literal(lctype, val)
        if lctype.is_buffer():
            return dtype, lcapi.builder().buffer_binding(lctype, val.handle, 0, val.bytesize) # offset defaults to 0
        if lctype.is_texture():
            return dtype, lcapi.builder().texture_binding(lctype, val.handle, 0) # miplevel defaults to 0
        if lctype.is_bindless_array():
            return dtype, lcapi.builder().bindless_array_binding(val.handle)
        if lctype.is_accel():
            return dtype, lcapi.builder().accel_binding(val.handle)
        if lctype.is_array():
            # create array and assign each element
            expr = lcapi.builder().local(lctype)
            for idx,x in enumerate(val.values):
                sliceexpr = lcapi.builder().literal(to_lctype(int), idx)
                lhs = lcapi.builder().access(lctype, expr, sliceexpr)
                rhs = lcapi.builder().literal(lctype.element(), x)
                lcapi.builder().assign(lhs, rhs)
            return dtype, expr
        if lctype.is_structure():
            # create struct and assign each element
            expr = lcapi.builder().local(lctype)
            for idx,x in enumerate(val.values):
                lhs = lcapi.builder().member(to_lctype(dtype.membertype[idx]), expr, idx)
                rhs = captured_expr(x)
                assert rhs.dtype == dtype.membertype[idx]
                lcapi.builder().assign(lhs, rhs.expr)
            return dtype, expr
        raise Exception("unrecognized closure var type:", type(val))

    @staticmethod
    def build_Name(node, allow_none = False):
        # Note: in Python all local variables are function-scoped
        if node.id in builtin_func_names:
            node.dtype, node.expr = BuiltinFuncType, node.id
        elif node.id in ctx().local_variable:
            varinfo = ctx().local_variable[node.id]
            node.dtype = varinfo.dtype
            node.expr = varinfo.expr
            node.is_arg = varinfo.is_arg
        else:
            val = ctx().closure_variable.get(node.id)
            # print("NAME:", node.id, "VALUE:", val)
            if val is None: # do not capture python builtin print
                if not allow_none:
                    raise NameError(f"undeclared idenfitier '{node.id}'")
                node.dtype = None
                return
            node.dtype, node.expr = build.captured_expr(val)

    @staticmethod
    def build_Constant(node):
        node.dtype = dtype_of(node.value)
        if node.dtype is str:
            node.expr = node.value
        else:
            node.expr = lcapi.builder().literal(to_lctype(node.dtype), node.value)

    @staticmethod
    def build_Assign(node):
        if len(node.targets) != 1:
            raise NotImplementedError('Chained assignment not supported')
        # allows left hand side to be undefined
        if type(node.targets[0]) is ast.Name:
            build.build_Name(node.targets[0], allow_none=True)
            if getattr(node.targets[0], "is_arg", False): # is argument
                if node.dtype not in (int, float, bool): # not scalar
                    raise TypeError("Assignment to non-scalar argument is not allowed.")
        else:
            build(node.targets[0])
        build(node.value)
        # create local variable if it doesn't exist yet
        if node.targets[0].dtype is None:
            dtype = node.value.dtype # craete variable with same type as rhs
            node.targets[0].expr = lcapi.builder().local(to_lctype(dtype))
            # store type & ptr info into name
            ctx().local_variable[node.targets[0].id] = VariableInfo(dtype, node.targets[0].expr)
            # Note: all local variables are function scope
        else:
            # must assign with same type; no implicit casting is allowed.
            if node.targets[0].dtype != node.value.dtype:
                raise TypeError(f"Can't assign to {node.targets[0].dtype} with {node.value.dtype} ")
        lcapi.builder().assign(node.targets[0].expr, node.value.expr)

    @staticmethod
    def build_AugAssign(node):
        build(node.target)
        build(node.value)
        dtype, expr = builtin_bin_op(type(node.op), node.target, node.value)
        lcapi.builder().assign(node.target.expr, expr)

    @staticmethod
    def build_UnaryOp(node):
        build(node.operand)
        node.dtype, node.expr = builtin_unary_op(type(node.op), node.operand)

    @staticmethod
    def build_BinOp(node):
        build(node.left)
        build(node.right)
        node.dtype, node.expr = builtin_bin_op(type(node.op), node.left, node.right)

    @staticmethod
    def build_Compare(node):
        if len(node.comparators) != 1:
            raise NotImplementedError('chained comparison not supported yet.')
        build(node.left)
        build(node.comparators[0])
        node.dtype, node.expr = builtin_bin_op(type(node.ops[0]), node.left, node.comparators[0])

    @staticmethod
    def build_BoolOp(node):
        # should be short-circuiting
        if len(node.values) != 2:
            raise NotImplementedError('chained bool op not supported yet. use brackets instead.')
        for x in node.values:
            build(x)
        node.dtype, node.expr = builtin_bin_op(type(node.op), node.values[0], node.values[1])

    @staticmethod
    def build_If(node):
        # condition
        build(node.test)
        if node.test.dtype != bool:
            raise TypeError(f"If condition must be bool, got {node.test.dtype}")
        ifstmt = lcapi.builder().if_(node.test.expr)
        # true branch
        lcapi.builder().push_scope(ifstmt.true_branch())
        for x in node.body:
            build(x)
        lcapi.builder().pop_scope(ifstmt.true_branch())
        # false branch
        lcapi.builder().push_scope(ifstmt.false_branch())
        for x in node.orelse:
            build(x)
        lcapi.builder().pop_scope(ifstmt.false_branch())

    @staticmethod
    def build_IfExp(node):
        build(node.body)
        build(node.test)
        build(node.orelse)
        if node.test.dtype != bool:
            raise TypeError(f"IfExp condition must be bool, got {node.test.dtype}")
        if node.body.dtype != node.orelse.dtype:
            raise TypeError(f"Both result expressions of IfExp must be of same type. ({node.body.dtype} vs {node.orelse.dtype})")
        node.dtype = node.body.dtype
        node.expr = lcapi.builder().call(to_lctype(node.dtype), lcapi.CallOp.SELECT, [node.orelse.expr, node.body.expr, node.test.expr])

    @staticmethod
    def build_For(node):
        assert type(node.target) is ast.Name
        assert type(node.iter) is ast.Call and type(node.iter.func) is ast.Name
        assert node.iter.func.id == "range" and len(node.iter.args) in {1,2,3}
        for x in node.iter.args:
            build(x)
            assert x.dtype is int
        if len(node.iter.args) == 1:
            range_start = lcapi.builder().literal(to_lctype(int), 0)
            range_stop = node.iter.args[0].expr
            range_step = lcapi.builder().literal(to_lctype(int), 1)
        if len(node.iter.args) == 2:
            range_start, range_stop = [x.expr for x in node.iter.args]
            range_step = lcapi.builder().literal(to_lctype(int), 1)
        if len(node.iter.args) == 3:
            range_start, range_stop, range_step = [x.expr for x in node.iter.args]
        # loop variable
        varexpr = lcapi.builder().local(to_lctype(int))
        lcapi.builder().assign(varexpr, range_start)
        ctx().local_variable[node.target.id] = (int, varexpr)
        # build for statement
        condition = lcapi.builder().binary(to_lctype(bool), lcapi.BinaryOp.LESS, varexpr, range_stop)
        forstmt = lcapi.builder().for_(varexpr, condition, range_step)
        lcapi.builder().push_scope(forstmt.body())
        for x in node.body:
            build(x)
        lcapi.builder().pop_scope(forstmt.body())

    @staticmethod
    def build_While(node):
        loopstmt = lcapi.builder().loop_()
        lcapi.builder().push_scope(loopstmt.body())
        # condition
        build(node.test)
        ifstmt = lcapi.builder().if_(node.test.expr)
        lcapi.builder().push_scope(ifstmt.false_branch())
        lcapi.builder().break_()
        lcapi.builder().pop_scope(ifstmt.false_branch())
        # body
        for x in node.body:
            build(x)
        lcapi.builder().pop_scope(loopstmt.body())

    @staticmethod
    def build_Break(node):
        lcapi.builder().break_()

    @staticmethod
    def build_Continue(node):
        lcapi.builder().continue_()
    
build = ASTVisitor()
