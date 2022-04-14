import ast
import astpretty
import lcapi
from .builtin import deduce_unary_type, deduce_binary_type, builtin_func
from .types import scalar_types, basic_types
from .vector import is_swizzle_name, get_swizzle_code
from .buffer import Buffer

def dsl_local_array(node_dtype, node_size):
    # Note: arguments are AST nodes
    assert type(node_dtype) == ast.Name
    if node_dtype.id in ('int','uint','float','bool'):
        dtype = lcapi.Type.from_(node_dtype.id)
    else:
        raise Exception("array of vector/matrix not supported yet")
    assert type(node_size) == ast.Constant
    size = node_size.value

    # TODO: support direct initialization from list / np array 
    rettype = lcapi.Type.from_(f'array<{dtype.description()},{size}>')
    return rettype, lcapi.builder().local(rettype)

class ASTVisitor:
    def __call__(self, ctx, node):
        method = getattr(self, 'build_' + node.__class__.__name__, None)
        if method is None:
            raise Exception(f'Unsupported node {node}:\n{astpretty.pformat(node)}')
        return method(ctx, node)

    @staticmethod
    def build_FunctionDef(ctx, node):
        if node.returns is not None:
            raise Exception('Return value is not supported')
        # if len(node.args.args)!=0 or node.args.vararg is not None or node.args.kwarg is not None:
        #     raise Exception('Arguments are not supported')
        for x in node.body: # build over a list
            build(ctx, x)

    @staticmethod
    def build_Expr(ctx, node):
        if isinstance(node.value, ast.Call):
            build(ctx, node.value)
        else:
            print("WARNING: Expr discarded")


    @staticmethod
    def build_Call(ctx, node):
        if node.func.__class__.__name__ == "Name": # static function
            # check for builtins
            if node.func.id == 'Array':
                node.dtype, node.expr = dsl_local_array(*node.args)
            elif node.func.id == 'Struct':
                raise Exception("struct not supported")
            else: # builtin functions
                for x in node.args:
                    build(ctx, x)
                node.dtype, node.expr = builtin_func(node.func.id, node.args)
        elif node.func.__class__.__name__ == "Attribute": # class method
            for x in node.args:
                build(ctx, x)
            build(ctx, node.func.value)
            builtin_op = None
            return_type = None
            # check for builtin methods (buffer, etc.)
            if node.func.value.dtype.is_buffer():
                if node.func.attr == "read":
                    builtin_op = lcapi.CallOp.BUFFER_READ
                    return_type = node.func.value.dtype.element()
                if node.func.attr == "write":
                    builtin_op = lcapi.CallOp.BUFFER_WRITE
            if builtin_op is None:
                raise Exception('unsupported method')
            node.dtype = return_type
            if return_type is None:
                lcapi.builder().call(builtin_op, [node.func.value.expr] + [x.expr for x in node.args])
            else: # function call has return value
                node.expr = lcapi.builder().call(return_type, builtin_op, [node.func.value.expr] + [x.expr for x in node.args])
        else:
            raise Exception('unrecognized call func type')

    @staticmethod
    def build_Attribute(ctx, node):
        build(ctx, node.value)
        # vector swizzle
        if node.value.dtype.is_vector():
            if is_swizzle_name(node.attr):
                original_size = node.value.dtype.dimension()
                swizzle_size = len(node.attr)
                if swizzle_size == 1:
                    node.dtype = node.value.dtype.element()
                else:
                    node.dtype = lcapi.Type.from_(f'vector<{node.value.dtype.element().description()},{swizzle_size}>')
                swizzle_code = get_swizzle_code(node.attr, original_size)
                node.expr = lcapi.builder().swizzle(node.dtype, node.value.expr, swizzle_size, swizzle_code)

    @staticmethod
    def build_Subscript(ctx, node):
        build(ctx, node.value)
        build(ctx, node.slice)
        assert node.value.dtype.is_array() # TODO: atomic
        node.dtype = node.value.dtype.element()
        node.expr = lcapi.builder().access(node.dtype, node.value.expr, node.slice.expr)

    @staticmethod
    def build_Name(ctx, node):
        # Note: in Python all local variables are function-scoped
        if node.id in ctx.local_variable:
            node.dtype, node.expr = ctx.local_variable[node.id]
        else:
            val = ctx.closure_variable.get(node.id)
            if val is None:
                node.expr = None
                return
            if type(val) in basic_types:
                node.dtype = basic_types[type(val)]
                node.expr = lcapi.builder().literal(node.dtype, val)
                return
            if type(val) is Buffer:
                node.dtype = lcapi.Type.from_("buffer<" + val.dtype.__name__ + ">")
                node.expr = lcapi.builder().buffer_binding(node.dtype, val.handle, 0)
                return

            raise Exception("unrecognized closure var type:", type(val), node.id)

    @staticmethod
    def build_Constant(ctx, node):
        if type(node.value) is str:
            raise Exception("String is not supported")
        node.dtype = scalar_types[type(node.value)]
        node.expr = lcapi.builder().literal(node.dtype, node.value)

    @staticmethod
    def build_Assign(ctx, node):
        if len(node.targets)!=1:
            raise Exception('Tuple assignment not supported')
        build(ctx, node.targets[0])
        build(ctx, node.value)
        # create local variable if it doesn't exist yet
        if node.targets[0].expr is None:
            dtype = node.value.dtype # deduced data type
            node.targets[0].expr = lcapi.builder().local(dtype)
            # store type & ptr info into name
            ctx.local_variable[node.targets[0].id] = (dtype, node.targets[0].expr)
            # all local variables are function scope
        lcapi.builder().assign(node.targets[0].expr, node.value.expr)

    @staticmethod
    def build_AugAssign(ctx, node):
        build(ctx, node.target)
        build(ctx, node.value)
        op = {
            ast.Add: lcapi.BinaryOp.ADD,
            ast.Sub: lcapi.BinaryOp.SUB,
            ast.Mult: lcapi.BinaryOp.MUL,
            ast.Div: lcapi.BinaryOp.DIV, # TODO type: int/int->float?
            ast.FloorDiv: lcapi.BinaryOp.DIV, # TODO type check: int only
            ast.Mod: lcapi.BinaryOp.MOD, # TODO support fmod using builtins
            ast.LShift: lcapi.BinaryOp.SHL,
            ast.RShift: lcapi.BinaryOp.SHR,
            ast.BitOr: lcapi.BinaryOp.BIT_OR,
            ast.BitXor: lcapi.BinaryOp.BIT_XOR,
            ast.BitAnd: lcapi.BinaryOp.BIT_AND,
        }.get(type(node.op))
        # ast.Pow, ast.MatMult is not supported
        if op is None:
            raise Exception(f'Unsupported binary operation: {type(node.op)}')
        x = lcapi.builder().binary(node.target.dtype, op, node.target.expr, node.value.expr)
        lcapi.builder().assign(node.target.expr, x)

    @staticmethod
    def build_UnaryOp(ctx, node):
        build(ctx, node.operand)
        op = {
            ast.UAdd: lcapi.UnaryOp.PLUS,
            ast.USub: lcapi.UnaryOp.MINUS,
            ast.Not: lcapi.UnaryOp.NOT,
            ast.Invert: lcapi.UnaryOp.BIT_NOT
        }.get(type(node.op))
        if op is None:
            raise Exception(f'Unsupported binary operation: {type(node.op)}')
        node.dtype = deduce_unary_type(node.op, node.operand.dtype)
        node.expr = lcapi.builder().unary(node.dtype, op, node.operand.expr)

    @staticmethod
    def build_BinOp(ctx, node):
        build(ctx, node.left)
        build(ctx, node.right)
        op = {
            ast.Add: lcapi.BinaryOp.ADD,
            ast.Sub: lcapi.BinaryOp.SUB,
            ast.Mult: lcapi.BinaryOp.MUL,
            ast.Div: lcapi.BinaryOp.DIV, # TODO type: int/int->float?
            ast.FloorDiv: lcapi.BinaryOp.DIV, # TODO type check: int only
            ast.Mod: lcapi.BinaryOp.MOD, # TODO support fmod using builtins
            ast.LShift: lcapi.BinaryOp.SHL,
            ast.RShift: lcapi.BinaryOp.SHR,
            ast.BitOr: lcapi.BinaryOp.BIT_OR,
            ast.BitXor: lcapi.BinaryOp.BIT_XOR,
            ast.BitAnd: lcapi.BinaryOp.BIT_AND,
        }.get(type(node.op))
        # ast.Pow, ast.MatMult is not supported
        if op is None:
            raise Exception(f'Unsupported binary operation: {type(node.op)}')
        node.dtype = deduce_binary_type(node.op, node.left.dtype, node.right.dtype)
        node.expr = lcapi.builder().binary(node.dtype, op, node.left.expr, node.right.expr)

    @staticmethod
    def build_Compare(ctx, node):
        if len(node.comparators)!=1:
            raise Exception('chained comparison not supported yet.')
        build(ctx, node.left)
        build(ctx, node.comparators[0])
        op = {
            ast.Eq: lcapi.BinaryOp.EQUAL,
            ast.NotEq: lcapi.BinaryOp.NOT_EQUAL,
            ast.Lt: lcapi.BinaryOp.LESS,
            ast.LtE: lcapi.BinaryOp.LESS_EQUAL,
            ast.Gt: lcapi.BinaryOp.GREATER,
            ast.GtE: lcapi.BinaryOp.GREATER_EQUAL
        }.get(type(node.ops[0])) # TODO ops
        if op is None:
            raise Exception(f'Unsupported compare operation: {type(node.op)}')
        # TODO support chained comparison
        node.dtype = lcapi.Type.from_("bool")
        node.expr = lcapi.builder().binary(node.dtype, op, node.left.expr, node.comparators[0].expr)

    @staticmethod
    def build_BoolOp(ctx, node):
        # should be short-circuiting
        if len(node.values)!=2:
            raise Exception('chained bool op not supported yet. use brackets instead.')
        for x in node.values:
            build(ctx, x)
        op = {
            ast.And: lcapi.BinaryOp.AND,
            ast.Or:  lcapi.BinaryOp.OR
        }.get(type(node.op))
        node.dtype = lcapi.Type.from_("bool")
        node.expr = lcapi.builder().binary(node.dtype, op, node.values[0].expr, node.values[1].expr)

    @staticmethod
    def build_If(ctx, node):
        # condition
        build(ctx, node.test)
        ifstmt = lcapi.builder().if_(node.test.expr)
        # true branch
        lcapi.builder().push_scope(ifstmt.true_branch())
        for x in node.body:
            build(ctx, x)
        lcapi.builder().pop_scope(ifstmt.true_branch())
        # false branch
        lcapi.builder().push_scope(ifstmt.false_branch())
        for x in node.orelse:
            build(ctx, x)
        lcapi.builder().pop_scope(ifstmt.false_branch())

    @staticmethod
    def build_For(ctx, node):
        raise Exception('for loop not supported yet')
        pass

    @staticmethod
    def build_While(ctx, node):
        loopstmt = lcapi.builder().loop_()
        lcapi.builder().push_scope(loopstmt.body())
        # condition
        build(ctx, node.test)
        ifstmt = lcapi.builder().if_(node.test.expr)
        lcapi.builder().push_scope(ifstmt.false_branch())
        lcapi.builder().break_()
        lcapi.builder().pop_scope(ifstmt.false_branch())
        # body
        for x in node.body:
            build(ctx, x)
        lcapi.builder().pop_scope(loopstmt.body())

    @staticmethod
    def build_Break(ctx, node):
        lcapi.builder().break_()

    @staticmethod
    def build_Continue(ctx, node):
        lcapi.builder().continue_()
    
build = ASTVisitor()
