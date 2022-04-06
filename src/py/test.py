import inspect
import ast
import astpretty


def deduce_literal_type(value):
    if isinstance(value, int):
        return lcapi.Type.from_("int")
    if isinstance(value, float):
        return lcapi.Type.from_("float")
    if isinstance(value, bool):
        return lcapi.Type.from_("bool")
    raise Exception(f'unrecognized literal type: {value}')
    
def deduce_unary_type(op, dtype):
    # TODO: Type check
    return dtype
    
def deduce_binary_type(op, dtype1, dtype2):
    # TODO: Type check
    # TODO: upcast
    return dtype1


# build context
local_variable = {}
closure_variable = {}



class Buffer:
    def __init__(self, size, dtype):
        if not dtype in {int, float, bool}:
            raise Exception('invalid buffer dtype')
        self.dtype = dtype
        self.size = size
        self.bytesize = size * lcapi.Type.from_(dtype.__name__).size()
        self.handle = device.impl().create_buffer(self.bytesize)

    def async_copy_from(self, arr): # arr: numpy array
        assert arr.size * arr.itemsize == self.bytesize
        ulcmd = lcapi.BufferUploadCommand.create(self.handle, 0, self.bytesize, arr)
        stream.add(ulcmd)

    def async_copy_to(self, arr): # arr: numpy array
        assert arr.size * arr.itemsize == self.bytesize
        dlcmd = lcapi.BufferDownloadCommand.create(self.handle, 0, self.bytesize, arr)
        stream.add(dlcmd)

    def read(self, idx):
        raise Exception("Method can only be called in Luisa kernel / callable")

    def write(self, idx):
        raise Exception("Method can only be called in Luisa kernel / callable")



class ASTVisitor:
    def __call__(self, node):
        method = getattr(self, 'build_' + node.__class__.__name__, None)
        if method is None:
            raise Exception(f'Unsupported node {node}:\n{astpretty.pformat(node)}')
        return method(node)

    @staticmethod
    def build_FunctionDef(node):
        if node.returns is not None:
            raise Exception('Return value is not supported')
        if len(node.args.args)!=0 or node.args.vararg is not None or node.args.kwarg is not None:
            raise Exception('Arguments are not supported')
        for x in node.body: # build over a list
            build(x)

    @staticmethod
    def build_Expr(node):
        print("!!!!!!!!!!!!!!!!!!!", isinstance(node.value, ast.Call))
        if isinstance(node.value, ast.Call):
            build(node.value)
        else:
            print("WARNING: Expr discarded")

    @staticmethod
    def build_Call(node):
        for x in node.args:
            build(x)
        if node.func.__class__.__name__ == "Name": # static function
            # TODO check for builtins
            pass
        elif node.func.__class__.__name__ == "Attribute": # class method
            build(node.func.value)
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
            if return_type is None:
                node.dtype = None
                lcapi.builder().call(builtin_op, [node.func.value.ptr] + [x.ptr for x in node.args])
            else: # function call has return value
                node.dtype = return_type
                node.ptr = lcapi.builder().call(return_type, builtin_op, [node.func.value.ptr] + [x.ptr for x in node.args])
        else:
            raise Exception('unrecognized call func type')

    @staticmethod
    def build_Name(node):
        # Note: in Python all local variables are function-scoped
        if node.id in local_variable:
            node.dtype, node.ptr = local_variable[node.id]
        else:
            val = closure_variable.get(node.id)
            if val is None:
                node.ptr = None
                return
            if type(val) in {int, float, bool}:
                node.dtype = deduce_literal_type(val)
                node.ptr = lcapi.builder().literal(node.dtype, val)
                return
            if type(val) is Buffer:
                node.dtype = lcapi.Type.from_("buffer<" + val.dtype.__name__ + ">")
                node.ptr = lcapi.builder().buffer_binding(node.dtype, val.handle, 0)
                return
            raise Exception("unrecognized closure var type")

    @staticmethod
    def build_Constant(node):
        node.dtype = deduce_literal_type(node.value)
        node.ptr = lcapi.builder().literal(node.dtype, node.value)

    @staticmethod
    def build_Assign(node):
        if len(node.targets)!=1:
            raise Exception('Tuple assignment not supported')
        build(node.targets[0])
        build(node.value)
        # create local variable if it doesn't exist yet
        if node.targets[0].ptr is None:
            dtype = node.value.dtype # deduced data type
            node.targets[0].ptr = lcapi.builder().local(dtype)
            # store type & ptr info into name
            local_variable[node.targets[0].id] = (dtype, node.targets[0].ptr)
            # all local variables are function scope
        lcapi.builder().assign(node.targets[0].ptr, node.value.ptr)

    @staticmethod
    def build_UnaryOp(node):
        build(node.operand)
        node.dtype = deduce_unary_type(node.op, node.operand.dtype)
        node.ptr = lcapi.builder().unary(node.dtype, op, node.operand.ptr)

    @staticmethod
    def build_BinOp(node):
        build(node.left)
        build(node.right)
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
        node.ptr = lcapi.builder().binary(node.dtype, op, node.left.ptr, node.right.ptr)

    @staticmethod
    def build_Compare(node):
        if len(node.comparators)!=1:
            raise Exception('chained comparison not supported yet.')
        build(node.left)
        build(node.comparators[0])
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
        node.ptr = lcapi.builder().binary(node.dtype, op, node.left.ptr, node.comparators[0].ptr)

    @staticmethod
    def build_BoolOp(node):
        # should be short-circuiting
        if len(node.values)!=2:
            raise Exception('chained bool op not supported yet. use brackets instead.')
        for x in node.values:
            build(x)
        op = {
            ast.And: lcapi.BinaryOp.AND,
            ast.Or:  lcapi.BinaryOp.OR
        }.get(type(node.op))
        node.dtype = lcapi.Type.from_("bool")
        node.ptr = lcapi.builder().binary(node.dtype, op, node.values[0].ptr, node.values[1].ptr)

    @staticmethod
    def build_If(node):
        # condition
        build(node.test)
        ifstmt = lcapi.builder().if_(node.test.ptr)
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
    def build_For(node):
        raise Exception('for loop not supported yet')
        pass

    @staticmethod
    def build_While(node):
        loopstmt = lcapi.builder().loop_()
        lcapi.builder().push_scope(loopstmt.body())
        # condition
        build(node.test)
        ifstmt = lcapi.builder().if_(node.test.ptr)
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


# ============= test script ================

import numpy as np
import lcapi
context = lcapi.Context(lcapi.FsPath(""))
device = context.create_device("ispc")
stream = device.create_stream()


def test_astgen():
    lcapi.builder().set_block_size(256,1,1)
    int_type = lcapi.Type.from_("int")
    buf_type = lcapi.Type.from_("buffer<int>")
    buf = lcapi.builder().buffer_binding(buf_type, buffer_handle, 0)
    idx3 = lcapi.builder().dispatch_id()
    idx = lcapi.builder().access(int_type, idx3, lcapi.builder().literal(int_type, 0))
    index = lcapi.builder().literal(int_type, 1)
    value = lcapi.builder().literal(int_type, 42)
    lcapi.builder().call(lcapi.CallOp.BUFFER_WRITE, [buf, idx, idx])


# user code

b = Buffer(100, int)

def f():
    a = b.read(0)
    b.write(a+1, 42)

# generate AST
tree = ast.parse(inspect.getsource(f))
_closure_vars = inspect.getclosurevars(f)
closure_variable = {
    **_closure_vars.globals,
    **_closure_vars.nonlocals,
    **_closure_vars.builtins
}
local_variable = {}

def astgen():
    print(astpretty.pformat(tree.body[0]))
    lcapi.builder().set_block_size(256,1,1)
    build(tree.body[0])



# ============= test script ================

arr = np.ones(100, dtype='int32')
arr1 = np.zeros(100, dtype='int32')

# upload command
b.async_copy_from(arr)

# compile kernel
builder = lcapi.FunctionBuilder.define_kernel(astgen)
func = builder.function()
shader_handle = device.impl().create_shader(func)
# call kernel
command = lcapi.ShaderDispatchCommand.create(shader_handle, func)
command.set_dispatch_size(1,1,1)
stream.add(command)

# download command
b.async_copy_to(arr1)

stream.synchronize()

print(arr1)
