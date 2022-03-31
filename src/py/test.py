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

local_variable = {}


class ASTVisitor:
    def __call__(self, node):
        method = getattr(self, 'build_' + node.__class__.__name__, None)
        if method is None:
            raise Exception(f'Unsupported node {node}:\n{astpretty.pformat(node)}')
        return method(node)
    
    def build_FunctionDef(self, node):
        if node.returns is not None:
            raise Exception('Return value is not supported')
        if len(node.args.args)!=0 or node.args.vararg is not None or node.args.kwarg is not None:
            raise Exception('Arguments are not supported')
        for x in node.body: # build over a list
            build(x)
            
    def build_Expr(self, node):
        print("WARNING: Expr discarded")
        # TODO: callable?
        
    def build_Name(self, node):
        # TODO: find name in scopes (node.id)
        if node.id in local_variable:
            node.dtype, node.ptr = local_variable[node.id]
        else:
            node.ptr = None
        
    def build_Constant(self, node):
        node.dtype = deduce_literal_type(node.value)
        node.ptr = lcapi.builder().literal(node.dtype, node.value)
        
    def build_Assign(self, node):
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
        
    def build_UnaryOp(self, node):
        build(node.operand)
        node.dtype = deduce_unary_type(node.op, node.operand.dtype)
        node.ptr = lcapi.builder().unary(node.dtype, op, node.operand.ptr)
        
    def build_BinOp(self, node):
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
        
    def build_Compare(self, node):
        if len(node.comparators)!=1:
            raise Exception('chained comparison not supported yet. use brackets instead.')
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

    def build_BoolOp(self, node):
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
    
    def build_If(self, node):
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
        
    def build_For(self, node):
        raise Exception('for loop not supported yet')
        pass
    
    def build_While(self, node):
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

    def build_Break(self, node):
        lcapi.builder().break_()

    def build_Continue(self, node):
        lcapi.builder().continue_()
    
build = ASTVisitor()


import numpy as np
import lcapi
context = lcapi.Context(lcapi.FsPath(""))
device = context.create_device("ispc")
stream = device.create_stream()

buffer_handle = device.impl().create_buffer(400)

print("BUFFER HANDLE: ", buffer_handle)

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

def f():
    a = 1 + 2
    if a < 4:
        b = 3.14

tree = ast.parse(inspect.getsource(f))
def astgen():
    print(astpretty.pformat(tree.body[0]))
    lcapi.builder().set_block_size(256,1,1)
    build(tree.body[0])

arr = np.ones(100, dtype='int32')
arr1 = np.zeros(100, dtype='int32')

ulcmd = lcapi.BufferUploadCommand.create(buffer_handle, 0, 400, arr)
stream.add(ulcmd)

builder = lcapi.FunctionBuilder.define_kernel(test_astgen)
func = builder.function()
shader_handle = device.impl().create_shader(func)
command = lcapi.ShaderDispatchCommand.create(shader_handle, func)
command.encode_pending_bindings()
command.set_dispatch_size(100,1,1)
stream.add(command)

dlcmd = lcapi.BufferDownloadCommand.create(buffer_handle, 0, 400, arr1)
stream.add(dlcmd)

stream.synchronize()

print(arr1)
