import inspect
import ast
import astpretty
import struct

import lcapi
from . import globalvars
from .types import basic_types, types
from .buffer import Buffer
from .arraytype import ArrayType
from . import astbuilder


def init(backend_name):
    globalvars.context = lcapi.Context(lcapi.FsPath(""))
    globalvars.device = globalvars.context.create_device(backend_name)
    globalvars.stream = globalvars.device.create_stream()


def create_param_exprs(params):
    # supports positional arguments only
    l = [] # (name, dtype, expr)
    for name in params:
        anno = params[name].annotation
        if anno == None:
            raise Exception("arguments must be annotated")
        elif type(anno) is lcapi.Type:
            l.append((name, anno, lcapi.builder().argument(anno)))
        # TODO: elif type(anno) is ref:
        elif anno in basic_types:
            dtype = basic_types[anno]
            l.append((name, dtype, lcapi.builder().argument(dtype)))
        elif type(anno) is ArrayType:
            dtype = anno.luisa_type()
            l.append((name, dtype, lcapi.builder().argument(dtype)))
        elif type(anno) is str:
            dtype = lcapi.Type.from_(anno)
            if dtype.is_buffer():
                l.append((name, dtype, lcapi.builder().buffer(dtype)))
            elif dtype.is_texture():
                l.append((name, dtype, lcapi.builder().texture(dtype)))
            elif dtype.is_bindless_array():
                l.append((name, dtype, lcapi.builder().bindless_array()))
            elif dtype.is_accel():
                l.append((name, dtype, lcapi.builder().accel()))
            else:
                l.append((name, dtype, lcapi.builder().argument(dtype)))
        else:
            raise Exception("argument unsupported")
    return l

class kernel:
    # creates a luisa kernel with given function
    def __init__(self, func):
        # get python AST & context
        self.tree = ast.parse(inspect.getsource(func))
        _closure_vars = inspect.getclosurevars(func)
        self.closure_variable = {
            **_closure_vars.globals,
            **_closure_vars.nonlocals,
            **_closure_vars.builtins
        }
        self.local_variable = {}

        param_list = inspect.signature(func).parameters

        def astgen():
            print(astpretty.pformat(self.tree.body[0]))
            lcapi.builder().set_block_size(256,1,1)
            # get parameters
            self.params = create_param_exprs(param_list)
            for name, dtype, expr in self.params:
                self.local_variable[name] = dtype, expr
            # build function body AST
            astbuilder.build(self, self.tree.body[0])

        self.builder = lcapi.FunctionBuilder.define_kernel(astgen)
        self.func = self.builder.function()
        # compile shader
        self.shader_handle = globalvars.device.impl().create_shader(self.func)


    # dispatch shader to stream
    def __call__(self, *args, dispatch_size, sync = True, stream = None):
        if stream is None:
            stream = globalvars.stream
        command = lcapi.ShaderDispatchCommand.create(self.shader_handle, self.func)
        # check & push arguments
        if len(args) != len(self.params):
            raise Exception("")
        for argid, arg in enumerate(args):
            dtype = self.params[argid][1]
            if dtype.is_basic():
                # TODO argument type cast? (e.g. int to uint)
                assert basic_types[type(arg)] == dtype
                # command.encode_literal(arg)
                command.encode_uniform(lcapi.to_bytes(arg), dtype.size(), dtype.alignment())
            elif dtype.is_array():
                assert len(arg) == dtype.dimension()
                packed_bytes = b''
                for x in arg:
                    assert basic_types[type(x)] == dtype.element()
                    packed_bytes += lcapi.to_bytes(x)
                command.encode_uniform(packed_bytes, dtype.size(), dtype.alignment())
            elif dtype.is_buffer():
                print(type(arg), dtype.element().description(), arg.dtype.description())
                assert type(arg) is Buffer and dtype.element() == arg.dtype
                command.encode_buffer(arg.handle, 0)
            else:
                assert False

        command.set_dispatch_size(*dispatch_size)
        stream.add(command)
        if sync:
            stream.synchronize()

def synchronize(stream = None):
    if stream is None:
        stream = globalvars.stream
    stream.synchronize()

