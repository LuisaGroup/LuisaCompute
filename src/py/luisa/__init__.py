import inspect
import ast
import astpretty
import struct

import lcapi
from . import globalvars
from .types import scalar_types
from .buffer import Buffer
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
        elif anno in scalar_types:
            dtype = scalar_types[anno]
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
        # push arguments
        for idx, arg in enumerate(args):
            # TODO check arg types
            if type(arg) is int:
                aa1 = struct.pack('i',arg)
                command.encode_uniform(aa1,4,4)# TODO
        command.set_dispatch_size(*dispatch_size)
        stream.add(command)
        if sync:
            stream.synchronize()

def synchronize(stream = None):
    if stream is None:
        stream = globalvars.stream
    stream.synchronize()

