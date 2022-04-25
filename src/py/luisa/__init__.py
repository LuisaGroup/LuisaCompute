import inspect
import ast
import astpretty
import struct

import lcapi
from . import globalvars
from .globalvars import get_global_device
from .types import dtype_of, to_lctype, ref
from .buffer import Buffer, BufferType
from .arraytype import ArrayType
from .structtype import StructType
from . import astbuilder
from .mathtypes import *
from .printer import Printer


def init(backend_name = None):
    globalvars.context = lcapi.Context(lcapi.FsPath("."))
    # auto select backend if not specified
    if backend_name == None:
        backends = globalvars.context.installed_backends()
        assert len(backends) > 0
        print("detected backends:", backends, "selecting first one.")
        backend_name = backends[0]
    globalvars.device = globalvars.context.create_device(backend_name)
    globalvars.stream = globalvars.device.create_stream()
    globalvars.printer = Printer()


def create_param_exprs(params, allow_ref = False):
    # supports positional arguments only
    l = [] # (name, dtype, expr)
    for name in params:
        anno = params[name].annotation
        if anno == None:
            raise Exception("arguments must be annotated")
        if type(anno) is ref:
            if not allow_ref:
                raise Exception("reference is only supported as type of callable arguments")
            lctype = to_lctype(anno.dtype) # also checking that it's valid dtype
            if lctype.is_basic() or lctype.is_array() or lctype.is_structure():
                l.append((name, anno.dtype, lcapi.builder().reference(lctype)))
            else:
                raise Exception(f"type {anno.dtype} can not be referenced")
        else:
            lctype = to_lctype(anno) # also checking that it's valid dtype
            if lctype.is_basic() or lctype.is_array() or lctype.is_structure():
                # uniform argument
                l.append((name, anno, lcapi.builder().argument(lctype)))
            elif lctype.is_buffer():
                l.append((name, anno, lcapi.builder().buffer(lctype)))
            elif lctype.is_texture():
                l.append((name, anno, lcapi.builder().texture(lctype)))
            elif lctype.is_bindless_array():
                l.append((name, anno, lcapi.builder().bindless_array()))
            elif lctype.is_accel():
                l.append((name, anno, lcapi.builder().accel()))
            else:
                raise Exception("unsupported argument annotation")
    return l

class kernel:
    # creates a luisa kernel with given function
    def __init__(self, func, is_device_callable = False):
        device = get_global_device()
        # get python AST & context
        self.tree = ast.parse(inspect.getsource(func))
        self.original_func = func
        _closure_vars = inspect.getclosurevars(func)
        self.closure_variable = {
            **_closure_vars.globals,
            **_closure_vars.nonlocals,
            **_closure_vars.builtins
        }
        self.local_variable = {} # dict: name -> (dtype, expr)
        self.is_device_callable = is_device_callable

        self.parameters = inspect.signature(func).parameters

        def astgen():
            # print(astpretty.pformat(self.tree.body[0]))
            if not self.is_device_callable:
                lcapi.builder().set_block_size(256,1,1)
            # get parameters
            self.params = create_param_exprs(self.parameters, allow_ref = is_device_callable)
            for name, dtype, expr in self.params:
                self.local_variable[name] = dtype, expr
            # build function body AST
            astbuilder.build(self, self.tree.body[0])

        if is_device_callable:
            self.builder = lcapi.FunctionBuilder.define_callable(astgen)
        else:
            self.builder = lcapi.FunctionBuilder.define_kernel(astgen)
        # Note: self.params[*][2] (expr) is invalidated

        self.func = self.builder.function()
        # compile shader
        if not is_device_callable:
            self.shader_handle = device.impl().create_shader(self.func)


    # dispatch shader to stream
    def __call__(self, *args, dispatch_size, stream = None):
        if self.is_device_callable:
            raise Exception("callable can't be called on host")
        if stream is None:
            stream = globalvars.stream
        command = lcapi.ShaderDispatchCommand.create(self.shader_handle, self.func)
        # check & push arguments
        if len(args) != len(self.params):
            raise Exception(f"calling kernel with {len(args)} arguments ({len(self.params)} expected).")
        for argid, arg in enumerate(args):
            dtype = self.params[argid][1]
            assert dtype_of(arg) == dtype
            lctype = to_lctype(dtype)
            if lctype.is_basic():
                # TODO argument type cast? (e.g. int to uint)
                command.encode_uniform(lcapi.to_bytes(arg), lctype.size(), lctype.alignment())
            elif lctype.is_array() or lctype.is_structure():
                command.encode_uniform(arg.to_bytes(), lctype.size(), lctype.alignment())
            elif lctype.is_buffer():
                command.encode_buffer(arg.handle, 0)
            else:
                assert False

        globalvars.printer.reset()
        command.set_dispatch_size(*dispatch_size)
        stream.add(command)
        globalvars.printer.final_print() # Note: This will FORCE synchronize
        # Note: need to test overhead of printer operations & synchronization


def callable(func):
    return kernel(func, is_device_callable = True)

def synchronize(stream = None):
    if stream is None:
        stream = globalvars.stream
    stream.synchronize()

