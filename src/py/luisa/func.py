try:
    import sourceinspect
except ImportError:
    print('sourceinspect not installed. This may cause issues in interactive mode (REPL).')
    import inspect as sourceinspect
    # need sourceinspect for getting source. see (#10)
import inspect
import ast

import lcapi
from . import globalvars, astbuilder
from .globalvars import get_global_device
from .types import dtype_of, to_lctype, implicit_covertable
from .astbuilder import VariableInfo
from. meshformat import MeshFormat
import textwrap
import os
import sys
from .raster import appdata

def create_arg_expr(dtype, allow_ref):
    # Note: scalars are always passed by value
    #       vectors/matrices/arrays/structs are passed by reference if (allow_ref==True)
    #       resources are always passed by reference (without specifying ref)
    lctype = to_lctype(dtype) # also checking that it's valid data dtype
    if lctype.is_scalar():
        return lcapi.builder().argument(lctype)
    if lctype.is_vector() or lctype.is_matrix() or lctype.is_array() or lctype.is_structure() or lctype.is_custom():
        if allow_ref:
            return lcapi.builder().reference(lctype)
        else:
            return lcapi.builder().argument(lctype)
    elif lctype.is_buffer():
        return lcapi.builder().buffer(lctype)
    elif lctype.is_texture():
        return lcapi.builder().texture(lctype)
    elif lctype.is_bindless_array():
        return lcapi.builder().bindless_array()
    elif lctype.is_accel():
        return lcapi.builder().accel()
    else:
        assert False

# annotation can be used (but not required) to specify argument type
def annotation_type_check(funcname, parameters, argtypes):
    def anno_str(anno):
        if anno == inspect._empty:
            return ""
        if hasattr(anno, '__name__'):
            return ":" + anno.__name__
        return ":" + repr(anno)
    count = len(argtypes)
    for idx, name in enumerate(parameters):
        if idx >= count:
            break
        anno = parameters[name].annotation
        if anno != inspect._empty and not implicit_covertable(anno, argtypes[idx]):
            hint = funcname + '(' + ', '.join([n + anno_str(parameters[n].annotation) for n in parameters]) + ')'
            raise TypeError(f"argument '{name}' expects {anno}, got {argtypes[idx]}. calling {hint}")

# variables, information and compiled result are stored per func instance (argument type specialization)
class FuncInstanceInfo:
    def __init__(self, func, call_from_host, argtypes):
        self.func = func
        self.__name__ = func.__name__
        self.sourcelines = func.sourcelines
        self.uses_printer = False
        # self.return_type is not defined until a return statement is met
        self.call_from_host = call_from_host
        self.argtypes = argtypes
        _closure_vars = inspect.getclosurevars(func.pyfunc)
        self.closure_variable = {
            **_closure_vars.globals,
            **_closure_vars.nonlocals,
            **_closure_vars.builtins
        }
        self.local_variable = {} # dict: name -> VariableInfo(dtype, expr, is_arg)
        self.function = None
        self.shader_handle = None
    def __del__(self):
        if self.shader_handle != None:
            device = get_global_device()
            if device != None:
                device.impl().destroy_shader(self.shader_handle)
    def build_arguments(self, allow_ref: bool, arg_info=None):
        if arg_info == None:
            for idx, name in enumerate(self.func.parameters):
                if idx >= len(self.argtypes): break
                dtype = self.argtypes[idx]
                expr = create_arg_expr(dtype, allow_ref = allow_ref)
                self.local_variable[name] = VariableInfo(dtype, expr, is_arg=True)
        else:
            for idx, name in enumerate(self.func.parameters):
                if idx >= len(self.argtypes): break
                var_info = arg_info.get(idx)
                dtype = self.argtypes[idx]
                if var_info != None:
                    self.local_variable[name] = var_info["var"]
                else:
                    expr = create_arg_expr(dtype, allow_ref = allow_ref)
                    self.local_variable[name] = VariableInfo(dtype, expr, is_arg=True)
class CompileError(Exception):
    pass

class func:
    # creates a luisa function with given function
    # A luisa function can be run on accelarated device (CPU/GPU).
    # It can either be called in parallel by python code,
    # or be called by another luisa function.
    # pyfunc: python function
    def __init__(self, pyfunc):
        self.pyfunc = pyfunc
        self.__name__ = pyfunc.__name__
        self.compiled_results = {} # maps (arg_type_tuple) to (function, shader_handle)
        frameinfo = inspect.getframeinfo(inspect.stack()[1][0])
        self.filename = frameinfo.filename
        self.lineno = frameinfo.lineno
        
    def save(self, argtypes: tuple, name=None, async_build: bool=True):
        self.sourcelines = sourceinspect.getsourcelines(self.pyfunc)[0]
        self.sourcelines = [textwrap.fill(line, tabsize=4, width=9999) for line in self.sourcelines]
        self.tree = ast.parse(textwrap.dedent("\n".join(self.sourcelines)))
        self.parameters = inspect.signature(self.pyfunc).parameters
        if len(argtypes) > len(self.parameters):
            raise Exception(f"calling {self.__name__} with {len(argtypes)} arguments ({len(self.parameters)} or less expected).")
        annotation_type_check(self.__name__, self.parameters, argtypes)
        f = FuncInstanceInfo(self, True, argtypes)
        # build function callback
        def astgen():
            lcapi.builder().set_block_size(256,1,1)
            f.build_arguments(False)
            # push context & build function body AST
            top = globalvars.current_context
            globalvars.current_context = f
            try:
                lcapi.begin_analyzer()
                astbuilder.build(self.tree.body[0])
            finally:
                lcapi.end_analyzer()
                globalvars.current_context = top
        # build function
        # Note: must retain the builder object
        f.builder = lcapi.FunctionBuilder.define_kernel(astgen)
        f.function = f.builder.function()
        # compile shader
        if name == None:
            name = self.__name__
        if async_build:
            get_global_device().impl().save_shader_async(f.builder, name)
        else:
            get_global_device().impl().save_shader(f.function, name)
    # compiles an argument-type-specialized callable/kernel
    # returns FuncInstanceInfo
    def compile(self, call_from_host: bool, allow_ref: bool, argtypes: tuple, arg_info=None):
        # get python AST & context
        self.sourcelines = sourceinspect.getsourcelines(self.pyfunc)[0]
        self.sourcelines = [textwrap.fill(line, tabsize=4, width=9999) for line in self.sourcelines]
        self.tree = ast.parse(textwrap.dedent("\n".join(self.sourcelines)))
        self.parameters = inspect.signature(self.pyfunc).parameters
        if len(argtypes) > len(self.parameters):
            raise Exception(f"calling {self.__name__} with {len(argtypes)} arguments ({len(self.parameters)} or less expected).")
        annotation_type_check(self.__name__, self.parameters, argtypes)
        f = FuncInstanceInfo(self, call_from_host, argtypes)
        # build function callback
        def astgen():
            if call_from_host:
                lcapi.builder().set_block_size(256,1,1)
            f.build_arguments(allow_ref=allow_ref, arg_info=arg_info)
            # push context & build function body AST
            top = globalvars.current_context
            globalvars.current_context = f
            try:
                lcapi.begin_analyzer()
                astbuilder.build(self.tree.body[0])
            finally:
                lcapi.end_analyzer()
                globalvars.current_context = top
        # build function
        # Note: must retain the builder object
        if call_from_host:
            f.builder = lcapi.FunctionBuilder.define_kernel(astgen)
        else:
            f.builder = lcapi.FunctionBuilder.define_callable(astgen)
        f.function = f.builder.function()
        # compile shader
        if call_from_host:
            name = ".cache/" + os.path.basename(sys.argv[0]).replace(".py", "") + '-' + self.__name__
            f.shader_handle = get_global_device().impl().create_shader(f.function, name)
        return f

    # looks up arg_type_tuple; compile if not existing
    # returns FuncInstanceInfo
    def get_compiled(self, call_from_host: bool, allow_ref:bool, argtypes: tuple, arg_info=None):
        if (call_from_host,) + argtypes not in self.compiled_results:
            try:
                self.compiled_results[(call_from_host,) + argtypes] = self.compile(call_from_host, allow_ref, argtypes, arg_info)
            except Exception as e:
                if hasattr(e, "already_printed"):
                    # hide the verbose traceback in AST builder
                    e1 = CompileError(f"Failed to compile luisa.func '{self.__name__}'")
                    e1.func = self
                    raise e1 from None
                else:
                    raise
        return self.compiled_results[(call_from_host,) + argtypes]


    # dispatch shader to stream
    def __call__(self, *args, dispatch_size, stream = None):
        get_global_device() # check device is initialized
        if stream is None:
            stream = globalvars.stream
        # get 3D dispatch size
        is_buffer = False
        if type(dispatch_size) is int:
            dispatch_size = (dispatch_size,1,1)
        elif (type(dispatch_size) == tuple or type(dispatch_size) == list) and (len(dispatch_size) in (1,2,3)):
            dispatch_size = (*dispatch_size, *[1]*(3-len(dispatch_size)))
        else:
            is_buffer = True
        # get types of arguments and compile
        argtypes = tuple(dtype_of(a) for a in args)
        f = self.get_compiled(call_from_host=True, allow_ref=False, argtypes=argtypes)
        # create command
        command = lcapi.ComputeDispatchCmdEncoder.create(len(args), f.shader_handle, f.function)
        # push arguments
        for a in args:
            lctype = to_lctype(dtype_of(a))
            if lctype.is_basic():
                command.encode_uniform(lcapi.to_bytes(a), lctype.size())
            elif lctype.is_array() or lctype.is_structure():
                command.encode_uniform(a.to_bytes(), lctype.size())
            elif lctype.is_buffer():
                command.encode_buffer(a.handle, 0, a.bytesize)
            elif lctype.is_texture():
                command.encode_texture(a.handle, 0)
            elif lctype.is_bindless_array():
                command.encode_bindless_array(a.handle)
            elif lctype.is_accel():
                command.encode_accel(a.handle)
            else:
                assert False
        # dispatch
        if is_buffer:
            command.set_dispatch_buffer(dispatch_size.handle)
        else:
            command.set_dispatch_size(*dispatch_size)
        stream.add(command.build())
        if f.uses_printer: # assume that this property doesn't change with argtypes
            globalvars.printer.final_print()
            # Note: printing will FORCE synchronize (#21)
            globalvars.printer.reset()


def save_raster_shader(mesh_format: MeshFormat, vertex: func, pixel: func, vert_argtypes, pixel_argtypes, name: str, async_builder: bool=True):
    vert_f = vertex.get_compiled(False, False,(appdata,) + vert_argtypes)
    pixel_f = pixel.get_compiled(False, False, (vert_f.return_type, ) + pixel_argtypes)
    device = get_global_device().impl()
    check_val = device.check_raster_shader(vert_f.function, pixel_f.function)
    if (check_val > 0):
        if(check_val == 1):
            raise TypeError("Vertex return type unmatch with pixel's first arg's type.")
        elif(check_val == 2):
            raise TypeError("Illegal vertex to pixel struct type.")
        elif(check_val == 3):
            raise TypeError("Pixel shader's output required less than 8.")
        elif(check_val == 4):
            raise TypeError("Pixel shader's return type illegal.")
        elif(check_val == 5):
            raise TypeError("Vertex or pixel shader is not callable.")
        else:
            raise TypeError("Vertex shader's first argument must be appdata type.")
    if async_builder:
        device.save_raster_shader_async(mesh_format.handle, vert_f.builder, pixel_f.builder, name)
    else:
        device.save_raster_shader(mesh_format.handle, vert_f.function, pixel_f.function, name)