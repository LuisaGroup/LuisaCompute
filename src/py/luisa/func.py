try:
    import sourceinspect
except ImportError:
    print('sourceinspect not installed. This may cause issues in interactive mode (REPL).')
    import inspect as sourceinspect
    # need sourceinspect for getting source. see (#10)
import inspect
import ast

from .dylibs import lcapi
from . import globalvars, astbuilder
from .globalvars import get_global_device
from .types import dtype_of, to_lctype, implicit_covertable, basic_dtypes, uint
from .astbuilder import VariableInfo
import textwrap
from pathlib import Path
import sys


def create_arg_expr(dtype, allow_ref):
    # Note: scalars are always passed by value
    #       vectors/matrices/arrays/structs are passed by reference if (allow_ref==True)
    #       resources are always passed by reference (without specifying ref)
    lctype = to_lctype(dtype)  # also checking that it's valid data dtype
    if lctype.is_scalar() or lctype.is_vector() or lctype.is_matrix():
        return lcapi.builder().argument(lctype)
    elif lctype.is_array() or lctype.is_structure():
        if allow_ref:
            return lcapi.builder().reference(lctype)
        else:
            return lcapi.builder().argument(lctype)
    elif lctype.is_buffer() or lctype.is_custom_buffer():
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
        self.local_variable = {}  # dict: name -> VariableInfo(dtype, expr, is_arg)
        self.function = None
        self.shader_handle = None

    def __del__(self):
        if self.shader_handle is not None:
            device = get_global_device()
            if device is not None:
                device.impl().destroy_shader(self.shader_handle)

    def build_arguments(self, allow_ref: bool, arg_info=None):
        if arg_info is None:
            for idx, name in enumerate(self.func.parameters):
                if idx >= len(self.argtypes): break
                dtype = self.argtypes[idx]
                expr = create_arg_expr(dtype, allow_ref=allow_ref)
                self.local_variable[name] = VariableInfo(dtype, expr, is_arg=True)
        else:
            for idx, name in enumerate(self.func.parameters):
                if idx >= len(self.argtypes): break
                var_info = arg_info.get(idx)
                dtype = self.argtypes[idx]
                if var_info is not None:
                    self.local_variable[name] = var_info["var"]
                else:
                    expr = create_arg_expr(dtype, allow_ref=allow_ref)
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
        self.compiled_results = {}  # maps (arg_type_tuple) to (function, shader_handle)
        frameinfo = inspect.getframeinfo(inspect.stack()[1][0])
        self.filename = frameinfo.filename
        self.lineno = frameinfo.lineno

    def save(self, argtypes: tuple, name=None, async_build: bool = True, print_cpp_header = False):
        self.sourcelines = sourceinspect.getsourcelines(self.pyfunc)[0]
        self.sourcelines = [textwrap.fill(line, tabsize=4, width=9999) for line in self.sourcelines]
        self.tree = ast.parse(textwrap.dedent("\n".join(self.sourcelines)))
        self.parameters = inspect.signature(self.pyfunc).parameters
        if len(argtypes) > len(self.parameters):
            raise Exception(
                f"calling {self.__name__} with {len(argtypes)} arguments ({len(self.parameters)} or less expected).")
        annotation_type_check(self.__name__, self.parameters, argtypes)
        f = FuncInstanceInfo(self, True, argtypes)

        # build function callback
        def astgen():
            lcapi.builder().set_block_size(256, 1, 1)
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
        if name is None:
            name = self.__name__
        if async_build:
            get_global_device().impl().save_shader_async(f.builder, name)
        else:
            get_global_device().impl().save_shader(f.function, name)
        if print_cpp_header:
            front = '''#pragma once
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
'''
            type_idx = 0
            type_map = {}
            type_defines = []
            r = ""
            shader_path = Path(name.replace("\\","/"))
            shader_name = shader_path.name.split(".")[0]
            def get_value_type_name(dtype, r):
                if dtype in basic_dtypes:
                    if dtype in {float, int, bool}:
                        return dtype.__name__, r
                    return "luisa::" + dtype.__name__, r
                elif type(dtype).__name__ == "StructType":
                    name = type_map.get(arg)
                    if name == None:
                        name = "Arg" + str(type_idx)
                        type_map[arg] = name

                        r += f"struct {name} " + "{\n"
                        for idx, ele_type in arg._py_args.items():
                            ele_name, r = get_value_type_name(ele_type, r)
                            r += f"    {ele_name} {idx};\n"
                        r += "};\n"
                    return name, r
                else:
                    return None, r
            buffer_declared = False
            volume_declared = False
            image_declared = False
            for arg in argtypes:
                name, r = get_value_type_name(arg, r)
                if name:
                    type_defines.append(name)
                elif type(arg).__name__ == "Texture2DType":
                    dtype_name, r = get_value_type_name(arg.dtype, r)
                    if name == None:
                        name = f"Image<{dtype_name}>"
                        type_map[arg] = name
                        if not image_declared:
                            front += "#include <luisa/runtime/image.h>\n"
                            image_declared = True
                    type_defines.append("luisa::compute::" + name)
                elif type(arg).__name__ == "Texture3DType":
                    dtype_name, r = get_value_type_name(arg.dtype, r)
                    if name == None:
                        name = f"Volume<{dtype_name}>"
                        type_map[arg] = name
                        if not volume_declared:
                            front += "#include <luisa/runtime/volume.h>\n"
                            volume_declared = True
                    type_defines.append("luisa::compute::" + name)
                elif type(arg).__name__ == "BufferType":
                    dtype_name, r = get_value_type_name(arg.dtype, r)
                    if name == None:
                        name = f"Buffer<{dtype_name}>"
                        type_map[arg] = name
                        if not buffer_declared:
                            front += "#include <luisa/runtime/buffer.h>\n"
                            buffer_declared = True
                    type_defines.append("luisa::compute::" + name)
                elif arg.__name__ == "BindlessArray":
                    name = type_map.get(arg)
                    if name == None:
                        name = "BindlessArray"
                        type_map[arg] = name
                        front += "#include <luisa/runtime/bindless_array.h>\n"
                    type_defines.append("luisa::compute::" + name)
                elif arg.__name__ == "Accel":
                    name = type_map.get(arg)
                    if name == None:
                        name = "Accel"
                        type_map[arg] = name
                        front += "#include <luisa/runtime/rtx/accel.h>\n"
                    type_defines.append("luisa::compute::" + name)
                elif arg.__name__ == "IndirectDispatchBuffer":
                    name = type_map.get(arg)
                    if name == None:
                        name = "IndirectDispatchBuffer"
                        type_map[arg] = name
                        front += "#include <luisa/runtime/dispatch_buffer.h>\n"
                    type_defines.append("luisa::compute::" + name)
                else:
                    assert False
                    
            dimension = f.builder.dimension()
            func_declare = "" 
            func_declare += f"luisa::compute::Shader{dimension}D<"
            type_name = ""
            sz = 0
            for i in type_defines:
                type_name += f"{i}"
                sz += 1
                if sz != len(type_defines):
                    type_name += ", "
            func_declare += type_name + ">"
            shader_path = str(shader_path)
            if sys.platform == 'win32':
                shader_path = shader_path.replace("\\", "/")
            r += f"inline {func_declare} load" + "(luisa::compute::Device &device) {\n    return device.load_shader<" + str(dimension) + ", " + type_name + ">(\"" + shader_path + "\");\n}\n"
            return front + "namespace " + shader_name + ' {\n' +  r + '}// namespace ' + shader_name + '\n'

    # compiles an argument-type-specialized callable/kernel
    # returns FuncInstanceInfo
    def compile(self, func_type: int, allow_ref: bool, argtypes: tuple, arg_info=None):
        call_from_host = func_type == 0
        # get python AST & context
        self.sourcelines = sourceinspect.getsourcelines(self.pyfunc)[0]
        self.sourcelines = [textwrap.fill(line, tabsize=4, width=9999) for line in self.sourcelines]
        self.tree = ast.parse(textwrap.dedent("\n".join(self.sourcelines)))
        self.parameters = inspect.signature(self.pyfunc).parameters
        if len(argtypes) > len(self.parameters):
            raise Exception(
                f"calling {self.__name__} with {len(argtypes)} arguments ({len(self.parameters)} or less expected).")
        annotation_type_check(self.__name__, self.parameters, argtypes)
        f = FuncInstanceInfo(self, call_from_host, argtypes)

        # build function callback
        def astgen():
            if call_from_host:
                lcapi.builder().set_block_size(256, 1, 1)
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
        match func_type:
            case 0:
                f.builder = lcapi.FunctionBuilder.define_kernel(astgen)
            case 1:
                f.builder = lcapi.FunctionBuilder.define_callable(astgen)
            case 2:
                f.builder = lcapi.FunctionBuilder.define_raster_stage(astgen)
        f.function = f.builder.function()
        # compile shader
        if call_from_host:
            f.shader_handle = get_global_device().impl().create_shader(f.function)
        return f

    # looks up arg_type_tuple; compile if not existing
    # returns FuncInstanceInfo
    def get_compiled(self, func_type: int, allow_ref: bool, argtypes: tuple, arg_info=None):
        if (func_type,) + argtypes not in self.compiled_results:
            try:
                self.compiled_results[(func_type,) + argtypes] = self.compile(func_type, allow_ref, argtypes, arg_info)
            except Exception as e:
                if hasattr(e, "already_printed"):
                    # hide the verbose traceback in AST builder
                    e1 = CompileError(f"Failed to compile luisa.func '{self.__name__}'")
                    e1.func = self
                    raise e1 from None
                else:
                    raise
        return self.compiled_results[(func_type,) + argtypes]

    # dispatch shader to stream
    def __call__(self, *args, dispatch_size, stream=None):
        get_global_device()  # check device is initialized
        if stream is None:
            stream = globalvars.vars.stream
        # get 3D dispatch size
        is_buffer = False
        if type(dispatch_size) is int:
            dispatch_size = (dispatch_size, 1, 1)
        elif (type(dispatch_size) == tuple or type(dispatch_size) == list) and (len(dispatch_size) in (1, 2, 3)):
            dispatch_size = (*dispatch_size, *[1] * (3 - len(dispatch_size)))
        else:
            is_buffer = True
        # get types of arguments and compile
        argtypes = tuple(dtype_of(a) for a in args)
        f = self.get_compiled(func_type=0, allow_ref=False, argtypes=argtypes)
        # create command
        command = lcapi.ComputeDispatchCmdEncoder.create(f.function.argument_size(), f.shader_handle, f.function)
        # push arguments
        for a in args:
            lctype = to_lctype(dtype_of(a))
            if lctype.is_basic():
                command.encode_uniform(lcapi.to_bytes(a), lctype.size())
            elif lctype.is_array() or lctype.is_structure():
                command.encode_uniform(a.to_bytes(), lctype.size())
            elif lctype.is_buffer() or lctype.is_custom_buffer():
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
        if f.uses_printer:  # assume that this property doesn't change with argtypes
            globalvars.printer.final_print()
            # Note: printing will FORCE synchronize (#21)
            globalvars.printer.reset()
