try:
    import sourceinspect
except ImportError:
    print('sourceinspect not installed. This may cause issues in interactive mode (REPL).')
    import inspect as sourceinspect
    # need sourceinspect for getting source. see (#10)
import inspect
import ast
import astpretty

import lcapi
from . import globalvars, astbuilder
from .globalvars import get_global_device
from .types import dtype_of, to_lctype, ref, CallableType
from .struct import StructType
from .astbuilder import VariableInfo
import textwrap


def create_arg_expr(dtype, allow_ref):
    # Note: scalars are always passed by value
    #       vectors/matrices/arrays/structs are passed by reference if (allow_ref==True)
    #       resources are always passed by reference (without specifying ref)
    lctype = to_lctype(dtype) # also checking that it's valid data dtype
    if lctype.is_scalar():
        return lcapi.builder().argument(lctype)
    elif lctype.is_vector() or lctype.is_matrix() or lctype.is_array() or lctype.is_structure():
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
    for idx, name in enumerate(parameters):
        anno = parameters[name].annotation
        if anno != inspect._empty and anno != argtypes[idx]:
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

    def build_arguments(self):
        for idx, name in enumerate(self.func.parameters):
            dtype = self.argtypes[idx]
            expr = create_arg_expr(dtype, allow_ref = (not self.call_from_host))
            self.local_variable[name] = VariableInfo(dtype, expr, is_arg=True)


class device_func:
    pass

class host_func:
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
        self.__doc__ = pyfunc.__doc__
        self.compiled_results = {} # maps (arg_type_tuple) to (function, shader_handle)

    # compiles an argument-type-specialized callable/kernel
    # returns FuncInstanceInfo
    def compile(self, call_from_host: bool, argtypes: tuple):
        # get python AST & context
        self.sourcelines = sourceinspect.getsourcelines(self.pyfunc)[0]
        self.tree = ast.parse(textwrap.dedent(sourceinspect.getsource(self.pyfunc)))
        self.parameters = inspect.signature(self.pyfunc).parameters
        if len(argtypes) != len(self.parameters):
            raise Exception(f"calling {self.__name__} with {len(argtypes)} arguments ({len(self.parameters)} expected).")
        annotation_type_check(self.__name__, self.parameters, argtypes)
        f = FuncInstanceInfo(self, call_from_host, argtypes)
        # build function callback
        def astgen():
            if call_from_host:
                lcapi.builder().set_block_size(256,1,1)
            f.build_arguments()
            # push context & build function body AST
            top = globalvars.current_context
            globalvars.current_context = f
            astbuilder.build(self.tree.body[0])
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
            f.shader_handle = get_global_device().impl().create_shader(f.function)
        return f


    # looks up arg_type_tuple; compile if not existing
    # returns FuncInstanceInfo
    def get_compiled(self, call_from_host: bool, argtypes: tuple):
        if (call_from_host,) + argtypes not in self.compiled_results:
            self.compiled_results[(call_from_host,) + argtypes] = self.compile(call_from_host, argtypes)
        return self.compiled_results[(call_from_host,) + argtypes]


    # dispatch shader to stream
    def __call__(self, *args, dispatch_size, stream = None):
        get_global_device() # check device is initialized
        if stream is None:
            stream = globalvars.stream
        # get types of arguments and compile
        argtypes = tuple(dtype_of(a) for a in args)
        f = self.get_compiled(call_from_host=True, argtypes=argtypes)
        # create command
        command = lcapi.ShaderDispatchCommand.create(f.shader_handle, f.function)
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
        command.set_dispatch_size(*dispatch_size)
        stream.add(command)
        if f.uses_printer: # assume that this property doesn't change with argtypes
            globalvars.printer.final_print()
            # Note: printing will FORCE synchronize (#21)
            globalvars.printer.reset()

