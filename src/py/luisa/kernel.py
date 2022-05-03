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
from .structtype import StructType
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

# variables, information and compiled result are stored per kernel instance (argument type specialization)
class KernelInstanceInfo:
    def __init__(self, kernel):
        self.kernel = kernel
        self.__name__ = kernel.__name__
        self.sourcelines = kernel.sourcelines
        self.is_device_callable = kernel.is_device_callable
        self.uses_printer = False
        # self.return_type is not defined until a return statement is met
        _closure_vars = inspect.getclosurevars(kernel.func)
        self.closure_variable = {
            **_closure_vars.globals,
            **_closure_vars.nonlocals,
            **_closure_vars.builtins
        }
        self.local_variable = {} # dict: name -> VariableInfo(dtype, expr, is_arg)
        self.function = None
        self.shader_handle = None

    def build_arguments(self, argtypes):
        for idx, name in enumerate(self.kernel.parameters):
            dtype = argtypes[idx]
            expr = create_arg_expr(dtype, allow_ref = self.kernel.is_device_callable)
            self.local_variable[name] = VariableInfo(dtype, expr, is_arg=True)


class kernel:
    # creates a luisa kernel with given function
    # func: python function
    # is_device_callable: True if it's callable, False if it's kernel
    def __init__(self, func, is_device_callable = False):
        self.func = func
        self.is_device_callable = is_device_callable
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.compiled_results = {} # maps (arg_type_tuple) to (function, shader_handle)

    # compiles an argument-type-specialized callable/kernel
    # returns KernelInstanceInfo
    def compile(self, argtypes):
        # get python AST & context
        self.sourcelines = sourceinspect.getsourcelines(self.func)[0]
        self.tree = ast.parse(textwrap.dedent(sourceinspect.getsource(self.func)))
        self.parameters = inspect.signature(self.func).parameters
        if len(argtypes) != len(self.parameters):
            raise Exception(f"calling kernel with {len(argtypes)} arguments ({len(self.params)} expected).")
        annotation_type_check(self.__name__, self.parameters, argtypes)
        f = KernelInstanceInfo(self)
        # compile callback
        def astgen():
            if not self.is_device_callable:
                lcapi.builder().set_block_size(256,1,1)
            f.build_arguments(argtypes)
            # push context & build function body AST
            top = globalvars.current_context
            globalvars.current_context = f
            astbuilder.build(self.tree.body[0])
            globalvars.current_context = top
        # compile
        # Note: must retain the builder object
        if self.is_device_callable:
            f.builder = lcapi.FunctionBuilder.define_callable(astgen)
        else:
            f.builder = lcapi.FunctionBuilder.define_kernel(astgen)
        # get LuisaCompute function
        f.function = f.builder.function()
        # compile shader
        if not self.is_device_callable:
            f.shader_handle = get_global_device().impl().create_shader(f.function)
        return f


    # looks up arg_type_tuple; compile if not existing
    # argtypes: tuple of dtype
    # returns KernelInstanceInfo
    def get_compiled(self, argtypes):
        if argtypes not in self.compiled_results:
            self.compiled_results[argtypes] = self.compile(argtypes)
        return self.compiled_results[argtypes]


    # dispatch shader to stream if it's kernel
    # callables can't be called directly
    def __call__(self, *args, dispatch_size, stream = None):
        if self.is_device_callable:
            raise TypeError("callable can't be called on host")
        get_global_device() # check device is initialized
        if stream is None:
            stream = globalvars.stream
        # get types of arguments and compile
        argtypes = tuple(dtype_of(a) for a in args)
        f = self.get_compiled(argtypes)
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


def callable(func):
    return kernel(func, is_device_callable = True)


def callable_method(struct):
    assert type(struct) is StructType
    def add_method(func):
        name = func.__name__
        # check name collision
        if name in struct.idx_dict:
            raise NameError("struct method can't have same name as its data members")
        struct.idx_dict[name] = len(struct.membertype)
        struct.membertype.append(CallableType)
        # check first parameter
        params = list(inspect.signature(func).parameters.items())
        if len(params) == 0:
            raise TypeError("struct method must have at lease 1 argument (self)")
        anno = params[0][1].annotation
        if type(anno) != ref or anno.dtype != struct:
            raise TypeError("annotation of first argument must be luisa.ref(T) where T is the struct type")
        struct.method_dict[name] = callable(func)
        if name == '__init__' and getattr(struct.method_dict[name], 'return_type', None) != None:
            raise TypeError(f'__init__() should return None, not {struct.method_dict[name].return_type}')
        # FIXME
    return add_method
