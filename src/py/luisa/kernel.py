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


def create_arg_expr(dtype, allow_ref):
    # Note: scalars are always passed by value
    #       vectors/matrices/arrays/structs are passed by reference if (allow_ref==True)
    #       resources are always passed by reference (without specifying ref)
    lctype = to_lctype(dtype) # also checking that it's valid data dtype
    if lctype.is_scalar():
        return lcapi.builder().argument(lctype)
    elif lctype.is_vector() or lctype.is_matrix() or lctype.is_array() or lctype.is_structure():
        return lcapi.builder().reference(lctype)
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
    # returns (LCfunction, shader_handle)
    # shader_handle is None if it's callable
    def compile(self, argtypes):
        # get python AST & context
        self.sourcelines = sourceinspect.getsourcelines(self.func)[0]
        self.tree = ast.parse(sourceinspect.getsource(self.func))
        _closure_vars = inspect.getclosurevars(self.func)
        self.closure_variable = {
            **_closure_vars.globals,
            **_closure_vars.nonlocals,
            **_closure_vars.builtins
        }
        self.local_variable = {} # dict: name -> VariableInfo(dtype, expr, is_arg)
        self.parameters = inspect.signature(self.func).parameters
        if len(args) != len(self.parameters):
            raise Exception(f"calling kernel with {len(args)} arguments ({len(self.params)} expected).")
        self.uses_printer = False
        # compile callback
        def astgen():
            if not self.is_device_callable:
                lcapi.builder().set_block_size(256,1,1)
            # get parameters
            for idx, name in enumerate(self.parameters):
                dtype = argtypes[idx]
                expr = create_arg_expr(dtype, allow_ref = self.is_device_callable)
                self.local_variable[name] = VariableInfo(dtype, expr, is_arg=True)
            # push context & build function body AST
            top = globalvars.current_kernel
            globalvars.current_kernel = self
            astbuilder.build(self.tree.body[0])
            globalvars.current_kernel = top
        # compile
        if is_device_callable:
            self.builder = lcapi.FunctionBuilder.define_callable(astgen)
        else:
            self.builder = lcapi.FunctionBuilder.define_kernel(astgen)
        # get LuisaCompute function
        function = self.builder.function()
        if is_device_callable:
            return function, None
        # compile shader
        shader_handle = get_global_device().impl().create_shader(self.lcfunction)
        return function, shader_handle


    # looks up arg_type_tuple; compile if not existing
    # argtypes: tuple of dtype
    # returns (function, shader_handle)
    def get_compiled(self, argtypes):
        if argtypes not in self.compiled_results:
            self.compiled_results[argtypes] = self.compile(argtypes)
        return self.compiled_results[argtypes]


    # dispatch shader to stream if it's kernel
    # callables can't be called directly
    def __call__(self, *args, dispatch_size, stream = None):
        if self.is_device_callable:
            raise TypeError("callable can't be called on host")
        if stream is None:
            stream = globalvars.stream
        # get types of arguments and compile
        argtypes = tuple(dtype_of(a) for a in args)
        lcfunction, shader_handle = self.get_compiled(argtypes)
        # create command
        command = lcapi.ShaderDispatchCommand.create(shader_handle, lcfunction)
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
        if self.uses_printer: # assume that this property doesn't change with argtypes
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
    return add_method
