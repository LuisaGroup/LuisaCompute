import functools
import lcapi
from .mathtypes import *
from .builtin import set_block_size, synchronize_block
from .types import nameof, to_lctype


def with_signature(*signatures):
    signatures_printed = [', '.join([nameof(x) for x in signature]) for signature in signatures]

    def wrapper(function_builder):
        def match(func_builder, signature, *args) -> (bool, str):
            signature_printed = ', '.join([nameof(x) for x in signature])
            if len(args) != len(signature):
                return False, f"{nameof(func_builder)} takes exactly {len(signature)} arguments ({signature_printed}), {len(args)} given. "
            for arg, dtype in zip(args, signature):
                if arg.dtype != dtype:
                    given = ', '.join([nameof(x.dtype) for x in args])
                    return False, f"{nameof(func_builder)} expects ({signature}). Calling with ({given})"
            return True, "Everything's alright."

        @functools.wraps(function_builder)
        def decorated(*args):
            reason = "Everything's alright."
            for signature in signatures:
                matched, reason = match(function_builder, signature, args)
                if matched:
                    return function_builder(*args)
            raise TypeError(reason)


class BuiltinFunctionCall:
    def __call__(self, name, *args, **kwargs):
        pass

    @staticmethod
    def _invoke_dispatch_related(name, *args, **kwargs):
        expr = getattr(lcapi.builder(), name)()
        dtype = int3
        expr1 = lcapi.builder().call(to_lctype(dtype), lcapi.CallOp.MAKE_INT3, [expr])
        tmp = lcapi.builder().local(to_lctype(dtype))
        lcapi.builder().assign(tmp, expr1)
        return dtype, tmp

    @staticmethod
    def _invoke_make_vector_n(data_type, vector_length, *args, **kwargs):
        op = getattr(lcapi.CallOp, f"make_{data_type}{vector_length}".upper())
        dtype = getattr(lcapi, f'{data_type}{vector_length}')
        return dtype, lcapi.builder().call(to_lctype(dtype), op, [x.expr for x in args])

    @staticmethod
    @with_signature((int, int, int),)
    def invoke_set_block_size(_, *args, **kwargs):
        return set_block_size.builder(*args, **kwargs)

    @staticmethod
    @with_signature((),)
    def invoke_synchronize_block(_, *args, **kwargs):
        return synchronize_block.builder(*args, **kwargs)

    @staticmethod
    @with_signature((),)
    def invoke_thread_id(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_dispatch_related(name, *args, **kwargs)

    @staticmethod
    @with_signature((),)
    def invoke_block_id(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_dispatch_related(name, *args, **kwargs)

    @staticmethod
    @with_signature((),)
    def invoke_dispatch_id(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_dispatch_related(name, *args, **kwargs)

    @staticmethod
    @with_signature((),)
    def invoke_dispatch_size(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_dispatch_related(name, *args, **kwargs)

    @staticmethod
    @with_signature((int, int),)
    def invoke_make_int2(name, *args, **kwargs):
        return BuiltinFunctionCall._invoke_make_vector_n(int, 2, *args, **kwargs)