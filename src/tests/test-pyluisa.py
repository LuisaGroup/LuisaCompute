from pyluisa.compute.ast import *
from pyluisa.compute.compile import *


def kernel(f):
    def wrapper(*args, **kwargs):
        with Builder(Function.Tag.KERNEL) as builder:
            f(*args, **kwargs)
            return builder.uid()
    
    return wrapper


@kernel
def something():
    Builder.current().thread_id()
    Builder.current().block_id()


def compile(f):
    f = Function.at(f())
    scratch = Scratch()
    codegen = CppCodegen(scratch)
    codegen.emit(f)
    print(scratch)


compile(something)
