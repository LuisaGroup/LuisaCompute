from pyluisa.compute.ast import *


def kernel(f):
    def wrapper(*args, **kwargs):
        with Builder(Function.Tag.KERNEL) as builder:
            f(*args, **kwargs)
    
    return wrapper


@kernel
def something(someone):
    print(f"Hello, {someone}!")


something("Mike")

