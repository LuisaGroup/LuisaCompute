from .dylibs import lcapi
from .func import func


class CallableLibrary:
    def __init__(self):
        self.callable_lib = lcapi.CallableLibrary()

    def add_callable(self, f: func, argtypes: tuple):
        compiled_func = f.get_compiled(1, True, argtypes)
        self.callable_lib.add_callable(f.__name__, compiled_func.builder)
    
    def save(self, path):
        self.callable_lib.serialize(path)
