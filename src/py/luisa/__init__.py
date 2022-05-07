import lcapi
from . import globalvars
from .types import ref

from .func import func
from .mathtypes import *
from .arraytype import ArrayType
from .structtype import StructType
from .buffer import buffer, Buffer, BufferType
from .texture2d import texture2d, Texture2D, Texture2DType
from lcapi import PixelStorage
from .gui import GUI

from .printer import Printer
from .accel import accel, Ray, Hit, Accel, Mesh
from .bindless import bindless_array, BindlessArray

from lcapi import log_level_verbose, log_level_info, log_level_warning, log_level_error
from os.path import realpath


def init(backend_name = None):
    globalvars.context = lcapi.Context(lcapi.FsPath(realpath(lcapi.__file__)))
    # auto select backend if not specified
    backends = globalvars.context.installed_backends()
    assert len(backends) > 0
    if backend_name == None:
        print(f"detected backends: {backends}. Selecting {backends[0]}.")
        backend_name = backends[0]
    elif backend_name not in backends:
        raise NameError(f"backend '{backend_name}' is not installed.")
    globalvars.device = globalvars.context.create_device(backend_name)
    globalvars.stream = globalvars.device.create_stream()
    globalvars.printer = Printer()


def synchronize(stream = None):
    if stream is None:
        stream = globalvars.stream
    stream.synchronize()

