import lcapi
from . import globalvars
from .types import ref

from .func import func
from .mathtypes import *
from .arraytype import ArrayType
from .structtype import StructType
from .buffer import Buffer, BufferType
from .buffer.Buffer import buffer
from .texture2d import Texture2D, Texture2DType
from .texture2d.Texture2D import texture2d
from lcapi import PixelStorage

from .printer import Printer
from .accel import Ray, Hit, Accel, Mesh
from .accel.Accel import accel
from .bindless import BindlessArray
from .bindless.BindlessArray import bindless_array

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

