import lcapi
from . import globalvars
# from .types import ref

from .func import func, save_raster_shader
from .mathtypes import *
from .array import array, ArrayType, SharedArrayType
from .struct import struct, StructType
from .buffer import buffer, Buffer, BufferType, DispatchIndirectBuffer
from .texture2d import texture2d, Texture2D, Texture2DType
from .texture3d import texture3d, Texture3D, Texture3DType
from lcapi import PixelStorage
from .gui import GUI

from .printer import Printer
from .accel import Ray, Hit, Accel
from .bindless import bindless_array, BindlessArray
from .util import RandomSampler
from .meshformat import MeshFormat

from lcapi import log_level_verbose, log_level_info, log_level_warning, log_level_error
from os.path import realpath


def init(backend_name = None, shader_path = None, support_gui = True):
    if globalvars.device != None:
        return
    if globalvars.context == None:
        globalvars.context = lcapi.Context(realpath(lcapi.__file__))
    # auto select backend if not specified
    backends = globalvars.context.installed_backends()
    assert len(backends) > 0
    if backend_name == None:
        print(f"detected backends: {backends}. Selecting {backends[0]}.")
        backend_name = backends[0]
    elif backend_name not in backends:
        raise NameError(f"backend '{backend_name}' is not installed.")
    globalvars.device = globalvars.context.create_device(backend_name)
    globalvars.stream = globalvars.device.create_stream(support_gui)
    globalvars.stream_support_gui = support_gui
    if shader_path != None:
        globalvars.context.set_shader_path(shader_path)

def init_headless(backend_name = None, shader_path = None):
    if globalvars.device != None:
        return
    if globalvars.context == None:
        globalvars.context = lcapi.Context(realpath(lcapi.__file__))
    # auto select backend if not specified
    backends = globalvars.context.installed_backends()
    assert len(backends) > 0
    if backend_name == None:
        print(f"detected backends: {backends}. Selecting {backends[0]}.")
        backend_name = backends[0]
    elif backend_name not in backends:
        raise NameError(f"backend '{backend_name}' is not installed.")
    globalvars.device = globalvars.context.create_headless_device(backend_name)
    if shader_path != None:
        globalvars.context.set_shader_path(shader_path)
def del_device():
    if globalvars.device != None:
        del globalvars.device
        globalvars.device = None

def synchronize(stream = None):
    if stream is None:
        stream = globalvars.stream
    stream.synchronize()

def execute(stream = None):
    if stream is None:
        stream = globalvars.stream
    stream.execute()