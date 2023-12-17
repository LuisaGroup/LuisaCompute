from .dylibs import lcapi


# callback: (level: string, message: string) -> None
def set_log_callback(callback):
    lcapi.set_log_callback(callback)


try:
    shell = get_ipython().__class__.__name__
    if shell != "TerminalInteractiveShell":
        from datetime import datetime


        def _default_log_callback(level, message):
            now = datetime.now()
            match level:
                case "D":
                    color, level = 96, "debug"
                case "I":
                    color, level = 92, "info"
                case "W":
                    color, level = 93, "warning"
                case _:
                    color, level = 91, "error"
            print(f"[{now}] [luisa] [\033[{color}m{level}\033[00m] {message}")


        set_log_callback(_default_log_callback)
except NameError:
    pass

from . import globalvars
from .types import half, short, ushort, half2, short2, ushort2, half3, short3, ushort3, half4, short4, ushort4

from .func import func
from .raster_func import save_raster_shader
from .mathtypes import *
from .array import array, ArrayType, SharedArrayType
from .struct import struct, StructType
from .buffer import buffer, Buffer, ByteBuffer, BufferType, ByteBufferType, IndirectDispatchBuffer
from .image2d import image2d, Image2D, Texture2DType
from .image3d import image3d, Image3D, Texture3DType
from .dylibs.lcapi import PixelStorage
from .gui import GUI

from .accel import Accel, make_ray, inf_ray, offset_ray_origin
from .hit import TriangleHit, CommittedHit, ProceduralHit
from .rayquery import RayQueryAllType, RayQueryAnyType, is_triangle, is_procedural, Ray
from .bindless import bindless_array, BindlessArray
from .util import RandomSampler
from .meshformat import MeshFormat

from .dylibs.lcapi import log_level_verbose, log_level_info, log_level_warning, log_level_error
from os.path import realpath
import platform
import sys
from os import environ
import inspect


def _select_backend(backends):
    if "LUISA_BACKEND" in environ:
        backend_name = environ["LUISA_BACKEND"].lower()
        if backend_name in backends:
            print(f"Detected backends: {backends}.",
                  f"Selecting {backend_name} according to environment variable `LUISA_BACKEND`.")
            return backend_name
        else:
            print(f"Detected backends: {backends}.",
                  f"Environment variable `LUISA_BACKEND` is set to {backend_name}, but it is not installed.",
                  f"LuisaCompute will select an alternative backend automatically.",
                  file=sys.stderr)
    platform_str = str(platform.platform()).lower()
    if platform_str.find("windows") >= 0:
        backend_name = "dx" if "dx" in backends else backends[0]
    elif platform_str.find("linux") >= 0:
        backend_name = "cuda" if "cuda" in backends else backends[0]
    elif platform_str.find("macos") >= 0:
        backend_name = "metal" if "metal" in backends else backends[0]
    else:
        backend_name = backends[0]
    print(f"Detected backends: {backends}. Selecting {backend_name}.")
    return backend_name


def verbose(fmt: str, *args, **kwargs):
    lcapi.log_verbose(fmt.format(*args, **kwargs))


def info(fmt: str, *args, **kwargs):
    lcapi.log_info(fmt.format(*args, **kwargs))


def warning(fmt: str, *args, **kwargs):
    lcapi.log_warning(fmt.format(*args, **kwargs))


def error(fmt: str, *args, **kwargs):
    lcapi.log_error(fmt.format(*args, **kwargs))


def _get_source_location_log_suffix():
    frame = inspect.currentframe().f_back.f_back
    return f" [{inspect.getfile(frame)}:{inspect.getlineno(frame)}]"


def verbose_with_location(fmt: str, *args, **kwargs):
    lcapi.log_verbose(fmt.format(*args, **kwargs) + _get_source_location_log_suffix())


def info_with_location(fmt: str, *args, **kwargs):
    lcapi.log_info(fmt.format(*args, **kwargs) + _get_source_location_log_suffix())


def warning_with_location(fmt: str, *args, **kwargs):
    lcapi.log_warning(fmt.format(*args, **kwargs) + _get_source_location_log_suffix())


def error_with_location(fmt: str, *args, **kwargs):
    lcapi.log_error(fmt.format(*args, **kwargs) + _get_source_location_log_suffix())


def init(backend_name=None, shader_path=None, support_gui=True):
    if globalvars.vars is not None:
        return
    globalvars.vars = globalvars.Vars()
    globalvars.vars.context = lcapi.Context(realpath(lcapi.__file__))
    # auto select backend if not specified
    backends = globalvars.vars.context.installed_backends()
    assert len(backends) > 0
    if backend_name is None:
        backend_name = _select_backend(backends)
    elif backend_name not in backends:
        raise NameError(f"backend '{backend_name}' is not installed.")
    globalvars.device = globalvars.vars.context.create_device(backend_name)
    globalvars.vars.stream = globalvars.device.create_stream(support_gui)
    globalvars.vars.stream_support_gui = support_gui
    if shader_path is not None:
        globalvars.vars.context.set_shader_path(shader_path)


def init_headless(backend_name=None, shader_path=None):
    if globalvars.vars is not None:
        return
    globalvars.vars = globalvars.Vars()
    if globalvars.vars.context is None:
        globalvars.vars.context = lcapi.Context(realpath(lcapi.__file__))
    # auto select backend if not specified
    backends = globalvars.vars.context.installed_backends()
    assert len(backends) > 0
    if backend_name is None:
        backend_name = _select_backend(backends)
    elif backend_name not in backends:
        raise NameError(f"backend '{backend_name}' is not installed.")
    globalvars.device = globalvars.vars.context.create_headless_device(backend_name)
    if shader_path is not None:
        globalvars.vars.context.set_shader_path(shader_path)


def del_device():
    if globalvars.vars is not None:
        del globalvars.vars


def synchronize(stream=None):
    if stream is None:
        stream = globalvars.vars.stream
    stream.synchronize()


def execute(stream=None):
    if stream is None:
        stream = globalvars.vars.stream
    stream.execute()
