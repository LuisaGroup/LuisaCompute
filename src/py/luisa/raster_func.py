# from .func import *
from .raster import AppData
from .meshformat import MeshFormat
from .func import func
from .globalvars import get_global_device

def save_raster_shader(mesh_format: MeshFormat, vertex: func, pixel: func, vert_argtypes, pixel_argtypes, name: str,
                       async_builder: bool = True):
    vert_f = vertex.get_compiled(2, False, (AppData,) + vert_argtypes)
    pixel_f = pixel.get_compiled(2, False, (vert_f.return_type,) + pixel_argtypes)
    device = get_global_device().impl()
    check_val = device.check_raster_shader(vert_f.function, pixel_f.function)
    if (check_val > 0):
        if (check_val == 1):
            raise TypeError("Vertex return type unmatch with pixel's first arg's type.")
        elif (check_val == 2):
            raise TypeError("Illegal vertex to pixel struct type.")
        elif (check_val == 3):
            raise TypeError("Pixel shader's output required less than 8.")
        elif (check_val == 4):
            raise TypeError("Pixel shader's return type illegal.")
        elif (check_val == 5):
            raise TypeError("Vertex or pixel shader is not callable.")
        else:
            raise TypeError("Vertex shader's first argument must be AppData type.")
    if async_builder:
        device.save_raster_shader_async(mesh_format.handle, vert_f.builder, pixel_f.builder, name)
    else:
        device.save_raster_shader(mesh_format.handle, vert_f.function, pixel_f.function, name)
