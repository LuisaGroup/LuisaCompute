from ctypes import c_void_p, c_char_p, c_int, c_int32, c_uint32, c_int64, c_uint64, c_size_t
from .config import dll


dll.luisa_compute_context_create.restype = c_void_p
dll.luisa_compute_context_create.argtypes = [c_char_p]


def context_create(exe_path):
    return dll.luisa_compute_context_create(exe_path.encode())


dll.luisa_compute_context_destroy.restype = None
dll.luisa_compute_context_destroy.argtypes = [c_void_p]


def context_destroy(ctx):
    dll.luisa_compute_context_destroy(ctx)


dll.luisa_compute_context_runtime_directory.restype = c_char_p
dll.luisa_compute_context_runtime_directory.argtypes = [c_void_p]


def context_runtime_directory(ctx):
    return bytes(dll.luisa_compute_context_runtime_directory(ctx)).decode()


dll.luisa_compute_context_cache_directory.restype = c_char_p
dll.luisa_compute_context_cache_directory.argtypes = [c_void_p]


def context_cache_directory(ctx):
    return bytes(dll.luisa_compute_context_cache_directory(ctx)).decode()


dll.luisa_compute_device_create.restype = c_void_p
dll.luisa_compute_device_create.argtypes = [c_void_p, c_char_p, c_uint32]


def device_create(ctx, name, index):
    return dll.luisa_compute_device_create(ctx, name.encode(), index)


dll.luisa_compute_device_destroy.restype = None
dll.luisa_compute_device_destroy.argtypes = [c_void_p]


def device_destroy(device):
    dll.luisa_compute_device_destroy(device)


dll.luisa_compute_buffer_create.restype = c_uint64
dll.luisa_compute_buffer_create.argtypes = [c_void_p, c_size_t, c_uint64, c_uint32]


def buffer_create(device, size, heap_handle, index_in_heap):
    return dll.luisa_compute_buffer_create(device, size, heap_handle, index_in_heap)


dll.luisa_compute_buffer_destroy.restype = None
dll.luisa_compute_buffer_destroy.argtypes = [c_void_p, c_uint64, c_uint64, c_uint32]


def buffer_destroy(device, handle, heap_handle, index_in_heap):
    dll.luisa_compute_buffer_destroy(device, handle, heap_handle, index_in_heap)


dll.luisa_compute_texture_create.restype = c_uint64
dll.luisa_compute_texture_create.argtypes = [c_void_p, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint64, c_uint32]


def texture_create(device, format, dim, w, h, d, mips, sampler, heap, index_in_heap):
    return dll.luisa_compute_texture_create(device, format, dim, w, h, d, mips, sampler, heap, index_in_heap)


dll.luisa_compute_texture_destroy.restype = None
dll.luisa_compute_texture_destroy.argtypes = [c_void_p, c_uint64, c_uint64, c_uint32]


def texture_destroy(device, handle, heap_handle, index_in_heap):
    dll.luisa_compute_texture_destroy(device, handle, heap_handle, index_in_heap)


dll.luisa_compute_heap_create.restype = c_uint64
dll.luisa_compute_heap_create.argtypes = [c_void_p, c_size_t]


def heap_create(device, size):
    return dll.luisa_compute_heap_create(device, size)


dll.luisa_compute_heap_destroy.restype = None
dll.luisa_compute_heap_destroy.argtypes = [c_void_p, c_uint64]


def heap_destroy(device, handle):
    dll.luisa_compute_heap_destroy(device, handle)


dll.luisa_compute_stream_create.restype = c_uint64
dll.luisa_compute_stream_create.argtypes = [c_void_p]


def stream_create(device):
    return dll.luisa_compute_stream_create(device)


dll.luisa_compute_stream_destroy.restype = None
dll.luisa_compute_stream_destroy.argtypes = [c_void_p, c_uint64]


def stream_destroy(device, handle):
    dll.luisa_compute_stream_destroy(device, handle)


dll.luisa_compute_stream_synchronize.restype = None
dll.luisa_compute_stream_synchronize.argtypes = [c_void_p, c_uint64]


def stream_synchronize(device, handle):
    dll.luisa_compute_stream_synchronize(device, handle)


dll.luisa_compute_stream_dispatch.restype = None
dll.luisa_compute_stream_dispatch.argtypes = [c_void_p, c_uint64, c_void_p]


def stream_dispatch(device, handle, cmd_list):
    dll.luisa_compute_stream_dispatch(device, handle, cmd_list)


dll.luisa_compute_shader_create.restype = c_uint64
dll.luisa_compute_shader_create.argtypes = [c_void_p, c_void_p]


def shader_create(device, function):
    return dll.luisa_compute_shader_create(device, function)


dll.luisa_compute_shader_destroy.restype = None
dll.luisa_compute_shader_destroy.argtypes = [c_void_p, c_uint64]


def shader_destroy(device, handle):
    dll.luisa_compute_shader_destroy(device, handle)


dll.luisa_compute_event_create.restype = c_uint64
dll.luisa_compute_event_create.argtypes = [c_void_p]


def event_create(device):
    return dll.luisa_compute_event_create(device)


dll.luisa_compute_event_destroy.restype = None
dll.luisa_compute_event_destroy.argtypes = [c_void_p, c_uint64]


def event_destroy(device, handle):
    dll.luisa_compute_event_destroy(device, handle)


dll.luisa_compute_event_signal.restype = None
dll.luisa_compute_event_signal.argtypes = [c_void_p, c_uint64, c_uint64]


def event_signal(device, handle, stream):
    dll.luisa_compute_event_signal(device, handle, stream)


dll.luisa_compute_event_wait.restype = None
dll.luisa_compute_event_wait.argtypes = [c_void_p, c_uint64, c_uint64]


def event_wait(device, handle, stream):
    dll.luisa_compute_event_wait(device, handle, stream)


dll.luisa_compute_event_synchronize.restype = None
dll.luisa_compute_event_synchronize.argtypes = [c_void_p, c_uint64]


def event_synchronize(device, handle):
    dll.luisa_compute_event_synchronize(device, handle)


dll.luisa_compute_mesh_create.restype = c_uint64
dll.luisa_compute_mesh_create.argtypes = [c_void_p]


def mesh_create(device):
    return dll.luisa_compute_mesh_create(device)


dll.luisa_compute_mesh_destroy.restype = None
dll.luisa_compute_mesh_destroy.argtypes = [c_void_p, c_uint64]


def mesh_destroy(device, handle):
    dll.luisa_compute_mesh_destroy(device, handle)


dll.luisa_compute_accel_create.restype = c_uint64
dll.luisa_compute_accel_create.argtypes = [c_void_p]


def accel_create(device):
    return dll.luisa_compute_accel_create(device)


dll.luisa_compute_accel_destroy.restype = None
dll.luisa_compute_accel_destroy.argtypes = [c_void_p, c_uint64]


def accel_destroy(device, handle):
    dll.luisa_compute_accel_destroy(device, handle)


dll.luisa_compute_command_list_create.restype = c_void_p
dll.luisa_compute_command_list_create.argtypes = []


def command_list_create():
    return dll.luisa_compute_command_list_create()


dll.luisa_compute_command_list_append.restype = None
dll.luisa_compute_command_list_append.argtypes = [c_void_p, c_void_p]


def command_list_append(list, command):
    dll.luisa_compute_command_list_append(list, command)
