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
dll.luisa_compute_device_destroy.argtypes = [c_void_p, c_void_p]


def device_destroy(ctx, device):
    dll.luisa_compute_device_destroy(ctx, device)


dll.luisa_compute_buffer_create.restype = c_uint64
dll.luisa_compute_buffer_create.argtypes = [c_void_p, c_size_t, c_uint64, c_uint32]


def buffer_create(device, size, heap_handle, index_in_heap):
    return dll.luisa_compute_buffer_create(device, size, heap_handle, index_in_heap)


dll.luisa_compute_buffer_destroy.restype = None
dll.luisa_compute_buffer_destroy.argtypes = [c_void_p, c_uint64]


def buffer_destroy(device, handle):
    dll.luisa_compute_buffer_destroy(device, handle)


dll.luisa_compute_pixel_format_to_storage.restype = c_uint32
dll.luisa_compute_pixel_format_to_storage.argtypes = [c_uint32]


def pixel_format_to_storage(format):
    return dll.luisa_compute_pixel_format_to_storage(format)


dll.luisa_compute_texture_create.restype = c_uint64
dll.luisa_compute_texture_create.argtypes = [c_void_p, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint64, c_uint32]


def texture_create(device, format, dim, w, h, d, mips, sampler, heap, index_in_heap):
    return dll.luisa_compute_texture_create(device, format, dim, w, h, d, mips, sampler, heap, index_in_heap)


dll.luisa_compute_texture_destroy.restype = None
dll.luisa_compute_texture_destroy.argtypes = [c_void_p, c_uint64]


def texture_destroy(device, handle):
    dll.luisa_compute_texture_destroy(device, handle)


dll.luisa_compute_heap_create.restype = c_uint64
dll.luisa_compute_heap_create.argtypes = [c_void_p, c_size_t]


def heap_create(device, size):
    return dll.luisa_compute_heap_create(device, size)


dll.luisa_compute_heap_destroy.restype = None
dll.luisa_compute_heap_destroy.argtypes = [c_void_p, c_uint64]


def heap_destroy(device, handle):
    dll.luisa_compute_heap_destroy(device, handle)


dll.luisa_compute_heap_query_usage.restype = c_size_t
dll.luisa_compute_heap_query_usage.argtypes = [c_void_p, c_uint64]


def heap_query_usage(device, handle):
    return dll.luisa_compute_heap_query_usage(device, handle)


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


dll.luisa_compute_command_list_empty.restype = c_int
dll.luisa_compute_command_list_empty.argtypes = [c_void_p]


def command_list_empty(list):
    return dll.luisa_compute_command_list_empty(list)


dll.luisa_compute_command_upload_buffer.restype = c_void_p
dll.luisa_compute_command_upload_buffer.argtypes = [c_uint64, c_size_t, c_size_t, c_void_p]


def command_upload_buffer(buffer, offset, size, data):
    return dll.luisa_compute_command_upload_buffer(buffer, offset, size, data)


dll.luisa_compute_command_download_buffer.restype = c_void_p
dll.luisa_compute_command_download_buffer.argtypes = [c_uint64, c_size_t, c_size_t, c_void_p]


def command_download_buffer(buffer, offset, size, data):
    return dll.luisa_compute_command_download_buffer(buffer, offset, size, data)


dll.luisa_compute_command_copy_buffer_to_buffer.restype = c_void_p
dll.luisa_compute_command_copy_buffer_to_buffer.argtypes = [c_uint64, c_size_t, c_uint64, c_size_t, c_size_t]


def command_copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size):
    return dll.luisa_compute_command_copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size)


dll.luisa_compute_command_copy_buffer_to_texture.restype = c_void_p
dll.luisa_compute_command_copy_buffer_to_texture.argtypes = [c_uint64, c_size_t, c_uint64, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32]


def command_copy_buffer_to_texture(buffer, buffer_offset, tex, tex_storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z):
    return dll.luisa_compute_command_copy_buffer_to_texture(buffer, buffer_offset, tex, tex_storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z)


dll.luisa_compute_command_copy_texture_to_buffer.restype = c_void_p
dll.luisa_compute_command_copy_texture_to_buffer.argtypes = [c_uint64, c_size_t, c_uint64, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32]


def command_copy_texture_to_buffer(buffer, buffer_offset, tex, tex_storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z):
    return dll.luisa_compute_command_copy_texture_to_buffer(buffer, buffer_offset, tex, tex_storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z)


dll.luisa_compute_command_copy_texture_to_texture.restype = c_void_p
dll.luisa_compute_command_copy_texture_to_texture.argtypes = [c_uint64, c_uint32, c_uint32, c_uint32, c_uint32, c_uint64, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32]


def command_copy_texture_to_texture(src, src_level, src_offset_x, src_offset_y, src_offset_z, dst, dst_level, dst_offset_x, dst_offset_y, dst_offset_z, size_x, size_y, size_z):
    return dll.luisa_compute_command_copy_texture_to_texture(src, src_level, src_offset_x, src_offset_y, src_offset_z, dst, dst_level, dst_offset_x, dst_offset_y, dst_offset_z, size_x, size_y, size_z)


dll.luisa_compute_command_upload_texture.restype = c_void_p
dll.luisa_compute_command_upload_texture.argtypes = [c_uint64, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_void_p]


def command_upload_texture(handle, storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z, data):
    return dll.luisa_compute_command_upload_texture(handle, storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z, data)


dll.luisa_compute_command_download_texture.restype = c_void_p
dll.luisa_compute_command_download_texture.argtypes = [c_uint64, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_void_p]


def command_download_texture(handle, storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z, data):
    return dll.luisa_compute_command_download_texture(handle, storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z, data)


dll.luisa_compute_command_dispatch_shader.restype = c_void_p
dll.luisa_compute_command_dispatch_shader.argtypes = [c_uint64, c_void_p]


def command_dispatch_shader(handle, kernel):
    return dll.luisa_compute_command_dispatch_shader(handle, kernel)


dll.luisa_compute_command_dispatch_shader_set_size.restype = None
dll.luisa_compute_command_dispatch_shader_set_size.argtypes = [c_void_p, c_uint32, c_uint32, c_uint32]


def command_dispatch_shader_set_size(cmd, sx, sy, sz):
    dll.luisa_compute_command_dispatch_shader_set_size(cmd, sx, sy, sz)


dll.luisa_compute_command_dispatch_shader_encode_buffer.restype = None
dll.luisa_compute_command_dispatch_shader_encode_buffer.argtypes = [c_void_p, c_uint32, c_uint64, c_size_t, c_uint32]


def command_dispatch_shader_encode_buffer(cmd, vid, buffer, offset, usage):
    dll.luisa_compute_command_dispatch_shader_encode_buffer(cmd, vid, buffer, offset, usage)


dll.luisa_compute_command_dispatch_shader_encode_texture.restype = None
dll.luisa_compute_command_dispatch_shader_encode_texture.argtypes = [c_void_p, c_uint32, c_uint64, c_uint32]


def command_dispatch_shader_encode_texture(cmd, vid, tex, usage):
    dll.luisa_compute_command_dispatch_shader_encode_texture(cmd, vid, tex, usage)


dll.luisa_compute_command_dispatch_shader_encode_uniform.restype = None
dll.luisa_compute_command_dispatch_shader_encode_uniform.argtypes = [c_void_p, c_uint32, c_void_p, c_size_t, c_size_t]


def command_dispatch_shader_encode_uniform(cmd, vid, data, size, alignment):
    dll.luisa_compute_command_dispatch_shader_encode_uniform(cmd, vid, data, size, alignment)


dll.luisa_compute_command_dispatch_shader_encode_heap.restype = None
dll.luisa_compute_command_dispatch_shader_encode_heap.argtypes = [c_void_p, c_uint32, c_uint64]


def command_dispatch_shader_encode_heap(cmd, vid, heap):
    dll.luisa_compute_command_dispatch_shader_encode_heap(cmd, vid, heap)


dll.luisa_compute_command_dispatch_shader_encode_accel.restype = None
dll.luisa_compute_command_dispatch_shader_encode_accel.argtypes = [c_void_p, c_uint32, c_uint64]


def command_dispatch_shader_encode_accel(cmd, vid, accel):
    dll.luisa_compute_command_dispatch_shader_encode_accel(cmd, vid, accel)


dll.luisa_compute_command_build_mesh.restype = c_void_p
dll.luisa_compute_command_build_mesh.argtypes = [c_uint64, c_uint32, c_uint64, c_size_t, c_size_t, c_size_t, c_uint64, c_size_t, c_size_t]


def command_build_mesh(handle, hint, v_buffer, v_offset, v_stride, v_count, t_buffer, t_offset, t_count):
    return dll.luisa_compute_command_build_mesh(handle, hint, v_buffer, v_offset, v_stride, v_count, t_buffer, t_offset, t_count)


dll.luisa_compute_command_update_mesh.restype = c_void_p
dll.luisa_compute_command_update_mesh.argtypes = [c_uint64]


def command_update_mesh(handle):
    return dll.luisa_compute_command_update_mesh(handle)


dll.luisa_compute_command_build_accel.restype = c_void_p
dll.luisa_compute_command_build_accel.argtypes = [c_uint64, c_uint32, c_void_p, c_void_p, c_size_t]


def command_build_accel(handle, hint, instance_mesh_handles, instance_transforms, instance_count):
    return dll.luisa_compute_command_build_accel(handle, hint, instance_mesh_handles, instance_transforms, instance_count)


dll.luisa_compute_command_update_accel.restype = c_void_p
dll.luisa_compute_command_update_accel.argtypes = [c_uint64, c_void_p, c_size_t, c_size_t]


def command_update_accel(handle, transforms, offset, count):
    return dll.luisa_compute_command_update_accel(handle, transforms, offset, count)
