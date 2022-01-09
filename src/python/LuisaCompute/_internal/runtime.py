from ctypes import c_void_p, c_char_p, c_int, c_int32, c_uint32, c_int64, c_uint64, c_size_t, c_float
from .config import dll


dll.luisa_compute_free_c_string.restype = None
dll.luisa_compute_free_c_string.argtypes = [c_char_p]


def free_c_string(cs):
    dll.luisa_compute_free_c_string(cs.encode())


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
dll.luisa_compute_device_create.argtypes = [c_void_p, c_char_p, c_char_p]


def device_create(ctx, name, properties):
    return dll.luisa_compute_device_create(ctx, name.encode(), properties.encode())


dll.luisa_compute_device_destroy.restype = None
dll.luisa_compute_device_destroy.argtypes = [c_void_p, c_void_p]


def device_destroy(ctx, device):
    dll.luisa_compute_device_destroy(ctx, device)


dll.luisa_compute_buffer_create.restype = c_void_p
dll.luisa_compute_buffer_create.argtypes = [c_void_p, c_size_t]


def buffer_create(device, size):
    return dll.luisa_compute_buffer_create(device, size)


dll.luisa_compute_buffer_destroy.restype = None
dll.luisa_compute_buffer_destroy.argtypes = [c_void_p]


def buffer_destroy(buffer):
    dll.luisa_compute_buffer_destroy(buffer)


dll.luisa_compute_pixel_format_to_storage.restype = c_uint32
dll.luisa_compute_pixel_format_to_storage.argtypes = [c_uint32]


def pixel_format_to_storage(format):
    return dll.luisa_compute_pixel_format_to_storage(format)


dll.luisa_compute_texture_create.restype = c_void_p
dll.luisa_compute_texture_create.argtypes = [c_void_p, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32]


def texture_create(device, format, dim, w, h, d, mips):
    return dll.luisa_compute_texture_create(device, format, dim, w, h, d, mips)


dll.luisa_compute_texture_destroy.restype = None
dll.luisa_compute_texture_destroy.argtypes = [c_void_p]


def texture_destroy(texture):
    dll.luisa_compute_texture_destroy(texture)


dll.luisa_compute_stream_create.restype = c_void_p
dll.luisa_compute_stream_create.argtypes = [c_void_p]


def stream_create(device):
    return dll.luisa_compute_stream_create(device)


dll.luisa_compute_stream_destroy.restype = None
dll.luisa_compute_stream_destroy.argtypes = [c_void_p]


def stream_destroy(stream):
    dll.luisa_compute_stream_destroy(stream)


dll.luisa_compute_stream_synchronize.restype = None
dll.luisa_compute_stream_synchronize.argtypes = [c_void_p]


def stream_synchronize(stream):
    dll.luisa_compute_stream_synchronize(stream)


dll.luisa_compute_stream_dispatch.restype = None
dll.luisa_compute_stream_dispatch.argtypes = [c_void_p, c_void_p]


def stream_dispatch(stream, cmd_list):
    dll.luisa_compute_stream_dispatch(stream, cmd_list)


dll.luisa_compute_shader_create.restype = c_void_p
dll.luisa_compute_shader_create.argtypes = [c_void_p, c_void_p, c_char_p]


def shader_create(device, function, options):
    return dll.luisa_compute_shader_create(device, function, options.encode())


dll.luisa_compute_shader_destroy.restype = None
dll.luisa_compute_shader_destroy.argtypes = [c_void_p]


def shader_destroy(shader):
    dll.luisa_compute_shader_destroy(shader)


dll.luisa_compute_event_create.restype = c_void_p
dll.luisa_compute_event_create.argtypes = [c_void_p]


def event_create(device):
    return dll.luisa_compute_event_create(device)


dll.luisa_compute_event_destroy.restype = None
dll.luisa_compute_event_destroy.argtypes = [c_void_p]


def event_destroy(event):
    dll.luisa_compute_event_destroy(event)


dll.luisa_compute_event_signal.restype = None
dll.luisa_compute_event_signal.argtypes = [c_void_p, c_void_p]


def event_signal(event, stream):
    dll.luisa_compute_event_signal(event, stream)


dll.luisa_compute_event_wait.restype = None
dll.luisa_compute_event_wait.argtypes = [c_void_p, c_void_p]


def event_wait(event, stream):
    dll.luisa_compute_event_wait(event, stream)


dll.luisa_compute_event_synchronize.restype = None
dll.luisa_compute_event_synchronize.argtypes = [c_void_p]


def event_synchronize(event):
    dll.luisa_compute_event_synchronize(event)


dll.luisa_compute_bindless_array_create.restype = c_void_p
dll.luisa_compute_bindless_array_create.argtypes = [c_void_p, c_size_t]


def bindless_array_create(device, n):
    return dll.luisa_compute_bindless_array_create(device, n)


dll.luisa_compute_bindless_array_destroy.restype = None
dll.luisa_compute_bindless_array_destroy.argtypes = [c_void_p]


def bindless_array_destroy(array):
    dll.luisa_compute_bindless_array_destroy(array)


dll.luisa_compute_bindless_array_emplace_buffer.restype = None
dll.luisa_compute_bindless_array_emplace_buffer.argtypes = [c_void_p, c_size_t, c_void_p]


def bindless_array_emplace_buffer(array, index, buffer):
    dll.luisa_compute_bindless_array_emplace_buffer(array, index, buffer)


dll.luisa_compute_bindless_array_emplace_tex2d.restype = None
dll.luisa_compute_bindless_array_emplace_tex2d.argtypes = [c_void_p, c_size_t, c_void_p, c_uint32]


def bindless_array_emplace_tex2d(array, index, texture, sampler):
    dll.luisa_compute_bindless_array_emplace_tex2d(array, index, texture, sampler)


dll.luisa_compute_bindless_array_emplace_tex3d.restype = None
dll.luisa_compute_bindless_array_emplace_tex3d.argtypes = [c_void_p, c_size_t, c_void_p, c_uint32]


def bindless_array_emplace_tex3d(array, index, texture, sampler):
    dll.luisa_compute_bindless_array_emplace_tex3d(array, index, texture, sampler)


dll.luisa_compute_bindless_array_remove_buffer.restype = None
dll.luisa_compute_bindless_array_remove_buffer.argtypes = [c_void_p, c_size_t]


def bindless_array_remove_buffer(array, index):
    dll.luisa_compute_bindless_array_remove_buffer(array, index)


dll.luisa_compute_bindless_array_remove_tex2d.restype = None
dll.luisa_compute_bindless_array_remove_tex2d.argtypes = [c_void_p, c_size_t]


def bindless_array_remove_tex2d(array, index):
    dll.luisa_compute_bindless_array_remove_tex2d(array, index)


dll.luisa_compute_bindless_array_remove_tex3d.restype = None
dll.luisa_compute_bindless_array_remove_tex3d.argtypes = [c_void_p, c_size_t]


def bindless_array_remove_tex3d(array, index):
    dll.luisa_compute_bindless_array_remove_tex3d(array, index)


dll.luisa_compute_mesh_create.restype = c_void_p
dll.luisa_compute_mesh_create.argtypes = [c_void_p, c_void_p, c_size_t, c_size_t, c_size_t, c_void_p, c_size_t, c_size_t, c_uint32]


def mesh_create(device, v_buffer, v_offset, v_stride, v_count, t_buffer, t_offset, t_count, hint):
    return dll.luisa_compute_mesh_create(device, v_buffer, v_offset, v_stride, v_count, t_buffer, t_offset, t_count, hint)


dll.luisa_compute_mesh_destroy.restype = None
dll.luisa_compute_mesh_destroy.argtypes = [c_void_p]


def mesh_destroy(mesh):
    dll.luisa_compute_mesh_destroy(mesh)


dll.luisa_compute_accel_create.restype = c_void_p
dll.luisa_compute_accel_create.argtypes = [c_void_p, c_uint32]


def accel_create(device, hint):
    return dll.luisa_compute_accel_create(device, hint)


dll.luisa_compute_accel_destroy.restype = None
dll.luisa_compute_accel_destroy.argtypes = [c_void_p]


def accel_destroy(accel):
    dll.luisa_compute_accel_destroy(accel)


dll.luisa_compute_accel_emplace_back.restype = None
dll.luisa_compute_accel_emplace_back.argtypes = [c_void_p, c_void_p, c_void_p, c_int]


def accel_emplace_back(accel, mesh, transform, visibility):
    dll.luisa_compute_accel_emplace_back(accel, mesh, transform, visibility)


dll.luisa_compute_accel_emplace.restype = None
dll.luisa_compute_accel_emplace.argtypes = [c_void_p, c_size_t, c_void_p, c_void_p, c_int]


def accel_emplace(accel, index, mesh, transform, visibility):
    dll.luisa_compute_accel_emplace(accel, index, mesh, transform, visibility)


dll.luisa_compute_accel_set_transform.restype = None
dll.luisa_compute_accel_set_transform.argtypes = [c_void_p, c_size_t, c_void_p]


def accel_set_transform(accel, index, transform):
    dll.luisa_compute_accel_set_transform(accel, index, transform)


dll.luisa_compute_accel_set_visibility.restype = None
dll.luisa_compute_accel_set_visibility.argtypes = [c_void_p, c_size_t, c_int]


def accel_set_visibility(accel, index, visibility):
    dll.luisa_compute_accel_set_visibility(accel, index, visibility)


dll.luisa_compute_accel_pop_back.restype = None
dll.luisa_compute_accel_pop_back.argtypes = [c_void_p]


def accel_pop_back(accel):
    dll.luisa_compute_accel_pop_back(accel)


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
dll.luisa_compute_command_upload_buffer.argtypes = [c_void_p, c_size_t, c_size_t, c_void_p]


def command_upload_buffer(buffer, offset, size, data):
    return dll.luisa_compute_command_upload_buffer(buffer, offset, size, data)


dll.luisa_compute_command_download_buffer.restype = c_void_p
dll.luisa_compute_command_download_buffer.argtypes = [c_void_p, c_size_t, c_size_t, c_void_p]


def command_download_buffer(buffer, offset, size, data):
    return dll.luisa_compute_command_download_buffer(buffer, offset, size, data)


dll.luisa_compute_command_copy_buffer_to_buffer.restype = c_void_p
dll.luisa_compute_command_copy_buffer_to_buffer.argtypes = [c_void_p, c_size_t, c_void_p, c_size_t, c_size_t]


def command_copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size):
    return dll.luisa_compute_command_copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size)


dll.luisa_compute_command_copy_buffer_to_texture.restype = c_void_p
dll.luisa_compute_command_copy_buffer_to_texture.argtypes = [c_void_p, c_size_t, c_void_p, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32]


def command_copy_buffer_to_texture(buffer, buffer_offset, tex, tex_storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z):
    return dll.luisa_compute_command_copy_buffer_to_texture(buffer, buffer_offset, tex, tex_storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z)


dll.luisa_compute_command_copy_texture_to_buffer.restype = c_void_p
dll.luisa_compute_command_copy_texture_to_buffer.argtypes = [c_uint64, c_size_t, c_uint64, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32]


def command_copy_texture_to_buffer(buffer, buffer_offset, tex, tex_storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z):
    return dll.luisa_compute_command_copy_texture_to_buffer(buffer, buffer_offset, tex, tex_storage, level, offset_x, offset_y, offset_z, size_x, size_y, size_z)


dll.luisa_compute_command_copy_texture_to_texture.restype = c_void_p
dll.luisa_compute_command_copy_texture_to_texture.argtypes = [c_uint64, c_uint32, c_uint32, c_uint32, c_uint32, c_uint64, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32, c_uint32]


def command_copy_texture_to_texture(src, src_level, src_offset_x, src_offset_y, src_offset_z, dst, dst_level, dst_offset_x, dst_offset_y, dst_offset_z, storage, size_x, size_y, size_z):
    return dll.luisa_compute_command_copy_texture_to_texture(src, src_level, src_offset_x, src_offset_y, src_offset_z, dst, dst_level, dst_offset_x, dst_offset_y, dst_offset_z, storage, size_x, size_y, size_z)


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
dll.luisa_compute_command_dispatch_shader_encode_texture.argtypes = [c_void_p, c_uint32, c_uint64, c_uint32, c_uint32]


def command_dispatch_shader_encode_texture(cmd, vid, tex, level, usage):
    dll.luisa_compute_command_dispatch_shader_encode_texture(cmd, vid, tex, level, usage)


dll.luisa_compute_command_dispatch_shader_encode_uniform.restype = None
dll.luisa_compute_command_dispatch_shader_encode_uniform.argtypes = [c_void_p, c_uint32, c_void_p, c_size_t, c_size_t]


def command_dispatch_shader_encode_uniform(cmd, vid, data, size, alignment):
    dll.luisa_compute_command_dispatch_shader_encode_uniform(cmd, vid, data, size, alignment)


dll.luisa_compute_command_dispatch_shader_encode_bindless_array.restype = None
dll.luisa_compute_command_dispatch_shader_encode_bindless_array.argtypes = [c_void_p, c_uint32, c_uint64]


def command_dispatch_shader_encode_bindless_array(cmd, vid, heap):
    dll.luisa_compute_command_dispatch_shader_encode_bindless_array(cmd, vid, heap)


dll.luisa_compute_command_dispatch_shader_encode_accel.restype = None
dll.luisa_compute_command_dispatch_shader_encode_accel.argtypes = [c_void_p, c_uint32, c_uint64]


def command_dispatch_shader_encode_accel(cmd, vid, accel):
    dll.luisa_compute_command_dispatch_shader_encode_accel(cmd, vid, accel)


dll.luisa_compute_command_build_mesh.restype = c_void_p
dll.luisa_compute_command_build_mesh.argtypes = [c_uint64]


def command_build_mesh(handle):
    return dll.luisa_compute_command_build_mesh(handle)


dll.luisa_compute_command_update_mesh.restype = c_void_p
dll.luisa_compute_command_update_mesh.argtypes = [c_uint64]


def command_update_mesh(handle):
    return dll.luisa_compute_command_update_mesh(handle)


dll.luisa_compute_command_build_accel.restype = c_void_p
dll.luisa_compute_command_build_accel.argtypes = [c_uint64]


def command_build_accel(handle):
    return dll.luisa_compute_command_build_accel(handle)


dll.luisa_compute_command_update_accel.restype = c_void_p
dll.luisa_compute_command_update_accel.argtypes = [c_uint64]


def command_update_accel(handle):
    return dll.luisa_compute_command_update_accel(handle)
