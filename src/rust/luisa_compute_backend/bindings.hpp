#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>
#include "luisa_compute_ir/common.h"


namespace luisa::compute::backend {

struct LCDispatchCallback {
    void (*callback)(uint8_t*);
    uint8_t *user_data;
};

using namespace luisa::compute::ir;

extern "C" {

CreatedResourceInfo lc_rs_create_accel(void *backend, AccelOption option);

void *lc_rs_create_backend(const char *name);

CreatedResourceInfo lc_rs_create_bindless_array(void *backend, size_t size);

CreatedBufferInfo lc_rs_create_buffer(void *backend, const CArc<Type> *ty, size_t count);

CreatedResourceInfo lc_rs_create_event(void *backend);

CreatedResourceInfo lc_rs_create_mesh(void *backend, AccelOption option);

CreatedResourceInfo lc_rs_create_procedural_primitive(void *backend, AccelOption option);

CreatedShaderInfo lc_rs_create_shader(void *backend,
                                      CArc<KernelModule> kernel,
                                      const ShaderOption *option);

CreatedResourceInfo lc_rs_create_stream(void *backend, StreamTag tag);

CreatedResourceInfo lc_rs_create_texture(void *backend,
                                         PixelFormat format,
                                         uint32_t dimension,
                                         uint32_t width,
                                         uint32_t height,
                                         uint32_t depth,
                                         uint32_t mipmap_levels);

void lc_rs_destroy_accel(void *backend, Accel accel);

void lc_rs_destroy_backend(void *ptr);

void lc_rs_destroy_bindless_array(void *backend, BindlessArray array);

void lc_rs_destroy_buffer(void *backend, Buffer buffer);

void lc_rs_destroy_event(void *backend, Event event);

void lc_rs_destroy_mesh(void *backend, Mesh mesh);

void lc_rs_destroy_shader(void *backend, Shader shader);

void lc_rs_destroy_stream(void *backend, Stream stream);

void lc_rs_destroy_texture(void *backend, Texture texture);

bool lc_rs_dispatch(void *backend,
                    Stream stream,
                    const Command *command_list,
                    size_t command_count,
                    LCDispatchCallback callback);

void lc_rs_query(void *backend, char *property, char *result, size_t result_size);

bool lc_rs_synchronize_stream(void *backend, Stream stream);

CBoxedSlice<uint8_t> luisa_compute_decode_const_data(const uint8_t *data,
                                                     size_t len,
                                                     const Type *ty);

} // extern "C"

} // namespace luisa::compute::backend
