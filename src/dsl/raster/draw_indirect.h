#pragma once

#include <dsl/expr.h>
#include <dsl/var.h>
#include <dsl/struct.h>
#include <runtime/command.h>

namespace luisa::compute {
struct DrawIndirectArgs {};
struct DrawIndexedIndirectArgs {};
struct VertexBufferAddress {};
struct IndexBufferAddress {};
}// namespace luisa::compute

LUISA_CUSTOM_STRUCT(DrawIndirectArgs)
LUISA_CUSTOM_STRUCT(DrawIndexedIndirectArgs)
LUISA_CUSTOM_STRUCT(VertexBufferAddress)
LUISA_CUSTOM_STRUCT(IndexBufferAddress)

/*
namespace luisa::compute {
template<typename T>
luisa::unique_ptr<BufferUploadCommand> upload_vertex_buffer(
    Buffer<VertexBufferAddress> const &buffer,
    size_t buffer_index,
    BufferView<T> vertex_buffer) {
  
}

template<typename T>
luisa::unique_ptr<BufferUploadCommand> upload_vertex_buffer(
    Buffer<VertexBufferAddress> const &buffer,
    size_t buffer_index,
    Buffer<T> const &vertex_buffer) {
    return upload_vertex_buffer(buffer, buffer_index, vertex_buffer.view());
}
}// namespace luisa::compute
*/
