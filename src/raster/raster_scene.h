#pragma once
#include <runtime/buffer.h>
#include <runtime/dynamic_buffer.h>
namespace luisa::compute {
class VertexBufferView {
    uint64_t _handle{};
    uint64_t _offset{};
    uint64_t _size{};
    uint64_t _stride{};

public:
    uint64_t handle() const noexcept { return _handle; }
    uint64_t offset() const noexcept { return _offset; }
    uint64_t size() const noexcept { return _size; }
    uint64_t stride() const noexcept { return _stride; }
    template<typename T>
        requires(is_buffer_view_v<T>)
    VertexBufferView(T const &buffer_view) {
        _handle = buffer_view.handle();
        _offset = buffer_view.offset_bytes();
        _size = buffer_view.size_bytes();
        _stride = buffer_view.stride();
    }

    template<typename T>
        requires(is_buffer_v<T>)
    VertexBufferView(T const &buffer) {
        _handle = buffer.handle();
        _offset = 0;
        _size = buffer.size_bytes();
        _stride = buffer.stride();
    }
};
class RasterMesh {
    luisa::span<VertexBufferView const> _vertex_buffers{};
    luisa::variant<BufferView<uint>, uint> _index_buffer;
    uint _instance_count{};
    uint _object_id{};

public:
    luisa::span<VertexBufferView const> vertex_buffers() const noexcept { return _vertex_buffers; }
    decltype(auto) index() const noexcept { return _index_buffer; };
    uint instance_count() const noexcept { return _instance_count; }
    uint object_id() const noexcept { return _object_id; }
    RasterMesh(
        luisa::span<VertexBufferView const> vertex_buffers,
        BufferView<uint> index_buffer,
        uint instance_count,
        uint object_id)
        : _vertex_buffers(vertex_buffers),
          _index_buffer(index_buffer),
          _instance_count(instance_count),
          _object_id(object_id) {
    }
    RasterMesh(
        luisa::span<VertexBufferView const> vertex_buffers,
        uint vertex_count,
        uint instance_count,
        uint object_id)
        : _vertex_buffers(vertex_buffers),
          _index_buffer(vertex_count),
          _instance_count(instance_count),
          _object_id(object_id) {
    }
};
}// namespace luisa::compute