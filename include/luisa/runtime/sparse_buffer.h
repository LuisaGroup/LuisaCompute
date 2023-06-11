#pragma once

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream_event.h>
#include <luisa/runtime/rhi/tile_modification.h>

namespace luisa::compute {
namespace detail {
LC_RUNTIME_API void check_sparse_buffer_map(size_t size_bytes, size_t tile_size, uint start_tile, uint tile_count);
LC_RUNTIME_API void check_sparse_buffer_unmap(size_t size_bytes, size_t tile_size, uint start_tile);
}// namespace detail
struct LC_RUNTIME_API SparseBufferUpdateTiles {
    uint64_t handle;
    luisa::vector<SparseBufferOperation> operations;
    void operator()(DeviceInterface *device, uint64_t stream_handle) && noexcept;
};
struct LC_RUNTIME_API SparseBufferClearTiles {
    uint64_t handle;
    void operator()(DeviceInterface *device, uint64_t stream_handle) && noexcept;
};
template<typename T>
class SparseBuffer final : public Resource {
public:
    static_assert(is_valid_buffer_element_v<T>);

private:
    luisa::vector<SparseBufferOperation> _operations;
    size_t _size{};
    size_t _element_stride{};
    size_t _tile_size{};

private:
    friend class Device;
    friend class ResourceGenerator;
    SparseBuffer(DeviceInterface *device, const SparseBufferCreationInfo &info) noexcept
        : Resource{device, Tag::SPARSE_BUFFER, info},
          _size{info.total_size_bytes / info.element_stride},
          _element_stride{info.element_stride},
          _tile_size{info.tile_size_bytes} {}
    SparseBuffer(DeviceInterface *device, size_t size) noexcept
        : SparseBuffer{
              device,
              [&] {
                  if (size == 0) [[unlikely]] {
                      detail::buffer_size_zero_error();
                  }
                  return device->create_sparse_buffer(Type::of<T>(), size);
              }()} {}

public:
    SparseBuffer() noexcept = default;
    ~SparseBuffer() noexcept override {
        if (*this) { device()->destroy_sparse_buffer(handle()); }
    }
    SparseBuffer(SparseBuffer &&) noexcept = default;
    SparseBuffer(SparseBuffer const &) noexcept = delete;
    SparseBuffer &operator=(SparseBuffer &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    void map_tile(uint start_tile, uint tile_count) noexcept {
        detail::check_sparse_buffer_map(size_bytes(), _tile_size, start_tile, tile_count);
        _operations.emplace_back(SparseBufferMapOperation{
            .start_tile = start_tile,
            .tile_count = tile_count});
    }
    void unmap_tile(uint start_tile) noexcept {
        detail::check_sparse_buffer_unmap(size_bytes(), _tile_size, start_tile);
        _operations.emplace_back(SparseBufferUnMapOperation{
            .start_tile = start_tile});
    }

    SparseBuffer &operator=(SparseBuffer const &) noexcept = delete;
    using Resource::operator bool;
    // properties
    [[nodiscard]] auto size() const noexcept {
        return _size;
    }
    [[nodiscard]] constexpr auto stride() const noexcept {
        return _element_stride;
    }
    [[nodiscard]] auto size_bytes() const noexcept {
        return _size * _element_stride;
    }
    [[nodiscard]] auto tile_size() const noexcept {
        return _tile_size;
    }
    [[nodiscard]] auto view() const noexcept {
        return BufferView<T>{this->device(), this->handle(), _element_stride, 0u, _size, _size};
    }
    [[nodiscard]] auto view(size_t offset, size_t count) const noexcept {
        return view().subview(offset, count);
    }
    // commands
    // copy buffer's data to pointer
    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return this->view().copy_to(data);
    }
    // copy pointer's data to buffer
    [[nodiscard]] auto copy_from(const void *data) noexcept {
        return this->view().copy_from(data);
    }
    // copy source buffer's data to buffer
    [[nodiscard]] auto copy_from(BufferView<T> source) noexcept {
        return this->view().copy_from(source);
    }
    [[nodiscard]] auto update() noexcept {
        return SparseBufferUpdateTiles{handle(), std::move(_operations)};
    }
    [[nodiscard]] auto clear_tiles() noexcept {
        return SparseBufferClearTiles{handle()};
    }
};

LUISA_MARK_STREAM_EVENT_TYPE(SparseBufferUpdateTiles)
LUISA_MARK_STREAM_EVENT_TYPE(SparseBufferClearTiles)

namespace detail {

template<typename T>
struct is_buffer_impl<SparseBuffer<T>> : std::true_type {};

template<typename T>
struct buffer_element_impl<SparseBuffer<T>> {
    using type = T;
};

}// namespace detail
}// namespace luisa::compute
