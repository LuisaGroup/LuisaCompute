#pragma once

#include <luisa/core/logging.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream_event.h>
#include <luisa/runtime/rhi/tile_modification.h>
#include <luisa/runtime/sparse_heap.h>

namespace luisa::compute {

namespace detail {
LUISA_RUNTIME_API void check_sparse_buffer_tile_region(
    size_t size_bytes,
    size_t tile_size,
    uint32_t start_tile,
    uint32_t tile_count) noexcept;
}// namespace detail

template<typename T>
class SparseBuffer final : public Resource {
public:
    static_assert(is_valid_buffer_element_v<T>);

private:
    size_t _size{};
    size_t _element_stride{};
    size_t _tile_size_bytes{};

private:
    friend class Device;
    friend class ResourceGenerator;
    SparseBuffer(DeviceInterface *device, const SparseBufferCreationInfo &info) noexcept
        : Resource{device, Tag::SPARSE_BUFFER, info},
          _size{info.total_size_bytes / info.element_stride},
          _element_stride{info.element_stride},
          _tile_size_bytes{info.tile_size_bytes} {}
    SparseBuffer(DeviceInterface *device, size_t size) noexcept
        : SparseBuffer{
              device,
              [&] {
                  if (size == 0) [[unlikely]] {
                      detail::error_buffer_size_is_zero();
                  }
                  auto info = device->create_sparse_buffer(Type::of<T>(), size);
#ifdef LUISA_ENABLE_SAFE_MODE
                  if (!info.valid()) {
                      LUISA_ERROR("Failed to create sparse buffer.");
                  }
#endif
                  return info;
              }()} {
    }

public:
    SparseBuffer() noexcept = default;
    ~SparseBuffer() noexcept override {
        if (*this) { device()->destroy_sparse_buffer(handle()); }
    }
    SparseBufferCreationInfo release() noexcept {
        SparseBufferCreationInfo r{Resource::release()};
        r.element_stride = _element_stride;
        r.tile_size_bytes = _tile_size_bytes;
        return r;
    }
    SparseBuffer(SparseBuffer &&) noexcept = default;
    SparseBuffer(SparseBuffer const &) noexcept = delete;
    SparseBuffer &operator=(SparseBuffer &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    [[nodiscard]] auto map_tile(uint start_tile, uint tile_count, const SparseBufferHeap &heap) noexcept {
        _check_is_valid();
        detail::check_sparse_buffer_tile_region(
            size_bytes(), _tile_size_bytes,
            start_tile, tile_count);
        detail::check_sparse_heap_provenance(*this, heap);
        return SparseUpdateTile{
            .handle = handle(),
            .operations = SparseBufferMapOperation{
                .allocated_heap = heap.handle(),
                .start_tile = start_tile,
                .tile_count = tile_count}};
    }
    [[nodiscard]] auto unmap_tile(uint start_tile, uint tile_count) noexcept {
        _check_is_valid();
        detail::check_sparse_buffer_tile_region(
            size_bytes(), _tile_size_bytes,
            start_tile, tile_count);
        return SparseUpdateTile{
            .handle = handle(),
            .operations = SparseBufferUnMapOperation{
                .start_tile = start_tile,
                .tile_count = tile_count}};
    }

    SparseBuffer &operator=(SparseBuffer const &) noexcept = delete;
    using Resource::operator bool;
    // properties
    [[nodiscard]] auto size() const noexcept {
        _check_is_valid();
        return _size;
    }
    [[nodiscard]] constexpr auto stride() const noexcept {
        _check_is_valid();
        return _element_stride;
    }
    [[nodiscard]] auto size_bytes() const noexcept {
        _check_is_valid();
        return _size * _element_stride;
    }
    [[nodiscard]] auto tile_size() const noexcept {
        _check_is_valid();
        LUISA_ASSERT(
            _element_stride != 0u &&
                _tile_size_bytes % _element_stride == 0u,
            "Sparse buffer tile size {} bytes is not an integral number of "
            "{}-byte elements; use tile_size_bytes() for this resource.",
            _tile_size_bytes, _element_stride);
        return _tile_size_bytes / _element_stride;
    }
    [[nodiscard]] auto tile_size_bytes() const noexcept {
        _check_is_valid();
        return _tile_size_bytes;
    }
    [[nodiscard]] auto view() const noexcept {
        _check_is_valid();
        return BufferView<T>{this->device(), this->handle(), _element_stride, 0u, _size, _size};
    }
    [[nodiscard]] auto view(size_t offset, size_t count) const noexcept {
        return view().subview(offset, count);
    }
    // commands
    // copy buffer's data to pointer
    [[nodiscard]] auto copy_to(void *data) const noexcept {
        _check_is_valid();
        return this->view().copy_to(data);
    }
    // copy pointer's data to buffer
    [[nodiscard]] auto copy_from(const void *data) noexcept {
        _check_is_valid();
        return this->view().copy_from(data);
    }
    // copy source buffer's data to buffer
    [[nodiscard]] auto copy_from(BufferView<T> source) noexcept {
        _check_is_valid();
        return this->view().copy_from(source);
    }
};

namespace detail {

template<typename T>
struct is_buffer_impl<SparseBuffer<T>> : std::true_type {};

template<typename T>
struct buffer_element_impl<SparseBuffer<T>> {
    using type = T;
};

}// namespace detail

}// namespace luisa::compute
