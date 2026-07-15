#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

ByteBuffer::ByteBuffer(DeviceInterface *device, const BufferCreationInfo &info) noexcept
    : Resource{device, Tag::BUFFER, info},
      _size_bytes{info.total_size_bytes} {}

ByteBuffer::ByteBuffer(DeviceInterface *device, size_t size_bytes) noexcept
    : ByteBuffer{
          device,
          [&] {
              if (size_bytes == 0u) [[unlikely]] { detail::error_buffer_size_is_zero(); }
              if (size_bytes % 4u != 0u) [[unlikely]] {
                  auto aligned_size = luisa::align(size_bytes, 4u);
                  LUISA_WARNING_WITH_LOCATION("Buffer size ({}) is not aligned to 4 bytes. "
                                              "We will align it for you to {}.",
                                              size_bytes, aligned_size);
                  size_bytes = aligned_size;
              }
              auto info = device->create_buffer(Type::of<void>(), size_bytes, nullptr);
#ifdef LUISA_ENABLE_SAFE_MODE
              if (!info.valid()) {
                  LUISA_ERROR("Failed to create byte buffer.");
              }
#endif
              return info;
          }()} {}

ByteBuffer::~ByteBuffer() noexcept {
    if (*this) { device()->destroy_buffer(handle()); }
}

ByteBuffer Device::create_byte_buffer(size_t byte_size) const noexcept {
    return ByteBuffer{impl(), byte_size};
}

ByteBuffer Device::import_external_byte_buffer(void *external_memory, size_t byte_size) const noexcept {
    auto info = impl()->create_buffer(Type::of<uint>(),
                                      (byte_size + sizeof(uint) - 1u) / sizeof(uint),
                                      external_memory);
    return ByteBuffer{impl(), info};
}

ByteBufferView ByteBuffer::view() const noexcept {
    return ByteBufferView{native_handle(), handle(), 0u, _size_bytes, _size_bytes};
}

ByteBufferView ByteBuffer::view(size_t offset_bytes, size_t size_bytes) const noexcept {
    return view().subview(offset_bytes, size_bytes);
}

luisa::unique_ptr<BufferDownloadCommand> ByteBuffer::copy_to(void *data) const noexcept {
    _check_is_valid();
    return view().copy_to(data);
}

luisa::unique_ptr<BufferUploadCommand> ByteBuffer::copy_from(const void *data) const noexcept {
    _check_is_valid();
    return view().copy_from(data);
}

[[nodiscard]] luisa::unique_ptr<BufferCopyCommand> ByteBuffer::copy_from(ByteBufferView source) const noexcept {
    _check_is_valid();
    return view().copy_from(source);
}

luisa::unique_ptr<BufferCopyCommand> ByteBuffer::copy_to(ByteBufferView source) const noexcept {
    _check_is_valid();
    return view().copy_to(source);
}

ByteBufferView ByteBufferView::original() const noexcept {
    return ByteBufferView{_native_handle, _handle, 0u, _total_size, _total_size};
}

ByteBufferView ByteBufferView::subview(size_t offset_bytes, size_t size_bytes) const noexcept {
    if (offset_bytes + size_bytes > _size) [[unlikely]] {
        detail::error_buffer_subview_overflow(offset_bytes, size_bytes, _size);
    }
    return ByteBufferView{_native_handle, _handle, _offset_bytes + offset_bytes, size_bytes, _total_size};
}

luisa::unique_ptr<BufferDownloadCommand> ByteBufferView::copy_to(void *data) const noexcept {
    return luisa::make_unique<BufferDownloadCommand>(_handle, offset_bytes(), size_bytes(), data);
}

luisa::unique_ptr<BufferUploadCommand> ByteBufferView::copy_from(const void *data) const noexcept {
    return luisa::make_unique<BufferUploadCommand>(this->handle(), this->offset_bytes(), this->size_bytes(), data);
}

luisa::unique_ptr<BufferCopyCommand> ByteBufferView::copy_from(ByteBufferView source) const noexcept {
    if (source.size_bytes() != this->size_bytes()) [[unlikely]] {
        detail::error_buffer_copy_sizes_mismatch(source.size_bytes(), this->size_bytes());
    }
    return luisa::make_unique<BufferCopyCommand>(
        source.handle(), this->handle(),
        source.offset_bytes(), this->offset_bytes(),
        this->size_bytes());
}

luisa::unique_ptr<BufferCopyCommand> ByteBufferView::copy_to(ByteBufferView source) const noexcept {
    return source.copy_from(*this);
}

namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const ByteBufferView &buffer) noexcept {
    _encoder.encode_buffer(buffer.handle(), buffer.offset_bytes(), buffer.size_bytes());
    return *this;
}

ShaderInvokeBase &ShaderInvokeBase::operator<<(const ByteBuffer &buffer) noexcept {
    buffer._check_is_valid();
    _encoder.encode_buffer(buffer.handle(), 0u, buffer.size_bytes());
    return *this;
}

}// namespace detail

}// namespace luisa::compute
