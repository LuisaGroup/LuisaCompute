#pragma once

#include <runtime/buffer.h>

namespace luisa::compute {

template<typename T>
struct Expr;

template<typename T>
struct Var;

class DynamicStruct;

template<>
class BufferView<DynamicStruct> {
    friend class Buffer<DynamicStruct>;
    uint64_t _handle;
    size_t _offset_bytes;
    const Type *_type;
    size_t _size;
    size_t _total_size;
    BufferView(uint64_t handle, size_t offset, const Type *type, size_t size, size_t total_size) noexcept
        : _handle(handle), _offset_bytes(offset), _type(type), _size(size), _total_size(total_size) {}

public:
    BufferView() noexcept : BufferView(Resource::invalid_handle, 0, nullptr, 0, 0) {}
    [[nodiscard]] explicit operator bool() const noexcept { return _handle != Resource::invalid_handle; }
    BufferView(const Buffer<DynamicStruct> &buffer) noexcept;

    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto stride() const noexcept { return _type->size(); }
    [[nodiscard]] auto offset() const noexcept { return _offset_bytes / stride(); }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * stride(); }

    BufferView subview(size_t offset_elements, size_t size_elements) const noexcept {
        return BufferView{_handle, _offset_bytes + offset_elements * stride(), _type, size_elements, _total_size};
    }
    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return BufferDownloadCommand::create(_handle, offset_bytes(), size_bytes(), data);
    }

    [[nodiscard]] auto copy_from(const void *data) noexcept {
        return BufferUploadCommand::create(this->handle(), this->offset_bytes(), this->size_bytes(), data);
    }

    [[nodiscard]] auto copy_from(BufferView source) noexcept {
        if (source.size() != this->size()) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(source.size(), this->size());
        }
        return BufferCopyCommand::create(
            source.handle(), this->handle(),
            source.offset_bytes(), this->offset_bytes(),
            this->size_bytes());
    }
    template<typename I>
    [[nodiscard]] Var<DynamicStruct> read(I &&i) const noexcept;
    template<typename U>
    [[nodiscard]] auto as() const noexcept {
        if (this->size_bytes() < sizeof(U)) [[unlikely]] {
            detail::error_buffer_reinterpret_size_too_small(sizeof(U), this->size_bytes());
        }
        return BufferView<U>{_handle, _offset_bytes, this->size_bytes() / sizeof(U), _total_size};
    }
    [[nodiscard]] auto as(const Type *type) const noexcept {
        auto stride = type->size();
        if (this->size_bytes() < stride) [[unlikely]] {
            detail::error_buffer_reinterpret_size_too_small(stride, this->size_bytes());
        }
        return BufferView<DynamicStruct>{_handle, _offset_bytes, type, this->size_bytes() / stride, _total_size};
    }

    template<typename I, typename V>
    void write(I &&i, V &&v) const noexcept;
};

template<typename T>
[[nodiscard]] BufferView<DynamicStruct> BufferView<T>::as(const Type *type) const noexcept {
    auto stride = type->size();
    if (this->size_bytes() < stride) [[unlikely]] {
        detail::error_buffer_reinterpret_size_too_small(stride, this->size_bytes());
    }
    return BufferView<DynamicStruct>{_handle, _offset_bytes, type, this->size_bytes() / stride, _total_size};
}

template<>
class Buffer<DynamicStruct> : public Resource {
private:
    const Type *_type{};
    size_t _size{};

public:
    Buffer(DeviceInterface *device, const Type *type, size_t size) noexcept
        : Resource(device, Tag::BUFFER, device->create_buffer(size * type->size())), _size(size), _type(type) {}
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto stride() const noexcept { return _type->size(); }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * _type->size(); }
    [[nodiscard]] auto view() const noexcept { return BufferView<DynamicStruct>{this->handle(), 0u, _type, _size, _size}; }
    [[nodiscard]] auto view(size_t offset, size_t count) const noexcept { return view().subview(offset, count); }

    [[nodiscard]] auto copy_to(void *data) const noexcept { return this->view().copy_to(data); }
    [[nodiscard]] auto copy_from(const void *data) noexcept { return this->view().copy_from(data); }
    [[nodiscard]] auto copy_from(BufferView<DynamicStruct> source) noexcept { return this->view().copy_from(source); }

    template<typename I>
    [[nodiscard]] decltype(auto) read(I &&i) const noexcept { return this->view().read(std::forward<I>(i)); }

    template<typename I, typename V>
    void write(I &&i, V &&v) const noexcept { this->view().write(std::forward<I>(i), std::forward<V>(v)); }
};

inline BufferView<DynamicStruct>::BufferView(const Buffer<DynamicStruct> &buffer) noexcept
    : BufferView{buffer.view()} {}

}// namespace luisa::compute
