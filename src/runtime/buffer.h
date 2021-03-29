//
// Created by Mike Smith on 2021/3/17.
//

#pragma once

#include <type_traits>

#include <core/concepts.h>
#include <runtime/device.h>

namespace luisa::compute {

namespace dsl {
template<typename T>
class BufferView;
}

#define LUISA_CHECK_BUFFER_ELEMENT_TYPE(T)                    \
    static_assert(alignof(T) <= 16u);                         \
    static_assert(std::is_same_v<T, std::remove_cvref_t<T>>); \
    static_assert(std::is_trivially_copyable_v<T>);           \
    static_assert(std::is_trivially_destructible_v<T>);

template<typename T>
class alignas(8) Buffer : public concepts::Noncopyable {

    LUISA_CHECK_BUFFER_ELEMENT_TYPE(T)

private:
    Device *_device;
    size_t _size;
    uint64_t _handle;

private:
    friend class Device;
    Buffer(Device *device, size_t size, uint64_t handle) noexcept
        : _device{device},
          _size{size},
          _handle{handle} {}

public:
    Buffer(Buffer &&another) noexcept
        : _device{another._device},
          _handle{another._handle},
          _size{another._handle} { another._device = nullptr; }

    Buffer &operator=(Buffer &&rhs) noexcept {
        if (&rhs != this) {
            _device->_dispose_buffer(_handle);
            _device = rhs._device;
            _handle = rhs._handle;
            _size = rhs._handle;
            rhs._device = nullptr;
        }
        return *this;
    }

    ~Buffer() noexcept {
        if (_device != nullptr /* not moved */) {
            _device->_dispose_buffer(_handle);
        }
    }

    [[nodiscard]] auto view() const noexcept { return dsl::BufferView<T>{_handle, 0u, _size}; }
    
    [[nodiscard]] auto copy_to(T *data) const noexcept { return this->view().copy_to(data); }
    [[nodiscard]] auto copy_from(const T *data) { return this->view().copy_from(data); }
    [[nodiscard]] auto copy_from(dsl::BufferView<T> source) { return this->view().copy_from(source); }
    
    template<typename I>
    [[nodiscard]] decltype(auto) operator[](I &&i) const noexcept {
        return this->view()[std::forward<I>(i)];
    }
};

}// namespace luisa::compute
