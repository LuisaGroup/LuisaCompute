//
// Created by Mike Smith on 2021/3/2.
//

#pragma once

#include <core/concepts.h>
#include <core/mathematics.h>
#include <runtime/command.h>
#include <runtime/resource.h>

namespace luisa::compute {

template<typename T>
struct Expr;

template<typename T>
class BufferView;

template<typename T>
using is_valid_buffer_element = std::conjunction<
    std::is_same<T, std::remove_cvref_t<T>>,
    std::is_trivially_copyable<T>,
    std::is_trivially_destructible<T>,
    std::bool_constant<(alignof(T) >= 4u)>>;

template<typename T>
constexpr auto is_valid_buffer_element_v = is_valid_buffer_element<T>::value;

template<typename T>
class Buffer final : public Resource {

    static_assert(is_valid_buffer_element_v<T>);

private:
    size_t _size{};

private:
    friend class Device;
    Buffer(Device::Interface *device, size_t size) noexcept
        : Resource{
              device, Tag::BUFFER,
              device->create_buffer(size * sizeof(T))},
          _size{size} {}

public:
    Buffer() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] void *native_handle() const noexcept { return device()->buffer_native_handle(handle()); }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * sizeof(T); }
    [[nodiscard]] auto view() const noexcept { return BufferView<T>{this->handle(), 0u, _size, _size}; }
    [[nodiscard]] auto view(size_t offset, size_t count) const noexcept { return view().subview(offset, count); }

    [[nodiscard]] auto copy_to(void *data) const noexcept { return this->view().copy_to(data); }
    [[nodiscard]] auto copy_from(const void *data) noexcept { return this->view().copy_from(data); }
    [[nodiscard]] auto copy_from(BufferView<T> source) noexcept { return this->view().copy_from(source); }

    template<typename I>
    [[nodiscard]] decltype(auto) read(I &&i) const noexcept { return this->view().read(std::forward<I>(i)); }

    template<typename I, typename V>
    void write(I &&i, V &&v) const noexcept { this->view().write(std::forward<I>(i), std::forward<V>(v)); }

    template<typename I>
    [[nodiscard]] decltype(auto) atomic(I &&i) const noexcept {
        return this->view().atomic(std::forward<I>(i));
    }
};

template<typename T>
class BufferView {

    static_assert(is_valid_buffer_element_v<T>);

private:
    uint64_t _handle;
    size_t _offset_bytes;
    size_t _size;
    size_t _total_size;

private:
    friend class Heap;
    friend class Buffer<T>;
    template<typename U>
    friend class BufferView;
    BufferView(uint64_t handle, size_t offset_bytes, size_t size, size_t total_size) noexcept
        : _handle{handle}, _offset_bytes{offset_bytes}, _size{size}, _total_size{total_size} {
        if (_offset_bytes % alignof(T) != 0u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid buffer view offset {} for elements with alignment {}.",
                _offset_bytes, alignof(T));
        }
    }

public:
    BufferView(const Buffer<T> &buffer) noexcept : BufferView{buffer.view()} {}

    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto offset() const noexcept { return _offset_bytes / sizeof(T); }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * sizeof(T); }

    [[nodiscard]] auto original() const noexcept {
        return BufferView{_handle, 0u, _total_size, _total_size};
    }
    [[nodiscard]] auto subview(size_t offset_elements, size_t size_elements) const noexcept {
        if (size_elements + offset_elements > _size) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Subview (with offset_elements = {}, size_elements = {}) "
                "overflows buffer view (with size_elements = {}).",
                offset_elements, size_elements, _size);
        }
        return BufferView{_handle, _offset_bytes + offset_elements * sizeof(T), size_elements, _total_size};
    }

    template<typename U>
    [[nodiscard]] auto as() const noexcept {
        if (this->size_bytes() < sizeof(U)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Unable to hold any element (with size = {}) in buffer view (with size = {}).",
                sizeof(U), this->size_bytes());
        }
        return BufferView<U>{_handle, _offset_bytes, this->size_bytes() / sizeof(U), _total_size};
    }

    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return BufferDownloadCommand::create(_handle, offset_bytes(), size_bytes(), data);
    }

    [[nodiscard]] auto copy_from(const void *data) noexcept {
        return BufferUploadCommand::create(this->handle(), this->offset_bytes(), this->size_bytes(), data);
    }

    [[nodiscard]] auto copy_from(BufferView<T> source) noexcept {
        if (source.size() != this->size()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Incompatible buffer views with different element counts (src = {}, dst = {}).",
                source.size(), this->size());
        }
        return BufferCopyCommand::create(
            source.handle(), this->handle(),
            source.offset_bytes(), this->offset_bytes(),
            this->size_bytes());
    }

    template<typename I>
    [[nodiscard]] decltype(auto) read(I &&i) const noexcept {
        return Expr<Buffer<T>>{*this}.read(std::forward<I>(i));
    }

    template<typename I, typename V>
    void write(I &&i, V &&v) const noexcept {
        Expr<Buffer<T>>{*this}.write(std::forward<I>(i), std::forward<V>(v));
    }

    template<typename I>
    [[nodiscard]] decltype(auto) atomic(I &&i) const noexcept {
        return Expr<Buffer<T>>{*this}.atomic(std::forward<I>(i));
    }
};

template<typename T>
BufferView(const Buffer<T> &) -> BufferView<T>;

template<typename T>
BufferView(BufferView<T>) -> BufferView<T>;

namespace detail {

template<typename T>
struct is_buffer_impl : std::false_type {};

template<typename T>
struct is_buffer_impl<Buffer<T>> : std::true_type {};

template<typename T>
struct is_buffer_view_impl : std::false_type {};

template<typename T>
struct is_buffer_view_impl<BufferView<T>> : std::true_type {};

template<typename T>
struct buffer_element_impl {
    using type = T;
};

template<typename T>
struct buffer_element_impl<Buffer<T>> {
    using type = T;
};

template<typename T>
struct buffer_element_impl<BufferView<T>> {
    using type = T;
};

}// namespace detail

template<typename T>
using is_buffer = detail::is_buffer_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_buffer_view = detail::is_buffer_view_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_buffer_or_view = std::disjunction<is_buffer<T>, is_buffer_view<T>>;

template<typename T>
constexpr auto is_buffer_v = is_buffer<T>::value;

template<typename T>
constexpr auto is_buffer_view_v = is_buffer_view<T>::value;

template<typename T>
constexpr auto is_buffer_or_view_v = is_buffer_or_view<T>::value;

template<typename T>
using buffer_element = detail::buffer_element_impl<std::remove_cvref_t<T>>;

template<typename T>
using buffer_element_t = typename buffer_element<T>::type;

class BufferArena {

private:
    std::mutex _mutex;
    Device &_device;
    luisa::vector<luisa::unique_ptr<Resource>> _buffers;
    luisa::optional<BufferView<float4>> _current_buffer;
    size_t _capacity;

public:
    explicit BufferArena(Device &device, size_t capacity = 4_mb) noexcept
        : _device{device}, _capacity{std::max(next_pow2(capacity), 64_kb) / sizeof(float4)} {}

    template<typename T>
    [[nodiscard]] BufferView<T> allocate(size_t n) noexcept {
        static_assert(alignof(T) <= 16u);
        std::scoped_lock lock{_mutex};
        auto size = n * sizeof(T);
        auto n_elem = (size + sizeof(float4) - 1u) / sizeof(float4);
        if (n_elem > _capacity) {// too big, will not use the arena
            auto buffer = luisa::make_unique<Buffer<T>>(_device.create_buffer<T>(n));
            auto view = buffer->view();
            _buffers.emplace_back(std::move(buffer));
            return view;
        }
        if (!_current_buffer || n_elem > _current_buffer->size()) {
            auto buffer = luisa::make_unique<Buffer<float4>>(
                _device.create_buffer<float4>(_capacity));
            _current_buffer = buffer->view();
            _buffers.emplace_back(std::move(buffer));
        }
        auto view = _current_buffer->subview(0u, n_elem);
        _current_buffer = _current_buffer->subview(
            n_elem, _current_buffer->size() - n_elem);
        return view.template as<T>().subview(0u, n);
    }
};

}// namespace luisa::compute
