//
// Created by Mike Smith on 2021/3/2.
//

#pragma once

#include <runtime/command.h>
#include <runtime/device.h>

#include <dsl/expr.h>
#include <dsl/arg.h>

namespace luisa::compute::dsl {

#define LUISA_CHECK_BUFFER_ELEMENT_TYPE(T)                    \
    static_assert(alignof(T) <= 16u);                         \
    static_assert(std::is_same_v<T, std::remove_cvref_t<T>>); \
    static_assert(std::is_trivially_copyable_v<T>);           \
    static_assert(std::is_trivially_destructible_v<T>);

template<typename T>
class BufferView;

template<typename T>
class alignas(8) Buffer : public concepts::Noncopyable {
    
    LUISA_CHECK_BUFFER_ELEMENT_TYPE(T)

private:
    Device *_device;
    size_t _size;
    uint64_t _handle;

public:
    Buffer(Device *device, size_t size) noexcept
        : _device{device},
          _handle{device->create_buffer(size * sizeof(T))},
          _size{size} {}

    Buffer(Buffer &&another) noexcept
        : _device{another._device},
          _handle{another._handle},
          _size{another._handle} { another._device = nullptr; }

    Buffer &operator=(Buffer &&rhs) noexcept {
        if (&rhs != this) {
            _device->dispose_buffer(_handle);
            _device = rhs._device;
            _handle = rhs._handle;
            _size = rhs._handle;
            rhs._device = nullptr;
        }
        return *this;
    }

    ~Buffer() noexcept {
        if (_device != nullptr /* not moved */) {
            _device->dispose_buffer(_handle);
        }
    }

    [[nodiscard]] auto view() const noexcept { return BufferView<T>{_device, _handle, 0u, _size}; }
    
    template<typename I>
    [[nodiscard]] decltype(auto) operator[](I &&i) const noexcept {
        return this->view()[std::forward<I>(i)];
    }
    
//    template<typename Index>
//    [[nodiscard]] decltype(auto) load(Index &&index) const noexcept { return view().load(std::forward<Index>(index)); }
//
//    template<typename Index, typename Value>
//    void store(Index &&index, Value &&value) const noexcept { view().store(std::forward<Index>(index), std::forward<Value>(value)); }
};

template<typename T>
class alignas(8) BufferView {
    
    LUISA_CHECK_BUFFER_ELEMENT_TYPE(T)

private:
    Device *_device{nullptr};
    uint64_t _handle{0u};
    size_t _offset_bytes{0u};
    size_t _size{0u};
    const RefExpr *_expression{nullptr};

protected:
    friend class Buffer<T>;
    BufferView(Device *device, uint64_t handle, size_t offset_bytes, size_t size) noexcept
        : _device{device}, _handle{handle}, _offset_bytes{offset_bytes}, _size{size} {
        if (_offset_bytes % alignof(T) != 0u) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid buffer view offset {} for elements with alignment {}.",
                _offset_bytes, alignof(T));
        }
    }

    // for creating function args
    template<typename U>
    friend class Kernel;

    template<typename U>
    friend class Callable;

    explicit BufferView(detail::ArgumentCreation) noexcept
        : _expression{FunctionBuilder::current()->buffer(Type::of<T>())} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }

public:
    BufferView(const Buffer<T> &buffer) noexcept : BufferView{buffer.view()} {}

    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * sizeof(T); }

    [[nodiscard]] auto subview(size_t offset_elements, size_t size_elements) const noexcept {
        if (size_elements * sizeof(T) + offset_elements > _size) {
            LUISA_ERROR_WITH_LOCATION(
                "Subview (with offset_elements = {}, size_elements = {}) overflows buffer view (with size_elements = {}).",
                offset_elements, size_elements, _size);
        }
        return BufferView{_device, _handle, _offset_bytes + offset_elements * sizeof(T), size_elements};
    }

    template<typename U>
    [[nodiscard]] auto as() const noexcept {
        auto byte_size = this->size() * sizeof(T);
        if (byte_size < sizeof(U)) {
            LUISA_ERROR_WITH_LOCATION(
                "Unable to hold any element (with size = {}) in buffer view (with size = {}).",
                sizeof(U), byte_size);
        }
        return BufferView{this->device(), this->handle(), this->offset_bytes(), byte_size / sizeof(U)};
    }

    [[nodiscard]] auto download(T *data) const {
        if (reinterpret_cast<size_t>(data) % alignof(T) != 0u) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid host pointer {} for elements with alignment {}.",
                fmt::ptr(data), alignof(T));
        }
        BufferDownloadCommand{_handle, offset_bytes(), size_bytes(), data};
    }

    [[nodiscard]] auto upload(const T *data) {
        return BufferUploadCommand{this->handle(), this->offset_bytes(), this->size_bytes(), data};
    }

    [[nodiscard]] auto copy(BufferView<T> source) {
        if (source.device() != this->device()) {
            LUISA_ERROR_WITH_LOCATION("Incompatible buffer views created on different devices.");
        }
        if (source.size() != this->size()) {
            LUISA_ERROR_WITH_LOCATION(
                "Incompatible buffer views with different element counts (src = {}, dst = {}).",
                source.size(), this->size());
        }
        return BufferCopyCommand{
            source.handle(), this->handle(),
            source.offset_bytes(), this->offset_bytes(),
            this->size_bytes()};
    }
    
//    template<typename I>
//    [[nodiscard]] auto load(I &&index) const noexcept {
//        // TODO: mark read
//        return (*this)[std::forward<I>(index)];
//    }
//
//    template<typename I, typename Value>
//    void store(I &&i, Value &&v) const noexcept {
//        // TODO: mark write
//        auto p = (*this)[std::forward<I>(i)];
//        p = std::forward<Value>(v);
//    }
    
    
    
    template<concepts::Integral I>
    [[nodiscard]] auto operator[](I i) const noexcept { return this->operator[](detail::Expr{i}); }
    
    template<concepts::Integral I>
    [[nodiscard]] auto operator[](detail::Expr<I> i) const noexcept {
        auto self = _expression ? _expression : FunctionBuilder::current()->buffer_binding(Type::of<T>(), _handle, _offset_bytes);
        auto expr = FunctionBuilder::current()->access(Type::of<T>(), self, i.expression());
        return detail::Expr<T>{expr};
    }
};

template<typename T>
BufferView(const Buffer<T> &) -> BufferView<T>;

template<typename T>
BufferView(BufferView<T>) -> BufferView<T>;

#undef LUISA_CHECK_BUFFER_ELEMENT_TYPE

template<typename T>
struct is_buffer : std::false_type {};

template<typename T>
struct is_buffer<Buffer<T>> : std::true_type {};

template<typename T>
struct is_buffer_view : std::false_type {};

template<typename T>
struct is_buffer_view<BufferView<T>> : std::true_type {};

template<typename T>
using is_buffer_or_view = std::disjunction<is_buffer<T>, is_buffer_view<T>>;

template<typename T>
constexpr auto is_buffer_v = is_buffer<T>::value;

template<typename T>
constexpr auto is_buffer_view_v = is_buffer_view<T>::value;

template<typename T>
constexpr auto is_buffer_or_view_v = is_buffer_or_view<T>::view;

}// namespace luisa::compute::dsl
