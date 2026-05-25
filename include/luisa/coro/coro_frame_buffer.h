//
// Created by Mike on 2024/5/10.
//

#pragma once

#include <luisa/runtime/buffer.h>
#include <luisa/dsl/resource.h>
#include <luisa/coro/coro_frame.h>

namespace luisa::compute {

namespace detail {

[[noreturn]] LUISA_CORO_API void error_coro_frame_buffer_invalid_element_size(size_t stride, size_t expected) noexcept;

template<typename T>
class BufferExprProxy;

}// namespace detail

template<>
class BufferView<coroutine::CoroFrame> {

    friend class lc::validation::Stream;

private:
    luisa::shared_ptr<const coroutine::CoroFrameDesc> _desc;
    void *_native_handle;
    uint64_t _handle;
    size_t _offset_bytes;
    size_t _size;
    size_t _total_size;

private:
    friend class Buffer<coroutine::CoroFrame>;
    friend class SparseBuffer<coroutine::CoroFrame>;

    template<typename U>
    friend class BufferView;

public:
    BufferView(luisa::shared_ptr<const coroutine::CoroFrameDesc> desc,
               void *native_handle, uint64_t handle,
               size_t offset_bytes, size_t size, size_t total_size) noexcept
        : _desc{std::move(desc)}, _native_handle{native_handle}, _handle{handle},
          _offset_bytes{offset_bytes}, _size{size}, _total_size{total_size} {
        if (auto a = _desc->type()->alignment(); _offset_bytes % a != 0u) [[unlikely]] {
            detail::error_buffer_invalid_alignment(_offset_bytes, a);
        }
    }

    template<template<typename> typename B>
        requires(is_buffer_v<B<coroutine::CoroFrame>>)
    BufferView(const B<coroutine::CoroFrame> &buffer) noexcept : BufferView{buffer.view()} {}

    BufferView() noexcept : BufferView{nullptr, nullptr, invalid_resource_handle, 0, 0, 0} {}
    [[nodiscard]] explicit operator bool() const noexcept { return _handle != invalid_resource_handle; }

    // properties
    [[nodiscard]] auto desc() const noexcept { return _desc.get(); }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto native_handle() const noexcept { return _native_handle; }
    [[nodiscard]] auto stride() const noexcept { return _desc->type()->size(); }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto offset() const noexcept { return _offset_bytes / stride(); }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * stride(); }

    [[nodiscard]] auto original() const noexcept {
        return BufferView{_desc, _native_handle, _handle, 0u, _total_size, _total_size};
    }
    [[nodiscard]] auto subview(size_t offset_elements, size_t size_elements) const noexcept {
        if (size_elements + offset_elements > _size) [[unlikely]] {
            detail::error_buffer_subview_overflow(offset_elements, size_elements, _size);
        }
        return BufferView{_desc, _native_handle, _handle,
                          _offset_bytes + offset_elements * stride(),
                          size_elements, _total_size};
    }
    // reinterpret cast buffer to another type U
    template<typename U>
        requires(!is_custom_struct_v<U>)
    [[nodiscard]] auto as() const noexcept {
        if (this->size_bytes() < sizeof(U)) [[unlikely]] {
            detail::error_buffer_reinterpret_size_too_small(sizeof(U), this->size_bytes());
        }
        auto total_size_bytes = _total_size * stride();
        return BufferView<U>{_native_handle, _handle, sizeof(U), _offset_bytes,
                             this->size_bytes() / sizeof(U), total_size_bytes / sizeof(U)};
    }
    // commands
    // copy buffer's data to pointer
    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return luisa::make_unique<BufferDownloadCommand>(_handle, offset_bytes(), size_bytes(), data);
    }
    // copy pointer's data to buffer
    [[nodiscard]] auto copy_from(const void *data) noexcept {
        return luisa::make_unique<BufferUploadCommand>(this->handle(), this->offset_bytes(), this->size_bytes(), data);
    }
    // copy source buffer's data to buffer
    [[nodiscard]] auto copy_from(BufferView<coroutine::CoroFrame> source) noexcept {
        if (source.size() != this->size()) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(source.size(), this->size());
        }
        return luisa::make_unique<BufferCopyCommand>(
            source.handle(), this->handle(),
            source.offset_bytes(), this->offset_bytes(),
            this->size_bytes());
    }
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::BufferExprProxy<BufferView<coroutine::CoroFrame>> *>(this);
    }
};

template<>
class Buffer<coroutine::CoroFrame> final : public Resource {

private:
    luisa::shared_ptr<const coroutine::CoroFrameDesc> _desc;
    size_t _size{};

private:
    friend class Device;
    friend class ResourceGenerator;
    friend class DxCudaInterop;
    friend class PinnedMemoryExt;

    Buffer(DeviceInterface *device,
           const BufferCreationInfo &info,
           luisa::shared_ptr<const coroutine::CoroFrameDesc> desc) noexcept
        : Resource{device, Tag::BUFFER, info},
          _desc{std::move(desc)},
          _size{info.total_size_bytes / info.element_stride} {
        if (info.element_stride != _desc->type()->size()) [[unlikely]] {
            detail::error_coro_frame_buffer_invalid_element_size(
                info.element_stride, _desc->type()->size());
        }
    }

    Buffer(DeviceInterface *device,
           const luisa::shared_ptr<const coroutine::CoroFrameDesc> &desc,
           size_t size) noexcept
        : Buffer{device,
                 [&] {
                     if (size == 0) [[unlikely]] {
                         detail::error_buffer_size_is_zero();
                     }
                     return device->create_buffer(desc->type(), size, nullptr);
                 }(),
                 desc} {}

    Buffer(DeviceInterface *device,
           const coroutine::CoroFrameDesc *desc,
           size_t size) noexcept
        : Buffer{device, desc->shared_from_this(), size} {}

public:
    Buffer() noexcept = default;
    ~Buffer() noexcept override {
        if (*this) { device()->destroy_buffer(handle()); }
    }
    Buffer(Buffer &&) noexcept = default;
    Buffer(Buffer const &) noexcept = delete;
    Buffer &operator=(Buffer &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Buffer &operator=(Buffer const &) noexcept = delete;
    using Resource::operator bool;
    // properties
    [[nodiscard]] auto desc() const noexcept { return _desc.get(); }
    [[nodiscard]] auto size() const noexcept {
        _check_is_valid();
        return _size;
    }
    [[nodiscard]] auto stride() const noexcept {
        _check_is_valid();
        return _desc->type()->size();
    }
    [[nodiscard]] auto size_bytes() const noexcept {
        _check_is_valid();
        return _size * stride();
    }
    [[nodiscard]] auto view() const noexcept {
        _check_is_valid();
        return BufferView<coroutine::CoroFrame>{_desc, this->native_handle(), this->handle(), 0u, _size, _size};
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
    [[nodiscard]] auto copy_from(BufferView<coroutine::CoroFrame> source) noexcept {
        return this->view().copy_from(source);
    }
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        _check_is_valid();
        return reinterpret_cast<const detail::BufferExprProxy<Buffer<coroutine::CoroFrame>> *>(this);
    }
};

template<>
struct Expr<Buffer<coroutine::CoroFrame>> {

private:
    luisa::shared_ptr<const coroutine::CoroFrameDesc> _desc;
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(luisa::shared_ptr<const coroutine::CoroFrameDesc> desc,
                  const RefExpr *expr) noexcept
        : _desc{std::move(desc)}, _expression{expr} {}

    Expr(BufferView<coroutine::CoroFrame> buffer) noexcept
        : _desc{buffer.desc()->shared_from_this()},
          _expression{detail::FunctionBuilder::current()->buffer_binding(
              Type::buffer(buffer.desc()->type()), buffer.handle(),
              buffer.offset_bytes(), buffer.size_bytes())} {}

    /// Contruct from Buffer. Will call buffer_binding() to bind buffer
    Expr(const Buffer<coroutine::CoroFrame> &buffer) noexcept
        : Expr{BufferView{buffer}} {}

    /// Construct from Var<Buffer<T>>.
    Expr(const Var<Buffer<coroutine::CoroFrame>> &buffer) noexcept
        : Expr{buffer.desc()->shared_from_this(), buffer.expression()} {}

    /// Construct from Var<BufferView<T>>.
    Expr(const Var<BufferView<coroutine::CoroFrame>> &buffer) noexcept
        : Expr{buffer.desc()->shared_from_this(), buffer.expression()} {}

    [[nodiscard]] const coroutine::CoroFrameDesc *desc() const noexcept { return _desc.get(); }
    [[nodiscard]] const RefExpr *expression() const noexcept { return _expression; }

    /// Read buffer at index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        auto f = detail::FunctionBuilder::current();
        auto expr = f->call(_desc->type(), CallOp::BUFFER_READ,
                            {_expression, detail::extract_expression(std::forward<I>(index))});
        auto frame = f->local(_desc->type());
        f->assign(frame, expr);
        return coroutine::CoroFrame{_desc, frame};
    }

    template<typename I>
        requires is_integral_expr_v<I>
    void write(I &&index, const coroutine::CoroFrame &value) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::BUFFER_WRITE,
            {_expression,
             detail::extract_expression(std::forward<I>(index)),
             value.expression()});
    }

    [[nodiscard]] Expr<uint64_t> device_address() const noexcept {
        return def<uint64_t>(detail::FunctionBuilder::current()->call(
            Type::of<uint64_t>(), CallOp::BUFFER_ADDRESS, {_expression}));
    }

public:
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

namespace detail {

template<template<typename T> typename B>
class BufferExprProxy<B<coroutine::CoroFrame>> {

private:
    using CoroFrameBuffer = B<coroutine::CoroFrame>;
    CoroFrameBuffer _buffer;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(BufferExprProxy)

public:
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        return Expr<CoroFrameBuffer>{_buffer}.read(std::forward<I>(index));
    }
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void write(I &&index, V &&value) const noexcept {
        Expr<CoroFrameBuffer>{_buffer}.write(std::forward<I>(index), std::forward<V>(value));
    }
    [[nodiscard]] Expr<uint64_t> device_address() const noexcept {
        return Expr<CoroFrameBuffer>{_buffer}.device_address();
    }
};

}// namespace detail

}// namespace luisa::compute
