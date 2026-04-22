//
// Created by ChenXin on 2024/5/12.
//

#pragma once

#include <luisa/runtime/device.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/builtin.h>
#include <luisa/coro/coro_frame.h>
#include <luisa/core/logging.h>

#include <utility>

namespace luisa::compute {

namespace detail {
template<typename T>
class SOAExprProxy;
}// namespace detail

template<typename T>
class SOA;

struct SOABase {
protected:
    luisa::shared_ptr<const coroutine::CoroFrameDesc> _desc;
    luisa::shared_ptr<luisa::vector<size_t>> _field_offsets;
    size_t _offset_elements{0u}, _size_elements{0u};

public:
    SOABase() noexcept = default;
    SOABase(SOABase &&) noexcept = default;
    SOABase(const SOABase &) noexcept = default;
    SOABase(luisa::shared_ptr<const coroutine::CoroFrameDesc> desc,
            luisa::shared_ptr<luisa::vector<size_t>> field_offsets,
            size_t offset_elements, size_t size_elements) noexcept
        : _desc{std::move(desc)},
          _field_offsets{std::move(field_offsets)},
          _offset_elements{offset_elements}, _size_elements{size_elements} {}
};

template<>
class SOAView<coroutine::CoroFrame> : public SOABase {
private:
    ByteBufferView _buffer_view;

public:
    SOAView() noexcept = default;
    SOAView(luisa::shared_ptr<const coroutine::CoroFrameDesc> desc, ByteBufferView buffer_view,
            luisa::shared_ptr<luisa::vector<size_t>> field_offsets,
            size_t offset_elements, size_t size_elements) noexcept
        : SOABase{std::move(desc), std::move(field_offsets), offset_elements, size_elements},
          _buffer_view{buffer_view} {
        LUISA_ASSERT((_buffer_view.offset_bytes() == 0u) &&
                         (_buffer_view.size_bytes() == _buffer_view.total_size_bytes()),
                     "Invalid buffer view for SOA.");
    }
    template<typename T>
        requires std::same_as<T, SOA<coroutine::CoroFrame>>
    SOAView(const T &soa) noexcept
        : SOAView{soa.view()} {}
    SOAView(const SOAView &) noexcept = default;
    SOAView(SOAView &&) noexcept = default;
    ~SOAView() noexcept = default;

    [[nodiscard]] auto subview(uint offset_elements, uint size_elements) noexcept {
        LUISA_ASSERT(offset_elements + size_elements <= _size_elements,
                     "SOA subview overflow: offset {} + size {} > {}.",
                     offset_elements, size_elements, _size_elements);
        return SOAView{_desc, _buffer_view,
                       _field_offsets, _offset_elements + offset_elements, size_elements};
    }
    [[nodiscard]] auto desc() const noexcept { return _desc.get(); }
    [[nodiscard]] auto handle() const noexcept { return _buffer_view.handle(); }
    [[nodiscard]] auto offset_elements() const noexcept { return _offset_elements; }
    [[nodiscard]] auto size_elements() const noexcept { return _size_elements; }
    [[nodiscard]] auto buffer_offset() const noexcept { return _buffer_view.offset_bytes(); }
    [[nodiscard]] auto size_bytes() const noexcept { return _buffer_view.size_bytes(); }
    [[nodiscard]] auto field_offsets() const noexcept { return _field_offsets; }
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::SOAExprProxy<SOAView<coroutine::CoroFrame>> *>(this);
    }
};

template<>
class SOA<coroutine::CoroFrame> : public SOABase {

private:
    ByteBuffer _buffer;

public:
    SOA(Device &device, luisa::shared_ptr<const coroutine::CoroFrameDesc> desc, size_t n) noexcept
        : SOABase{std::move(desc), luisa::make_shared<luisa::vector<size_t>>(),
                  0u, n} {
        size_t size_bytes = 0u;
        auto fields = _desc->type()->members();
        _field_offsets->reserve(fields.size());
        for (const auto field_type : fields) {
            auto alignment = std::max<size_t>(field_type->alignment(), 4u);
            size_bytes = (size_bytes + alignment - 1u) & ~(alignment - 1u);
            _field_offsets->emplace_back(size_bytes);
            auto aligned_size = (field_type->size() + alignment - 1u) & ~(alignment - 1u);
            size_bytes += aligned_size * _size_elements;
        }
        size_bytes = (size_bytes + 3u) & ~3u;
        _buffer = device.create_byte_buffer(size_bytes);
    }
    SOA(Device &device, const coroutine::CoroFrameDesc *desc, size_t n) noexcept
        : SOA{device, desc->shared_from_this(), n} {}
    SOA() noexcept = default;
    SOA(const SOA &) = delete;
    SOA(SOA &&) noexcept = default;
    SOA &operator=(const SOA &) = delete;
    SOA &operator=(SOA &&x) noexcept {
        _desc = std::move(x._desc);
        _offset_elements = x._offset_elements;
        _size_elements = x._size_elements;
        _field_offsets = std::move(x._field_offsets);
        _buffer = std::move(x._buffer);
        return *this;
    }
    ~SOA() noexcept = default;

    // properties
    [[nodiscard]] auto desc() const noexcept { return _desc.get(); }
    [[nodiscard]] auto view() const noexcept {
        return SOAView<coroutine::CoroFrame>{
            _desc,
            _buffer.view(),
            _field_offsets,
            0u,
            _size_elements};
    }
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::SOAExprProxy<SOA<coroutine::CoroFrame>> *>(this);
    }
};

/// Class of Expr<SOA<coroutine::CoroFrame>>
template<>
struct Expr<SOA<coroutine::CoroFrame>> : SOABase {

private:
    const RefExpr *_expression{nullptr};

public:
    /// Construct from SOAView<coroutine::CoroFrame>. Will call buffer_binding() to bind buffer
    Expr(const SOAView<coroutine::CoroFrame> &soa_view) noexcept
        : SOABase{soa_view.desc()->shared_from_this(), soa_view.field_offsets(),
                  soa_view.offset_elements(), soa_view.size_elements()},
          _expression{detail::FunctionBuilder::current()->buffer_binding(
              Type::of<ByteBuffer>(), soa_view.handle(),
              soa_view.buffer_offset(), soa_view.size_bytes())} {}

    /// Construct from SOA<coroutine::CoroFrame>. Will call buffer_binding() to bind buffer
    Expr(const SOA<coroutine::CoroFrame> &soa) noexcept
        : Expr{SOAView<coroutine::CoroFrame>{soa}} {}

    /// Return RefExpr
    [[nodiscard]] const RefExpr *expression() const noexcept { return _expression; }

    /// Read field with field_index at index
    template<typename V, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] Var<V> read_field(I &&index, uint field_index) const noexcept {
        auto fb = detail::FunctionBuilder::current();
        auto field_type = _desc->type()->members()[field_index];
        auto offset = _field_offsets->at(field_index);
        auto alignment = std::max<size_t>(field_type->alignment(), 4u);
        auto aligned_size = (field_type->size() + alignment - 1u) & ~(alignment - 1u);
        auto offset_var = offset + (_offset_elements + ULong(index)) * aligned_size;
        auto f = fb->local(field_type);
        auto s = fb->call(
            field_type, CallOp::BYTE_BUFFER_READ,
            {_expression, detail::extract_expression(offset_var)});
        fb->assign(f, s);
        return Var<V>(f);
    }

    /// Write field with field_index at index
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void write_field(I &&index, V &&value, uint field_index) const noexcept {
        auto fb = detail::FunctionBuilder::current();
        auto field_type = _desc->type()->members()[field_index];
        auto offset = _field_offsets->at(field_index);
        auto alignment = std::max<size_t>(field_type->alignment(), 4u);
        auto aligned_size = (field_type->size() + alignment - 1u) & ~(alignment - 1u);
        auto offset_var = offset + (_offset_elements + ULong(index)) * aligned_size;
        fb->call(CallOp::BYTE_BUFFER_WRITE,
                 {_expression,
                  detail::extract_expression(offset_var),
                  detail::extract_expression(value)});
    }

    /// Read index with active fields
    template<typename I>
    [[nodiscard]] auto read(
        I &&index, luisa::optional<luisa::span<const uint>> active_fields = luisa::nullopt) const noexcept {
        auto fb = detail::FunctionBuilder::current();
        auto frame = fb->local(_desc->type());
        auto fields = _desc->type()->members();
        for (auto i = 0u; i < fields.size(); i++) {
            if (active_fields && std::find(active_fields->begin(), active_fields->end(), i) == active_fields->end()) { continue; }
            auto field_type = fields[i];
            auto offset = _field_offsets->at(i);
            auto alignment = std::max<size_t>(field_type->alignment(), 4u);
            auto aligned_size = (field_type->size() + alignment - 1u) & ~(alignment - 1u);
            auto offset_var = offset + (_offset_elements + ULong(index)) * aligned_size;
            auto s = fb->call(
                field_type, CallOp::BYTE_BUFFER_READ,
                {_expression, detail::extract_expression(offset_var)});
            auto f = fb->member(field_type, frame, i);
            fb->assign(f, s);
        }
        return coroutine::CoroFrame{_desc, frame};
    }

    /// Write index with active fields
    template<typename I>
    void write(I &&index, const coroutine::CoroFrame &frame,
               luisa::optional<luisa::span<const uint>> active_fields = luisa::nullopt) const noexcept {
        auto fb = detail::FunctionBuilder::current();
        auto fields = _desc->type()->members();
        for (auto i = 0u; i < fields.size(); i++) {
            if (active_fields && std::find(active_fields->begin(), active_fields->end(), i) == active_fields->end()) { continue; }
            auto field_type = fields[i];
            auto offset = _field_offsets->at(i);
            auto alignment = std::max<size_t>(field_type->alignment(), 4u);
            auto aligned_size = (field_type->size() + alignment - 1u) & ~(alignment - 1u);
            auto offset_var = offset + (_offset_elements + ULong(index)) * aligned_size;
            auto f = fb->member(field_type, frame.expression(), i);
            fb->call(CallOp::BYTE_BUFFER_WRITE,
                     {_expression, detail::extract_expression(offset_var), f});
        }
    }

    [[nodiscard]] Expr<uint64_t> device_address() const noexcept {
        return def<uint64_t>(detail::FunctionBuilder::current()->call(
            Type::of<uint64_t>(), CallOp::BUFFER_ADDRESS, {_expression}));
    }

    /// Self-pointer to unify the interfaces of the captured Buffer<T> and Expr<Buffer<T>>
    [[nodiscard]] auto operator->() const noexcept { return this; }
};

/// Class of Expr<SOAView<coroutine::CoroFrame>>
template<>
struct Expr<SOAView<coroutine::CoroFrame>> : public Expr<SOA<coroutine::CoroFrame>> {
    using Expr<SOA<coroutine::CoroFrame>>::Expr;
};

namespace detail {

template<template<typename T> typename B>
class SOAExprProxy<B<coroutine::CoroFrame>> {
private:
    using SOAOrView = B<coroutine::CoroFrame>;
    SOAOrView _soa;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(SOAExprProxy)

public:
    /// Read field with field_index at index
    template<typename V, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] Var<V> read_field(I &&index, uint field_index) const noexcept {
        return Expr<SOAOrView>{_soa}.template read_field<V>(std::forward<I>(index), field_index);
    }
    /// Read field named with "name" at index
    template<typename V, typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] Var<V> read_field(I &&index, luisa::string_view name) const noexcept {
        return read_field<V>(std::forward<I>(index), _soa.desc()->designated_field(name));
    }

    /// Write field with field_index at index
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void write_field(I &&index, V &&value, uint field_index) const noexcept {
        Expr<SOAOrView>{_soa}.write_field(std::forward<I>(index),
                                          std::forward<V>(value),
                                          field_index);
    }
    /// Write field named with "name" at index
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void write_field(I &&index, V &&value, luisa::string_view name) const noexcept {
        write_field(std::forward<I>(index),
                    std::forward<V>(value),
                    _soa.desc()->designated_field(name));
    }

    /// Read index
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        return Expr<SOAOrView>{_soa}.read(std::forward<I>(index), luisa::nullopt);
    }
    /// Read index with active fields
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&index, luisa::span<const uint> active_fields) const noexcept {
        return Expr<SOAOrView>{_soa}.read(std::forward<I>(index), luisa::make_optional(active_fields));
    }

    /// Write index
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void write(I &&index, V &&value) const noexcept {
        Expr<SOAOrView>{_soa}.write(std::forward<I>(index),
                                    std::forward<V>(value),
                                    luisa::nullopt);
    }
    /// Write index with active fields
    template<typename I, typename V>
        requires is_integral_expr_v<I>
    void write(I &&index, V &&value, luisa::span<const uint> active_fields) const noexcept {
        Expr<SOAOrView>{_soa}.write(std::forward<I>(index),
                                    std::forward<V>(value),
                                    luisa::make_optional(active_fields));
    }

    [[nodiscard]] Expr<uint64_t> device_address() const noexcept {
        return Expr<SOAOrView>{_soa}.device_address();
    }
};

}// namespace detail

}// namespace luisa::compute
