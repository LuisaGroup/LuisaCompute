#pragma once

#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/vector.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/shared.h>
#include <luisa/runtime/byte_buffer.h>

namespace luisa::compute::coro {

struct CoroFrameStorageLayout {
    luisa::vector<size_t> field_offsets;
    luisa::vector<size_t> field_strides;
    size_t frame_stride{0u};
    size_t size_bytes{0u};

    [[nodiscard]] static auto make_aos(const CoroFrameDesc &desc, size_t capacity) noexcept {
        CoroFrameStorageLayout layout;
        auto *frame_type = desc.frame_type();
        layout.frame_stride = frame_type->size();
        layout.size_bytes = layout.frame_stride * capacity;
        layout.field_offsets.reserve(desc.frame_field_count());
        size_t offset = 0u;
        for (auto i = 0u; i < desc.frame_field_count(); i++) {
            auto *type = desc.frame_field_type(i);
            auto alignment = type->alignment();
            offset = (offset + alignment - 1u) / alignment * alignment;
            layout.field_offsets.emplace_back(offset);
            layout.field_strides.emplace_back(0u);
            offset += type->size();
        }
        return layout;
    }

    [[nodiscard]] static auto make_soa(const CoroFrameDesc &desc, size_t capacity) noexcept {
        CoroFrameStorageLayout layout;
        layout.field_offsets.reserve(desc.frame_field_count());
        layout.field_strides.reserve(desc.frame_field_count());
        size_t size_bytes = 0u;
        for (auto i = 0u; i < desc.frame_field_count(); i++) {
            auto *type = desc.frame_field_type(i);
            auto alignment = std::max<size_t>(type->alignment(), 4u);
            auto stride = (type->size() + alignment - 1u) / alignment * alignment;
            size_bytes = (size_bytes + alignment - 1u) / alignment * alignment;
            layout.field_offsets.emplace_back(size_bytes);
            layout.field_strides.emplace_back(stride);
            size_bytes += stride * capacity;
        }
        layout.size_bytes = (size_bytes + 3u) / 4u * 4u;
        return layout;
    }
};

[[nodiscard]] inline auto coro_frame_is_active_field(
    size_t index, luisa::optional<luisa::span<const size_t>> active_fields) noexcept {
    if (index < CoroFrameDesc::reserved_field_count) { return true; }
    return !active_fields || std::find(active_fields->begin(), active_fields->end(), index) != active_fields->end();
}

inline void coro_frame_append_unique(luisa::vector<size_t> &fields, size_t field) noexcept {
    if (std::find(fields.begin(), fields.end(), field) == fields.end()) {
        fields.emplace_back(field);
    }
}

inline void coro_frame_append_reserved_fields(luisa::vector<size_t> &fields) noexcept {
    for (auto i = 0u; i < CoroFrameDesc::reserved_field_count; i++) {
        coro_frame_append_unique(fields, i);
    }
}

[[nodiscard]] inline auto coro_frame_collect_input_fields(const CoroGraph &graph, size_t node_index) noexcept {
    return luisa::vector<size_t>{graph.node(node_index).input_fields};
}

[[nodiscard]] inline auto coro_frame_collect_output_fields(const CoroGraph &graph, size_t node_index) noexcept {
    return luisa::vector<size_t>{graph.node(node_index).output_fields};
}

[[nodiscard]] inline auto coro_frame_collect_output_fields(const CoroGraph &graph,
                                                          size_t node_index,
                                                          size_t target_index) noexcept {
    luisa::vector<size_t> fields;
    coro_frame_append_reserved_fields(fields);
    if (auto *edge = graph.edge(node_index, target_index)) {
        for (auto field : edge->store_fields) {
            coro_frame_append_unique(fields, field);
        }
    }
    luisa::sort(fields.begin(), fields.end());
    return fields;
}

class CoroFrameSharedStorage {

private:
    const CoroFrameDesc *_desc{nullptr};
    luisa::vector<const RefExpr *> _expressions;
    size_t _size{0u};

private:
    [[nodiscard]] auto _is_soa() const noexcept { return _expressions.size() > 1u; }

public:
    CoroFrameSharedStorage(const CoroFrameDesc *desc, size_t size, bool soa) noexcept
        : _desc{desc}, _size{size} {
        auto *fb = luisa::compute::detail::FunctionBuilder::current();
        if (!soa) {
            _expressions.emplace_back(fb->shared(Type::array(_desc->frame_type(), _size)));
        } else {
            _expressions.reserve(_desc->frame_field_count());
            for (auto i = 0u; i < _desc->frame_field_count(); i++) {
                auto *type = _desc->frame_field_type(i);
                _expressions.emplace_back(fb->shared(Type::array(type, _size)));
            }
        }
    }

    CoroFrameSharedStorage(CoroFrameSharedStorage &&) noexcept = default;
    CoroFrameSharedStorage(const CoroFrameSharedStorage &) noexcept = delete;
    CoroFrameSharedStorage &operator=(CoroFrameSharedStorage &&) noexcept = delete;
    CoroFrameSharedStorage &operator=(const CoroFrameSharedStorage &) noexcept = delete;

    [[nodiscard]] auto desc() const noexcept { return _desc; }
    [[nodiscard]] auto size() const noexcept { return _size; }

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(
        I &&index,
        luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt) const noexcept {
        auto i = def(std::forward<I>(index));
        auto *fb = luisa::compute::detail::FunctionBuilder::current();
        auto *frame_expr = fb->local(_desc->frame_type());
        fb->assign(frame_expr, fb->call(_desc->frame_type(), CallOp::ZERO, {}));
        if (!_is_soa()) {
            auto *src = fb->access(_desc->frame_type(), _expressions.front(), i.expression());
            if (!active_fields) {
                fb->assign(frame_expr, src);
            } else {
                for (auto field_index : *active_fields) {
                    auto *type = _desc->frame_field_type(field_index);
                    auto *dst = fb->member(type, frame_expr, field_index);
                    auto *field = fb->member(type, src, field_index);
                    fb->assign(dst, field);
                }
            }
        } else {
            for (auto field_index = 0u; field_index < _desc->frame_field_count(); field_index++) {
                if (!coro_frame_is_active_field(field_index, active_fields)) { continue; }
                auto *type = _desc->frame_field_type(field_index);
                auto *dst = fb->member(type, frame_expr, field_index);
                auto *src = fb->access(type, _expressions[field_index], i.expression());
                fb->assign(dst, src);
            }
        }
        return CoroFrame{_desc, static_cast<const Expression *>(frame_expr)};
    }

    template<typename I>
        requires is_integral_expr_v<I>
    void write(
        I &&index, const CoroFrame &frame,
        luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt) const noexcept {
        auto i = def(std::forward<I>(index));
        auto *fb = luisa::compute::detail::FunctionBuilder::current();
        if (!_is_soa()) {
            auto *dst = fb->access(_desc->frame_type(), _expressions.front(), i.expression());
            if (!active_fields) {
                fb->assign(dst, frame.expression());
            } else {
                for (auto field_index : *active_fields) {
                    auto *type = _desc->frame_field_type(field_index);
                    auto *src = fb->member(type, frame.expression(), field_index);
                    auto *field = fb->member(type, dst, field_index);
                    fb->assign(field, src);
                }
            }
        } else {
            for (auto field_index = 0u; field_index < _desc->frame_field_count(); field_index++) {
                if (!coro_frame_is_active_field(field_index, active_fields)) { continue; }
                auto *type = _desc->frame_field_type(field_index);
                auto *src = fb->member(type, frame.expression(), field_index);
                auto *dst = fb->access(type, _expressions[field_index], i.expression());
                fb->assign(dst, src);
            }
        }
    }
};

inline void coro_frame_store_aos(
    const Var<ByteBuffer> &buffer, Expr<uint> frame_index, const CoroFrame &frame,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false) noexcept {
    auto base = frame_index * static_cast<uint>(layout.frame_stride);
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    for (auto field_index = 0u; field_index < frame.desc()->frame_field_count(); field_index++) {
        if (!coro_frame_is_active_field(field_index, active_fields)) { continue; }
        auto *type = frame.desc()->frame_field_type(field_index);
        auto *member = fb->member(type, frame.expression(), field_index);
        fb->call(is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_WRITE : CallOp::BYTE_BUFFER_WRITE,
                 {buffer.expression(),
                  luisa::compute::detail::extract_expression(base + static_cast<uint>(layout.field_offsets[field_index])),
                  member});
    }
}

[[nodiscard]] inline auto coro_frame_load_aos(
    const CoroFrameDesc *desc, const Var<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false) noexcept {
    auto frame = CoroFrame::create(desc);
    auto base = frame_index * static_cast<uint>(layout.frame_stride);
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    for (auto field_index = 0u; field_index < desc->frame_field_count(); field_index++) {
        if (!coro_frame_is_active_field(field_index, active_fields)) { continue; }
        auto *type = desc->frame_field_type(field_index);
        auto *value = fb->call(type, is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_READ : CallOp::BYTE_BUFFER_READ,
                               {buffer.expression(),
                                luisa::compute::detail::extract_expression(base + static_cast<uint>(layout.field_offsets[field_index]))});
        auto *member = fb->member(type, frame.expression(), field_index);
        fb->assign(member, value);
    }
    return frame;
}

inline void coro_frame_store_soa(
    const Var<ByteBuffer> &buffer, Expr<uint> frame_index, const CoroFrame &frame,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false) noexcept {
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    for (auto field_index = 0u; field_index < frame.desc()->frame_field_count(); field_index++) {
        if (!coro_frame_is_active_field(field_index, active_fields)) { continue; }
        auto offset = static_cast<uint>(layout.field_offsets[field_index]) +
                      frame_index * static_cast<uint>(layout.field_strides[field_index]);
        auto *type = frame.desc()->frame_field_type(field_index);
        auto *member = fb->member(type, frame.expression(), field_index);
        fb->call(is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_WRITE : CallOp::BYTE_BUFFER_WRITE,
                 {buffer.expression(), luisa::compute::detail::extract_expression(offset), member});
    }
}

[[nodiscard]] inline auto coro_frame_load_soa(
    const CoroFrameDesc *desc, const Var<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false) noexcept {
    auto frame = CoroFrame::create(desc);
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    for (auto field_index = 0u; field_index < desc->frame_field_count(); field_index++) {
        if (!coro_frame_is_active_field(field_index, active_fields)) { continue; }
        auto offset = static_cast<uint>(layout.field_offsets[field_index]) +
                      frame_index * static_cast<uint>(layout.field_strides[field_index]);
        auto *type = desc->frame_field_type(field_index);
        auto *value = fb->call(type, is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_READ : CallOp::BYTE_BUFFER_READ,
                               {buffer.expression(), luisa::compute::detail::extract_expression(offset)});
        auto *member = fb->member(type, frame.expression(), field_index);
        fb->assign(member, value);
    }
    return frame;
}

inline void coro_frame_store(
    const Var<ByteBuffer> &buffer, Expr<uint> frame_index, const CoroFrame &frame,
    const CoroFrameStorageLayout &layout, bool soa,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false) noexcept {
    if (soa) {
        coro_frame_store_soa(buffer, frame_index, frame, layout, active_fields, is_volatile);
    } else {
        coro_frame_store_aos(buffer, frame_index, frame, layout, active_fields, is_volatile);
    }
}

[[nodiscard]] inline auto coro_frame_load(
    const CoroFrameDesc *desc, const Var<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout, bool soa,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false) noexcept {
    return soa ?
               coro_frame_load_soa(desc, buffer, frame_index, layout, active_fields, is_volatile) :
               coro_frame_load_aos(desc, buffer, frame_index, layout, active_fields, is_volatile);
}

template<typename T>
[[nodiscard]] inline auto coro_frame_read_field(
    const Var<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout, bool soa, size_t field_index) noexcept {
    auto offset = def(static_cast<uint>(layout.field_offsets[field_index]));
    offset += frame_index * static_cast<uint>(soa ? layout.field_strides[field_index] : layout.frame_stride);
    return buffer.template read<T>(offset);
}

template<typename V>
inline void coro_frame_write_field(
    const Var<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout, bool soa, size_t field_index, V &&value) noexcept {
    auto offset = def(static_cast<uint>(layout.field_offsets[field_index]));
    offset += frame_index * static_cast<uint>(soa ? layout.field_strides[field_index] : layout.frame_stride);
    buffer.write(offset, std::forward<V>(value));
}

}// namespace luisa::compute::coro
