#pragma once

#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/vector.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/shared.h>
#include <luisa/runtime/byte_buffer.h>

#include <limits>

namespace luisa::compute::coro {

struct CoroFrameStorageLayout {
    luisa::vector<size_t> field_offsets;
    luisa::vector<size_t> field_strides;
    // For a runtime-capacity SoA layout, field i starts at
    //   field_offsets[i] + capacity * field_capacity_strides[i].
    // Static SoA and AoS layouts leave this vector empty. Keeping the pool
    // capacity out of the constant term makes shader structure (and therefore
    // its cache identity) independent of the allocation size.
    luisa::vector<size_t> field_capacity_strides;
    size_t frame_stride{0u};
    size_t size_bytes{0u};

    [[nodiscard]] auto has_runtime_capacity() const noexcept {
        return !field_capacity_strides.empty();
    }

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

    /// Build an SoA layout whose device-side address expressions accept the
    /// frame-pool capacity as a shader argument. Fields are placed in
    /// non-increasing alignment order. Since DSL alignments are powers of two,
    /// every preceding field stride is then a multiple of the current field's
    /// alignment, so each base has the linear form `prefix_stride * capacity`
    /// for every positive capacity; no capacity-dependent padding constant is
    /// required.
    [[nodiscard]] static auto make_runtime_soa(
        const CoroFrameDesc &desc, size_t capacity) noexcept {
        LUISA_ASSERT(capacity != 0u,
                     "A runtime-capacity coroutine SoA must contain at least one frame.");
        CoroFrameStorageLayout layout;
        auto field_count = desc.frame_field_count();
        layout.field_offsets.resize(field_count, 0u);
        layout.field_strides.resize(field_count, 0u);
        layout.field_capacity_strides.resize(field_count, 0u);

        luisa::vector<size_t> storage_order(field_count);
        for (auto i = 0u; i < field_count; i++) { storage_order[i] = i; }
        luisa::sort(
            storage_order.begin(), storage_order.end(),
            [&desc](auto lhs, auto rhs) noexcept {
                auto lhs_alignment =
                    std::max<size_t>(desc.frame_field_type(lhs)->alignment(), 4u);
                auto rhs_alignment =
                    std::max<size_t>(desc.frame_field_type(rhs)->alignment(), 4u);
                return lhs_alignment != rhs_alignment ?
                           lhs_alignment > rhs_alignment :
                           lhs < rhs;
            });

        size_t prefix_stride = 0u;
        for (auto field_index : storage_order) {
            auto *type = desc.frame_field_type(field_index);
            auto alignment = std::max<size_t>(type->alignment(), 4u);
            LUISA_ASSERT((alignment & (alignment - 1u)) == 0u,
                         "Coroutine frame field alignment {} is not a power of two.",
                         alignment);
            auto stride =
                (type->size() + alignment - 1u) / alignment * alignment;
            LUISA_ASSERT(prefix_stride % alignment == 0u,
                         "Runtime coroutine SoA ordering failed to preserve field alignment.");
            LUISA_ASSERT(
                prefix_stride <= std::numeric_limits<size_t>::max() - stride,
                "Coroutine SoA per-frame stride overflows size_t.");
            layout.field_capacity_strides[field_index] = prefix_stride;
            layout.field_strides[field_index] = stride;
            prefix_stride += stride;
        }
        LUISA_ASSERT(
            prefix_stride == 0u ||
                capacity <= std::numeric_limits<size_t>::max() / prefix_stride,
            "Coroutine SoA allocation size overflows size_t (capacity={}, stride={}).",
            capacity, prefix_stride);
        layout.frame_stride = prefix_stride;
        layout.size_bytes = prefix_stride * capacity;
        LUISA_ASSERT(
            layout.size_bytes <= std::numeric_limits<uint>::max(),
            "Coroutine SoA byte offsets exceed the current 32-bit byte-buffer ABI "
            "(capacity={}, stride={}, bytes={}).",
            capacity, prefix_stride, layout.size_bytes);
        return layout;
    }
};

[[nodiscard]] inline auto coro_frame_is_active_field(
    size_t index, luisa::optional<luisa::span<const size_t>> active_fields,
    bool include_reserved_fields = true) noexcept {
    if (include_reserved_fields &&
        index < CoroFrameDesc::reserved_field_count) { return true; }
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

/// Return the exact token-indexed frame state that must survive relocation of
/// a queued continuation. CoroGraph projects cfg-distill's already solved
/// live_begin sets onto physical frame fields; do not infer this state from
/// immediate callable inputs or materialized stores, because both omit
/// dormant pass-through values by design.
[[nodiscard]] inline auto coro_frame_collect_relocation_fields(
    const CoroGraph &graph, size_t frame_field_count) noexcept {
    luisa::vector<luisa::vector<size_t>> relocation_fields;
    relocation_fields.reserve(graph.node_count());
    for (auto &&node : graph.nodes()) {
        auto fields = luisa::vector<size_t>{node.relocation_fields};
        for (auto field : fields) {
            LUISA_ASSERT(
                field < frame_field_count,
                "Coroutine node {} relocation field {} is outside frame field "
                "count {}.",
                node.index, field, frame_field_count);
        }
        luisa::sort(fields.begin(), fields.end());
        fields.erase(std::unique(fields.begin(), fields.end()), fields.end());
        relocation_fields.emplace_back(std::move(fields));
    }
    return relocation_fields;
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
    void read_into(
        I &&index, const CoroFrame &frame,
        luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
        bool include_reserved_fields = true) const noexcept {
        auto i = def(std::forward<I>(index));
        auto *fb = luisa::compute::detail::FunctionBuilder::current();
        if (!_is_soa()) {
            auto *src = fb->access(_desc->frame_type(), _expressions.front(), i.expression());
            if (!active_fields) {
                fb->assign(frame.expression(), src);
            } else {
                for (auto field_index = 0u;
                     field_index < _desc->frame_field_count();
                     field_index++) {
                    if (!coro_frame_is_active_field(
                            field_index, active_fields,
                            include_reserved_fields)) { continue; }
                    auto *type = _desc->frame_field_type(field_index);
                    auto *dst = fb->member(
                        type, frame.expression(), field_index);
                    auto *field = fb->member(type, src, field_index);
                    fb->assign(dst, field);
                }
            }
        } else {
            for (auto field_index = 0u; field_index < _desc->frame_field_count(); field_index++) {
                if (!coro_frame_is_active_field(
                        field_index, active_fields,
                        include_reserved_fields)) { continue; }
                auto *type = _desc->frame_field_type(field_index);
                auto *dst = fb->member(
                    type, frame.expression(), field_index);
                auto *src = fb->access(type, _expressions[field_index], i.expression());
                fb->assign(dst, src);
            }
        }
    }

    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(
        I &&index,
        luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt) const noexcept {
        auto frame = CoroFrame::create(_desc);
        read_into(
            std::forward<I>(index), frame, active_fields);
        return frame;
    }

    template<typename I>
        requires is_integral_expr_v<I>
    void write(
        I &&index, const CoroFrame &frame,
        luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
        bool include_reserved_fields = true) const noexcept {
        auto i = def(std::forward<I>(index));
        auto *fb = luisa::compute::detail::FunctionBuilder::current();
        if (!_is_soa()) {
            auto *dst = fb->access(_desc->frame_type(), _expressions.front(), i.expression());
            if (!active_fields) {
                fb->assign(dst, frame.expression());
            } else {
                for (auto field_index = 0u;
                     field_index < _desc->frame_field_count();
                     field_index++) {
                    if (!coro_frame_is_active_field(
                            field_index, active_fields,
                            include_reserved_fields)) { continue; }
                    auto *type = _desc->frame_field_type(field_index);
                    auto *src = fb->member(type, frame.expression(), field_index);
                    auto *field = fb->member(type, dst, field_index);
                    fb->assign(field, src);
                }
            }
        } else {
            for (auto field_index = 0u; field_index < _desc->frame_field_count(); field_index++) {
                if (!coro_frame_is_active_field(
                        field_index, active_fields,
                        include_reserved_fields)) { continue; }
                auto *type = _desc->frame_field_type(field_index);
                auto *src = fb->member(type, frame.expression(), field_index);
                auto *dst = fb->access(type, _expressions[field_index], i.expression());
                fb->assign(dst, src);
            }
        }
    }
};

inline void coro_frame_store_aos(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index, const CoroFrame &frame,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    auto base = frame_index * static_cast<uint>(layout.frame_stride);
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    for (auto field_index = 0u; field_index < frame.desc()->frame_field_count(); field_index++) {
        if (!coro_frame_is_active_field(
                field_index, active_fields,
                include_reserved_fields)) { continue; }
        auto *type = frame.desc()->frame_field_type(field_index);
        auto *member = fb->member(type, frame.expression(), field_index);
        fb->call(is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_WRITE : CallOp::BYTE_BUFFER_WRITE,
                 {buffer.expression(),
                  luisa::compute::detail::extract_expression(base + static_cast<uint>(layout.field_offsets[field_index])),
                  member});
    }
}

inline void coro_frame_load_aos_into(
    const CoroFrame &frame, const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    auto base = frame_index * static_cast<uint>(layout.frame_stride);
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    for (auto field_index = 0u;
         field_index < frame.desc()->frame_field_count(); field_index++) {
        if (!coro_frame_is_active_field(
                field_index, active_fields,
                include_reserved_fields)) { continue; }
        auto *type = frame.desc()->frame_field_type(field_index);
        auto *value = fb->call(type, is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_READ : CallOp::BYTE_BUFFER_READ,
                               {buffer.expression(),
                                luisa::compute::detail::extract_expression(base + static_cast<uint>(layout.field_offsets[field_index]))});
        auto *member = fb->member(type, frame.expression(), field_index);
        fb->assign(member, value);
    }
}

[[nodiscard]] inline auto coro_frame_load_aos(
    const CoroFrameDesc *desc, const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false) noexcept {
    auto frame = CoroFrame::create(desc);
    coro_frame_load_aos_into(
        frame, buffer, frame_index, layout, active_fields, is_volatile);
    return frame;
}

inline void coro_frame_store_soa(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index, const CoroFrame &frame,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    for (auto field_index = 0u; field_index < frame.desc()->frame_field_count(); field_index++) {
        if (!coro_frame_is_active_field(
                field_index, active_fields,
                include_reserved_fields)) { continue; }
        auto offset = static_cast<uint>(layout.field_offsets[field_index]) +
                      frame_index * static_cast<uint>(layout.field_strides[field_index]);
        auto *type = frame.desc()->frame_field_type(field_index);
        auto *member = fb->member(type, frame.expression(), field_index);
        fb->call(is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_WRITE : CallOp::BYTE_BUFFER_WRITE,
                 {buffer.expression(), luisa::compute::detail::extract_expression(offset), member});
    }
}

inline void coro_frame_load_soa_into(
    const CoroFrame &frame, const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    for (auto field_index = 0u;
         field_index < frame.desc()->frame_field_count(); field_index++) {
        if (!coro_frame_is_active_field(
                field_index, active_fields,
                include_reserved_fields)) { continue; }
        auto offset = static_cast<uint>(layout.field_offsets[field_index]) +
                      frame_index * static_cast<uint>(layout.field_strides[field_index]);
        auto *type = frame.desc()->frame_field_type(field_index);
        auto *value = fb->call(type, is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_READ : CallOp::BYTE_BUFFER_READ,
                               {buffer.expression(), luisa::compute::detail::extract_expression(offset)});
        auto *member = fb->member(type, frame.expression(), field_index);
        fb->assign(member, value);
    }
}

[[nodiscard]] inline auto coro_frame_load_soa(
    const CoroFrameDesc *desc, const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false) noexcept {
    auto frame = CoroFrame::create(desc);
    coro_frame_load_soa_into(
        frame, buffer, frame_index, layout, active_fields, is_volatile);
    return frame;
}

[[nodiscard]] inline auto coro_frame_runtime_soa_offset(
    Expr<uint> frame_index, Expr<uint> capacity,
    const CoroFrameStorageLayout &layout, size_t field_index) noexcept {
    LUISA_ASSERT(layout.has_runtime_capacity(),
                 "A runtime SoA address requires a capacity-parameterized layout.");
    LUISA_ASSERT(field_index < layout.field_offsets.size(),
                 "Coroutine frame field index {} is out of range {}.",
                 field_index, layout.field_offsets.size());
    return capacity * static_cast<uint>(layout.field_capacity_strides[field_index]) +
           frame_index * static_cast<uint>(layout.field_strides[field_index]) +
           static_cast<uint>(layout.field_offsets[field_index]);
}

inline void coro_frame_store_runtime_soa(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    Expr<uint> capacity, const CoroFrame &frame,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    for (auto field_index = 0u;
         field_index < frame.desc()->frame_field_count(); field_index++) {
        if (!coro_frame_is_active_field(
                field_index, active_fields,
                include_reserved_fields)) { continue; }
        auto offset = coro_frame_runtime_soa_offset(
            frame_index, capacity, layout, field_index);
        auto *type = frame.desc()->frame_field_type(field_index);
        auto *member = fb->member(type, frame.expression(), field_index);
        fb->call(is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_WRITE :
                               CallOp::BYTE_BUFFER_WRITE,
                 {buffer.expression(),
                  luisa::compute::detail::extract_expression(offset), member});
    }
}

inline void coro_frame_load_runtime_soa_into(
    const CoroFrame &frame, const Expr<ByteBuffer> &buffer,
    Expr<uint> frame_index, Expr<uint> capacity,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    for (auto field_index = 0u;
         field_index < frame.desc()->frame_field_count(); field_index++) {
        if (!coro_frame_is_active_field(
                field_index, active_fields,
                include_reserved_fields)) { continue; }
        auto offset = coro_frame_runtime_soa_offset(
            frame_index, capacity, layout, field_index);
        auto *type = frame.desc()->frame_field_type(field_index);
        auto *value = fb->call(
            type,
            is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_READ :
                          CallOp::BYTE_BUFFER_READ,
            {buffer.expression(),
             luisa::compute::detail::extract_expression(offset)});
        auto *member = fb->member(type, frame.expression(), field_index);
        fb->assign(member, value);
    }
}

[[nodiscard]] inline auto coro_frame_load_runtime_soa(
    const CoroFrameDesc *desc, const Expr<ByteBuffer> &buffer,
    Expr<uint> frame_index, Expr<uint> capacity,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false) noexcept {
    auto frame = CoroFrame::create(desc);
    coro_frame_load_runtime_soa_into(
        frame, buffer, frame_index, capacity,
        layout, active_fields, is_volatile);
    return frame;
}

inline void coro_frame_store(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index, const CoroFrame &frame,
    const CoroFrameStorageLayout &layout, bool soa,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    if (soa) {
        coro_frame_store_soa(
            buffer, frame_index, frame, layout, active_fields,
            is_volatile, include_reserved_fields);
    } else {
        coro_frame_store_aos(
            buffer, frame_index, frame, layout, active_fields,
            is_volatile, include_reserved_fields);
    }
}

inline void coro_frame_store(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    Expr<uint> soa_capacity, const CoroFrame &frame,
    const CoroFrameStorageLayout &layout, bool soa,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    if (soa) {
        coro_frame_store_runtime_soa(
            buffer, frame_index, soa_capacity, frame,
            layout, active_fields, is_volatile,
            include_reserved_fields);
    } else {
        coro_frame_store_aos(
            buffer, frame_index, frame,
            layout, active_fields, is_volatile,
            include_reserved_fields);
    }
}

inline void coro_frame_load_into(
    const CoroFrame &frame, const Expr<ByteBuffer> &buffer,
    Expr<uint> frame_index, const CoroFrameStorageLayout &layout, bool soa,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    if (soa) {
        coro_frame_load_soa_into(
            frame, buffer, frame_index, layout,
            active_fields, is_volatile,
            include_reserved_fields);
    } else {
        coro_frame_load_aos_into(
            frame, buffer, frame_index, layout,
            active_fields, is_volatile,
            include_reserved_fields);
    }
}

inline void coro_frame_load_into(
    const CoroFrame &frame, const Expr<ByteBuffer> &buffer,
    Expr<uint> frame_index, Expr<uint> soa_capacity,
    const CoroFrameStorageLayout &layout, bool soa,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    if (soa) {
        coro_frame_load_runtime_soa_into(
            frame, buffer, frame_index, soa_capacity,
            layout, active_fields, is_volatile,
            include_reserved_fields);
    } else {
        coro_frame_load_aos_into(
            frame, buffer, frame_index, layout,
            active_fields, is_volatile,
            include_reserved_fields);
    }
}

[[nodiscard]] inline auto coro_frame_load(
    const CoroFrameDesc *desc, const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout, bool soa,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false) noexcept {
    return soa ?
               coro_frame_load_soa(desc, buffer, frame_index, layout, active_fields, is_volatile) :
               coro_frame_load_aos(desc, buffer, frame_index, layout, active_fields, is_volatile);
}

[[nodiscard]] inline auto coro_frame_load(
    const CoroFrameDesc *desc, const Expr<ByteBuffer> &buffer,
    Expr<uint> frame_index, Expr<uint> soa_capacity,
    const CoroFrameStorageLayout &layout, bool soa,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false) noexcept {
    return soa ?
               coro_frame_load_runtime_soa(
                   desc, buffer, frame_index, soa_capacity,
                   layout, active_fields, is_volatile) :
               coro_frame_load_aos(
                   desc, buffer, frame_index,
                   layout, active_fields, is_volatile);
}

template<typename T>
[[nodiscard]] inline auto coro_frame_read_field(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout, bool soa, size_t field_index) noexcept {
    auto offset = def(static_cast<uint>(layout.field_offsets[field_index]));
    offset += frame_index * static_cast<uint>(soa ? layout.field_strides[field_index] : layout.frame_stride);
    return buffer.template read<T>(offset);
}

template<typename V>
inline void coro_frame_write_field(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout, bool soa, size_t field_index, V &&value) noexcept {
    auto offset = def(static_cast<uint>(layout.field_offsets[field_index]));
    offset += frame_index * static_cast<uint>(soa ? layout.field_strides[field_index] : layout.frame_stride);
    buffer.write(offset, std::forward<V>(value));
}

template<typename T>
[[nodiscard]] inline auto coro_frame_read_field(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    Expr<uint> soa_capacity, const CoroFrameStorageLayout &layout,
    bool soa, size_t field_index) noexcept {
    if (soa) {
        auto offset = coro_frame_runtime_soa_offset(
            frame_index, soa_capacity, layout, field_index);
        return buffer.template read<T>(offset);
    }
    return coro_frame_read_field<T>(
        buffer, frame_index, layout, false, field_index);
}

template<typename V>
inline void coro_frame_write_field(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    Expr<uint> soa_capacity, const CoroFrameStorageLayout &layout,
    bool soa, size_t field_index, V &&value) noexcept {
    if (soa) {
        auto offset = coro_frame_runtime_soa_offset(
            frame_index, soa_capacity, layout, field_index);
        buffer.write(offset, std::forward<V>(value));
    } else {
        coro_frame_write_field(
            buffer, frame_index, layout, false,
            field_index, std::forward<V>(value));
    }
}

}// namespace luisa::compute::coro
