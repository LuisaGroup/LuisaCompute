#pragma once

#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/vector.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/resource.h>
#include <luisa/dsl/shared.h>
#include <luisa/dsl/stmt.h>
#include <luisa/runtime/byte_buffer.h>

#include <array>
#include <limits>

namespace luisa::compute::coro {

inline constexpr luisa::string_view coro_frame_raw_access_comment =
    "luisa.coro.frame.raw";

struct CoroFrameStorageLayout {
    struct PackedChunk {
        size_t capacity_stride{0u};
        uint8_t width{0u};
        std::array<uint32_t, 4u> fields{0u, 0u, 0u, 0u};
    };

    luisa::vector<size_t> field_offsets;
    luisa::vector<size_t> field_strides;
    // For a runtime-capacity SoA layout, field i starts at
    //   field_offsets[i] + capacity * field_capacity_strides[i] + frame_index * field_strides[i].
    // Static SoA and AoS layouts leave this vector empty. Keeping the pool
    // capacity out of the constant term makes shader structure (and therefore
    // its cache identity) independent of the allocation size.
    luisa::vector<size_t> field_capacity_strides;
    luisa::vector<PackedChunk> packed_chunks;
    size_t frame_stride{0u};
    size_t size_bytes{0u};

    [[nodiscard]] auto has_runtime_capacity() const noexcept {
        return !field_capacity_strides.empty();
    }

    [[nodiscard]] static auto is_packable_dword_field_type(const Type *type) noexcept {
        return type != nullptr &&
               type->size() == 4u &&
               type->alignment() <= 4u &&
               (type->is_scalar() || type->is_vector()) &&
               !type->is_bool_or_bool_vector();
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

    /// Build a runtime-capacity SoA layout that packs 4-byte scalar/vector
    /// fields belonging to the exact same co-access equivalence class into
    /// wide 16-byte (`uint4`) and 8-byte (`uint2`) chunks while keeping
    /// single-field-queried slots (`target_token`, hints, extension keys) and
    /// reserved slots in dense standalone arrays.
    [[nodiscard]] static auto make_packed_runtime_soa(
        const CoroFrameDesc &desc, size_t capacity,
        luisa::span<const luisa::vector<size_t>> active_field_sets,
        luisa::span<const size_t> standalone_fields = {}) noexcept {
        LUISA_ASSERT(capacity != 0u,
                     "A runtime-capacity coroutine SoA must contain at least one frame.");
        if (active_field_sets.empty()) {
            return make_runtime_soa(desc, capacity);
        }
        auto field_count = desc.frame_field_count();
        luisa::vector<uint8_t> standalone(field_count, 0u);
        for (size_t f = 0u;
             f < std::min<size_t>(field_count, CoroFrameDesc::reserved_field_count);
             ++f) {
            standalone[f] = 1u;
        }
        for (auto f : standalone_fields) {
            if (f < field_count) { standalone[f] = 1u; }
        }
        for (size_t f = 0u; f < field_count; ++f) {
            if (!is_packable_dword_field_type(desc.frame_field_type(f))) {
                standalone[f] = 1u;
            }
        }

        auto word_count = (active_field_sets.size() + 63u) / 64u;
        luisa::vector<luisa::vector<uint64_t>> signatures(
            field_count, luisa::vector<uint64_t>(word_count, 0ull));
        for (size_t s = 0u; s < active_field_sets.size(); ++s) {
            auto word = s >> 6u;
            auto bit = uint64_t{1ull} << (s & 63u);
            for (auto f : active_field_sets[s]) {
                if (f < field_count && standalone[f] == 0u) {
                    signatures[f][word] |= bit;
                }
            }
        }

        struct StorageGroup {
            size_t alignment{4u};
            size_t stride{4u};
            uint8_t width{1u};
            std::array<uint32_t, 4u> fields{0u, 0u, 0u, 0u};
        };
        luisa::vector<StorageGroup> groups;
        groups.reserve(field_count);

        luisa::vector<uint8_t> placed(field_count, 0u);
        for (size_t f = 0u; f < field_count; ++f) {
            if (placed[f] != 0u) { continue; }
            auto non_zero = std::any_of(
                signatures[f].begin(), signatures[f].end(),
                [](uint64_t w) noexcept { return w != 0ull; });
            if (standalone[f] != 0u || !non_zero) {
                placed[f] = 1u;
                auto *type = desc.frame_field_type(f);
                auto alignment = std::max<size_t>(type->alignment(), 4u);
                LUISA_ASSERT((alignment & (alignment - 1u)) == 0u,
                             "Coroutine frame field alignment {} is not a power of two.",
                             alignment);
                auto stride = (type->size() + alignment - 1u) / alignment * alignment;
                groups.emplace_back(StorageGroup{
                    .alignment = alignment,
                    .stride = stride,
                    .width = 1u,
                    .fields = {static_cast<uint32_t>(f), 0u, 0u, 0u}});
                continue;
            }
            luisa::vector<uint32_t> equiv;
            equiv.emplace_back(static_cast<uint32_t>(f));
            placed[f] = 1u;
            for (size_t g = f + 1u; g < field_count; ++g) {
                if (placed[g] == 0u && standalone[g] == 0u &&
                    signatures[g] == signatures[f]) {
                    equiv.emplace_back(static_cast<uint32_t>(g));
                    placed[g] = 1u;
                }
            }
            size_t cursor = 0u;
            while (cursor + 4u <= equiv.size()) {
                groups.emplace_back(StorageGroup{
                    .alignment = 16u,
                    .stride = 16u,
                    .width = 4u,
                    .fields = {equiv[cursor], equiv[cursor + 1u],
                               equiv[cursor + 2u], equiv[cursor + 3u]}});
                cursor += 4u;
            }
            if (cursor + 2u <= equiv.size()) {
                groups.emplace_back(StorageGroup{
                    .alignment = 8u,
                    .stride = 8u,
                    .width = 2u,
                    .fields = {equiv[cursor], equiv[cursor + 1u], 0u, 0u}});
                cursor += 2u;
            }
            while (cursor < equiv.size()) {
                groups.emplace_back(StorageGroup{
                    .alignment = 4u,
                    .stride = 4u,
                    .width = 1u,
                    .fields = {equiv[cursor], 0u, 0u, 0u}});
                ++cursor;
            }
        }

        luisa::sort(groups.begin(), groups.end(),
                    [](const StorageGroup &lhs, const StorageGroup &rhs) noexcept {
                        return lhs.alignment != rhs.alignment ?
                                   lhs.alignment > rhs.alignment :
                                   lhs.fields[0] < rhs.fields[0];
                    });

        CoroFrameStorageLayout layout;
        layout.field_offsets.resize(field_count, 0u);
        layout.field_strides.resize(field_count, 0u);
        layout.field_capacity_strides.resize(field_count, 0u);
        size_t prefix_stride = 0u;
        for (const auto &group : groups) {
            LUISA_ASSERT(prefix_stride % group.alignment == 0u,
                         "Packed runtime coroutine SoA ordering failed to preserve alignment.");
            LUISA_ASSERT(
                prefix_stride <= std::numeric_limits<size_t>::max() - group.stride,
                "Coroutine SoA per-frame stride overflows size_t.");
            if (group.width == 4u || group.width == 2u) {
                layout.packed_chunks.emplace_back(PackedChunk{
                    .capacity_stride = prefix_stride,
                    .width = group.width,
                    .fields = group.fields});
                for (uint8_t lane = 0u; lane < group.width; ++lane) {
                    auto field_index = group.fields[lane];
                    layout.field_capacity_strides[field_index] = prefix_stride;
                    layout.field_strides[field_index] = group.stride;
                    layout.field_offsets[field_index] = static_cast<size_t>(lane) * 4u;
                }
            } else {
                auto field_index = group.fields[0];
                layout.field_capacity_strides[field_index] = prefix_stride;
                layout.field_strides[field_index] = group.stride;
                layout.field_offsets[field_index] = 0u;
            }
            prefix_stride += group.stride;
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

inline void coro_frame_mark_raw_access(
    const CoroFrameStorageLayout &layout) noexcept {
    if (layout.size_bytes <= std::numeric_limits<uint>::max()) {
        luisa::compute::detail::comment(coro_frame_raw_access_comment);
    }
}

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

/// Scheduler-visible coroutine frame transport contract.
///
/// For continuation t and graph edge s -> t:
///
///   load[t]       = fields actually read by continuation t
///   store[s][t]   = fields defined on s -> t and live in t
///   relocate[t]   = complete token-indexed resident payload R[t]
///
/// Impossible graph edges have an empty store set. A scheduler that keeps its
/// routing token in frame storage may request field 6 in every realizable
/// output (including the implicit normal-return outcome zero) and queued
/// relocation payload;
/// schedulers with a separate token/index queue must not duplicate it.
struct CoroFrameIOPlanConfig {
    bool externalize_target_token{false};
};

struct CoroFrameIOPlan {
    luisa::vector<luisa::vector<size_t>> input_fields;
    luisa::vector<luisa::vector<luisa::vector<size_t>>>
        transition_output_fields;
    luisa::vector<luisa::vector<size_t>> relocation_fields;

    [[nodiscard]] auto input(size_t node) const noexcept {
        return luisa::span<const size_t>{input_fields[node]};
    }
    [[nodiscard]] auto output(size_t source, size_t target) const noexcept {
        return luisa::span<const size_t>{
            transition_output_fields[source][target]};
    }
    [[nodiscard]] auto relocation(size_t node) const noexcept {
        return luisa::span<const size_t>{relocation_fields[node]};
    }
};

[[nodiscard]] inline auto coro_frame_make_io_plan(
    const CoroGraph &graph, size_t frame_field_count,
    CoroFrameIOPlanConfig config = {}) noexcept {
    CoroFrameIOPlan plan;
    auto node_count = graph.node_count();
    plan.input_fields.resize(node_count);
    plan.transition_output_fields.resize(node_count);
    plan.relocation_fields = coro_frame_collect_relocation_fields(
        graph, frame_field_count);
    for (auto node = 0u; node < node_count; ++node) {
        plan.input_fields[node] = coro_frame_collect_input_fields(
            graph, node);
        plan.transition_output_fields[node].resize(node_count);
        for (auto field : plan.input_fields[node]) {
            LUISA_ASSERT(field < frame_field_count,
                         "Coroutine input field {} is outside frame field "
                         "count {}.",
                         field, frame_field_count);
        }
    }
    for (auto &&edge : graph.edges()) {
        LUISA_ASSERT(edge.from_index < node_count &&
                         edge.to_index < node_count,
                     "Coroutine edge {} -> {} is outside node count {}.",
                     edge.from_index, edge.to_index, node_count);
        auto &fields =
            plan.transition_output_fields[edge.from_index][edge.to_index];
        fields = edge.store_fields;
        for (auto field : fields) {
            LUISA_ASSERT(field < frame_field_count,
                         "Coroutine transition field {} is outside frame "
                         "field count {}.",
                         field, frame_field_count);
        }
        if (config.externalize_target_token) {
            coro_frame_append_unique(fields, 6u);
        }
        luisa::sort(fields.begin(), fields.end());
        fields.erase(std::unique(fields.begin(), fields.end()), fields.end());
    }
    if (config.externalize_target_token) {
        for (auto &&node : graph.nodes()) {
            // Graph edges model suspend-to-resume transitions. Normal return
            // is the scheduler's implicit outcome zero and is therefore not a
            // CoroGraph edge (nor equivalent to Scope::is_terminal, which
            // records an explicit CoroTerminate path). Every materialized
            // callable has this ABI outcome; the dynamic `next == 0` guard
            // ensures the token write executes only on that outcome.
            coro_frame_append_unique(
                plan.transition_output_fields[node.index][0u], 6u);
            if (node.index != graph.entry_index()) {
                coro_frame_append_unique(
                    plan.relocation_fields[node.index], 6u);
                luisa::sort(
                    plan.relocation_fields[node.index].begin(),
                    plan.relocation_fields[node.index].end());
            }
        }
    }
    return plan;
}

/// Factor token-indexed relocation payloads R[t] into
///
///   C = intersection(t != 0) R[t]
///   D[t] = R[t] - C.
///
/// A queued coroutine frame is a tagged union: for target token t, only R[t]
/// is defined. Schedulers may copy C unconditionally and dispatch only D[t]
/// on the token, which is exactly R[t] without reading inactive physical
/// slots. The factorization is a code-size optimization, not a liveness
/// approximation; CoroGraph remains the sole source of the relocation proof.
struct CoroFrameRelocationFieldPartition {
    luisa::vector<size_t> common_fields;
    luisa::vector<luisa::vector<size_t>> residual_fields;
};

[[nodiscard]] inline auto coro_frame_partition_relocation_fields(
    luisa::vector<luisa::vector<size_t>> exact_fields,
    const CoroFrameStorageLayout *layout = nullptr) noexcept {
    CoroFrameRelocationFieldPartition partition;
    partition.residual_fields = exact_fields;
    if (exact_fields.size() > 1u) {
        partition.common_fields = exact_fields[1u];
        for (auto token = 2u; token < exact_fields.size(); ++token) {
            auto &fields = exact_fields[token];
            partition.common_fields.erase(
                std::remove_if(
                    partition.common_fields.begin(),
                    partition.common_fields.end(),
                    [&](auto field) noexcept {
                        return std::find(
                                   fields.begin(), fields.end(), field) ==
                               fields.end();
                    }),
                partition.common_fields.end());
        }
        if (layout != nullptr && !layout->packed_chunks.empty()) {
            auto contains_field = [](const auto &vec, size_t f) noexcept {
                return std::find(vec.begin(), vec.end(), f) != vec.end();
            };
            for (const auto &chunk : layout->packed_chunks) {
                bool all_in_common = true;
                bool any_in_common = false;
                for (uint8_t lane = 0u; lane < chunk.width; ++lane) {
                    const bool in_c =
                        contains_field(partition.common_fields, chunk.fields[lane]);
                    all_in_common = all_in_common && in_c;
                    any_in_common = any_in_common || in_c;
                }
                if (any_in_common && !all_in_common) {
                    bool full_in_some_token = false;
                    for (auto token = 1u; token < exact_fields.size(); ++token) {
                        bool full = true;
                        for (uint8_t lane = 0u; lane < chunk.width; ++lane) {
                            if (!contains_field(exact_fields[token],
                                                chunk.fields[lane])) {
                                full = false;
                                break;
                            }
                        }
                        if (full) {
                            full_in_some_token = true;
                            break;
                        }
                    }
                    if (full_in_some_token) {
                        for (uint8_t lane = 0u; lane < chunk.width; ++lane) {
                            partition.common_fields.erase(
                                std::remove(partition.common_fields.begin(),
                                            partition.common_fields.end(),
                                            chunk.fields[lane]),
                                partition.common_fields.end());
                        }
                    }
                }
            }
        }
    }
    for (auto token = 1u;
         token < partition.residual_fields.size(); ++token) {
        auto &fields = partition.residual_fields[token];
        fields.erase(
            std::remove_if(
                fields.begin(), fields.end(),
                [&](auto field) noexcept {
                    return std::find(
                               partition.common_fields.begin(),
                               partition.common_fields.end(), field) !=
                           partition.common_fields.end();
                }),
            fields.end());
    }
    if (!partition.residual_fields.empty()) {
        partition.residual_fields.front().clear();
    }
    return partition;
}

[[nodiscard]] inline auto coro_frame_partition_relocation_fields(
    const CoroGraph &graph, size_t frame_field_count) noexcept {
    return coro_frame_partition_relocation_fields(
        coro_frame_collect_relocation_fields(graph, frame_field_count));
}

[[nodiscard]] inline auto coro_frame_partition_relocation_fields(
    const CoroFrameIOPlan &plan) noexcept {
    return coro_frame_partition_relocation_fields(
        plan.relocation_fields);
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
        luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
        bool include_reserved_fields = true) const noexcept {
        auto frame = CoroFrame::create(_desc);
        read_into(
            std::forward<I>(index), frame, active_fields,
            include_reserved_fields);
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
        auto offset = def(base + static_cast<uint>(layout.field_offsets[field_index]));
        coro_frame_mark_raw_access(layout);
        fb->call(is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_WRITE : CallOp::BYTE_BUFFER_WRITE,
                 {buffer.expression(),
                  luisa::compute::detail::extract_expression(offset),
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
        auto offset = def(base + static_cast<uint>(layout.field_offsets[field_index]));
        coro_frame_mark_raw_access(layout);
        auto *value = fb->call(type, is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_READ : CallOp::BYTE_BUFFER_READ,
                               {buffer.expression(),
                                luisa::compute::detail::extract_expression(offset)});
        auto *member = fb->member(type, frame.expression(), field_index);
        fb->assign(member, value);
    }
}

[[nodiscard]] inline auto coro_frame_load_aos(
    const CoroFrameDesc *desc, const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    auto frame = CoroFrame::create(desc);
    coro_frame_load_aos_into(
        frame, buffer, frame_index, layout, active_fields, is_volatile,
        include_reserved_fields);
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
        auto offset = def(static_cast<uint>(layout.field_offsets[field_index]) +
                          frame_index * static_cast<uint>(layout.field_strides[field_index]));
        auto *type = frame.desc()->frame_field_type(field_index);
        auto *member = fb->member(type, frame.expression(), field_index);
        coro_frame_mark_raw_access(layout);
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
        auto offset = def(static_cast<uint>(layout.field_offsets[field_index]) +
                          frame_index * static_cast<uint>(layout.field_strides[field_index]));
        auto *type = frame.desc()->frame_field_type(field_index);
        coro_frame_mark_raw_access(layout);
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
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    auto frame = CoroFrame::create(desc);
    coro_frame_load_soa_into(
        frame, buffer, frame_index, layout, active_fields, is_volatile,
        include_reserved_fields);
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

/// Copy exactly the fields proved live for one tagged coroutine state.
/// Source and destination frame indices must be distinct, and concurrent
/// invocations must own disjoint destinations. Copying through a complete
/// CoroFrame aggregate is intentionally avoided: fields outside the token's
/// relocation set are undefined and therefore may not be loaded merely to
/// simplify storage movement.
inline void coro_frame_copy_fields(
    const Expr<ByteBuffer> &buffer, Expr<uint> source_index,
    Expr<uint> destination_index, Expr<uint> soa_capacity,
    const CoroFrameDesc *desc, const CoroFrameStorageLayout &layout,
    bool soa, luisa::span<const size_t> fields,
    bool is_volatile = false) noexcept {
    LUISA_ASSERT(desc != nullptr, "Coroutine frame descriptor is null.");
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    luisa::vector<uint8_t> field_remaining;
    if (soa && layout.has_runtime_capacity() && !layout.packed_chunks.empty()) {
        field_remaining.assign(desc->frame_field_count(), 0u);
        for (auto field_index : fields) {
            LUISA_ASSERT(
                field_index < desc->frame_field_count(),
                "Coroutine relocation field {} is outside frame field count {}.",
                field_index, desc->frame_field_count());
            field_remaining[field_index] = 1u;
        }
        for (const auto &chunk : layout.packed_chunks) {
            auto all_active = true;
            for (uint8_t lane = 0u; lane < chunk.width; ++lane) {
                if (field_remaining[chunk.fields[lane]] == 0u) {
                    all_active = false;
                    break;
                }
            }
            if (!all_active) { continue; }
            auto *vec_type = chunk.width == 4u ? Type::of<uint4>() : Type::of<uint2>();
            auto chunk_stride = static_cast<uint>(chunk.width) * 4u;
            auto base_offset = soa_capacity * static_cast<uint>(chunk.capacity_stride);
            auto source_byte_offset = def(base_offset + source_index * chunk_stride);
            auto destination_byte_offset = def(base_offset + destination_index * chunk_stride);
            coro_frame_mark_raw_access(layout);
            auto *value = fb->call(
                vec_type,
                is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_READ :
                              CallOp::BYTE_BUFFER_READ,
                {buffer.expression(),
                 luisa::compute::detail::extract_expression(source_byte_offset)});
            coro_frame_mark_raw_access(layout);
            fb->call(
                is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_WRITE :
                              CallOp::BYTE_BUFFER_WRITE,
                {buffer.expression(),
                 luisa::compute::detail::extract_expression(destination_byte_offset),
                 value});
            for (uint8_t lane = 0u; lane < chunk.width; ++lane) {
                field_remaining[chunk.fields[lane]] = 0u;
            }
        }
    }
    for (auto field_index : fields) {
        LUISA_ASSERT(
            field_index < desc->frame_field_count(),
            "Coroutine relocation field {} is outside frame field count {}.",
            field_index, desc->frame_field_count());
        if (!field_remaining.empty() && field_remaining[field_index] == 0u) {
            continue;
        }
        auto *type = desc->frame_field_type(field_index);
        auto copy_at_offsets = [&](auto source_offset,
                                   auto destination_offset) noexcept {
            auto source_byte_offset = def(source_offset);
            auto destination_byte_offset = def(destination_offset);
            coro_frame_mark_raw_access(layout);
            auto *value = fb->call(
                type,
                is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_READ :
                              CallOp::BYTE_BUFFER_READ,
                {buffer.expression(),
                 luisa::compute::detail::extract_expression(source_byte_offset)});
            coro_frame_mark_raw_access(layout);
            fb->call(
                is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_WRITE :
                              CallOp::BYTE_BUFFER_WRITE,
                {buffer.expression(),
                 luisa::compute::detail::extract_expression(
                     destination_byte_offset),
                 value});
        };
        if (soa) {
            auto source_offset = coro_frame_runtime_soa_offset(
                source_index, soa_capacity, layout, field_index);
            auto destination_offset = coro_frame_runtime_soa_offset(
                destination_index, soa_capacity, layout, field_index);
            copy_at_offsets(source_offset, destination_offset);
        } else {
            auto stride = static_cast<uint>(layout.frame_stride);
            auto field_offset =
                static_cast<uint>(layout.field_offsets[field_index]);
            auto source_offset = source_index * stride + field_offset;
            auto destination_offset =
                destination_index * stride + field_offset;
            copy_at_offsets(source_offset, destination_offset);
        }
    }
}

inline void coro_frame_store_runtime_soa(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    Expr<uint> capacity, const CoroFrame &frame,
    const CoroFrameStorageLayout &layout,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    auto field_count = frame.desc()->frame_field_count();
    luisa::vector<uint8_t> field_active(field_count, 0u);
    for (auto field_index = 0u; field_index < field_count; ++field_index) {
        if (coro_frame_is_active_field(
                field_index, active_fields, include_reserved_fields)) {
            field_active[field_index] = 1u;
        }
    }
    if (!layout.packed_chunks.empty()) {
        auto *uint_type = Type::of<uint>();
        for (const auto &chunk : layout.packed_chunks) {
            auto all_active = true;
            for (uint8_t lane = 0u; lane < chunk.width; ++lane) {
                if (field_active[chunk.fields[lane]] == 0u) {
                    all_active = false;
                    break;
                }
            }
            if (!all_active) { continue; }
            std::array<const Expression *, 4u> words{};
            for (uint8_t lane = 0u; lane < chunk.width; ++lane) {
                auto field_index = chunk.fields[lane];
                auto *type = frame.desc()->frame_field_type(field_index);
                const Expression *member =
                    fb->member(type, frame.expression(), field_index);
                words[lane] =
                    type == uint_type ?
                        member :
                        static_cast<const Expression *>(
                            fb->cast(uint_type, CastOp::BITWISE, member));
                field_active[field_index] = 0u;
            }
            auto *vec_type = chunk.width == 4u ? Type::of<uint4>() : Type::of<uint2>();
            auto *vec = fb->call(
                vec_type,
                chunk.width == 4u ? CallOp::MAKE_UINT4 : CallOp::MAKE_UINT2,
                luisa::span<const Expression *const>{words.data(), chunk.width});
            auto offset = def(capacity * static_cast<uint>(chunk.capacity_stride) +
                              frame_index * (static_cast<uint>(chunk.width) * 4u));
            coro_frame_mark_raw_access(layout);
            fb->call(is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_WRITE :
                                   CallOp::BYTE_BUFFER_WRITE,
                     {buffer.expression(),
                      luisa::compute::detail::extract_expression(offset), vec});
        }
    }
    for (auto field_index = 0u; field_index < field_count; field_index++) {
        if (field_active[field_index] == 0u) { continue; }
        auto offset = def(coro_frame_runtime_soa_offset(
            frame_index, capacity, layout, field_index));
        auto *type = frame.desc()->frame_field_type(field_index);
        auto *member = fb->member(type, frame.expression(), field_index);
        coro_frame_mark_raw_access(layout);
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
    auto field_count = frame.desc()->frame_field_count();
    luisa::vector<uint8_t> field_active(field_count, 0u);
    for (auto field_index = 0u; field_index < field_count; ++field_index) {
        if (coro_frame_is_active_field(
                field_index, active_fields, include_reserved_fields)) {
            field_active[field_index] = 1u;
        }
    }
    if (!layout.packed_chunks.empty()) {
        auto *uint_type = Type::of<uint>();
        for (const auto &chunk : layout.packed_chunks) {
            auto all_active = true;
            for (uint8_t lane = 0u; lane < chunk.width; ++lane) {
                if (field_active[chunk.fields[lane]] == 0u) {
                    all_active = false;
                    break;
                }
            }
            if (!all_active) { continue; }
            auto *vec_type = chunk.width == 4u ? Type::of<uint4>() : Type::of<uint2>();
            auto offset = def(capacity * static_cast<uint>(chunk.capacity_stride) +
                              frame_index * (static_cast<uint>(chunk.width) * 4u));
            coro_frame_mark_raw_access(layout);
            auto *vec_call = fb->call(
                vec_type,
                is_volatile ? CallOp::BYTE_BUFFER_VOLATILE_READ :
                              CallOp::BYTE_BUFFER_READ,
                {buffer.expression(),
                 luisa::compute::detail::extract_expression(offset)});
            auto *vec_local = fb->local(vec_type);
            fb->assign(vec_local, vec_call);
            for (uint8_t lane = 0u; lane < chunk.width; ++lane) {
                auto field_index = chunk.fields[lane];
                auto *type = frame.desc()->frame_field_type(field_index);
                const Expression *lane_expr =
                    fb->swizzle(uint_type, vec_local, 1u, lane);
                const Expression *value =
                    type == uint_type ?
                        lane_expr :
                        static_cast<const Expression *>(
                            fb->cast(type, CastOp::BITWISE, lane_expr));
                auto *member = fb->member(type, frame.expression(), field_index);
                fb->assign(member, value);
                field_active[field_index] = 0u;
            }
        }
    }
    for (auto field_index = 0u; field_index < field_count; field_index++) {
        if (field_active[field_index] == 0u) { continue; }
        auto offset = def(coro_frame_runtime_soa_offset(
            frame_index, capacity, layout, field_index));
        auto *type = frame.desc()->frame_field_type(field_index);
        coro_frame_mark_raw_access(layout);
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
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    auto frame = CoroFrame::create(desc);
    coro_frame_load_runtime_soa_into(
        frame, buffer, frame_index, capacity,
        layout, active_fields, is_volatile,
        include_reserved_fields);
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
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    return soa ?
               coro_frame_load_soa(desc, buffer, frame_index, layout,
                                   active_fields, is_volatile,
                                   include_reserved_fields) :
               coro_frame_load_aos(desc, buffer, frame_index, layout,
                                   active_fields, is_volatile,
                                   include_reserved_fields);
}

[[nodiscard]] inline auto coro_frame_load(
    const CoroFrameDesc *desc, const Expr<ByteBuffer> &buffer,
    Expr<uint> frame_index, Expr<uint> soa_capacity,
    const CoroFrameStorageLayout &layout, bool soa,
    luisa::optional<luisa::span<const size_t>> active_fields = luisa::nullopt,
    bool is_volatile = false,
    bool include_reserved_fields = true) noexcept {
    return soa ?
               coro_frame_load_runtime_soa(
                   desc, buffer, frame_index, soa_capacity,
                   layout, active_fields, is_volatile,
                   include_reserved_fields) :
               coro_frame_load_aos(
                   desc, buffer, frame_index,
                   layout, active_fields, is_volatile,
                   include_reserved_fields);
}

template<typename T>
[[nodiscard]] inline auto coro_frame_read_field(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout, bool soa, size_t field_index) noexcept {
    auto offset = def(static_cast<uint>(layout.field_offsets[field_index]));
    offset += frame_index * static_cast<uint>(soa ? layout.field_strides[field_index] : layout.frame_stride);
    coro_frame_mark_raw_access(layout);
    return buffer.template read<T>(offset);
}

template<typename V>
inline void coro_frame_write_field(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    const CoroFrameStorageLayout &layout, bool soa, size_t field_index, V &&value) noexcept {
    auto offset = def(static_cast<uint>(layout.field_offsets[field_index]));
    offset += frame_index * static_cast<uint>(soa ? layout.field_strides[field_index] : layout.frame_stride);
    coro_frame_mark_raw_access(layout);
    buffer.write(offset, std::forward<V>(value));
}

template<typename T>
[[nodiscard]] inline auto coro_frame_read_field(
    const Expr<ByteBuffer> &buffer, Expr<uint> frame_index,
    Expr<uint> soa_capacity, const CoroFrameStorageLayout &layout,
    bool soa, size_t field_index) noexcept {
    if (soa) {
        auto offset = def(coro_frame_runtime_soa_offset(
            frame_index, soa_capacity, layout, field_index));
        coro_frame_mark_raw_access(layout);
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
        auto offset = def(coro_frame_runtime_soa_offset(
            frame_index, soa_capacity, layout, field_index));
        coro_frame_mark_raw_access(layout);
        buffer.write(offset, std::forward<V>(value));
    } else {
        coro_frame_write_field(
            buffer, frame_index, layout, false,
            field_index, std::forward<V>(value));
    }
}

}// namespace luisa::compute::coro
