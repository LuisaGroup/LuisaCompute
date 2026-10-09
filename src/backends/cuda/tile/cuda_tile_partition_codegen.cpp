#include "cuda_tile_partition_codegen.h"
#include <bit>
#include <limits>
#include <luisa/core/stl/format.h>

namespace luisa::compute::cuda::native_tile {
namespace {

namespace cuda_tile_partition_codegen_detail {
[[nodiscard]] luisa::string_view scalar(tile::ScalarType type) noexcept {
    using tile::ScalarType;
    switch (type) {
        case ScalarType::BOOL: return "bool";
        case ScalarType::INT32: return "int";
        case ScalarType::UINT32: return "unsigned";
        case ScalarType::INT64: return "long long";
        case ScalarType::UINT64: return "unsigned long long";
        case ScalarType::FLOAT16: return "__half";
        case ScalarType::BFLOAT16: return "__nv_bfloat16";
        case ScalarType::FLOAT32: return "float";
        default: return {};
    }
}

[[nodiscard]] luisa::string emit(const tile::IndependentCollectivePlan &p, const Artifact &original) noexcept {
    auto source = luisa::string{"\nextern \"C\" __tile_global__ void luisa_tile_partition("};
    for (auto i = size_t{0u}; i < original.arguments.size(); i++) {
        auto type = scalar(original.arguments[i].element);
        if (type.empty()) { return {}; }
        if (i != 0u) { source += ", "; }
        source += luisa::format("{} *buffer{}", type, i);
    }
    auto line = [&](luisa::string_view text) { source.append(text.data(), text.size()); source += '\n'; };
    auto rows = p.candidate.independent_extent_per_program;
    auto width = p.tile_contribution_extent;
    auto full_rows = p.candidate.tail_valid_extent == 0u;
    auto full_input = full_rows && p.logical_contribution_extent == width;
    auto input_slot = p.disjoint.input.argument_index, output_slot = p.disjoint.output.argument_index;
    line(") {");
    line(luisa::format("    using InputIndices = ct::tile<long long, ct::shape<{}, {}>>;", rows, width));
    line(luisa::format("    using OutputIndices = ct::tile<long long, ct::shape<{}, 1>>;", rows));
    line("    auto lane = ct::iota<InputIndices>();");
    line(luisa::format("    auto row_base = ct::element_cast<long long>(ct::bid().x) * {}ll;", rows));
    line(luisa::format("    auto row = row_base + lane / {}ll;", width));
    line(luisa::format("    auto column = lane % {}ll;", width));
    if (full_input) {
        line(luisa::format("    auto offset = row * {}ll + column;", p.logical_contribution_extent));
        line(luisa::format("    auto loaded = ct::load(buffer{} + offset);", input_slot));
    } else {
        line(luisa::format("    auto valid = (row < {}ll) && (column < {}ll);", p.logical_independent_extent, p.logical_contribution_extent));
        line(luisa::format("    auto offset = ct::select(valid, row * {}ll + column, ct::full<InputIndices>(0ll));", p.logical_contribution_extent));
        line(luisa::format("    auto loaded = ct::load_masked(buffer{} + offset, valid, ct::element_cast<{}>(0));", input_slot, scalar(p.input_storage)));
    }
    line("    auto input = ct::element_cast<float>(loaded);");
    if (p.kind == tile::CollectiveKind::MAXIMUM) {
        line("    auto identity = ct::element_bitcast<float>(0xff800000u);");
    }
    if (p.contribution_identity_mask) {
        auto identity = p.kind == tile::CollectiveKind::SUM ? "0.0f" : "identity";
        line(luisa::format("    input = ct::select(column < {}ll, input, ct::full<decltype(input)>({}));", p.logical_contribution_extent, identity));
    }
    if (p.kind == tile::CollectiveKind::SUM) {
        line("    auto reduced = ct::sum(input, ct::integral_constant<1>{}, ct::round_ties_to_even_t{}, ct::preserve_subnormals_t{});");
        line("    auto with_seed = ct::add(0.0f, reduced, ct::round_ties_to_even_t{}, ct::preserve_subnormals_t{});");
    } else {
        line("    auto reduced = ct::reduce_max(input, ct::integral_constant<1>{}, ct::suppress_nan_t{}, ct::preserve_subnormals_t{});");
        line("    auto with_seed = ct::max(identity, reduced, ct::suppress_nan_t{}, ct::preserve_subnormals_t{});");
    }
    line(luisa::format("    auto stored = ct::element_cast<{}>(with_seed);", scalar(p.output_storage)));
    line("    auto output_row = row_base + ct::iota<OutputIndices>();");
    if (full_rows) {
        line(luisa::format("    ct::store(buffer{} + output_row, stored);", output_slot));
    } else {
        line(luisa::format("    auto output_valid = output_row < {}ll;", p.logical_independent_extent));
        line("    auto output_offset = ct::select(output_valid, output_row, ct::full<OutputIndices>(0ll));");
        line(luisa::format("    ct::store_masked(buffer{} + output_offset, stored, output_valid);", output_slot));
    }
    line("}");
    return source;
}

}  // namespace cuda_tile_partition_codegen_detail
}// namespace

void append_program_partition(Artifact &original, const tile::Function &function, uint32_t target_rows) noexcept {
    if (target_rows == 0u || !original.ok()) { return; }
    original.partition_rows = target_rows;
    auto fail = [&](luisa::string_view reason) { original.partition_diagnostic = reason; };
    if ((target_rows != 1u && target_rows != 2u && target_rows != 4u) ||
        !original.partition_entry.empty() || !original.streaming_scan_entry.empty() || !original.aligned16_entry.empty() ||
        original.scan_chunk_extent != 0u || original.independent_axis_extent != 0u) {
        return fail("program partition requires target rows 1/2/4 and an untransformed original artifact");
    }
    auto p = tile::plan_independent_collective(function, {.target_extent_per_program = target_rows});
    if (!p.ok()) { return fail(p.error); }
    auto storage = p.input_storage;
    if ((storage != tile::ScalarType::FLOAT32 && storage != tile::ScalarType::FLOAT16 && storage != tile::ScalarType::BFLOAT16) ||
        p.output_storage != storage || (p.kind != tile::CollectiveKind::SUM && p.kind != tile::CollectiveKind::MAXIMUM) ||
        p.input_independent_axis != 0u || p.input_contribution_axis != 1u ||
        p.output_independent_axis != 0u || (p.output_rank != 1u && p.output_rank != 2u)) {
        return fail("CUDA program partition requires matching floating storage and contiguous last-axis SUM/MAXIMUM");
    }
    auto width = p.tile_contribution_extent;
    auto old_rows = p.original.independent_extent_per_program;
    constexpr auto kGridMax = uint64_t{0x7fffffffu};
    if ((old_rows != 4u && old_rows != 8u) || target_rows >= old_rows || old_rows % target_rows != 0u ||
        p.candidate.independent_extent_per_program != target_rows ||
        p.logical_contribution_extent == 0u || p.logical_contribution_extent > 65536u ||
        width != std::bit_ceil(p.logical_contribution_extent) ||
        p.original.programs == 0u || p.original.programs > kGridMax || p.candidate.programs == 0u || p.candidate.programs > kGridMax ||
        original.grid != std::array<uint32_t, 3u>{static_cast<uint32_t>(p.original.programs), 1u, 1u} ||
        original.block != std::array<uint32_t, 3u>{1u, 1u, 1u}) {
        return fail("CUDA program partition geometry is outside its bounded static subset");
    }
    auto input = p.disjoint.input, output = p.disjoint.output;
    if (input.argument_index >= original.arguments.size() || output.argument_index >= original.arguments.size() ||
        input.argument_index == output.argument_index || input.byte_offset != 0u || output.byte_offset != 0u ||
        input.byte_count == 0u || output.byte_count == 0u ||
        input.byte_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        output.byte_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return fail("CUDA program partition requires bounded complete root-view intervals");
    }
    for (auto i = size_t{0u}; i < original.arguments.size(); i++) {
        auto &&argument = original.arguments[i];
        if (i == input.argument_index) {
            if (argument.element != storage || argument.minimum_size_bytes != input.byte_count || !argument.read || argument.written) {
                return fail("program partition input metadata disagrees with its original artifact");
            }
        } else if (i == output.argument_index) {
            if (argument.element != storage || argument.minimum_size_bytes != output.byte_count || argument.read || !argument.written) {
                return fail("program partition output metadata disagrees with its original artifact");
            }
        } else if (argument.read || argument.written) {
            return fail("program partition cannot discard another live root resource");
        }
    }
    auto source = cuda_tile_partition_codegen_detail::emit(p, original);
    if (source.empty()) { return fail("program partition source generation rejected its argument types"); }
    original.partition_source_offset = original.source.size();
    original.partition_original_rows = static_cast<uint32_t>(old_rows);
    original.partition_grid = {static_cast<uint32_t>(p.candidate.programs), 1u, 1u};
    original.partition_guard = {input.argument_index, output.argument_index, input.byte_count, output.byte_count};
    original.partition_entry = "luisa_tile_partition";
    original.source += source;
}

}// namespace luisa::compute::cuda::native_tile
