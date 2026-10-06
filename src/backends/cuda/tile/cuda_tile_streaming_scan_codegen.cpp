// Emit a separate streaming entry; never rewrite the general emitter source.
#include "cuda_tile_streaming_scan.h"
#include <luisa/core/stl/format.h>

namespace luisa::compute::cuda::native_tile {
namespace {
[[nodiscard]] luisa::string_view scalar_type_name(tile::ScalarType type) noexcept {
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
}// namespace

luisa::string emit_streaming_scan_entry(const StreamingScanPlan &p, const Artifact &original,
                                        uint32_t worker_warps, uint32_t target_sm) noexcept {
    if (!p.ok() || !original.ok() || (worker_warps != 0u && ((worker_warps != 4u && worker_warps != 8u) || target_sm != 89u))) { return {}; }
    auto source = worker_warps == 0u ? luisa::string{"\nextern \"C\" __tile_global__ void luisa_tile_stream_scan("} :
                                       luisa::format("\nextern \"C\" {{\n[[cutile::hint({}, num_worker_warps_per_cta = {})]]\n__tile_global__ void luisa_tile_stream_scan(", target_sm * 10u, worker_warps);
    for (auto i = size_t{0u}; i < original.arguments.size(); i++) {
        auto type = scalar_type_name(original.arguments[i].element);
        if (type.empty()) { return {}; }
        if (i != 0u) { source += ", "; }
        source += luisa::format("{} *buffer{}", type, i);
    }
    auto line = [&](luisa::string_view value) { source.append(value.data(), value.size()); source += '\n'; };
    auto fully_in_bounds = p.rows % p.rows_per_program == 0u && p.columns == p.padded_columns;
    line(") {");
    line(luisa::format("    using Indices = ct::tile<long long, ct::shape<{}, {}>>;", p.rows_per_program, p.chunk_extent));
    line(luisa::format("    using Carry = ct::tile<float, ct::shape<{}, 1>>;", p.rows_per_program));
    line("    auto lane = ct::iota<Indices>();");
    if (!fully_in_bounds) { line("    auto index_zero = ct::full<Indices>(0ll);"); }
    line(luisa::format("    auto row = ct::element_cast<long long>(ct::bid().x) * {}ll + lane / {}ll;", p.rows_per_program, p.chunk_extent));
    line(luisa::format("    auto lane_column = lane % {}ll;", p.chunk_extent));
    line("    auto carry = ct::full<Carry>(0.0f);");
    line(luisa::format("    for (long long base = 0ll; base < {}ll; base += {}ll) {{", p.padded_columns, p.chunk_extent));
    line("        auto column = base + lane_column;");
    if (fully_in_bounds) {
        line(luisa::format("        auto offset = row * {}ll + column;", p.columns));
        line(luisa::format("        auto loaded = ct::load(buffer{} + offset);", p.input_slot));
    } else {
        line(luisa::format("        auto valid = (row < {}ll) && (column < {}ll);", p.rows, p.columns));
        line(luisa::format("        auto offset = ct::select(valid, row * {}ll + column, index_zero);", p.columns));
        line(luisa::format("        auto loaded = ct::load_masked(buffer{} + offset, valid, ct::element_cast<{}>(0));", p.input_slot, scalar_type_name(p.storage)));
    }
    line("        auto input = ct::element_cast<float>(loaded);");
    line("        auto cumulative = ct::partial_sum(input, ct::integral_constant<1>{}, ct::round_ties_to_even_t{}, ct::preserve_subnormals_t{}, ct::scan_forward_t{});");
    line("        if (base != 0ll) {");
    line("            cumulative = ct::add(carry, cumulative, ct::round_ties_to_even_t{}, ct::preserve_subnormals_t{});");
    line("        }");
    line("        auto with_seed = ct::add(0.0f, cumulative, ct::round_ties_to_even_t{}, ct::preserve_subnormals_t{});");
    line(luisa::format("        auto stored = ct::element_cast<{}>(with_seed);", scalar_type_name(p.storage)));
    line(fully_in_bounds ? luisa::format("        ct::store(buffer{} + offset, stored);", p.output_slot) :
                           luisa::format("        ct::store_masked(buffer{} + offset, stored, valid);", p.output_slot));
    line(luisa::format("        carry = ct::extract(cumulative, ct::shape<{}, 1>{{}}, 0ull, {}ull);", p.rows_per_program, p.chunk_extent - 1u));
    line("    }");
    line("}");
    if (worker_warps != 0u) { line("}"); }
    return source;
}
void append_streaming_scan(Artifact &original, const tile::Function &function,
                           uint32_t chunk, uint32_t worker_warps, uint32_t target_sm) noexcept {
    if (chunk == 0u || !original.ok()) { return; }
    original.streaming_scan_chunk_extent = chunk;
    auto plan = match_streaming_scan(function, original, chunk);
    if (!plan.ok()) {
        original.streaming_scan_diagnostic = std::move(plan.error);
        return;
    }
    auto source = emit_streaming_scan_entry(plan, original, worker_warps, target_sm);
    if (source.empty()) {
        original.streaming_scan_diagnostic = "streaming source generation rejected its private plan";
        return;
    }
    original.streaming_scan_source_offset = original.source.size();
    original.streaming_scan_guard = StreamingScanGuard{
        plan.input_slot, plan.output_slot,
        original.arguments[plan.input_slot].minimum_size_bytes,
        original.arguments[plan.output_slot].minimum_size_bytes};
    original.streaming_scan_entry = "luisa_tile_stream_scan";
    original.source += source;
}
}// namespace luisa::compute::cuda::native_tile
