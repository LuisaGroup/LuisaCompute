#pragma once

#include <array>
#include <cmath>
#include <limits>
#include "cuda_tile_cub_scan.h"

namespace luisa::compute::cuda::native_tile {

// Frozen before independent validation; the score is not a confidence bound.
inline constexpr luisa::string_view kScanCostProfile = "sm89-24-cuda134-scan-nnls-v1";
inline constexpr luisa::string_view kScanCostFit = "67127e819f80a395aeecae55cd999c8e3f8b1bcf894fdf0672c58611cca87f66";
inline constexpr luisa::string_view kScanCostProfileFileSha256 = "d38f317cb43b1e646fcdb959e30ea89675b79722508d973ee037609db24d9186";
inline constexpr std::array kScanCostThreads{128u, 256u, 512u, 1024u};
inline constexpr std::array kScanCostTileCoefficients{0.7284119403590213, 0.0, 0.00026641302734655293};
inline constexpr std::array kScanCostCubCoefficients{0.6027021529393383, 12.227593513162688, 0.027854484240835357, 0.022923736019806327};
inline constexpr double kScanCostImprovementRatio = 0.95;

struct ScanCostDevice {
    bool query_ok{false};
    uint32_t compute_capability{0u};
    uint32_t processors{0u};
    uint32_t subgroup_width{0u};
    uint32_t resident_threads{0u};
    uint32_t driver_api_version{0u};
    uint32_t toolkit_version{0u};
    // CUDACompiler::nvrtc_version(): major * 10000 + minor * 100.
    uint32_t nvrtc_version{0u};
};

// These are private observations of one live, entry-resolved ordinary CUDA
// candidate. No CUmodule ownership or Driver calls are introduced here.
// The factory must keep the queried function alive until the decision is used.
// Temporary owner => loaded-candidate; final transfer => installed-entry.
// Never relabel an attempted/unloaded query as an installed function receipt.
struct CompiledScanResources {
    bool loaded_entry_verified{false};
    bool function_query_ok{false};
    bool capacity_query_ok{false};
    int32_t registers{-1};
    int32_t static_shared_bytes{-1};
    int32_t local_bytes{-1};
    int32_t maximum_threads{-1};
    int32_t resident_cta_capacity{-1};
    uint32_t capacity_threads{0u};
    uint64_t capacity_dynamic_shared_bytes{0u};
};

struct CompiledScanCandidate {
    const CubScanArtifact *artifact{nullptr};
    CompiledScanResources resources;
    bool compile_key_known{false};
    // A valid key may be zero; unknown is represented separately above.
    uint64_t compile_key{0u};
};

struct ScanCostCandidateScore {
    uint32_t threads{0u};
    uint64_t compile_key{0u};
    bool compile_key_known{false};
    bool has_score{false};
    std::array<double, 4u> features{};
    double score{0.0};
    luisa::string_view reason{"candidate-unavailable"};
};

struct ScanCostChoice {
    uint32_t selected_threads{0u};
    bool has_score{false};
    std::array<double, 3u> original_features{};
    double original_score{0.0};
    double selected_score{0.0};
    std::array<ScanCostCandidateScore, 4u> candidates{};
    luisa::string_view status{"ineligible"};
    luisa::string_view reason{"device-query"};
};

namespace scan_cost_detail {

[[nodiscard]] inline bool multiply(uint64_t a, uint64_t b, uint64_t &result) noexcept {
    if (a == 0u || b == 0u || a > std::numeric_limits<uint64_t>::max() / b) { return false; }
    result = a * b;
    return true;
}

[[nodiscard]] inline bool add(uint64_t a, uint64_t b, uint64_t &result) noexcept {
    if (a > std::numeric_limits<uint64_t>::max() - b) { return false; }
    result = a + b;
    return true;
}

[[nodiscard]] inline bool ceiling(uint64_t numerator, uint64_t denominator, uint64_t &result) noexcept {
    if (numerator == 0u || denominator == 0u) { return false; }
    // Quotient plus remainder cannot overflow for positive uint64 operands.
    result = numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0u);
    return true;
}

template<size_t N>
[[nodiscard]] inline bool score(const std::array<double, N> &coefficients,
                                const std::array<double, N> &features, double &result) noexcept {
    auto total = 0.0;
    for (auto i = size_t{0u}; i < N; i++) {
        if (!std::isfinite(coefficients[i]) || coefficients[i] < 0.0 ||
            !std::isfinite(features[i]) || features[i] < 0.0) { return false; }
        total += coefficients[i] * features[i];
    }
    if (!std::isfinite(total) || total <= 0.0) { return false; }
    result = total;
    return true;
}

[[nodiscard]] inline bool supports_device(const ScanCostDevice &device) noexcept {
    return device.query_ok && device.compute_capability == 89u && device.processors == 24u &&
           device.subgroup_width == 32u && device.resident_threads == 1536u &&
           device.driver_api_version == 13040u && device.toolkit_version == 13040u &&
           device.nvrtc_version == 130400u;
}

[[nodiscard]] inline bool plain_original(const Artifact &original) noexcept {
    return original.ok() && original.scan_chunk_extent == 0u && original.chunked_scan_operations == 0u &&
           original.independent_axis_extent == 0u && original.partitioned_collective_operations == 0u &&
           original.streaming_scan_chunk_extent == 0u && original.streaming_scan_entry.empty() &&
           original.partition_rows == 0u && original.partition_entry.empty() &&
           original.aligned16_buffer_mask == 0u && original.aligned16_partition_loads == 0u && original.aligned16_entry.empty();
}

[[nodiscard]] inline bool original_facts(const tile::ClosedPrefixAnalysis &proof, const Artifact &original,
                                         uint64_t &bytes_per_buffer) noexcept {
    using tile::ScalarType;
    if (!proof.ok() || !plain_original(original) ||
        (proof.storage != ScalarType::FLOAT16 && proof.storage != ScalarType::BFLOAT16) ||
        proof.collective.kind != tile::CollectiveKind::INCLUSIVE_SUM || proof.collective.element != ScalarType::FLOAT32 ||
        proof.original.independent_extent_per_program != 1u || proof.collective.independent_elements != 1u ||
        proof.original.tail_valid_extent != 0u || proof.logical_independent_extent == 0u || proof.logical_contribution_extent == 0u ||
        proof.original.programs != proof.logical_independent_extent || proof.original.full_programs != proof.logical_independent_extent ||
        proof.collective.contribution_extent != proof.logical_contribution_extent ||
        proof.collective.input_elements != proof.logical_contribution_extent) { return false; }
    constexpr auto launch_limit = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    auto rows = proof.logical_independent_extent;
    auto width = proof.logical_contribution_extent;
    if (rows > launch_limit || width > launch_limit ||
        original.grid != std::array<uint32_t, 3u>{static_cast<uint32_t>(rows), 1u, 1u} ||
        original.block != std::array<uint32_t, 3u>{1u, 1u, 1u}) { return false; }
    // block1 is checked only as the original Tile ABI, never a physical count.
    auto input = proof.disjoint.input;
    auto output = proof.disjoint.output;
    uint64_t elements{};
    if (!multiply(rows, width, elements) || !multiply(elements, 2u, bytes_per_buffer) ||
        input.argument_index >= original.arguments.size() || output.argument_index >= original.arguments.size() ||
        input.argument_index == output.argument_index || original.arguments.size() > 31u ||
        input.byte_offset != 0u || output.byte_offset != 0u ||
        input.byte_count != bytes_per_buffer || output.byte_count != bytes_per_buffer) { return false; }
    auto root = proof.function->body().block(0u);
    if (root->argument_count() != original.arguments.size()) { return false; }
    for (auto slot = size_t{0u}; slot < original.arguments.size(); slot++) {
        auto &argument = original.arguments[slot];
        auto &type = root->argument(slot)->type();
        if (!type.is_view() || type.scalar_type() != argument.element ||
            argument.read != (slot == input.argument_index) || argument.written != (slot == output.argument_index)) { return false; }
        if ((slot == input.argument_index || slot == output.argument_index) &&
            (argument.element != proof.storage || argument.minimum_size_bytes != bytes_per_buffer)) { return false; }
    }
    return true;
}

[[nodiscard]] inline bool memory_feature(uint64_t bytes_per_buffer, const ScanCostDevice &device, double &result) noexcept {
    uint64_t total{}, divisor{};
    if (!add(bytes_per_buffer, bytes_per_buffer, total) || !multiply(device.processors, uint64_t{1u} << 20u, divisor)) { return false; }
    result = static_cast<double>(total) / static_cast<double>(divisor);
    return std::isfinite(result) && result > 0.0;
}

[[nodiscard]] inline bool tile_features(uint64_t rows, uint64_t width, uint64_t bytes_per_buffer,
                                        const ScanCostDevice &device, std::array<double, 3u> &features) noexcept {
    uint64_t batches{}, logical_demand{};
    double memory{};
    if (!ceiling(rows, device.processors, batches) || !multiply(batches, width, logical_demand) ||
        !memory_feature(bytes_per_buffer, device, memory)) { return false; }
    features = {1.0, memory, static_cast<double>(logical_demand)};
    return std::isfinite(features[2u]);
}

[[nodiscard]] inline bool cub_features(uint64_t rows, uint64_t width, uint64_t bytes_per_buffer,
                                       uint32_t threads, uint64_t capacity, const ScanCostDevice &device,
                                       std::array<double, 4u> &features) noexcept {
    uint64_t chunk{}, capacity_total{}, batches{}, warps{}, scaled_chunks{}, local{}, group{};
    double memory{};
    if (!multiply(threads, 8u, chunk) || width == 0u || width % chunk != 0u ||
        !multiply(device.processors, capacity, capacity_total) || !ceiling(rows, capacity_total, batches) ||
        !ceiling(threads, device.subgroup_width, warps) || !multiply(batches, width / chunk, scaled_chunks) ||
        !multiply(scaled_chunks, 8u, local) || !multiply(scaled_chunks, warps, group) ||
        !memory_feature(bytes_per_buffer, device, memory)) { return false; }
    features = {1.0, memory, static_cast<double>(local), static_cast<double>(group)};
    return std::isfinite(features[2u]) && std::isfinite(features[3u]);
}

[[nodiscard]] inline luisa::string_view resource_rejection(const CompiledScanResources &resource,
                                                           uint32_t threads) noexcept {
    if (!resource.loaded_entry_verified) { return "entry-unavailable"; }
    if (!resource.function_query_ok || resource.registers < 0 || resource.static_shared_bytes < 0 ||
        resource.local_bytes < 0 || resource.maximum_threads <= 0) { return "compiled-resources-unknown"; }
    if (!resource.capacity_query_ok || resource.resident_cta_capacity <= 0) { return "resident-capacity-unknown"; }
    if (resource.capacity_threads != threads || resource.capacity_dynamic_shared_bytes != 0u) { return "capacity-launch-mismatch"; }
    if (resource.maximum_threads < static_cast<int32_t>(threads)) { return "compiled-block-limit"; }
    if (resource.local_bytes != 0) { return "unmodelled-local-memory-regime"; }
    return {};
}

[[nodiscard]] inline bool matching_candidate(const CubScanArtifact &candidate, uint32_t threads,
                                              const tile::ClosedPrefixAnalysis &proof, uint64_t bytes_per_buffer) noexcept {
    auto input = proof.disjoint.input.argument_index;
    auto output = proof.disjoint.output.argument_index;
    // The original ABI gate has checked that both shifts are smaller than 31.
    auto mask = (uint32_t{1u} << input) | (uint32_t{1u} << output);
    return candidate.ok() && candidate.entry == "luisa_tile_cub_scan" && candidate.threads == threads &&
           candidate.chunk_extent == threads * 8u && proof.logical_contribution_extent % candidate.chunk_extent == 0u &&
           candidate.grid == std::array<uint32_t, 3u>{static_cast<uint32_t>(proof.logical_independent_extent), 1u, 1u} &&
           candidate.block == std::array<uint32_t, 3u>{threads, 1u, 1u} && candidate.alignment_mask == mask &&
           candidate.guard.input_slot == input && candidate.guard.output_slot == output &&
           candidate.guard.input_bytes == bytes_per_buffer && candidate.guard.output_bytes == bytes_per_buffer;
}

[[nodiscard]] inline bool predicts_saving(double candidate, double original) noexcept {
    return std::isfinite(candidate) && candidate > 0.0 && std::isfinite(original) && original > 0.0 &&
           candidate / original < kScanCostImprovementRatio;
}

}// namespace scan_cost_detail

// Call only in explicit cost mode. 'proof' must be freshly obtained from the
// unchanged actual Function. generate_cub_scan independently rechecks semantics
// for each artifact; this score does not authorize a rewrite or prove noalias.
// Each array position is the exact corresponding kScanCostThreads recipe.
// Other experimental worker/layout/fast-math options remain mutually exclusive.
[[nodiscard]] inline ScanCostChoice choose_cub_scan_cost(
    const tile::ClosedPrefixAnalysis &proof, const Artifact &original,
    const std::array<CompiledScanCandidate, 4u> &candidates,
    const ScanCostDevice &device, bool fast_math, bool other_experiment_requested) noexcept {
    ScanCostChoice choice;
    for (auto i = size_t{0u}; i < candidates.size(); i++) {
        choice.candidates[i].threads = kScanCostThreads[i];
        choice.candidates[i].compile_key = candidates[i].compile_key;
        choice.candidates[i].compile_key_known = candidates[i].compile_key_known;
    }
    if (!scan_cost_detail::supports_device(device)) {
        choice.reason = device.query_ok ? "target-profile" : "device-query";
        return choice;
    }
    if (fast_math || other_experiment_requested) {
        choice.reason = fast_math ? "fast-math" : "other-experiment";
        return choice;
    }
    uint64_t bytes_per_buffer{};
    if (!scan_cost_detail::original_facts(proof, original, bytes_per_buffer)) {
        choice.reason = "analysis-or-layout";
        return choice;
    }
    auto rows = proof.logical_independent_extent;
    auto width = proof.logical_contribution_extent;
    if (!scan_cost_detail::tile_features(rows, width, bytes_per_buffer, device, choice.original_features) ||
        !scan_cost_detail::score(kScanCostTileCoefficients, choice.original_features, choice.original_score)) {
        choice.reason = "original-model-facts";
        return choice;
    }
    choice.has_score = true;
    choice.selected_score = choice.original_score;
    choice.status = "retained";
    choice.reason = "predicted-original";
    auto best_threads = 0u;
    auto best_score = choice.original_score;
    for (auto i = size_t{0u}; i < candidates.size(); i++) {
        auto &candidate = candidates[i];
        auto &record = choice.candidates[i];
        auto threads = record.threads;
        if (candidate.artifact == nullptr || !candidate.artifact->ok()) { continue; }
        if (!scan_cost_detail::matching_candidate(*candidate.artifact, threads, proof, bytes_per_buffer)) {
            record.reason = "candidate-abi-or-layout";
            continue;
        }
        if (!candidate.compile_key_known) {
            record.reason = "compilation-identity-unknown";
            continue;
        }
        if (auto reason = scan_cost_detail::resource_rejection(candidate.resources, threads); !reason.empty()) {
            record.reason = reason;
            continue;
        }
        if (!scan_cost_detail::cub_features(rows, width, bytes_per_buffer, threads,
                                            static_cast<uint64_t>(candidate.resources.resident_cta_capacity), device, record.features) ||
            !scan_cost_detail::score(kScanCostCubCoefficients, record.features, record.score)) {
            record.reason = "candidate-model-facts";
            continue;
        }
        record.has_score = true;
        record.reason = "scored";
        // Strict comparison preserves original ties and ascending-T candidate ties.
        if (record.score < best_score) {
            best_score = record.score;
            best_threads = threads;
        }
    }
    if (best_threads != 0u && scan_cost_detail::predicts_saving(best_score, choice.original_score)) {
        choice.selected_threads = best_threads;
        choice.selected_score = best_score;
        choice.status = "selected";
        choice.reason = "predicted-saving";
    }
    // This is a prediction, not installation or invocation selection. Factory
    // failure must retain the original; final-pointer alias/alignment guards
    // remain in CUDAShaderTile::select_launch for both direct and graph paths.
    return choice;
}

}// namespace luisa::compute::cuda::native_tile
