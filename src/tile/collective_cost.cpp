#include <luisa/tile/collective_cost.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace luisa::compute::tile {

CollectiveCostFeatures collective_cost_features(const CollectiveWorkAnalysis &a,
                                                CollectiveCostDeviceReference device) noexcept {
    auto fail = [](luisa::string_view error) { return CollectiveCostFeatures{{}, error}; };
    if (!a.ok() || a.programs == 0u || a.largest_materialized_tile_elements == 0u ||
        a.materialized_tile_peak_bytes == 0u || device.processor_count == 0u || device.subgroup_width == 0u) {
        return fail("collective cost requires admitted nonempty work and positive device references");
    }
    auto add = [](uint64_t &total, uint64_t value) {
        if (value > std::numeric_limits<uint64_t>::max() - total) { return false; }
        total += value;
        return true;
    };
    uint64_t contributions = 0u, independent = 0u, sum_inputs = 0u, max_inputs = 0u, max_width = 0u;
    for (auto &&work : a.collectives) {
        if (work.element != ScalarType::FLOAT32 || work.contribution_extent == 0u || work.independent_elements == 0u ||
            work.independent_elements > std::numeric_limits<uint64_t>::max() / work.contribution_extent ||
            work.input_elements != work.contribution_extent * work.independent_elements) {
            return fail("collective cost received inconsistent FP32 contribution facts");
        }
        switch (work.kind) {
            case CollectiveKind::SUM:
                if (!add(sum_inputs, work.input_elements)) { return fail("collective cost SUM count overflow"); }
                break;
            case CollectiveKind::MAXIMUM:
                if (!add(max_inputs, work.input_elements)) { return fail("collective cost MAXIMUM count overflow"); }
                break;
            case CollectiveKind::MINIMUM:
            case CollectiveKind::INCLUSIVE_SUM: break;
            default: return fail("collective cost received an unknown collective kind");
        }
        if (!add(contributions, work.input_elements) || !add(independent, work.independent_elements)) {
            return fail("collective cost aggregate count overflow");
        }
        max_width = std::max(max_width, work.contribution_extent);
    }
    auto bytes = a.global_read_bytes_per_program;
    if (!add(bytes, a.global_write_bytes_per_program)) { return fail("collective cost byte count overflow"); }
    auto count = static_cast<double>(contributions);
    auto subgroup = static_cast<double>(device.subgroup_width);
    auto log = [](double x) { return std::log2(1.0 + x); };
    CollectiveCostFeatures result;
    result.values = {
        log(static_cast<double>(a.programs) / static_cast<double>(device.processor_count)),
        log(static_cast<double>(a.largest_materialized_tile_elements)),
        log(static_cast<double>(a.materialized_tile_peak_bytes) / (subgroup * 4.0)),
        log(static_cast<double>(a.elementwise_elements_per_program) / count),
        log(static_cast<double>(max_width) / subgroup),
        log(static_cast<double>(independent)),
        static_cast<double>(sum_inputs) / count,
        static_cast<double>(max_inputs) / count,
        log(static_cast<double>(bytes) / (4.0 * count))};
    for (auto value : result.values) {
        if (!std::isfinite(value)) { return fail("collective cost derived a non-finite feature"); }
    }
    return result;
}

CollectiveCostResult evaluate_collective_cost_tree(luisa::span<const CollectiveCostTreeNode> nodes,
                                                   luisa::span<const double> features) noexcept {
    auto fail = [](luisa::string_view error) { return CollectiveCostResult{0.0, error}; };
    if (nodes.empty() || nodes.size() > kCollectiveCostMaxNodes ||
        features.empty() || features.size() > kCollectiveCostMaxFeatures) {
        return fail("collective cost tree or feature count is outside its bounded interface");
    }
    for (auto value : features) {
        if (!std::isfinite(value)) { return fail("collective cost feature is not finite"); }
    }
    for (auto &&node : nodes) {
        if (!std::isfinite(node.threshold) || !std::isfinite(node.log_score)) {
            return fail("collective cost node contains a non-finite value");
        }
        if (node.feature < -1 || (node.feature >= 0 &&
                                  (static_cast<size_t>(node.feature) >= features.size() ||
                                   node.left >= nodes.size() || node.right >= nodes.size()))) {
            return fail("collective cost node references an invalid feature or child");
        }
    }
    // Validate every connected component, including unreachable nodes. A fixed
    // explicit stack bounds both memory and work and avoids recursive descent.
    std::array<uint8_t, kCollectiveCostMaxNodes> colors{};
    std::array<uint32_t, kCollectiveCostMaxNodes> stack{};
    for (auto start = size_t{0u}; start < nodes.size(); start++) {
        if (colors[start] != 0u) { continue; }
        auto depth = size_t{1u};
        stack[0u] = static_cast<uint32_t>(start);
        while (depth != 0u) {
            auto index = stack[depth - 1u];
            auto &&node = nodes[index];
            if (colors[index] == 0u) {
                colors[index] = 1u;
                if (node.feature == -1) {
                    colors[index] = 2u;
                    depth--;
                    continue;
                }
                if (colors[node.left] == 1u) { return fail("collective cost tree contains a cycle"); }
                if (colors[node.left] == 0u) {
                    stack[depth++] = node.left;
                    continue;
                }
            }
            if (colors[node.right] == 1u) { return fail("collective cost tree contains a cycle"); }
            if (colors[node.right] == 0u) {
                stack[depth++] = node.right;
                continue;
            }
            colors[index] = 2u;
            depth--;
        }
    }
    auto index = uint32_t{0u};
    for (auto remaining = nodes.size(); remaining != 0u; remaining--) {
        auto &&node = nodes[index];
        if (node.feature == -1) { return CollectiveCostResult{node.log_score, {}}; }
        index = features[static_cast<size_t>(node.feature)] <= node.threshold ? node.left : node.right;
    }
    return fail("collective cost traversal exceeded its node bound");
}

}// namespace luisa::compute::tile
