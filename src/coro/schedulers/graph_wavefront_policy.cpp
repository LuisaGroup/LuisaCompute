#include <luisa/coro/schedulers/graph_wavefront_policy.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <luisa/core/logging.h>

namespace luisa::compute::coro {

namespace {

[[nodiscard]] size_t target_index(
    luisa::span<const uint> targets, uint target) noexcept {
    auto iter = std::find(targets.begin(), targets.end(), target);
    return static_cast<size_t>(iter - targets.begin());
}

[[nodiscard]] double live_count(
    const GraphWavefrontPopulation &population) noexcept {
    auto live = 0.0;
    for (auto node = 1u; node < population.queues.size(); ++node) {
        live += population.queues[node];
    }
    return live;
}

}// namespace

uint graph_wavefront_resolve_tail_threshold(
    uint configured_threshold,
    uint active_frame_capacity,
    uint execution_block_size) noexcept {
    LUISA_ASSERT(execution_block_size != 0u,
                 "Graph-wavefront tail threshold requires a positive "
                 "execution block size.");
    if (configured_threshold == 0u ||
        active_frame_capacity == 0u) {
        return 0u;
    }
    if (configured_threshold !=
        graph_wavefront_auto_tail_threshold) {
        return std::min(configured_threshold,
                        active_frame_capacity);
    }
    auto scaled = static_cast<uint64_t>(active_frame_capacity) * 3u / 8u;
    auto aligned = scaled / execution_block_size * execution_block_size;
    if (aligned == 0u) {
        aligned = std::min(active_frame_capacity,
                           execution_block_size);
    }
    return static_cast<uint>(
        std::min<uint64_t>(aligned, active_frame_capacity));
}

GraphWavefrontMarkovModel::GraphWavefrontMarkovModel(
    luisa::vector<luisa::vector<uint>> targets, double prior) noexcept
    : _targets{std::move(targets)} {
    LUISA_ASSERT(!_targets.empty(),
                 "Graph-wavefront Markov model requires an entry node.");
    LUISA_ASSERT(std::isfinite(prior) && prior > 0.0,
                 "Graph-wavefront Markov prior must be finite and positive.");
    _alpha.resize(_targets.size());
    for (auto source = 0u; source < _targets.size(); ++source) {
        auto &row = _targets[source];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        for (auto target : row) {
            LUISA_ASSERT(target != 0u && target < _targets.size(),
                         "Graph-wavefront transition {} -> {} is out of range.",
                         source, target);
        }
        // Column zero is terminal. Keeping it in every row is conservative:
        // a callable may return without a materialized continuation edge.
        _alpha[source].resize(row.size() + 1u, prior);
    }
}

size_t GraphWavefrontMarkovModel::node_count() const noexcept {
    return _targets.size();
}

bool GraphWavefrontMarkovModel::supports(uint from, uint to) const noexcept {
    if (from >= _targets.size() || to >= _targets.size()) { return false; }
    return to == 0u ||
           target_index(luisa::span{_targets[from]}, to) !=
               _targets[from].size();
}

double GraphWavefrontMarkovModel::probability(
    uint from, uint to) const noexcept {
    if (!supports(from, to)) { return 0.0; }
    auto &&row = _alpha[from];
    auto sum = 0.0;
    for (auto alpha : row) { sum += alpha; }
    if (to == 0u) { return row.front() / sum; }
    auto index = target_index(luisa::span{_targets[from]}, to) + 1u;
    return row[index] / sum;
}

GraphWavefrontPopulation GraphWavefrontMarkovModel::predict(
    const GraphWavefrontPopulation &source,
    GraphWavefrontAction action,
    double logical_count,
    double active_capacity) const noexcept {
    LUISA_ASSERT(source.queues.size() == node_count(),
                 "Graph-wavefront population/model size mismatch.");
    LUISA_ASSERT(action.selected_node < node_count(),
                 "Graph-wavefront action node is out of range.");
    auto destination = source;
    auto free_count = source.queues[0u];
    auto admit_count = action.admit_entry ?
                           std::min(free_count,
                                    std::max(logical_count -
                                                 source.generated_count,
                                             0.0)) :
                           0.0;
    destination.generated_count += admit_count;
    destination.queues[0u] -= admit_count;

    auto apply_row = [&](uint source_node, double count) noexcept {
        if (count == 0.0) { return; }
        destination.queues[0u] +=
            count * probability(source_node, 0u);
        for (auto target : _targets[source_node]) {
            destination.queues[target] +=
                count * probability(source_node, target);
        }
    };
    apply_row(0u, admit_count);
    if (action.selected_node != 0u) {
        auto selected_count = source.queues[action.selected_node];
        destination.queues[action.selected_node] = 0.0;
        apply_row(action.selected_node, selected_count);
    }

    auto ownership = 0.0;
    for (auto count : destination.queues) { ownership += count; }
    auto tolerance = std::max(active_capacity, 1.0) * 1e-9;
    LUISA_ASSERT(std::abs(ownership - active_capacity) <= tolerance,
                 "Graph-wavefront predicted ownership is not conserved: "
                 "{} vs {}.", ownership, active_capacity);
    return destination;
}

void GraphWavefrontMarkovModel::observe(
    const GraphWavefrontPopulation &source,
    GraphWavefrontAction action,
    const GraphWavefrontPopulation &destination) noexcept {
    LUISA_ASSERT(source.queues.size() == node_count() &&
                     destination.queues.size() == node_count(),
                 "Graph-wavefront observation/model size mismatch.");
    LUISA_ASSERT(action.selected_node < node_count(),
                 "Graph-wavefront observed action node is out of range.");

    auto update_row = [&](uint source_node, double executed,
                          luisa::span<const double> emitted) noexcept {
        if (executed <= 0.0) { return; }
        auto emitted_total = 0.0;
        for (auto value : emitted) { emitted_total += value; }
        auto terminal = std::max(executed - emitted_total, 0.0);
        _alpha[source_node][0u] += terminal;
        for (auto target : _targets[source_node]) {
            _alpha[source_node][
                target_index(
                    luisa::span{_targets[source_node]}, target) +
                1u] += emitted[target];
        }
    };

    // A selective action carries every unselected queue unchanged. Therefore
    // destination - carried_source is the exact emitted vector. When entry and
    // a continuation execute together their emissions are not identifiable
    // separately from aggregate counters, so defer learning for that mixed
    // action rather than inventing evidence.
    auto selected = action.selected_node;
    auto entry_executed = destination.generated_count - source.generated_count;
    if (entry_executed > 0.0 && selected != 0u) { return; }
    auto emitted = luisa::vector<double>(node_count(), 0.0);
    for (auto target = 1u; target < node_count(); ++target) {
        auto carried = target == selected ? 0.0 : source.queues[target];
        emitted[target] = std::max(destination.queues[target] - carried, 0.0);
        LUISA_ASSERT(
            supports(selected, target) || emitted[target] == 0.0,
            "Graph-wavefront observation contains unsupported transition "
            "{} -> {} with population {}.",
            selected, target, emitted[target]);
    }
    if (selected == 0u) {
        update_row(0u, entry_executed, luisa::span{emitted});
    } else {
        update_row(selected, source.queues[selected], luisa::span{emitted});
    }
}

void graph_wavefront_advance_wait_actions(
    const GraphWavefrontPopulation &source,
    uint selected_node,
    luisa::span<uint64_t> wait_actions) noexcept {
    LUISA_ASSERT(wait_actions.size() == source.queues.size(),
                 "Graph-wavefront waiting-age/population size mismatch.");
    LUISA_ASSERT(selected_node < source.queues.size(),
                 "Graph-wavefront serviced node is out of range.");
    wait_actions[0u] = 0u;
    for (auto node = 1u; node < source.queues.size(); ++node) {
        if (node == selected_node || source.queues[node] <= 0.0) {
            wait_actions[node] = 0u;
        } else {
            LUISA_ASSERT(wait_actions[node] !=
                             std::numeric_limits<uint64_t>::max(),
                         "Graph-wavefront waiting age overflowed.");
            wait_actions[node]++;
        }
    }
}

GraphWavefrontAction graph_wavefront_select_action(
    const GraphWavefrontPopulation &population,
    double logical_count,
    double active_capacity,
    double refill_threshold,
    luisa::span<const uint> refill_nodes,
    luisa::span<const uint64_t> wait_actions,
    uint64_t max_queue_wait_actions) noexcept {
    LUISA_ASSERT(!population.queues.empty(),
                 "Graph-wavefront population requires an entry queue.");
    LUISA_ASSERT(wait_actions.empty() ||
                     wait_actions.size() == population.queues.size(),
                 "Graph-wavefront waiting-age/population size mismatch.");
    auto selected = 0u;
    auto selected_count = 0.0;
    for (auto node = 1u; node < population.queues.size(); ++node) {
        if (population.queues[node] > selected_count) {
            selected = node;
            selected_count = population.queues[node];
        }
    }
    auto forced_by_fairness = false;
    if (max_queue_wait_actions != 0u && !wait_actions.empty()) {
        auto oldest = uint64_t{0u};
        auto overdue = 0u;
        for (auto node = 1u; node < population.queues.size(); ++node) {
            if (population.queues[node] > 0.0 &&
                wait_actions[node] >= max_queue_wait_actions &&
                wait_actions[node] > oldest) {
                overdue = node;
                oldest = wait_actions[node];
            }
        }
        if (overdue != 0u) {
            selected = overdue;
            forced_by_fairness = true;
        }
    }
    auto live = live_count(population);
    auto threshold = refill_threshold == 0.0 ?
                         active_capacity * 0.5 :
                         std::min(refill_threshold, active_capacity);
    auto aligned = refill_nodes.empty() || selected == 0u ||
                   target_index(refill_nodes, selected) != refill_nodes.size();
    auto work_remaining = population.generated_count < logical_count;
    auto admit = work_remaining && population.queues[0u] > 0.0 && aligned &&
                 (live == 0.0 || live < threshold);
    return {.selected_node = selected,
            .admit_entry = admit,
            .forced_by_fairness = forced_by_fairness};
}

}// namespace luisa::compute::coro
