#pragma once

#include <cstdint>

#include <luisa/core/basic_types.h>
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::coro {

/// Host-side state estimate used by the graph-wavefront action scheduler.
/// Queue zero is the stable free-frame stack; queue i > 0 belongs to CoroGraph
/// node i. `generated_count` is monotone and bounded by the logical launch.
struct GraphWavefrontPopulation {
    luisa::vector<double> queues;
    double generated_count{0.0};
};

/// One legal graph-wavefront action. Node zero means entry-only admission;
/// nonzero means consume exactly that continuation queue. Entry admission may
/// be combined with a continuation only when the host refill policy permits.
struct GraphWavefrontAction {
    uint selected_node{0u};
    bool admit_entry{false};
};

/// A support-constrained first-order transition model. Every row has one
/// additional terminal outcome at column zero. Unsupported CoroGraph edges
/// are fixed to probability zero. Observed action deltas update a Dirichlet
/// posterior over the supported row, while prediction can never affect the
/// exact device ownership certificate or queue bounds.
class LUISA_CORO_API GraphWavefrontMarkovModel {

private:
    luisa::vector<luisa::vector<uint>> _targets;
    luisa::vector<luisa::vector<double>> _alpha;

public:
    explicit GraphWavefrontMarkovModel(
        luisa::vector<luisa::vector<uint>> targets,
        double prior = 1.0) noexcept;

    [[nodiscard]] size_t node_count() const noexcept;
    [[nodiscard]] bool supports(uint from, uint to) const noexcept;
    [[nodiscard]] double probability(uint from, uint to) const noexcept;
    [[nodiscard]] GraphWavefrontPopulation predict(
        const GraphWavefrontPopulation &source,
        GraphWavefrontAction action,
        double logical_count,
        double active_capacity) const noexcept;
    void observe(const GraphWavefrontPopulation &source,
                 GraphWavefrontAction action,
                 const GraphWavefrontPopulation &destination) noexcept;
};

/// Select the largest predicted continuation queue. Strict comparison makes
/// the lowest node index the deterministic winner on ties. Entry is admitted
/// when the scheduler is empty, and otherwise only below `refill_threshold`;
/// a zero threshold means half of active capacity. If `refill_nodes` is not
/// empty, mixed admission is restricted to those selected continuations.
[[nodiscard]] LUISA_CORO_API GraphWavefrontAction graph_wavefront_select_action(
    const GraphWavefrontPopulation &population,
    double logical_count,
    double active_capacity,
    double refill_threshold,
    luisa::span<const uint> refill_nodes = {}) noexcept;

}// namespace luisa::compute::coro
