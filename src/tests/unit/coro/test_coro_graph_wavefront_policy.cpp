#include "ut/ut.hpp"

#include <luisa/coro/schedulers/graph_wavefront_policy.h>

#include <cmath>

using namespace luisa;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool near(double a, double b) noexcept {
    return std::abs(a - b) <= 1e-9;
}

}// namespace

int main() {

    "graph_wavefront_action_selection_is_deterministic_and_aligned"_test = [] {
        GraphWavefrontPopulation p{
            .queues = {20.0, 10.0, 40.0, 30.0},
            .generated_count = 80.0};
        auto action = graph_wavefront_select_action(
            p, 200.0, 100.0, 0.0, luisa::span<const uint>{});
        expect(action.selected_node == 2u);
        expect(!action.admit_entry)
            << "half-capacity refill uses a strict live-count inequality";

        p.queues = {60.0, 20.0, 10.0, 10.0};
        uint aligned[]{1u};
        action = graph_wavefront_select_action(
            p, 200.0, 100.0, 50.0, luisa::span{aligned});
        expect(action.selected_node == 1u);
        expect(action.admit_entry);

        p.queues = {70.0, 10.0, 10.0, 10.0};
        action = graph_wavefront_select_action(
            p, 200.0, 100.0, 50.0, luisa::span{aligned});
        expect(action.selected_node == 1u)
            << "strict max selection breaks equal-size ties by node index";
        expect(action.admit_entry);

        p.queues = {70.0, 10.0, 20.0, 0.0};
        action = graph_wavefront_select_action(
            p, 200.0, 100.0, 50.0, luisa::span{aligned});
        expect(action.selected_node == 2u);
        expect(!action.admit_entry)
            << "a non-aligned selected continuation blocks mixed admission";

        p.queues = {100.0, 0.0, 0.0, 0.0};
        action = graph_wavefront_select_action(
            p, 200.0, 100.0, 0.0, luisa::span{aligned});
        expect(action.selected_node == 0u);
        expect(action.admit_entry)
            << "the empty state must never be a fixed point while work remains";
    };

    "graph_wavefront_markov_prediction_conserves_ownership"_test = [] {
        GraphWavefrontMarkovModel model{{{1u}, {1u, 2u}, {1u}}};
        expect(model.supports(1u, 0u));
        expect(model.supports(1u, 1u));
        expect(model.supports(1u, 2u));
        expect(!model.supports(2u, 2u));

        GraphWavefrontPopulation source{
            .queues = {20.0, 60.0, 20.0},
            .generated_count = 80.0};
        auto predicted = model.predict(
            source,
            {.selected_node = 1u, .admit_entry = false},
            200.0, 100.0);
        auto ownership = 0.0;
        for (auto q : predicted.queues) { ownership += q; }
        expect(near(ownership, 100.0));
        expect(near(predicted.queues[0u], 40.0));
        expect(near(predicted.queues[1u], 20.0));
        expect(near(predicted.queues[2u], 40.0));
    };

    "graph_wavefront_markov_learning_respects_graph_support"_test = [] {
        GraphWavefrontMarkovModel model{{{1u}, {1u, 2u}, {1u}}};
        GraphWavefrontPopulation source{
            .queues = {20.0, 60.0, 20.0},
            .generated_count = 80.0};
        GraphWavefrontPopulation destination{
            .queues = {30.0, 40.0, 30.0},
            .generated_count = 80.0};
        model.observe(source,
                      {.selected_node = 1u, .admit_entry = false},
                      destination);
        expect(model.probability(1u, 1u) >
               model.probability(1u, 2u));
        expect(model.probability(1u, 2u) ==
               model.probability(1u, 0u))
            << "equal observed terminal and node-2 counts retain equal "
               "posterior mass";
        expect(model.probability(1u, 3u) == 0.0)
            << "Bayesian updates cannot create an unsupported transition";

        auto p_before = model.probability(1u, 1u);
        auto mixed = destination;
        mixed.generated_count = 90.0;
        model.observe(source,
                      {.selected_node = 1u, .admit_entry = true}, mixed);
        expect(near(model.probability(1u, 1u), p_before))
            << "unidentifiable mixed entry/continuation emissions are not "
               "misattributed";
    };

    return 0;
}
