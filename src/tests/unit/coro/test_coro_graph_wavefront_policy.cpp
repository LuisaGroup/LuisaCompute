#include "ut/ut.hpp"

#include <luisa/coro/schedulers/graph_wavefront_policy.h>

#include <cmath>

using namespace luisa;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool is_near(double a, double b) noexcept {
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

    "graph_wavefront_bounded_fairness_serves_adversarial_sparse_queue"_test = [] {
        GraphWavefrontPopulation population{
            .queues = {0.0, 1000.0, 1.0, 1.0},
            .generated_count = 1002.0};
        luisa::vector<uint64_t> wait_actions(4u, 0u);
        constexpr auto horizon = uint64_t{3u};
        auto node2_last_service = uint64_t{0u};
        auto node3_last_service = uint64_t{0u};
        for (auto action_index = uint64_t{1u};
             action_index <= 24u; ++action_index) {
            auto action = graph_wavefront_select_action(
                population, 1002.0, 1002.0, 0.0,
                luisa::span<const uint>{}, luisa::span{wait_actions}, horizon);
            if (action_index <= horizon) {
                expect(action.selected_node == 1u)
                    << "largest-queue throughput policy wins before the "
                       "fairness horizon";
                expect(!action.forced_by_fairness);
            }
            if (action.selected_node == 2u) {
                expect(action_index - node2_last_service <= horizon + 2u)
                    << "node 2 violated the N-queue bounded-service proof";
                node2_last_service = action_index;
            }
            if (action.selected_node == 3u) {
                expect(action_index - node3_last_service <= horizon + 2u)
                    << "node 3 violated the N-queue bounded-service proof";
                node3_last_service = action_index;
            }
            // All three queues are adversarial self-loops: their populations
            // remain non-empty and the hot queue is always numerically largest.
            graph_wavefront_advance_wait_actions(
                population, action.selected_node, luisa::span{wait_actions});
        }
        expect(node2_last_service != 0u);
        expect(node3_last_service != 0u);
    };

    "graph_wavefront_fairness_age_tracks_source_nonempty_interval"_test = [] {
        GraphWavefrontPopulation population{
            .queues = {9.0, 1.0, 0.0}, .generated_count = 1.0};
        luisa::vector<uint64_t> wait_actions{17u, 4u, 9u};
        graph_wavefront_advance_wait_actions(
            population, 0u, luisa::span{wait_actions});
        expect(wait_actions[0u] == 0u);
        expect(wait_actions[1u] == 5u);
        expect(wait_actions[2u] == 0u)
            << "a source-empty queue cannot inherit stale waiting age";
        population.queues[2u] = 1.0;
        auto action = graph_wavefront_select_action(
            population, 10.0, 10.0, 0.0,
            luisa::span<const uint>{}, luisa::span{wait_actions}, 5u);
        expect(action.selected_node == 1u);
        expect(action.forced_by_fairness);
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
        expect(is_near(ownership, 100.0));
        expect(is_near(predicted.queues[0u], 40.0));
        expect(is_near(predicted.queues[1u], 20.0));
        expect(is_near(predicted.queues[2u], 40.0));
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
        expect(is_near(model.probability(1u, 1u), p_before))
            << "unidentifiable mixed entry/continuation emissions are not "
               "misattributed";
    };

    return 0;
}
