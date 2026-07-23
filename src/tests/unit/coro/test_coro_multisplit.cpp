// Test for stable_multisplit clustering utility.
// Covers: interleaved tokens, uniform tokens, empty input, mixed alive/dead.

#include "ut/ut.hpp"

#include <luisa/core/logging.h>
#include <luisa/coro/schedulers/multisplit.h>

using namespace luisa;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_multisplit_interleaved() {

    "multisplit_interleaved_two_tokens"_test = [] {
        // 4 instances with tokens {0, 1, 0, 1}
        // → two groups: token0={0,2}, token1={1,3}
        luisa::vector<uint32_t> tokens = {0u, 1u, 0u, 1u};
        luisa::vector<bool> alive = {true, true, true, true};

        auto result = stable_multisplit(tokens, alive);

        expect(result.groups.size() == 2u);

        // token 0 → indices {0, 2}
        auto it0 = result.groups.find(0u);
        expect(it0 != result.groups.end());
        expect(it0->second.size() == 2u);
        expect(it0->second[0u] == 0u);
        expect(it0->second[1u] == 2u);

        // token 1 → indices {1, 3}
        auto it1 = result.groups.find(1u);
        expect(it1 != result.groups.end());
        expect(it1->second.size() == 2u);
        expect(it1->second[0u] == 1u);
        expect(it1->second[1u] == 3u);
    };
}

void reg_multisplit_uniform() {

    "multisplit_uniform_all_same_token"_test = [] {
        // All instances have token 7 → one group with all indices
        luisa::vector<uint32_t> tokens = {7u, 7u, 7u, 7u, 7u};
        luisa::vector<bool> alive = {true, true, true, true, true};

        auto result = stable_multisplit(tokens, alive);

        expect(result.groups.size() == 1u);

        auto it = result.groups.find(7u);
        expect(it != result.groups.end());
        expect(it->second.size() == 5u);
        for (size_t i = 0u; i < 5u; ++i) {
            expect(it->second[i] == i);
        }
    };
}

void reg_multisplit_empty() {

    "multisplit_empty_input"_test = [] {
        luisa::vector<uint32_t> tokens;
        luisa::vector<bool> alive;

        auto result = stable_multisplit(tokens, alive);

        expect(result.groups.empty());
        expect(result.groups.size() == 0u);
    };
}

void reg_multisplit_mixed_alive_dead() {

    "multisplit_mixed_alive_dead"_test = [] {
        // 6 instances: tokens {0, 1, 0, 1, 0, 1}
        // alive:      {T, T, F, T, T, F}
        // → token 0: alive at 0, 4 → {0, 4}
        // → token 1: alive at 1, 3 → {1, 3}  (indices 2 and 5 are dead/skipped)
        luisa::vector<uint32_t> tokens = {0u, 1u, 0u, 1u, 0u, 1u};
        luisa::vector<bool> alive = {true, true, false, true, true, false};

        auto result = stable_multisplit(tokens, alive);

        expect(result.groups.size() == 2u);

        // token 0 → only alive indices {0, 4}
        auto it0 = result.groups.find(0u);
        expect(it0 != result.groups.end());
        expect(it0->second.size() == 2u);
        expect(it0->second[0u] == 0u);
        expect(it0->second[1u] == 4u);

        // token 1 → only alive indices {1, 3}
        auto it1 = result.groups.find(1u);
        expect(it1 != result.groups.end());
        expect(it1->second.size() == 2u);
        expect(it1->second[0u] == 1u);
        expect(it1->second[1u] == 3u);
    };

    "multisplit_all_dead"_test = [] {
        // All instances dead → all groups empty (no entries in map)
        luisa::vector<uint32_t> tokens = {0u, 1u, 2u};
        luisa::vector<bool> alive = {false, false, false};

        auto result = stable_multisplit(tokens, alive);

        expect(result.groups.empty());
        expect(result.groups.size() == 0u);
    };

    "multisplit_some_tokens_no_alive"_test = [] {
        // Token 0 has alive instances, token 1 has none
        luisa::vector<uint32_t> tokens = {0u, 0u, 1u, 1u};
        luisa::vector<bool> alive = {true, true, false, false};

        auto result = stable_multisplit(tokens, alive);

        expect(result.groups.size() == 1u);
        expect(result.groups.find(0u) != result.groups.end());
        expect(result.groups.find(1u) == result.groups.end());
    };
}

void reg_multisplit_stability() {

    "multisplit_preserves_original_order"_test = [] {
        // Verify that within each token group, indices appear in original order.
        // Use 10 instances with repeated tokens to test stability.
        luisa::vector<uint32_t> tokens;
        luisa::vector<bool> alive;

        for (size_t i = 0u; i < 10u; ++i) {
            tokens.push_back(static_cast<uint32_t>(i % 3u));
            alive.push_back(true);
        }

        auto result = stable_multisplit(tokens, alive);

        // Token 0: indices {0, 3, 6, 9}
        auto it0 = result.groups.find(0u);
        expect(it0 != result.groups.end());
        expect(it0->second.size() == 4u);
        expect(it0->second[0u] == 0u);
        expect(it0->second[1u] == 3u);
        expect(it0->second[2u] == 6u);
        expect(it0->second[3u] == 9u);

        // Token 1: indices {1, 4, 7}
        auto it1 = result.groups.find(1u);
        expect(it1 != result.groups.end());
        expect(it1->second.size() == 3u);
        expect(it1->second[0u] == 1u);
        expect(it1->second[1u] == 4u);
        expect(it1->second[2u] == 7u);

        // Token 2: indices {2, 5, 8}
        auto it2 = result.groups.find(2u);
        expect(it2 != result.groups.end());
        expect(it2->second.size() == 3u);
        expect(it2->second[0u] == 2u);
        expect(it2->second[1u] == 5u);
        expect(it2->second[2u] == 8u);
    };
}

int main(int /*argc*/, char * /*argv*/[]) {
    reg_multisplit_interleaved();
    reg_multisplit_uniform();
    reg_multisplit_empty();
    reg_multisplit_mixed_alive_dead();
    reg_multisplit_stability();
    return 0;
}
