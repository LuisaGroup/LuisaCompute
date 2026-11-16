// Check existential BDD projection against an independent Boolean truth table.
#include "ut/ut.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/module.h>
#include <luisa/xir/function.h>
#include "../../../xir/passes/coro_guarded_scalar_relation.h"

using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

int main() {
    "existential_projection_matches_truth_tables"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        std::array<Value *, 6u> predicates;
        for (auto &predicate : predicates) {
            predicate = kernel->create_value_argument(Type::of<bool>());
        }
        luisa::compute::xir::detail::CoroBooleanSetManager sets;
        std::array<luisa::compute::xir::detail::CoroBooleanSetManager::Set, 64u> cubes;
        for (uint32_t assignment = 0u; assignment < cubes.size(); ++assignment) {
            auto cube = sets.universe();
            for (uint32_t bit = 0u; bit < predicates.size(); ++bit) {
                cube = sets.intersect(cube, sets.literal(predicates[bit], (assignment & (1u << bit)) != 0u));
            }
            cubes[assignment] = cube;
        }
        uint64_t state = 0xf03a57e97c81d249ull;
        size_t checked = 0u;
        for (uint32_t test = 0u; test < 20u; ++test) {
            state ^= state << 13u;
            state ^= state >> 7u;
            state ^= state << 17u;
            auto truth = test == 0u ? 0ull : test == 1u ? ~0ull : state;
            auto formula = sets.empty_set();
            for (uint32_t assignment = 0u; assignment < cubes.size(); ++assignment) {
                if ((truth >> assignment) & 1ull) { formula = sets.unite(formula, cubes[assignment]); }
            }
            for (uint32_t keep = 0u; keep < 64u; ++keep) {
                luisa::vector<Value *> retained;
                for (uint32_t bit = 0u; bit < predicates.size(); ++bit) {
                    if (keep & (1u << bit)) { retained.emplace_back(predicates[bit]); }
                }
                auto projected = sets.project(formula, retained);
                if (std::popcount(keep) == 5) {
                    auto removed = static_cast<uint32_t>(std::countr_zero((~keep) & 63u));
                    expect(sets.forget(formula, predicates[removed]) == projected);
                    // A repeated query must agree with the memoized result.
                    expect(sets.forget(formula, predicates[removed]) == projected);
                }
                for (uint32_t assignment = 0u; assignment < cubes.size(); ++assignment) {
                    bool expected = false;
                    for (uint32_t witness = 0u; witness < cubes.size(); ++witness) {
                        expected |= ((truth >> witness) & 1ull) != 0u &&
                                    (witness & keep) == (assignment & keep);
                    }
                    auto actual = !sets.is_empty(sets.intersect(projected, cubes[assignment]));
                    expect(actual == expected);
                    ++checked;
                }
                if (keep == 63u) { expect(projected == formula); }
            }
        }
        expect(checked == 81920u);
        expect(!sets.widened());
        auto *unseen = kernel->create_value_argument(Type::of<bool>());
        std::array<Value *, 1u> only_unseen{unseen};
        expect(sets.project(cubes[5u], only_unseen) == sets.universe());
        auto first = sets.project(cubes[5u], predicates);
        (void)sets.literal(unseen, true);
        expect(sets.project(cubes[5u], predicates) == first);
    };
}
