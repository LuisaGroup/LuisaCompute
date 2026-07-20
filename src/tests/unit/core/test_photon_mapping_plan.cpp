// Test for the deterministic photon-mapping storage and accumulation plan.
// This test covers:
// - unique path/depth photon slots and checked capacity boundaries
// - per-term and complete-sum fixed-point overflow boundaries
// - order-independent two-word accumulation across uint32 carries
// - bounded quantization and malformed-term rejection

#include "ut/ut.hpp"

#include "photon_mapping_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto production_storage =
    photon_mapping::plan_photon_storage(1'000'000u, 8u);
constexpr auto production_accumulator =
    photon_mapping::plan_fixed_point_accumulator(
        production_storage.capacity, 70u, 24u);

static_assert(production_storage.valid);
static_assert(production_storage.capacity == 8'000'000u);
static_assert(production_accumulator.valid);

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "photon_storage_assigns_unique_path_depth_slots"_test = [] {
        std::array<bool, 12u> seen{};
        constexpr auto plan =
            photon_mapping::plan_photon_storage(3u, 4u);
        expect(plan.valid);
        expect(eq(plan.capacity, 12u));
        for (auto path = 0u; path < plan.path_count; ++path) {
            for (auto depth = 0u; depth < plan.max_depth; ++depth) {
                auto slot = photon_mapping::photon_slot_index(
                    plan, path, depth);
                expect(slot.has_value());
                expect(*slot < seen.size());
                expect(!seen[*slot]);
                seen[*slot] = true;
            }
        }
        expect(std::all_of(seen.begin(), seen.end(),
                           [](bool value) noexcept { return value; }));
        expect(!photon_mapping::photon_slot_index(
                    plan, plan.path_count, 0u)
                    .has_value());
        expect(!photon_mapping::photon_slot_index(
                    plan, 0u, plan.max_depth)
                    .has_value());
    };

    "photon_storage_rejects_invalid_and_overflowing_capacity"_test = [] {
        expect(!photon_mapping::plan_photon_storage(0u, 1u).valid);
        expect(!photon_mapping::plan_photon_storage(1u, 0u).valid);
        expect(photon_mapping::plan_photon_storage(
                   std::numeric_limits<uint32_t>::max(), 1u)
                   .valid);
        expect(!photon_mapping::plan_photon_storage(
                    std::numeric_limits<uint32_t>::max(), 2u)
                    .valid);
    };

    "photon_fixed_point_plan_proves_both_numeric_boundaries"_test = [] {
        expect(production_accumulator.valid);
        expect(eq(production_accumulator.scale, 1u << 24u));
        expect(eq(production_accumulator.max_quantized_term,
                  70u << 24u));
        expect(production_accumulator.max_quantized_sum <
               std::numeric_limits<uint64_t>::max());
        expect(production_accumulator.max_input_quantization_error <
               0.24);

        expect(!photon_mapping::plan_fixed_point_accumulator(
                    0u, 70u, 24u)
                    .valid);
        expect(!photon_mapping::plan_fixed_point_accumulator(
                    1u, 0u, 24u)
                    .valid);
        expect(!photon_mapping::plan_fixed_point_accumulator(
                    1u, 1u, 32u)
                    .valid);
        expect(!photon_mapping::plan_fixed_point_accumulator(
                    production_storage.capacity, 256u, 24u)
                    .valid)
            << "one term must fit the uint32 low-word input";
        expect(!photon_mapping::plan_fixed_point_accumulator(
                    std::numeric_limits<uint64_t>::max(), 1u, 1u)
                    .valid)
            << "the complete sum must fit two uint32 words";
    };

    "photon_fixed_point_accumulation_is_order_independent"_test = [] {
        constexpr std::array<uint32_t, 5u> terms{
            std::numeric_limits<uint32_t>::max(),
            17u,
            0x80000000u,
            3u,
            0x7fffffffu,
        };
        auto accumulate = [](auto begin, auto end) noexcept {
            photon_mapping::FixedPointWords sum;
            for (auto iter = begin; iter != end; ++iter) {
                sum = photon_mapping::add_fixed_point_term(sum, *iter);
            }
            return sum;
        };
        auto forward = accumulate(terms.begin(), terms.end());
        auto reverse = accumulate(terms.rbegin(), terms.rend());
        expect(eq(forward.low, reverse.low));
        expect(eq(forward.high, reverse.high));
        expect(eq(photon_mapping::fixed_point_word_value(forward),
                  uint64_t{0x0000000200000012u}));
    };

    "photon_fixed_point_quantization_is_bounded_and_fail_closed"_test = [] {
        constexpr std::array<float, 6u> terms{
            0.0f, 0.1f, 0.25f, 1.0f, 17.125f, 69.75f};
        photon_mapping::FixedPointWords sum;
        double exact_sum = 0.0;
        for (auto term : terms) {
            auto quantized = photon_mapping::quantize_fixed_point_term(
                term, production_accumulator);
            expect(quantized.has_value());
            sum = photon_mapping::add_fixed_point_term(
                sum, *quantized);
            exact_sum += static_cast<double>(term);
        }
        auto decoded = photon_mapping::decode_fixed_point_words(
            sum, production_accumulator);
        auto error_bound = static_cast<double>(terms.size()) /
                           (2.0 * production_accumulator.scale);
        expect(std::abs(decoded - exact_sum) <= error_bound);

        expect(!photon_mapping::quantize_fixed_point_term(
                    -1.0f, production_accumulator)
                    .has_value());
        expect(!photon_mapping::quantize_fixed_point_term(
                    70.5f, production_accumulator)
                    .has_value());
        expect(!photon_mapping::quantize_fixed_point_term(
                    std::numeric_limits<float>::infinity(),
                    production_accumulator)
                    .has_value());
        expect(!photon_mapping::quantize_fixed_point_term(
                    std::numeric_limits<float>::quiet_NaN(),
                    production_accumulator)
                    .has_value());
    };
}
