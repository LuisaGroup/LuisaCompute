#include "reference_warp_collectives.h"

#include <iostream>
#include <string_view>

using namespace luisa::compute::simd;

namespace {

[[nodiscard]] bool check(bool condition, const char *expression,
                         const char *file, int line) noexcept {
    if (!condition) {
        std::cerr << file << ':' << line << ": check failed: "
                  << expression << '\n';
    }
    return condition;
}

#define CHECK(EXPR)                                                           \
    do {                                                                      \
        if (!check(static_cast<bool>(EXPR), #EXPR, __FILE__, __LINE__)) {     \
            return false;                                                     \
        }                                                                     \
    } while (false)

template<size_t Width>
[[nodiscard]] bool test_full_warp() noexcept {
    using Ops = reference::WarpCollectives<Width>;
    using Mask = typename Ops::Mask;
    typename Ops::template Lanes<uint32_t> values{};
    typename Ops::template Lanes<bool> predicate{};
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        values[lane] = static_cast<uint32_t>(lane + 1u);
        predicate[lane] = lane % 2u == 0u;
    }
    auto participants = Mask::full();
    auto expected_sum = static_cast<uint32_t>(Width * (Width + 1u) / 2u);
    CHECK(Ops::active_sum(participants, values) == expected_sum);
    CHECK(Ops::active_product(Mask::single(0u), values) == 1u);
    CHECK(Ops::active_min(participants, values) == 1u);
    CHECK(Ops::active_max(participants, values) == Width);
    CHECK(Ops::first_active_lane(participants) == 0u);
    auto first = Ops::is_first_active_lane(participants);
    CHECK(first[0u]);
    for (auto lane = size_t{1u}; lane < Width; lane++) {
        CHECK(!first[lane]);
    }
    CHECK(Ops::active_count_bits(participants, predicate) ==
          (Width + 1u) / 2u);
    CHECK(Ops::active_any(participants, predicate));
    if constexpr (Width > 1u) {
        CHECK(!Ops::active_all(participants, predicate));
    }
    auto prefix = Ops::prefix_sum(participants, values);
    auto running = uint32_t{0u};
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        CHECK(prefix[lane] == running);
        running += values[lane];
    }
    return true;
}

template<size_t Width>
[[nodiscard]] bool test_sparse_warp() noexcept {
    static_assert(Width >= 8u);
    using Ops = reference::WarpCollectives<Width>;
    using Mask = typename Ops::Mask;
    auto participants = Mask::from_indices({0u, 1u, 6u});
    typename Ops::template Lanes<uint32_t> values{};
    typename Ops::template Lanes<bool> predicate{};
    for (auto lane = size_t{0u}; lane < Width; lane++) {
        values[lane] = static_cast<uint32_t>(lane + 2u);
    }
    predicate[0u] = true;
    predicate[6u] = true;

    CHECK(Ops::active_sum(participants, values) == 13u);
    CHECK(Ops::active_product(participants, values) == 48u);
    CHECK(Ops::active_count_bits(participants, predicate) == 2u);
    auto ballot = Ops::active_bit_mask(participants, predicate);
    CHECK(ballot[0u] == ((1u << 0u) | (1u << 6u)));
    CHECK(ballot[1u] == 0u && ballot[2u] == 0u && ballot[3u] == 0u);

    auto prefix_sum = Ops::prefix_sum(participants, values);
    CHECK(prefix_sum[0u] == 0u);
    CHECK(prefix_sum[1u] == 2u);
    CHECK(prefix_sum[6u] == 5u);
    auto prefix_product = Ops::prefix_product(participants, values);
    CHECK(prefix_product[0u] == 1u);
    CHECK(prefix_product[1u] == 2u);
    CHECK(prefix_product[6u] == 6u);
    auto prefix_bits = Ops::prefix_count_bits(participants, predicate);
    CHECK(prefix_bits[0u] == 0u);
    CHECK(prefix_bits[1u] == 1u);
    CHECK(prefix_bits[6u] == 1u);

    auto broadcast = Ops::read_lane(participants, values, 6u);
    CHECK(broadcast.valid());
    participants.for_each([&](auto lane) noexcept {
        if (broadcast.values[lane] != 8u) {
            broadcast.invalid_lanes.set(lane);
        }
    });
    CHECK(broadcast.valid());
    auto invalid = Ops::read_lane(participants, values, 2u);
    CHECK(!invalid.valid());
    CHECK(invalid.invalid_lanes == participants);
    CHECK(Ops::read_first_active_lane(participants, values) == 2u);
    return true;
}

}// namespace

int main() {
    struct Test {
        std::string_view name;
        bool (*run)();
    };
    constexpr Test tests[]{
        {"full warp1", &test_full_warp<1u>},
        {"full warp4", &test_full_warp<4u>},
        {"full warp8", &test_full_warp<8u>},
        {"full warp16", &test_full_warp<16u>},
        {"sparse warp8", &test_sparse_warp<8u>},
        {"sparse warp128", &test_sparse_warp<128u>},
    };
    auto failures = 0u;
    for (auto test : tests) {
        if (test.run()) {
            std::cout << "[pass] " << test.name << '\n';
        } else {
            std::cerr << "[fail] " << test.name << '\n';
            ++failures;
        }
    }
    return failures == 0u ? 0 : 1;
}
