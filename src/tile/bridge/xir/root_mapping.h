#pragma once

#include <algorithm>
#include <stdexcept>
#include <luisa/tile/dimension.h>

namespace luisa::compute::tile::bridge::xir::detail {

struct RootDigit {
    uint32_t axis;
    uint32_t extent;
    uint32_t scale;
};

// Mixed-radix execution coordinates, unrelated to a buffer's memory layout.
// T factors the original axes; order permutes both outer and inner digits.
// Adjacent compatible digits are merged so identity factorizations retain
// exactly the old lowering and cost path, including unit/single-axis domains.
struct RootMapping {
    luisa::vector<RootDigit> digits;
    uint32_t volume{1u};
    bool identity{true};
    uint32_t decode_arithmetic{0u};
};

[[nodiscard]] inline RootMapping root_mapping(
    const IndexSpace &domain, luisa::span<const uint32_t> order,
    luisa::span<const uint32_t> tiles) {
    auto rank = domain.rank();
    if (rank == 0u || (!order.empty() && order.size() != rank)) {
        throw std::invalid_argument{"XIR root axis order must be a complete permutation"};
    }
    if (!tiles.empty() && tiles.size() != rank) {
        throw std::invalid_argument{"XIR root axis tiles must specify every original axis"};
    }
    RootMapping result;
    luisa::vector<uint32_t> extents(rank), factors(rank, 1u);
    luisa::vector<bool> seen(rank, false);
    for (size_t i = 0u; i < rank; i++) {
        auto axis = order.empty() ? i : order[i];
        if (axis >= rank || seen[axis]) {
            throw std::invalid_argument{"XIR root axis order must be a complete permutation"};
        }
        seen[axis] = true;
        auto &extent = domain.axis(i).extent;
        if (!extent.is_constant() || extent.constant_value() == 0u || extent.constant_value() > UINT32_MAX / result.volume) {
            throw std::invalid_argument{"XIR root traversal requires positive static extents and uint32 volume"};
        }
        extents[i] = static_cast<uint32_t>(extent.constant_value());
        result.volume *= extents[i];
        factors[i] = tiles.empty() ? 1u : tiles[i];
        if (factors[i] == 0u || extents[i] % factors[i] != 0u) {
            throw std::invalid_argument{"XIR root axis tiles must be positive divisors of the original extents"};
        }
    }
    auto append = [&](RootDigit digit) {
        if (digit.extent == 1u) { return; }
        if (!result.digits.empty()) {
            auto &previous = result.digits.back();
            if (previous.axis == digit.axis && static_cast<uint64_t>(digit.scale) * digit.extent == previous.scale) {
                previous.extent *= digit.extent;// bounded by the original axis extent
                previous.scale = digit.scale;
                return;
            }
        }
        result.digits.emplace_back(digit);
    };
    for (auto inner : {false, true}) {
        for (size_t i = 0u; i < rank; i++) {
            auto axis = static_cast<uint32_t>(order.empty() ? i : order[i]);
            append({axis, inner ? factors[axis] : extents[axis] / factors[axis], inner ? 1u : factors[axis]});
        }
    }
    auto position = size_t{0u};
    for (size_t i = 0u; i < rank; i++) {
        auto axis = order.empty() ? i : order[i];
        if (extents[axis] == 1u) { continue; }
        result.identity &= position < result.digits.size() && result.digits[position].axis == axis &&
                           result.digits[position].extent == extents[axis] && result.digits[position].scale == 1u;
        position++;
    }
    result.identity &= position == result.digits.size();
    if (!result.identity) {
        std::fill(seen.begin(), seen.end(), false);
        for (size_t i = 0u; i < result.digits.size(); i++) {
            auto digit = result.digits[i];
            // One division, optional remainder, scale and coordinate sum.
            // This is an uncalibrated integer-work prior, not native cycles.
            result.decode_arithmetic += 1u + (i != 0u) + (digit.scale != 1u) + seen[digit.axis];
            seen[digit.axis] = true;
        }
    }
    return result;
}

}// namespace luisa::compute::tile::bridge::xir::detail
