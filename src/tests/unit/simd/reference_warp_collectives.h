#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <type_traits>

#include "../schedule/cohort_scheduler.h"

namespace luisa::compute::simd::reference {

template<typename T, size_t Width>
using LaneArray = std::array<T, Width>;

template<typename T, size_t Width>
struct LaneReadResult {
    LaneArray<T, Width> values{};
    schedule::LaneMask<Width> invalid_lanes{};

    [[nodiscard]] constexpr bool valid() const noexcept {
        return invalid_lanes.none();
    }
};

// Unit-test oracle only. This scalar implementation is deliberately outside
// the backend: production lowering always represents warp N as LLVM <N x T>
// and delegates SIMD instruction selection and scheduling to LLVM.
template<size_t Width>
class WarpCollectives {

public:
    using Mask = schedule::LaneMask<Width>;

    template<typename T>
    using Lanes = LaneArray<T, Width>;

private:
    template<typename T, typename Binary>
    [[nodiscard]] static constexpr std::optional<T> _reduce(
        Mask participants, const Lanes<T> &values,
        Binary &&binary) noexcept {
        auto first = participants.first();
        if (!first) { return std::nullopt; }
        auto result = values[*first];
        auto rest = participants - Mask::single(*first);
        rest.for_each([&](auto lane) noexcept {
            result = std::invoke(binary, result, values[lane]);
        });
        return result;
    }

    template<typename T, typename Binary>
    [[nodiscard]] static constexpr Lanes<T> _exclusive_scan(
        Mask participants, const Lanes<T> &values, T identity,
        Binary &&binary) noexcept {
        Lanes<T> result{};
        auto accumulated = identity;
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            if (!participants.test(lane)) { continue; }
            result[lane] = accumulated;
            accumulated = std::invoke(
                binary, accumulated, values[lane]);
        }
        return result;
    }

public:
    [[nodiscard]] static constexpr std::optional<uint32_t>
    first_active_lane(Mask participants) noexcept {
        auto first = participants.first();
        return first ?
                   std::optional<uint32_t>{
                       static_cast<uint32_t>(*first)} :
                   std::nullopt;
    }

    [[nodiscard]] static constexpr Lanes<bool>
    is_first_active_lane(Mask participants) noexcept {
        Lanes<bool> result{};
        if (auto first = participants.first()) {
            result[*first] = true;
        }
        return result;
    }

    template<typename T>
    [[nodiscard]] static constexpr std::optional<T>
    read_first_active_lane(Mask participants,
                           const Lanes<T> &values) noexcept {
        auto first = participants.first();
        return first ? std::optional<T>{values[*first]} : std::nullopt;
    }

    template<typename T, typename Index>
        requires std::is_integral_v<Index>
    [[nodiscard]] static constexpr LaneReadResult<T, Width> read_lane(
        Mask participants, const Lanes<T> &values,
        const Lanes<Index> &source_lanes) noexcept {
        LaneReadResult<T, Width> result{};
        participants.for_each([&](auto lane) noexcept {
            auto source = source_lanes[lane];
            auto valid = [&]() noexcept {
                if constexpr (std::is_signed_v<Index>) {
                    if (source < 0) { return false; }
                }
                return static_cast<uint64_t>(source) < Width &&
                       participants.test(static_cast<size_t>(source));
            }();
            if (valid) {
                result.values[lane] =
                    values[static_cast<size_t>(source)];
            } else {
                result.invalid_lanes.set(lane);
            }
        });
        return result;
    }

    template<typename T, typename Index>
        requires std::is_integral_v<Index>
    [[nodiscard]] static constexpr LaneReadResult<T, Width> read_lane(
        Mask participants, const Lanes<T> &values,
        Index source_lane) noexcept {
        Lanes<Index> source_lanes{};
        source_lanes.fill(source_lane);
        return read_lane(participants, values, source_lanes);
    }

    [[nodiscard]] static constexpr uint32_t active_count_bits(
        Mask participants, const Lanes<bool> &predicate) noexcept {
        auto count = uint32_t{0u};
        participants.for_each([&](auto lane) noexcept {
            count += static_cast<uint32_t>(predicate[lane]);
        });
        return count;
    }

    [[nodiscard]] static constexpr bool active_all(
        Mask participants, const Lanes<bool> &predicate) noexcept {
        auto result = true;
        participants.for_each([&](auto lane) noexcept {
            result &= predicate[lane];
        });
        return result;
    }

    [[nodiscard]] static constexpr bool active_any(
        Mask participants, const Lanes<bool> &predicate) noexcept {
        auto result = false;
        participants.for_each([&](auto lane) noexcept {
            result |= predicate[lane];
        });
        return result;
    }

    template<typename T>
    [[nodiscard]] static constexpr bool active_all_equal(
        Mask participants, const Lanes<T> &values) noexcept {
        auto first = participants.first();
        if (!first) { return true; }
        auto result = true;
        auto rest = participants - Mask::single(*first);
        rest.for_each([&](auto lane) noexcept {
            result &= values[lane] == values[*first];
        });
        return result;
    }

    [[nodiscard]] static constexpr std::array<uint32_t, 4u>
    active_bit_mask(Mask participants,
                    const Lanes<bool> &predicate) noexcept {
        std::array<uint32_t, 4u> result{};
        participants.for_each([&](auto lane) noexcept {
            if (predicate[lane]) {
                result[lane / 32u] |=
                    uint32_t{1u} << (lane % 32u);
            }
        });
        return result;
    }

    [[nodiscard]] static constexpr Lanes<uint32_t>
    prefix_count_bits(Mask participants,
                      const Lanes<bool> &predicate) noexcept {
        Lanes<uint32_t> result{};
        auto count = uint32_t{0u};
        for (auto lane = size_t{0u}; lane < Width; lane++) {
            if (!participants.test(lane)) { continue; }
            result[lane] = count;
            count += static_cast<uint32_t>(predicate[lane]);
        }
        return result;
    }

    template<typename T>
    [[nodiscard]] static constexpr std::optional<T> active_sum(
        Mask participants, const Lanes<T> &values) noexcept {
        return _reduce(participants, values, std::plus<>{});
    }

    template<typename T>
    [[nodiscard]] static constexpr std::optional<T> active_product(
        Mask participants, const Lanes<T> &values) noexcept {
        return _reduce(participants, values, std::multiplies<>{});
    }

    template<typename T>
    [[nodiscard]] static constexpr std::optional<T> active_min(
        Mask participants, const Lanes<T> &values) noexcept {
        return _reduce(participants, values,
                       [](auto lhs, auto rhs) noexcept {
                           return std::min(lhs, rhs);
                       });
    }

    template<typename T>
    [[nodiscard]] static constexpr std::optional<T> active_max(
        Mask participants, const Lanes<T> &values) noexcept {
        return _reduce(participants, values,
                       [](auto lhs, auto rhs) noexcept {
                           return std::max(lhs, rhs);
                       });
    }

    template<typename T>
        requires std::is_integral_v<T>
    [[nodiscard]] static constexpr std::optional<T> active_bit_and(
        Mask participants, const Lanes<T> &values) noexcept {
        return _reduce(participants, values, std::bit_and<>{});
    }

    template<typename T>
        requires std::is_integral_v<T>
    [[nodiscard]] static constexpr std::optional<T> active_bit_or(
        Mask participants, const Lanes<T> &values) noexcept {
        return _reduce(participants, values, std::bit_or<>{});
    }

    template<typename T>
        requires std::is_integral_v<T>
    [[nodiscard]] static constexpr std::optional<T> active_bit_xor(
        Mask participants, const Lanes<T> &values) noexcept {
        return _reduce(participants, values, std::bit_xor<>{});
    }

    template<typename T>
    [[nodiscard]] static constexpr Lanes<T> prefix_sum(
        Mask participants, const Lanes<T> &values) noexcept {
        return _exclusive_scan(
            participants, values, T{0}, std::plus<>{});
    }

    template<typename T>
    [[nodiscard]] static constexpr Lanes<T> prefix_product(
        Mask participants, const Lanes<T> &values) noexcept {
        return _exclusive_scan(
            participants, values, T{1}, std::multiplies<>{});
    }
};

}// namespace luisa::compute::simd::reference
