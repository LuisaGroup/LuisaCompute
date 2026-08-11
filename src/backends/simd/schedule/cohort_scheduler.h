#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>
#include <vector>

namespace luisa::compute::simd::schedule {

template<size_t Width>
class LaneMask {

    static_assert(Width > 0u && Width <= 128u,
                  "A Luisa logical warp must contain between 1 and 128 lanes.");

public:
    static constexpr auto width = Width;
    static constexpr auto word_count = (Width + 63u) / 64u;

private:
    std::array<uint64_t, word_count> _words{};

private:
    [[nodiscard]] static consteval uint64_t _last_word_mask() noexcept {
        if constexpr (Width % 64u == 0u) {
            return ~uint64_t{0u};
        } else {
            return (uint64_t{1u} << (Width % 64u)) - 1u;
        }
    }

    constexpr void _normalize() noexcept {
        _words.back() &= _last_word_mask();
    }

public:
    constexpr LaneMask() noexcept = default;

    [[nodiscard]] static constexpr LaneMask full() noexcept {
        LaneMask mask;
        for (auto &word : mask._words) { word = ~uint64_t{0u}; }
        mask._normalize();
        return mask;
    }

    [[nodiscard]] static constexpr LaneMask first_n(size_t count) noexcept {
        LaneMask mask;
        if (count >= Width) { return full(); }
        auto full_words = count / 64u;
        for (auto i = 0u; i < full_words; i++) {
            mask._words[i] = ~uint64_t{0u};
        }
        if (auto remainder = count % 64u; remainder != 0u) {
            mask._words[full_words] =
                (uint64_t{1u} << remainder) - 1u;
        }
        return mask;
    }

    [[nodiscard]] static constexpr LaneMask single(size_t lane) noexcept {
        LaneMask mask;
        if (lane < Width) { mask.set(lane); }
        return mask;
    }

    [[nodiscard]] static constexpr LaneMask from_indices(
        std::initializer_list<size_t> lanes) noexcept {
        LaneMask mask;
        for (auto lane : lanes) {
            if (lane < Width) { mask.set(lane); }
        }
        return mask;
    }

    constexpr void set(size_t lane, bool value = true) noexcept {
        if (lane >= Width) { return; }
        auto &word = _words[lane / 64u];
        auto bit = uint64_t{1u} << (lane % 64u);
        if (value) {
            word |= bit;
        } else {
            word &= ~bit;
        }
    }

    [[nodiscard]] constexpr bool test(size_t lane) const noexcept {
        return lane < Width &&
               (_words[lane / 64u] &
                (uint64_t{1u} << (lane % 64u))) != 0u;
    }

    [[nodiscard]] constexpr uint64_t word(size_t index) const noexcept {
        return index < word_count ? _words[index] : 0u;
    }

    [[nodiscard]] constexpr bool any() const noexcept {
        for (auto word : _words) {
            if (word != 0u) { return true; }
        }
        return false;
    }

    [[nodiscard]] constexpr bool none() const noexcept { return !any(); }

    [[nodiscard]] constexpr size_t count() const noexcept {
        auto result = size_t{0u};
        for (auto word : _words) {
            result += static_cast<size_t>(std::popcount(word));
        }
        return result;
    }

    [[nodiscard]] constexpr std::optional<size_t> first() const noexcept {
        for (auto i = 0u; i < word_count; i++) {
            if (auto word = _words[i]; word != 0u) {
                return i * 64u + static_cast<size_t>(std::countr_zero(word));
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] constexpr bool intersects(LaneMask rhs) const noexcept {
        for (auto i = 0u; i < word_count; i++) {
            if ((_words[i] & rhs._words[i]) != 0u) { return true; }
        }
        return false;
    }

    [[nodiscard]] constexpr bool is_subset_of(LaneMask rhs) const noexcept {
        for (auto i = 0u; i < word_count; i++) {
            if ((_words[i] & ~rhs._words[i]) != 0u) { return false; }
        }
        return true;
    }

    template<typename F>
    constexpr void for_each(F &&f) const noexcept {
        for (auto word_index = 0u; word_index < word_count; word_index++) {
            auto remaining = _words[word_index];
            while (remaining != 0u) {
                auto bit = static_cast<size_t>(std::countr_zero(remaining));
                f(word_index * 64u + bit);
                remaining &= remaining - 1u;
            }
        }
    }

    constexpr LaneMask &operator|=(LaneMask rhs) noexcept {
        for (auto i = 0u; i < word_count; i++) {
            _words[i] |= rhs._words[i];
        }
        return *this;
    }

    constexpr LaneMask &operator&=(LaneMask rhs) noexcept {
        for (auto i = 0u; i < word_count; i++) {
            _words[i] &= rhs._words[i];
        }
        return *this;
    }

    constexpr LaneMask &operator^=(LaneMask rhs) noexcept {
        for (auto i = 0u; i < word_count; i++) {
            _words[i] ^= rhs._words[i];
        }
        _normalize();
        return *this;
    }

    friend constexpr LaneMask operator|(LaneMask lhs, LaneMask rhs) noexcept {
        lhs |= rhs;
        return lhs;
    }

    friend constexpr LaneMask operator&(LaneMask lhs, LaneMask rhs) noexcept {
        lhs &= rhs;
        return lhs;
    }

    friend constexpr LaneMask operator^(LaneMask lhs, LaneMask rhs) noexcept {
        lhs ^= rhs;
        return lhs;
    }

    friend constexpr LaneMask operator~(LaneMask value) noexcept {
        for (auto &word : value._words) { word = ~word; }
        value._normalize();
        return value;
    }

    friend constexpr LaneMask operator-(LaneMask lhs, LaneMask rhs) noexcept {
        return lhs & ~rhs;
    }

    friend constexpr bool operator==(LaneMask, LaneMask) noexcept = default;
};

// A continuation identifies a dynamic Schedule IR point. The loop epoch is
// deliberately part of the key: lanes at the same static PC in different
// loop iterations must not be combined into one collective instance.
struct Continuation {
    uint32_t pc{0u};
    uint32_t convergence_token{0u};
    uint32_t loop_epoch{0u};

    friend constexpr bool operator==(Continuation,
                                     Continuation) noexcept = default;
};

enum struct SchedulingPolicy {
    depth_first,
    largest_cohort,
};

enum struct EnqueueResult {
    inserted,
    merged,
    empty,
    conflict,
};

enum struct ConvergenceResult {
    waiting,
    released,
    empty,
    missing,
    conflict,
};

template<size_t Width>
struct Cohort {
    Continuation continuation{};
    LaneMask<Width> mask{};
    uint64_t sequence{0u};
};

// Dependency-light semantic model for the dynamic cohort scheduler. This is
// intentionally not tied to XIR or LLVM: it exercises continuation identity,
// convergence, termination, and policy independence before code generation is
// introduced.
template<size_t Width>
class CohortScheduler {

public:
    using Mask = LaneMask<Width>;
    using CohortType = Cohort<Width>;

private:
    struct ConvergenceGate {
        Continuation continuation{};
        Mask expected{};
        Mask arrived{};
        uint64_t sequence{0u};
    };

    SchedulingPolicy _policy;
    Mask _live;
    Mask _queued;
    Mask _parked;
    std::vector<CohortType> _ready;
    std::vector<ConvergenceGate> _gates;
    uint64_t _next_sequence{0u};

private:
    [[nodiscard]] auto _find_ready(Continuation continuation) noexcept {
        return std::find_if(
            _ready.begin(), _ready.end(),
            [continuation](const auto &cohort) noexcept {
                return cohort.continuation == continuation;
            });
    }

    [[nodiscard]] auto _find_gate(Continuation continuation) noexcept {
        return std::find_if(
            _gates.begin(), _gates.end(),
            [continuation](const auto &gate) noexcept {
                return gate.continuation == continuation;
            });
    }

    [[nodiscard]] EnqueueResult _enqueue(Continuation continuation,
                                         Mask mask) noexcept {
        mask &= _live;
        if (mask.none()) { return EnqueueResult::empty; }
        if (mask.intersects(_parked)) { return EnqueueResult::conflict; }

        auto same = _find_ready(continuation);
        auto same_mask = same == _ready.end() ? Mask{} : same->mask;
        if (mask.intersects(same_mask)) {
            return EnqueueResult::conflict;
        }
        if (mask.intersects(_queued - same_mask)) {
            return EnqueueResult::conflict;
        }
        if (same != _ready.end()) {
            same->mask |= mask;
            _queued |= mask;
            return EnqueueResult::merged;
        }
        _ready.emplace_back(CohortType{
            .continuation = continuation,
            .mask = mask,
            .sequence = _next_sequence++,
        });
        _queued |= mask;
        return EnqueueResult::inserted;
    }

    void _release_satisfied_gates() noexcept {
        auto changed = true;
        while (changed) {
            changed = false;
            for (auto i = size_t{0u}; i < _gates.size(); i++) {
                auto expected_live = _gates[i].expected & _live;
                if (_gates[i].arrived != expected_live) { continue; }
                auto continuation = _gates[i].continuation;
                auto arrived = _gates[i].arrived;
                _parked = _parked - arrived;
                _gates.erase(_gates.begin() + static_cast<std::ptrdiff_t>(i));
                if (arrived.any()) {
                    static_cast<void>(_enqueue(continuation, arrived));
                }
                changed = true;
                break;
            }
        }
    }

public:
    explicit CohortScheduler(
        Mask initial_live = Mask::full(),
        SchedulingPolicy policy = SchedulingPolicy::depth_first) noexcept
        : _policy{policy}, _live{initial_live} {}

    [[nodiscard]] EnqueueResult enqueue(Continuation continuation,
                                        Mask mask) noexcept {
        return _enqueue(continuation, mask);
    }

    // Declares the lanes that must reach (or terminate before) a dynamic
    // reconvergence point. A continuation key identifies exactly one dynamic
    // instance, including its loop epoch.
    [[nodiscard]] bool declare_convergence(Continuation continuation,
                                           Mask expected) noexcept {
        expected &= _live;
        if (_find_gate(continuation) != _gates.end()) { return false; }
        _gates.emplace_back(ConvergenceGate{
            .continuation = continuation,
            .expected = expected,
            .arrived = {},
            .sequence = _next_sequence++,
        });
        _release_satisfied_gates();
        return true;
    }

    [[nodiscard]] ConvergenceResult arrive(Continuation continuation,
                                           Mask mask) noexcept {
        auto gate = _find_gate(continuation);
        if (gate == _gates.end()) { return ConvergenceResult::missing; }
        mask &= _live;
        if (mask.none()) { return ConvergenceResult::empty; }
        if (!mask.is_subset_of(gate->expected) ||
            mask.intersects(_queued) ||
            mask.intersects(_parked)) {
            return ConvergenceResult::conflict;
        }
        gate->arrived |= mask;
        _parked |= mask;
        auto expected_live = gate->expected & _live;
        if (gate->arrived != expected_live) {
            return ConvergenceResult::waiting;
        }
        auto arrived = gate->arrived;
        _parked = _parked - arrived;
        _gates.erase(gate);
        auto result = _enqueue(continuation, arrived);
        return result == EnqueueResult::conflict ?
                   ConvergenceResult::conflict :
                   ConvergenceResult::released;
    }

    // Termination is only valid for a cohort currently owned by the caller.
    // A queued or parked lane cannot be terminated behind the scheduler's
    // back; doing so would hide a control-flow bug in the lowering.
    [[nodiscard]] bool terminate(Mask mask) noexcept {
        mask &= _live;
        if (mask.intersects(_queued) || mask.intersects(_parked)) {
            return false;
        }
        _live = _live - mask;
        _release_satisfied_gates();
        return true;
    }

    [[nodiscard]] std::optional<CohortType> take() noexcept {
        if (_ready.empty()) { return std::nullopt; }
        auto selected = size_t{0u};
        for (auto i = size_t{1u}; i < _ready.size(); i++) {
            auto choose = false;
            if (_policy == SchedulingPolicy::depth_first) {
                choose = _ready[i].sequence > _ready[selected].sequence;
            } else {
                auto lhs_count = _ready[i].mask.count();
                auto rhs_count = _ready[selected].mask.count();
                choose = lhs_count > rhs_count ||
                         (lhs_count == rhs_count &&
                          _ready[i].sequence < _ready[selected].sequence);
            }
            if (choose) { selected = i; }
        }
        auto cohort = _ready[selected];
        _queued = _queued - cohort.mask;
        _ready.erase(_ready.begin() +
                     static_cast<std::ptrdiff_t>(selected));
        return cohort;
    }

    [[nodiscard]] auto policy() const noexcept { return _policy; }
    [[nodiscard]] auto live_mask() const noexcept { return _live; }
    [[nodiscard]] auto queued_mask() const noexcept { return _queued; }
    [[nodiscard]] auto parked_mask() const noexcept { return _parked; }
    [[nodiscard]] auto ready_count() const noexcept { return _ready.size(); }
    [[nodiscard]] auto pending_convergence_count() const noexcept {
        return _gates.size();
    }
    [[nodiscard]] bool has_ready() const noexcept { return !_ready.empty(); }
    [[nodiscard]] bool complete() const noexcept {
        return _live.none() && _ready.empty() && _gates.empty();
    }
    [[nodiscard]] bool stalled() const noexcept {
        return _live.any() && _ready.empty() && !_gates.empty();
    }
};

}// namespace luisa::compute::simd::schedule
