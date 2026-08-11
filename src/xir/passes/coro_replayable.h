#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::xir {

class Value;

namespace detail {

// Classifies expression DAGs that may be reconstructed after a coroutine
// suspension instead of being carried in the frame. The analysis is deliberately
// stricter than generic purity: only value computations rooted entirely in
// constants, stable arguments, or scheduler-preserved special registers qualify.
// Resource operations, loads, calls, clocks, and ray queries fail closed.
class CoroReplayableValueAnalysis {

private:
    enum class State : uint8_t {
        VISITING,
        NOT_REPLAYABLE,
        REPLAYABLE,
    };

    struct Entry {
        State state{State::VISITING};
        size_t instruction_cost{0u};
    };

    luisa::unordered_map<const Value *, Entry> _cache;
    size_t _replayable_value_count{0u};
    size_t _rejected_value_count{0u};

private:
    [[nodiscard]] Entry _classify(const Value *value) noexcept;

public:
    [[nodiscard]] static size_t instruction_budget(
        const Type *type) noexcept;

    [[nodiscard]] bool detect(const Value *value) noexcept {
        return _classify(value).state == State::REPLAYABLE;
    }

    [[nodiscard]] size_t instruction_cost(const Value *value) noexcept {
        auto result = _classify(value);
        return result.state == State::REPLAYABLE ?
                   result.instruction_cost :
                   0u;
    }

    [[nodiscard]] size_t replayable_value_count() const noexcept {
        return _replayable_value_count;
    }

    [[nodiscard]] size_t rejected_value_count() const noexcept {
        return _rejected_value_count;
    }
};

}// namespace detail
}// namespace luisa::compute::xir
