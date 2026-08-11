#include <algorithm>

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/special_register.h>

#include "coro_replayable.h"

namespace luisa::compute::xir::detail {

namespace {

[[nodiscard]] bool is_stable_replay_root(const Value *value) noexcept {
    if (value == nullptr) { return true; }
    switch (value->derived_value_tag()) {
        case DerivedValueTag::UNDEFINED:
        case DerivedValueTag::FUNCTION:
        case DerivedValueTag::BASIC_BLOCK:
        case DerivedValueTag::CONSTANT:
        case DerivedValueTag::ARGUMENT:
            return true;
        case DerivedValueTag::SPECIAL_REGISTER: {
            // Dispatch identity is explicitly preserved in the reserved frame
            // header by coro-split. Hardware worker identity (thread/block/warp
            // registers) is not: a resumed path may run on another worker.
            auto tag = static_cast<const SpecialRegister *>(value)
                           ->derived_special_register_tag();
            return tag == DerivedSpecialRegisterTag::DISPATCH_ID ||
                   tag == DerivedSpecialRegisterTag::DISPATCH_SIZE;
        }
        case DerivedValueTag::INSTRUCTION:
            return false;
    }
    return false;
}

[[nodiscard]] bool is_replayable_instruction_kind(
    const Instruction *instruction) noexcept {
    switch (instruction->derived_instruction_tag()) {
        // These instructions are deterministic value computations. Their
        // operands are checked recursively below. In particular, a GEP rooted
        // in local storage is rejected because the alloca operand is not a
        // stable root.
        case DerivedInstructionTag::ARITHMETIC:
        case DerivedInstructionTag::CAST:
        case DerivedInstructionTag::GEP:
            return true;
        default:
            return false;
    }
}

}// namespace

size_t CoroReplayableValueAnalysis::instruction_budget(
    const Type *type) noexcept {
    // One frame value incurs at least one store and one load. Compare that
    // minimum traffic in 32-bit words with the number of XIR value operations
    // needed to rebuild it. Cap the budget so a large aggregate cannot expand
    // an arbitrarily large expression DAG into every continuation.
    constexpr size_t max_instruction_budget = 8u;
    auto words = type == nullptr ? 1u : (type->size() + 3u) / 4u;
    return std::min(
        max_instruction_budget,
        std::max(size_t{1u}, words * size_t{2u}));
}

CoroReplayableValueAnalysis::Entry
CoroReplayableValueAnalysis::_classify(const Value *value) noexcept {
    if (is_stable_replay_root(value)) {
        return Entry{State::REPLAYABLE, 0u};
    }
    if (value == nullptr ||
        value->derived_value_tag() != DerivedValueTag::INSTRUCTION) {
        return Entry{State::NOT_REPLAYABLE, 0u};
    }
    if (auto iter = _cache.find(value); iter != _cache.end()) {
        // A back-edge in the value graph is invalid SSA for replay purposes.
        // Fail closed rather than recursing forever on malformed input.
        return iter->second.state == State::VISITING ?
                   Entry{State::NOT_REPLAYABLE, 0u} :
                   iter->second;
    }

    auto was_inserted = _cache.emplace(
        value, Entry{State::VISITING, 0u}).second;
    LUISA_ASSERT(
        was_inserted,
        "A previously classified replay value must have been found above.");
    auto *instruction = static_cast<const Instruction *>(value);
    auto result = Entry{State::NOT_REPLAYABLE, 0u};
    if (is_replayable_instruction_kind(instruction)) {
        auto cost = size_t{1u};
        auto budget = instruction_budget(instruction->type());
        auto valid = true;
        for (auto *operand_use : instruction->operand_uses()) {
            auto operand = _classify(operand_use->value());
            if (operand.state != State::REPLAYABLE) {
                valid = false;
                break;
            }
            if (cost > budget ||
                operand.instruction_cost > budget - cost) {
                valid = false;
                break;
            }
            cost += operand.instruction_cost;
        }
        if (valid && cost <= budget) {
            result = Entry{State::REPLAYABLE, cost};
        }
    }
    // Operand classification recursively inserts into the same dense hash
    // table and may rehash it. Its iterators are therefore not stable across
    // the recursion. The key itself is an IR pointer and remains stable, so
    // re-establish the iterator only after the recursive fixed point returns.
    auto final_iter = _cache.find(value);
    LUISA_ASSERT(
        final_iter != _cache.end(),
        "The visiting replay value disappeared during classification.");
    final_iter->second = result;
    if (result.state == State::REPLAYABLE) {
        ++_replayable_value_count;
    } else {
        ++_rejected_value_count;
    }
    return result;
}

}// namespace luisa::compute::xir::detail
