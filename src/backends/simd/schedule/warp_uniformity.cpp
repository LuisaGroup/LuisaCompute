#include "warp_uniformity.h"

#include <vector>

#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/thread_group.h>
#include <luisa/xir/special_register.h>

namespace luisa::compute::simd::schedule {

void WarpUniformityAnalysis::clear() noexcept {
    _function = nullptr;
    _states.clear();
}

WarpUniformityAnalysis::State WarpUniformityAnalysis::_state(
    const xir::Value *value) const noexcept {
    if (value == nullptr) { return State::varying; }
    if (auto iter = _states.find(value); iter != _states.end()) {
        return iter->second;
    }
    using Tag = xir::DerivedValueTag;
    switch (value->derived_value_tag()) {
        case Tag::CONSTANT: return State::warp_uniform;
        case Tag::FUNCTION: return State::warp_uniform;
        case Tag::BASIC_BLOCK: return State::warp_uniform;
        case Tag::UNDEFINED: return State::varying;
        case Tag::ARGUMENT: {
            auto argument = static_cast<const xir::Argument *>(value);
            return _function != nullptr &&
                           _function->isa<xir::KernelFunction>() &&
                           argument->parent_function() == _function ?
                       State::warp_uniform :
                       State::varying;
        }
        case Tag::SPECIAL_REGISTER: {
            using S = xir::DerivedSpecialRegisterTag;
            switch (static_cast<const xir::SpecialRegister *>(value)
                        ->derived_special_register_tag()) {
                case S::BLOCK_ID:
                case S::KERNEL_ID:
                case S::BLOCK_SIZE:
                case S::WARP_SIZE:
                case S::DISPATCH_SIZE: return State::warp_uniform;
                case S::THREAD_ID:
                case S::WARP_LANE_ID:
                case S::DISPATCH_ID:
                case S::RASTER_OBJECT_ID:
                case S::RASTER_BARYCENTRICS: return State::varying;
            }
            return State::varying;
        }
        case Tag::INSTRUCTION: return State::unknown;
    }
    return State::varying;
}

void WarpUniformityAnalysis::analyze(
    const xir::Function *function) noexcept {
    clear();
    _function = function;
    if (function == nullptr || function->definition() == nullptr ||
        function->definition()->body_block() == nullptr) {
        return;
    }

    std::vector<const xir::Instruction *> instructions;
    function->definition()->traverse_instructions(
        [&](const xir::Instruction *instruction) noexcept {
            instructions.emplace_back(instruction);
        });
    auto argument_count = size_t{0u};
    for (auto *argument : function->arguments()) {
        static_cast<void>(argument);
        ++argument_count;
    }
    _states.reserve(argument_count + instructions.size());
    for (auto *argument : function->arguments()) {
        _states.emplace(
            argument,
            function->isa<xir::KernelFunction>() ?
                State::warp_uniform :
                State::varying);
    }

    struct Pending {
        uint32_t unresolved_count{0u};
        State state{State::warp_uniform};
    };
    std::unordered_map<const xir::Instruction *, Pending> pending;
    std::unordered_map<
        const xir::Value *, std::vector<const xir::Instruction *>>
        dependents;
    std::vector<std::pair<const xir::Instruction *, State>> ready;
    pending.reserve(instructions.size());
    dependents.reserve(instructions.size());
    ready.reserve(instructions.size());
    auto join_state = [](State lhs, State rhs) noexcept {
        if (lhs == State::unknown) { return rhs; }
        if (rhs == State::unknown) { return lhs; }
        return static_cast<uint32_t>(lhs) >= static_cast<uint32_t>(rhs) ?
                   lhs :
                   rhs;
    };
    auto resolve = [&](const xir::Instruction *instruction,
                       State state) noexcept {
        if (_states.emplace(instruction, state).second) {
            ready.emplace_back(instruction, state);
        }
    };

    std::vector<const xir::Value *> dependencies;
    for (auto *instruction : instructions) {
        dependencies.clear();
        auto has_immediate = false;
        auto immediate = State::varying;
        auto floor = State::warp_uniform;
        auto set_immediate = [&](State state) noexcept {
            has_immediate = true;
            immediate = state;
        };

        using Tag = xir::DerivedInstructionTag;
        switch (instruction->derived_instruction_tag()) {
            case Tag::ARITHMETIC:
            case Tag::CAST:
            case Tag::GEP:
            case Tag::RESOURCE_QUERY:
                for (auto *operand_use : instruction->operand_uses()) {
                    dependencies.emplace_back(operand_use->value());
                }
                break;
            case Tag::PHI: {
                auto *phi = static_cast<const xir::PhiInst *>(instruction);
                if (phi->incoming_count() == 0u) {
                    set_immediate(State::varying);
                    break;
                }
                auto *first = phi->incoming(0u).value;
                auto same_value = true;
                for (auto i = 1u; i < phi->incoming_count(); i++) {
                    same_value &= phi->incoming(i).value == first;
                }
                // Selecting distinct definitions may vary even if each one is
                // independently uniform. A later control-dependence analysis
                // may refine this conservative result.
                if (!same_value) {
                    set_immediate(State::varying);
                } else {
                    dependencies.emplace_back(first);
                }
                break;
            }
            case Tag::THREAD_GROUP: {
                auto *thread_group =
                    static_cast<const xir::ThreadGroupInst *>(instruction);
                using Op = xir::ThreadGroupOp;
                switch (thread_group->op()) {
                    case Op::WARP_FIRST_ACTIVE_LANE:
                    case Op::WARP_ACTIVE_ALL_EQUAL:
                    case Op::WARP_ACTIVE_BIT_AND:
                    case Op::WARP_ACTIVE_BIT_OR:
                    case Op::WARP_ACTIVE_BIT_XOR:
                    case Op::WARP_ACTIVE_COUNT_BITS:
                    case Op::WARP_ACTIVE_MAX:
                    case Op::WARP_ACTIVE_MIN:
                    case Op::WARP_ACTIVE_PRODUCT:
                    case Op::WARP_ACTIVE_SUM:
                    case Op::WARP_ACTIVE_ALL:
                    case Op::WARP_ACTIVE_ANY:
                    case Op::WARP_ACTIVE_BIT_MASK:
                    case Op::WARP_READ_FIRST_ACTIVE_LANE:
                    case Op::SHADER_EXECUTION_REORDER:
                    case Op::SYNCHRONIZE_BLOCK:
                        // The result is scalar inside this dynamic cohort, but
                        // a sibling path or another loop epoch may observe a
                        // different active set and therefore a different value.
                        set_immediate(State::cohort_uniform);
                        break;
                    case Op::WARP_READ_LANE:
                        if (thread_group->operand_count() < 2u) {
                            set_immediate(State::varying);
                        } else {
                            // A non-varying source-lane index produces one
                            // broadcast value for the current cohort. It is
                            // not warp-global because the participating set
                            // and source value can differ by dynamic instance.
                            floor = State::cohort_uniform;
                            dependencies.emplace_back(
                                thread_group->operand(1u));
                        }
                        break;
                    case Op::WARP_IS_FIRST_ACTIVE_LANE:
                    case Op::WARP_PREFIX_COUNT_BITS:
                    case Op::WARP_PREFIX_SUM:
                    case Op::WARP_PREFIX_PRODUCT:
                    case Op::RASTER_QUAD_DDX:
                    case Op::RASTER_QUAD_DDY:
                        set_immediate(State::varying);
                        break;
                }
                break;
            }

            // Lane-local values and mutable observations are varying.
            case Tag::ALLOCA:
            case Tag::LOAD:
            case Tag::ATOMIC:
            case Tag::RESOURCE_READ:
            case Tag::RAY_QUERY_LOOP:
            case Tag::RAY_QUERY_DISPATCH:
            case Tag::RAY_QUERY_OBJECT_READ:
            case Tag::RAY_QUERY_OBJECT_WRITE:
            case Tag::RAY_QUERY_PIPELINE:
            case Tag::CLOCK:
            case Tag::CALL:
            case Tag::AUTODIFF_SCOPE:
            case Tag::AUTODIFF_INTRINSIC:
                set_immediate(State::varying);
                break;

            // Terminators and side-effect-only instructions do not form SIMD
            // data values; their classification is ignored by lowering.
            case Tag::IF:
            case Tag::SWITCH:
            case Tag::INDEXED_BRANCH:
            case Tag::LOOP:
            case Tag::SIMPLE_LOOP:
            case Tag::BRANCH:
            case Tag::CONDITIONAL_BRANCH:
            case Tag::UNREACHABLE:
            case Tag::BREAK:
            case Tag::CONTINUE:
            case Tag::RETURN:
            case Tag::RASTER_DISCARD:
            case Tag::CORO_SUSPEND:
            case Tag::CORO_RESUME:
            case Tag::CORO_TERMINATE:
            case Tag::STORE:
            case Tag::RESOURCE_WRITE:
            case Tag::PRINT:
            case Tag::DEBUG_BREAK:
            case Tag::ASSERT:
            case Tag::ASSUME:
            case Tag::OUTLINE:
                set_immediate(State::warp_uniform);
                break;
        }

        if (has_immediate) {
            resolve(instruction, immediate);
            continue;
        }
        auto resolved_state = floor;
        auto has_varying = false;
        for (auto *dependency : dependencies) {
            auto dependency_state = _state(dependency);
            if (dependency_state == State::varying) {
                has_varying = true;
                break;
            }
            if (dependency_state != State::unknown) {
                resolved_state = join_state(
                    resolved_state, dependency_state);
            }
        }
        if (has_varying) {
            resolve(instruction, State::varying);
            continue;
        }
        auto unresolved_count = uint32_t{0u};
        for (auto *dependency : dependencies) {
            if (_state(dependency) == State::unknown) {
                ++unresolved_count;
                dependents[dependency].emplace_back(instruction);
            }
        }
        if (unresolved_count == 0u) {
            resolve(instruction, resolved_state);
        } else {
            pending.emplace(
                instruction,
                Pending{unresolved_count, resolved_state});
        }
    }

    for (auto ready_index = size_t{0u};
         ready_index < ready.size(); ready_index++) {
        auto [resolved_value, resolved_state] = ready[ready_index];
        auto iter = dependents.find(resolved_value);
        if (iter == dependents.end()) { continue; }
        for (auto *dependent : iter->second) {
            auto pending_iter = pending.find(dependent);
            if (pending_iter == pending.end()) { continue; }
            if (resolved_state == State::varying) {
                pending.erase(pending_iter);
                resolve(dependent, State::varying);
            } else {
                pending_iter->second.state = join_state(
                    pending_iter->second.state, resolved_state);
                if (--pending_iter->second.unresolved_count == 0u) {
                    auto state = pending_iter->second.state;
                    pending.erase(pending_iter);
                    resolve(dependent, state);
                }
            }
        }
    }
    // Cycles through PHIs or other facts that remain unproven are varying.
    for (auto instruction : instructions) {
        _states.try_emplace(instruction, State::varying);
    }
}

ValueClass WarpUniformityAnalysis::classify(
    const xir::Value *value) const noexcept {
    switch (_state(value)) {
        case State::warp_uniform: return ValueClass::warp_uniform;
        case State::cohort_uniform: return ValueClass::cohort_uniform;
        case State::unknown:
        case State::varying: return ValueClass::varying;
    }
    return ValueClass::varying;
}

}// namespace luisa::compute::simd::schedule
