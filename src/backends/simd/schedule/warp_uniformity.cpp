#include "warp_uniformity.h"

#include <deque>
#include <vector>

#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/indexed_branch.h>
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
        // Every reachable instruction is seeded before dependency
        // propagation begins. An instruction absent from the table is
        // unreachable, foreign, or malformed and must stay conservative.
        case Tag::INSTRUCTION: return State::varying;
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

    // Values form a monotone lattice:
    //
    //   warp_uniform < cohort_uniform < varying
    //
    // Start pure SSA cycles optimistically and only move downward. This proves
    // uniform loop-carried expressions without whole-function rescans. Each
    // value changes class at most twice and each use is therefore visited at
    // most twice after graph construction.
    std::vector<const xir::BasicBlock *> blocks;
    function->definition()->traverse_basic_blocks(
        xir::BasicBlockTraversalOrder::REVERSE_POST_ORDER,
        [&](const xir::BasicBlock *block) noexcept {
            blocks.emplace_back(block);
        });
    std::unordered_map<const xir::BasicBlock *, size_t> block_indices;
    block_indices.reserve(blocks.size());
    for (auto i = size_t{0u}; i < blocks.size(); i++) {
        block_indices.emplace(blocks[i], i);
    }

    std::vector<const xir::Instruction *> instructions;
    for (auto *block : blocks) {
        block->traverse_instructions(
            [&](const xir::Instruction *instruction) noexcept {
                instructions.emplace_back(instruction);
            });
    }
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

    auto join_state = [](State lhs, State rhs) noexcept {
        if (lhs == State::unknown) { return rhs; }
        if (rhs == State::unknown) { return lhs; }
        return static_cast<uint32_t>(lhs) >= static_cast<uint32_t>(rhs) ?
                   lhs :
                   rhs;
    };

    struct Rule {
        State floor{State::warp_uniform};
        std::vector<const xir::Value *> dependencies;
        bool distinct_phi{false};
    };
    std::vector<Rule> rules(instructions.size());
    std::unordered_map<
        const xir::Value *, std::vector<size_t>>
        dependents;
    dependents.reserve(instructions.size());

    for (auto instruction_index = size_t{0u};
         instruction_index < instructions.size(); instruction_index++) {
        auto *instruction = instructions[instruction_index];
        auto &rule = rules[instruction_index];
        auto set_immediate = [&](State state) noexcept {
            rule.floor = state;
            rule.dependencies.clear();
        };
        auto add_all_operands = [&] {
            for (auto *operand_use : instruction->operand_uses()) {
                rule.dependencies.emplace_back(operand_use->value());
            }
        };

        using Tag = xir::DerivedInstructionTag;
        switch (instruction->derived_instruction_tag()) {
            case Tag::ARITHMETIC:
            case Tag::CAST:
            case Tag::GEP:
            case Tag::RESOURCE_QUERY:
                add_all_operands();
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
                if (same_value) {
                    rule.dependencies.emplace_back(first);
                    break;
                }
                rule.distinct_phi = true;
                add_all_operands();

                // A PHI selected once by warp-uniform acyclic control can be
                // warp-global. A recurrent PHI changes by dynamic loop epoch,
                // so even a lane-coherent loop needs lane-wise state whenever
                // it crosses a scheduler suspension.
                auto *phi_block = instruction->parent_block();
                auto phi_block_iter = block_indices.find(phi_block);
                if (phi_block_iter == block_indices.end()) {
                    rule.floor = State::varying;
                    break;
                }
                for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
                    auto incoming_iter = block_indices.find(
                        phi->incoming(i).block);
                    if (incoming_iter != block_indices.end() &&
                        incoming_iter->second >= phi_block_iter->second) {
                        rule.floor = State::cohort_uniform;
                        break;
                    }
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
                            rule.floor = State::cohort_uniform;
                            rule.dependencies.emplace_back(
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
        _states.emplace(instruction, rule.floor);
        for (auto *dependency : rule.dependencies) {
            dependents[dependency].emplace_back(instruction_index);
        }
    }

    std::deque<size_t> value_worklist;
    std::vector<uint8_t> value_queued(instructions.size(), uint8_t{0u});
    auto enqueue_value = [&](size_t index) noexcept {
        if (value_queued[index] == 0u) {
            value_queued[index] = 1u;
            value_worklist.emplace_back(index);
        }
    };
    auto degrade_value = [&](size_t index, State state) noexcept {
        auto *instruction = instructions[index];
        auto old_state = _states.at(instruction);
        auto new_state = join_state(old_state, state);
        if (new_state != old_state) {
            _states[instruction] = new_state;
            enqueue_value(index);
            return true;
        }
        return false;
    };

    // Fold external facts and the optimistic initial states into every rule
    // once. Later propagation is incremental over use edges.
    for (auto instruction_index = size_t{0u};
         instruction_index < instructions.size(); instruction_index++) {
        auto state = rules[instruction_index].floor;
        for (auto *dependency : rules[instruction_index].dependencies) {
            state = join_state(state, _state(dependency));
        }
        _states[instructions[instruction_index]] = state;
        if (state != State::warp_uniform) {
            enqueue_value(instruction_index);
        }
    }

    // Track whether every static path decision reaching a block is provably
    // warp-uniform. A single non-uniform incoming edge permanently downgrades
    // the block, so propagation touches every CFG edge at most once.
    std::vector<std::vector<size_t>> successors(blocks.size());
    for (auto block_index = size_t{0u}; block_index < blocks.size();
         block_index++) {
        blocks[block_index]->traverse_successors(
            false, [&](const xir::BasicBlock *successor) noexcept {
                if (auto iter = block_indices.find(successor);
                    iter != block_indices.end()) {
                    successors[block_index].emplace_back(iter->second);
                }
            });
    }
    std::vector<std::vector<size_t>> block_phis(blocks.size());
    for (auto instruction_index = size_t{0u};
         instruction_index < instructions.size(); instruction_index++) {
        if (!rules[instruction_index].distinct_phi) { continue; }
        auto iter = block_indices.find(
            instructions[instruction_index]->parent_block());
        if (iter != block_indices.end()) {
            block_phis[iter->second].emplace_back(instruction_index);
        }
    }
    std::vector<uint8_t> warp_uniform_path(blocks.size(), uint8_t{1u});
    std::deque<size_t> block_worklist;
    auto degrade_block = [&](size_t index) noexcept {
        if (warp_uniform_path[index] != 0u) {
            warp_uniform_path[index] = 0u;
            block_worklist.emplace_back(index);
        }
    };
    std::unordered_map<const xir::Value *, std::vector<size_t>>
        selector_blocks;
    selector_blocks.reserve(blocks.size());
    for (auto block_index = size_t{0u}; block_index < blocks.size();
         block_index++) {
        auto *terminator = blocks[block_index]->terminator();
        const xir::Value *selector = nullptr;
        if (terminator != nullptr) {
            using Tag = xir::DerivedInstructionTag;
            switch (terminator->derived_instruction_tag()) {
                case Tag::CONDITIONAL_BRANCH:
                    selector = static_cast<
                        const xir::ConditionalBranchInst *>(terminator)
                                   ->condition();
                    break;
                case Tag::INDEXED_BRANCH:
                    selector = static_cast<
                        const xir::IndexedBranchInst *>(terminator)
                                   ->value();
                    break;
                default: break;
            }
        }
        if (selector != nullptr) {
            selector_blocks[selector].emplace_back(block_index);
            if (_state(selector) != State::warp_uniform) {
                for (auto successor : successors[block_index]) {
                    degrade_block(successor);
                }
            }
        } else if (successors[block_index].size() > 1u) {
            // Structured/multi-way control should be destructured before this
            // analysis. Stay conservative if an unrecognized split remains.
            for (auto successor : successors[block_index]) {
                degrade_block(successor);
            }
        }
    }

    while (!value_worklist.empty() || !block_worklist.empty()) {
        while (!value_worklist.empty()) {
            auto instruction_index = value_worklist.front();
            value_worklist.pop_front();
            value_queued[instruction_index] = 0u;
            auto *instruction = instructions[instruction_index];
            auto state = _states.at(instruction);
            if (auto iter = dependents.find(instruction);
                iter != dependents.end()) {
                for (auto dependent : iter->second) {
                    degrade_value(dependent, state);
                }
            }
            if (state != State::warp_uniform) {
                if (auto iter = selector_blocks.find(instruction);
                    iter != selector_blocks.end()) {
                    for (auto source : iter->second) {
                        for (auto successor : successors[source]) {
                            degrade_block(successor);
                        }
                    }
                }
            }
        }
        if (!block_worklist.empty()) {
            auto block_index = block_worklist.front();
            block_worklist.pop_front();
            for (auto phi : block_phis[block_index]) {
                degrade_value(phi, State::varying);
            }
            for (auto successor : successors[block_index]) {
                degrade_block(successor);
            }
        }
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
