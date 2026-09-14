#include "warp_uniformity.h"

#include <algorithm>
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
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/thread_group.h>
#include <luisa/xir/special_register.h>

#include "../../../xir/passes/natural_loop.h"

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
                case S::DISPATCH_SIZE:
                case S::RASTER_BASE_INSTANCE: return State::warp_uniform;
                case S::THREAD_ID:
                case S::WARP_LANE_ID:
                case S::DISPATCH_ID:
                case S::RASTER_OBJECT_ID:
                case S::RASTER_BARYCENTRICS:
                case S::RASTER_FRONT_FACING: return State::varying;
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
    const xir::Function *function,
    std::span<const ValueClass> parameter_value_classes) noexcept {
    _analyze(function, parameter_value_classes, {}, false);
}

void WarpUniformityAnalysis::analyze(
    const xir::Function *function,
    std::span<const ValueClass> parameter_value_classes,
    std::span<const xir::NaturalLoop> natural_loops) noexcept {
    _analyze(function, parameter_value_classes, natural_loops, true);
}

void WarpUniformityAnalysis::_analyze(
    const xir::Function *function,
    std::span<const ValueClass> parameter_value_classes,
    std::span<const xir::NaturalLoop> natural_loops,
    bool supplied_natural_loops) noexcept {
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

    // NaturalLoop consumes a plain CFG. Standalone analysis is also used on
    // incomplete/structured input: do not introduce a terminator assertion or
    // pretend that an unavailable loop analysis proves no epoch crossings.
    auto plain_cfg = true;
    for (auto *block : blocks) {
        if (!block->is_terminated()) {
            plain_cfg = false;
            break;
        }
        using Tag = xir::DerivedInstructionTag;
        switch (block->terminator()->derived_instruction_tag()) {
            case Tag::BRANCH:
            case Tag::CONDITIONAL_BRANCH:
            case Tag::INDEXED_BRANCH:
            case Tag::RETURN:
            case Tag::UNREACHABLE: break;
            default: plain_cfg = false; break;
        }
        if (!plain_cfg) { break; }
    }
    luisa::vector<xir::NaturalLoop> discovered_loops;
    if (plain_cfg && !supplied_natural_loops) {
        auto *mutable_function = const_cast<xir::Function *>(function);
        auto dom_tree = xir::compute_dom_tree(
            mutable_function, {.compute_dominance_frontiers = false});
        // Lowering rejects irreducible CFGs before calling the overload with
        // supplied loops. Standalone analysis instead stays conservative if a
        // reverse-postorder retreating edge is not a natural back-edge.
        for (auto i = size_t{0u}; i < blocks.size(); i++) {
            blocks[i]->traverse_successors(false, [&](const xir::BasicBlock *successor) noexcept {
                if (auto iter = block_indices.find(successor);
                    iter != block_indices.end() && iter->second <= i &&
                    !dom_tree.dominates(const_cast<xir::BasicBlock *>(successor),
                                        const_cast<xir::BasicBlock *>(blocks[i]))) {
                    plain_cfg = false;
                }
            });
        }
        if (plain_cfg) {
            discovered_loops = xir::discover_natural_loops(
                mutable_function->definition(), dom_tree);
            natural_loops = {discovered_loops.data(), discovered_loops.size()};
        }
    }

    std::vector<const xir::Instruction *> instructions;
    for (auto *block : blocks) {
        block->traverse_instructions(
            [&](const xir::Instruction *instruction) noexcept {
                instructions.emplace_back(instruction);
            });
    }
    std::unordered_map<const xir::Value *, size_t> instruction_indices;
    instruction_indices.reserve(instructions.size());
    for (auto i = size_t{0u}; i < instructions.size(); i++) {
        instruction_indices.emplace(instructions[i], i);
    }
    std::vector<std::vector<size_t>> containing_loops(blocks.size());
    if (plain_cfg) {
        for (auto i = size_t{0u}; i < natural_loops.size(); i++) {
            auto add_membership = [&](const xir::BasicBlock *block) noexcept {
                if (auto iter = block_indices.find(block); iter != block_indices.end()) {
                    containing_loops[iter->second].emplace_back(i);
                }
            };
            add_membership(natural_loops[i].header);
            for (auto *block : natural_loops[i].body_blocks) { add_membership(block); }
        }
    }
    std::vector<uint8_t> escapes_loop_epoch(instructions.size(), plain_cfg ? uint8_t{0u} : uint8_t{1u});
    auto contains = [&](size_t loop, const xir::BasicBlock *block) noexcept {
        auto iter = block_indices.find(block);
        if (iter == block_indices.end()) { return false; }
        auto &&memberships = containing_loops[iter->second];
        return std::binary_search(memberships.cbegin(), memberships.cend(), loop);
    };
    auto mark_escape = [&](const xir::Value *value, const xir::BasicBlock *use_block,
                           const xir::BasicBlock *incoming_block, bool phi_use) noexcept {
        auto iter = instruction_indices.find(value);
        if (iter == instruction_indices.end() || escapes_loop_epoch[iter->second] != 0u) { return; }
        auto definition = block_indices.find(instructions[iter->second]->parent_block());
        if (definition == block_indices.end()) {
            escapes_loop_epoch[iter->second] = 1u;
            return;
        }
        for (auto loop : containing_loops[definition->second]) {
            if (!contains(loop, use_block) || (phi_use && !contains(loop, incoming_block))) {
                escapes_loop_epoch[iter->second] = 1u;
                break;
            }
        }
    };
    if (plain_cfg) {
        for (auto *instruction : instructions) {
            if (instruction->isa<xir::PhiInst>()) {
                auto *phi = static_cast<const xir::PhiInst *>(instruction);
                for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
                    auto incoming = phi->incoming(i);
                    // An exit PHI consumes a snapshot across the edge even
                    // when its incoming predecessor is still in the loop.
                    mark_escape(incoming.value, phi->parent_block(), incoming.block, true);
                }
            } else {
                for (auto *use : instruction->operand_uses()) {
                    mark_escape(use->value(), instruction->parent_block(), nullptr, false);
                }
            }
        }
    }
    auto normalize_state = [&](size_t index, State state) noexcept {
        // Cohort equality belongs to one dynamic epoch. Different lanes may
        // preserve different last values at exit. Keep truly warp-stable
        // values scalar, and keep nonescaping cohort-local fast paths intact.
        return escapes_loop_epoch[index] != 0u && state == State::cohort_uniform ?
                   State::varying :
                   state;
    };
    auto argument_count = size_t{0u};
    for (auto *argument : function->arguments()) {
        static_cast<void>(argument);
        ++argument_count;
    }
    _states.reserve(argument_count + instructions.size());
    auto argument_index = size_t{0u};
    for (auto *argument : function->arguments()) {
        auto state = function->isa<xir::KernelFunction>() ?
                         State::warp_uniform :
                         State::varying;
        if (!parameter_value_classes.empty()) {
            switch (parameter_value_classes[argument_index]) {
                case ValueClass::warp_uniform:
                    state = State::warp_uniform;
                    break;
                case ValueClass::cohort_uniform:
                    state = State::cohort_uniform;
                    break;
                case ValueClass::varying:
                    state = State::varying;
                    break;
                case ValueClass::mask:
                case ValueClass::token:
                    // Parameters cannot carry scheduler-only classes. The
                    // lowering preflight rejects these before analysis.
                    state = State::varying;
                    break;
            }
        }
        _states.emplace(
            argument, state);
        argument_index++;
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
                add_all_operands();
                break;
            case Tag::RESOURCE_QUERY: {
                auto op = static_cast<const xir::ResourceQueryInst *>(
                              instruction)
                              ->op();
                if (op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL ||
                    op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY ||
                    op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR ||
                    op == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR) {
                    // A query is mutable lane-local state. Even a uniform ray
                    // must receive one state object per physical lane because
                    // its handlers may make varying commit decisions.
                    set_immediate(State::varying);
                } else {
                    add_all_operands();
                }
                break;
            }
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
        _states.emplace(instruction, normalize_state(instruction_index, rule.floor));
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
        auto new_state = normalize_state(index, join_state(old_state, state));
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
        state = normalize_state(instruction_index, state);
        _states[instructions[instruction_index]] = state;
        if (state != State::warp_uniform) {
            enqueue_value(instruction_index);
        }
    }

    // Track the strongest control class reaching each block. Cohort-uniform
    // control does not split the current cohort, so a PHI selected by such a
    // path remains scalar inside that cohort. Varying control still degrades
    // the PHI to lane-wise state. This is the same monotone lattice as value
    // propagation, and every block changes class at most twice.
    std::vector<std::vector<size_t>> successors(blocks.size());
    for (auto block_index = size_t{0u}; block_index < blocks.size();
         block_index++) {
        if (!blocks[block_index]->is_terminated()) { continue; }
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
    std::vector<State> control_states(
        blocks.size(), State::warp_uniform);
    std::deque<size_t> block_worklist;
    auto degrade_block = [&](size_t index, State state) noexcept {
        auto old_state = control_states[index];
        auto new_state = join_state(old_state, state);
        if (new_state != old_state) {
            control_states[index] = new_state;
            block_worklist.emplace_back(index);
        }
    };
    std::unordered_map<const xir::Value *, std::vector<size_t>>
        selector_blocks;
    selector_blocks.reserve(blocks.size());
    for (auto block_index = size_t{0u}; block_index < blocks.size();
         block_index++) {
        auto *terminator = blocks[block_index]->is_terminated() ?
                               blocks[block_index]->terminator() :
                               nullptr;
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
            auto state = _state(selector);
            if (state != State::warp_uniform) {
                for (auto successor : successors[block_index]) {
                    degrade_block(successor, state);
                }
            }
        } else if (successors[block_index].size() > 1u) {
            // Structured/multi-way control should be destructured before this
            // analysis. Stay conservative if an unrecognized split remains.
            for (auto successor : successors[block_index]) {
                degrade_block(successor, State::varying);
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
                            degrade_block(successor, state);
                        }
                    }
                }
            }
        }
        if (!block_worklist.empty()) {
            auto block_index = block_worklist.front();
            block_worklist.pop_front();
            auto state = control_states[block_index];
            for (auto phi : block_phis[block_index]) {
                degrade_value(phi, state);
            }
            for (auto successor : successors[block_index]) {
                degrade_block(successor, state);
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
