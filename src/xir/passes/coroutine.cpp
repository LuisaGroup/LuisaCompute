#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/type.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/coroutine.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coroutine.h>

namespace luisa::compute::xir {

namespace {

[[nodiscard]] bool is_exit_terminator(const Instruction *inst) noexcept {
    if (inst == nullptr) { return true; }
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::RETURN:
        case DerivedInstructionTag::UNREACHABLE:
        case DerivedInstructionTag::RASTER_DISCARD: return true;
        default: return false;
    }
}

[[nodiscard]] luisa::vector<BasicBlock *> reachable_blocks(FunctionDefinition *def) noexcept {
    luisa::vector<BasicBlock *> blocks;
    def->traverse_basic_blocks([&](BasicBlock *block) noexcept { blocks.emplace_back(block); });
    return blocks;
}

[[nodiscard]] luisa::vector<BasicBlock *> successor_blocks(BasicBlock *block) noexcept {
    luisa::vector<BasicBlock *> successors;
    block->traverse_successors(false, [&](BasicBlock *successor) noexcept { successors.emplace_back(successor); });
    return successors;
}

[[nodiscard]] Instruction *next_instruction(BasicBlock *block, Instruction *inst) noexcept {
    auto found = false;
    for (auto candidate : block->instructions()) {
        if (found) { return candidate->is_terminator() ? nullptr : candidate; }
        if (candidate == inst) { found = true; }
    }
    return nullptr;
}

[[nodiscard]] AllocaInst *local_alloca_from_value(Value *value) noexcept {
    if (value == nullptr || !value->isa<AllocaInst>()) { return nullptr; }
    auto alloca = static_cast<AllocaInst *>(value);
    return alloca->is_local() ? alloca : nullptr;
}

[[nodiscard]] Constant *uint_constant(Module *module, uint32_t value) noexcept {
    return module->create_constant(Type::of<uint32_t>(), &value);
}

struct CoroutineFrameSlot {
    AllocaInst *alloca{};
    const Type *type{};
    uint32_t index{};
};

[[nodiscard]] GEPInst *create_frame_slot_pointer(XIRBuilder &builder, AllocaInst *frame_alloca, const CoroutineFrameSlot &slot) noexcept {
    auto index = uint_constant(frame_alloca->parent_module(), slot.index);
    return builder.gep(slot.type, frame_alloca, {index});
}

[[nodiscard]] size_t transition_target_for_suspend(const CoroutineAnalysisInfo &analysis, size_t suspend_id) noexcept {
    for (auto transition : analysis.transitions) {
        if (transition.suspend_id == suspend_id) { return transition.exits ? analysis.continuations.size() : transition.to_continuation; }
    }
    return analysis.continuations.size();
}

[[nodiscard]] bool candidate_is_live_at_suspend(const CoroutineFrameCandidateInfo &candidate, size_t suspend_id) noexcept {
    for (auto id : candidate.live_across_suspend_ids) {
        if (id == suspend_id) { return true; }
    }
    return false;
}

}

CoroutineAnalysisInfo coroutine_analysis_run_on_function(Function *function) noexcept {
    CoroutineAnalysisInfo info;
    if (function == nullptr || !function->is_definition()) { return info; }
    auto def = static_cast<FunctionDefinition *>(function);
    auto blocks = reachable_blocks(def);
    luisa::unordered_map<Instruction *, size_t> suspend_by_inst;
    for (auto block : blocks) {
        for (auto inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::PHI:
                    info.diagnostics.emplace_back("coroutine analysis expects reg2mem-normalized XIR without reachable PHI nodes");
                    break;
                case DerivedInstructionTag::CORO_REGISTER: {
                    auto id = info.registers.size();
                    info.registers.emplace_back(CoroutineMarkerInfo{inst, block, id});
                    info.is_coroutine = true;
                    break;
                }
                case DerivedInstructionTag::CORO_SUSPEND: {
                    auto id = info.suspends.size();
                    suspend_by_inst.emplace(inst, id);
                    info.suspends.emplace_back(CoroutineMarkerInfo{inst, block, id});
                    info.is_coroutine = true;
                    break;
                }
                default: break;
            }
        }
    }
    if (!info.is_coroutine) { return info; }
    auto add_continuation = [&](BasicBlock *entry, Instruction *entry_inst = nullptr) noexcept {
        for (auto &&continuation : info.continuations) {
            if (continuation.entry_block == entry && continuation.entry_inst == entry_inst) { return continuation.id; }
        }
        auto id = info.continuations.size();
        info.continuations.emplace_back(CoroutineContinuationInfo{.id = id, .entry_block = entry, .entry_inst = entry_inst});
        return id;
    };
    auto entry_id = add_continuation(def->body_block());
    for (auto marker : info.suspends) {
        if (auto next = next_instruction(marker.block, marker.inst)) {
            static_cast<void>(add_continuation(marker.block, next));
        } else {
            for (auto successor : successor_blocks(marker.block)) { static_cast<void>(add_continuation(successor)); }
        }
    }
    for (auto continuation_index = 0u; continuation_index < info.continuations.size(); continuation_index++) {
        auto continuation_id = info.continuations[continuation_index].id;
        auto continuation_entry_block = info.continuations[continuation_index].entry_block;
        auto continuation_entry_inst = info.continuations[continuation_index].entry_inst;
        luisa::vector<BasicBlock *> worklist{continuation_entry_block};
        luisa::unordered_map<BasicBlock *, Instruction *> entry_insts;
        if (continuation_entry_inst != nullptr) { entry_insts.emplace(continuation_entry_block, continuation_entry_inst); }
        luisa::unordered_set<BasicBlock *> visited;
        while (!worklist.empty()) {
            auto block = worklist.back();
            worklist.pop_back();
            if (block == nullptr || !visited.emplace(block).second) { continue; }
            info.continuations[continuation_index].blocks.emplace_back(block);
            auto start = entry_insts.contains(block) ? entry_insts.at(block) : nullptr;
            auto started = start == nullptr;
            auto stopped_at_suspend = false;
            for (auto inst : block->instructions()) {
                if (!started) {
                    started = inst == start;
                    if (!started) { continue; }
                }
                if (auto iter = suspend_by_inst.find(inst); iter != suspend_by_inst.end()) {
                    auto suspend_id = iter->second;
                    info.continuations[continuation_index].suspend_ids.emplace_back(suspend_id);
                    if (auto next = next_instruction(block, inst)) {
                        auto target_id = add_continuation(block, next);
                        info.transitions.emplace_back(CoroutineTransitionInfo{continuation_id, suspend_id, target_id, false});
                    } else {
                        auto successors = successor_blocks(block);
                        if (successors.empty() || is_exit_terminator(block->terminator())) {
                            info.transitions.emplace_back(CoroutineTransitionInfo{continuation_id, suspend_id, 0u, true});
                        } else {
                            for (auto successor : successors) {
                                auto target_id = add_continuation(successor);
                                info.transitions.emplace_back(CoroutineTransitionInfo{continuation_id, suspend_id, target_id, false});
                            }
                        }
                    }
                    stopped_at_suspend = true;
                    break;
                }
            }
            if (stopped_at_suspend) { continue; }
            auto term = block->terminator();
            if (is_exit_terminator(term)) { continue; }
            for (auto successor : successor_blocks(block)) { worklist.emplace_back(successor); }
        }
    }
    luisa::unordered_map<AllocaInst *, luisa::unordered_set<size_t>> frame_candidates;
    for (auto marker : info.suspends) {
        auto scan_block = [&](BasicBlock *block, Instruction *start) noexcept {
            auto started = start == nullptr;
            for (auto inst : block->instructions()) {
                if (!started) {
                    started = inst == start;
                    if (!started) { continue; }
                }
                if (suspend_by_inst.contains(inst)) { break; }
                if (inst->isa<StoreInst>()) {
                    auto store = static_cast<StoreInst *>(inst);
                    if (auto alloca = local_alloca_from_value(store->variable())) { frame_candidates[alloca].emplace(marker.id); }
                }
                for (auto use : inst->operand_uses()) {
                    if (auto alloca = local_alloca_from_value(use->value())) { frame_candidates[alloca].emplace(marker.id); }
                }
            }
        };
        if (auto next = next_instruction(marker.block, marker.inst)) { scan_block(marker.block, next); }
        luisa::unordered_set<BasicBlock *> after_suspend;
        luisa::vector<BasicBlock *> worklist;
        if (next_instruction(marker.block, marker.inst) == nullptr) { worklist = successor_blocks(marker.block); }
        while (!worklist.empty()) {
            auto block = worklist.back();
            worklist.pop_back();
            if (block == nullptr || !after_suspend.emplace(block).second) { continue; }
            auto has_suspend = false;
            for (auto inst : block->instructions()) {
                if (suspend_by_inst.contains(inst)) {
                    has_suspend = true;
                    break;
                }
            }
            if (has_suspend) { continue; }
            if (is_exit_terminator(block->terminator())) { continue; }
            for (auto successor : successor_blocks(block)) { worklist.emplace_back(successor); }
        }
        for (auto block : after_suspend) { scan_block(block, nullptr); }
    }
    info.frame_candidates.reserve(frame_candidates.size());
    for (auto &&[alloca, suspend_ids] : frame_candidates) {
        CoroutineFrameCandidateInfo candidate{.alloca = alloca};
        candidate.live_across_suspend_ids.reserve(suspend_ids.size());
        for (auto id : suspend_ids) { candidate.live_across_suspend_ids.emplace_back(id); }
        info.frame_candidates.emplace_back(std::move(candidate));
    }
    static_cast<void>(entry_id);
    return info;
}

CoroutineAnalysisInfo coroutine_analysis_run_on_module(Module *module) noexcept {
    CoroutineAnalysisInfo merged;
    if (module == nullptr) { return merged; }
    for (auto function : module->function_list()) {
        auto sub = coroutine_analysis_run_on_function(function);
        merged.is_coroutine |= sub.is_coroutine;
        auto register_offset = merged.registers.size();
        auto suspend_offset = merged.suspends.size();
        auto continuation_offset = merged.continuations.size();
        for (auto marker : sub.registers) {
            marker.id += register_offset;
            merged.registers.emplace_back(marker);
        }
        for (auto marker : sub.suspends) {
            marker.id += suspend_offset;
            merged.suspends.emplace_back(marker);
        }
        for (auto continuation : sub.continuations) {
            continuation.id += continuation_offset;
            for (auto &id : continuation.suspend_ids) { id += suspend_offset; }
            merged.continuations.emplace_back(std::move(continuation));
        }
        for (auto transition : sub.transitions) {
            transition.from_continuation += continuation_offset;
            transition.suspend_id += suspend_offset;
            if (!transition.exits) { transition.to_continuation += continuation_offset; }
            merged.transitions.emplace_back(transition);
        }
        for (auto candidate : sub.frame_candidates) {
            for (auto &id : candidate.live_across_suspend_ids) { id += suspend_offset; }
            merged.frame_candidates.emplace_back(std::move(candidate));
        }
        merged.diagnostics.insert(merged.diagnostics.end(), sub.diagnostics.begin(), sub.diagnostics.end());
    }
    return merged;
}

CoroutineLowerInfo coroutine_lower_run_on_function(Function *function) noexcept {
    CoroutineLowerInfo lower;
    auto analysis = coroutine_analysis_run_on_function(function);
    lower.diagnostics = analysis.diagnostics;
    if (function == nullptr || !function->is_definition() || !analysis.is_coroutine) { return lower; }
    auto def = static_cast<FunctionDefinition *>(function);
    auto module = function->parent_module();
    luisa::vector<CoroutineFrameSlot> slots;
    slots.reserve(analysis.frame_candidates.size());
    luisa::sort(analysis.frame_candidates.begin(), analysis.frame_candidates.end(), [](auto lhs, auto rhs) noexcept { return lhs.alloca < rhs.alloca; });
    for (auto candidate : analysis.frame_candidates) {
        if (candidate.alloca == nullptr) { continue; }
        slots.emplace_back(CoroutineFrameSlot{.alloca = candidate.alloca, .type = candidate.alloca->type(), .index = static_cast<uint32_t>(slots.size())});
    }
    luisa::unordered_map<AllocaInst *, CoroutineFrameSlot> slot_map;
    luisa::vector<const Type *> frame_types;
    frame_types.reserve(slots.size());
    for (auto slot : slots) {
        slot_map.emplace(slot.alloca, slot);
        frame_types.emplace_back(slot.type);
    }
    auto frame_type = frame_types.empty() ? Type::of<uint32_t>() : Type::structure(frame_types);
    auto old_body = def->body_block();
    auto pre_entry = def->create_basic_block();
    auto dispatch_default = def->create_basic_block();
    auto dispatch_merge = def->create_basic_block();
    def->set_body_block(pre_entry);
    XIRBuilder builder;
    builder.set_insertion_point(pre_entry);
    auto state_alloca = builder.alloca_local(Type::of<uint32_t>());
    state_alloca->add_comment("coroutine state");
    auto frame_alloca = builder.alloca_local(frame_type);
    frame_alloca->add_comment("coroutine frame");
    auto entry_id = uint_constant(module, 0u);
    builder.store(state_alloca, entry_id);
    auto state = builder.load(Type::of<uint32_t>(), state_alloca);
    auto dispatch = builder.switch_(state);
    dispatch->add_comment("coroutine dispatcher");
    dispatch->add_case(0, old_body);
    dispatch->set_default_block(dispatch_default);
    dispatch->set_merge_block(dispatch_merge);
    lower.created_state_alloca_count = 1u;
    lower.created_frame_alloca_count = 1u;
    lower.created_switch_count = 1u;
    builder.set_insertion_point(dispatch_default);
    builder.unreachable_("invalid coroutine continuation");
    builder.set_insertion_point(dispatch_merge);
    builder.unreachable_("coroutine dispatcher merge unreachable");
    for (auto marker : analysis.registers) {
        auto register_inst = static_cast<CoroRegisterInst *>(marker.inst);
        if (auto alloca = local_alloca_from_value(register_inst->value())) {
            if (auto iter = slot_map.find(alloca); iter != slot_map.end()) {
                builder.set_insertion_point(register_inst->prev());
                auto frame_ptr = create_frame_slot_pointer(builder, frame_alloca, iter->second);
                auto value = builder.load(alloca->type(), alloca);
                builder.store(frame_ptr, value);
            }
        }
        register_inst->remove_self();
        lower.removed_register_count++;
    }
    for (auto marker : analysis.suspends) {
        auto suspend_inst = static_cast<CoroSuspendInst *>(marker.inst);
        builder.set_insertion_point(suspend_inst->prev());
        for (auto candidate : analysis.frame_candidates) {
            if (!candidate_is_live_at_suspend(candidate, marker.id)) { continue; }
            if (auto iter = slot_map.find(candidate.alloca); iter != slot_map.end()) {
                auto frame_ptr = create_frame_slot_pointer(builder, frame_alloca, iter->second);
                auto value = builder.load(candidate.alloca->type(), candidate.alloca);
                builder.store(frame_ptr, value);
            }
        }
        auto next_id = static_cast<uint32_t>(transition_target_for_suspend(analysis, marker.id));
        builder.store(state_alloca, uint_constant(module, next_id));
        for (auto candidate : analysis.frame_candidates) {
            if (!candidate_is_live_at_suspend(candidate, marker.id)) { continue; }
            if (auto iter = slot_map.find(candidate.alloca); iter != slot_map.end()) {
                auto frame_ptr = create_frame_slot_pointer(builder, frame_alloca, iter->second);
                auto value = builder.load(candidate.alloca->type(), frame_ptr);
                builder.store(candidate.alloca, value);
            }
        }
        suspend_inst->remove_self();
        lower.removed_suspend_count++;
    }
    lower.changed = lower.removed_register_count != 0u || lower.removed_suspend_count != 0u;
    return lower;
}

CoroutineLowerInfo coroutine_lower_run_on_module(Module *module) noexcept {
    CoroutineLowerInfo merged;
    if (module == nullptr) { return merged; }
    for (auto function : module->function_list()) {
        auto sub = coroutine_lower_run_on_function(function);
        merged.changed |= sub.changed;
        merged.removed_register_count += sub.removed_register_count;
        merged.removed_suspend_count += sub.removed_suspend_count;
        merged.created_state_alloca_count += sub.created_state_alloca_count;
        merged.created_frame_alloca_count += sub.created_frame_alloca_count;
        merged.created_switch_count += sub.created_switch_count;
        merged.diagnostics.insert(merged.diagnostics.end(), sub.diagnostics.begin(), sub.diagnostics.end());
    }
    return merged;
}

}
