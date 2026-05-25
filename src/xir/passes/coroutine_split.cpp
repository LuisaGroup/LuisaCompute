#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/argument.h>
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
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coroutine.h>
#include <luisa/xir/passes/coroutine_split.h>
#include <luisa/xir/undefined.h>

namespace luisa::compute::xir {

namespace {

class CoroutineCloneResolver final : public InstructionCloneValueResolver {

private:
    luisa::unordered_map<const Value *, Value *> _map;

public:
    void emplace(const Value *from, Value *to) noexcept {
        _map.emplace(from, to);
    }

    void bind(const Value *from, Value *to) noexcept {
        _map.insert_or_assign(from, to);
    }

    [[nodiscard]] bool contains(const Value *v) const noexcept {
        return _map.find(v) != _map.end();
    }

    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) { return nullptr; }
        switch (value->derived_value_tag()) {
            case DerivedValueTag::UNDEFINED:
            case DerivedValueTag::FUNCTION:
            case DerivedValueTag::CONSTANT:
            case DerivedValueTag::SPECIAL_REGISTER:
                return const_cast<Value *>(value);
            default: break;
        }
        if (auto it = _map.find(value); it != _map.end()) { return it->second; }
        if (value->derived_value_tag() == DerivedValueTag::BASIC_BLOCK) {
            // Branch target not in this continuation — should not happen after
            // the branch handling logic, but return nullptr to catch bugs early.
            LUISA_ERROR_WITH_LOCATION(
                "coroutine_split: unresolved basic block during cloning");
        }
        if (value->derived_value_tag() == DerivedValueTag::INSTRUCTION) {
            auto inst = static_cast<const Instruction *>(value);
            LUISA_ERROR_WITH_LOCATION(
                "coroutine_split: unresolved value during cloning (tag={}, type={})",
                static_cast<int>(inst->derived_instruction_tag()),
                inst->type() ? inst->type()->description() : "null");
        }
        LUISA_ERROR_WITH_LOCATION("coroutine_split: unresolved value during cloning");
    }
};

[[nodiscard]] Constant *uint_constant(Module *module, uint32_t value) noexcept {
    return module->create_constant(luisa::compute::Type::of<uint32_t>(), &value);
}

[[nodiscard]] Argument *mirror_argument(Function *callable, const Argument *src) noexcept {
    switch (src->derived_argument_tag()) {
        case DerivedArgumentTag::VALUE:
            return callable->create_value_argument(src->type());
        case DerivedArgumentTag::REFERENCE:
            return callable->create_reference_argument(src->type());
        case DerivedArgumentTag::RESOURCE:
            return callable->create_resource_argument(src->type());
    }
    return nullptr;
}

[[nodiscard]] Instruction *next_instruction_including_terminator(BasicBlock *block, Instruction *inst) noexcept {
    auto found = false;
    for (auto candidate : block->instructions()) {
        if (found) { return candidate; }
        if (candidate == inst) { found = true; }
    }
    return nullptr;
}

[[nodiscard]] bool is_frame_slot(const luisa::vector<CoroutineSplitFrameSlot> &slots, Instruction *inst) noexcept {
    return std::any_of(slots.begin(), slots.end(), [&](auto slot) noexcept { return slot.source_alloca == inst; });
}

void emit_terminate(XIRBuilder &builder, Module *module, Value *state_gep) noexcept {
    builder.store(state_gep, uint_constant(module, 0u));
    builder.return_void();
}

void emit_suspend(XIRBuilder &builder, Module *module, Value *state_gep, uint32_t next_token) noexcept {
    builder.store(state_gep, uint_constant(module, next_token));
    builder.return_void();
}

[[nodiscard]] uint32_t next_token_for_suspend(const CoroutineAnalysisInfo &analysis, size_t suspend_id) noexcept {
    for (auto transition : analysis.transitions) {
        if (transition.suspend_id == suspend_id) {
            return transition.exits ? 0u : static_cast<uint32_t>(transition.to_continuation);
        }
    }
    return 0u;
}

[[nodiscard]] bool continuation_contains_block(const CoroutineContinuationInfo &continuation, BasicBlock *block) noexcept {
    for (auto candidate : continuation.blocks) {
        if (candidate == block) { return true; }
    }
    return false;
}

[[nodiscard]] bool is_unstructured_terminator(Instruction *inst) noexcept {
    if (inst == nullptr) { return true; }
    switch (inst->derived_instruction_tag()) {
        case DerivedInstructionTag::BRANCH:
        case DerivedInstructionTag::CONDITIONAL_BRANCH:
        case DerivedInstructionTag::SWITCH:
        case DerivedInstructionTag::RETURN:
        case DerivedInstructionTag::UNREACHABLE:
        case DerivedInstructionTag::RASTER_DISCARD: return true;
        default: return !inst->is_terminator();
    }
}

}// namespace

CoroutineSplitInfo coroutine_split_run_on_function(Function *function) noexcept {
    CoroutineSplitInfo info;
    if (function == nullptr || !function->is_definition()) {
        info.diagnostics.emplace_back("coroutine_split: function is null or not a definition");
        return info;
    }
    auto analysis = coroutine_analysis_run_on_function(function);
    info.diagnostics = analysis.diagnostics;
    if (!analysis.is_coroutine) { return info; }
    auto def = static_cast<FunctionDefinition *>(function);
    auto module = function->parent_module();
    luisa::vector<CoroutineFrameCandidateInfo> sorted_candidates = analysis.frame_candidates;
    luisa::sort(sorted_candidates.begin(), sorted_candidates.end(),
                [](auto lhs, auto rhs) noexcept { return lhs.alloca < rhs.alloca; });
    luisa::vector<const luisa::compute::Type *> frame_member_types;
    frame_member_types.reserve(sorted_candidates.size() + 1u);
    frame_member_types.emplace_back(luisa::compute::Type::of<uint32_t>());// slot 0: target_token
    for (auto candidate : sorted_candidates) {
        if (candidate.alloca == nullptr) { continue; }
        CoroutineSplitFrameSlot slot{
            .source_alloca = candidate.alloca,
            .field_index = static_cast<size_t>(frame_member_types.size()),
            .type = candidate.alloca->type(),
        };
        frame_member_types.emplace_back(slot.type);
        info.frame_slots.emplace_back(slot);
    }
    info.frame_type = luisa::compute::Type::structure(frame_member_types);

    luisa::unordered_map<const Instruction *, size_t> suspend_ids;
    for (auto marker : analysis.suspends) { suspend_ids.emplace(marker.inst, marker.id); }
    luisa::unordered_set<const Instruction *> registers;
    for (auto marker : analysis.registers) { registers.emplace(marker.inst); }

    info.continuations.clear();
    info.continuations.reserve(analysis.continuations.size());
    for (auto &continuation : analysis.continuations) {
        auto callable = module->create_callable(nullptr);
        auto frame_ref = callable->create_reference_argument(info.frame_type);
        luisa::vector<Argument *> mirrored_args;
        for (auto src_arg : function->arguments()) {
            mirrored_args.emplace_back(mirror_argument(callable, src_arg));
        }
        luisa::unordered_map<BasicBlock *, BasicBlock *> block_map;
        for (auto src_block : continuation.blocks) {
            auto cloned_block = src_block == continuation.entry_block ? callable->create_body_block() : callable->create_basic_block();
            block_map.emplace(src_block, cloned_block);
        }
        if (block_map.empty()) {
            auto cloned_block = callable->create_body_block();
            block_map.emplace(continuation.entry_block, cloned_block);
        }

        CoroutineCloneResolver resolver;
        for (auto &&[src_block, cloned_block] : block_map) { resolver.emplace(src_block, cloned_block); }
        size_t arg_index = 0;
        for (auto src_arg : function->arguments()) {
            resolver.emplace(src_arg, mirrored_args[arg_index]);
            arg_index++;
        }

        XIRBuilder builder;
        builder.set_insertion_point(block_map.at(continuation.entry_block));
        for (auto slot : info.frame_slots) {
            auto idx = uint_constant(module, static_cast<uint32_t>(slot.field_index));
            auto gep = builder.gep(slot.type, frame_ref, {idx});
            resolver.emplace(slot.source_alloca, gep);
        }
        auto state_idx = uint_constant(module, 0u);
        auto state_gep = builder.gep(luisa::compute::Type::of<uint32_t>(), frame_ref, {state_idx});

        // Pre-clone non-frame-slot allocas into the entry block.
        def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            for (auto inst : block->instructions()) {
                if (inst->derived_instruction_tag() != DerivedInstructionTag::ALLOCA) continue;
                if (resolver.contains(inst)) continue;
                builder.set_insertion_point(block_map.at(continuation.entry_block));
                auto local = builder.alloca_local(inst->type());
                resolver.bind(inst, local);
            }
        });

        for (auto src_block : continuation.blocks) {
            auto cloned_block = block_map.at(src_block);
            builder.set_insertion_point(cloned_block);
            auto started = src_block != continuation.entry_block || continuation.entry_inst == nullptr;
            auto terminated = false;
            for (auto inst : src_block->instructions()) {
                if (!started) {
                    started = inst == continuation.entry_inst;
                    if (!started) { continue; }
                }
                if (registers.contains(inst)) { continue; }
                if (auto suspend_iter = suspend_ids.find(inst); suspend_iter != suspend_ids.end()) {
                    emit_suspend(builder, module, state_gep, next_token_for_suspend(analysis, suspend_iter->second));
                    terminated = true;
                    break;
                }
                switch (inst->derived_instruction_tag()) {
                    case DerivedInstructionTag::ALLOCA: {
                        if (!is_frame_slot(info.frame_slots, inst)) {
                            auto cloned = inst->clone_with_metadata(builder, resolver);
                            resolver.bind(inst, cloned);
                        }
                        break;
                    }
                    case DerivedInstructionTag::RETURN: {
                        emit_terminate(builder, module, state_gep);
                        terminated = true;
                        break;
                    }
                    case DerivedInstructionTag::BRANCH: {
                        auto br = static_cast<BranchInst *>(inst);
                        if (br->target_block() && !block_map.contains(br->target_block())) {
                            emit_terminate(builder, module, state_gep);
                        } else {
                            auto cloned = inst->clone_with_metadata(builder, resolver);
                            resolver.bind(inst, cloned);
                        }
                        terminated = true;
                        break;
                    }
                    case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                        auto cbr = static_cast<ConditionalBranchInst *>(inst);
                        auto true_in = cbr->true_block() && block_map.contains(cbr->true_block());
                        auto false_in = cbr->false_block() && block_map.contains(cbr->false_block());
                        if (true_in && false_in) {
                            auto cloned = inst->clone_with_metadata(builder, resolver);
                            resolver.bind(inst, cloned);
                        } else if (true_in) {
                            builder.br(block_map.at(cbr->true_block()));
                        } else if (false_in) {
                            builder.br(block_map.at(cbr->false_block()));
                        } else {
                            emit_terminate(builder, module, state_gep);
                        }
                        terminated = true;
                        break;
                    }
                    case DerivedInstructionTag::SWITCH: {
                        auto cloned = inst->clone_with_metadata(builder, resolver);
                        resolver.bind(inst, cloned);
                        terminated = true;
                        break;
                    }
                    case DerivedInstructionTag::CORO_REGISTER:
                    case DerivedInstructionTag::CORO_SUSPEND: break;
                    default: {
                        if (!is_unstructured_terminator(inst)) {
                            info.diagnostics.emplace_back("coroutine_split: structured terminator found in unstructured coroutine CFG");
                            emit_terminate(builder, module, state_gep);
                            terminated = true;
                            break;
                        }
                        auto cloned = inst->clone_with_metadata(builder, resolver);
                        resolver.bind(inst, cloned);
                        if (inst->is_terminator()) { terminated = true; }
                        break;
                    }
                }
                if (terminated) { break; }
            }
            if (!terminated && !cloned_block->is_terminated()) {
                emit_terminate(builder, module, state_gep);
            }
        }
        // Ensure all blocks in this callable are terminated.
        auto ensure_terminated = [&](BasicBlock *block) noexcept {
            if (block && !block->is_terminated()) {
                builder.set_insertion_point(block);
                emit_terminate(builder, module, state_gep);
            }
        };
        ensure_terminated(callable->body_block());
        for (auto block : callable->basic_blocks()) { ensure_terminated(block); }
        for (auto &&[_, cloned_block] : block_map) { ensure_terminated(cloned_block); }
        // Verify: no unterminated blocks should remain.
        LUISA_ASSERT(!callable->body_block() || callable->body_block()->is_terminated(),
                     "coroutine_split: body block still unterminated after safety pass");
        for (auto block : callable->basic_blocks()) {
            LUISA_ASSERT(block->is_terminated(),
                         "coroutine_split: block still unterminated after safety pass");
        }

        CoroutineSplitContinuation entry{
            .id = continuation.id,
            .callable = callable,
            .outgoing_suspends = {},
        };
        for (auto suspend_id : continuation.suspend_ids) { entry.outgoing_suspends.emplace_back(suspend_id); }
        info.continuations.emplace_back(std::move(entry));
    }

    info.is_supported = true;
    info.changed = true;
    return info;
}

luisa::vector<CoroutineSplitInfo> coroutine_split_run_on_module(Module *module) noexcept {
    luisa::vector<CoroutineSplitInfo> results;
    if (module == nullptr) { return results; }
    luisa::vector<Function *> targets;
    for (auto function : module->function_list()) { targets.emplace_back(function); }
    for (auto function : targets) {
        auto sub = coroutine_split_run_on_function(function);
        if (!sub.continuations.empty() || !sub.diagnostics.empty()) {
            results.emplace_back(std::move(sub));
        }
    }
    return results;
}

}// namespace luisa::compute::xir
