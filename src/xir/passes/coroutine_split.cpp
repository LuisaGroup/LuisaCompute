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

// A flat coroutine in this initial pass means:
//   - every CoroSuspend / CoroRegister marker lives directly in the entry
//     body block (no enclosing structured-CFG container);
//   - the body block contains no IF / SWITCH / LOOP / SIMPLE_LOOP /
//     RAY_QUERY_LOOP / OUTLINE / AUTODIFF_SCOPE instructions;
//   - the body block ends with a ReturnInst (no out-edges to other blocks).
//
// The condition-replay extension in the SIGGRAPH Asia 2024 paper (appendix
// Sec. CF reconstruction) lifts the latter two constraints, but is the next
// milestone for this pass; PT/SDF should keep using `coroutine_lower` until
// then.
[[nodiscard]] bool is_flat_coroutine(FunctionDefinition *def, luisa::vector<luisa::string> &diagnostics) noexcept {
    auto body = def->body_block();
    if (body == nullptr) { return true; }
    luisa::vector<BasicBlock *> blocks_with_markers;
    def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
        for (auto inst : block->instructions()) {
            if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_SUSPEND ||
                inst->derived_instruction_tag() == DerivedInstructionTag::CORO_REGISTER) {
                blocks_with_markers.emplace_back(block);
                return;
            }
        }
    });
    for (auto block : blocks_with_markers) {
        if (block != body) {
            diagnostics.emplace_back(
                "coroutine_split: suspend/register marker found outside the entry body block; "
                "this typically means the marker is reached through a structured control-flow "
                "container (loop/if/switch). Only flat coroutines are supported in this initial "
                "pass; condition-replay loop handling is the next milestone "
                "(see GPU Coroutines paper, appendix Sec. CF reconstruction)");
            return false;
        }
    }
    for (auto inst : body->instructions()) {
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::IF:
            case DerivedInstructionTag::SWITCH:
            case DerivedInstructionTag::LOOP:
            case DerivedInstructionTag::SIMPLE_LOOP:
            case DerivedInstructionTag::RAY_QUERY_LOOP:
            case DerivedInstructionTag::OUTLINE:
            case DerivedInstructionTag::AUTODIFF_SCOPE: {
                diagnostics.emplace_back(
                    "coroutine_split: body contains a structured control-flow container "
                    "(only flat coroutines are supported in this initial pass; "
                    "condition-replay loop handling is the next milestone)");
                return false;
            }
            default: break;
        }
    }
    auto term = body->terminator();
    if (term == nullptr ||
        term->derived_instruction_tag() != DerivedInstructionTag::RETURN) {
        diagnostics.emplace_back(
            "coroutine_split: body block does not terminate in ReturnInst "
            "(multi-block flat coroutines are not yet handled by the cloner)");
        return false;
    }
    return true;
}

// InstructionCloneValueResolver implementation that maps:
//   - the source coroutine's arguments → the new callable's arguments
//   - the source allocas listed as frame slots → GEPs into the frame argument
//   - cloned instructions → their clones (inserted incrementally)
class CoroutineCloneResolver final : public InstructionCloneValueResolver {

private:
    luisa::unordered_map<const Value *, Value *> _map;

public:
    void emplace(const Value *from, Value *to) noexcept {
        _map.emplace(from, to);
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
        LUISA_ERROR_WITH_LOCATION("coroutine_split: unresolved value during cloning");
    }
};

[[nodiscard]] Constant *uint_constant(Module *module, uint32_t value) noexcept {
    return module->create_constant(luisa::compute::Type::of<uint32_t>(), &value);
}

// Mirror an argument from the source coroutine into the new continuation
// callable. Returns the new argument so the resolver can map it.
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

}// namespace

CoroutineSplitInfo coroutine_split_run_on_function(Function *function) noexcept {
    CoroutineSplitInfo info;
    if (function == nullptr || !function->is_definition()) {
        info.diagnostics.emplace_back("coroutine_split: function is null or not a definition");
        return info;
    }
    auto def = static_cast<FunctionDefinition *>(function);
    auto analysis = coroutine_analysis_run_on_function(function);
    info.diagnostics = analysis.diagnostics;
    if (!analysis.is_coroutine) { return info; }
    if (!is_flat_coroutine(def, info.diagnostics)) {
        info.is_supported = false;
        return info;
    }
    auto module = function->parent_module();

    // --- Build frame type -----------------------------------------------------
    // Slot 0 = target_token (uint). Slots 1..N map to allocas selected by the
    // analysis pass, deterministically ordered by alloca pointer (matches
    // coroutine_lower). The frame type is a struct of those member types.
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

    // --- Slice the body block by suspends ------------------------------------
    // Each slice corresponds to one continuation. The first slice is the
    // entry; subsequent slices begin right after a suspend marker. Suspend
    // and register markers are NOT cloned — they get rewritten in place.
    auto body = def->body_block();
    luisa::vector<luisa::vector<Instruction *>> slices(1);
    luisa::vector<size_t> slice_terminator_suspend_id;
    auto suspend_to_id = luisa::unordered_map<const Instruction *, size_t>{};
    for (auto marker : analysis.suspends) { suspend_to_id.emplace(marker.inst, marker.id); }
    auto register_set = luisa::unordered_set<const Instruction *>{};
    for (auto marker : analysis.registers) { register_set.emplace(marker.inst); }
    for (auto inst : body->instructions()) {
        if (auto reg_iter = register_set.find(inst); reg_iter != register_set.end()) {
            continue;// drop coro_register markers
        }
        if (auto sus_iter = suspend_to_id.find(inst); sus_iter != suspend_to_id.end()) {
            slice_terminator_suspend_id.emplace_back(sus_iter->second);
            slices.emplace_back();
            continue;
        }
        slices.back().emplace_back(inst);
    }
    slice_terminator_suspend_id.emplace_back(static_cast<size_t>(-1));// final slice ends with original return

    // --- Resolve next-token per suspend --------------------------------------
    luisa::unordered_map<size_t, size_t> suspend_next;
    for (auto t : analysis.transitions) {
        suspend_next.emplace(t.suspend_id, t.exits ? 0u : t.to_continuation);
    }

    // --- Emit one CallableFunction per slice ---------------------------------
    info.continuations.clear();
    info.continuations.reserve(slices.size());
    for (size_t k = 0; k < slices.size(); ++k) {
        auto callable = module->create_callable(/* ret_type */ nullptr);
        auto frame_ref = callable->create_reference_argument(info.frame_type);
        // Mirror the source coroutine's arguments after the frame ref.
        luisa::vector<Argument *> mirrored_args;
        for (auto src_arg : function->arguments()) {
            mirrored_args.emplace_back(mirror_argument(callable, src_arg));
        }
        auto block = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(block);

        CoroutineCloneResolver resolver;
        // Map original arguments to the mirrored ones in this callable.
        {
            size_t i = 0;
            for (auto src_arg : function->arguments()) {
                resolver.emplace(src_arg, mirrored_args[i]);
                ++i;
            }
        }
        // Map source allocas to GEPs into the frame.
        for (auto slot : info.frame_slots) {
            auto idx = uint_constant(module, static_cast<uint32_t>(slot.field_index));
            auto gep = builder.gep(slot.type, frame_ref, {idx});
            resolver.emplace(slot.source_alloca, gep);
        }
        // GEP for the target_token slot (slot 0).
        auto state_idx = uint_constant(module, 0u);
        auto state_gep = builder.gep(luisa::compute::Type::of<uint32_t>(), frame_ref, {state_idx});

        // Clone the body slice.
        for (auto inst : slices[k]) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::ALLOCA: {
                    auto is_frame_slot = std::any_of(
                        info.frame_slots.begin(), info.frame_slots.end(),
                        [&](auto slot) noexcept { return slot.source_alloca == inst; });
                    if (!is_frame_slot) {
                        // Local-only alloca that isn't live across suspends — clone in-place.
                        auto cloned = inst->clone_with_metadata(builder, resolver);
                        resolver.emplace(inst, cloned);
                    }
                    break;
                }
                case DerivedInstructionTag::RETURN: {
                    builder.store(state_gep, uint_constant(module, 0u));
                    builder.return_void();
                    break;
                }
                default: {
                    auto cloned = inst->clone_with_metadata(builder, resolver);
                    resolver.emplace(inst, cloned);
                    break;
                }
            }
        }

        // If this slice ended with a suspend (not a return), emit the
        // store-then-return-void epilogue using the precomputed next token.
        auto term_suspend_id = slice_terminator_suspend_id[k];
        if (term_suspend_id != static_cast<size_t>(-1)) {
            auto next_iter = suspend_next.find(term_suspend_id);
            auto next_token = next_iter == suspend_next.end() ? 0u : static_cast<uint32_t>(next_iter->second);
            builder.store(state_gep, uint_constant(module, next_token));
            builder.return_void();
        } else if (slices[k].empty() ||
                   slices[k].back()->derived_instruction_tag() != DerivedInstructionTag::RETURN) {
            builder.store(state_gep, uint_constant(module, 0u));
            builder.return_void();
        }

        CoroutineSplitContinuation entry{
            .id = k,
            .callable = callable,
            .outgoing_suspends = {},
        };
        if (term_suspend_id != static_cast<size_t>(-1)) {
            entry.outgoing_suspends.emplace_back(term_suspend_id);
        }
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
