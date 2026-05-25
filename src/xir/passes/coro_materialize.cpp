#include <luisa/core/logging.h>
#include <luisa/core/stl/algorithm.h>
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
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coroutine.h>
#include <luisa/xir/passes/coro_graph_analysis.h>
#include <luisa/xir/passes/coro_materialize.h>

namespace luisa::compute::xir::coro {

namespace {

[[nodiscard]] Constant *uint_constant(Module *module, uint32_t value) noexcept {
    return module->create_constant(luisa::compute::Type::of<uint32_t>(), &value);
}

[[nodiscard]] Constant *bool_constant(Module *module, bool value) noexcept {
    return module->create_constant(luisa::compute::Type::of<bool>(), &value);
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

// Clone resolver that falls through to the original value if not mapped.
// This is intentional: values defined outside the current scope (constants,
// special registers, arguments) should resolve to themselves.
class MaterializeResolver final : public InstructionCloneValueResolver {
    luisa::unordered_map<const Value *, Value *> _map;
    XIRBuilder *_builder{};
public:
    void set_builder(XIRBuilder *b) noexcept { _builder = b; }
    void emplace(const Value *from, Value *to) noexcept { _map.emplace(from, to); }
    bool contains(const Value *v) const noexcept { return _map.contains(v); }
    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) return nullptr;
        switch (value->derived_value_tag()) {
            case DerivedValueTag::UNDEFINED:
            case DerivedValueTag::FUNCTION:
            case DerivedValueTag::CONSTANT:
            case DerivedValueTag::SPECIAL_REGISTER:
                return const_cast<Value *>(value);
            default: break;
        }
        if (auto it = _map.find(value); it != _map.end()) return it->second;
        // Replay: clone non-terminator, non-alloca instructions on demand.
        // This handles pure computations defined before the suspend point
        // that are used after (equivalent to the Rust impl's replay_value).
        if (value->derived_value_tag() == DerivedValueTag::INSTRUCTION && _builder != nullptr) {
            auto inst = const_cast<Instruction *>(static_cast<const Instruction *>(value));
            if (inst->derived_instruction_tag() == DerivedInstructionTag::ALLOCA) {
                auto alloca_inst = static_cast<AllocaInst *>(inst);
                LUISA_ASSERT(!alloca_inst->type()->is_resource(),
                             "coro_materialize: resource-typed alloca is ill-formed");
            }
            if (!inst->is_terminator()) {
                if (inst->derived_instruction_tag() == DerivedInstructionTag::GEP) {
                    auto gep = static_cast<GEPInst *>(inst);
                    if (gep->base()->type() && gep->base()->type()->is_resource()) {
                        auto resolved_base = this->resolve(gep->base());
                        _map.emplace(value, resolved_base);
                        return resolved_base;
                    }
                }
                auto cloned = inst->clone_with_metadata(*_builder, *this);
                _map.emplace(value, cloned);
                return cloned;
            }
        }
        LUISA_ERROR_WITH_LOCATION(
            "coro_materialize: unresolved value (tag={}, type={})",
            static_cast<int>(value->derived_value_tag()),
            value->type() ? value->type()->description() : "null");
    }
};

struct ScopeMaterializer {
    Module *module{};
    Function *source_function{};
    const CoroGraphInfo *graph{};
    const CoroutineSplitInfo *split_info{};

    // Per-callable state
    CallableFunction *callable{};
    XIRBuilder builder;
    MaterializeResolver resolver;
    Value *frame_ref{};
    Value *state_gep{};// GEP to frame[0] (target_token)

    // First-flag allocas indexed by the CoroInstrRef of the MakeFirstFlag node
    luisa::unordered_map<size_t, Value *> first_flags;

    void materialize_scope(CoroScopeRef scope_ref, uint32_t next_token_on_suspend) noexcept {
        auto &scope = graph->scopes[scope_ref.index];
        for (auto ref : scope.instructions) {
            materialize_instr(ref, next_token_on_suspend);
        }
    }

    void materialize_instr(CoroInstrRef ref, uint32_t next_token_on_suspend) noexcept {
        if (!ref.valid()) return;
        auto &instr = graph->preliminary.instructions[ref.index];
        switch (instr.tag) {
            case CoroInstruction::Tag::SIMPLE:
                materialize_simple(instr);
                break;
            case CoroInstruction::Tag::CONDITION_STACK_REPLAY:
                materialize_condition_replay(instr);
                break;
            case CoroInstruction::Tag::MAKE_FIRST_FLAG:
                materialize_make_first_flag(ref);
                break;
            case CoroInstruction::Tag::SKIP_IF_FIRST_FLAG:
                materialize_skip_if_first_flag(instr, next_token_on_suspend);
                break;
            case CoroInstruction::Tag::CLEAR_FIRST_FLAG:
                materialize_clear_first_flag(instr);
                break;
            case CoroInstruction::Tag::LOOP:
                materialize_loop(instr, next_token_on_suspend);
                break;
            case CoroInstruction::Tag::IF:
                materialize_if(instr, next_token_on_suspend);
                break;
            case CoroInstruction::Tag::SWITCH:
                materialize_switch(instr, next_token_on_suspend);
                break;
            case CoroInstruction::Tag::SUSPEND:
                materialize_suspend(instr, next_token_on_suspend);
                break;
            case CoroInstruction::Tag::TERMINATE:
                materialize_terminate();
                break;
            default:
                break;
        }
    }

    void materialize_simple(const CoroInstruction &instr) noexcept {
        auto *inst = instr.source_inst;
        if (inst == nullptr) return;
        if (inst->derived_instruction_tag() == DerivedInstructionTag::CORO_REGISTER) return;
        // Skip terminators — control flow is handled structurally by the materializer.
        if (inst->is_terminator()) return;
        if (inst->derived_instruction_tag() == DerivedInstructionTag::ALLOCA) {
            bool is_frame_slot = std::any_of(
                split_info->frame_slots.begin(), split_info->frame_slots.end(),
                [&](auto &slot) { return slot.source_alloca == inst; });
            if (is_frame_slot) return;
            auto cloned = inst->clone_with_metadata(builder, resolver);
            resolver.emplace(inst, cloned);
            return;
        }
        // Skip GEPs on resource bases — they're resolved inline by ResourceQueryInst.
        if (inst->derived_instruction_tag() == DerivedInstructionTag::GEP) {
            auto gep = static_cast<GEPInst *>(inst);
            if (gep->base()->type() && gep->base()->type()->is_resource()) {
                return;
            }
        }
        if (resolver.contains(inst)) return;
        auto cloned = inst->clone_with_metadata(builder, resolver);
        resolver.emplace(inst, cloned);
    }

    void materialize_condition_replay(const CoroInstruction &instr) noexcept {
        // For each replay item, emit a constant with the replayed value.
        // The condition value is what the if/switch branched on; we store the
        // constant so subsequent cloned if/switch instructions see the right value.
        for (auto &item : instr.replay_items) {
            if (item.control_flow_inst == nullptr) continue;
            // The condition is the operand of the if/switch instruction.
            Value *original_cond = nullptr;
            if (item.control_flow_inst->derived_instruction_tag() == DerivedInstructionTag::IF) {
                auto if_inst = static_cast<IfInst *>(item.control_flow_inst);
                original_cond = if_inst->operand(0);// condition operand
            } else if (item.control_flow_inst->derived_instruction_tag() == DerivedInstructionTag::SWITCH) {
                auto sw = static_cast<SwitchInst *>(item.control_flow_inst);
                original_cond = sw->operand(0);// value operand
            }
            if (original_cond == nullptr) continue;
            // Emit a constant for the replayed value.
            Constant *replayed = nullptr;
            if (original_cond->type() == luisa::compute::Type::of<bool>()) {
                replayed = bool_constant(module, item.value != 0);
            } else {
                replayed = module->create_constant(luisa::compute::Type::of<int>(), &item.value);
            }
            resolver.emplace(original_cond, replayed);
        }
    }

    void materialize_make_first_flag(CoroInstrRef ref) noexcept {
        auto flag = builder.alloca_local(luisa::compute::Type::of<bool>());
        builder.store(flag, bool_constant(module, true));
        first_flags.emplace(ref.index, flag);
    }

    void materialize_skip_if_first_flag(const CoroInstruction &instr, uint32_t next_token) noexcept {
        auto flag_it = first_flags.find(instr.first_flag.index);
        if (flag_it == first_flags.end()) return;
        auto flag = flag_it->second;
        // Emit: if (!flag) { ...body... }
        auto loaded = builder.load(luisa::compute::Type::of<bool>(), flag);
        auto not_flag = builder.call(luisa::compute::Type::of<bool>(),
                                     ArithmeticOp::UNARY_BIT_NOT, {loaded});
        auto if_inst = builder.if_(not_flag);
        auto true_block = if_inst->create_true_block();
        auto false_block = if_inst->create_false_block();
        auto merge_block = if_inst->create_merge_block();
        // True branch: execute the body (first_flag is false → not first run)
        builder.set_insertion_point(true_block);
        for (auto body_ref : instr.body) {
            materialize_instr(body_ref, next_token);
        }
        builder.br(merge_block);
        // False branch: skip (first run)
        builder.set_insertion_point(false_block);
        builder.br(merge_block);
        // Continue after merge
        builder.set_insertion_point(merge_block);
    }

    void materialize_clear_first_flag(const CoroInstruction &instr) noexcept {
        auto flag_it = first_flags.find(instr.first_flag.index);
        if (flag_it == first_flags.end()) return;
        builder.store(flag_it->second, bool_constant(module, false));
    }

    void materialize_loop(const CoroInstruction &instr, uint32_t next_token) noexcept {
        // XIR LoopInst: prepare → body → update → (back to prepare)
        // instr.body = prepare instructions
        // instr.true_branch = body_block instructions
        // instr.false_branch = update instructions
        auto loop = builder.loop();
        auto prepare = loop->create_prepare_block();
        auto loop_body = loop->create_body_block();
        auto update = loop->create_update_block();
        auto merge = loop->create_merge_block();

        builder.set_insertion_point(prepare);
        for (auto r : instr.body) { materialize_instr(r, next_token); }
        if (!prepare->is_terminated()) { builder.br(loop_body); }

        builder.set_insertion_point(loop_body);
        for (auto r : instr.true_branch) { materialize_instr(r, next_token); }
        if (!loop_body->is_terminated()) { builder.br(update); }

        builder.set_insertion_point(update);
        for (auto r : instr.false_branch) { materialize_instr(r, next_token); }
        if (!update->is_terminated()) { builder.br(prepare); }

        builder.set_insertion_point(merge);
    }

    void materialize_if(const CoroInstruction &instr, uint32_t next_token) noexcept {
        auto *src = instr.source_inst;
        // Resolve the condition
        Value *cond = nullptr;
        if (src != nullptr && src->derived_instruction_tag() == DerivedInstructionTag::IF) {
            auto orig_if = static_cast<IfInst *>(src);
            auto orig_cond = orig_if->operand(0);
            cond = resolver.resolve(orig_cond);
        }
        if (cond == nullptr) {
            cond = bool_constant(module, true);
        }
        auto if_inst = builder.if_(cond);
        auto true_block = if_inst->create_true_block();
        auto false_block = if_inst->create_false_block();
        auto merge_block = if_inst->create_merge_block();

        builder.set_insertion_point(true_block);
        for (auto r : instr.true_branch) { materialize_instr(r, next_token); }
        if (!true_block->is_terminated()) { builder.br(merge_block); }

        builder.set_insertion_point(false_block);
        for (auto r : instr.false_branch) { materialize_instr(r, next_token); }
        if (!false_block->is_terminated()) { builder.br(merge_block); }

        builder.set_insertion_point(merge_block);
    }

    void materialize_switch(const CoroInstruction &instr, uint32_t next_token) noexcept {
        auto *src = instr.source_inst;
        Value *value = nullptr;
        if (src != nullptr && src->derived_instruction_tag() == DerivedInstructionTag::SWITCH) {
            auto orig_sw = static_cast<SwitchInst *>(src);
            value = resolver.resolve(orig_sw->operand(0));
        }
        if (value == nullptr) {
            value = uint_constant(module, 0u);
        }
        auto sw = builder.switch_(value);
        auto default_block = callable->create_basic_block();
        auto merge_block = callable->create_basic_block();
        sw->set_default_block(default_block);
        sw->set_merge_block(merge_block);

        for (auto &c : instr.cases) {
            auto case_block = callable->create_basic_block();
            sw->add_case(c.value, case_block);
            builder.set_insertion_point(case_block);
            for (auto r : c.body) { materialize_instr(r, next_token); }
            if (!case_block->is_terminated()) { builder.br(merge_block); }
        }

        builder.set_insertion_point(default_block);
        for (auto r : instr.default_branch) { materialize_instr(r, next_token); }
        if (!default_block->is_terminated()) { builder.br(merge_block); }

        builder.set_insertion_point(merge_block);
    }

    void materialize_suspend(const CoroInstruction &instr, uint32_t next_token) noexcept {
        // Store the next continuation token and return.
        builder.store(state_gep, uint_constant(module, next_token));
        builder.return_void();
    }

    void materialize_terminate() noexcept {
        // Store 0 (terminated) and return.
        builder.store(state_gep, uint_constant(module, 0u));
        builder.return_void();
    }
};

}// namespace

CoroMaterializeResult coro_materialize_run_on_function(Function *function) noexcept {
    CoroMaterializeResult result;
    if (function == nullptr || !function->is_definition()) {
        result.diagnostics.emplace_back("coro_materialize: function is null or not a definition");
        return result;
    }
    auto module = function->parent_module();
    auto graph = coro_graph_run_on_function(function);
    if (!graph.ok || graph.scopes.empty()) {
        result.diagnostics = std::move(graph.diagnostics);
        return result;
    }

    // Build frame type from the existing coroutine analysis (frame_candidates).
    auto analysis = coroutine_analysis_run_on_function(function);
    if (!analysis.is_coroutine) {
        result.diagnostics.emplace_back("coro_materialize: function is not a coroutine");
        return result;
    }

    // Sort frame candidates deterministically.
    luisa::vector<CoroutineFrameCandidateInfo> sorted_candidates = analysis.frame_candidates;
    luisa::sort(sorted_candidates.begin(), sorted_candidates.end(),
                [](auto lhs, auto rhs) noexcept { return lhs.alloca < rhs.alloca; });

    // Build frame type: slot 0 = target_token (uint), slots 1..N = alloca types.
    luisa::vector<const luisa::compute::Type *> frame_member_types;
    frame_member_types.emplace_back(luisa::compute::Type::of<uint32_t>());
    for (auto &candidate : sorted_candidates) {
        if (candidate.alloca == nullptr) continue;
        auto t = candidate.alloca->type();
        if (t == nullptr || t->is_resource() || t->is_custom()) continue;
        CoroutineSplitFrameSlot slot{
            .source_alloca = candidate.alloca,
            .field_index = frame_member_types.size(),
            .type = t,
        };
        frame_member_types.emplace_back(slot.type);
        result.split_info.frame_slots.emplace_back(slot);
    }
    result.split_info.frame_type = luisa::compute::Type::structure(frame_member_types);

    // Determine next-token for each suspend. Use the scope index + 1 as the token
    // (0 = terminated, 1 = entry, 2+ = continuations).
    luisa::unordered_map<uint32_t, uint32_t> suspend_to_next_token;
    for (auto &[token, scope_ref] : graph.token_to_scope) {
        suspend_to_next_token.emplace(token, static_cast<uint32_t>(scope_ref.index + 1));
    }

    // Emit one callable per scope.
    for (size_t si = 0; si < graph.scopes.size(); ++si) {
        auto callable = module->create_callable(nullptr);
        auto frame_ref = callable->create_reference_argument(result.split_info.frame_type);
        // Mirror source function arguments.
        luisa::vector<Argument *> mirrored;
        for (auto src_arg : function->arguments()) {
            mirrored.emplace_back(mirror_argument(callable, src_arg));
        }
        auto block = callable->create_body_block();

        ScopeMaterializer mat;
        mat.module = module;
        mat.source_function = function;
        mat.graph = &graph;
        mat.split_info = &result.split_info;
        mat.callable = callable;
        mat.frame_ref = frame_ref;
        mat.builder.set_insertion_point(block);
        mat.resolver.set_builder(&mat.builder);

        // Map source arguments to mirrored.
        {
            size_t i = 0;
            for (auto src_arg : function->arguments()) {
                mat.resolver.emplace(src_arg, mirrored[i]);
                ++i;
            }
        }
        // Map frame slot allocas to GEPs.
        for (auto &slot : result.split_info.frame_slots) {
            auto idx = uint_constant(module, static_cast<uint32_t>(slot.field_index));
            auto gep = mat.builder.gep(slot.type, frame_ref, {idx});
            mat.resolver.emplace(slot.source_alloca, gep);
        }
        // GEP for target_token (slot 0).
        auto state_idx = uint_constant(module, 0u);
        mat.state_gep = mat.builder.gep(luisa::compute::Type::of<uint32_t>(), frame_ref, {state_idx});

        // Determine next_token for suspends in this scope.
        // For simplicity: use the first suspend token found in the scope's instructions.
        uint32_t next_token = 0u;
        for (auto ref : graph.scopes[si].instructions) {
            auto &instr = graph.preliminary.instructions[ref.index];
            if (instr.tag == CoroInstruction::Tag::SUSPEND) {
                auto it = suspend_to_next_token.find(instr.suspend_token);
                if (it != suspend_to_next_token.end()) { next_token = it->second; }
                break;
            }
        }

        // Materialize the scope.
        mat.materialize_scope(CoroScopeRef{si}, next_token);

        // If the block isn't terminated, add a default terminate.
        if (!block->is_terminated()) {
            mat.builder.store(mat.state_gep, uint_constant(module, 0u));
            mat.builder.return_void();
        }

        CoroutineSplitContinuation cont{
            .id = si,
            .callable = callable,
            .outgoing_suspends = {},
        };
        // Collect outgoing suspend tokens from this scope.
        for (auto ref : graph.scopes[si].instructions) {
            auto &instr = graph.preliminary.instructions[ref.index];
            if (instr.tag == CoroInstruction::Tag::SUSPEND) {
                cont.outgoing_suspends.emplace_back(instr.suspend_token);
            }
        }
        result.split_info.continuations.emplace_back(std::move(cont));
    }

    result.split_info.is_supported = true;
    result.split_info.changed = true;
    result.ok = true;
    return result;
}

}// namespace luisa::compute::xir::coro
