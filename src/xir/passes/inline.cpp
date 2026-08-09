#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/builder.h>
#include <luisa/core/logging.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/function.h>
#include <luisa/xir/value.h>
#include <luisa/xir/metadata/signature_constraint.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static luisa::vector<CallInst *> collect_call_sites(Function *callee) noexcept {
    luisa::vector<CallInst *> calls;
    for (auto &&use : callee->use_list()) {
        if (auto user = use->user(); user != nullptr && user->isa<CallInst>()) {
            auto call = static_cast<CallInst *>(user);
            if (call->callee() == callee &&
                use == call->operand_use(CallInst::operand_index_callee)) {
                calls.push_back(call);
            }
        }
    }
    return calls;
}

class InlineValueResolver final : public InstructionCloneValueResolver {
    luisa::unordered_map<const Value *, Value *> _map;
    Module *_module;
public:
    explicit InlineValueResolver(Function *caller_func) noexcept
        : _module{caller_func->parent_module()} {}
    void emplace(const Value *from, Value *to) noexcept { _map.emplace(from, to); }
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
        auto it = _map.find(value);
        if (it == _map.end()) {
            if (value->derived_value_tag() == DerivedValueTag::BASIC_BLOCK) {
                return nullptr;
            }
            if (value->type() != nullptr) {
                auto undef = _module->create_undefined(value->type());
                _map.emplace(value, undef);
                return undef;
            }
            LUISA_ERROR("Inline: unresolved value (tag={}).", to_string(value->derived_value_tag()));
        }
        return it->second;
    }
};

[[nodiscard]] static size_t count_instructions(FunctionDefinition *def) noexcept {
    size_t n = 0;
    if (def) def->traverse_instructions([&](const Instruction *) noexcept { ++n; });
    return n;
}

[[nodiscard]] static bool has_single_block(FunctionDefinition *def) noexcept {
    size_t count = 0u;
    for (auto *block : def->basic_blocks()) {
        static_cast<void>(block);
        ++count;
    }
    return count == 1u && def->body_block() != nullptr;
}

[[nodiscard]] static bool can_inline_single_block(
    FunctionDefinition *def) noexcept {
    if (!has_single_block(def)) { return false; }
    auto *block = def->body_block();
    if (!block->is_terminated() ||
        !block->terminator()->isa<ReturnInst>()) {
        return false;
    }
    for (auto *inst : block->instructions()) {
        if ((inst->is_terminator() && !inst->isa<ReturnInst>()) ||
            inst->isa<PhiInst>()) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] static bool contains_inline_barrier(FunctionDefinition *def,
                                                  bool allow_autodiff_scope) noexcept {
    for (auto *block : def->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::IF:
                case DerivedInstructionTag::SWITCH:
                case DerivedInstructionTag::LOOP:
                case DerivedInstructionTag::SIMPLE_LOOP:
                case DerivedInstructionTag::BREAK:
                case DerivedInstructionTag::CONTINUE:
                case DerivedInstructionTag::RAY_QUERY_LOOP:
                case DerivedInstructionTag::RAY_QUERY_DISPATCH:
                case DerivedInstructionTag::OUTLINE:
                case DerivedInstructionTag::CORO_SUSPEND:
                case DerivedInstructionTag::CORO_RESUME:
                case DerivedInstructionTag::CORO_TERMINATE:
                    return true;
                case DerivedInstructionTag::AUTODIFF_SCOPE:
                    if (!allow_autodiff_scope) { return true; }
                    break;
                default: break;
            }
        }
    }
    return false;
}

[[nodiscard]] static bool typed_value_operand_valid(const Value *value) noexcept {
    return value != nullptr && value->type() != nullptr &&
           !value->isa<BasicBlock>() && !value->isa<Function>() &&
           !value->type()->is_resource();
}

[[nodiscard]] static bool rvalue_operand_valid(const Value *value) noexcept {
    return typed_value_operand_valid(value) && !value->is_lvalue();
}

[[nodiscard]] static bool argument_matches(const Argument *formal,
                                           const Value *actual) noexcept {
    if (formal == nullptr || actual == nullptr ||
        actual->type() != formal->type()) {
        return false;
    }
    if (formal->is_resource()) {
        return actual->isa<ResourceArgument>() && !actual->is_lvalue();
    }
    if (formal->is_reference()) {
        return typed_value_operand_valid(actual) && actual->is_lvalue();
    }
    return rvalue_operand_valid(actual);
}

struct InlineFunctionSummary {
    bool has_valid_definition{false};
    bool has_single_block{false};
    bool can_inline_single_block{false};
    bool return_shape_is_valid{false};
    bool has_return_metadata{false};
    bool has_single_body_metadata{false};
    bool contains_barrier_disallow_autodiff{false};
    bool contains_barrier_allow_autodiff{false};
};

[[nodiscard]] static InlineFunctionSummary summarize_inline_function(
    Function *function, InlineInfo &info) noexcept {
    InlineFunctionSummary summary;
    auto *definition =
        function == nullptr ? nullptr : function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        return summary;
    }
    summary.has_valid_definition = true;
    ++info.call_site_summary_function_count;
    auto block_count = size_t{0u};
    auto return_count = size_t{0u};
    auto single_block_forbidden = false;
    auto return_shape_is_valid = true;
    for (auto *block : definition->basic_blocks()) {
        ++block_count;
        for (auto *inst : block->instructions()) {
            ++info.call_site_summary_instruction_scan_count;
            single_block_forbidden |=
                (inst->is_terminator() &&
                 !inst->isa<ReturnInst>()) ||
                inst->isa<PhiInst>();
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::IF:
                case DerivedInstructionTag::SWITCH:
                case DerivedInstructionTag::LOOP:
                case DerivedInstructionTag::SIMPLE_LOOP:
                case DerivedInstructionTag::BREAK:
                case DerivedInstructionTag::CONTINUE:
                case DerivedInstructionTag::RAY_QUERY_LOOP:
                case DerivedInstructionTag::RAY_QUERY_DISPATCH:
                case DerivedInstructionTag::OUTLINE:
                case DerivedInstructionTag::CORO_SUSPEND:
                case DerivedInstructionTag::CORO_RESUME:
                case DerivedInstructionTag::CORO_TERMINATE:
                    summary.contains_barrier_disallow_autodiff = true;
                    summary.contains_barrier_allow_autodiff = true;
                    break;
                case DerivedInstructionTag::AUTODIFF_SCOPE:
                    summary.contains_barrier_disallow_autodiff = true;
                    break;
                default: break;
            }
            if (!inst->isa<ReturnInst>()) { continue; }
            auto *return_inst = static_cast<ReturnInst *>(inst);
            auto *return_value = return_inst->return_value();
            return_shape_is_valid &=
                (function->type() == nullptr) ==
                    (return_value == nullptr) &&
                (return_value == nullptr ||
                 return_value->type() == function->type());
            summary.has_return_metadata |=
                !inst->metadata_list().empty();
            ++return_count;
        }
    }
    summary.has_single_block = block_count == 1u;
    if (summary.has_single_block) {
        auto *body = definition->body_block();
        summary.has_single_body_metadata =
            !body->metadata_list().empty();
        summary.can_inline_single_block =
            body->is_terminated() &&
            body->terminator()->isa<ReturnInst>() &&
            !single_block_forbidden;
    }
    summary.return_shape_is_valid =
        return_shape_is_valid &&
        (function->type() == nullptr || return_count != 0u);
    return summary;
}

[[nodiscard]] static bool validate_call_shape(
    CallInst *call, Function *callee,
    const InlineFunctionSummary &summary) noexcept {
    if (call->type() != callee->type() ||
        call->argument_count() !=
            callee->arguments().count_size() ||
        !summary.has_valid_definition ||
        !summary.return_shape_is_valid) {
        return false;
    }
    auto argument_index = 0u;
    for (auto *formal : callee->arguments()) {
        if (!argument_matches(
                formal, call->argument(argument_index++))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] static bool has_unmappable_inline_metadata(
    CallInst *call,
    const InlineFunctionSummary &summary) noexcept {
    if (call == nullptr || !summary.has_valid_definition ||
        !call->metadata_list().empty() ||
        summary.has_return_metadata) {
        return true;
    }
    if (summary.has_single_block) {
        return summary.has_single_body_metadata;
    }
    auto *call_block = call->parent_block();
    return call_block == nullptr ||
           !call_block->metadata_list().empty();
}

[[nodiscard]] static bool validate_call_shape(CallInst *call,
                                              Function *callee) noexcept {
    if (call->type() != callee->type()) { return false; }
    if (call->argument_count() != callee->arguments().count_size()) { return false; }
    auto argument_index = 0u;
    for (auto *formal : callee->arguments()) {
        auto *actual = call->argument(argument_index++);
        if (!argument_matches(formal, actual)) { return false; }
    }
    auto *definition = callee->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        return false;
    }
    auto return_count = 0u;
    for (auto *block : definition->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            if (!inst->isa<ReturnInst>()) { continue; }
            auto *return_inst = static_cast<ReturnInst *>(inst);
            auto *return_value = return_inst->return_value();
            if ((call->type() == nullptr) != (return_value == nullptr)) { return false; }
            if (return_value != nullptr && return_value->type() != call->type()) { return false; }
            return_count++;
        }
    }
    return call->type() == nullptr || return_count != 0u;
}

[[nodiscard]] static bool has_unmappable_inline_metadata(
    CallInst *call, FunctionDefinition *callee_def) noexcept {
    if (call == nullptr || callee_def == nullptr) { return true; }
    if (!call->metadata_list().empty()) { return true; }
    if (has_single_block(callee_def)) {
        // Single-block inlining splices instructions into the caller's
        // existing block. The callee block itself has no one-to-one
        // replacement, and merging its metadata into the caller block can
        // create duplicate metadata kinds or change the annotation's scope.
        if (auto *body = callee_def->body_block();
            body != nullptr && !body->metadata_list().empty()) {
            return true;
        }
    } else {
        auto *call_block = call->parent_block();
        if (call_block == nullptr ||
            !call_block->metadata_list().empty()) {
            return true;
        }
    }
    for (auto *block : callee_def->basic_blocks()) {
        for (auto *inst : block->instructions()) {
            if (inst->isa<ReturnInst>() &&
                !inst->metadata_list().empty()) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] static bool inline_single_block_call(CallInst *call,
                                                   Function *callee,
                                                   bool prevalidated = false) noexcept {
    auto *callee_def = callee->definition();
    auto *caller = call->parent_function();
    if (callee_def == nullptr || caller == nullptr ||
        (!prevalidated && !can_inline_single_block(callee_def))) {
        return false;
    }
    auto *block = callee_def->body_block();
    XIRBuilder builder;
    builder.set_insertion_point(call);
    InlineValueResolver resolver{caller};
    auto *module = caller->parent_module();
    size_t i = 0u;
    for (auto *arg : callee->arguments()) {
        auto *call_arg = i < call->argument_count() ?
                             call->argument(i) :
                             static_cast<Value *>(module->create_undefined(arg->type()));
        if (arg->is_lvalue() && !call_arg->is_lvalue()) {
            auto *tmp = builder.alloca_local(arg->type());
            builder.store(tmp, call_arg);
            resolver.emplace(arg, tmp);
        } else {
            resolver.emplace(arg, call_arg);
        }
        ++i;
    }
    for (auto *inst : block->instructions()) {
        if (inst->isa<ReturnInst>()) {
            auto *ret = static_cast<ReturnInst *>(inst);
            if (call->type() != nullptr) {
                if (ret->return_value() == nullptr) { return false; }
                call->replace_all_uses_with(resolver.resolve(ret->return_value()));
            }
            call->remove_self();
            return true;
        }
        auto *clone = inst->clone_with_metadata(builder, resolver);
        LUISA_ASSERT(clone != nullptr, "Inline: clone failed.");
        resolver.emplace(inst, clone);
    }
    return false;
}

[[nodiscard]] static bool inline_multi_block_call(CallInst *call, Function *callee) noexcept {
    auto callee_def = callee->definition();
    if (!callee_def) return false;
    auto caller_func = call->parent_function();
    if (!caller_func) return false;
    auto caller_def = caller_func->definition();
    if (!caller_def) return false;

    auto call_block = call->parent_block();
    luisa::vector<Instruction *> to_move;
    auto past_call = false;
    for (auto *inst : call_block->instructions()) {
        if (inst == call) {
            past_call = true;
        } else if (past_call) {
            to_move.emplace_back(inst);
        }
    }

    auto module = caller_func->parent_module();
    XIRBuilder builder;
    InlineValueResolver resolver{caller_func};

    // Map callee args -> call args
    {
        size_t i = 0;
        for (auto arg : callee->arguments()) {
            auto call_arg = i < call->argument_count() ? call->argument(i) : static_cast<Value *>(module->create_undefined(arg->type()));
            if (arg->is_lvalue() && !call_arg->is_lvalue()) {
                builder.set_insertion_point(call);
                auto tmp = builder.alloca_local(arg->type());
                builder.store(tmp, call_arg);
                resolver.emplace(arg, tmp);
            } else {
                resolver.emplace(arg, call_arg);
            }
            ++i;
        }
    }

    // Collect reachable callee blocks in RPO for instruction cloning.
    luisa::vector<BasicBlock *> callee_blocks;
    callee_def->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER, [&](BasicBlock *bb) noexcept { callee_blocks.push_back(bb); });
    luisa::unordered_set<const BasicBlock *> callee_reachable{
        callee_blocks.begin(), callee_blocks.end()};

    luisa::unordered_map<BasicBlock *, BasicBlock *> block_map;
    luisa::vector<BasicBlock *> new_blocks;
    for (auto bb : callee_blocks) {
        auto nb = caller_func->create_basic_block();
        for (auto *metadata : bb->metadata_list()) {
            nb->metadata_list().push_front(metadata->clone());
        }
        block_map[bb] = nb;
        new_blocks.push_back(nb);
        resolver.emplace(bb, nb);
    }

    // Create single-exit merge block and return value alloca
    auto merge_bb = caller_func->create_basic_block();

    // Map unreachable blocks to dedicated empty blocks so structured
    // terminators (IfInst, LoopInst) referencing them get valid targets.
    {
        for (auto bb : callee_def->basic_blocks()) {
            if (!callee_reachable.contains(bb)) {
                auto nb = caller_func->create_basic_block();
                for (auto *metadata : bb->metadata_list()) {
                    nb->metadata_list().push_front(metadata->clone());
                }
                block_map[bb] = nb;
                resolver.emplace(bb, nb);
                builder.set_insertion_point(nb);
                builder.unreachable_();
            }
        }
    }
    Instruction *ret_alloca = nullptr;
    if (call->type()) {
        builder.set_insertion_point(call);
        ret_alloca = builder.alloca_local(call->type());
    }

    // Clone instructions from callee into new blocks.
    // We make two passes:
    //   Pass 1: clone all alloca instructions first. They have no operand
    //   dependencies and may be referenced by instructions that appear
    //   earlier in RPO (e.g., alloca inside a branch referenced from a
    //   predecessor block after previous inlining).
    //   Pass 2: clone everything else.
    luisa::vector<std::pair<const PhiInst *, PhiInst *>> phi_nodes;
    for (size_t i = 0; i < callee_blocks.size(); ++i) {
        builder.set_insertion_point(new_blocks[i]);
        for (auto inst : callee_blocks[i]->instructions()) {
            if (inst->isa<AllocaInst>()) {
                auto c = inst->clone_with_metadata(builder, resolver);
                LUISA_ASSERT(c, "Inline: clone failed.");
                resolver.emplace(inst, c);
            }
        }
    }
    for (size_t i = 0; i < callee_blocks.size(); ++i) {
        builder.set_insertion_point(new_blocks[i]);
        for (auto inst : callee_blocks[i]->instructions()) {
            if (inst->isa<ReturnInst>()) {
                auto r = static_cast<ReturnInst *>(inst);
                if (ret_alloca && r->operand_count() > 0) {
                    auto val = resolver.resolve(r->operand(0));
                    builder.store(ret_alloca, val);
                }
                builder.br(merge_bb);
            } else if (inst->isa<PhiInst>()) {
                auto phi = static_cast<PhiInst *>(inst);
                auto dup_phi = builder.phi(phi->type());
                for (auto *metadata : phi->metadata_list()) {
                    dup_phi->metadata_list().push_front(metadata->clone());
                }
                phi_nodes.emplace_back(phi, dup_phi);
                resolver.emplace(inst, dup_phi);
            } else if (!inst->isa<AllocaInst>()) {
                auto c = inst->clone_with_metadata(builder, resolver);
                LUISA_ASSERT(c, "Inline: clone failed.");
                resolver.emplace(inst, c);
            }
        }
    }
    // Patch phi node operands now that all blocks and values are mapped.
    for (auto [original_phi, dup_phi] : phi_nodes) {
        // Only executable callee blocks were cloned. Disconnected owned
        // blocks are represented by terminal empty shells, so their original
        // outgoing edges no longer exist and must not survive as Phi labels.
        // Keeping those labels creates an incoming-without-predecessor pair.
        for (size_t i = 0; i < original_phi->incoming_count(); i++) {
            auto incoming = original_phi->incoming(i);
            if (!callee_reachable.contains(incoming.block)) { continue; }
            auto resolved_value = resolver.resolve(incoming.value);
            auto resolved_block = resolver.resolve(incoming.block);
            dup_phi->add_incoming(
                resolved_value, static_cast<BasicBlock *>(resolved_block));
        }
        if (original_phi->parent_block() == callee_def->body_block()) {
            // A function entry is reached by an implicit invocation edge,
            // which is not represented in a standalone function's Phi list.
            // Inlining materializes that edge as call_block -> cloned entry.
            // The entry value on that formerly implicit edge is undefined.
            dup_phi->add_incoming(
                module->create_undefined(original_phi->type()), call_block);
        }
    }

    // Wire caller: split the call block
    auto entry_block = block_map[callee_def->body_block()];
    luisa::vector<BasicBlock *> original_successors;
    if (call_block->is_terminated()) {
        call_block->traverse_successors(
            false, [&](BasicBlock *successor) noexcept {
                original_successors.emplace_back(successor);
            });
    }

    // Load return value in merge block
    if (ret_alloca) {
        builder.set_insertion_point(merge_bb);
        auto loaded = builder.load(call->type(), ret_alloca);
        call->replace_all_uses_with(loaded);
    }

    // Remove the call
    call->remove_self();

    // Move post-call instructions to merge_bb
    builder.set_insertion_point(merge_bb);
    for (auto inst : to_move) {
        if (!inst->is_terminator()) {
            auto m = inst->remove_self();
            builder.append(std::move(m));
        }
    }

    // Move terminator from call_block to merge_bb
    if (call_block->is_terminated()) {
        auto m = call_block->terminator()->remove_self();
        builder.set_insertion_point(merge_bb);
        if (merge_bb->is_terminated()) merge_bb->terminator()->remove_self();
        builder.append(std::move(m));
        // Moving the terminator transfers every original outgoing edge from
        // call_block to merge_bb. Phi incoming labels describe predecessor
        // edges, so their labels must move with those edges. This also covers
        // duplicate branch targets and a former self-edge to call_block.
        for (auto *successor : original_successors) {
            for (auto *inst : successor->instructions()) {
                if (!inst->isa<PhiInst>()) { break; }
                auto *phi = static_cast<PhiInst *>(inst);
                for (auto i = 0u; i < phi->incoming_count(); ++i) {
                    auto incoming = phi->incoming(i);
                    if (incoming.block == call_block) {
                        phi->set_incoming(i, incoming.value, merge_bb);
                    }
                }
            }
        }
    }

    // Branch from call_block to inlined entry
    builder.set_insertion_point(call_block);
    builder.br(entry_block);

    // Defensive: if merge_bb has no terminator (can happen when call_block
    // was already unterminated in malformed IR), add unreachable.
    if (!merge_bb->is_terminated()) {
        builder.set_insertion_point(merge_bb);
        builder.unreachable_();
    }

    return true;
}

[[nodiscard]] static bool inline_call(CallInst *call, Function *callee,
                                      InlineInfo &info,
                                      InlineOptions options = {},
                                      luisa::unordered_set<CallInst *> *reported_malformed_calls = nullptr) noexcept {
    auto *callee_def = callee->definition();
    auto *caller = call->parent_function();
    auto *caller_def = caller == nullptr ? nullptr : caller->definition();
    if (callee_def == nullptr || caller_def == nullptr) { return false; }
    if (callee_def->body_block() == nullptr) {
        ++info.skipped_declaration_call_count;
        return false;
    }
    if (!validate_call_shape(call, callee)) {
        if (reported_malformed_calls == nullptr ||
            reported_malformed_calls->emplace(call).second) {
            ++info.rejected_malformed_call_count;
        }
        return false;
    }
    if (callee->find_metadata<SignatureConstraintMD>() != nullptr) {
        ++info.skipped_constrained_call_count;
        return false;
    }
    if (has_unmappable_inline_metadata(call, callee_def)) {
        ++info.skipped_metadata_call_count;
        return false;
    }
    if (contains_inline_barrier(callee_def, false)) {
        ++info.skipped_structured_call_count;
        return false;
    }
    if (has_single_block(callee_def)) {
        return inline_single_block_call(call, callee);
    }
    if (contains_inline_barrier(caller_def, options.allow_autodiff_scope_in_caller)) {
        ++info.skipped_structured_call_count;
        return false;
    }
    return inline_multi_block_call(call, callee);
}

[[nodiscard]] static luisa::unordered_set<Function *>
find_recursive_callables(luisa::span<Function *const> callables) noexcept {
    luisa::unordered_set<Function *> callable_set;
    callable_set.reserve(callables.size());
    for (auto *callable : callables) { callable_set.emplace(callable); }
    luisa::unordered_map<Function *, luisa::vector<Function *>> edges;
    for (auto *function : callables) {
        if (auto *def = function->definition()) {
            // collect_call_sites() observes every owned CallInst through the
            // callee use list. Recursion discovery must use the same domain;
            // otherwise a self-call in a disconnected block can be mistaken
            // for a non-recursive inlining candidate.
            for (auto *block : def->basic_blocks()) {
                for (auto *inst : block->instructions()) {
                    if (inst->isa<CallInst>()) {
                        auto *callee =
                            static_cast<CallInst *>(inst)->callee();
                        if (callee != nullptr &&
                            callable_set.contains(callee)) {
                            edges[function].emplace_back(callee);
                        }
                    }
                }
            }
        }
    }
    luisa::unordered_set<Function *> recursive;
    for (auto *start : callables) {
        luisa::unordered_set<Function *> visited;
        luisa::vector<Function *> worklist{start};
        while (!worklist.empty()) {
            auto *current = worklist.back();
            worklist.pop_back();
            if (!visited.emplace(current).second) { continue; }
            for (auto *next : edges[current]) {
                if (next == start) {
                    recursive.emplace(start);
                    worklist.clear();
                    break;
                }
                worklist.emplace_back(next);
            }
        }
    }
    return recursive;
}

static void run(Module *module, InlineInfo &info) noexcept {
    if (module == nullptr) { return; }
    // Early exit if no callables
    bool has_callables = false;
    for (auto f : module->function_list()) {
        if (f->derived_function_tag() == DerivedFunctionTag::CALLABLE) {
            has_callables = true;
            break;
        }
    }
    if (!has_callables) return;

    // Collect callables (safe iteration before modification)
    luisa::vector<Function *> callables;
    for (auto f : module->function_list())
        if (f->derived_function_tag() == DerivedFunctionTag::CALLABLE)
            callables.push_back(f);

    auto recursive = find_recursive_callables(callables);

    // Defer removal to after iteration to avoid corrupting the list
    luisa::vector<Function *> to_remove;
    for (auto callee : callables) {
        auto def = callee->definition();
        if (!def) continue;
        if (recursive.contains(callee)) {
            ++info.skipped_recursive_callable_count;
            continue;
        }
        auto edges = collect_call_sites(callee);
        if (edges.empty()) continue;

        size_t n = edges.size();
        bool doit = (n == 1) || (n <= 3 && count_instructions(def) <= 50);
        if (!doit) continue;

        for (auto call : edges)
            if (inline_call(call, callee, info))
                info.inlined_call_count++;

        if (callee->use_list().empty()) { to_remove.push_back(callee); }
    }
    for (auto callee : to_remove) {
        callee->remove_self();
        info.removed_callable_count++;
    }
}

}// namespace detail

namespace {

void set_inline_report(const InlineInfo &info, PassReport *report) noexcept {
    if (report == nullptr) { return; }
    report->set("inlined_call", info.inlined_call_count);
    report->set("removed_callable", info.removed_callable_count);
    report->set("skipped_recursive_callable",
                info.skipped_recursive_callable_count);
    report->set("skipped_structured_call",
                info.skipped_structured_call_count);
    report->set("skipped_constrained_call",
                info.skipped_constrained_call_count);
    report->set("skipped_metadata_call",
                info.skipped_metadata_call_count);
    report->set("skipped_declaration_call",
                info.skipped_declaration_call_count);
    report->set("rejected_malformed_call",
                info.rejected_malformed_call_count);
    report->set("call_site_summary_function",
                info.call_site_summary_function_count);
    report->set("call_site_summary_instruction_scan",
                info.call_site_summary_instruction_scan_count);
    report->set("call_site_cached_apply",
                info.call_site_cached_apply_count);
    report->set("call_site_revalidated_apply",
                info.call_site_revalidated_apply_count);
}

}// namespace

InlineInfo inline_pass_run_on_module(Module *module, PassReport *report) noexcept {
    InlineInfo info;
    if (module != nullptr) {
        detail::run(module, info);
    }
    set_inline_report(info, report);
    return info;
}

InlineInfo inline_all_pass_run_on_module(Module *module, PassReport *report) noexcept {
    return inline_all_pass_run_on_module(module, {}, report);
}

InlineInfo inline_all_pass_run_on_module(Module *module, InlineOptions options, PassReport *report) noexcept {
    InlineInfo info;
    if (!module) {
        set_inline_report(info, report);
        return info;
    }
    luisa::unordered_set<CallInst *> reported_malformed_calls;
    for (;;) {
        luisa::vector<Function *> callables;
        for (auto f : module->function_list())
            if (f->derived_function_tag() == DerivedFunctionTag::CALLABLE)
                callables.push_back(f);
        if (callables.empty()) break;
        auto recursive = detail::find_recursive_callables(callables);
        luisa::unordered_set<Function *> callable_set{callables.begin(), callables.end()};
        luisa::vector<Function *> leaves;
        for (auto callee : callables) {
            if (recursive.contains(callee)) {
                ++info.skipped_recursive_callable_count;
                continue;
            }
            auto def = callee->definition();
            if (!def) continue;
            bool is_leaf = true;
            for (auto *block : def->basic_blocks()) {
                for (auto *inst : block->instructions()) {
                    if (!is_leaf) { break; }
                    if (inst->derived_instruction_tag() ==
                        DerivedInstructionTag::CALL) {
                        auto *call = static_cast<const CallInst *>(inst);
                        if (callable_set.contains(const_cast<Function *>(
                                static_cast<const Function *>(
                                    call->callee())))) {
                            is_leaf = false;
                        }
                    }
                }
                if (!is_leaf) { break; }
            }
            if (is_leaf) leaves.push_back(callee);
        }
        if (leaves.empty()) break;
        bool progress = false;
        for (auto callee : leaves) {
            auto def = callee->definition();
            if (!def) continue;
            auto edges = detail::collect_call_sites(callee);
            for (auto call : edges) {
                if (detail::inline_call(call, callee, info, options, &reported_malformed_calls)) {
                    info.inlined_call_count++;
                    progress = true;
                }
            }
        }
        if (!progress) break;
        for (auto *callee : leaves) {
            if (callee->use_list().empty()) {
                callee->remove_self();
                ++info.removed_callable_count;
            }
        }
    }
    set_inline_report(info, report);
    return info;
}

InlineInfo inline_call_sites_pass_run_on_module(
    Module *module, luisa::span<CallInst *const> call_sites,
    InlineOptions options, PassReport *report) noexcept {
    InlineInfo info;
    if (module == nullptr || call_sites.empty()) {
        set_inline_report(info, report);
        return info;
    }
    luisa::unordered_set<CallInst *> reported_malformed_calls;
    luisa::vector<Function *> all_callables;
    for (auto *function : module->function_list()) {
        if (function->derived_function_tag() ==
            DerivedFunctionTag::CALLABLE) {
            all_callables.emplace_back(function);
        }
    }
    auto recursive = detail::find_recursive_callables(all_callables);
    luisa::unordered_map<Function *, detail::InlineFunctionSummary>
        function_summaries;
    auto summary_of = [&](Function *function) noexcept {
        if (auto iter = function_summaries.find(function);
            iter != function_summaries.end()) {
            return iter->second;
        }
        auto summary =
            detail::summarize_inline_function(function, info);
        function_summaries.emplace(function, summary);
        return summary;
    };
    struct PreparedInlineCall {
        CallInst *call;
        Function *callee;
        bool single_block;
    };
    luisa::unordered_set<Function *> reported_recursive;
    luisa::unordered_set<CallInst *> seen_calls;
    luisa::vector<PreparedInlineCall> plan;
    plan.reserve(call_sites.size());
    for (auto *call : call_sites) {
        if (call == nullptr) {
            ++info.rejected_malformed_call_count;
            continue;
        }
        if (!seen_calls.emplace(call).second) { continue; }
        auto *callee = call->callee();
        auto *caller = call->parent_function();
        auto malformed = callee == nullptr || caller == nullptr ||
                         caller->parent_module() != module ||
                         callee->parent_module() != module ||
                         callee->derived_function_tag() !=
                             DerivedFunctionTag::CALLABLE ||
                         callee->definition() == nullptr;
        if (!malformed &&
            callee->definition()->body_block() == nullptr) {
            ++info.skipped_declaration_call_count;
            continue;
        }
        auto callee_summary = malformed ?
                                  detail::InlineFunctionSummary{} :
                                  summary_of(callee);
        malformed |= !malformed &&
                     !detail::validate_call_shape(
                         call, callee, callee_summary);
        if (!malformed && callee_summary.has_single_block &&
            !callee_summary.can_inline_single_block) {
            malformed = true;
        }
        if (malformed) {
            ++info.rejected_malformed_call_count;
            continue;
        }
        if (recursive.contains(callee)) {
            if (reported_recursive.emplace(callee).second) {
                ++info.skipped_recursive_callable_count;
            }
            continue;
        }
        if (callee->find_metadata<SignatureConstraintMD>() != nullptr) {
            ++info.skipped_constrained_call_count;
            continue;
        }
        if (detail::has_unmappable_inline_metadata(
                call, callee_summary)) {
            ++info.skipped_metadata_call_count;
            continue;
        }
        auto caller_contains_barrier = false;
        if (!callee_summary.has_single_block) {
            auto caller_summary = summary_of(caller);
            caller_contains_barrier =
                options.allow_autodiff_scope_in_caller ?
                    caller_summary.contains_barrier_allow_autodiff :
                    caller_summary.contains_barrier_disallow_autodiff;
        }
        if (callee_summary.contains_barrier_disallow_autodiff ||
            caller_contains_barrier) {
            ++info.skipped_structured_call_count;
            continue;
        }
        plan.emplace_back(call, callee,
                          callee_summary.has_single_block);
    }
    if (info.rejected_malformed_call_count != 0u ||
        info.skipped_recursive_callable_count != 0u ||
        info.skipped_structured_call_count != 0u ||
        info.skipped_constrained_call_count != 0u ||
        info.skipped_metadata_call_count != 0u ||
        info.skipped_declaration_call_count != 0u ||
        plan.size() != seen_calls.size()) {
        set_inline_report(info, report);
        return info;
    }
    // Every summary above describes an immutable function definition. An
    // inline operation mutates only its caller, so a prepared callee remains
    // valid unless that function was itself an earlier caller in this plan.
    // Track exactly that invalidation frontier: independent call sites reuse
    // their preflight decision, while nested call chains retain the complete
    // generic validation path after their callee changes.
    luisa::unordered_set<Function *> mutated_functions;
    for (auto &&prepared : plan) {
        auto *call = prepared.call;
        auto *callee = prepared.callee;
        auto *caller = call->parent_function();
        auto revalidate = mutated_functions.contains(callee);
        auto succeeded = false;
        if (revalidate) {
            ++info.call_site_revalidated_apply_count;
            succeeded = detail::inline_call(
                call, callee, info, options,
                &reported_malformed_calls);
        } else {
            ++info.call_site_cached_apply_count;
            succeeded = prepared.single_block ?
                            detail::inline_single_block_call(
                                call, callee, true) :
                            detail::inline_multi_block_call(
                                call, callee);
        }
        if (!succeeded) {
            LUISA_ERROR_WITH_LOCATION(
                "Inline call-site plan changed after successful preflight.");
        }
        mutated_functions.emplace(caller);
        ++info.inlined_call_count;
    }
    luisa::unordered_set<Function *> planned_callees;
    for (auto &&prepared : plan) {
        planned_callees.emplace(prepared.callee);
    }
    luisa::vector<Function *> unused_callables;
    for (auto *function : module->function_list()) {
        if (planned_callees.contains(function) &&
            function->use_list().empty()) {
            unused_callables.emplace_back(function);
        }
    }
    for (auto *callee : unused_callables) {
        callee->remove_self();
        ++info.removed_callable_count;
    }
    set_inline_report(info, report);
    return info;
}

}// namespace luisa::compute::xir
