#include <luisa/xir/module.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/passes/resource_origin.h>

namespace luisa::compute::xir {

UniqueResourceOriginMap analyze_unique_resource_origins(const Module *module) noexcept {
    UniqueResourceOriginMap origins;
    if (module == nullptr) { return origins; }

    struct OriginState {
        const Argument *origin{nullptr};
        luisa::vector<const Argument *> actuals;
        bool conflicting{false};
    };
    luisa::unordered_map<const Argument *, OriginState> states;
    luisa::unordered_map<const User *, const Function *> owned_calls;
    for (auto *function : module->function_list()) {
        if (function->parent_module() != module) { continue; }
        for (auto *argument : function->arguments()) {
            if (argument->parent_function() != function ||
                !argument->is_resource() || argument->type() == nullptr) {
                continue;
            }
            switch (function->derived_function_tag()) {
                case DerivedFunctionTag::KERNEL: origins.emplace(argument, argument); break;
                case DerivedFunctionTag::CALLABLE: states.emplace(argument, OriginState{}); break;
                default: break;
            }
        }
        // Do not require a complete CFG: owned orphan blocks are conservative
        // incoming edges too, and may contain calls in partially built IR.
        for (auto *block : function->basic_blocks()) {
            if (block->parent_function() != function) { continue; }
            for (auto *inst : block->instructions()) {
                if (inst->parent_block() == block &&
                    (inst->isa<CallInst>() || inst->isa<RayQueryPipelineInst>())) {
                    owned_calls.emplace(inst, function);
                }
            }
        }
    }

    for (auto *function : module->function_list()) {
        if (function->parent_module() != module ||
            function->derived_function_tag() != DerivedFunctionTag::CALLABLE) {
            continue;
        }
        auto reject_function = [&]() noexcept {
            for (auto *formal : function->arguments()) {
                if (auto iter = states.find(formal); iter != states.end()) {
                    iter->second.conflicting = true;
                }
            }
        };
        auto formal_count = function->arguments().count_size();
        for (auto *use : function->use_list()) {
            auto *user = use->user();
            auto caller = owned_calls.find(user);
            if (caller == owned_calls.end()) {
                // Escaped function values and detached/out-of-module users
                // leave the set of possible actual arguments unknown.
                reject_function();
                break;
            }
            size_t operand_offset = 0u;
            size_t formal_offset = 0u;
            if (user->isa<CallInst>() &&
                user->operand_count() >= CallInst::operand_index_argument_offset &&
                use == user->operand_use(CallInst::operand_index_callee)) {
                operand_offset = CallInst::operand_index_argument_offset;
            } else if (user->isa<RayQueryPipelineInst>() &&
                       user->operand_count() >= RayQueryPipelineInst::operand_index_offset_captured_arguments &&
                       (use == user->operand_use(RayQueryPipelineInst::operand_index_on_surface_function) ||
                        use == user->operand_use(RayQueryPipelineInst::operand_index_on_procedural_function))) {
                operand_offset = RayQueryPipelineInst::operand_index_offset_captured_arguments;
                formal_offset = 1u;
                auto *query = user->operand(RayQueryPipelineInst::operand_index_query_object);
                if (formal_count == 0u || query == nullptr ||
                    !function->arguments().front()->is_reference() ||
                    function->arguments().front()->type() != query->type() || !query->is_lvalue()) {
                    reject_function();
                    break;
                }
            } else {
                // The same function can be a legal callee and an ordinary
                // operand of another call. The latter is not a call edge.
                reject_function();
                break;
            }
            if (formal_count != user->operand_count() - operand_offset + formal_offset) {
                reject_function();
                break;
            }
            auto formal_index = size_t{0u};
            for (auto *formal : function->arguments()) {
                auto index = formal_index++;
                if (index < formal_offset) { continue; }
                auto state = states.find(formal);
                if (state == states.end()) { continue; }
                auto *value = user->operand(operand_offset + index - formal_offset);
                if (value == nullptr || !value->isa<Argument>()) {
                    state->second.conflicting = true;
                    continue;
                }
                auto *actual = static_cast<const Argument *>(value);
                if (!actual->is_resource() || actual->type() != formal->type() ||
                    actual->parent_function() != caller->second) {
                    state->second.conflicting = true;
                    continue;
                }
                state->second.actuals.emplace_back(actual);
            }
        }
    }

    // Match the resource-origin lattice used by the SPIR-V argument analysis:
    // unresolved -> unique(root) or conflicting. Publish only after *all*
    // incoming values have resolved, never provisionally from one caller.
    // Consequently self/mutual recursion remains unresolved even if another
    // caller supplies a kernel root. No recursive C++ traversal is needed.
    for (auto changed = true; changed;) {
        changed = false;
        for (auto &[formal, state] : states) {
            if (state.conflicting || state.origin != nullptr || state.actuals.empty()) { continue; }
            const Argument *candidate = nullptr;
            auto unresolved = false;
            for (auto *actual : state.actuals) {
                const Argument *origin = nullptr;
                if (auto root = origins.find(actual); root != origins.end()) {
                    origin = root->second;
                } else if (auto dependency = states.find(actual); dependency != states.end() && !dependency->second.conflicting) {
                    origin = dependency->second.origin;
                    if (origin == nullptr) {
                        unresolved = true;
                        continue;
                    }
                } else {
                    state.conflicting = true;
                    break;
                }
                if (origin->type() != formal->type() || (candidate != nullptr && candidate != origin)) {
                    state.conflicting = true;
                    break;
                }
                candidate = origin;
            }
            if (state.conflicting) {
                changed = true;
            } else if (!unresolved && candidate != nullptr) {
                state.origin = candidate;
                changed = true;
            }
        }
    }
    for (auto &&[formal, state] : states) {
        if (!state.conflicting && state.origin != nullptr) { origins.emplace(formal, state.origin); }
    }
    return origins;
}

}// namespace luisa::compute::xir
