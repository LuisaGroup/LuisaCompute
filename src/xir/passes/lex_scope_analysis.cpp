#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/autodiff.h>
#include <luisa/xir/instructions/outline.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/passes/lex_scope_analysis.h>

namespace luisa::compute::xir {

namespace detail {

class LexScopeStack {

public:
    using Set = luisa::unordered_set<const Instruction *>;
    using Stack = luisa::vector<Set>;

private:
    Stack _stack;

public:
    void push() noexcept { _stack.emplace_back(); }
    void pop() noexcept {
        LUISA_DEBUG_ASSERT(!_stack.empty(), "Lexical scope stack is empty.");
        _stack.pop_back();
    }
    template<typename F>
    void with_scope(F &&f) noexcept {
        push();
        f();
        pop();
    }
    void define(const Instruction *inst) noexcept {
        LUISA_DEBUG_ASSERT(!_stack.empty(), "Lexical scope stack is empty.");
        _stack.back().emplace(inst);
    }
    [[nodiscard]] bool is_in_scope(const Value *value) const noexcept {
        // do not consider non-instruction values as they can never be out of scope
        if (value == nullptr || !value->isa<Instruction>()) { return true; }
        // check if the value is in the current scope or any parent scope
        for (auto it = _stack.rbegin(); it != _stack.rend(); ++it) {
            if (it->contains(static_cast<const Instruction *>(value))) { return true; }
        }
        // the value is not in any scope
        return false;
    }
};

static void walk_lexical_scopes_recursively(const BasicBlock *block,
                                            const LexScopeAnalysisConfig &config,
                                            LexScopeStack &stack,
                                            LexScopeInfo &info) noexcept {
    if (block == nullptr) { return; }
    for (auto inst : block->instructions()) {
        for (auto op_use : inst->operand_uses()) {
            if (auto value = op_use->value(); !stack.is_in_scope(op_use->value())) {
                if (auto op_inst = static_cast<const Instruction *>(value);
                    info.lexical_scope_breakers.emplace(op_inst).second) {
                    info.lexical_scope_breaks_ordered.emplace_back(op_inst);
                }
            }
        }
        stack.define(inst);
        // recursively walk the successors if the instruction is a control flow instruction
        switch (inst->derived_instruction_tag()) {
            case DerivedInstructionTag::IF: {
                auto if_inst = static_cast<const IfInst *>(inst);
                // scope of the true branch
                stack.with_scope([&] {
                    walk_lexical_scopes_recursively(if_inst->true_block(), config, stack, info);
                });
                // scope of the false branch
                stack.with_scope([&] {
                    walk_lexical_scopes_recursively(if_inst->false_block(), config, stack, info);
                });
                // merge block should be of the parent block
                walk_lexical_scopes_recursively(if_inst->merge_block(), config, stack, info);
                break;
            }
            case DerivedInstructionTag::SWITCH: {
                auto switch_inst = static_cast<const SwitchInst *>(inst);
                // scopes of the case blocks
                for (size_t i = 0; i < switch_inst->case_count(); i++) {
                    stack.with_scope([&] {
                        walk_lexical_scopes_recursively(switch_inst->case_block(i), config, stack, info);
                    });
                }
                // scope of the default block
                stack.with_scope([&] {
                    walk_lexical_scopes_recursively(switch_inst->default_block(), config, stack, info);
                });
                // merge block should be of the parent block
                walk_lexical_scopes_recursively(switch_inst->merge_block(), config, stack, info);
                break;
            }
            case DerivedInstructionTag::LOOP: {
                auto loop_inst = static_cast<const LoopInst *>(inst);
                // scope of the loop
                stack.with_scope([&] {
                    // prepare block should be of the loop scope
                    walk_lexical_scopes_recursively(loop_inst->prepare_block(), config, stack, info);
                    if (config.loop_body_is_nested) {// scope of the body block is nested into the loop scope
                        stack.with_scope([&] {
                            walk_lexical_scopes_recursively(loop_inst->body_block(), config, stack, info);
                        });
                    } else {// otherwise loop body is of the same scope as the loop itself
                        walk_lexical_scopes_recursively(loop_inst->body_block(), config, stack, info);
                    }
                    // update block should be of the loop scope
                    walk_lexical_scopes_recursively(loop_inst->update_block(), config, stack, info);
                });
                // merge block should be of the parent block
                walk_lexical_scopes_recursively(loop_inst->merge_block(), config, stack, info);
                break;
            }
            case DerivedInstructionTag::SIMPLE_LOOP: {
                auto simple_loop_inst = static_cast<const SimpleLoopInst *>(inst);
                // scope of the body block
                stack.with_scope([&] {
                    walk_lexical_scopes_recursively(simple_loop_inst->body_block(), config, stack, info);
                });
                // merge block should be of the parent block
                walk_lexical_scopes_recursively(simple_loop_inst->merge_block(), config, stack, info);
                break;
            }
            case DerivedInstructionTag::RAY_QUERY_LOOP: {
                auto ray_query_loop = static_cast<const RayQueryLoopInst *>(inst);
                stack.with_scope([&] {
                    walk_lexical_scopes_recursively(
                        ray_query_loop->dispatch_block(), config, stack, info);
                });
                walk_lexical_scopes_recursively(
                    ray_query_loop->merge_block(), config, stack, info);
                break;
            }
            case DerivedInstructionTag::RAY_QUERY_DISPATCH: {
                auto dispatch = static_cast<const RayQueryDispatchInst *>(inst);
                stack.with_scope([&] {
                    walk_lexical_scopes_recursively(
                        dispatch->on_surface_candidate_block(), config, stack, info);
                });
                stack.with_scope([&] {
                    walk_lexical_scopes_recursively(
                        dispatch->on_procedural_candidate_block(), config, stack, info);
                });
                break;
            }
            case DerivedInstructionTag::AUTODIFF_SCOPE: {
                auto scope = static_cast<const AutodiffScopeInst *>(inst);
                stack.with_scope([&] {
                    walk_lexical_scopes_recursively(
                        scope->entry_block(), config, stack, info);
                });
                walk_lexical_scopes_recursively(
                    scope->merge_block(), config, stack, info);
                break;
            }
            case DerivedInstructionTag::OUTLINE: {
                auto outline = static_cast<const OutlineInst *>(inst);
                stack.with_scope([&] {
                    walk_lexical_scopes_recursively(
                        outline->target_block(), config, stack, info);
                });
                walk_lexical_scopes_recursively(
                    outline->merge_block(), config, stack, info);
                break;
            }
            default: {
                LUISA_ASSERT(inst->control_flow_merge() == nullptr,
                             "Unexpected control flow {:?} in lexical scope analysis.",
                             inst->intrinsic_identifier());
                break;
            }
        }
    }
}

static void analyze_lexical_scopes_in_function(const Function *function,
                                               const LexScopeAnalysisConfig &config,
                                               LexScopeInfo &info) noexcept {
    if (function == nullptr) { return; }
    if (auto def = function->definition();
        def != nullptr && def->body_block() != nullptr) {
        LexScopeStack stack;
        stack.with_scope([&] {
            walk_lexical_scopes_recursively(def->body_block(), config, stack, info);
        });
    }
    LUISA_DEBUG_ASSERT(info.lexical_scope_breakers.size() == info.lexical_scope_breaks_ordered.size(),
                       "Lexical scope analysis failed: size mismatch.");
}

}// namespace detail

LexScopeInfo lex_scope_analysis_pass_run_on_function(const Function *function,
                                                     const LexScopeAnalysisConfig &config) noexcept {
    LexScopeInfo info;
    detail::analyze_lexical_scopes_in_function(function, config, info);
    return info;
}

}// namespace luisa::compute::xir
