#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/destructure_cfg.h>

namespace luisa::compute::xir {

namespace detail {

static void destructure_ray_query_loop(RayQueryLoopInst *rq_loop, XIRBuilder &b,
                                       DestructureCFGInfo &info) noexcept {
    if (rq_loop == nullptr) { return; }
    auto parent_block = rq_loop->parent_block();
    auto dispatch_block = rq_loop->dispatch_block();
    auto merge_block = rq_loop->merge_block();
    if (parent_block == nullptr || dispatch_block == nullptr || merge_block == nullptr) {
        LUISA_WARNING_WITH_LOCATION("destructure_ray_query_loop: skipping RayQueryLoop with null parent/dispatch/merge block.");
        return;
    }
    auto dispatch_term = dispatch_block->terminator();
    if (dispatch_term == nullptr || !dispatch_term->isa<RayQueryDispatchInst>()) {
        LUISA_WARNING_WITH_LOCATION("destructure_ray_query_loop: dispatch block not terminated with RayQueryDispatch.");
        return;
    }
    auto dispatch_inst = static_cast<RayQueryDispatchInst *>(dispatch_term);
    auto query_object = dispatch_inst->query_object();
    auto on_surface_block = dispatch_inst->on_surface_candidate_block();
    auto on_procedural_block = dispatch_inst->on_procedural_candidate_block();
    if (query_object == nullptr || on_surface_block == nullptr || on_procedural_block == nullptr) {
        LUISA_WARNING_WITH_LOCATION("destructure_ray_query_loop: RayQueryDispatch with null operand.");
        return;
    }
    auto function = parent_block->parent_function();
    if (function == nullptr) {
        LUISA_WARNING_WITH_LOCATION("destructure_ray_query_loop: parent block without parent function.");
        return;
    }
    auto def = function->definition();
    if (def == nullptr) {
        LUISA_WARNING_WITH_LOCATION("destructure_ray_query_loop: function has no definition.");
        return;
    }
    auto bool_type = Type::of<bool>();
    rq_loop->remove_self();
    b.set_insertion_point(parent_block);
    auto loop_inst = b.loop();
    loop_inst->set_merge_block(merge_block);
    auto prepare_block = loop_inst->create_prepare_block();
    auto body_block = loop_inst->create_body_block();
    auto update_block = loop_inst->create_update_block();
    b.set_insertion_point(prepare_block);
    b.br(body_block);
    b.set_insertion_point(update_block);
    b.br(prepare_block);
    b.set_insertion_point(body_block);
    b.call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED, {query_object});
    auto is_terminated = b.call(bool_type, RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, {query_object});
    auto is_triangle = b.call(bool_type, RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE, {query_object});
    auto is_procedural = b.call(bool_type, RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE, {query_object});
    auto after_term_block = def->create_basic_block();
    b.cond_br(is_terminated, merge_block, after_term_block);
    b.set_insertion_point(after_term_block);
    auto after_triangle_block = def->create_basic_block();
    b.cond_br(is_triangle, on_surface_block, after_triangle_block);
    b.set_insertion_point(after_triangle_block);
    b.cond_br(is_procedural, on_procedural_block, update_block);
    auto rewrite_candidate_back_edge = [&](BasicBlock *block) noexcept {
        if (block == nullptr || !block->is_terminated()) { return; }
        auto term = block->terminator();
        if (term == nullptr) { return; }
        if (term->isa<BranchInst>()) {
            auto br = static_cast<BranchInst *>(term);
            if (br->target_block() == dispatch_block) {
                br->set_target_block(update_block);
            }
        } else if (term->isa<ConditionalBranchInst>()) {
            auto cbr = static_cast<ConditionalBranchInst *>(term);
            if (cbr->true_block() == dispatch_block) { cbr->set_true_target(update_block); }
            if (cbr->false_block() == dispatch_block) { cbr->set_false_target(update_block); }
        }
    };
    rewrite_candidate_back_edge(on_surface_block);
    rewrite_candidate_back_edge(on_procedural_block);
    dispatch_inst->remove_self();
    info.destructured_ray_query_loop_count += 1u;
}

static void destructure_in_function(Function *function, DestructureCFGInfo &info) noexcept {
    if (function == nullptr) { return; }
    auto def = function->definition();
    if (def == nullptr) { return; }
    for (;;) {
        luisa::vector<IfInst *> if_insts;
        luisa::vector<LoopInst *> loop_insts;
        luisa::vector<SimpleLoopInst *> simple_loop_insts;
        luisa::vector<BreakInst *> break_insts;
        luisa::vector<ContinueInst *> continue_insts;
        luisa::vector<RayQueryLoopInst *> rq_loop_insts;
        def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            if (block == nullptr || !block->is_terminated()) { return; }
            auto term = block->terminator();
            if (term == nullptr) { return; }
            switch (term->derived_instruction_tag()) {
                case DerivedInstructionTag::IF:
                    if_insts.emplace_back(static_cast<IfInst *>(term));
                    break;
                case DerivedInstructionTag::LOOP:
                    loop_insts.emplace_back(static_cast<LoopInst *>(term));
                    break;
                case DerivedInstructionTag::SIMPLE_LOOP:
                    simple_loop_insts.emplace_back(static_cast<SimpleLoopInst *>(term));
                    break;
                case DerivedInstructionTag::BREAK:
                    break_insts.emplace_back(static_cast<BreakInst *>(term));
                    break;
                case DerivedInstructionTag::CONTINUE:
                    continue_insts.emplace_back(static_cast<ContinueInst *>(term));
                    break;
                case DerivedInstructionTag::RAY_QUERY_LOOP:
                    rq_loop_insts.emplace_back(static_cast<RayQueryLoopInst *>(term));
                    break;
                default: break;
            }
        });
        if (if_insts.empty() && loop_insts.empty() && simple_loop_insts.empty() &&
            break_insts.empty() && continue_insts.empty() && rq_loop_insts.empty()) {
            break;
        }
        XIRBuilder b;
        for (auto brk : break_insts) {
            if (brk == nullptr) { continue; }
            auto block = brk->parent_block();
            auto target = brk->target_block();
            if (block == nullptr || target == nullptr) {
                LUISA_WARNING_WITH_LOCATION("destructure_cfg: skipping break with null parent/target.");
                continue;
            }
            brk->remove_self();
            b.set_insertion_point(block);
            b.br(target);
        }
        for (auto cont : continue_insts) {
            if (cont == nullptr) { continue; }
            auto block = cont->parent_block();
            auto target = cont->target_block();
            if (block == nullptr || target == nullptr) {
                LUISA_WARNING_WITH_LOCATION("destructure_cfg: skipping continue with null parent/target.");
                continue;
            }
            cont->remove_self();
            b.set_insertion_point(block);
            b.br(target);
        }
        for (auto if_inst : if_insts) {
            if (if_inst == nullptr) { continue; }
            auto block = if_inst->parent_block();
            auto cond = if_inst->condition();
            auto true_block = if_inst->true_block();
            auto false_block = if_inst->false_block();
            if (block == nullptr || cond == nullptr || true_block == nullptr || false_block == nullptr) {
                LUISA_WARNING_WITH_LOCATION("destructure_cfg: skipping IfInst with null operand.");
                continue;
            }
            if_inst->remove_self();
            b.set_insertion_point(block);
            b.cond_br(cond, true_block, false_block);
        }
        for (auto loop_inst : loop_insts) {
            if (loop_inst == nullptr) { continue; }
            auto block = loop_inst->parent_block();
            auto prepare = loop_inst->prepare_block();
            if (block == nullptr || prepare == nullptr) {
                LUISA_WARNING_WITH_LOCATION("destructure_cfg: skipping LoopInst with null parent/prepare block.");
                continue;
            }
            loop_inst->remove_self();
            b.set_insertion_point(block);
            b.br(prepare);
        }
        for (auto sl : simple_loop_insts) {
            if (sl == nullptr) { continue; }
            auto block = sl->parent_block();
            auto body = sl->body_block();
            if (block == nullptr || body == nullptr) {
                LUISA_WARNING_WITH_LOCATION("destructure_cfg: skipping SimpleLoopInst with null parent/body block.");
                continue;
            }
            sl->remove_self();
            b.set_insertion_point(block);
            b.br(body);
        }
        for (auto rq : rq_loop_insts) {
            destructure_ray_query_loop(rq, b, info);
        }
        info.destructured_if_count += if_insts.size();
        info.destructured_loop_count += loop_insts.size();
        info.destructured_simple_loop_count += simple_loop_insts.size();
        info.destructured_break_count += break_insts.size();
        info.destructured_continue_count += continue_insts.size();
    }
}

}// namespace detail

DestructureCFGInfo destructure_cfg_pass_run_on_function(Function *function) noexcept {
    DestructureCFGInfo info;
    if (function == nullptr) { return info; }
    detail::destructure_in_function(function, info);
    return info;
}

DestructureCFGInfo destructure_cfg_pass_run_on_module(Module *module) noexcept {
    DestructureCFGInfo info;
    if (module == nullptr) { return info; }
    for (auto f : module->function_list()) {
        detail::destructure_in_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
