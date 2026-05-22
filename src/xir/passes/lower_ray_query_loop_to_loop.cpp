#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_ray_query_loop_to_loop.h>

namespace luisa::compute::xir {

namespace detail {

// Lowers RayQueryLoopInst into a LoopInst with structured candidate dispatch.
//
// Output structure (LoopInst with prepare/body/update/merge):
//   prepare: proceed(rq); cond_br(is_terminated, merge, body)
//   body:    IfInst(is_triangle, on_surface_block, else_block, candidate_continue)
//              else_block: IfInst(is_procedural, on_procedural_block, skip, candidate_continue)
//                skip: br(candidate_continue)
//            candidate_continue: br(update)
//   update:  br(prepare)
//   merge:   (original merge — post-loop code)
static bool lower_one_ray_query_loop(RayQueryLoopInst *rq_loop, XIRBuilder &b,
                                     LowerRayQueryLoopToLoopInfo &info) noexcept {
    if (rq_loop == nullptr) { return false; }
    auto parent_block = rq_loop->parent_block();
    auto dispatch_block = rq_loop->dispatch_block();
    auto merge_block = rq_loop->merge_block();
    if (parent_block == nullptr || dispatch_block == nullptr || merge_block == nullptr) {
        LUISA_WARNING_WITH_LOCATION("lower_ray_query_loop_to_loop: skipping RayQueryLoop with null parent/dispatch/merge block.");
        return false;
    }
    auto dispatch_term = dispatch_block->terminator();
    if (dispatch_term == nullptr || !dispatch_term->isa<RayQueryDispatchInst>()) {
        LUISA_WARNING_WITH_LOCATION("lower_ray_query_loop_to_loop: dispatch block not terminated with RayQueryDispatch.");
        return false;
    }
    auto dispatch_inst = static_cast<RayQueryDispatchInst *>(dispatch_term);
    auto query_object = dispatch_inst->query_object();
    auto on_surface_block = dispatch_inst->on_surface_candidate_block();
    auto on_procedural_block = dispatch_inst->on_procedural_candidate_block();
    if (query_object == nullptr || on_surface_block == nullptr || on_procedural_block == nullptr) {
        LUISA_WARNING_WITH_LOCATION("lower_ray_query_loop_to_loop: RayQueryDispatch with null operand.");
        return false;
    }
    auto function = parent_block->parent_function();
    if (function == nullptr) {
        LUISA_WARNING_WITH_LOCATION("lower_ray_query_loop_to_loop: parent block without parent function.");
        return false;
    }
    auto def = function->definition();
    if (def == nullptr) {
        LUISA_WARNING_WITH_LOCATION("lower_ray_query_loop_to_loop: function has no definition.");
        return false;
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
    b.call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED, {query_object});
    auto is_terminated = b.call(bool_type, RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, {query_object});
    b.cond_br(is_terminated, merge_block, body_block);

    b.set_insertion_point(update_block);
    b.br(prepare_block);

    b.set_insertion_point(body_block);
    auto is_triangle = b.call(bool_type, RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE, {query_object});
    auto is_procedural = b.call(bool_type, RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE, {query_object});

    auto tri_merge = def->create_basic_block();
    auto tri_else_block = def->create_basic_block();
    auto tri_if = b.if_(is_triangle);
    tri_if->set_true_target(on_surface_block);
    tri_if->set_false_target(tri_else_block);
    tri_if->set_merge_block(tri_merge);

    b.set_insertion_point(tri_merge);
    b.br(update_block);

    auto proc_merge = def->create_basic_block();
    auto proc_skip_block = def->create_basic_block();
    b.set_insertion_point(tri_else_block);
    auto proc_if = b.if_(is_procedural);
    proc_if->set_true_target(on_procedural_block);
    proc_if->set_false_target(proc_skip_block);
    proc_if->set_merge_block(proc_merge);

    b.set_insertion_point(proc_merge);
    b.br(tri_merge);

    b.set_insertion_point(proc_skip_block);
    b.br(proc_merge);

    // Retarget handler terminators: br(dispatch_block) → br(tri_merge)
    // Only retarget BranchInst targets, not merge_block fields of nested constructs.
    dispatch_inst->remove_self();
    auto retarget_handler_branches = [&](BasicBlock *handler_entry, BasicBlock *target) noexcept {
        luisa::vector<BasicBlock *> worklist;
        luisa::unordered_set<BasicBlock *> visited;
        worklist.emplace_back(handler_entry);
        while (!worklist.empty()) {
            auto *bb = worklist.back();
            worklist.pop_back();
            if (!visited.emplace(bb).second) { continue; }
            if (!bb->is_terminated()) { continue; }
            auto *term = bb->terminator();
            if (term->derived_instruction_tag() == DerivedInstructionTag::BRANCH) {
                auto *br = static_cast<BranchInst *>(term);
                if (br->target_block() == dispatch_block) {
                    br->set_target_block(target);
                    continue;
                }
            }
            bb->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                if (!visited.contains(succ) && succ != tri_merge && succ != proc_merge) {
                    worklist.emplace_back(succ);
                }
            });
        }
    };
    retarget_handler_branches(on_surface_block, tri_merge);
    retarget_handler_branches(on_procedural_block, proc_merge);

    info.lowered_ray_query_loop_count += 1;
    return true;
}

static void lower_in_function(Function *function, LowerRayQueryLoopToLoopInfo &info) noexcept {
    if (function == nullptr) { return; }
    auto def = function->definition();
    if (def == nullptr) { return; }
    for (;;) {
        luisa::vector<RayQueryLoopInst *> rq_loops;
        def->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            if (block == nullptr || !block->is_terminated()) { return; }
            auto term = block->terminator();
            if (term != nullptr && term->isa<RayQueryLoopInst>()) {
                rq_loops.emplace_back(static_cast<RayQueryLoopInst *>(term));
            }
        });
        if (rq_loops.empty()) { break; }
        XIRBuilder b;
        bool any_lowered = false;
        for (auto rq : rq_loops) {
            any_lowered |= lower_one_ray_query_loop(rq, b, info);
        }
        if (!any_lowered) { break; }
    }
}

}// namespace detail

LowerRayQueryLoopToLoopInfo lower_ray_query_loop_to_loop_pass_run_on_function(Function *function) noexcept {
    LowerRayQueryLoopToLoopInfo info;
    if (function == nullptr) { return info; }
    detail::lower_in_function(function, info);
    return info;
}

LowerRayQueryLoopToLoopInfo lower_ray_query_loop_to_loop_pass_run_on_module(Module *module) noexcept {
    LowerRayQueryLoopToLoopInfo info;
    if (module == nullptr) { return info; }
    for (auto f : module->function_list()) {
        detail::lower_in_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
