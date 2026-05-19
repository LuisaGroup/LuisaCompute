#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_ray_query_loop_to_loop.h>

namespace luisa::compute::xir {

namespace detail {

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
    auto candidate_continue_block = def->create_basic_block();
    b.set_insertion_point(candidate_continue_block);
    b.br(update_block);
    dispatch_block->replace_all_uses_with(candidate_continue_block);
    dispatch_inst->remove_self();
    info.lowered_ray_query_loop_count += 1u;
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
