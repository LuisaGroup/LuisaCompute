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
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/lower_ray_query_loop_to_loop.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include <algorithm>

namespace luisa::compute::xir {

namespace detail {

static void clone_metadata_impl(const MetadataListMixin &source,
                                       MetadataListMixin &target) noexcept {
    for (auto *metadata : source.metadata_list()) {
        target.metadata_list().push_front(metadata->clone());
    }
}

[[nodiscard]] static bool contains_phi(BasicBlock *block) noexcept {
    if (block == nullptr) { return false; }
    for (auto *inst : block->instructions()) {
        if (inst->isa<PhiInst>()) { return true; }
    }
    return false;
}

static void replace_phi_predecessor(BasicBlock *block,
                                    BasicBlock *old_predecessor,
                                    BasicBlock *new_predecessor) noexcept {
    if (block == nullptr || old_predecessor == new_predecessor) { return; }
    for (auto *inst : block->instructions()) {
        if (!inst->isa<PhiInst>()) { continue; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (size_t i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            if (incoming.block == old_predecessor) {
                phi->set_incoming(i, incoming.value, new_predecessor);
            }
        }
    }
}

struct RetargetableHandlerRegion {
    luisa::unordered_set<BasicBlock *> blocks;
    size_t dispatch_exit_count{0u};
};

[[nodiscard]] static bool collect_retargetable_handler_region(
    BasicBlock *entry, BasicBlock *dispatch, BasicBlock *loop_merge,
    RetargetableHandlerRegion &region) noexcept {
    if (entry == nullptr || dispatch == nullptr || loop_merge == nullptr) { return false; }
    auto *owner = dispatch->parent_function();
    luisa::vector<BasicBlock *> worklist{entry};
    while (!worklist.empty()) {
        auto *block = worklist.back();
        worklist.pop_back();
        if (block == dispatch) { continue; }
        if (block == nullptr || block == loop_merge ||
            block->parent_function() != owner) {
            return false;
        }
        if (!region.blocks.emplace(block).second) { continue; }
        if (!block->is_terminated()) { return false; }
        auto *term = block->terminator();
        auto valid = true;
        if (term->isa<BranchInst>() &&
            static_cast<BranchInst *>(term)->target_block() == nullptr) {
            return false;
        }
        if (term->isa<ConditionalBranchInst>()) {
            auto *branch = static_cast<ConditionalBranchInst *>(term);
            if (branch->condition() == nullptr || branch->true_block() == nullptr ||
                branch->false_block() == nullptr) {
                return false;
            }
        }
        block->traverse_successors(false, [&](BasicBlock *successor) noexcept {
            if (successor == dispatch) {
                valid &= term->isa<BranchInst>() &&
                         static_cast<BranchInst *>(term)->target_block() == dispatch;
                if (valid) { ++region.dispatch_exit_count; }
            } else if (!region.blocks.contains(successor)) {
                worklist.emplace_back(successor);
            }
        });
        if (!valid) { return false; }
    }
    if (region.dispatch_exit_count == 0u) { return false; }
    for (auto *block : region.blocks) {
        if (auto *merge = block->terminator()->control_flow_merge(); merge != nullptr) {
            auto *merge_block = merge->merge_block();
            if (merge_block != nullptr && !region.blocks.contains(merge_block)) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] static bool lower_ray_query_loop_to_loop_handler_region_has_external_predecessor(
    BasicBlock *entry, BasicBlock *dispatch,
    const RetargetableHandlerRegion &region) noexcept {
    for (auto *block : region.blocks) {
        auto invalid = false;
        block->traverse_predecessors(false, [&](BasicBlock *predecessor) noexcept {
            invalid |= !region.blocks.contains(predecessor) &&
                       !(block == entry && predecessor == dispatch);
        });
        if (invalid) { return true; }
    }
    return false;
}

[[nodiscard]] static bool lower_ray_query_loop_to_loop_handler_regions_overlap(
    const RetargetableHandlerRegion &lhs,
    const RetargetableHandlerRegion &rhs) noexcept {
    auto *smaller = &lhs.blocks;
    auto *larger = &rhs.blocks;
    if (smaller->size() > larger->size()) { std::swap(smaller, larger); }
    for (auto *block : *smaller) {
        if (larger->contains(block)) { return true; }
    }
    return false;
}

[[nodiscard]] static bool is_ray_query_object_impl(
    const Value *value) noexcept {
    if (value == nullptr || !value->is_lvalue()) { return false; }
    auto *type = value->type();
    return type == Type::custom("LC_RayQueryAll") ||
           type == Type::custom("LC_RayQueryAny");
}

// Lowers RayQueryLoopInst into a LoopInst with structured candidate dispatch.
//
// Output structure (LoopInst with prepare/body/update/merge):
//   prepare: proceed(rq); cond_br(is_active, body, merge)
//   body:    IfInst(is_triangle, on_surface_block, else_block, candidate_continue)
//              else_block: IfInst(is_procedural, on_procedural_block, skip, candidate_continue)
//                skip: br(candidate_continue)
//            candidate_continue: br(update)
//   update:  br(prepare)
//   merge:   (original merge — post-loop code)
static bool lower_one_ray_query_loop(RayQueryLoopInst *rq_loop, XIRBuilder &b,
                                     LowerRayQueryLoopToLoopInfo &info,
                                     bool preflight_only) noexcept {
    auto reject = [&](luisa::string_view reason) noexcept {
        LUISA_WARNING_WITH_LOCATION("lower_ray_query_loop_to_loop: rejecting loop: {}", reason);
        ++info.error_count;
        return false;
    };
    if (rq_loop == nullptr) { return reject("null RayQueryLoopInst"); }
    auto parent_block = rq_loop->parent_block();
    auto dispatch_block = rq_loop->dispatch_block();
    auto merge_block = rq_loop->merge_block();
    if (parent_block == nullptr || dispatch_block == nullptr || merge_block == nullptr) {
        return reject("null parent, dispatch, or merge block");
    }
    if (!dispatch_block->is_terminated()) {
        return reject("dispatch block is unterminated");
    }
    auto dispatch_term = dispatch_block->terminator();
    if (!dispatch_term->isa<RayQueryDispatchInst>()) {
        return reject("dispatch block is not terminated with RayQueryDispatchInst");
    }
    auto dispatch_inst = static_cast<RayQueryDispatchInst *>(dispatch_term);
    auto query_object = dispatch_inst->query_object();
    auto on_surface_block = dispatch_inst->on_surface_candidate_block();
    auto on_procedural_block = dispatch_inst->on_procedural_candidate_block();
    if (!is_ray_query_object_impl(query_object) ||
        on_surface_block == nullptr || on_procedural_block == nullptr) {
        return reject(
            "RayQueryDispatchInst requires an lvalue LC_RayQueryAll/"
            "LC_RayQueryAny object and non-null handler operands");
    }
    auto function = parent_block->parent_function();
    if (function == nullptr) {
        return reject("parent block has no parent function");
    }
    auto def = function->definition();
    if (def == nullptr) {
        return reject("parent function has no definition");
    }
    if (dispatch_block->parent_function() != function ||
        merge_block->parent_function() != function) {
        return reject("RayQueryLoop references a block outside its function");
    }
    if (dispatch_inst->exit_block() != merge_block) {
        return reject("dispatch exit does not match the RayQueryLoop merge block");
    }
    if (dispatch_block == merge_block || dispatch_block == parent_block ||
        merge_block == parent_block || on_surface_block == dispatch_block ||
        on_procedural_block == dispatch_block || on_surface_block == merge_block ||
        on_procedural_block == merge_block) {
        return reject("ray-query loop reuses a parent, dispatch, merge, or handler block");
    }
    // Dispatch values and handler-entry PHIs would need edge-sensitive SSA
    // migration. Reject them before creating a single replacement block.
    if (contains_phi(dispatch_block) || contains_phi(on_surface_block) ||
        contains_phi(on_procedural_block)) {
        return reject("dispatch or handler-entry PHI requires SSA migration");
    }
    RetargetableHandlerRegion surface_region;
    RetargetableHandlerRegion procedural_region;
    if (!collect_retargetable_handler_region(on_surface_block, dispatch_block,
                                             merge_block, surface_region) ||
        !collect_retargetable_handler_region(on_procedural_block, dispatch_block,
                                             merge_block, procedural_region)) {
        return reject("handler has an unterminated or non-Branch edge to dispatch");
    }
    if (lower_ray_query_loop_to_loop_handler_regions_overlap(surface_region, procedural_region)) {
        return reject("surface and procedural handler regions overlap");
    }
    auto dispatch_has_external_predecessor = false;
    dispatch_block->traverse_predecessors(
        false, [&](BasicBlock *predecessor) noexcept {
            dispatch_has_external_predecessor |=
                predecessor != parent_block &&
                !surface_region.blocks.contains(predecessor) &&
                !procedural_region.blocks.contains(predecessor);
        });
    if (dispatch_has_external_predecessor) {
        return reject("dispatch has a predecessor outside the ray-query loop");
    }
    if (lower_ray_query_loop_to_loop_handler_region_has_external_predecessor(on_surface_block, dispatch_block, surface_region) ||
        lower_ray_query_loop_to_loop_handler_region_has_external_predecessor(on_procedural_block, dispatch_block, procedural_region)) {
        return reject("handler has a predecessor outside its region");
    }
    auto merge_has_external_predecessor = false;
    merge_block->traverse_predecessors(
        false, [&](BasicBlock *predecessor) noexcept {
            merge_has_external_predecessor |=
                predecessor != dispatch_block &&
                predecessor != parent_block;
        });
    if (merge_has_external_predecessor) {
        return reject("merge has a predecessor outside the ray-query loop");
    }
    for (auto *inst : dispatch_block->instructions()) {
        if (inst != dispatch_term) {
            return reject("dispatch block contains non-terminator instructions");
        }
    }
    if (preflight_only) { return true; }
    auto bool_type = Type::of<bool>();

    auto removed_loop = rq_loop->remove_self();
    b.set_insertion_point(parent_block);
    auto loop_inst = b.loop();
    clone_metadata_impl(*removed_loop, *loop_inst);
    loop_inst->set_merge_block(merge_block);
    auto prepare_block = loop_inst->create_prepare_block();
    auto body_block = loop_inst->create_body_block();
    auto update_block = loop_inst->create_update_block();

    b.set_insertion_point(prepare_block);
    b.call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED, {query_object});
    auto is_terminated = b.call(bool_type, RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, {query_object});
    auto is_active = b.call(bool_type, ArithmeticOp::UNARY_BIT_NOT, {is_terminated});
    b.cond_br(is_active, body_block, merge_block);
    replace_phi_predecessor(merge_block, dispatch_block, prepare_block);

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
    auto removed_dispatch = dispatch_inst->remove_self();
    // The outer triangle test is the unique replacement for candidate
    // dispatch. The nested procedural test only refines its false arm.
    clone_metadata_impl(*removed_dispatch, *tri_if);
    b.set_insertion_point(dispatch_block);
    b.unreachable_();
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

[[nodiscard]] static luisa::vector<RayQueryLoopInst *>
collect_ray_query_loops_impl(Function *function) noexcept {
    luisa::vector<RayQueryLoopInst *> rq_loops;
    if (function != nullptr) {
        if (auto *def = function->definition()) {
            for (auto *block : def->basic_blocks()) {
                if (block != nullptr && block->is_terminated()) {
                    auto *terminator = block->terminator();
                    if (terminator->isa<RayQueryLoopInst>()) {
                        rq_loops.emplace_back(
                            static_cast<RayQueryLoopInst *>(terminator));
                    }
                }
            }
        }
    }
    return rq_loops;
}

[[nodiscard]] static bool lower_ray_query_loop_to_loop_preflight_ray_query_loops(
    luisa::span<RayQueryLoopInst *const> rq_loops,
    XIRBuilder &b, LowerRayQueryLoopToLoopInfo &info) noexcept {
    auto accepted = true;
    for (auto *rq : rq_loops) {
        accepted &= lower_one_ray_query_loop(rq, b, info, true);
    }
    return accepted;
}

static void lower_ray_query_loop_to_loop_lower_preflighted_ray_query_loops(
    luisa::span<RayQueryLoopInst *const> rq_loops,
    XIRBuilder &b, LowerRayQueryLoopToLoopInfo &info) noexcept {
    for (auto *rq : rq_loops) {
        static_cast<void>(
            lower_one_ray_query_loop(rq, b, info, false));
    }
}

static void lower_in_function(
    Function *function, LowerRayQueryLoopToLoopInfo &info) noexcept {
    auto rq_loops = collect_ray_query_loops_impl(function);
    XIRBuilder b;
    if (!lower_ray_query_loop_to_loop_preflight_ray_query_loops(
            luisa::span{rq_loops}, b, info)) {
        return;
    }
    lower_ray_query_loop_to_loop_lower_preflighted_ray_query_loops(
        luisa::span{rq_loops}, b, info);
}

}// namespace detail

LowerRayQueryLoopToLoopInfo lower_ray_query_loop_to_loop_pass_run_on_function(Function *function) noexcept {
    LowerRayQueryLoopToLoopInfo info;
    if (function == nullptr) { return info; }
    detail::lower_in_function(function, info);
    return info;
}

LowerRayQueryLoopToLoopInfo lower_ray_query_loop_to_loop_pass_run_on_module(Module *module, PassReport *report) noexcept {
    LowerRayQueryLoopToLoopInfo info;
    if (module != nullptr) {
        struct FunctionWork {
            Function *function;
            luisa::vector<RayQueryLoopInst *> loops;
        };
        luisa::vector<FunctionWork> work;
        for (auto *function : module->function_list()) {
            work.emplace_back(FunctionWork{
                .function = function,
                .loops = detail::collect_ray_query_loops_impl(function)});
        }
        XIRBuilder b;
        auto accepted = true;
        for (auto &item : work) {
            accepted &= detail::lower_ray_query_loop_to_loop_preflight_ray_query_loops(
                luisa::span{item.loops}, b, info);
        }
        if (accepted) {
            for (auto &item : work) {
                detail::lower_ray_query_loop_to_loop_lower_preflighted_ray_query_loops(
                    luisa::span{item.loops}, b, info);
            }
        }
    }
    if (report != nullptr) {
        report->set("lowered_ray_query_loop", info.lowered_ray_query_loop_count);
        report->set("error", info.error_count);
    }
    return info;
}

}// namespace luisa::compute::xir
