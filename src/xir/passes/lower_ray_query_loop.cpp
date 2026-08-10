#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include <algorithm>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

struct RayQueryHandlerRegion {
    luisa::unordered_set<BasicBlock *> blocks;
    size_t dispatch_exit_count{0u};
};

static void clone_metadata(const MetadataListMixin &source,
                           MetadataListMixin &target) noexcept {
    for (auto *metadata : source.metadata_list()) {
        target.metadata_list().push_front(metadata->clone());
    }
}

[[nodiscard]] static bool is_ray_query_object(
    const Value *value) noexcept {
    if (value == nullptr || !value->is_lvalue()) { return false; }
    auto *type = value->type();
    return type == Type::custom("LC_RayQueryAll") ||
           type == Type::custom("LC_RayQueryAny");
}

[[nodiscard]] static bool collect_outlineable_handler_region(
    BasicBlock *entry, BasicBlock *dispatch, BasicBlock *loop_merge,
    RayQueryHandlerRegion &region, luisa::string_view &reason) noexcept {
    if (entry == nullptr || dispatch == nullptr || loop_merge == nullptr) {
        reason = "null handler entry, dispatch, or loop merge block";
        return false;
    }
    auto *owner = dispatch->parent_function();
    luisa::vector<BasicBlock *> worklist{entry};
    while (!worklist.empty()) {
        auto *block = worklist.back();
        worklist.pop_back();
        if (block == dispatch) { continue; }
        if (block == loop_merge) {
            reason = "candidate handler reaches the ray-query loop merge directly";
            return false;
        }
        if (block == nullptr || block->parent_function() != owner) {
            reason = "candidate handler references a block outside its function";
            return false;
        }
        if (!region.blocks.emplace(block).second) { continue; }
        if (!block->is_terminated()) {
            reason = "candidate handler contains an unterminated block";
            return false;
        }
        auto *term = block->terminator();
        if (term->isa<ReturnInst>()) {
            // Returning from the parent function is not equivalent to returning
            // from an outlined candidate callback.
            reason = "candidate handler returns from the parent function";
            return false;
        }
        auto valid = true;
        block->traverse_successors(false, [&](BasicBlock *successor) noexcept {
            if (successor == dispatch) {
                if (term->isa<BranchInst>() &&
                    static_cast<BranchInst *>(term)->target_block() == dispatch) {
                    ++region.dispatch_exit_count;
                } else {
                    valid = false;
                }
            } else if (!region.blocks.contains(successor)) {
                worklist.emplace_back(successor);
            }
        });
        if (!valid) {
            reason = "candidate handler has a non-Branch edge to dispatch";
            return false;
        }
    }
    if (region.dispatch_exit_count != 1u) {
        reason = "candidate handler does not have exactly one exit to dispatch";
        return false;
    }
    // Raw structured merge markers are not CFG operands and therefore are not
    // discovered by successor traversal. The outliner can only resolve merge
    // blocks that are cloned as part of this handler region.
    for (auto *block : region.blocks) {
        if (auto *merge = block->terminator()->control_flow_merge(); merge != nullptr) {
            auto *merge_block = merge->merge_block();
            if (merge_block != nullptr && !region.blocks.contains(merge_block)) {
                reason = "candidate handler has a structured merge outside its region";
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] static bool lower_ray_query_loop_handler_regions_overlap(
    const RayQueryHandlerRegion &lhs,
    const RayQueryHandlerRegion &rhs) noexcept {
    auto *smaller = &lhs.blocks;
    auto *larger = &rhs.blocks;
    if (smaller->size() > larger->size()) { std::swap(smaller, larger); }
    for (auto *block : *smaller) {
        if (larger->contains(block)) { return true; }
    }
    return false;
}

[[nodiscard]] static bool lower_ray_query_loop_handler_region_has_external_predecessor(
    BasicBlock *entry, BasicBlock *dispatch,
    const RayQueryHandlerRegion &handler) noexcept {
    for (auto *block : handler.blocks) {
        auto invalid = false;
        block->traverse_predecessors(false, [&](BasicBlock *predecessor) noexcept {
            invalid |= !handler.blocks.contains(predecessor) &&
                       !(block == entry && predecessor == dispatch);
        });
        if (invalid) { return true; }
    }
    return false;
}

[[nodiscard]] static bool value_is_outline_resolvable(
    Value *value, Value *query_object, BasicBlock *dispatch,
    const RayQueryHandlerRegion &handler,
    const luisa::unordered_set<BasicBlock *> &loop_blocks) noexcept {
    if (value == nullptr || value == query_object) { return true; }
    switch (value->derived_value_tag()) {
        case DerivedValueTag::UNDEFINED: [[fallthrough]];
        case DerivedValueTag::FUNCTION: [[fallthrough]];
        case DerivedValueTag::CONSTANT: [[fallthrough]];
        case DerivedValueTag::SPECIAL_REGISTER: return true;
        case DerivedValueTag::ARGUMENT:
            // Function arguments are materialized as callback captures.
            return true;
        case DerivedValueTag::BASIC_BLOCK:
            return handler.blocks.contains(static_cast<BasicBlock *>(value));
        case DerivedValueTag::INSTRUCTION: {
            auto *inst = static_cast<Instruction *>(value);
            auto *parent = inst->parent_block();
            if (handler.blocks.contains(parent)) { return true; }
            // Dispatch PHIs are lowered to callback-local loads before cloning.
            if (parent == dispatch && inst->isa<PhiInst>()) { return true; }
            // Instructions defined outside the ray-query loop are captured.
            return !loop_blocks.contains(parent);
        }
        default: return false;
    }
}

[[nodiscard]] static bool validate_outline_resolver_inputs(
    const RayQueryHandlerRegion &handler, Value *query_object,
    BasicBlock *dispatch, const luisa::unordered_set<BasicBlock *> &loop_blocks,
    luisa::string_view &reason) noexcept {
    for (auto *block : handler.blocks) {
        for (auto *inst : block->instructions()) {
            // Lowering an outer loop first would clone a nested RayQueryLoopInst
            // into the new callback while the function worklist still points to
            // the dead original. Supporting that shape requires a deliberate
            // inner-to-outer pipeline, so reject it atomically for now.
            if (inst->isa<RayQueryLoopInst>()) {
                reason = "nested ray-query loops in candidate handlers are not supported";
                return false;
            }
            if (inst->isa<PhiInst>()) {
                auto *phi = static_cast<PhiInst *>(inst);
                for (size_t i = 0u; i < phi->incoming_count(); ++i) {
                    auto incoming = phi->incoming(i);
                    if (!handler.blocks.contains(incoming.block) ||
                        !value_is_outline_resolvable(incoming.value, query_object,
                                                     dispatch, handler, loop_blocks)) {
                        reason = "candidate handler PHI has an incoming edge/value outside its outline region";
                        return false;
                    }
                }
            }
            // Reject cross-handler SSA operands that the global loop capture
            // analysis intentionally regards as internal and therefore would
            // not turn into callback arguments.
            if (inst->isa<BranchInst>() &&
                static_cast<BranchInst *>(inst)->target_block() == dispatch) {
                // This terminator is materialized as ReturnInst by the outliner;
                // its dispatch target is deliberately absent from the resolver.
                continue;
            }
            for (auto *use : inst->operand_uses()) {
                if (!value_is_outline_resolvable(use->value(), query_object,
                                                 dispatch, handler, loop_blocks)) {
                    reason = "candidate handler uses an SSA value unavailable to its outlined callback";
                    return false;
                }
            }
        }
    }
    return true;
}

[[nodiscard]] static bool can_lower_ray_query_loop(RayQueryLoopInst *loop,
                                                   luisa::string_view &reason) noexcept {
    if (loop == nullptr || loop->parent_block() == nullptr) {
        reason = "null loop or parent block";
        return false;
    }
    auto *dispatch_block = loop->dispatch_block();
    auto *merge_block = loop->merge_block();
    if (dispatch_block == nullptr || merge_block == nullptr || !dispatch_block->is_terminated()) {
        reason = "null merge/dispatch block or unterminated dispatch block";
        return false;
    }
    auto *term = dispatch_block->terminator();
    if (!term->isa<RayQueryDispatchInst>()) {
        reason = "dispatch block is not terminated by RayQueryDispatchInst";
        return false;
    }
    for (auto *inst : dispatch_block->instructions()) {
        if (inst != term && !inst->isa<PhiInst>()) {
            reason = "dispatch block contains an unsupported non-PHI instruction";
            return false;
        }
    }
    auto *dispatch = static_cast<RayQueryDispatchInst *>(term);
    if (!is_ray_query_object(dispatch->query_object()) ||
        dispatch->exit_block() != merge_block) {
        reason = "dispatch requires an lvalue LC_RayQueryAll/"
                 "LC_RayQueryAny object and an exit matching the loop";
        return false;
    }
    auto *owner = loop->parent_block()->parent_function();
    if (owner == nullptr || dispatch_block->parent_function() != owner ||
        merge_block->parent_function() != owner) {
        reason = "ray-query loop references a block outside its function";
        return false;
    }
    for (auto *inst : dispatch_block->instructions()) {
        if (!inst->isa<PhiInst>()) { continue; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (size_t i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            if (incoming.block == nullptr ||
                incoming.block->parent_function() != owner ||
                !incoming.block->is_terminated()) {
                reason = "dispatch PHI has a null, foreign, or unterminated predecessor";
                return false;
            }
        }
    }
    auto *surface = dispatch->on_surface_candidate_block();
    auto *procedural = dispatch->on_procedural_candidate_block();
    if (surface == nullptr || procedural == nullptr) {
        reason = "dispatch has a null candidate handler";
        return false;
    }
    if (dispatch_block == merge_block || dispatch_block == loop->parent_block() ||
        merge_block == loop->parent_block() || surface == dispatch_block ||
        procedural == dispatch_block || surface == merge_block ||
        procedural == merge_block) {
        reason = "ray-query loop reuses a parent, dispatch, merge, or handler block";
        return false;
    }
    for (auto *inst : merge_block->instructions()) {
        if (inst->isa<PhiInst>()) {
            reason = "ray-query loop merge contains a PHI that cannot be moved atomically";
            return false;
        }
    }
    auto merge_has_external_predecessor = false;
    merge_block->traverse_predecessors(false, [&](BasicBlock *predecessor) noexcept {
        merge_has_external_predecessor |= predecessor != dispatch_block &&
                                          predecessor != loop->parent_block();
    });
    if (merge_has_external_predecessor) {
        reason = "ray-query loop merge has a predecessor outside the loop";
        return false;
    }
    auto merge_has_self_successor = false;
    merge_block->traverse_successors(false, [&](BasicBlock *successor) noexcept {
        merge_has_self_successor |= successor == merge_block;
    });
    if (merge_has_self_successor) {
        reason = "ray-query loop merge has a self edge";
        return false;
    }
    RayQueryHandlerRegion surface_region;
    RayQueryHandlerRegion procedural_region;
    if (!collect_outlineable_handler_region(surface, dispatch_block, merge_block,
                                            surface_region, reason) ||
        !collect_outlineable_handler_region(procedural, dispatch_block, merge_block,
                                            procedural_region, reason)) {
        return false;
    }
    if (lower_ray_query_loop_handler_regions_overlap(surface_region, procedural_region)) {
        reason = "surface and procedural candidate handler regions overlap";
        return false;
    }
    auto dispatch_has_external_predecessor = false;
    dispatch_block->traverse_predecessors(
        false, [&](BasicBlock *predecessor) noexcept {
            dispatch_has_external_predecessor |=
                predecessor != loop->parent_block() &&
                !surface_region.blocks.contains(predecessor) &&
                !procedural_region.blocks.contains(predecessor);
        });
    if (dispatch_has_external_predecessor) {
        reason = "ray-query dispatch has a predecessor outside the loop";
        return false;
    }
    if (lower_ray_query_loop_handler_region_has_external_predecessor(surface, dispatch_block, surface_region) ||
        lower_ray_query_loop_handler_region_has_external_predecessor(procedural, dispatch_block, procedural_region)) {
        reason = "candidate handler has a predecessor outside its outline region";
        return false;
    }
    luisa::unordered_set<BasicBlock *> loop_blocks;
    loop_blocks.emplace(dispatch_block);
    loop_blocks.insert(surface_region.blocks.begin(), surface_region.blocks.end());
    loop_blocks.insert(procedural_region.blocks.begin(), procedural_region.blocks.end());
    if (!validate_outline_resolver_inputs(surface_region, dispatch->query_object(),
                                          dispatch_block, loop_blocks, reason) ||
        !validate_outline_resolver_inputs(procedural_region, dispatch->query_object(),
                                          dispatch_block, loop_blocks, reason)) {
        return false;
    }
    return true;
}

[[nodiscard]] static luisa::vector<RayQueryLoopInst *>
collect_ray_query_loops(Function *function) noexcept {
    luisa::vector<RayQueryLoopInst *> loops;
    if (function == nullptr) { return loops; }
    if (auto *def = function->definition()) {
        for (auto *block : def->basic_blocks()) {
            if (block != nullptr && block->is_terminated()) {
                auto *terminator = block->terminator();
                if (terminator->isa<RayQueryLoopInst>()) {
                    loops.emplace_back(
                        static_cast<RayQueryLoopInst *>(terminator));
                }
            }
        }
    }
    return loops;
}

[[nodiscard]] static bool lower_ray_query_loop_preflight_ray_query_loops(
    luisa::span<RayQueryLoopInst *const> loops,
    RayQueryLoopLowerInfo &info) noexcept {
    auto rejected = false;
    for (auto *loop : loops) {
        luisa::string_view reason;
        if (!can_lower_ray_query_loop(loop, reason)) {
            LUISA_WARNING_WITH_LOCATION(
                "lower_ray_query_loop: rejecting loop: {}", reason);
            ++info.error_count;
            rejected = true;
        }
    }
    return !rejected;
}

struct RayQueryLoopSubgraph {
    Value *query_object;
    luisa::unordered_set<BasicBlock *> unordered;
    luisa::vector<BasicBlock *> reverse_post_order;
};

static void collect_ray_query_loop_basic_blocks_post_order(BasicBlock *block, const BasicBlock *merge,
                                                           RayQueryLoopSubgraph &subgraph) noexcept {
    if (block != merge && subgraph.unordered.emplace(block).second) {
        block->traverse_successors(true, [&](BasicBlock *succ) noexcept {
            collect_ray_query_loop_basic_blocks_post_order(succ, merge, subgraph);
        });
        // note that we are collecting post-order here
        subgraph.reverse_post_order.emplace_back(block);
    }
}

[[nodiscard]] static auto collect_ray_query_loop_subgraph(RayQueryLoopInst *loop) noexcept {
    // get dispatch and merge blocks
    auto dispatch_block = loop->dispatch_block();
    LUISA_DEBUG_ASSERT(dispatch_block != nullptr, "Invalid ray query loop dispatch block.");
    // get query object from dispatch block
    auto dispatch_inst = dispatch_block->terminator();
    LUISA_DEBUG_ASSERT(dispatch_inst != nullptr &&
                           dispatch_inst == dispatch_block->instructions().front() &&
                           dispatch_inst->isa<RayQueryDispatchInst>(),
                       "Invalid ray query loop dispatch instruction.");
    auto query_object = static_cast<RayQueryDispatchInst *>(dispatch_inst)->query_object();
    LUISA_DEBUG_ASSERT(query_object != nullptr, "Invalid ray query loop query object.");
    auto loop_merge = loop->control_flow_merge();
    LUISA_DEBUG_ASSERT(loop_merge != nullptr, "Invalid ray query loop control flow merge.");
    auto merge_block = loop_merge->merge_block();
    LUISA_DEBUG_ASSERT(merge_block != nullptr, "Invalid ray query loop merge block.");
    // collect subgraph
    RayQueryLoopSubgraph subgraph{.query_object = query_object};
    collect_ray_query_loop_basic_blocks_post_order(dispatch_block, merge_block, subgraph);
    // post-order to reverse post-order
    std::reverse(subgraph.reverse_post_order.begin(), subgraph.reverse_post_order.end());
    LUISA_DEBUG_ASSERT(subgraph.reverse_post_order.front() == dispatch_block, "Invalid ray query loop dispatch block.");
    return subgraph;
}

struct RayQueryLoopCaptureList {
    // values that are defined outside the loop but used inside (including
    // variables, excluding the query object and other non-instruction values)
    luisa::vector<Value *> in_values;
    // values that are defined inside the loop but used outside, which we
    // must create variables for passing them out of the loop
    luisa::vector<Instruction *> out_values;
};

static void collect_ray_query_loop_capture_list_in_inst(Instruction *inst, const Value *query_object,
                                                        const luisa::unordered_set<Value *> &internal,
                                                        luisa::unordered_set<Value *> &known_in,
                                                        RayQueryLoopCaptureList &list) noexcept {
    // check if any user of the value is outside the loop
    for (auto &&use : inst->use_list()) {
        if (auto user = use->user(); user != nullptr && !internal.contains(user)) {
            list.out_values.emplace_back(inst);
            break;
        }
    }
    // check if any operand of the value is outside the loop
    auto is_interested_value = [&](Value *value) noexcept {
        // check non-null and not query object
        if (value == nullptr || value == query_object) { return false; }
        // check value type
        switch (value->derived_value_tag()) {
            case DerivedValueTag::UNDEFINED: [[fallthrough]];
            case DerivedValueTag::FUNCTION: [[fallthrough]];
            case DerivedValueTag::BASIC_BLOCK: [[fallthrough]];
            case DerivedValueTag::CONSTANT: [[fallthrough]];
            case DerivedValueTag::SPECIAL_REGISTER: return false;
            case DerivedValueTag::INSTRUCTION: [[fallthrough]];
            case DerivedValueTag::ARGUMENT: break;
            default: LUISA_ERROR_WITH_LOCATION("Unknown derived value tag.");
        }
        // check if the value is defined inside the loop
        if (internal.contains(value)) { return false; }
        // check if the value is already known
        return known_in.emplace(value).second;
    };
    for (auto &&op_use : inst->operand_uses()) {
        if (auto op = op_use->value(); is_interested_value(op)) {
            list.in_values.emplace_back(op);
        }
    }
}

[[nodiscard]] static auto collect_ray_query_loop_capture_list(const RayQueryLoopSubgraph &subgraph) noexcept {
    RayQueryLoopCaptureList capture_list;
    luisa::unordered_set<Value *> known_in;
    luisa::unordered_set<Value *> internal;
    for (auto block : subgraph.reverse_post_order) {
        for (auto inst : block->instructions()) {
            internal.emplace(inst);
        }
    }
    for (auto block : subgraph.reverse_post_order) {
        for (auto &&inst : block->instructions()) {
            collect_ray_query_loop_capture_list_in_inst(
                inst, subgraph.query_object,
                internal, known_in, capture_list);
        }
    }
    return capture_list;
}

class RayQueryLowerPassValueResolver final : public InstructionCloneValueResolver {

private:
    luisa::unordered_map<const Value *, Value *> _value_map;

public:
    bool emplace(const Value *original, Value *duplicate) noexcept {
        return _value_map.emplace(original, duplicate).second;
    }
    [[nodiscard]] Value *resolve_or_null(const Value *value) noexcept {
        if (value == nullptr) { return nullptr; }
        switch (value->derived_value_tag()) {
            case DerivedValueTag::UNDEFINED: [[fallthrough]];
            case DerivedValueTag::FUNCTION: [[fallthrough]];
            case DerivedValueTag::CONSTANT: [[fallthrough]];
            case DerivedValueTag::SPECIAL_REGISTER: return const_cast<Value *>(value);
            case DerivedValueTag::BASIC_BLOCK: [[fallthrough]];
            case DerivedValueTag::INSTRUCTION: [[fallthrough]];
            case DerivedValueTag::ARGUMENT: break;
            default: LUISA_ERROR_WITH_LOCATION("Invalid value.");
        }
        auto iter = _value_map.find(value);
        return iter == _value_map.end() ? nullptr : iter->second;
    }
    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) { return nullptr; }
        auto resolved = resolve_or_null(value);
        LUISA_DEBUG_ASSERT(resolved != nullptr, "Value not found in the resolver.");
        return resolved;
    }
};

static BasicBlock *duplicate_basic_block_for_ray_query_loop_dispatch_branch(const BasicBlock *original, const BasicBlock *merge,
                                                                            luisa::vector<std::pair<const PhiInst *, PhiInst *>> &phi_nodes,
                                                                            RayQueryLowerPassValueResolver &resolver) noexcept {
    auto bb = static_cast<BasicBlock *>(resolver.resolve(original));
    clone_metadata(*original, *bb);
    XIRBuilder b;
    b.set_insertion_point(bb);
    for (auto inst : original->instructions()) {
        // special case: branch to the merge block
        if (inst->is_terminator() && inst->isa<BranchInst>() &&
            static_cast<const BranchInst *>(inst)->target_block() == merge) {
            auto *return_inst = b.return_void();
            clone_metadata(*inst, *return_inst);
        } else if (inst->isa<PhiInst>()) {
            auto dup_phi = b.phi(inst->type());
            clone_metadata(*inst, *dup_phi);
            phi_nodes.emplace_back(static_cast<const PhiInst *>(inst), dup_phi);
            resolver.emplace(inst, dup_phi);
        } else {
            auto dup_inst = inst->clone_with_metadata(b, resolver);
            LUISA_DEBUG_ASSERT(dup_inst != nullptr, "Failed to duplicate instruction.");
            resolver.emplace(inst, dup_inst);
        }
    }
    return bb;
}

[[nodiscard]] static Function *outline_ray_query_loop_dispatch_branch(Module *module, BasicBlock *branch,
                                                                      Value *query_object, const BasicBlock *dispatch,
                                                                      const RayQueryLoopCaptureList &capture_list,
                                                                      luisa::string_view comment) noexcept {
    // check if the branch is nullptr
    if (branch == nullptr) { return nullptr; }
    // create the function
    auto function = module->create_callable(nullptr);
    function->add_comment(comment);
    // compute the subgraph for the branch block
    RayQueryLoopSubgraph subgraph{.query_object = query_object};
    collect_ray_query_loop_basic_blocks_post_order(branch, dispatch, subgraph);
    std::reverse(subgraph.reverse_post_order.begin(), subgraph.reverse_post_order.end());
    // check that the first block is the branch
    LUISA_DEBUG_ASSERT(subgraph.reverse_post_order.front() == branch, "Invalid branch block.");
    // value map for renaming
    RayQueryLowerPassValueResolver resolver;
    // create an argument for the query object
    LUISA_DEBUG_ASSERT(query_object != nullptr && query_object->is_lvalue(), "Invalid query object.");
    auto query_arg = function->create_reference_argument(query_object->type());
    resolver.emplace(query_object, query_arg);
    // create arguments for in values
    for (auto in_value : capture_list.in_values) {
        auto in_arg = function->create_argument(in_value->type(), in_value->is_lvalue());
        resolver.emplace(in_value, in_arg);
    }
    // create blocks for the function
    for (auto block : subgraph.reverse_post_order) {
        auto local_block = function->create_basic_block();
        resolver.emplace(block, local_block);
    }
    // set function body
    function->set_body_block(static_cast<BasicBlock *>(resolver.resolve(branch)));
    // duplicate the blocks
    auto already_returned = false;
    luisa::vector<std::pair<const PhiInst *, PhiInst *>> phi_nodes;
    for (auto block : subgraph.reverse_post_order) {
        if (auto bb = duplicate_basic_block_for_ray_query_loop_dispatch_branch(block, dispatch, phi_nodes, resolver);
            bb->terminator()->isa<ReturnInst>()) {
            LUISA_ASSERT(!already_returned, "Multiple return instructions in the branch block.");
            already_returned = true;
            // generate store instructions for out values
            XIRBuilder b;
            b.set_insertion_point(bb->terminator()->prev());
            for (auto out_value : capture_list.out_values) {
                auto out_arg = function->create_reference_argument(out_value->type());
                if (auto resolved = resolver.resolve_or_null(out_value)) {
                    b.store(out_arg, resolved);
                }
            }
        }
    }
    // fix phi nodes
    for (auto [original_phi, dup_phi] : phi_nodes) {
        dup_phi->set_incoming_count(original_phi->incoming_count());
        for (size_t i = 0; i < original_phi->incoming_count(); i++) {
            auto incoming = original_phi->incoming(i);
            auto resolved_value = resolver.resolve(incoming.value);
            auto resolved_block = resolver.resolve(incoming.block);
            LUISA_DEBUG_ASSERT(resolved_block->isa<BasicBlock>(), "Invalid resolved block.");
            dup_phi->set_incoming(i, resolved_value, static_cast<BasicBlock *>(resolved_block));
        }
    }
    return function;
}

static void lower_ray_query_loop(Function *function, RayQueryLoopInst *loop, RayQueryLoopLowerInfo &info) noexcept {
    auto subgraph = collect_ray_query_loop_subgraph(loop);
    auto capture_list = collect_ray_query_loop_capture_list(subgraph);
    auto dispatch = static_cast<RayQueryDispatchInst *>(subgraph.reverse_post_order.front()->terminator());
    auto merge_block = loop->control_flow_merge()->merge_block();
    LUISA_DEBUG_ASSERT(dispatch->exit_block() == merge_block, "Invalid ray query loop exit block.");
    LUISA_DEBUG_ASSERT(function->parent_module() != nullptr, "Invalid function module.");
    auto on_surface = outline_ray_query_loop_dispatch_branch(
        function->parent_module(), dispatch->on_surface_candidate_block(), subgraph.query_object,
        subgraph.reverse_post_order.front(), capture_list,
        "on_surface function outlined from ray query loop");
    auto on_procedural = outline_ray_query_loop_dispatch_branch(
        function->parent_module(), dispatch->on_procedural_candidate_block(), subgraph.query_object,
        subgraph.reverse_post_order.front(), capture_list,
        "on_procedural function outlined from ray query loop");
    // prepare captured arguments
    luisa::vector<Value *> captured_args;
    captured_args.reserve(capture_list.in_values.size() + capture_list.out_values.size());
    for (auto in_value : capture_list.in_values) {
        captured_args.emplace_back(in_value);
    }
    // create variables for out values
    if (!capture_list.out_values.empty()) {
        XIRBuilder b;
        b.set_insertion_point(function->definition()->body_block()->instructions().front());
        for (auto out_value : capture_list.out_values) {
            auto variable = b.alloca_local(out_value->type());
            variable->add_comment("alloca for ray query output value");
            captured_args.emplace_back(variable);
        }
    }
    // create ray query pipeline
    XIRBuilder b;
    b.set_insertion_point(loop->prev());
    auto loop_parent_block = loop->parent_block();
    auto *pipeline = b.ray_query_pipeline(
        subgraph.query_object, on_surface, on_procedural, captured_args);
    // The pipeline is the unique semantic replacement for both the candidate
    // dispatch and its enclosing loop. Clone dispatch provenance first so the
    // enclosing loop's name/location remains the primary identity if both
    // sources carry single-valued metadata.
    clone_metadata(*dispatch, *pipeline);
    clone_metadata(*loop, *pipeline);
    // remove the loop and record the change
    {
        loop->remove_self();
        info.lowered_loop_count++;
    }
    // load the out values and replace the uses
    auto out_variables = luisa::span{captured_args}.subspan(capture_list.in_values.size());
    for (size_t i = 0; i < capture_list.out_values.size(); i++) {
        auto old_out_value = capture_list.out_values[i];
        auto out_variable = out_variables[i];
        auto out_value = b.load(old_out_value->type(), out_variable);
        out_value->add_comment("load from ray query output alloca");
        old_out_value->replace_all_uses_with(out_value);
    }
    // rewrite the PHI nodes in merge block's successors
    merge_block->traverse_successors(false, [&](BasicBlock *succ) noexcept {
        LUISA_ASSERT(succ != merge_block, "Invalid successor.");
        succ->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<PhiInst>()) {
                auto phi = static_cast<PhiInst *>(inst);
                for (size_t i = 0; i < phi->incoming_count(); i++) {
                    if (auto incoming = phi->incoming(i); incoming.block == merge_block) {
                        phi->set_incoming(i, incoming.value, loop_parent_block);
                    }
                }
            }
        });
    });
    // move the instructions from the merge block to the loop parent block
    while (!merge_block->instructions().empty()) {
        auto inst = merge_block->instructions().front();
        LUISA_ASSERT(!inst->isa<PhiInst>(), "Invalid phi instruction in merge block.");
        b.append(inst->remove_self());
    }
    // add an unreachable instruction to the merge block
    b.set_insertion_point(merge_block);
    b.unreachable_();
}

static void collect_blocks_in_ray_query_dispatch_branch(BasicBlock *block, BasicBlock *dispatch_block,
                                                        luisa::unordered_set<BasicBlock *> &collected) noexcept {
    if (block != nullptr && block != dispatch_block && collected.emplace(block).second) {
        block->traverse_successors(true, [&](BasicBlock *succ) noexcept {
            collect_blocks_in_ray_query_dispatch_branch(succ, dispatch_block, collected);
        });
    }
}

static void replace_phi_uses_with_local_load_in_blocks(BasicBlock *block, PhiInst *phi, AllocaInst *phi_alloca,
                                                       const luisa::unordered_set<BasicBlock *> &collected_blocks) noexcept {
    if (block != nullptr) {
        luisa::fixed_vector<Use *, 64> local_uses;
        for (auto &&use : phi->use_list()) {
            if (auto user = use->user()) {
                LUISA_DEBUG_ASSERT(user->isa<Instruction>(), "Invalid user.");
                if (auto user_inst = static_cast<Instruction *>(user); collected_blocks.contains(user_inst->parent_block())) {
                    local_uses.emplace_back(use);
                }
            }
        }
        if (!local_uses.empty()) {
            XIRBuilder b;
            b.set_insertion_point(block->instructions().head_sentinel());
            auto phi_load = b.load(phi->type(), phi_alloca);
            phi_load->add_comment("load from phi alloca");
            clone_metadata(*phi, *phi_load);
            for (auto use : local_uses) {
                User::set_operand_use_value(use, phi_load);
            }
        }
    }
}

static void lower_phi_nodes_in_loop_dispatch_block(FunctionDefinition *f, RayQueryLoopInst *loop) noexcept {
    auto dispatch_block = loop->dispatch_block();
    LUISA_DEBUG_ASSERT(dispatch_block != nullptr, "Invalid dispatch block.");
    // collect phi nodes
    luisa::fixed_vector<PhiInst *, 16> phi_nodes;
    for (auto inst : dispatch_block->instructions()) {
        switch (auto tag = inst->derived_instruction_tag()) {
            case DerivedInstructionTag::RAY_QUERY_DISPATCH: {
                LUISA_DEBUG_ASSERT(inst == dispatch_block->terminator(),
                                   "Invalid terminator.");
                break;
            }
            case DerivedInstructionTag::PHI: {
                phi_nodes.emplace_back(static_cast<PhiInst *>(inst));
                break;
            }
            default: LUISA_ERROR_WITH_LOCATION(
                "Unexpected instruction {} in ray query loop dispatch block.",
                xir::to_string(tag));
        }
    }
    if (!phi_nodes.empty()) {
        auto dispatch_inst = [&] {
            auto terminator = dispatch_block->terminator();
            LUISA_DEBUG_ASSERT(terminator->isa<RayQueryDispatchInst>(), "Invalid terminator.");
            return static_cast<RayQueryDispatchInst *>(terminator);
        }();
        // collect surface and procedural blocks
        auto surface_block = dispatch_inst->on_surface_candidate_block();
        auto procedural_block = dispatch_inst->on_procedural_candidate_block();
        luisa::unordered_set<BasicBlock *> surface_blocks;
        luisa::unordered_set<BasicBlock *> procedural_blocks;
        collect_blocks_in_ray_query_dispatch_branch(surface_block, dispatch_block, surface_blocks);
        collect_blocks_in_ray_query_dispatch_branch(procedural_block, dispatch_block, procedural_blocks);
        // lower the phi nodes to local variables
        XIRBuilder b;
        for (auto phi : phi_nodes) {
            b.set_insertion_point(f->body_block()->instructions().head_sentinel());
            auto phi_alloca = b.alloca_local(phi->type());
            phi_alloca->add_comment("alloca to lower phi node in ray query loop");
            clone_metadata(*phi, *phi_alloca);
            static constexpr auto is_undef = [](Value *v) noexcept {
                return v == nullptr || v->isa<Undefined>();
            };
            for (size_t i = 0; i < phi->incoming_count(); i++) {
                if (auto incoming = phi->incoming(i); !is_undef(incoming.value)) {
                    b.set_insertion_point(incoming.block->terminator()->prev());
                    b.store(phi_alloca, incoming.value);
                }
            }
            replace_phi_uses_with_local_load_in_blocks(surface_block, phi, phi_alloca, surface_blocks);
            replace_phi_uses_with_local_load_in_blocks(procedural_block, phi, phi_alloca, procedural_blocks);
#ifndef NDEBUG
            for (auto &&use : phi->use_list()) {
                if (auto user = use->user()) {
                    LUISA_DEBUG_ASSERT(user->isa<Instruction>(), "Invalid user.");
                    auto user_block = static_cast<Instruction *>(user)->parent_block();
                    LUISA_DEBUG_ASSERT(!surface_blocks.contains(user_block) && !procedural_blocks.contains(user_block),
                                       "Phi node uses should have been lowered in surface or procedural blocks.");
                }
            }
#endif
            if (auto exit_block = dispatch_inst->exit_block()) {
                b.set_insertion_point(exit_block->instructions().head_sentinel());
                auto phi_load = b.load(phi->type(), phi_alloca);
                phi_load->add_comment("load from phi alloca in ray query exit block");
                clone_metadata(*phi, *phi_load);
                phi->replace_all_uses_with(phi_load);
            }
            LUISA_DEBUG_ASSERT(phi->use_list().empty(), "Phi node has uses but no exit block.");
            phi->remove_self();
        }
    }
}

static void lower_ray_query_loop_lower_preflighted_ray_query_loops(
    Function *function, luisa::span<RayQueryLoopInst *const> loops,
    RayQueryLoopLowerInfo &info) noexcept {
    auto *def = function == nullptr ? nullptr : function->definition();
    if (def == nullptr) { return; }
    auto lowered_before = info.lowered_loop_count;
    for (auto *loop : loops) {
        lower_phi_nodes_in_loop_dispatch_block(def, loop);
        hoist_alloca_instructions_to_entry_block(def);
        lower_ray_query_loop(function, loop, info);
    }
    // Remove dead code after lowering using the DCE pass.
    if (info.lowered_loop_count != lowered_before) {
        auto dce_info = dce_pass_run_on_function(function);
        LUISA_VERBOSE(
            "Removed {} dead instruction(s) and {} dead block(s) after "
            "lowering ray query loop(s).",
            dce_info.removed_inst_count, dce_info.removed_block_count);
    }
}

static void run_lower_ray_query_loop_pass_on_function(
    Function *function, RayQueryLoopLowerInfo &info) noexcept {
    auto loops = collect_ray_query_loops(function);
    // Preflight the complete function before touching dispatch PHIs, hoisting
    // allocas, creating callbacks, or running function-wide DCE.
    if (!lower_ray_query_loop_preflight_ray_query_loops(luisa::span{loops}, info)) { return; }
    lower_ray_query_loop_lower_preflighted_ray_query_loops(
        function, luisa::span{loops}, info);
}

}// namespace detail

RayQueryLoopLowerInfo lower_ray_query_loop_pass_run_on_function(Function *function) noexcept {
    RayQueryLoopLowerInfo info;
    detail::run_lower_ray_query_loop_pass_on_function(function, info);
    return info;
}

RayQueryLoopLowerInfo lower_ray_query_loop_pass_run_on_module(Module *module, PassReport *report) noexcept {
    RayQueryLoopLowerInfo info;
    struct FunctionWork {
        Function *function;
        luisa::vector<RayQueryLoopInst *> loops;
    };
    luisa::vector<FunctionWork> work;
    if (module != nullptr) {
        // Snapshot the original function list: successful lowering appends two
        // callback functions per loop, and those generated functions are not
        // part of this pass invocation's input domain.
        for (auto *function : module->function_list()) {
            work.emplace_back(FunctionWork{
                .function = function,
                .loops = detail::collect_ray_query_loops(function)});
        }
        // The module overload is transactional as well as the function
        // overload. Discover every rejection before the first callback,
        // alloca, pipeline, or DCE mutation is created.
        auto accepted = true;
        for (auto &item : work) {
            accepted &= detail::lower_ray_query_loop_preflight_ray_query_loops(
                luisa::span{item.loops}, info);
        }
        if (accepted) {
            for (auto &item : work) {
                detail::lower_ray_query_loop_lower_preflighted_ray_query_loops(
                    item.function, luisa::span{item.loops}, info);
            }
        }
    }
    if (report != nullptr) {
        report->set("lowered_loop", info.lowered_loop_count);
        report->set("error", info.error_count);
    }
    return info;
}

}// namespace luisa::compute::xir
