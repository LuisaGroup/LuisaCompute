#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/module.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/aggregate_field_bitmask.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>
#include <luisa/xir/passes/lower_ray_query_to_pipeline.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include <algorithm>
#include <limits>

#include "helpers.h"
#include "lower_ray_query_handler_graph.h"
#include "lower_ray_query_to_pipeline_capture_cost.h"
namespace luisa::compute::xir {

// This result is returned by the established exported overloads. Growing it
// changes the platform return convention (for example, register return to
// sret on x86-64), so new optional outputs belong in the options overload.
static_assert(sizeof(LowerRayQueryToPipelineInfo) == 2u * sizeof(size_t));

namespace detail {

struct LowerRayQueryToPipelineRunInfo : LowerRayQueryToPipelineInfo {
    size_t localized_alloca_count{0u};
    size_t selection_localization_analysis_count{0u};
    size_t handler_localization_instruction_evaluation_count{0u};
    size_t handler_localization_analysis_count{0u};
    size_t handler_localization_indexed_instruction_count{0u};
    size_t handler_localization_relevant_instruction_count{0u};
    size_t handler_localization_avoided_instruction_scan_count{0u};
    size_t handler_localization_block_evaluation_count{0u};
    bool verify_handler_scratch_graph{false};
};

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
    if (region.dispatch_exit_count == 0u) {
        reason = "candidate handler has no exit to dispatch";
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

[[nodiscard]] static bool lower_ray_query_to_pipeline_handler_regions_overlap(
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

[[nodiscard]] static bool lower_ray_query_to_pipeline_handler_region_has_external_predecessor(
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
    if (lower_ray_query_to_pipeline_handler_regions_overlap(surface_region, procedural_region)) {
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
    if (lower_ray_query_to_pipeline_handler_region_has_external_predecessor(surface, dispatch_block, surface_region) ||
        lower_ray_query_to_pipeline_handler_region_has_external_predecessor(procedural, dispatch_block, procedural_region)) {
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

[[nodiscard]] static bool lower_ray_query_to_pipeline_preflight_ray_query_loops(
    luisa::span<RayQueryLoopInst *const> loops,
    LowerRayQueryToPipelineRunInfo &info) noexcept {
    auto rejected = false;
    for (auto *loop : loops) {
        luisa::string_view reason;
        if (!can_lower_ray_query_loop(loop, reason)) {
            LUISA_WARNING_WITH_LOCATION(
                "lower_ray_query_to_pipeline: rejecting loop: {}", reason);
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

struct RayQueryHandlerLocalAllocas {
    luisa::vector<AllocaInst *> surface;
    luisa::vector<AllocaInst *> procedural;
    luisa::unordered_set<Value *> all;
};

struct RayQueryHandlerRootUseInfo {
    bool valid{true};
    bool used_by_surface{false};
    bool used_by_procedural{false};
};

// Pointer identity is already unique inside one XIR module. Hashing the bytes
// of an address with the general content hash is unnecessary work for these
// short-lived compiler analyses, especially when system STL is selected.
// Deliberately omit `is_avalanching`: dense hash tables may still mix the raw
// identity, while std::unordered_* receives the ordinary address hash.
struct RayQueryHandlerPointerIdentityHash {
    [[nodiscard]] size_t operator()(const void *pointer) const noexcept {
        return static_cast<size_t>(reinterpret_cast<uintptr_t>(pointer));
    }
};

template<typename T>
using RayQueryHandlerPointerSet =
    luisa::unordered_set<T *, RayQueryHandlerPointerIdentityHash>;

template<typename T, typename V>
using RayQueryHandlerPointerMap =
    luisa::unordered_map<T *, V, RayQueryHandlerPointerIdentityHash>;

// The root-use proof is deliberately separate from definite initialization.
// It establishes ownership: outside-handler writes are killed incoming state,
// while every observation or escape must belong to exactly one candidate kind.
// GEP is the only address-preserving XIR instruction, and a direct Callable is
// the only interprocedural edge through which an lvalue may legally flow.
static void collect_handler_alloca_root_uses(
    Value *pointer, const RayQueryHandlerRegion &surface_region,
    const RayQueryHandlerRegion &procedural_region,
    RayQueryHandlerRootUseInfo &info,
    RayQueryHandlerPointerSet<const Value> &visited) noexcept {
    if (!info.valid || pointer == nullptr ||
        !visited.emplace(pointer).second) {
        return;
    }
    for (auto &&use : pointer->use_list()) {
        auto *user = use->user();
        if (user == nullptr || !user->isa<Instruction>()) {
            info.valid = false;
            return;
        }
        auto *inst = static_cast<Instruction *>(user);
        if (inst->isa<GEPInst>() &&
            static_cast<GEPInst *>(inst)->base() == pointer) {
            collect_handler_alloca_root_uses(
                inst, surface_region, procedural_region, info, visited);
            continue;
        }
        auto *block = inst->parent_block();
        auto in_surface = surface_region.blocks.contains(block);
        auto in_procedural = procedural_region.blocks.contains(block);
        if (in_surface || in_procedural) {
            info.used_by_surface |= in_surface;
            info.used_by_procedural |= in_procedural;
            if (inst->isa<LoadInst>() &&
                static_cast<LoadInst *>(inst)->variable() == pointer) {
                continue;
            }
            if (inst->isa<StoreInst>() &&
                static_cast<StoreInst *>(inst)->variable() == pointer &&
                static_cast<StoreInst *>(inst)->value() != pointer) {
                continue;
            }
            if (inst->isa<CallInst>()) {
                auto *call = static_cast<CallInst *>(inst);
                auto *callee = call->callee();
                auto argument_index = std::numeric_limits<size_t>::max();
                for (auto i = 0u; i < call->argument_count(); ++i) {
                    if (call->argument(i) == pointer) {
                        argument_index = i;
                        break;
                    }
                }
                if (callee != nullptr && callee->definition() != nullptr &&
                    argument_index < callee->arguments().count_size()) {
                    Argument *argument = nullptr;
                    auto index = 0u;
                    for (auto *candidate : callee->arguments()) {
                        if (index++ == argument_index) {
                            argument = candidate;
                            break;
                        }
                    }
                    if (argument != nullptr && argument->is_reference()) {
                        continue;
                    }
                }
            }
            info.valid = false;
            return;
        }
        // A write outside the candidate supplies only the incoming value. The
        // field-level proof below must show that no handler read can observe it.
        if (inst->isa<StoreInst>() &&
            static_cast<StoreInst *>(inst)->variable() == pointer &&
            static_cast<StoreInst *>(inst)->value() != pointer) {
            continue;
        }
        info.valid = false;
        return;
    }
}

struct RayQueryHandlerPointerView {
    luisa::vector<luisa::optional<size_t>> access_pattern;
    // A runtime vector/array index identifies one stable field at execution
    // time, but does not prove a compile-time must-definition of every field in
    // its may-mask. Reads may use the union; writes generate no must bits.
    bool precise{true};
};

using RayQueryHandlerPointerEnvironment =
    RayQueryHandlerPointerMap<const Value, RayQueryHandlerPointerView>;

using RayQueryHandlerPointerSupport =
    RayQueryHandlerPointerSet<const Value>;
using RayQueryHandlerActiveFunctions =
    RayQueryHandlerPointerSet<const Function>;
using RayQueryHandlerInstructionSchedule =
    RayQueryHandlerPointerMap<const BasicBlock,
                              luisa::vector<const Instruction *>>;

struct RayQueryHandlerPointerResolveResult {
    bool valid{true};
    bool related{false};
    RayQueryHandlerPointerView view;
};

// `collect_handler_alloca_root_uses` follows every use edge from one root and
// recurses through the only address-preserving XIR instruction, GEP. Any other
// top-level use is either a classified load/store, an explicit reference-call
// boundary, or rejects localization. Its visited set is therefore the exact
// support of this root in the parent function's pointer product lattice. An
// instruction whose lvalue operands lie outside that support has the identity
// effect for this coordinate. Callable bodies are analyzed without this
// parent-function filter after binding their reference formals.

struct RayQueryHandlerScratchEffect {
    const Type *type;
    luisa::optional<AggregateFieldBitmask> need;
    luisa::optional<AggregateFieldBitmask> define;
    bool valid{true};

    explicit RayQueryHandlerScratchEffect(const Type *type) noexcept
        : type{type} {}

    [[nodiscard]] bool needs_input() const noexcept {
        return need && need->access().any();
    }
};

// Candidate-local scratch is a path effect over primitive aggregate leaves.
// For sequential effects A then B:
//   need = A.need union (B.need - A.define)
//   define = A.define union B.define.
// At a CFG join, need is path union and define is path intersection. These are
// the exact transfer/join operations for "may read before a must definition".
static void union_mask(
    luisa::optional<AggregateFieldBitmask> &target,
    const AggregateFieldBitmask &incoming) noexcept {
    if (target) {
        *target |= incoming;
    } else {
        target.emplace(incoming);
    }
}

static void append_handler_scratch_effect(
    RayQueryHandlerScratchEffect &first,
    const RayQueryHandlerScratchEffect &second) noexcept {
    LUISA_DEBUG_ASSERT(first.type == second.type, "Type mismatch.");
    first.valid &= second.valid;
    if (!first.valid) { return; }
    if (second.need) {
        if (first.define) {
            auto uncovered = *second.need & ~*first.define;
            if (uncovered.access().any()) {
                union_mask(first.need, uncovered);
            }
        } else {
            union_mask(first.need, *second.need);
        }
    }
    if (second.define) { union_mask(first.define, *second.define); }
}

struct RayQueryHandlerScratchBlockState {
    bool reached{false};
    RayQueryHandlerScratchEffect effect;

    explicit RayQueryHandlerScratchBlockState(const Type *type) noexcept
        : effect{type} {}
};

class RayQueryHandlerScratchAnalyzer {

private:
    const Type *_root_type;
    LowerRayQueryToPipelineRunInfo *_run_info;
    const RayQueryHandlerGraph *_graph;

private:
    [[nodiscard]] RayQueryHandlerPointerResolveResult resolve_pointer(
        const Value *value,
        const RayQueryHandlerPointerEnvironment &environment,
        RayQueryHandlerPointerMap<
            const Value, RayQueryHandlerPointerResolveResult> &cache)
        const noexcept {
        if (value == nullptr || !value->is_lvalue()) { return {}; }
        if (auto iter = cache.find(value); iter != cache.end()) {
            return iter->second;
        }
        if (auto iter = environment.find(value);
            iter != environment.end()) {
            RayQueryHandlerPointerResolveResult result;
            result.related = true;
            result.view = iter->second;
            cache.emplace(value, result);
            return result;
        }
        // A pessimistic cache entry is the DFS gray state. Encountering it
        // through a malformed cyclic GEP chain returns invalid; successful
        // resolution overwrites it with the black-state summary below.
        RayQueryHandlerPointerResolveResult visiting;
        visiting.valid = false;
        auto [cache_iter, inserted] = cache.emplace(value, visiting);
        LUISA_DEBUG_ASSERT(inserted, "Duplicate pointer-resolution state.");
        static_cast<void>(cache_iter);
        RayQueryHandlerPointerResolveResult result;
        if (value->isa<GEPInst>()) {
            auto *gep = static_cast<const GEPInst *>(value);
            result = resolve_pointer(
                gep->base(), environment, cache);
            if (result.valid && result.related) {
                for (auto i = 0u; i < gep->index_count(); ++i) {
                    auto *index = gep->index(i);
                    if (index == nullptr || index->type() == nullptr) {
                        result.valid = false;
                        break;
                    }
                    if (index->isa<Constant>()) {
                        uint64_t decoded = 0u;
                        if (!try_decode_constant_nonnegative_integer(
                                index, decoded) ||
                            decoded > static_cast<uint64_t>(SIZE_MAX)) {
                            result.valid = false;
                            break;
                        }
                        result.view.access_pattern.emplace_back(
                            static_cast<size_t>(decoded));
                    } else {
                        result.view.access_pattern.emplace_back(luisa::nullopt);
                        result.view.precise = false;
                    }
                }
            }
        }
        auto resolved_iter = cache.find(value);
        LUISA_DEBUG_ASSERT(resolved_iter != cache.end(),
                           "Lost pointer-resolution state.");
        resolved_iter->second = result;
        return result;
    }

    [[nodiscard]] RayQueryHandlerScratchEffect access_effect(
        const RayQueryHandlerPointerView &view, bool read,
        bool write) const noexcept {
        RayQueryHandlerScratchEffect effect{_root_type};
        AggregateFieldBitmask mask{_root_type};
        if (!mask.mark_access_pattern(view.access_pattern)) {
            effect.valid = false;
            return effect;
        }
        if (read) { effect.need.emplace(mask); }
        if (write && view.precise) {
            effect.define.emplace(std::move(mask));
        }
        return effect;
    }

    [[nodiscard]] RayQueryHandlerScratchEffect summarize_callable(
        const Function *function,
        RayQueryHandlerPointerEnvironment environment,
        RayQueryHandlerActiveFunctions &active_functions) const noexcept {
        RayQueryHandlerScratchEffect invalid{_root_type};
        invalid.valid = false;
        if (function == nullptr || function->definition() == nullptr ||
            function->definition()->body_block() == nullptr ||
            !active_functions.emplace(function).second) {
            return invalid;
        }
        auto *definition =
            const_cast<Function *>(function)->definition();
        luisa::unordered_set<BasicBlock *> blocks;
        for (auto *block : definition->basic_blocks()) {
            blocks.emplace(block);
        }
        RayQueryHandlerGraph graph{blocks};
        auto result = summarize_region(
            definition->body_block(), graph,
            std::move(environment), nullptr, nullptr, false,
            active_functions);
        active_functions.erase(function);
        return result;
    }

    [[nodiscard]] RayQueryHandlerScratchEffect instruction_effect(
        const Instruction *instruction,
        const RayQueryHandlerPointerEnvironment &environment,
        RayQueryHandlerPointerMap<
            const Value, RayQueryHandlerPointerResolveResult> &cache,
        const RayQueryHandlerPointerSupport *pointer_support,
        RayQueryHandlerActiveFunctions &active_functions) const noexcept {
        if (_run_info != nullptr) {
            ++_run_info->handler_localization_instruction_evaluation_count;
        }
        RayQueryHandlerScratchEffect effect{_root_type};
        if (instruction == nullptr) {
            effect.valid = false;
            return effect;
        }
        auto require_unrelated_operand = [&](size_t index) noexcept {
            auto *operand = instruction->operand(index);
            if (operand == nullptr || !operand->is_lvalue()) { return; }
            if (pointer_support != nullptr &&
                !pointer_support->contains(operand)) {
                return;
            }
            auto resolved = resolve_pointer(
                operand, environment, cache);
            effect.valid &= resolved.valid && !resolved.related;
        };
        switch (instruction->derived_instruction_tag()) {
            case DerivedInstructionTag::GEP: {
                if (instruction->operand_count() == 0u) {
                    effect.valid = false;
                    break;
                }
                auto *base =
                    static_cast<const GEPInst *>(instruction)->base();
                if (pointer_support != nullptr &&
                    !pointer_support->contains(base)) {
                    for (auto i = 1u;
                         effect.valid &&
                         i < instruction->operand_count();
                         ++i) {
                        require_unrelated_operand(i);
                    }
                    break;
                }
                auto resolved = resolve_pointer(
                    base, environment, cache);
                effect.valid &= resolved.valid;
                for (auto i = 1u;
                     effect.valid && i < instruction->operand_count(); ++i) {
                    require_unrelated_operand(i);
                }
                break;
            }
            case DerivedInstructionTag::LOAD: {
                if (instruction->operand_count() != 1u) {
                    effect.valid = false;
                    break;
                }
                auto *variable =
                    static_cast<const LoadInst *>(instruction)->variable();
                if (pointer_support != nullptr &&
                    !pointer_support->contains(variable)) {
                    break;
                }
                auto resolved = resolve_pointer(
                    variable, environment, cache);
                effect.valid &= resolved.valid;
                if (resolved.related) {
                    effect = access_effect(resolved.view, true, false);
                }
                break;
            }
            case DerivedInstructionTag::STORE: {
                if (instruction->operand_count() != 2u) {
                    effect.valid = false;
                    break;
                }
                auto *variable =
                    static_cast<const StoreInst *>(instruction)->variable();
                if (pointer_support != nullptr &&
                    !pointer_support->contains(variable)) {
                    require_unrelated_operand(1u);
                    break;
                }
                auto resolved = resolve_pointer(
                    variable, environment, cache);
                effect.valid &= resolved.valid;
                if (resolved.related) {
                    effect = access_effect(resolved.view, false, true);
                }
                if (effect.valid) { require_unrelated_operand(1u); }
                break;
            }
            case DerivedInstructionTag::CALL: {
                auto *call = static_cast<const CallInst *>(instruction);
                auto *callee = call->callee();
                if (callee == nullptr || call->argument_count() !=
                                             callee->arguments().count_size()) {
                    effect.valid = false;
                    break;
                }
                RayQueryHandlerPointerEnvironment callee_environment;
                auto formal = callee->arguments().begin();
                for (auto i = 0u; i < call->argument_count(); ++i, ++formal) {
                    auto *argument = call->argument(i);
                    if (pointer_support != nullptr &&
                        !pointer_support->contains(argument)) {
                        continue;
                    }
                    auto resolved =
                        argument == nullptr || !argument->is_lvalue() ?
                            RayQueryHandlerPointerResolveResult{} :
                            resolve_pointer(argument, environment, cache);
                    effect.valid &= resolved.valid;
                    if (!resolved.related) { continue; }
                    if (!(*formal)->is_reference() ||
                        !callee_environment.emplace(*formal,
                                                    std::move(resolved.view))
                             .second) {
                        effect.valid = false;
                        break;
                    }
                }
                if (effect.valid && !callee_environment.empty()) {
                    effect = summarize_callable(
                        callee, std::move(callee_environment),
                        active_functions);
                }
                break;
            }
            default: {
                for (auto i = 0u;
                     effect.valid && i < instruction->operand_count(); ++i) {
                    require_unrelated_operand(i);
                }
                break;
            }
        }
        return effect;
    }

    static bool join_effect(
        RayQueryHandlerScratchBlockState &target,
        const RayQueryHandlerScratchEffect &incoming) noexcept {
        if (!target.reached) {
            target.reached = true;
            target.effect = incoming;
            return true;
        }
        LUISA_DEBUG_ASSERT(target.effect.type == incoming.type,
                           "Type mismatch.");
        auto previous_need = target.effect.need;
        auto previous_define = target.effect.define;
        auto previous_valid = target.effect.valid;
        if (incoming.need) {
            union_mask(target.effect.need, *incoming.need);
        }
        if (target.effect.define) {
            if (incoming.define) {
                *target.effect.define &= *incoming.define;
            } else {
                target.effect.define.reset();
            }
        }
        target.effect.valid &= incoming.valid;
        return target.effect.need != previous_need ||
               target.effect.define != previous_define ||
               target.effect.valid != previous_valid;
    }

    [[nodiscard]] RayQueryHandlerScratchEffect summarize_region(
        BasicBlock *entry,
        const RayQueryHandlerGraph &graph,
        RayQueryHandlerPointerEnvironment environment,
        const RayQueryHandlerPointerSupport *pointer_support,
        const RayQueryHandlerInstructionSchedule *instruction_schedule,
        bool allow_external_exit,
        RayQueryHandlerActiveFunctions &active_functions) const noexcept {
        RayQueryHandlerScratchEffect invalid{_root_type};
        invalid.valid = false;
        auto entry_id = graph.block_id(entry);
        if (!graph.valid() || entry_id >= graph.size()) { return invalid; }
        luisa::vector<RayQueryHandlerScratchBlockState> states;
        states.reserve(graph.size());
        for (auto i = 0u; i < graph.size(); ++i) {
            states.emplace_back(_root_type);
        }
        RayQueryHandlerScratchEffect identity{_root_type};
        states[entry_id].reached = true;
        states[entry_id].effect = identity;
        luisa::vector<size_t> worklist{entry_id};
        luisa::vector<uint8_t> queued(graph.size(), uint8_t{0u});
        queued[entry_id] = 1u;
        RayQueryHandlerScratchBlockState exits{_root_type};
        // Pointer provenance depends only on the immutable environment and
        // value graph, not on the block or current dataflow fact. Reuse one
        // memo table across block revisits and fixed-point iterations.
        RayQueryHandlerPointerMap<
            const Value, RayQueryHandlerPointerResolveResult>
            pointer_cache;
        // A block effect is a pure function of the immutable pointer
        // environment and its ordered instruction effects. Memoizing it does
        // not change the CFG fixed point; it only avoids recomputing the same
        // transfer when a predecessor fact grows. Effects are created lazily
        // so malformed instructions in unreachable blocks remain irrelevant,
        // exactly as in the original reachable-worklist formulation.
        RayQueryHandlerPointerMap<
            BasicBlock,
            luisa::unique_ptr<RayQueryHandlerScratchEffect>>
            block_effects;
        auto block_effect = [&](BasicBlock *block) noexcept
            -> const RayQueryHandlerScratchEffect & {
            if (auto iter = block_effects.find(block);
                iter != block_effects.end()) {
                return *iter->second;
            }
            auto effect = luisa::make_unique<RayQueryHandlerScratchEffect>(
                _root_type);
            auto append_instruction =
                [&](const Instruction *instruction) noexcept {
                    auto next = instruction_effect(
                        instruction, environment, pointer_cache,
                        pointer_support, active_functions);
                    append_handler_scratch_effect(*effect, next);
                };
            if (instruction_schedule != nullptr) {
                if (auto iter = instruction_schedule->find(block);
                    iter != instruction_schedule->end()) {
                    for (auto *instruction : iter->second) {
                        append_instruction(instruction);
                        if (!effect->valid) { break; }
                    }
                }
            } else {
                for (auto *instruction : block->instructions()) {
                    append_instruction(instruction);
                    if (!effect->valid) { break; }
                }
            }
            auto [iter, inserted] =
                block_effects.emplace(block, std::move(effect));
            LUISA_DEBUG_ASSERT(inserted,
                               "Duplicate ray-query block effect.");
            return *iter->second;
        };
        while (!worklist.empty()) {
            if (_run_info != nullptr) {
                ++_run_info->handler_localization_block_evaluation_count;
            }
            auto block_id = worklist.back();
            worklist.pop_back();
            queued[block_id] = 0u;
            auto &graph_block = graph.block(block_id);
            auto *block = graph_block.source;
            auto current = states[block_id].effect;
            append_handler_scratch_effect(current, block_effect(block));
            if (!current.valid) { return current; }
            for (auto successor_id : graph_block.successors) {
                auto &state = states[successor_id];
                if (join_effect(state, current) && !queued[successor_id]) {
                    queued[successor_id] = 1u;
                    worklist.emplace_back(successor_id);
                }
            }
            if (graph_block.external_successor_count != 0u) {
                if (!allow_external_exit) { return invalid; }
                join_effect(exits, current);
            } else if (graph_block.successor_count == 0u) {
                auto *terminator = block->terminator();
                if (terminator != nullptr &&
                    terminator->isa<ReturnInst>()) {
                    join_effect(exits, current);
                } else if (terminator == nullptr ||
                           !terminator->isa<UnreachableInst>()) {
                    return invalid;
                }
            }
        }
        return exits.reached ? exits.effect : invalid;
    }

public:
    explicit RayQueryHandlerScratchAnalyzer(
        const Type *root_type,
        LowerRayQueryToPipelineRunInfo *run_info,
        const RayQueryHandlerGraph *graph) noexcept
        : _root_type{root_type},
          _run_info{run_info},
          _graph{graph} {}

    [[nodiscard]] RayQueryHandlerScratchEffect summarize(
        AllocaInst *alloca, BasicBlock *entry,
        const RayQueryHandlerPointerSupport &pointer_support) const noexcept {
        RayQueryHandlerScratchEffect invalid{_root_type};
        invalid.valid = false;
        if (alloca == nullptr || entry == nullptr ||
            alloca->type() != _root_type ||
            alloca->parent_function() == nullptr ||
            _graph == nullptr || !_graph->valid()) {
            return invalid;
        }
        if (_run_info != nullptr) {
            ++_run_info->handler_localization_analysis_count;
        }
        RayQueryHandlerPointerEnvironment environment;
        environment.emplace(alloca, RayQueryHandlerPointerView{});
        RayQueryHandlerActiveFunctions active_functions;
        active_functions.emplace(alloca->parent_function());

        // Let S be the GEP-closed support of this root. The instruction effect
        // is identity exactly when none of its operands belongs to S (the
        // provenance resolver recognizes no other address-preserving XIR
        // instruction). Therefore the ordered subsequence below is a lossless
        // sparse representation of each block's path effect.
        RayQueryHandlerPointerSet<const Instruction> relevant_instructions;
        RayQueryHandlerPointerSet<const BasicBlock> relevant_blocks;
        luisa::vector<const Instruction *> ordered_instructions;
        for (auto *pointer : pointer_support) {
            for (auto &&use : pointer->use_list()) {
                auto *user = use->user();
                if (user == nullptr || !user->isa<Instruction>()) {
                    continue;
                }
                auto *instruction =
                    static_cast<const Instruction *>(user);
                auto *block = instruction->parent_block();
                if (block != nullptr &&
                    _graph->contains(block) &&
                    relevant_instructions.emplace(instruction).second) {
                    relevant_blocks.emplace(block);
                    ordered_instructions.emplace_back(instruction);
                }
            }
        }
        RayQueryHandlerInstructionSchedule instruction_schedule;
        for (auto *instruction : ordered_instructions) {
            instruction_schedule[instruction->parent_block()]
                .emplace_back(instruction);
        }
        for (auto &[block, ordered] : instruction_schedule) {
            static_cast<void>(block);
            std::sort(
                ordered.begin(), ordered.end(),
                [&](const Instruction *lhs,
                    const Instruction *rhs) noexcept {
                    auto *lhs_location =
                        _graph->instruction_location(lhs);
                    auto *rhs_location =
                        _graph->instruction_location(rhs);
                    LUISA_DEBUG_ASSERT(
                        lhs_location != nullptr && rhs_location != nullptr &&
                            lhs_location->block_id == rhs_location->block_id,
                        "Invalid ray-query handler instruction order.");
                    return lhs_location->ordinal < rhs_location->ordinal;
                });
        }
        if (_run_info != nullptr) {
            _run_info->handler_localization_relevant_instruction_count +=
                ordered_instructions.size();
            for (auto *block : relevant_blocks) {
                _run_info
                    ->handler_localization_avoided_instruction_scan_count +=
                    _graph->block_instruction_count(block);
            }
        }
        if (_run_info != nullptr &&
            _run_info->verify_handler_scratch_graph) {
            RayQueryHandlerInstructionSchedule oracle;
            for (auto *block : relevant_blocks) {
                auto &linear = oracle[block];
                for (auto *instruction : block->instructions()) {
                    if (relevant_instructions.contains(instruction)) {
                        linear.emplace_back(instruction);
                    }
                }
            }
            LUISA_ASSERT(
                oracle.size() == instruction_schedule.size(),
                "Sparse ray-query handler schedule disagrees with the "
                "linear instruction-order oracle.");
            for (auto &[block, linear] : oracle) {
                auto iter = instruction_schedule.find(block);
                LUISA_ASSERT(
                    iter != instruction_schedule.end() &&
                        iter->second == linear,
                    "Sparse ray-query handler schedule disagrees with the "
                    "linear instruction-order oracle.");
            }
        }
        return summarize_region(
            entry, *_graph, std::move(environment),
            &pointer_support, &instruction_schedule, true,
            active_functions);
    }
};

[[nodiscard]] static RayQueryHandlerLocalAllocas
find_handler_local_allocas(
    const RayQueryLoopCaptureList &capture_list,
    BasicBlock *surface_entry,
    const RayQueryHandlerRegion &surface_region,
    BasicBlock *procedural_entry,
    const RayQueryHandlerRegion &procedural_region,
    LowerRayQueryToPipelineRunInfo &info) noexcept {
    RayQueryHandlerLocalAllocas result;
    luisa::vector<AllocaInst *> candidates;
    for (auto *value : capture_list.in_values) {
        if (value != nullptr && value->isa<AllocaInst>()) {
            auto *alloca = static_cast<AllocaInst *>(value);
            if (alloca->is_local()) { candidates.emplace_back(alloca); }
        }
    }
    if (candidates.empty()) { return result; }
    RayQueryHandlerGraph surface_graph{surface_region.blocks};
    RayQueryHandlerGraph procedural_graph{procedural_region.blocks};
    if (info.verify_handler_scratch_graph) {
        LUISA_ASSERT(
            surface_graph.verify(surface_region.blocks) &&
                procedural_graph.verify(procedural_region.blocks),
            "Dense ray-query handler graph disagrees with the source CFG.");
    }
    info.handler_localization_indexed_instruction_count +=
        surface_graph.instruction_count() +
        procedural_graph.instruction_count();
    for (auto *alloca : candidates) {
        RayQueryHandlerRootUseInfo uses;
        RayQueryHandlerPointerSupport visited;
        collect_handler_alloca_root_uses(
            alloca, surface_region, procedural_region, uses, visited);
        if (!uses.valid) { continue; }
        // A root used by both candidate kinds has two independent callback
        // lifetimes. It is safe to duplicate the storage into both outlined
        // functions iff each handler separately kills every incoming field
        // before observing it. Requiring a unique handler would retain false
        // cross-candidate state for ordinary DSL temporaries shared by the
        // two source lambdas.
        if (!uses.used_by_surface && !uses.used_by_procedural) { continue; }
        auto handler_is_invocation_local =
            [&](BasicBlock *entry,
                const RayQueryHandlerGraph &graph) noexcept {
                auto summary =
                    RayQueryHandlerScratchAnalyzer{
                        alloca->type(), &info, &graph}
                        .summarize(alloca, entry, visited);
                return summary.valid && !summary.needs_input();
            };
        auto surface_is_local =
            !uses.used_by_surface ||
            handler_is_invocation_local(surface_entry, surface_graph);
        auto procedural_is_local =
            !uses.used_by_procedural ||
            handler_is_invocation_local(procedural_entry, procedural_graph);
        if (!surface_is_local || !procedural_is_local) { continue; }
        if (uses.used_by_surface) { result.surface.emplace_back(alloca); }
        if (uses.used_by_procedural) {
            result.procedural.emplace_back(alloca);
        }
        result.all.emplace(alloca);
    }
    return result;
}

[[nodiscard]] static size_t count_handler_localized_input_captures(
    RayQueryLoopInst *loop,
    const RayQueryLoopCaptureList &capture_list,
    LowerRayQueryToPipelineRunInfo &info) noexcept {
    if (loop == nullptr || loop->dispatch_block() == nullptr) {
        return 0u;
    }
    auto *terminator = loop->dispatch_block()->terminator();
    if (terminator == nullptr ||
        !terminator->isa<RayQueryDispatchInst>()) {
        return 0u;
    }
    auto *dispatch =
        static_cast<RayQueryDispatchInst *>(terminator);
    RayQueryHandlerRegion surface_region;
    RayQueryHandlerRegion procedural_region;
    luisa::string_view reason;
    if (!collect_outlineable_handler_region(
            dispatch->on_surface_candidate_block(),
            loop->dispatch_block(), loop->merge_block(),
            surface_region, reason) ||
        !collect_outlineable_handler_region(
            dispatch->on_procedural_candidate_block(),
            loop->dispatch_block(), loop->merge_block(),
            procedural_region, reason)) {
        // Selection runs only after the atomic preflight, so this is a
        // defensive fallback rather than a second acceptance path.
        return 0u;
    }
    return find_handler_local_allocas(
               capture_list,
               dispatch->on_surface_candidate_block(), surface_region,
               dispatch->on_procedural_candidate_block(),
               procedural_region,
               info)
        .all.size();
}

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

[[nodiscard]] static luisa::vector<RayQueryLoopInst *>
select_ray_query_loops(
    luisa::span<RayQueryLoopInst *const> loops,
    LowerRayQueryToPipelineOptions options,
    LowerRayQueryToPipelineRunInfo &info) noexcept {
    struct Candidate {
        RayQueryLoopInst *loop;
        size_t handler_block_count;
        size_t handler_instruction_count;
        size_t input_capture_count;
        size_t output_capture_count;
        size_t capture_cost;
        bool capture_filter_eligible;
        bool capture_eligible;
    };
    auto capture_count_within_limit = [](size_t input_count,
                                         size_t output_count,
                                         size_t limit) noexcept {
        return output_count <= limit &&
               input_count <= limit - output_count;
    };
    luisa::vector<Candidate> candidates;
    candidates.reserve(loops.size());
    auto capture_eligible_loop_count = size_t{0u};
    for (auto *loop : loops) {
        auto subgraph = collect_ray_query_loop_subgraph(loop);
        auto capture_list = collect_ray_query_loop_capture_list(subgraph);
        auto handler_block_count = subgraph.reverse_post_order.size() - 1u;
        auto handler_instruction_count = size_t{0u};
        for (auto *block : subgraph.reverse_post_order) {
            if (block == loop->dispatch_block()) { continue; }
            for (auto *instruction : block->instructions()) {
                static_cast<void>(instruction);
                ++handler_instruction_count;
            }
        }
        // Localization affects count selection only when the raw count exceeds
        // its budget. Payload cost deliberately remains a conservative raw
        // bound; lowering performs the complete proof once before ABI changes.
        auto input_capture_count = capture_list.in_values.size();
        auto output_capture_count = capture_list.out_values.size();
        auto capture_cost = detail::ray_query_capture_cost(luisa::span{capture_list.in_values},
                                                           luisa::span{capture_list.out_values}, options);
        auto capture_filter_eligible = true;
        if (options.captured_argument_filter != nullptr) {
            for (auto *value : capture_list.in_values) {
                capture_filter_eligible &= options.captured_argument_filter(value, false);
            }
            for (auto *value : capture_list.out_values) {
                capture_filter_eligible &= options.captured_argument_filter(value, true);
            }
        }
        auto capture_eligible =
            capture_filter_eligible &&
            capture_cost.total() <= options.max_captured_argument_cost &&
            capture_count_within_limit(input_capture_count, output_capture_count,
                                       options.max_captured_argument_count);
        // Output captures cannot be localized. If they alone exceed the
        // budget, no input-localization result can make the loop eligible.
        if (!capture_eligible && capture_filter_eligible &&
            capture_cost.total() <= options.max_captured_argument_cost &&
            output_capture_count <= options.max_captured_argument_count) {
            ++info.selection_localization_analysis_count;
            auto localized_input_capture_count =
                count_handler_localized_input_captures(
                    loop, capture_list, info);
            LUISA_DEBUG_ASSERT(localized_input_capture_count <= input_capture_count,
                               "Localized ray-query handler captures exceed input captures.");
            input_capture_count -= localized_input_capture_count;
            capture_eligible =
                capture_filter_eligible &&
                capture_cost.total() <= options.max_captured_argument_cost &&
                capture_count_within_limit(input_capture_count, output_capture_count,
                                           options.max_captured_argument_count);
        }
        capture_eligible_loop_count += capture_eligible ? 1u : 0u;
        candidates.emplace_back(Candidate{
            .loop = loop,
            .handler_block_count = handler_block_count,
            .handler_instruction_count = handler_instruction_count,
            .input_capture_count = input_capture_count,
            .output_capture_count = output_capture_count,
            .capture_cost = capture_cost.total(),
            .capture_filter_eligible = capture_filter_eligible,
            .capture_eligible = capture_eligible});
    }
    luisa::vector<RayQueryLoopInst *> selected;
    selected.reserve(loops.size());
    for (auto &&candidate : candidates) {
        auto profitability_eligible =
            candidate.handler_instruction_count >=
                options.min_handler_instruction_count ||
            capture_eligible_loop_count >=
                options.min_small_handler_loop_count;
        if (candidate.capture_eligible && profitability_eligible) {
            LUISA_VERBOSE(
                "lower_ray_query_to_pipeline: selecting loop with {} handler block(s), {} handler instruction(s), at most {} input and {} output captured argument(s), costing {} payload byte(s).",
                candidate.handler_block_count,
                candidate.handler_instruction_count,
                candidate.input_capture_count,
                candidate.output_capture_count,
                candidate.capture_cost);
            selected.emplace_back(candidate.loop);
        } else {
            LUISA_VERBOSE(
                "lower_ray_query_to_pipeline: retaining loop with {} handler block(s), {} handler instruction(s), at most {} input and {} output captured argument(s), costing {} payload byte(s) (capture_filter={}, capture_limit={}, capture_cost_limit={}, min_handler_instructions={}, eligible_loop_count={}, small_handler_loop_threshold={}).",
                candidate.handler_block_count,
                candidate.handler_instruction_count,
                candidate.input_capture_count,
                candidate.output_capture_count,
                candidate.capture_cost,
                candidate.capture_filter_eligible,
                options.max_captured_argument_count,
                options.max_captured_argument_cost,
                options.min_handler_instruction_count,
                capture_eligible_loop_count,
                options.min_small_handler_loop_count);
            if (options.skipped_loop_count != nullptr) {
                ++*options.skipped_loop_count;
            }
        }
    }
    return selected;
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
                                                                      luisa::span<AllocaInst *const> local_allocas,
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
    // Both callbacks share the RayQueryPipeline capture ABI. Materialize every
    // output argument once, then write it at each outlined return. Creating
    // output arguments while visiting returns accidentally encoded a
    // single-exit restriction into the function signature.
    luisa::vector<Value *> out_args;
    out_args.reserve(capture_list.out_values.size());
    for (auto out_value : capture_list.out_values) {
        out_args.emplace_back(
            function->create_reference_argument(out_value->type()));
    }
    // create blocks for the function
    for (auto block : subgraph.reverse_post_order) {
        auto local_block = function->create_basic_block();
        resolver.emplace(block, local_block);
    }
    // set function body
    function->set_body_block(static_cast<BasicBlock *>(resolver.resolve(branch)));
    // Recreate proven invocation-local storage inside the outlined handler.
    // The resolver then rewrites every accepted direct load/store to this new
    // object; no callback ABI field or parent-function lifetime is required.
    XIRBuilder local_builder;
    local_builder.set_insertion_point(function->definition()->body_block());
    for (auto *original : local_allocas) {
        auto *local = local_builder.alloca_(original->type(), original->op());
        clone_metadata(*original, *local);
        LUISA_ASSERT(resolver.emplace(original, local),
                     "Duplicate localized ray-query handler alloca.");
    }
    // duplicate the blocks
    luisa::vector<BasicBlock *> return_blocks;
    luisa::vector<std::pair<const PhiInst *, PhiInst *>> phi_nodes;
    for (auto block : subgraph.reverse_post_order) {
        if (auto bb = duplicate_basic_block_for_ray_query_loop_dispatch_branch(block, dispatch, phi_nodes, resolver);
            bb->terminator()->isa<ReturnInst>()) {
            return_blocks.emplace_back(bb);
        }
    }
    LUISA_DEBUG_ASSERT(!return_blocks.empty(),
                       "Outlined ray-query handler has no return block.");
    // Generate output stores only after every block has been cloned, so an
    // output resolver entry never depends on reverse-post-order visitation.
    for (auto *return_block : return_blocks) {
        XIRBuilder b;
        b.set_insertion_point(return_block->terminator()->prev());
        for (auto i = 0u; i < capture_list.out_values.size(); i++) {
            if (auto resolved = resolver.resolve_or_null(
                    capture_list.out_values[i])) {
                b.store(out_args[i], resolved);
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

static void lower_ray_query_to_pipeline(
    Function *function, RayQueryLoopInst *loop,
    LowerRayQueryToPipelineRunInfo &info) noexcept {
    auto subgraph = collect_ray_query_loop_subgraph(loop);
    auto capture_list = collect_ray_query_loop_capture_list(subgraph);
    auto dispatch = static_cast<RayQueryDispatchInst *>(subgraph.reverse_post_order.front()->terminator());
    RayQueryHandlerRegion surface_region;
    RayQueryHandlerRegion procedural_region;
    luisa::string_view region_reason;
    LUISA_ASSERT(
        collect_outlineable_handler_region(
            dispatch->on_surface_candidate_block(),
            subgraph.reverse_post_order.front(), loop->merge_block(),
            surface_region, region_reason) &&
            collect_outlineable_handler_region(
                dispatch->on_procedural_candidate_block(),
                subgraph.reverse_post_order.front(), loop->merge_block(),
                procedural_region, region_reason),
        "Preflighted ray-query handler region became invalid: {}.",
        region_reason);
    auto local_allocas = find_handler_local_allocas(
        capture_list, dispatch->on_surface_candidate_block(), surface_region,
        dispatch->on_procedural_candidate_block(), procedural_region,
        info);
    if (!local_allocas.all.empty()) {
        capture_list.in_values.erase(
            std::remove_if(
                capture_list.in_values.begin(),
                capture_list.in_values.end(),
                [&](Value *value) noexcept {
                    return local_allocas.all.contains(value);
                }),
            capture_list.in_values.end());
        info.localized_alloca_count += local_allocas.all.size();
    }
    auto merge_block = loop->control_flow_merge()->merge_block();
    LUISA_DEBUG_ASSERT(dispatch->exit_block() == merge_block, "Invalid ray query loop exit block.");
    LUISA_DEBUG_ASSERT(function->parent_module() != nullptr, "Invalid function module.");
    auto on_surface = outline_ray_query_loop_dispatch_branch(
        function->parent_module(), dispatch->on_surface_candidate_block(), subgraph.query_object,
        subgraph.reverse_post_order.front(), capture_list,
        luisa::span{local_allocas.surface},
        "on_surface function outlined from ray query loop");
    auto on_procedural = outline_ray_query_loop_dispatch_branch(
        function->parent_module(), dispatch->on_procedural_candidate_block(), subgraph.query_object,
        subgraph.reverse_post_order.front(), capture_list,
        luisa::span{local_allocas.procedural},
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

static void lower_ray_query_to_pipeline_lower_preflighted_ray_query_loops(
    Function *function, luisa::span<RayQueryLoopInst *const> loops,
    LowerRayQueryToPipelineRunInfo &info) noexcept {
    auto *def = function == nullptr ? nullptr : function->definition();
    if (def == nullptr) { return; }
    auto lowered_before = info.lowered_loop_count;
    for (auto *loop : loops) {
        lower_phi_nodes_in_loop_dispatch_block(def, loop);
        hoist_alloca_instructions_to_entry_block(def);
        lower_ray_query_to_pipeline(function, loop, info);
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

static void run_lower_ray_query_to_pipeline_pass_on_function(
    Function *function, LowerRayQueryToPipelineOptions options,
    LowerRayQueryToPipelineRunInfo &info) noexcept {
    auto loops = collect_ray_query_loops(function);
    // Preflight the complete function before touching dispatch PHIs, hoisting
    // allocas, creating callbacks, or running function-wide DCE.
    if (!lower_ray_query_to_pipeline_preflight_ray_query_loops(luisa::span{loops}, info)) { return; }
    loops = select_ray_query_loops(
        luisa::span{loops}, options, info);
    lower_ray_query_to_pipeline_lower_preflighted_ray_query_loops(
        function, luisa::span{loops}, info);
}

}// namespace detail

LowerRayQueryToPipelineInfo
lower_ray_query_to_pipeline_pass_run_on_function(
    Function *function, LowerRayQueryToPipelineOptions options) noexcept {
    if (options.skipped_loop_count != nullptr) {
        *options.skipped_loop_count = 0u;
    }
    if (options.localized_alloca_count != nullptr) {
        *options.localized_alloca_count = 0u;
    }
    detail::LowerRayQueryToPipelineRunInfo info;
    info.verify_handler_scratch_graph =
        options.verify_handler_scratch_graph;
    detail::run_lower_ray_query_to_pipeline_pass_on_function(
        function, options, info);
    if (options.localized_alloca_count != nullptr) {
        *options.localized_alloca_count = info.localized_alloca_count;
    }
    return info;
}

LowerRayQueryToPipelineInfo
lower_ray_query_to_pipeline_pass_run_on_function(
    Function *function) noexcept {
    return lower_ray_query_to_pipeline_pass_run_on_function(
        function, {});
}

LowerRayQueryToPipelineInfo
lower_ray_query_to_pipeline_pass_run_on_module(
    Module *module, PassReport *report,
    LowerRayQueryToPipelineOptions options) noexcept {
    if (options.skipped_loop_count != nullptr) {
        *options.skipped_loop_count = 0u;
    }
    if (options.localized_alloca_count != nullptr) {
        *options.localized_alloca_count = 0u;
    }
    detail::LowerRayQueryToPipelineRunInfo info;
    info.verify_handler_scratch_graph =
        options.verify_handler_scratch_graph;
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
            accepted &= detail::lower_ray_query_to_pipeline_preflight_ray_query_loops(
                luisa::span{item.loops}, info);
        }
        if (accepted) {
            for (auto &item : work) {
                item.loops = detail::select_ray_query_loops(
                    luisa::span{item.loops}, options, info);
            }
            for (auto &item : work) {
                detail::lower_ray_query_to_pipeline_lower_preflighted_ray_query_loops(
                    item.function, luisa::span{item.loops}, info);
            }
        }
    }
    if (report != nullptr) {
        report->set(
            "lowered_ray_query_to_pipeline", info.lowered_loop_count);
        report->set("error", info.error_count);
        report->set("selection_localization_analysis",
                    info.selection_localization_analysis_count);
        report->set(
            "handler_localization_instruction_evaluation",
            info.handler_localization_instruction_evaluation_count);
        report->set("handler_localization_analysis",
                    info.handler_localization_analysis_count);
        report->set("handler_localization_indexed_instruction",
                    info.handler_localization_indexed_instruction_count);
        report->set("handler_localization_relevant_instruction",
                    info.handler_localization_relevant_instruction_count);
        report->set("handler_localization_avoided_instruction_scan",
                    info.handler_localization_avoided_instruction_scan_count);
        report->set("handler_localization_block_evaluation",
                    info.handler_localization_block_evaluation_count);
    }
    if (options.localized_alloca_count != nullptr) {
        *options.localized_alloca_count = info.localized_alloca_count;
    }
    return info;
}

LowerRayQueryToPipelineInfo
lower_ray_query_to_pipeline_pass_run_on_module(
    Module *module, PassReport *report) noexcept {
    return lower_ray_query_to_pipeline_pass_run_on_module(
        module, report, {});
}

RayQueryLoopLowerInfo
lower_ray_query_loop_pass_run_on_function(Function *function) noexcept {
    size_t localized_alloca_count = 0u;
    auto info = lower_ray_query_to_pipeline_pass_run_on_function(
        function,
        {.localized_alloca_count = &localized_alloca_count});
    return {
        .lowered_loop_count = info.lowered_loop_count,
        .localized_alloca_count = localized_alloca_count,
        .error_count = info.error_count};
}

RayQueryLoopLowerInfo
lower_ray_query_loop_pass_run_on_module(
    Module *module, PassReport *report) noexcept {
    size_t localized_alloca_count = 0u;
    auto info = lower_ray_query_to_pipeline_pass_run_on_module(
        module, nullptr,
        {.localized_alloca_count = &localized_alloca_count});
    if (report != nullptr) {
        report->set("lowered_loop", info.lowered_loop_count);
        report->set("localized_alloca", localized_alloca_count);
        report->set("error", info.error_count);
    }
    return {
        .lowered_loop_count = info.lowered_loop_count,
        .localized_alloca_count = localized_alloca_count,
        .error_count = info.error_count};
}

}// namespace luisa::compute::xir
