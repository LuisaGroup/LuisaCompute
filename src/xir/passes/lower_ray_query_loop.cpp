#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

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
    luisa::unordered_map<const Value *, Value *> value_map;

public:
    bool emplace(const Value *original, Value *duplicate) noexcept {
        return value_map.emplace(original, duplicate).second;
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
        auto iter = value_map.find(value);
        return iter == value_map.end() ? nullptr : iter->second;
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
    XIRBuilder b;
    b.set_insertion_point(bb);
    for (auto inst : original->instructions()) {
        // special case: branch to the merge block
        if (inst->is_terminator() && inst->isa<BranchInst>() &&
            static_cast<const BranchInst *>(inst)->target_block() == merge) {
            b.return_void();
        } else if (inst->isa<PhiInst>()) {
            auto dup_phi = b.phi(inst->type());
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
    (void)b.ray_query_pipeline(subgraph.query_object, on_surface, on_procedural, captured_args);
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
                phi->replace_all_uses_with(phi_load);
            }
            LUISA_DEBUG_ASSERT(phi->use_list().empty(), "Phi node has uses but no exit block.");
            phi->remove_self();
        }
    }
}

static void run_lower_ray_query_loop_pass_on_function(Function *function, RayQueryLoopLowerInfo &info) noexcept {
    if (auto def = function->definition()) {
        // discover all ray query loops
        luisa::vector<RayQueryLoopInst *> loops;
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<RayQueryLoopInst>()) {
                loops.emplace_back(static_cast<RayQueryLoopInst *>(inst));
            }
        });
        // lower each ray query loop
        for (auto loop : loops) {
            lower_phi_nodes_in_loop_dispatch_block(def, loop);
            hoist_alloca_instructions_to_entry_block(def);
            lower_ray_query_loop(function, loop, info);
        }
        // remove dead code after lowering using the DCE pass
        if (!loops.empty()) {
            auto dce_info = dce_pass_run_on_function(function);
            LUISA_VERBOSE("Removed {} dead instruction(s) and {} dead block(s) after lowering ray query loop(s).",
                          dce_info.removed_inst_count, dce_info.removed_block_count);
        }
    }
}

}// namespace detail

RayQueryLoopLowerInfo lower_ray_query_loop_pass_run_on_function(Function *function) noexcept {
    RayQueryLoopLowerInfo info;
    detail::run_lower_ray_query_loop_pass_on_function(function, info);
    return info;
}

RayQueryLoopLowerInfo lower_ray_query_loop_pass_run_on_module(Module *module) noexcept {
    RayQueryLoopLowerInfo info;
    for (auto f : module->function_list()) {
        detail::run_lower_ray_query_loop_pass_on_function(f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
