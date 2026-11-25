#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/reconstruct_ray_query_loop.h>
#include <luisa/core/stl/algorithm.h>
#include <luisa/xir/undefined.h>

#include <algorithm>
#include <array>

namespace luisa::compute::xir {

namespace detail {

static void reconstruct_clone_metadata(const MetadataListMixin &source,
                                       MetadataListMixin &target) noexcept {
    for (auto *metadata : source.metadata_list()) {
        target.metadata_list().push_front(metadata->clone());
    }
}

[[nodiscard]] static bool reconstruct_is_ray_query_object(
    const Value *value) noexcept {
    if (value == nullptr || !value->is_lvalue()) { return false; }
    auto *type = value->type();
    return type == Type::custom("LC_RayQueryAll") ||
           type == Type::custom("LC_RayQueryAny");
}

[[nodiscard]] static luisa::vector<Instruction *>
collect_instructions(BasicBlock *block) noexcept {
    luisa::vector<Instruction *> instructions;
    if (block != nullptr) {
        for (auto *inst : block->instructions()) {
            instructions.emplace_back(inst);
        }
    }
    return instructions;
}

[[nodiscard]] static bool is_ray_query_write(
    Instruction *inst, RayQueryObjectWriteOp op,
    Value *query = nullptr) noexcept {
    if (inst == nullptr || !inst->isa<RayQueryObjectWriteInst>()) {
        return false;
    }
    auto *write = static_cast<RayQueryObjectWriteInst *>(inst);
    return write->op() == op && write->operand_count() == 1u &&
           (query == nullptr || write->operand(0u) == query);
}

[[nodiscard]] static bool is_ray_query_read(
    Instruction *inst, RayQueryObjectReadOp op,
    Value *query) noexcept {
    if (inst == nullptr || !inst->isa<RayQueryObjectReadInst>()) {
        return false;
    }
    auto *read = static_cast<RayQueryObjectReadInst *>(inst);
    return read->op() == op && read->operand_count() == 1u &&
           read->operand(0u) == query && read->type() == Type::of<bool>();
}

[[nodiscard]] static bool has_exactly_one_use_by(
    Value *value, Instruction *expected_user) noexcept {
    auto use_count = 0u;
    for (auto *use : value->use_list()) {
        if (use->user() != expected_user) { return false; }
        ++use_count;
    }
    return use_count == 1u;
}

[[nodiscard]] static bool has_exactly_two_uses_by(
    Value *value, Instruction *first,
    Instruction *second) noexcept {
    auto first_count = 0u;
    auto second_count = 0u;
    for (auto *use : value->use_list()) {
        if (use->user() == first) {
            ++first_count;
        } else if (use->user() == second) {
            ++second_count;
        } else {
            return false;
        }
    }
    return first_count == 1u && second_count == 1u;
}

struct ReconstructHandlerRegion {
    luisa::unordered_set<BasicBlock *> blocks;
    luisa::vector<BranchInst *> exits;
};

struct GeneralInlineSpecialization {
    bool collapse_empty{false};
    luisa::vector<BasicBlock *> reachable_blocks;
    luisa::unordered_set<BasicBlock *> reachable_set;
    luisa::unordered_map<const Instruction *, bool> folded_bools;
    bool has_surface_ops{false};
    bool has_procedural_ops{false};
};

struct ReconstructCandidate {
    LoopInst *loop{nullptr};
    SimpleLoopInst *inline_loop{nullptr};
    BasicBlock *parent{nullptr};
    BasicBlock *prepare{nullptr};
    BasicBlock *body{nullptr};
    BasicBlock *update{nullptr};
    BasicBlock *merge{nullptr};
    Value *query{nullptr};
    Instruction *candidate_dispatch{nullptr};
    BasicBlock *surface_entry{nullptr};
    BasicBlock *procedural_entry{nullptr};
    bool surface_empty{false};
    bool procedural_empty{false};
    ReconstructHandlerRegion surface_region;
    ReconstructHandlerRegion procedural_region;
    BasicBlock *inline_break{nullptr};
    BasicBlock *inline_relay{nullptr};
    bool is_general_inline{false};
    luisa::vector<PhiInst *> loop_body_phis;
    luisa::unordered_set<BasicBlock *> general_body_blocks;
    GeneralInlineSpecialization surface_spec;
    GeneralInlineSpecialization procedural_spec;
};

enum class ReconstructMatch {
    ignored,
    accepted,
    rejected,
};

[[nodiscard]] static bool block_contains_ray_query_proceed(
    BasicBlock *block) noexcept {
    if (block == nullptr) { return false; }
    for (auto *inst : block->instructions()) {
        if (is_ray_query_write(
                inst,
                RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] static bool collect_handler_region(
    BasicBlock *entry, BasicBlock *body, BasicBlock *update,
    ReconstructHandlerRegion &region,
    luisa::string_view &reason) noexcept {
    if (entry == update) { return true; }
    if (entry == nullptr || body == nullptr || update == nullptr) {
        reason = "candidate handler has a null entry/body/update block";
        return false;
    }
    auto *owner = body->parent_function();
    luisa::vector<BasicBlock *> worklist{entry};
    while (!worklist.empty()) {
        auto *block = worklist.back();
        worklist.pop_back();
        if (block == update) { continue; }
        if (block == nullptr || block == body ||
            block->parent_function() != owner) {
            reason = "candidate handler escapes its canonical loop";
            return false;
        }
        if (!region.blocks.emplace(block).second) { continue; }
        if (!block->is_terminated()) {
            reason = "candidate handler contains an unterminated block";
            return false;
        }
        auto *terminator = block->terminator();
        auto valid = true;
        block->traverse_successors(
            false, [&](BasicBlock *successor) noexcept {
                if (successor == update) {
                    if (terminator->isa<BranchInst>() &&
                        static_cast<BranchInst *>(terminator)
                                ->target_block() == update) {
                        region.exits.emplace_back(
                            static_cast<BranchInst *>(terminator));
                    } else {
                        valid = false;
                    }
                } else if (!region.blocks.contains(successor)) {
                    worklist.emplace_back(successor);
                }
            });
        if (!valid) {
            reason = "candidate handler has a non-Branch edge to the loop update";
            return false;
        }
    }
    if (region.exits.empty()) {
        reason = "candidate handler has no exit to the loop update";
        return false;
    }
    for (auto *block : region.blocks) {
        if (auto *merge = block->terminator()->control_flow_merge();
            merge != nullptr) {
            auto *merge_block = merge->merge_block();
            if (merge_block != nullptr &&
                !region.blocks.contains(merge_block)) {
                reason = "candidate handler has a structured merge outside its region";
                return false;
            }
        }
        auto has_external_predecessor = false;
        block->traverse_predecessors(
            false, [&](BasicBlock *predecessor) noexcept {
                has_external_predecessor |=
                    !region.blocks.contains(predecessor) &&
                    !(block == entry && predecessor == body);
            });
        if (has_external_predecessor) {
            reason = "candidate handler has an external predecessor";
            return false;
        }
    }
    return true;
}

[[nodiscard]] static bool handler_regions_overlap(
    const ReconstructHandlerRegion &lhs,
    const ReconstructHandlerRegion &rhs) noexcept {
    auto *smaller = &lhs.blocks;
    auto *larger = &rhs.blocks;
    if (smaller->size() > larger->size()) {
        luisa::swap(smaller, larger);
    }
    for (auto *block : *smaller) {
        if (larger->contains(block)) { return true; }
    }
    return false;
}

[[nodiscard]] static bool predecessors_are_subset_of(
    BasicBlock *block,
    luisa::span<BasicBlock *const> allowed) noexcept {
    auto valid = true;
    block->traverse_predecessors(
        false, [&](BasicBlock *predecessor) noexcept {
            valid &= std::find(
                         allowed.begin(), allowed.end(), predecessor) !=
                     allowed.end();
        });
    return valid;
}

[[nodiscard]] static ReconstructMatch match_canonical_ray_query_loop(
    LoopInst *loop, ReconstructCandidate &candidate,
    luisa::string_view &reason) noexcept {
    if (loop == nullptr) { return ReconstructMatch::ignored; }
    auto *prepare = loop->prepare_block();
    auto *body = loop->body_block();
    auto *update = loop->update_block();
    if (!block_contains_ray_query_proceed(prepare) &&
        !block_contains_ray_query_proceed(body) &&
        !block_contains_ray_query_proceed(update)) {
        return ReconstructMatch::ignored;
    }
    auto reject = [&](luisa::string_view message) noexcept {
        reason = message;
        return ReconstructMatch::rejected;
    };
    auto *parent = loop->parent_block();
    auto *merge = loop->merge_block();
    if (parent == nullptr || prepare == nullptr || body == nullptr ||
        update == nullptr || merge == nullptr) {
        return reject("canonical ray-query loop has a null structural block");
    }
    auto *function = parent->parent_function();
    if (function == nullptr || prepare->parent_function() != function ||
        body->parent_function() != function ||
        update->parent_function() != function ||
        merge->parent_function() != function ||
        parent == prepare || parent == body || parent == update ||
        parent == merge || prepare == body || prepare == update ||
        prepare == merge || body == update || body == merge ||
        update == merge) {
        return reject("canonical ray-query loop has invalid block ownership");
    }
    if (!block_contains_ray_query_proceed(prepare)) {
        return reject("ray-query PROCEED is outside the canonical prepare block");
    }

    auto prepare_instructions = collect_instructions(prepare);
    if (prepare_instructions.size() != 4u ||
        !is_ray_query_write(
            prepare_instructions[0u],
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED)) {
        return reject("ray-query prepare block is not canonical");
    }
    auto *query = prepare_instructions[0u]->operand(0u);
    if (!reconstruct_is_ray_query_object(query) ||
        !is_ray_query_read(
            prepare_instructions[1u],
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
            query) ||
        !prepare_instructions[2u]->isa<ArithmeticInst>()) {
        return reject("ray-query prepare operations do not share one valid query object");
    }
    auto *not_terminated =
        static_cast<ArithmeticInst *>(prepare_instructions[2u]);
    if (not_terminated->op() != ArithmeticOp::UNARY_BIT_NOT ||
        not_terminated->operand_count() != 1u ||
        not_terminated->operand(0u) != prepare_instructions[1u] ||
        not_terminated->type() != Type::of<bool>() ||
        !prepare_instructions[3u]->isa<ConditionalBranchInst>()) {
        return reject("ray-query termination test is not canonical");
    }
    auto *prepare_branch =
        static_cast<ConditionalBranchInst *>(prepare_instructions[3u]);
    if (prepare_branch->condition() != not_terminated ||
        prepare_branch->true_block() != body ||
        prepare_branch->false_block() != merge ||
        !has_exactly_one_use_by(
            prepare_instructions[1u], not_terminated) ||
        !has_exactly_one_use_by(not_terminated, prepare_branch)) {
        return reject("ray-query prepare branch is not active-body/terminated-merge");
    }

    auto update_instructions = collect_instructions(update);
    if (update_instructions.size() != 1u ||
        !update_instructions.front()->isa<BranchInst>() ||
        static_cast<BranchInst *>(update_instructions.front())
                ->target_block() != prepare) {
        return reject("ray-query update block is not a canonical latch");
    }

    auto body_instructions = collect_instructions(body);
    BasicBlock *surface_entry = update;
    BasicBlock *procedural_entry = update;
    Instruction *candidate_dispatch = nullptr;
    if (body_instructions.size() == 1u &&
        body_instructions.front()->isa<BranchInst>() &&
        static_cast<BranchInst *>(body_instructions.front())
                ->target_block() == update) {
        candidate_dispatch = body_instructions.front();
    } else if (body_instructions.size() == 2u &&
               is_ray_query_read(
                   body_instructions[0u],
                   RayQueryObjectReadOp::
                       RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE,
                   query) &&
               body_instructions[1u]->isa<IfInst>()) {
        auto *candidate_if =
            static_cast<IfInst *>(body_instructions[1u]);
        if (candidate_if->condition() != body_instructions[0u] ||
            candidate_if->merge_block() != update ||
            candidate_if->true_block() == nullptr ||
            candidate_if->false_block() == nullptr ||
            !has_exactly_one_use_by(
                body_instructions[0u], candidate_if)) {
            return reject("ray-query candidate dispatch is not canonical");
        }
        surface_entry = candidate_if->true_block();
        procedural_entry = candidate_if->false_block();
        candidate_dispatch = candidate_if;
    } else {
        return reject("ray-query body is not a canonical candidate dispatch");
    }

    ReconstructHandlerRegion surface_region;
    ReconstructHandlerRegion procedural_region;
    if (!collect_handler_region(
            surface_entry, body, update, surface_region, reason) ||
        !collect_handler_region(
            procedural_entry, body, update, procedural_region, reason)) {
        return ReconstructMatch::rejected;
    }
    if (handler_regions_overlap(surface_region, procedural_region)) {
        return reject("surface and procedural handler regions overlap");
    }

    std::array prepare_predecessors{parent, update};
    std::array body_predecessors{prepare};
    if (!predecessors_are_subset_of(
            prepare, luisa::span{prepare_predecessors}) ||
        !predecessors_are_subset_of(
            body, luisa::span{body_predecessors})) {
        return reject("canonical ray-query shell has an external predecessor");
    }
    luisa::vector<BasicBlock *> update_predecessors{body};
    for (auto *exit : surface_region.exits) {
        update_predecessors.emplace_back(exit->parent_block());
    }
    for (auto *exit : procedural_region.exits) {
        update_predecessors.emplace_back(exit->parent_block());
    }
    if (!predecessors_are_subset_of(
            update, luisa::span{update_predecessors})) {
        return reject("canonical ray-query update has an external predecessor");
    }
    std::array merge_predecessors{parent, prepare};
    if (!predecessors_are_subset_of(
            merge, luisa::span{merge_predecessors})) {
        return reject("canonical ray-query merge has an external predecessor");
    }

    candidate = ReconstructCandidate{
        .loop = loop,
        .parent = parent,
        .prepare = prepare,
        .body = body,
        .update = update,
        .merge = merge,
        .query = query,
        .candidate_dispatch = candidate_dispatch,
        .surface_entry = surface_entry,
        .procedural_entry = procedural_entry,
        .surface_empty = surface_entry == update,
        .procedural_empty = procedural_entry == update,
        .surface_region = std::move(surface_region),
        .procedural_region = std::move(procedural_region)};
    return ReconstructMatch::accepted;
}

[[nodiscard]] static bool function_has_other_proceed(
    FunctionDefinition *definition, Value *query,
    const luisa::unordered_set<BasicBlock *> &excluded_blocks) noexcept {
    if (definition == nullptr) { return false; }
    for (auto *block : definition->basic_blocks()) {
        if (excluded_blocks.contains(block)) { continue; }
        for (auto *inst : block->instructions()) {
            if (is_ray_query_write(
                    inst, RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED, query)) {
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] static bool alloca_has_single_store_in_candidate_entry(
    const Value *variable, BasicBlock *candidate_entry,
    const luisa::unordered_set<BasicBlock *> &body_region,
    const StoreInst *&single_store) noexcept {
    if (variable == nullptr || !variable->isa<AllocaInst>()) { return false; }
    const StoreInst *found_store = nullptr;
    for (auto *use : variable->use_list()) {
        auto *user = use->user();
        if (user == nullptr || !user->isa<Instruction>()) { return false; }
        auto *inst = static_cast<const Instruction *>(user);
        if (inst->isa<StoreInst>()) {
            auto *store = static_cast<const StoreInst *>(inst);
            if (store->variable() != variable) { return false; }
            if (found_store != nullptr) { return false; }
            found_store = store;
        } else if (inst->isa<LoadInst>()) {
            auto *load = static_cast<const LoadInst *>(inst);
            if (load->variable() != variable ||
                !body_region.contains(const_cast<BasicBlock *>(load->parent_block()))) {
                return false;
            }
        } else {
            return false;
        }
    }
    if (found_store == nullptr ||
        found_store->parent_block() != candidate_entry) {
        return false;
    }
    auto seen_store = false;
    for (auto *inst : candidate_entry->instructions()) {
        if (inst == found_store) {
            seen_store = true;
            continue;
        }
        if (inst->isa<LoadInst>() &&
            static_cast<const LoadInst *>(inst)->variable() == variable &&
            !seen_store) {
            return false;
        }
    }
    single_store = found_store;
    return true;
}

[[nodiscard]] static luisa::optional<bool> eval_specialized_bool(
    const Value *value, Value *query, bool is_surface,
    BasicBlock *candidate_entry,
    const luisa::unordered_set<BasicBlock *> &body_region,
    luisa::unordered_map<const Instruction *, bool> &memo) noexcept {
    if (value == nullptr || value->type() != Type::of<bool>()) {
        return luisa::nullopt;
    }
    if (value->isa<Constant>()) {
        return static_cast<const Constant *>(value)->as<bool>();
    }
    if (!value->isa<Instruction>()) { return luisa::nullopt; }
    auto *inst = static_cast<const Instruction *>(value);
    if (auto iter = memo.find(inst); iter != memo.end()) {
        return iter->second;
    }
    luisa::optional<bool> result;
    if (inst->isa<RayQueryObjectReadInst>()) {
        auto *read = static_cast<const RayQueryObjectReadInst *>(inst);
        if (read->operand_count() == 1u && read->operand(0u) == query) {
            if (read->op() == RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE) {
                result = is_surface;
            } else if (read->op() == RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE) {
                result = !is_surface;
            }
        }
    } else if (inst->isa<LoadInst>()) {
        auto *load = static_cast<const LoadInst *>(inst);
        const StoreInst *single_store = nullptr;
        if (alloca_has_single_store_in_candidate_entry(
                load->variable(), candidate_entry, body_region, single_store)) {
            result = eval_specialized_bool(
                single_store->value(), query, is_surface,
                candidate_entry, body_region, memo);
        }
    } else if (inst->isa<ArithmeticInst>()) {
        auto *arith = static_cast<const ArithmeticInst *>(inst);
        switch (arith->op()) {
            case ArithmeticOp::UNARY_BIT_NOT: {
                if (auto op = eval_specialized_bool(
                        arith->operand(0u), query, is_surface,
                        candidate_entry, body_region, memo)) {
                    result = !*op;
                }
                break;
            }
            case ArithmeticOp::BINARY_BIT_AND: {
                auto lhs = eval_specialized_bool(
                    arith->operand(0u), query, is_surface,
                    candidate_entry, body_region, memo);
                auto rhs = eval_specialized_bool(
                    arith->operand(1u), query, is_surface,
                    candidate_entry, body_region, memo);
                if (lhs && !*lhs) {
                    result = false;
                } else if (rhs && !*rhs) {
                    result = false;
                } else if (lhs && rhs) {
                    result = *lhs && *rhs;
                }
                break;
            }
            case ArithmeticOp::BINARY_BIT_OR: {
                auto lhs = eval_specialized_bool(
                    arith->operand(0u), query, is_surface,
                    candidate_entry, body_region, memo);
                auto rhs = eval_specialized_bool(
                    arith->operand(1u), query, is_surface,
                    candidate_entry, body_region, memo);
                if (lhs && *lhs) {
                    result = true;
                } else if (rhs && *rhs) {
                    result = true;
                } else if (lhs && rhs) {
                    result = *lhs || *rhs;
                }
                break;
            }
            case ArithmeticOp::BINARY_BIT_XOR:
            case ArithmeticOp::BINARY_NOT_EQUAL: {
                auto lhs = eval_specialized_bool(
                    arith->operand(0u), query, is_surface,
                    candidate_entry, body_region, memo);
                auto rhs = eval_specialized_bool(
                    arith->operand(1u), query, is_surface,
                    candidate_entry, body_region, memo);
                if (lhs && rhs) { result = (*lhs != *rhs); }
                break;
            }
            case ArithmeticOp::BINARY_EQUAL: {
                auto lhs = eval_specialized_bool(
                    arith->operand(0u), query, is_surface,
                    candidate_entry, body_region, memo);
                auto rhs = eval_specialized_bool(
                    arith->operand(1u), query, is_surface,
                    candidate_entry, body_region, memo);
                if (lhs && rhs) { result = (*lhs == *rhs); }
                break;
            }
            default: break;
        }
    }
    if (result) { memo.emplace(inst, *result); }
    return result;
}

[[nodiscard]] static bool analyze_general_inline_specialization(
    BasicBlock *candidate_entry, BasicBlock *loop_body, BasicBlock *merge,
    const luisa::unordered_set<BasicBlock *> &body_region,
    Value *query, bool is_surface,
    GeneralInlineSpecialization &spec) noexcept {
    spec = {};
    luisa::unordered_map<const Instruction *, bool> bool_memo;
    luisa::vector<BasicBlock *> post_order;
    auto dfs = [&](auto &&self, BasicBlock *block) noexcept -> void {
        if (block == nullptr || block == loop_body || block == merge ||
            !body_region.contains(block) ||
            !spec.reachable_set.emplace(block).second) {
            return;
        }
        auto *terminator = block->terminator();
        if (terminator->isa<IfInst>() || terminator->isa<ConditionalBranchInst>()) {
            auto *cond_term = static_cast<ConditionalBranchTerminatorInstruction *>(terminator);
            if (auto folded = eval_specialized_bool(
                    cond_term->condition(), query, is_surface,
                    candidate_entry, body_region, bool_memo)) {
                spec.folded_bools.emplace(terminator, *folded);
                auto *taken = *folded ? cond_term->true_block() : cond_term->false_block();
                self(self, taken);
                post_order.emplace_back(block);
                return;
            }
        }
        block->traverse_successors(false, [&](BasicBlock *successor) noexcept {
            self(self, successor);
        });
        if (auto *cf_merge = terminator->control_flow_merge();
            cf_merge != nullptr && cf_merge->merge_block() != nullptr) {
            self(self, cf_merge->merge_block());
        }
        post_order.emplace_back(block);
    };
    dfs(dfs, candidate_entry);
    std::reverse(post_order.begin(), post_order.end());
    spec.reachable_blocks = std::move(post_order);

    for (auto *block : spec.reachable_blocks) {
        for (auto *inst : block->instructions()) {
            if (inst->isa<RayQueryObjectReadInst>()) {
                auto *read = static_cast<const RayQueryObjectReadInst *>(inst);
                if (read->operand_count() == 1u && read->operand(0u) == query) {
                    if (read->op() == RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT) {
                        spec.has_surface_ops = true;
                    } else if (read->op() == RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT) {
                        spec.has_procedural_ops = true;
                    }
                }
            } else if (inst->isa<RayQueryObjectWriteInst>()) {
                auto *write = static_cast<const RayQueryObjectWriteInst *>(inst);
                if (write->operand_count() >= 1u && write->operand(0u) == query) {
                    if (write->op() == RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE) {
                        spec.has_surface_ops = true;
                    } else if (write->op() == RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL) {
                        spec.has_procedural_ops = true;
                    }
                }
            }
        }
    }
    return !(spec.has_surface_ops && spec.has_procedural_ops);
}

[[nodiscard]] static ReconstructMatch
match_frontend_inline_ray_query_loop(
    SimpleLoopInst *loop, ReconstructCandidate &candidate,
    luisa::string_view &reason) noexcept {
    if (loop == nullptr) { return ReconstructMatch::ignored; }
    auto *loop_body = loop->body_block();
    if (!block_contains_ray_query_proceed(loop_body)) {
        return ReconstructMatch::ignored;
    }
    auto reject = [&](luisa::string_view message) noexcept {
        reason = message;
        return ReconstructMatch::rejected;
    };
    auto *parent = loop->parent_block();
    auto *merge = loop->merge_block();
    if (parent == nullptr || loop_body == nullptr || merge == nullptr) {
        return reject("frontend ray-query loop has a null structural block");
    }
    auto *function = parent->parent_function();
    if (function == nullptr || loop_body->parent_function() != function ||
        merge->parent_function() != function || parent == loop_body ||
        parent == merge || loop_body == merge) {
        return reject("frontend ray-query loop has invalid block ownership");
    }

    // The native DSL `$while (query.proceed())` guard translates to:
    //
    //   [optional PhiInsts after mem2reg]
    //   PROCEED(query)
    //   terminated = IS_TERMINATED(query)
    //   active = !terminated
    //   should_break = !active
    //   if (should_break) break
    //
    // Algebraic cleanup may remove both NOTs, so accept either zero or two,
    // and accept an optional private store/load temporary along the guard chain.
    auto loop_body_instructions = collect_instructions(loop_body);
    luisa::vector<PhiInst *> loop_body_phis;
    auto guard_start = size_t{0u};
    while (guard_start < loop_body_instructions.size() &&
           loop_body_instructions[guard_start]->isa<PhiInst>()) {
        loop_body_phis.emplace_back(
            static_cast<PhiInst *>(loop_body_instructions[guard_start]));
        ++guard_start;
    }
    if (guard_start + 3u > loop_body_instructions.size() ||
        !is_ray_query_write(
            loop_body_instructions[guard_start],
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED)) {
        return reject("frontend ray-query loop guard is not canonical");
    }
    auto *query = loop_body_instructions[guard_start]->operand(0u);
    if (!reconstruct_is_ray_query_object(query) ||
        !is_ray_query_read(
            loop_body_instructions[guard_start + 1u],
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
            query)) {
        return reject("frontend ray-query guard does not use one valid query object");
    }
    Instruction *guard_value = loop_body_instructions[guard_start + 1u];
    auto guard_value_index = guard_start + 2u;
    auto not_count = 0u;
    auto has_interleaved_temp = false;
    while (guard_value_index + 1u < loop_body_instructions.size()) {
        if (loop_body_instructions[guard_value_index]->isa<ArithmeticInst>()) {
            auto *not_inst = static_cast<ArithmeticInst *>(
                loop_body_instructions[guard_value_index]);
            if (not_inst->op() != ArithmeticOp::UNARY_BIT_NOT ||
                not_inst->operand_count() != 1u ||
                not_inst->operand(0u) != guard_value ||
                not_inst->type() != Type::of<bool>() ||
                !has_exactly_one_use_by(guard_value, not_inst)) {
                break;
            }
            guard_value = not_inst;
            ++guard_value_index;
            ++not_count;
            continue;
        }
        if (guard_value_index + 2u < loop_body_instructions.size() &&
            loop_body_instructions[guard_value_index]->isa<StoreInst>() &&
            loop_body_instructions[guard_value_index + 1u]->isa<LoadInst>()) {
            auto *store = static_cast<StoreInst *>(
                loop_body_instructions[guard_value_index]);
            auto *load = static_cast<LoadInst *>(
                loop_body_instructions[guard_value_index + 1u]);
            if (store->value() != guard_value ||
                load->variable() != store->variable() ||
                load->type() != Type::of<bool>() ||
                !has_exactly_one_use_by(guard_value, store) ||
                !has_exactly_two_uses_by(
                    store->variable(), store, load)) {
                return reject("frontend ray-query loop guard temporary escapes its shell");
            }
            if (not_count == 1u) { has_interleaved_temp = true; }
            guard_value = load;
            guard_value_index += 2u;
            continue;
        }
        break;
    }
    if (not_count != 0u && not_count != 2u) {
        return reject("frontend ray-query loop termination test is not canonical");
    }
    if (guard_value_index + 1u != loop_body_instructions.size() ||
        !loop_body_instructions[guard_value_index]->isa<IfInst>()) {
        return reject("frontend ray-query loop termination test is not canonical");
    }
    auto *guard_if = static_cast<IfInst *>(
        loop_body_instructions[guard_value_index]);
    if (guard_if->condition() != guard_value ||
        !has_exactly_one_use_by(guard_value, guard_if)) {
        return reject("frontend ray-query loop guard condition escapes its shell");
    }

    auto *break_block = guard_if->true_block();
    auto *relay_block = guard_if->false_block();
    auto *candidate_block = guard_if->merge_block();
    if (break_block == nullptr || relay_block == nullptr ||
        candidate_block == nullptr) {
        return reject("frontend ray-query loop guard has a null branch or merge");
    }
    auto break_instructions = collect_instructions(break_block);
    auto relay_instructions = collect_instructions(relay_block);
    if (break_instructions.size() != 1u ||
        !break_instructions.front()->isa<BreakInst>() ||
        static_cast<BreakInst *>(break_instructions.front())
                ->target_block() != merge ||
        relay_instructions.size() != 1u ||
        !relay_instructions.front()->isa<BranchInst>() ||
        static_cast<BranchInst *>(relay_instructions.front())
                ->target_block() != candidate_block) {
        return reject("frontend ray-query loop guard is not break/relay canonical");
    }

    // Fast path: match the exact canonical immediate surface/procedural split
    // without PHIs or interleaved temporaries so round-trips remain bitwise identical.
    auto try_canonical_fast_path = [&]() noexcept -> bool {
        if (!loop_body_phis.empty() || has_interleaved_temp) { return false; }
        auto candidate_instructions = collect_instructions(candidate_block);
        if (candidate_instructions.size() < 2u ||
            candidate_instructions.size() > 3u ||
            !candidate_instructions.front()->isa<RayQueryObjectReadInst>()) {
            return false;
        }
        auto *candidate_read = static_cast<RayQueryObjectReadInst *>(
            candidate_instructions.front());
        if (candidate_read->operand_count() != 1u ||
            candidate_read->operand(0u) != query ||
            candidate_read->type() != Type::of<bool>() ||
            (candidate_read->op() != RayQueryObjectReadOp::
                                         RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE &&
             candidate_read->op() != RayQueryObjectReadOp::
                                         RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE)) {
            return false;
        }
        auto *candidate_condition = static_cast<Instruction *>(candidate_read);
        auto candidate_if_index = 1u;
        auto candidate_negated = false;
        if (candidate_instructions.size() == 3u) {
            if (!candidate_instructions[1u]->isa<ArithmeticInst>()) { return false; }
            auto *not_inst = static_cast<ArithmeticInst *>(
                candidate_instructions[1u]);
            if (not_inst->op() != ArithmeticOp::UNARY_BIT_NOT ||
                not_inst->operand_count() != 1u ||
                not_inst->operand(0u) != candidate_read ||
                not_inst->type() != Type::of<bool>() ||
                !has_exactly_one_use_by(candidate_read, not_inst)) {
                return false;
            }
            candidate_condition = not_inst;
            candidate_if_index = 2u;
            candidate_negated = true;
        }
        if (!candidate_instructions[candidate_if_index]->isa<IfInst>()) { return false; }
        auto *candidate_if = static_cast<IfInst *>(
            candidate_instructions[candidate_if_index]);
        if (candidate_if->condition() != candidate_condition ||
            !has_exactly_one_use_by(candidate_condition, candidate_if)) {
            return false;
        }
        auto *latch = candidate_if->merge_block();
        auto *true_entry = candidate_if->true_block();
        auto *false_entry = candidate_if->false_block();
        if (latch == nullptr || true_entry == nullptr || false_entry == nullptr) {
            return false;
        }
        auto true_is_surface =
            candidate_read->op() == RayQueryObjectReadOp::
                                        RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE;
        true_is_surface ^= candidate_negated;
        auto *surface_entry = true_is_surface ? true_entry : false_entry;
        auto *procedural_entry = true_is_surface ? false_entry : true_entry;

        auto latch_instructions = collect_instructions(latch);
        if (latch_instructions.size() != 1u ||
            !latch_instructions.front()->isa<BranchInst>() ||
            static_cast<BranchInst *>(latch_instructions.front())
                    ->target_block() != loop_body) {
            return false;
        }

        std::array shell_blocks{
            parent, loop_body, break_block, relay_block,
            candidate_block, latch, merge};
        for (auto i = 0u; i < shell_blocks.size(); i++) {
            if (shell_blocks[i] == nullptr ||
                shell_blocks[i]->parent_function() != function) {
                return false;
            }
            for (auto j = i + 1u; j < shell_blocks.size(); j++) {
                if (shell_blocks[i] == shell_blocks[j]) { return false; }
            }
        }

        ReconstructHandlerRegion surface_region;
        ReconstructHandlerRegion procedural_region;
        luisa::string_view ignore_reason;
        if (!collect_handler_region(
                surface_entry, candidate_block, latch,
                surface_region, ignore_reason) ||
            !collect_handler_region(
                procedural_entry, candidate_block, latch,
                procedural_region, ignore_reason)) {
            return false;
        }
        if (handler_regions_overlap(surface_region, procedural_region)) {
            return false;
        }
        for (auto *block : surface_region.blocks) {
            if (block_contains_ray_query_proceed(block)) { return false; }
        }
        for (auto *block : procedural_region.blocks) {
            if (block_contains_ray_query_proceed(block)) { return false; }
        }

        std::array loop_body_predecessors{parent, latch};
        std::array guard_arm_predecessors{loop_body};
        std::array candidate_predecessors{relay_block};
        if (!predecessors_are_subset_of(
                loop_body, luisa::span{loop_body_predecessors}) ||
            !predecessors_are_subset_of(
                break_block, luisa::span{guard_arm_predecessors}) ||
            !predecessors_are_subset_of(
                relay_block, luisa::span{guard_arm_predecessors}) ||
            !predecessors_are_subset_of(
                candidate_block, luisa::span{candidate_predecessors})) {
            return false;
        }
        luisa::vector<BasicBlock *> latch_predecessors;
        for (auto *exit : surface_region.exits) {
            latch_predecessors.emplace_back(exit->parent_block());
        }
        for (auto *exit : procedural_region.exits) {
            latch_predecessors.emplace_back(exit->parent_block());
        }
        if (!predecessors_are_subset_of(
                latch, luisa::span{latch_predecessors})) {
            return false;
        }
        std::array merge_predecessors{break_block};
        if (!predecessors_are_subset_of(
                merge, luisa::span{merge_predecessors})) {
            return false;
        }

        candidate = ReconstructCandidate{
            .inline_loop = loop,
            .parent = parent,
            .prepare = loop_body,
            .body = candidate_block,
            .update = latch,
            .merge = merge,
            .query = query,
            .candidate_dispatch = candidate_if,
            .surface_entry = surface_entry,
            .procedural_entry = procedural_entry,
            .surface_empty = false,
            .procedural_empty = false,
            .surface_region = std::move(surface_region),
            .procedural_region = std::move(procedural_region),
            .inline_break = break_block,
            .inline_relay = relay_block};
        return true;
    };
    if (try_canonical_fast_path()) {
        return ReconstructMatch::accepted;
    }

    // General stateful inline path: collect all blocks reachable from
    // `candidate_block` bounded by `loop_body` (continue) and `merge` (break),
    // then preflight candidate-kind specialization into separate handlers.
    luisa::unordered_set<BasicBlock *> body_region;
    luisa::vector<BasicBlock *> worklist{candidate_block};
    luisa::vector<BasicBlock *> continue_exit_blocks;
    luisa::vector<BasicBlock *> break_exit_blocks;
    while (!worklist.empty()) {
        auto *block = worklist.back();
        worklist.pop_back();
        if (block == loop_body || block == merge) { continue; }
        if (block == nullptr || block->parent_function() != function ||
            block == parent || block == break_block || block == relay_block) {
            return reject("stateful inline ray-query body escapes its loop shell");
        }
        if (!body_region.emplace(block).second) { continue; }
        if (!block->is_terminated()) {
            return reject("stateful inline ray-query body contains an unterminated block");
        }
        auto *terminator = block->terminator();
        auto valid_edges = true;
        block->traverse_successors(false, [&](BasicBlock *successor) noexcept {
            if (successor == loop_body) {
                if ((terminator->isa<BranchInst>() || terminator->isa<ContinueInst>()) &&
                    static_cast<BranchTerminatorInstruction *>(terminator)->target_block() == loop_body) {
                    continue_exit_blocks.emplace_back(block);
                } else {
                    valid_edges = false;
                }
            } else if (successor == merge) {
                if (terminator->isa<BreakInst>() &&
                    static_cast<BreakInst *>(terminator)->target_block() == merge) {
                    break_exit_blocks.emplace_back(block);
                } else {
                    valid_edges = false;
                }
            } else if (!body_region.contains(successor)) {
                worklist.emplace_back(successor);
            }
        });
        if (!valid_edges) {
            return reject("stateful inline ray-query body has non-Branch/Continue/Break loop exit");
        }
        if (auto *cf_merge = terminator->control_flow_merge();
            cf_merge != nullptr && cf_merge->merge_block() != nullptr) {
            auto *cf_merge_block = cf_merge->merge_block();
            if (cf_merge_block == loop_body || cf_merge_block == merge ||
                cf_merge_block->parent_function() != function ||
                cf_merge_block == parent || cf_merge_block == break_block ||
                cf_merge_block == relay_block) {
                return reject("stateful inline ray-query body has a merge escaping its region");
            }
            if (!body_region.contains(cf_merge_block)) {
                worklist.emplace_back(cf_merge_block);
            }
        }
    }

    if (continue_exit_blocks.empty() && break_exit_blocks.empty()) {
        return reject("stateful inline ray-query body has no continue or break exit");
    }
    luisa::unordered_set<Instruction *> loop_phi_set{
        loop_body_phis.begin(), loop_body_phis.end()};
    for (auto *block : body_region) {
        if (auto *cf_merge = block->terminator()->control_flow_merge();
            cf_merge != nullptr && cf_merge->merge_block() != nullptr &&
            !body_region.contains(cf_merge->merge_block())) {
            return reject("stateful inline ray-query body has a merge outside its region");
        }
        auto has_external_pred = false;
        block->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
            has_external_pred |=
                !body_region.contains(pred) &&
                !(block == candidate_block && pred == relay_block);
        });
        if (has_external_pred) {
            return reject("stateful inline ray-query body has an external predecessor");
        }
        for (auto *inst : block->instructions()) {
            if (inst->isa<ReturnInst>() ||
                inst->isa<RayQueryLoopInst>() ||
                inst->isa<RayQueryPipelineInst>() ||
                inst->isa<CoroSuspendInst>() ||
                inst->isa<CoroResumeInst>() ||
                inst->isa<CoroTerminateInst>()) {
                return reject("stateful inline ray-query body contains an unsupported control-flow or ray-query instruction");
            }
            if (is_ray_query_write(inst, RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED)) {
                return reject("stateful inline ray-query body contains a nested PROCEED");
            }
            if (is_ray_query_read(inst, RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED, query)) {
                return reject("stateful inline ray-query body queries IS_TERMINATED on the active query");
            }
            for (auto *use : inst->use_list()) {
                auto *user = use->user();
                if (user == nullptr || !user->isa<Instruction>()) {
                    return reject("stateful inline ray-query body instruction escapes to non-instruction");
                }
                auto *user_inst = static_cast<Instruction *>(user);
                auto *user_block = user_inst->parent_block();
                if (!body_region.contains(user_block)) {
                    auto valid_loop_phi_backedge =
                        user_block == loop_body &&
                        user_inst->isa<PhiInst>() &&
                        loop_phi_set.contains(static_cast<PhiInst *>(user_inst));
                    if (!valid_loop_phi_backedge) {
                        return reject("stateful inline ray-query body defines an SSA value used outside the loop");
                    }
                }
            }
        }
    }
    if (!break_exit_blocks.empty()) {
        luisa::unordered_set<BasicBlock *> owned_blocks{body_region.begin(), body_region.end()};
        owned_blocks.emplace(loop_body);
        owned_blocks.emplace(break_block);
        owned_blocks.emplace(relay_block);
        if (function_has_other_proceed(function->definition(), query, owned_blocks)) {
            return reject("stateful inline ray-query with $break cannot safely lower if the same query proceeds again after the loop");
        }
    }

    luisa::vector<BasicBlock *> loop_body_predecessors{parent};
    loop_body_predecessors.insert(
        loop_body_predecessors.end(),
        continue_exit_blocks.begin(), continue_exit_blocks.end());
    std::array guard_arm_predecessors{loop_body};
    std::array candidate_predecessors{relay_block};
    if (!predecessors_are_subset_of(loop_body, luisa::span{loop_body_predecessors}) ||
        !predecessors_are_subset_of(break_block, luisa::span{guard_arm_predecessors}) ||
        !predecessors_are_subset_of(relay_block, luisa::span{guard_arm_predecessors}) ||
        !predecessors_are_subset_of(candidate_block, luisa::span{candidate_predecessors})) {
        return reject("stateful inline ray-query shell has an external predecessor");
    }
    luisa::vector<BasicBlock *> merge_predecessors{break_block};
    merge_predecessors.insert(
        merge_predecessors.end(),
        break_exit_blocks.begin(), break_exit_blocks.end());
    if (!predecessors_are_subset_of(merge, luisa::span{merge_predecessors})) {
        return reject("stateful inline ray-query merge has an external predecessor");
    }
    for (auto *inst : merge->instructions()) {
        if (inst->isa<PhiInst>()) {
            return reject("stateful inline ray-query merge contains a PHI node");
        }
    }
    for (auto *phi : loop_body_phis) {
        if (phi->incoming_count() != loop_body_predecessors.size()) {
            return reject("stateful inline ray-query header PHI does not match predecessors");
        }
        auto has_parent = false;
        for (size_t i = 0u; i < phi->incoming_count(); ++i) {
            auto inc = phi->incoming(i);
            if (inc.block == parent) {
                has_parent = true;
                if (inc.value->isa<Instruction>()) {
                    auto *inc_block = static_cast<Instruction *>(inc.value)->parent_block();
                    if (inc_block == loop_body || body_region.contains(inc_block)) {
                        return reject("stateful inline ray-query header PHI entry value comes from inside the loop");
                    }
                }
            } else if (!body_region.contains(inc.block)) {
                return reject("stateful inline ray-query header PHI references an external block");
            }
        }
        if (!has_parent) {
            return reject("stateful inline ray-query header PHI is missing loop entry edge");
        }
    }

    GeneralInlineSpecialization surface_spec;
    GeneralInlineSpecialization procedural_spec;
    if (!analyze_general_inline_specialization(
            candidate_block, loop_body, merge, body_region,
            query, true, surface_spec) ||
        !analyze_general_inline_specialization(
            candidate_block, loop_body, merge, body_region,
            query, false, procedural_spec)) {
        return reject("stateful inline ray-query mixes surface and procedural candidate operations without candidate-kind specialization");
    }
    if (surface_spec.has_procedural_ops && !surface_spec.has_surface_ops) {
        surface_spec.collapse_empty = true;
    }
    if (procedural_spec.has_surface_ops && !procedural_spec.has_procedural_ops) {
        procedural_spec.collapse_empty = true;
    }
    if (surface_spec.collapse_empty && procedural_spec.collapse_empty) {
        return reject("stateful inline ray-query specialization collapsed both candidate arms");
    }

    candidate = ReconstructCandidate{
        .inline_loop = loop,
        .parent = parent,
        .prepare = loop_body,
        .body = candidate_block,
        .merge = merge,
        .query = query,
        .candidate_dispatch = guard_if,
        .inline_break = break_block,
        .inline_relay = relay_block,
        .is_general_inline = true,
        .loop_body_phis = std::move(loop_body_phis),
        .general_body_blocks = std::move(body_region),
        .surface_spec = std::move(surface_spec),
        .procedural_spec = std::move(procedural_spec)};
    return ReconstructMatch::accepted;
}

static void replace_reconstructed_ray_query_phi_predecessor(
    BasicBlock *block, BasicBlock *old_predecessor,
    BasicBlock *new_predecessor) noexcept {
    for (auto *inst : block->instructions()) {
        if (!inst->isa<PhiInst>()) { continue; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (auto i = 0u; i < phi->incoming_count(); i++) {
            auto incoming = phi->incoming(i);
            if (incoming.block == old_predecessor) {
                phi->set_incoming(i, incoming.value, new_predecessor);
            }
        }
    }
}

class ReconstructInlineValueResolver final : public InstructionCloneValueResolver {
private:
    luisa::unordered_map<const Value *, Value *> _map;

public:
    void emplace(const Value *src, Value *dst) noexcept {
        _map[src] = dst;
    }
    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) { return nullptr; }
        if (auto iter = _map.find(value); iter != _map.end()) {
            return iter->second;
        }
        return const_cast<Value *>(value);
    }
};

static void reconstruct_general_inline_candidate(
    ReconstructCandidate &candidate,
    ReconstructRayQueryLoopInfo &info) noexcept {
    XIRBuilder builder;
    auto *function = candidate.parent->parent_function();
    auto *module = function->parent_module();
    auto removed_loop = candidate.inline_loop->remove_self();
    builder.set_insertion_point(candidate.parent);
    auto *ray_query_loop = builder.ray_query_loop();
    reconstruct_clone_metadata(*removed_loop, *ray_query_loop);
    ray_query_loop->set_merge_block(candidate.merge);

    auto *dispatch_block = ray_query_loop->create_dispatch_block();
    builder.set_insertion_point(dispatch_block);
    luisa::vector<PhiInst *> new_loop_phis;
    new_loop_phis.reserve(candidate.loop_body_phis.size());
    for (auto *orig_phi : candidate.loop_body_phis) {
        auto *new_phi = builder.phi(orig_phi->type());
        reconstruct_clone_metadata(*orig_phi, *new_phi);
        for (size_t i = 0u; i < orig_phi->incoming_count(); ++i) {
            auto inc = orig_phi->incoming(i);
            if (inc.block == candidate.parent) {
                new_phi->add_incoming(inc.value, candidate.parent);
                break;
            }
        }
        new_loop_phis.emplace_back(new_phi);
    }
    auto *dispatch = builder.ray_query_dispatch(candidate.query);
    reconstruct_clone_metadata(*candidate.candidate_dispatch, *dispatch);
    dispatch->set_exit_block(candidate.merge);

    auto *bool_true = module->create_constant_one(Type::of<bool>());
    auto *bool_false = module->create_constant_zero(Type::of<bool>());

    auto clone_specialized_handler = [&](const GeneralInlineSpecialization &spec,
                                         bool is_surface) noexcept {
        if (spec.collapse_empty || spec.reachable_blocks.empty()) {
            auto *empty_block = is_surface ?
                                    dispatch->create_on_surface_candidate_block() :
                                    dispatch->create_on_procedural_candidate_block();
            builder.set_insertion_point(empty_block);
            builder.br(dispatch_block);
            for (auto *new_phi : new_loop_phis) {
                new_phi->add_incoming(module->create_undefined(new_phi->type()), empty_block);
            }
            return;
        }

        ReconstructInlineValueResolver resolver;
        for (size_t i = 0u; i < candidate.loop_body_phis.size(); ++i) {
            resolver.emplace(candidate.loop_body_phis[i], new_loop_phis[i]);
        }

        luisa::unordered_map<const BasicBlock *, BasicBlock *> block_map;
        for (auto *orig_block : spec.reachable_blocks) {
            BasicBlock *cloned = orig_block == candidate.body ?
                                     (is_surface ?
                                          dispatch->create_on_surface_candidate_block() :
                                          dispatch->create_on_procedural_candidate_block()) :
                                     function->create_basic_block();
            reconstruct_clone_metadata(*orig_block, *cloned);
            block_map.emplace(orig_block, cloned);
            resolver.emplace(orig_block, cloned);
        }

        luisa::vector<std::pair<const PhiInst *, PhiInst *>> cloned_phis;
        luisa::vector<std::pair<const BasicBlock *, BasicBlock *>> continue_exits;
        luisa::vector<BasicBlock *> break_exits;
        for (auto *orig_block : spec.reachable_blocks) {
            auto *cloned = block_map.at(orig_block);
            builder.set_insertion_point(cloned);
            for (auto *inst : orig_block->instructions()) {
                if (inst->isa<RayQueryObjectReadInst>()) {
                    auto *read = static_cast<const RayQueryObjectReadInst *>(inst);
                    if (read->operand_count() == 1u && read->operand(0u) == candidate.query) {
                        if (read->op() == RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE) {
                            resolver.emplace(inst, is_surface ? bool_true : bool_false);
                            continue;
                        }
                        if (read->op() == RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE) {
                            resolver.emplace(inst, is_surface ? bool_false : bool_true);
                            continue;
                        }
                    }
                }
                if (inst->isa<PhiInst>()) {
                    auto *orig_phi = static_cast<const PhiInst *>(inst);
                    auto *dup_phi = builder.phi(orig_phi->type());
                    reconstruct_clone_metadata(*orig_phi, *dup_phi);
                    cloned_phis.emplace_back(orig_phi, dup_phi);
                    resolver.emplace(inst, dup_phi);
                    continue;
                }
                if (inst->is_terminator()) {
                    if (auto folded = spec.folded_bools.find(inst);
                        folded != spec.folded_bools.end()) {
                        auto *cond_term = static_cast<const ConditionalBranchTerminatorInstruction *>(inst);
                        auto *orig_target = folded->second ? cond_term->true_block() : cond_term->false_block();
                        if (orig_target == candidate.prepare) {
                            builder.br(dispatch_block);
                            continue_exits.emplace_back(orig_block, cloned);
                        } else if (orig_target == candidate.merge) {
                            builder.call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE, {candidate.query});
                            builder.br(dispatch_block);
                            break_exits.emplace_back(cloned);
                        } else {
                            builder.br(block_map.at(orig_target));
                        }
                        continue;
                    }
                    if ((inst->isa<BranchInst>() || inst->isa<ContinueInst>()) &&
                        static_cast<const BranchTerminatorInstruction *>(inst)->target_block() == candidate.prepare) {
                        auto *br = builder.br(dispatch_block);
                        reconstruct_clone_metadata(*inst, *br);
                        continue_exits.emplace_back(orig_block, cloned);
                        continue;
                    }
                    if (inst->isa<BreakInst>() &&
                        static_cast<const BreakInst *>(inst)->target_block() == candidate.merge) {
                        builder.call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE, {candidate.query});
                        auto *br = builder.br(dispatch_block);
                        reconstruct_clone_metadata(*inst, *br);
                        break_exits.emplace_back(cloned);
                        continue;
                    }
                }
                auto *dup_inst = inst->clone_with_metadata(builder, resolver);
                LUISA_DEBUG_ASSERT(dup_inst != nullptr, "Failed to clone instruction in inline ray-query specialization.");
                resolver.emplace(inst, dup_inst);
            }
        }

        // Wire cloned internal PHI nodes using actual CFG predecessors of each cloned block.
        for (auto &&[orig_phi, dup_phi] : cloned_phis) {
            luisa::unordered_set<BasicBlock *> actual_preds;
            dup_phi->parent_block()->traverse_predecessors(false, [&](BasicBlock *pred) noexcept {
                actual_preds.emplace(pred);
            });
            for (size_t i = 0u; i < orig_phi->incoming_count(); ++i) {
                auto inc = orig_phi->incoming(i);
                auto pred_iter = block_map.find(inc.block);
                if (pred_iter != block_map.end() && actual_preds.contains(pred_iter->second)) {
                    dup_phi->add_incoming(resolver.resolve(inc.value), pred_iter->second);
                }
            }
        }

        // Wire dispatch_block PHI incoming values from cloned exits.
        for (size_t phi_idx = 0u; phi_idx < candidate.loop_body_phis.size(); ++phi_idx) {
            auto *orig_phi = candidate.loop_body_phis[phi_idx];
            auto *new_phi = new_loop_phis[phi_idx];
            for (auto &&[orig_exit, cloned_exit] : continue_exits) {
                Value *resolved_val = module->create_undefined(new_phi->type());
                for (size_t i = 0u; i < orig_phi->incoming_count(); ++i) {
                    auto inc = orig_phi->incoming(i);
                    if (inc.block == orig_exit) {
                        resolved_val = resolver.resolve(inc.value);
                        break;
                    }
                }
                new_phi->add_incoming(resolved_val, cloned_exit);
            }
            for (auto *cloned_break : break_exits) {
                new_phi->add_incoming(module->create_undefined(new_phi->type()), cloned_break);
            }
        }
    };

    clone_specialized_handler(candidate.surface_spec, true);
    clone_specialized_handler(candidate.procedural_spec, false);

    for (size_t i = 0u; i < candidate.loop_body_phis.size(); ++i) {
        candidate.loop_body_phis[i]->replace_all_uses_with(new_loop_phis[i]);
    }
    replace_reconstructed_ray_query_phi_predecessor(
        candidate.merge, candidate.inline_break, dispatch_block);

    luisa::vector<BasicBlock *> old_blocks{
        candidate.prepare, candidate.inline_break, candidate.inline_relay};
    old_blocks.insert(
        old_blocks.end(),
        candidate.general_body_blocks.begin(),
        candidate.general_body_blocks.end());
    for (auto *block : old_blocks) {
        while (!block->instructions().empty()) {
            block->instructions().back()->remove_self();
        }
    }
    for (auto *block : old_blocks) {
        block->remove_self();
    }
    ++info.reconstructed_ray_query_loop_count;
}

static void reconstruct_candidate(
    ReconstructCandidate &candidate,
    ReconstructRayQueryLoopInfo &info) noexcept {
    if (candidate.is_general_inline) {
        reconstruct_general_inline_candidate(candidate, info);
        return;
    }
    XIRBuilder builder;
    auto is_inline = candidate.inline_loop != nullptr;
    RayQueryLoopInst *ray_query_loop = nullptr;
    if (is_inline) {
        auto removed_loop = candidate.inline_loop->remove_self();
        builder.set_insertion_point(candidate.parent);
        ray_query_loop = builder.ray_query_loop();
        reconstruct_clone_metadata(*removed_loop, *ray_query_loop);
    } else {
        auto removed_loop = candidate.loop->remove_self();
        builder.set_insertion_point(candidate.parent);
        ray_query_loop = builder.ray_query_loop();
        reconstruct_clone_metadata(*removed_loop, *ray_query_loop);
    }
    ray_query_loop->set_merge_block(candidate.merge);
    auto *dispatch_block = ray_query_loop->create_dispatch_block();
    builder.set_insertion_point(dispatch_block);
    auto *dispatch = builder.ray_query_dispatch(candidate.query);
    reconstruct_clone_metadata(*candidate.candidate_dispatch, *dispatch);
    dispatch->set_exit_block(candidate.merge);

    auto materialize_handler = [&](bool empty, BasicBlock *entry,
                                   bool surface) noexcept {
        if (!empty) {
            if (surface) {
                dispatch->set_on_surface_candidate_block(entry);
            } else {
                dispatch->set_on_procedural_candidate_block(entry);
            }
            return entry;
        }
        auto *block = surface ?
                          dispatch->create_on_surface_candidate_block() :
                          dispatch->create_on_procedural_candidate_block();
        builder.set_insertion_point(block);
        builder.br(dispatch_block);
        return block;
    };
    static_cast<void>(materialize_handler(
        candidate.surface_empty, candidate.surface_entry, true));
    static_cast<void>(materialize_handler(
        candidate.procedural_empty, candidate.procedural_entry, false));

    auto retarget_exits = [&](ReconstructHandlerRegion &region) noexcept {
        for (auto *exit : region.exits) {
            exit->set_target_block(dispatch_block);
        }
    };
    retarget_exits(candidate.surface_region);
    retarget_exits(candidate.procedural_region);
    replace_reconstructed_ray_query_phi_predecessor(
        candidate.merge,
        is_inline ?
            candidate.inline_break :
            candidate.prepare,
        dispatch_block);

    luisa::vector<BasicBlock *> shell_blocks{
        candidate.prepare, candidate.body, candidate.update};
    if (is_inline) {
        shell_blocks.emplace_back(candidate.inline_break);
        shell_blocks.emplace_back(candidate.inline_relay);
    }
    // Detach all users before unlinking the generated shell blocks.
    for (auto *block : shell_blocks) {
        while (!block->instructions().empty()) {
            block->instructions().back()->remove_self();
        }
    }
    for (auto *block : shell_blocks) {
        block->remove_self();
    }
    ++info.reconstructed_ray_query_loop_count;
}

struct FunctionReconstructionWork {
    Function *function{nullptr};
    luisa::vector<ReconstructCandidate> candidates;
};

static void preflight_function(
    Function *function, FunctionReconstructionWork &work,
    ReconstructRayQueryLoopInfo &info) noexcept {
    work.function = function;
    if (function == nullptr || function->definition() == nullptr) { return; }
    luisa::vector<Instruction *> loops;
    for (auto *block : function->definition()->basic_blocks()) {
        if (block->is_terminated() &&
            (block->terminator()->isa<LoopInst>() ||
             block->terminator()->isa<SimpleLoopInst>())) {
            loops.emplace_back(block->terminator());
        }
    }
    for (auto *loop : loops) {
        ReconstructCandidate candidate;
        luisa::string_view reason;
        auto match = loop->isa<LoopInst>() ?
                         match_canonical_ray_query_loop(
                             static_cast<LoopInst *>(loop),
                             candidate, reason) :
                         match_frontend_inline_ray_query_loop(
                             static_cast<SimpleLoopInst *>(loop),
                             candidate, reason);
        switch (match) {
            case ReconstructMatch::ignored:
                ++info.ignored_loop_count;
                break;
            case ReconstructMatch::accepted:
                work.candidates.emplace_back(std::move(candidate));
                break;
            case ReconstructMatch::rejected:
                LUISA_WARNING_WITH_LOCATION(
                    "reconstruct_ray_query_loop: rejecting ray-like loop: {}",
                    reason);
                ++info.error_count;
                break;
        }
    }
}

static void reconstruct_preflighted_function(
    FunctionReconstructionWork &work,
    ReconstructRayQueryLoopInfo &info) noexcept {
    for (auto &candidate : work.candidates) {
        reconstruct_candidate(candidate, info);
    }
}

}// namespace detail

ReconstructRayQueryLoopInfo
reconstruct_ray_query_loop_pass_run_on_function(
    Function *function) noexcept {
    ReconstructRayQueryLoopInfo info;
    detail::FunctionReconstructionWork work;
    detail::preflight_function(function, work, info);
    if (info.succeeded()) {
        detail::reconstruct_preflighted_function(work, info);
    }
    return info;
}

ReconstructRayQueryLoopInfo
reconstruct_ray_query_loop_pass_run_on_module(
    Module *module, PassReport *report) noexcept {
    ReconstructRayQueryLoopInfo info;
    luisa::vector<detail::FunctionReconstructionWork> work;
    if (module != nullptr) {
        for (auto *function : module->function_list()) {
            auto &item = work.emplace_back();
            detail::preflight_function(function, item, info);
        }
        if (info.succeeded()) {
            for (auto &item : work) {
                detail::reconstruct_preflighted_function(item, info);
            }
        }
    }
    if (report != nullptr) {
        report->set(
            "reconstructed_ray_query_loop",
            info.reconstructed_ray_query_loop_count);
        report->set("ignored_loop", info.ignored_loop_count);
        report->set("error", info.error_count);
    }
    return info;
}

}// namespace luisa::compute::xir
