#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/reconstruct_ray_query_loop.h>

#include <algorithm>
#include <array>

namespace luisa::compute::xir {

namespace detail {

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
        std::swap(smaller, larger);
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
    if (!is_ray_query_object(query) ||
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
    //   PROCEED(query)
    //   terminated = IS_TERMINATED(query)
    //   active = !terminated
    //   should_break = !active
    //   if (should_break) break
    //
    // Algebraic cleanup may remove both NOTs, so accept either zero or two,
    // but never one: one NOT would invert the loop termination semantics.
    auto loop_body_instructions = collect_instructions(loop_body);
    if (loop_body_instructions.size() < 3u ||
        !is_ray_query_write(
            loop_body_instructions[0u],
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED)) {
        return reject("frontend ray-query loop guard is not canonical");
    }
    auto *query = loop_body_instructions[0u]->operand(0u);
    if (!is_ray_query_object(query) ||
        !is_ray_query_read(
            loop_body_instructions[1u],
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
            query)) {
        return reject("frontend ray-query guard does not use one valid query object");
    }
    auto *guard_value = loop_body_instructions[1u];
    auto guard_value_index = 2u;
    auto not_count = 0u;
    while (guard_value_index + 1u < loop_body_instructions.size() &&
           loop_body_instructions[guard_value_index]
               ->isa<ArithmeticInst>()) {
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
    }
    if (not_count != 0u && not_count != 2u) {
        return reject("frontend ray-query loop termination test is not canonical");
    }
    // The generic DSL unary operator materializes its result as a temporary
    // Var<bool>. Match that one private store/load pair, but require the local
    // to have no other users so deleting the shell cannot alter user state.
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
        guard_value = load;
        guard_value_index += 2u;
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

    // The loop body must dispatch the published candidate exactly once. Both
    // public predicates are accepted, as is one explicit logical negation.
    auto candidate_instructions = collect_instructions(candidate_block);
    if (candidate_instructions.size() < 2u ||
        candidate_instructions.size() > 3u ||
        !candidate_instructions.front()->isa<RayQueryObjectReadInst>()) {
        return reject("frontend ray-query candidate dispatch is not canonical");
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
        return reject("frontend candidate predicate uses a different query object");
    }
    auto *candidate_condition = static_cast<Instruction *>(candidate_read);
    auto candidate_if_index = 1u;
    auto candidate_negated = false;
    if (candidate_instructions.size() == 3u) {
        if (!candidate_instructions[1u]->isa<ArithmeticInst>()) {
            return reject("frontend candidate predicate negation is malformed");
        }
        auto *not_inst = static_cast<ArithmeticInst *>(
            candidate_instructions[1u]);
        if (not_inst->op() != ArithmeticOp::UNARY_BIT_NOT ||
            not_inst->operand_count() != 1u ||
            not_inst->operand(0u) != candidate_read ||
            not_inst->type() != Type::of<bool>() ||
            !has_exactly_one_use_by(candidate_read, not_inst)) {
            return reject("frontend candidate predicate negation is malformed");
        }
        candidate_condition = not_inst;
        candidate_if_index = 2u;
        candidate_negated = true;
    }
    if (!candidate_instructions[candidate_if_index]->isa<IfInst>()) {
        return reject("frontend candidate predicate does not terminate in an IfInst");
    }
    auto *candidate_if = static_cast<IfInst *>(
        candidate_instructions[candidate_if_index]);
    if (candidate_if->condition() != candidate_condition ||
        !has_exactly_one_use_by(candidate_condition, candidate_if)) {
        return reject("frontend candidate predicate escapes its dispatch shell");
    }
    auto *latch = candidate_if->merge_block();
    auto *true_entry = candidate_if->true_block();
    auto *false_entry = candidate_if->false_block();
    if (latch == nullptr || true_entry == nullptr || false_entry == nullptr) {
        return reject("frontend candidate dispatch has a null branch or merge");
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
        return reject("frontend ray-query loop latch is not canonical");
    }

    std::array shell_blocks{
        parent, loop_body, break_block, relay_block,
        candidate_block, latch, merge};
    for (auto i = 0u; i < shell_blocks.size(); i++) {
        if (shell_blocks[i] == nullptr ||
            shell_blocks[i]->parent_function() != function) {
            return reject("frontend ray-query shell crosses a function boundary");
        }
        for (auto j = i + 1u; j < shell_blocks.size(); j++) {
            if (shell_blocks[i] == shell_blocks[j]) {
                return reject("frontend ray-query shell aliases structural blocks");
            }
        }
    }

    ReconstructHandlerRegion surface_region;
    ReconstructHandlerRegion procedural_region;
    if (!collect_handler_region(
            surface_entry, candidate_block, latch,
            surface_region, reason) ||
        !collect_handler_region(
            procedural_entry, candidate_block, latch,
            procedural_region, reason)) {
        return ReconstructMatch::rejected;
    }
    if (handler_regions_overlap(surface_region, procedural_region)) {
        return reject("frontend surface and procedural handler regions overlap");
    }
    for (auto *block : surface_region.blocks) {
        if (block_contains_ray_query_proceed(block)) {
            return reject("frontend surface handler contains a nested PROCEED");
        }
    }
    for (auto *block : procedural_region.blocks) {
        if (block_contains_ray_query_proceed(block)) {
            return reject("frontend procedural handler contains a nested PROCEED");
        }
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
        return reject("frontend ray-query shell has an external predecessor");
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
        return reject("frontend ray-query latch has an external predecessor");
    }
    std::array merge_predecessors{break_block};
    if (!predecessors_are_subset_of(
            merge, luisa::span{merge_predecessors})) {
        return reject("frontend ray-query merge has an external predecessor");
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
    return ReconstructMatch::accepted;
}

static void replace_phi_predecessor(
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

static void reconstruct_candidate(
    ReconstructCandidate &candidate,
    ReconstructRayQueryLoopInfo &info) noexcept {
    XIRBuilder builder;
    auto is_inline = candidate.inline_loop != nullptr;
    RayQueryLoopInst *ray_query_loop = nullptr;
    if (is_inline) {
        auto removed_loop = candidate.inline_loop->remove_self();
        builder.set_insertion_point(candidate.parent);
        ray_query_loop = builder.ray_query_loop();
        clone_metadata(*removed_loop, *ray_query_loop);
    } else {
        auto removed_loop = candidate.loop->remove_self();
        builder.set_insertion_point(candidate.parent);
        ray_query_loop = builder.ray_query_loop();
        clone_metadata(*removed_loop, *ray_query_loop);
    }
    ray_query_loop->set_merge_block(candidate.merge);
    auto *dispatch_block = ray_query_loop->create_dispatch_block();
    builder.set_insertion_point(dispatch_block);
    auto *dispatch = builder.ray_query_dispatch(candidate.query);
    clone_metadata(*candidate.candidate_dispatch, *dispatch);
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
    replace_phi_predecessor(
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
