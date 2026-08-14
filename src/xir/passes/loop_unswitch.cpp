#include <luisa/xir/passes/loop_unswitch.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/clock.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/undefined.h>

#include "helpers.h"
#include "natural_loop.h"

namespace luisa::compute::xir {

namespace detail {

namespace {

class LoopCloneValueResolver final : public InstructionCloneValueResolver {

private:
    luisa::unordered_map<const Value *, Value *> _values;

public:
    void map(const Value *source, Value *clone) noexcept {
        _values.emplace(source, clone);
    }

    [[nodiscard]] Value *resolve(const Value *value) noexcept override {
        if (value == nullptr) { return nullptr; }
        if (auto iter = _values.find(value); iter != _values.end()) {
            return iter->second;
        }
        return const_cast<Value *>(value);
    }
};

struct LiveOutPlan {
    Instruction *value{nullptr};
    luisa::vector<Use *> ordinary_uses;
};

struct UnswitchPlan {
    NaturalLoop loop;
    ConditionalBranchInst *candidate{nullptr};
    luisa::vector<BasicBlock *> blocks;
    luisa::vector<LiveOutPlan> live_outs;
    size_t instruction_count{0u};
    bool guarded_dynamic_trip{false};
    ConditionalBranchInst *guard_source{nullptr};
    BasicBlock *guard_exit{nullptr};
    bool guard_body_is_true_target{false};
    luisa::unordered_map<PhiInst *, Value *> guard_substitution;
};

template<typename T>
void clone_metadata(const T &source, T &destination) noexcept {
    for (auto *metadata : source.metadata_list()) {
        destination.metadata_list().push_front(metadata->clone());
    }
}

[[nodiscard]] bool is_exit_phi_edge_use(
    Use *use, PhiInst *phi, BasicBlock *exit_source) noexcept {
    if (use == nullptr || phi == nullptr) { return false; }
    for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
        auto incoming = phi->incoming_use(i);
        if (incoming.value == use && incoming.block == exit_source) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] luisa::vector<BasicBlock *> ordered_loop_blocks(
    FunctionDefinition *definition, const NaturalLoop &loop) noexcept {
    luisa::vector<BasicBlock *> blocks;
    definition->traverse_basic_blocks(
        BasicBlockTraversalOrder::REVERSE_POST_ORDER,
        [&](BasicBlock *block) noexcept {
            if (loop.contains(block)) { blocks.emplace_back(block); }
        });
    return blocks;
}

[[nodiscard]] bool loop_contains_nested_header(
    const NaturalLoop &loop,
    const luisa::vector<NaturalLoop> &all_loops) noexcept {
    for (auto &&other : all_loops) {
        if (other.header != loop.header && loop.contains(other.header)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool guard_value_is_clonable(
    Value *value, BasicBlock *header,
    const luisa::unordered_map<PhiInst *, Value *> &substitution) noexcept {
    if (value == nullptr) { return false; }
    if (!value->isa<Instruction>()) { return true; }
    auto *instruction = static_cast<Instruction *>(value);
    if (instruction->parent_block() != header) { return true; }
    if (instruction->isa<PhiInst>()) {
        return substitution.contains(
            static_cast<PhiInst *>(instruction));
    }
    if (!instruction->isa<ArithmeticInst>()) { return false; }
    for (auto *use : instruction->operand_uses()) {
        auto *operand = use->value();
        if (!guard_value_is_clonable(
                operand, header, substitution)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] Value *materialize_guard_value(
    XIRBuilder &builder, LoopCloneValueResolver &resolver,
    Value *value, BasicBlock *header) noexcept {
    if (value == nullptr || !value->isa<Instruction>()) {
        return value;
    }
    auto *instruction = static_cast<Instruction *>(value);
    if (instruction->parent_block() != header ||
        instruction->isa<PhiInst>()) {
        return resolver.resolve(value);
    }
    if (auto *resolved = resolver.resolve(value);
        resolved != value) {
        return resolved;
    }
    for (auto *use : instruction->operand_uses()) {
        static_cast<void>(materialize_guard_value(
            builder, resolver, use->value(), header));
    }
    auto *clone = instruction->clone_with_metadata(builder, resolver);
    LUISA_ASSERT(clone != nullptr,
                 "Loop unswitching failed to clone a guard value.");
    resolver.map(instruction, clone);
    return clone;
}

[[nodiscard]] bool loop_is_read_only_and_cohort_insensitive(
    const luisa::vector<BasicBlock *> &blocks) noexcept {
    for (auto *block : blocks) {
        auto *terminator = block->terminator();
        if (terminator == nullptr ||
            !(terminator->isa<BranchInst>() ||
              terminator->isa<ConditionalBranchInst>())) {
            return false;
        }
        for (auto *instruction : block->instructions()) {
            if (instruction->is_terminator() ||
                instruction->isa<PhiInst>()) {
                continue;
            }
            auto memory = get_memory_info(instruction);
            if (memory.is_volatile || memory.writes_memory() ||
                instruction->isa<ClockInst>()) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool collect_live_outs(
    UnswitchPlan &plan, const DomTree &dom_tree) noexcept {
    auto *exit_source = plan.loop.exit_edges.front().first;
    auto *exit_block = plan.loop.exit_edges.front().second;
    for (auto *block : plan.blocks) {
        for (auto *instruction : block->instructions()) {
            if (instruction->type() == nullptr ||
                instruction->use_list().empty()) {
                continue;
            }
            LiveOutPlan live_out{.value = instruction};
            for (auto *use : instruction->use_list()) {
                auto *user = use->user();
                if (user == nullptr || !user->isa<Instruction>()) {
                    return false;
                }
                auto *user_instruction =
                    static_cast<Instruction *>(user);
                auto *user_block = user_instruction->parent_block();
                if (user_block != nullptr &&
                    plan.loop.contains(user_block)) {
                    continue;
                }
                if (user_instruction->isa<PhiInst>() &&
                    user_block == exit_block &&
                    is_exit_phi_edge_use(
                        use, static_cast<PhiInst *>(user_instruction),
                        exit_source)) {
                    continue;
                }
                // A Phi in the exit consumes values on predecessor edges; a
                // newly inserted exit Phi cannot dominate such an operand.
                if (user_block == nullptr ||
                    (user_block == exit_block &&
                     user_instruction->isa<PhiInst>()) ||
                    instruction->is_lvalue() ||
                    !dom_tree.dominates(exit_block, user_block)) {
                    return false;
                }
                live_out.ordinary_uses.emplace_back(use);
            }
            if (!live_out.ordinary_uses.empty()) {
                plan.live_outs.emplace_back(std::move(live_out));
            }
        }
    }
    return true;
}

[[nodiscard]] bool make_plan(
    FunctionDefinition *definition, const NaturalLoop &loop,
    const luisa::vector<NaturalLoop> &all_loops,
    const DomTree &dom_tree, const LoopUnswitchOptions &options,
    UnswitchPlan &plan) noexcept {
    if (definition == nullptr || loop.header == nullptr ||
        loop.preheader == nullptr || loop.latches.size() != 1u ||
        loop.exit_edges.size() != 1u ||
        loop.exit_blocks.size() != 1u ||
        loop_contains_nested_header(loop, all_loops)) {
        return false;
    }
    auto *exit_block = loop.exit_edges.front().second;
    if (exit_block == nullptr || !exit_block->is_terminated()) {
        return false;
    }
    auto *exit_source = loop.exit_edges.front().first;
    for (auto *instruction : exit_block->instructions()) {
        if (!instruction->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(instruction);
        auto incoming_count = size_t{0u};
        for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
            incoming_count += phi->incoming(i).block == exit_source;
        }
        if (incoming_count != 1u) {
            return false;
        }
    }
    auto *preheader_terminator = loop.preheader->terminator();
    if (preheader_terminator == nullptr ||
        !preheader_terminator->isa<BranchInst>() ||
        static_cast<BranchInst *>(preheader_terminator)
                ->target_block() != loop.header) {
        return false;
    }
    auto bounds = analyze_loop_bounds(loop);
    auto guarded_dynamic_trip = false;
    if (!bounds.trip_count_is_constant) {
        guarded_dynamic_trip =
            options.enable_guarded_dynamic_trip;
        if (!guarded_dynamic_trip) { return false; }
    } else if (bounds.constant_trip_count <= 1u) {
        // A guard would be semantically valid here, but cannot amortize the
        // cloned loop and selector dispatch over repeated iterations.
        return false;
    }
    if (!guarded_dynamic_trip &&
        !preheader_terminator->metadata_list().empty()) {
        return false;
    }

    plan = {};
    plan.loop = loop;
    plan.guarded_dynamic_trip = guarded_dynamic_trip;
    plan.blocks = ordered_loop_blocks(definition, loop);
    if (plan.blocks.size() != loop.body_blocks.size() + 1u ||
        !loop_is_read_only_and_cohort_insensitive(plan.blocks)) {
        return false;
    }
    for (auto *block : plan.blocks) {
        for (auto *instruction : block->instructions()) {
            plan.instruction_count++;
            if (plan.instruction_count >
                options.max_loop_instruction_count) {
                return false;
            }
        }
    }

    if (guarded_dynamic_trip) {
        // Guarded unswitching handles the canonical top-tested shape. The
        // cloned guard evaluates only the source header condition with every
        // header Phi replaced by its preheader input. A false guard therefore
        // preserves the zero-trip path without evaluating the invariant
        // selector early.
        if (exit_source != loop.header ||
            !loop.header->terminator()->isa<ConditionalBranchInst>()) {
            return false;
        }
        auto *header_branch = static_cast<ConditionalBranchInst *>(
            loop.header->terminator());
        auto *header_condition = header_branch->condition();
        auto *true_target = header_branch->true_block();
        auto *false_target = header_branch->false_block();
        auto true_is_body = true_target != exit_block &&
                            loop.contains(true_target);
        auto false_is_body = false_target != exit_block &&
                             loop.contains(false_target);
        if (true_is_body == false_is_body ||
            (true_is_body ? false_target : true_target) != exit_block ||
            header_condition == nullptr ||
            header_condition->isa<Undefined>()) {
            return false;
        }
        for (auto *instruction : loop.header->instructions()) {
            if (!instruction->isa<PhiInst>()) { break; }
            auto *phi = static_cast<PhiInst *>(instruction);
            Value *initial = nullptr;
            for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
                auto incoming = phi->incoming(i);
                if (incoming.block == loop.preheader) {
                    initial = incoming.value;
                    break;
                }
            }
            if (initial == nullptr) { return false; }
            plan.guard_substitution.emplace(phi, initial);
        }
        if (!guard_value_is_clonable(
                header_condition, loop.header,
                plan.guard_substitution)) {
            return false;
        }
        for (auto *instruction : exit_block->instructions()) {
            if (!instruction->isa<PhiInst>()) { break; }
            auto *phi = static_cast<PhiInst *>(instruction);
            Value *exit_value = nullptr;
            for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
                auto incoming = phi->incoming(i);
                if (incoming.block == exit_source) {
                    exit_value = incoming.value;
                    break;
                }
            }
            if (!guard_value_is_clonable(
                    exit_value, loop.header,
                    plan.guard_substitution)) {
                return false;
            }
        }
        plan.guard_source = header_branch;
        plan.guard_exit = exit_block;
        plan.guard_body_is_true_target = true_is_body;
    }

    auto *latch = loop.latches.front();
    for (auto *block : plan.blocks) {
        auto *terminator = block->terminator();
        if (!terminator->isa<ConditionalBranchInst>()) { continue; }
        auto *branch = static_cast<ConditionalBranchInst *>(terminator);
        auto *condition = branch->condition();
        auto *condition_instruction =
            condition != nullptr && condition->isa<Instruction>() ?
                static_cast<Instruction *>(condition) :
                nullptr;
        if (condition == nullptr || condition->type() == nullptr ||
            !condition->type()->is_bool() || condition->isa<Constant>() ||
            condition->isa<Undefined>() ||
            branch->true_block() == nullptr ||
            branch->false_block() == nullptr ||
            branch->true_block() == branch->false_block() ||
            branch->true_block() == loop.header ||
            branch->false_block() == loop.header ||
            !loop.contains(branch->true_block()) ||
            !loop.contains(branch->false_block()) ||
            (condition_instruction != nullptr &&
             loop.contains(condition_instruction->parent_block())) ||
            !dom_tree.dominates(block, latch) ||
            (options.candidate_filter != nullptr &&
             !options.candidate_filter(
                 branch, options.candidate_filter_context))) {
            continue;
        }
        plan.candidate = branch;
        break;
    }
    if (plan.candidate == nullptr ||
        !collect_live_outs(plan, dom_tree)) {
        return false;
    }
    if (guarded_dynamic_trip) {
        for (auto &&live_out : plan.live_outs) {
            if (!guard_value_is_clonable(
                    live_out.value, loop.header,
                    plan.guard_substitution)) {
                return false;
            }
        }
    }
    return true;
}

void retarget_header_phi_predecessor(
    BasicBlock *header, BasicBlock *from,
    BasicBlock *to) noexcept {
    for (auto *instruction : header->instructions()) {
        if (!instruction->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(instruction);
        for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
            auto incoming = phi->incoming(i);
            if (incoming.block == from) {
                phi->set_incoming(i, incoming.value, to);
            }
        }
    }
}

void remove_phi_predecessor(
    BasicBlock *block, BasicBlock *predecessor) noexcept {
    if (block == nullptr || predecessor == nullptr) { return; }
    for (auto *instruction : block->instructions()) {
        if (!instruction->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(instruction);
        for (auto i = phi->incoming_count(); i-- > 0u;) {
            if (phi->incoming(i).block == predecessor) {
                phi->remove_incoming(i);
            }
        }
    }
}

void transform_plan(
    FunctionDefinition *definition, UnswitchPlan &plan,
    LoopUnswitchInfo &info) noexcept {
    LoopCloneValueResolver resolver;
    for (auto *block : plan.blocks) {
        auto *clone = definition->create_basic_block();
        clone_metadata(*block, *clone);
        resolver.map(block, clone);
    }

    // Materialize every Phi identity before cloning ordinary instructions so
    // loop-carried cycles and forward edge operands resolve to the clone.
    luisa::vector<std::pair<PhiInst *, PhiInst *>> phis;
    for (auto *block : plan.blocks) {
        auto *clone = static_cast<BasicBlock *>(resolver.resolve(block));
        XIRBuilder builder;
        builder.set_insertion_point(clone);
        for (auto *instruction : block->instructions()) {
            if (!instruction->isa<PhiInst>()) { break; }
            auto *source_phi = static_cast<PhiInst *>(instruction);
            auto *clone_phi = builder.phi(source_phi->type());
            clone_metadata(*source_phi, *clone_phi);
            resolver.map(source_phi, clone_phi);
            phis.emplace_back(source_phi, clone_phi);
        }
    }

    for (auto *block : plan.blocks) {
        auto *clone = static_cast<BasicBlock *>(resolver.resolve(block));
        XIRBuilder builder;
        builder.set_insertion_point(clone);
        for (auto *instruction : block->instructions()) {
            if (instruction->isa<PhiInst>()) { continue; }
            Instruction *cloned = nullptr;
            if (instruction == plan.candidate) {
                auto *false_target = static_cast<BasicBlock *>(
                    resolver.resolve(plan.candidate->false_block()));
                cloned = builder.br(false_target);
            } else {
                cloned = instruction->clone_with_metadata(
                    builder, resolver);
            }
            LUISA_ASSERT(cloned != nullptr,
                         "Loop unswitching failed to clone an instruction.");
            resolver.map(instruction, cloned);
        }
    }
    for (auto &&[source, clone] : phis) {
        for (auto i = size_t{0u}; i < source->incoming_count(); i++) {
            auto incoming = source->incoming(i);
            auto *value = resolver.resolve(incoming.value);
            auto *block = static_cast<BasicBlock *>(
                resolver.resolve(incoming.block));
            clone->add_incoming(value, block);
        }
    }
    auto *cloned_candidate_block = static_cast<BasicBlock *>(
        resolver.resolve(plan.candidate->parent_block()));
    auto *cloned_true_target = static_cast<BasicBlock *>(
        resolver.resolve(plan.candidate->true_block()));
    remove_phi_predecessor(
        cloned_true_target, cloned_candidate_block);

    auto *original_header = plan.loop.header;
    auto *cloned_header = static_cast<BasicBlock *>(
        resolver.resolve(original_header));
    auto *old_preheader = plan.loop.preheader;
    auto *true_preheader = definition->create_basic_block();
    auto *false_preheader = definition->create_basic_block();
    true_preheader->set_name("unswitch_true_preheader");
    false_preheader->set_name("unswitch_false_preheader");
    XIRBuilder builder;
    builder.set_insertion_point(true_preheader);
    builder.br(original_header);
    builder.set_insertion_point(false_preheader);
    builder.br(cloned_header);
    retarget_header_phi_predecessor(
        original_header, old_preheader, true_preheader);
    retarget_header_phi_predecessor(
        cloned_header, old_preheader, false_preheader);

    auto *old_preheader_branch = old_preheader->terminator();
    BasicBlock *entry_guard = nullptr;
    LoopCloneValueResolver guard_resolver;
    if (plan.guarded_dynamic_trip) {
        entry_guard = definition->create_basic_block();
        auto *dispatch_block = definition->create_basic_block();
        entry_guard->set_name("unswitch_entry_guard");
        dispatch_block->set_name("unswitch_guarded_dispatch");
        for (auto &&[phi, initial] : plan.guard_substitution) {
            guard_resolver.map(phi, initial);
        }

        builder.set_insertion_point(entry_guard);
        auto *guard_condition = materialize_guard_value(
            builder, guard_resolver,
            plan.guard_source->condition(), original_header);
        // Materialize every value that the new zero-trip exit edge will need
        // before terminating the guard block. The resolver caches cloned
        // header arithmetic so shared exit/live-out expressions are emitted
        // only once.
        auto *exit_source = plan.loop.exit_edges.front().first;
        for (auto *instruction : plan.guard_exit->instructions()) {
            if (!instruction->isa<PhiInst>()) { break; }
            auto *phi = static_cast<PhiInst *>(instruction);
            for (auto i = size_t{0u}; i < phi->incoming_count(); i++) {
                auto incoming = phi->incoming(i);
                if (incoming.block == exit_source) {
                    static_cast<void>(materialize_guard_value(
                        builder, guard_resolver, incoming.value,
                        original_header));
                    break;
                }
            }
        }
        for (auto &&live_out : plan.live_outs) {
            static_cast<void>(materialize_guard_value(
                builder, guard_resolver, live_out.value,
                original_header));
        }
        ConditionalBranchInst *guard_branch = nullptr;
        if (plan.guard_body_is_true_target) {
            guard_branch = builder.cond_br(
                guard_condition, dispatch_block, plan.guard_exit);
        } else {
            guard_branch = builder.cond_br(
                guard_condition, plan.guard_exit, dispatch_block);
        }
        clone_metadata(*plan.guard_source, *guard_branch);

        builder.set_insertion_point(dispatch_block);
        auto *dispatch = builder.cond_br(
            plan.candidate->condition(),
            true_preheader, false_preheader);
        clone_metadata(*plan.candidate, *dispatch);

        static_cast<BranchInst *>(old_preheader_branch)
            ->set_target_block(entry_guard);
    } else {
        builder.set_insertion_point(old_preheader_branch);
        auto *dispatch = builder.cond_br(
            plan.candidate->condition(),
            true_preheader, false_preheader);
        clone_metadata(*plan.candidate, *dispatch);
        static_cast<void>(old_preheader_branch->remove_self());
    }

    auto *original_candidate = plan.candidate;
    auto *original_candidate_block =
        original_candidate->parent_block();
    auto *dropped_false_target = original_candidate->false_block();
    builder.set_insertion_point(original_candidate);
    builder.br(original_candidate->true_block());
    static_cast<void>(original_candidate->remove_self());
    remove_phi_predecessor(
        dropped_false_target, original_candidate_block);

    auto *exit_source = plan.loop.exit_edges.front().first;
    auto *cloned_exit_source = static_cast<BasicBlock *>(
        resolver.resolve(exit_source));
    auto *exit_block = plan.loop.exit_edges.front().second;
    for (auto *instruction : exit_block->instructions()) {
        if (!instruction->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(instruction);
        auto original_incoming_count = phi->incoming_count();
        for (auto i = size_t{0u}; i < original_incoming_count; i++) {
            auto incoming = phi->incoming(i);
            if (incoming.block == exit_source) {
                phi->add_incoming(
                    resolver.resolve(incoming.value),
                    cloned_exit_source);
                if (entry_guard != nullptr) {
                    phi->add_incoming(
                        guard_resolver.resolve(incoming.value),
                        entry_guard);
                }
                break;
            }
        }
    }

    for (auto &live_out : plan.live_outs) {
        builder.set_insertion_point(
            exit_block->instructions().front()->prev());
        auto *merged = builder.phi(live_out.value->type());
        merged->add_incoming(live_out.value, exit_source);
        merged->add_incoming(
            resolver.resolve(live_out.value), cloned_exit_source);
        if (entry_guard != nullptr) {
            merged->add_incoming(
                guard_resolver.resolve(live_out.value), entry_guard);
        }
        for (auto *use : live_out.ordinary_uses) {
            User::set_operand_use_value(use, merged);
        }
        info.merged_live_out_count++;
    }

    info.unswitched_loop_count++;
    info.guarded_dynamic_loop_count += plan.guarded_dynamic_trip;
    info.cloned_block_count += plan.blocks.size();
    info.cloned_instruction_count += plan.instruction_count;
    info.created_preheader_count += 2u;
}

void run_on_definition(
    FunctionDefinition *definition,
    const LoopUnswitchOptions &options,
    LoopUnswitchInfo &info) noexcept {
    if (definition == nullptr ||
        options.max_unswitched_loop_count == 0u) {
        return;
    }
    if (contains_structured_control_flow(definition)) {
        info.structured_cfg_error_count++;
        LUISA_WARNING_WITH_LOCATION(
            "Loop unswitching rejected structured CFG; run "
            "destructure_cfg first. IR was left unchanged.");
        return;
    }
    auto initial_count = info.unswitched_loop_count;
    while (info.unswitched_loop_count - initial_count <
           options.max_unswitched_loop_count) {
        auto dom_tree = compute_dom_tree(definition);
        auto loops = discover_natural_loops(definition, dom_tree);
        auto transformed = false;
        for (auto &loop : loops) {
            UnswitchPlan plan;
            if (make_plan(
                    definition, loop, loops, dom_tree,
                    options, plan)) {
                transform_plan(definition, plan, info);
                transformed = true;
                break;
            }
        }
        if (!transformed) { break; }
    }
}

[[nodiscard]] bool preflight_module(
    Module *module, LoopUnswitchInfo &info) noexcept {
    if (module == nullptr) { return true; }
    for (auto *function : module->function_list()) {
        auto *definition = function == nullptr ?
                               nullptr :
                               function->definition();
        if (definition != nullptr &&
            contains_structured_control_flow(definition)) {
            info.structured_cfg_error_count++;
        }
    }
    if (info.structured_cfg_error_count != 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "Loop unswitching rejected a module containing structured CFG; "
            "run destructure_cfg first. The entire module was left unchanged.");
        return false;
    }
    return true;
}

}

}// namespace detail

LoopUnswitchInfo loop_unswitch_pass_run_on_function(
    Function *function, LoopUnswitchOptions options) noexcept {
    LoopUnswitchInfo info;
    detail::run_on_definition(
        function == nullptr ? nullptr : function->definition(),
        options, info);
    return info;
}

LoopUnswitchInfo loop_unswitch_pass_run_on_module(
    Module *module, LoopUnswitchOptions options,
    PassReport *report) noexcept {
    LoopUnswitchInfo info;
    if (detail::preflight_module(module, info) && module != nullptr) {
        for (auto *function : module->function_list()) {
            detail::run_on_definition(
                function == nullptr ?
                    nullptr :
                    function->definition(),
                options, info);
        }
    }
    if (report != nullptr) {
        report->set("unswitched-loop", info.unswitched_loop_count);
        report->set(
            "guarded-dynamic-loop",
            info.guarded_dynamic_loop_count);
        report->set("cloned-block", info.cloned_block_count);
        report->set(
            "cloned-instruction", info.cloned_instruction_count);
        report->set(
            "created-preheader", info.created_preheader_count);
        report->set("merged-live-out", info.merged_live_out_count);
        report->set(
            "structured-cfg-error", info.structured_cfg_error_count);
    }
    return info;
}

}// namespace luisa::compute::xir
