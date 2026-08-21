#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/coro_alloca_scope.h>
#include <luisa/xir/passes/pointer_usage.h>

#include "coro_frame_access.h"
#include "coro_initialized_prefix.h"
#include "coro_predicate_analysis.h"
#include "coro_semantic_graph.h"

namespace luisa::compute::xir {

namespace detail {

namespace {

struct AllocaUseRegion {
    bool valid{true};
    bool contains_phi_use{false};
    luisa::unordered_set<Value *> pointers;
    luisa::unordered_set<Instruction *> users;
    luisa::vector<BasicBlock *> blocks;
};

struct InstructionLocation {
    size_t block_id;
    size_t ordinal;
};

using InstructionLocationMap =
    luisa::unordered_map<Instruction *, InstructionLocation>;

[[nodiscard]] InstructionLocationMap make_instruction_locations(
    FunctionDefinition *definition,
    const CoroSemanticGraph &graph) noexcept {
    InstructionLocationMap locations;
    for (auto *block : definition->basic_blocks()) {
        auto block_id = graph.block_id(block);
        auto ordinal = size_t{0u};
        for (auto *instruction : block->instructions()) {
            locations.emplace(
                instruction,
                InstructionLocation{block_id, ordinal++});
        }
    }
    return locations;
}

[[nodiscard]] AllocaUseRegion collect_alloca_use_region(
    AllocaInst *alloca, FunctionDefinition *definition,
    const CoroSemanticGraph &graph) noexcept {
    AllocaUseRegion result;
    luisa::unordered_set<BasicBlock *> seen_blocks;
    luisa::vector<Value *> worklist{alloca};
    while (!worklist.empty() && result.valid) {
        auto *pointer = worklist.back();
        worklist.pop_back();
        if (!result.pointers.emplace(pointer).second) { continue; }
        for (auto *use : pointer->use_list()) {
            auto *user = use == nullptr ? nullptr : use->user();
            if (user == nullptr || !user->isa<Instruction>()) {
                result.valid = false;
                break;
            }
            auto *instruction = static_cast<Instruction *>(user);
            auto *block = instruction->parent_block();
            if (block == nullptr ||
                instruction->parent_function() != definition ||
                !graph.contains(block)) {
                result.valid = false;
                break;
            }
            result.users.emplace(instruction);
            if (seen_blocks.emplace(block).second) {
                result.blocks.emplace_back(block);
            }
            if (instruction->isa<PhiInst>()) {
                result.contains_phi_use = true;
                continue;
            }
            if (instruction->isa<GEPInst>() &&
                static_cast<GEPInst *>(instruction)->base() == pointer) {
                worklist.emplace_back(instruction);
            }
        }
    }
    return result;
}

struct InsertionPoint {
    Instruction *instruction{nullptr};
    bool follows_alloca{false};
    bool has_gap_after_alloca{false};
};

[[nodiscard]] bool instruction_strictly_precedes_linear(
    Instruction *before, Instruction *after) noexcept {
    if (before == nullptr || after == nullptr ||
        before->parent_block() != after->parent_block()) {
        return false;
    }
    for (auto *instruction :
         before->parent_block()->instructions()) {
        if (instruction == before) { return true; }
        if (instruction == after) { return false; }
    }
    return false;
}

[[nodiscard]] bool instruction_immediately_precedes(
    Instruction *before, Instruction *after) noexcept {
    if (before == nullptr || after == nullptr ||
        before->parent_block() != after->parent_block()) {
        return false;
    }
    return before->next() == after;
}

[[nodiscard]] InsertionPoint find_latest_insertion_point_linear(
    BasicBlock *target, AllocaInst *alloca,
    const luisa::unordered_set<Instruction *> &users) noexcept {
    InsertionPoint result;
    auto saw_alloca = false;
    for (auto *instruction : target->instructions()) {
        if (instruction == alloca) {
            saw_alloca = true;
            continue;
        }
        if (users.contains(instruction)) {
            result.instruction = instruction;
            result.follows_alloca = saw_alloca;
            return result;
        }
        if (instruction->is_terminator()) {
            result.instruction = instruction;
            result.follows_alloca = saw_alloca;
            return result;
        }
        if (saw_alloca) { result.has_gap_after_alloca = true; }
    }
    return result;
}

// The pass freezes its candidate set before mutation and moves only the
// candidate alloca and, when proved unique, its whole-object defining store.
// Neither instruction can be an observation of a different valid candidate.
// Consequently the relative order of every pair queried for an unprocessed
// candidate is invariant under earlier contractions. Snapshot ordinals answer
// strict-order queries exactly; current intrusive-list adjacency answers the
// one property that can change when unrelated nodes are inserted in a gap.
[[nodiscard]] bool instruction_strictly_precedes(
    Instruction *before, Instruction *after,
    const InstructionLocationMap &locations,
    bool verify_instruction_order,
    size_t &instruction_order_query_count) noexcept {
    ++instruction_order_query_count;
    auto result = false;
    if (before != nullptr && after != nullptr &&
        before->parent_block() == after->parent_block()) {
        auto before_iter = locations.find(before);
        auto after_iter = locations.find(after);
        result = before_iter != locations.end() &&
                 after_iter != locations.end() &&
                 before_iter->second.block_id ==
                     after_iter->second.block_id &&
                 before_iter->second.ordinal <
                     after_iter->second.ordinal;
    }
    if (verify_instruction_order) {
        LUISA_ASSERT(
            result == instruction_strictly_precedes_linear(
                          before, after),
            "Snapshot instruction order disagrees with the current "
            "intrusive list.");
    }
    return result;
}

[[nodiscard]] InsertionPoint find_latest_insertion_point(
    BasicBlock *target, AllocaInst *alloca,
    const luisa::unordered_set<Instruction *> &users,
    const InstructionLocationMap &locations,
    bool verify_instruction_order,
    size_t &instruction_order_query_count,
    size_t &placement_user_inspection_count) noexcept {
    ++instruction_order_query_count;
    InsertionPoint result;
    if (target == nullptr) { return result; }
    auto *insertion = static_cast<Instruction *>(
        target->terminator());
    auto insertion_iter = locations.find(insertion);
    if (insertion == nullptr ||
        insertion_iter == locations.end()) {
        return result;
    }
    auto insertion_ordinal = insertion_iter->second.ordinal;
    for (auto *user : users) {
        ++placement_user_inspection_count;
        if (user == nullptr || user->parent_block() != target) {
            continue;
        }
        auto iter = locations.find(user);
        if (iter == locations.end() ||
            iter->second.block_id !=
                insertion_iter->second.block_id) {
            continue;
        }
        if (iter->second.ordinal < insertion_ordinal) {
            insertion = user;
            insertion_ordinal = iter->second.ordinal;
        }
    }
    result.instruction = insertion;
    if (alloca != nullptr && alloca->parent_block() == target) {
        auto alloca_iter = locations.find(alloca);
        result.follows_alloca =
            alloca_iter != locations.end() &&
            alloca_iter->second.block_id ==
                insertion_iter->second.block_id &&
            alloca_iter->second.ordinal < insertion_ordinal;
        result.has_gap_after_alloca =
            result.follows_alloca && alloca->next() != insertion;
    }
    if (verify_instruction_order) {
        auto oracle = find_latest_insertion_point_linear(
            target, alloca, users);
        LUISA_ASSERT(
            result.instruction == oracle.instruction &&
                result.follows_alloca == oracle.follows_alloca &&
                result.has_gap_after_alloca ==
                    oracle.has_gap_after_alloca,
            "Snapshot insertion-point query disagrees with the current "
            "intrusive list.");
    }
    return result;
}

struct FirstDefinitionPlan {
    StoreInst *definition{nullptr};
    BasicBlock *target{nullptr};
    Instruction *insertion_instruction{nullptr};
};

[[nodiscard]] FirstDefinitionPlan plan_first_definition_delay(
    AllocaInst *alloca, const AllocaUseRegion &region,
    const CoroSemanticGraph &graph,
    const InstructionLocationMap &instruction_locations,
    bool verify_instruction_order,
    size_t &instruction_order_query_count,
    size_t &placement_user_inspection_count) noexcept {
    FirstDefinitionPlan plan;
    StoreInst *definition = nullptr;
    luisa::unordered_set<Instruction *> observations;
    luisa::vector<BasicBlock *> observation_blocks;
    luisa::unordered_set<BasicBlock *> seen_blocks;

    // Formal single-definition domain:
    //   * exactly one full-object store defines the local;
    //   * every other pointer use is a typed projection or observation;
    //   * no reference/atomic/unknown operation can expose store timing.
    // Moving that store with the lifetime start may execute it more often in
    // a loop, but with no second write version every load still observes the
    // same dominating SSA value as in the original program.
    for (auto *user : region.users) {
        if (user->isa<StoreInst>()) {
            auto *store = static_cast<StoreInst *>(user);
            if (definition != nullptr || store->variable() != alloca) {
                return plan;
            }
            definition = store;
            continue;
        }
        if (!user->isa<GEPInst>() && !user->isa<LoadInst>()) {
            return plan;
        }
        observations.emplace(user);
        if (seen_blocks.emplace(user->parent_block()).second) {
            observation_blocks.emplace_back(user->parent_block());
        }
    }
    if (definition == nullptr || observations.empty()) { return plan; }

    auto *target = graph.nearest_common_dominator(
        luisa::span{observation_blocks});
    auto *source = alloca->parent_block();
    auto *definition_block = definition->parent_block();
    if (target == nullptr || source == nullptr ||
        definition_block == nullptr ||
        !graph.dominates(source, target) ||
        !graph.dominates(definition_block, target)) {
        return plan;
    }
    auto insertion = find_latest_insertion_point(
        target, alloca, observations, instruction_locations,
        verify_instruction_order, instruction_order_query_count,
        placement_user_inspection_count);
    if (insertion.instruction == nullptr) { return plan; }
    if (definition_block == target &&
        !instruction_strictly_precedes(
            definition, insertion.instruction,
            instruction_locations, verify_instruction_order,
            instruction_order_query_count)) {
        return plan;
    }
    if (source == target &&
        instruction_immediately_precedes(alloca, definition) &&
        instruction_immediately_precedes(
            definition, insertion.instruction)) {
        return plan;
    }

    // The store is moved, not recomputed. Its SSA operand must therefore be
    // available at the new point. Function arguments and constants are
    // globally available; an instruction operand must dominate the target
    // and textually precede the insertion point when defined there.
    auto *value = definition->value();
    if (value == nullptr) { return plan; }
    if (value->isa<Instruction>()) {
        auto *value_instruction = static_cast<Instruction *>(value);
        auto *value_block = value_instruction->parent_block();
        if (value_block == nullptr ||
            !graph.dominates(value_block, target) ||
            (value_block == target &&
             !instruction_strictly_precedes(
                 value_instruction, insertion.instruction,
                 instruction_locations, verify_instruction_order,
                 instruction_order_query_count))) {
            return plan;
        }
    } else if (value->derived_value_tag() !=
                   DerivedValueTag::ARGUMENT &&
               value->derived_value_tag() !=
                   DerivedValueTag::CONSTANT) {
        // A special register is evaluated at its use site and may change at a
        // continuation boundary (notably the current coroutine token). It is
        // not an SSA snapshot unless materialized by an instruction.
        return plan;
    }
    plan.definition = definition;
    plan.target = target;
    plan.insertion_instruction = insertion.instruction;
    return plan;
}

void apply_first_definition_plan(
    AllocaInst *alloca,
    const FirstDefinitionPlan &plan) noexcept {
    auto alloca_owner = alloca->remove_self();
    auto definition_owner = plan.definition->remove_self();
    auto *moved_alloca = plan.insertion_instruction->insert_before_self(
        std::move(alloca_owner));
    auto *moved_definition =
        plan.insertion_instruction->insert_before_self(
            std::move(definition_owner));
    LUISA_DEBUG_ASSERT(
        moved_alloca == alloca &&
            moved_definition == plan.definition,
        "First-definition delay changed XIR instruction identity.");
}

enum class LifetimeEventKind : uint8_t {
    redefine_pointer,
    store,
    read
};

struct LifetimeEvent {
    LifetimeEventKind kind;
    Value *pointer;
    Instruction *instruction;
};

struct LifetimeFactLayout {
    luisa::unordered_map<size_t, size_t> atom_facts;
    luisa::unordered_map<Value *, size_t> pointer_facts;
    size_t fact_count{0u};
};

struct LifetimeProofResult {
    bool succeeded{false};
    bool guarded{false};
    Instruction *failing_read{nullptr};
    size_t failing_predicate_count{0u};
    size_t block_evaluation_count{0u};
    size_t guarded_state_evaluation_count{0u};
    size_t predicate_widening_count{0u};
};

struct LifetimeProofTimings {
    double problem_ms{0.0};
    double slice_ms{0.0};
    double layout_ms{0.0};
    double events_ms{0.0};
    double event_order_ms{0.0};
    double event_transfer_ms{0.0};
    double unconditional_ms{0.0};
    double guarded_ms{0.0};
};

using LifetimeFactState = luisa::vector<uint8_t>;

struct LifetimeProofProblem {
    bool valid{false};
    size_t target_id{0u};
    LifetimeFactLayout layout;
    luisa::vector<uint8_t> active;
    luisa::vector<size_t> active_blocks;
    // Parallel to active_blocks. Keeping the event domain sparse is
    // essential after SROA: a large coroutine may contain thousands of CFG
    // blocks while one scalar local touches only a handful of them.
    luisa::vector<luisa::vector<LifetimeEvent>> events;
};

struct ReferenceArgumentEffect {
    // True iff some execution can observe the incoming object before a
    // complete definition of the observed fields. This is a May property.
    bool may_read_prior_value{true};
    // True iff every normally returning execution has completely defined the
    // object. This is a Must property. Non-returning exits do not weaken it.
    bool fully_defines_on_return{false};
};

class ReferenceArgumentEffectAnalysis {
private:
    luisa::unordered_map<Argument *, ReferenceArgumentEffect> _effects;
    luisa::unordered_set<FunctionDefinition *> _analyzed;
    size_t _analysis_count{0u};
    double _analysis_ms{0.0};

private:
    void _analyze(FunctionDefinition *definition) noexcept {
        if (definition == nullptr || definition->body_block() == nullptr ||
            !_analyzed.emplace(definition).second) {
            return;
        }
        auto begin = std::chrono::steady_clock::now();
        ++_analysis_count;
        PointerUsageAnalysisInfo analysis_info;

        const auto analyze = [&]() noexcept {
            // PointerUsageAnalysis is a field-sensitive pair of finite
            // dataflow problems. At function entry, LIVE is exactly the set
            // of aggregate leaves that may be read before a definite
            // overwrite. At each normal return, KILL is exactly the set
            // definitely written on every path to that return. Calls inside
            // the callee remain conservative opaque read/writes, so an
            // unsupported or recursive dependency can only make this summary
            // less precise, never unsound.
            luisa::vector<Value *> reference_arguments;
            for (auto *argument : definition->arguments()) {
                if (argument->is_reference()) {
                    _effects.try_emplace(
                        argument, ReferenceArgumentEffect{});
                    reference_arguments.emplace_back(argument);
                }
            }
            if (reference_arguments.empty()) { return; }

            // Pointer usage is a product lattice over pointer views. Reference
            // summaries query only formal-reference coordinates, so solving
            // those coordinates is exactly equivalent to solving every local
            // alloca/GEP view and projecting afterward. Pointer discovery and
            // malformed-use validation remain whole-function and therefore
            // fail closed.
            PointerUsageAnalysis analysis;
            analysis_info = analysis.analyze(
                definition, luisa::span<Value *const>{reference_arguments});
            if (!analysis_info.succeeded()) { return; }
            // The summary pass below is read-only. Validate the captured IR
            // version once, then query the immutable block-result table
            // directly; validating the whole instruction snapshot for every
            // argument and return block would turn extraction into
            // O(queries * instructions).
            LUISA_ASSERT(
                analysis.is_current(),
                "Fresh pointer-usage analysis is unexpectedly stale.");

            for (auto *argument : definition->arguments()) {
                if (!argument->is_reference()) { continue; }
                auto *entry_block = analysis.current_block_usage(
                    definition->body_block());
                auto entry_iter = entry_block == nullptr ?
                                      PointerUsageMap::const_iterator{} :
                                      entry_block->in.find(argument);
                if (entry_block == nullptr ||
                    entry_iter == entry_block->in.end()) {
                    continue;
                }
                auto *entry = entry_iter->second.get();

                auto has_normal_return = false;
                auto defines_at_every_return = true;
                for (auto *block : definition->basic_blocks()) {
                    if (!block->is_terminated() ||
                        !block->terminator()->isa<ReturnInst>()) {
                        continue;
                    }
                    auto *block_usage =
                        analysis.current_block_usage(block);
                    auto usage_iter = block_usage == nullptr ?
                                          PointerUsageMap::const_iterator{} :
                                          block_usage->out.find(argument);
                    if (block_usage == nullptr ||
                        usage_iter == block_usage->out.end()) {
                        continue;
                    }
                    auto *usage = usage_iter->second.get();
                    has_normal_return = true;
                    defines_at_every_return &= usage->kill.access().all();
                }
                _effects[argument] = ReferenceArgumentEffect{
                    .may_read_prior_value = entry->live.access().any(),
                    .fully_defines_on_return =
                        has_normal_return && defines_at_every_return};
            }
        };
        analyze();
        auto elapsed_ms =
            std::chrono::duration<double, std::milli>{
                std::chrono::steady_clock::now() - begin}
                .count();
        _analysis_ms += elapsed_ms;
        if (auto profile = std::getenv(
                "LUISA_CORO_PROFILE_COMPILATION");
            profile != nullptr && luisa::string_view{profile} == "1") {
            LUISA_INFO(
                "Coroutine reference-effect summary: function='{}' "
                "tracked={} materialized={} blocks={} conservative={} "
                "invalid={} time={:.3f} ms.",
                definition->name().value_or("<unnamed>"),
                analysis_info.tracked_pointer_count,
                analysis_info.materialized_pointer_count,
                analysis_info.analyzed_block_count,
                analysis_info.conservative_access_count,
                analysis_info.invalid_access_count,
                elapsed_ms);
        }
    }

public:
    [[nodiscard]] ReferenceArgumentEffect effect(
        Argument *argument) noexcept {
        if (argument == nullptr || !argument->is_reference()) {
            return {};
        }
        _analyze(argument->parent_function()->definition());
        if (auto iter = _effects.find(argument); iter != _effects.end()) {
            return iter->second;
        }
        return {};
    }

    [[nodiscard]] size_t analysis_count() const noexcept {
        return _analysis_count;
    }

    [[nodiscard]] double analysis_ms() const noexcept {
        return _analysis_ms;
    }
};

[[nodiscard]] LifetimeFactLayout make_lifetime_fact_layout(
    luisa::span<const size_t> atom_indices,
    const AllocaUseRegion &region) noexcept {
    LifetimeFactLayout layout;
    layout.atom_facts.reserve(atom_indices.size());
    layout.pointer_facts.reserve(region.pointers.size());
    for (auto atom : atom_indices) {
        layout.atom_facts.emplace(atom, layout.fact_count++);
    }
    for (auto *pointer : region.pointers) {
        layout.pointer_facts.emplace(pointer, layout.fact_count++);
    }
    return layout;
}

void redefine_pointer(Value *pointer, LifetimeFactState &state,
                      const LifetimeFactLayout &layout) noexcept {
    if (auto iter = layout.pointer_facts.find(pointer);
        iter != layout.pointer_facts.end()) {
        state[iter->second] = 0u;
    }
}

void define_pointer(Value *pointer, LifetimeFactState &state,
                    const LifetimeFactLayout &layout,
                    const CoroFrameAtomDomain &domain) noexcept {
    if (auto iter = layout.pointer_facts.find(pointer);
        iter != layout.pointer_facts.end()) {
        // A typed XIR store overwrites the complete object denoted by this
        // dynamic pointer version, even when no fixed aggregate leaf is a
        // Must target (for example array[i]).
        state[iter->second] = 1u;
    }
    for (auto access : domain.memory_accesses(pointer)) {
        if (!access.covers_atom) { continue; }
        if (auto iter = layout.atom_facts.find(access.atom_index);
            iter != layout.atom_facts.end()) {
            state[iter->second] = 1u;
        }
    }
}

[[nodiscard]] bool pointer_is_defined(
    Value *pointer, const LifetimeFactState &state,
    const LifetimeFactLayout &layout,
    const CoroFrameAtomDomain &domain) noexcept {
    if (auto iter = layout.pointer_facts.find(pointer);
        iter != layout.pointer_facts.end() && state[iter->second] != 0u) {
        return true;
    }
    auto has_relevant_atom = false;
    for (auto access : domain.memory_accesses(pointer)) {
        if (auto iter = layout.atom_facts.find(access.atom_index);
            iter != layout.atom_facts.end()) {
            has_relevant_atom = true;
            if (state[iter->second] == 0u) { return false; }
        }
    }
    // A memory observation with no fact would mean the typed projection
    // analysis failed to represent a reachable use. Reject rather than
    // treating an empty conjunction as proof.
    return has_relevant_atom;
}

[[nodiscard]] bool apply_lifetime_events(
    luisa::span<const LifetimeEvent> events,
    LifetimeFactState &state,
    const LifetimeFactLayout &layout,
    const CoroFrameAtomDomain &domain,
    bool validate_reads,
    Instruction **failing_read = nullptr) noexcept {
    for (auto event : events) {
        switch (event.kind) {
            case LifetimeEventKind::redefine_pointer:
                // A static GEP instruction may execute repeatedly with a new
                // runtime index. Its previous exact-address fact therefore
                // cannot cross this definition.
                redefine_pointer(event.pointer, state, layout);
                break;
            case LifetimeEventKind::store:
                define_pointer(event.pointer, state, layout, domain);
                break;
            case LifetimeEventKind::read:
                if (validate_reads &&
                    !pointer_is_defined(
                        event.pointer, state, layout, domain)) {
                    if (failing_read != nullptr) {
                        *failing_read = event.instruction;
                    }
                    return false;
                }
                break;
        }
    }
    return true;
}

[[nodiscard]] bool append_call_lifetime_events(
    CallInst *call, const AllocaUseRegion &region,
    ReferenceArgumentEffectAnalysis &reference_effects,
    luisa::vector<LifetimeEvent> &events) noexcept {
    auto *callee = call == nullptr ? nullptr : call->callee();
    luisa::vector<Argument *> formals;
    if (callee != nullptr) {
        for (auto *argument : callee->arguments()) {
            formals.emplace_back(argument);
        }
    }
    auto signature_matches =
        callee != nullptr && formals.size() == call->argument_count();

    // A call is one unordered aliasing event at this abstraction level. Emit
    // every possible old-value observation before any Must definition. This
    // preserves soundness when the same object (or overlapping projections) is
    // bound to multiple reference formals: a write-only formal must not hide a
    // read-before-write through an aliasing formal merely because it appears
    // earlier in the signature.
    luisa::vector<Value *> reads;
    luisa::vector<Value *> definitions;
    luisa::unordered_set<Value *> seen_reads;
    luisa::unordered_set<Value *> seen_definitions;
    auto found_pointer_operand = false;
    for (size_t i = 0u; i < call->argument_count(); ++i) {
        auto *actual = call->argument(i);
        if (!region.pointers.contains(actual)) { continue; }
        found_pointer_operand = true;
        auto effect = ReferenceArgumentEffect{};
        if (signature_matches && formals[i]->is_reference() &&
            formals[i]->type() == actual->type()) {
            effect = reference_effects.effect(formals[i]);
        }
        if (effect.may_read_prior_value &&
            seen_reads.emplace(actual).second) {
            reads.emplace_back(actual);
        }
        if (effect.fully_defines_on_return &&
            seen_definitions.emplace(actual).second) {
            definitions.emplace_back(actual);
        }
    }
    for (auto *pointer : reads) {
        events.emplace_back(
            LifetimeEventKind::read, pointer, call);
    }
    for (auto *pointer : definitions) {
        events.emplace_back(
            LifetimeEventKind::store, pointer, call);
    }
    return found_pointer_operand;
}

[[nodiscard]] bool append_ray_query_pipeline_lifetime_events(
    RayQueryPipelineInst *pipeline, const AllocaUseRegion &region,
    ReferenceArgumentEffectAnalysis &reference_effects,
    luisa::vector<LifetimeEvent> &events) noexcept {
    if (pipeline == nullptr) { return false; }

    luisa::unordered_set<Value *> seen_reads;
    auto append_read = [&](Value *pointer) noexcept {
        if (region.pointers.contains(pointer) &&
            seen_reads.emplace(pointer).second) {
            events.emplace_back(
                LifetimeEventKind::read, pointer, pipeline);
            return true;
        }
        return false;
    };

    // The query object is state consumed by the traversal itself. Candidate
    // callbacks, however, execute zero or more times in backend-defined
    // candidate order. Therefore a capture may begin a fresh lifetime iff no
    // possible handler reads its incoming value before defining it. Callback
    // writes are deliberately not Must definitions of the pipeline: a query
    // with no matching candidate executes neither handler.
    auto found_pointer_operand = append_read(pipeline->query_object());
    auto capture_count = pipeline->captured_argument_count();
    auto handlers = std::array{
        pipeline->on_surface_function(),
        pipeline->on_procedural_function()};
    luisa::vector<luisa::vector<Argument *>> handler_formals;
    handler_formals.reserve(handlers.size());
    for (auto *handler : handlers) {
        auto &formals = handler_formals.emplace_back();
        if (handler != nullptr) {
            for (auto *argument : handler->arguments()) {
                formals.emplace_back(argument);
            }
        }
    }

    for (size_t capture_index = 0u;
         capture_index < capture_count; ++capture_index) {
        auto *actual = pipeline->captured_argument(capture_index);
        if (!region.pointers.contains(actual)) { continue; }
        found_pointer_operand = true;
        auto may_read_prior_value = false;
        for (size_t handler_index = 0u;
             handler_index < handlers.size(); ++handler_index) {
            auto *handler = handlers[handler_index];
            auto &formals = handler_formals[handler_index];
            auto signature_matches =
                handler != nullptr &&
                formals.size() == capture_count + 1u;
            if (!signature_matches) {
                may_read_prior_value = true;
                break;
            }
            auto *formal = formals[capture_index + 1u];
            if (!formal->is_reference() || formal->type() != actual->type() ||
                reference_effects.effect(formal).may_read_prior_value) {
                may_read_prior_value = true;
                break;
            }
        }
        if (may_read_prior_value) {
            static_cast<void>(append_read(actual));
        }
    }
    return found_pointer_operand;
}

[[nodiscard]] bool append_instruction_lifetime_events(
    Instruction *instruction, const AllocaUseRegion &region,
    ReferenceArgumentEffectAnalysis &reference_effects,
    luisa::vector<LifetimeEvent> &events) noexcept {
    if (instruction->isa<GEPInst>()) {
        events.emplace_back(LifetimeEvent{
            LifetimeEventKind::redefine_pointer, instruction,
            instruction});
        return true;
    }
    if (instruction->isa<LoadInst>()) {
        auto *pointer =
            static_cast<LoadInst *>(instruction)->variable();
        if (!region.pointers.contains(pointer)) { return false; }
        events.emplace_back(LifetimeEvent{
            LifetimeEventKind::read, pointer, instruction});
        return true;
    }
    if (instruction->isa<StoreInst>()) {
        auto *pointer =
            static_cast<StoreInst *>(instruction)->variable();
        if (!region.pointers.contains(pointer)) { return false; }
        events.emplace_back(LifetimeEvent{
            LifetimeEventKind::store, pointer, instruction});
        return true;
    }
    if (instruction->isa<CallInst>()) {
        return append_call_lifetime_events(
            static_cast<CallInst *>(instruction), region,
            reference_effects, events);
    }
    if (instruction->isa<RayQueryPipelineInst>()) {
        return append_ray_query_pipeline_lifetime_events(
            static_cast<RayQueryPipelineInst *>(instruction),
            region, reference_effects, events);
    }
    auto found_pointer_operand = false;
    luisa::unordered_set<Value *> seen_pointers;
    for (auto *operand_use : instruction->operand_uses()) {
        auto *pointer =
            operand_use == nullptr ? nullptr : operand_use->value();
        if (!region.pointers.contains(pointer) ||
            !seen_pointers.emplace(pointer).second) {
            continue;
        }
        found_pointer_operand = true;
        // Atomics and unknown pointer operations may observe the old value
        // before any possible write. Ordinary reference calls are handled
        // above by their field-sensitive callee summary.
        events.emplace_back(LifetimeEvent{
            LifetimeEventKind::read, pointer, instruction});
    }
    return found_pointer_operand;
}

struct OrderedLifetimeUser {
    Instruction *instruction;
    InstructionLocation location;
};

[[nodiscard]] LifetimeProofProblem make_lifetime_proof_problem(
    BasicBlock *target, Instruction *insertion_instruction,
    const AllocaUseRegion &region,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain,
    ReferenceArgumentEffectAnalysis &reference_effects,
    const InstructionLocationMap &instruction_locations,
    luisa::span<const size_t> atom_indices,
    LifetimeProofTimings *timings) noexcept {
    using Clock = std::chrono::steady_clock;
    auto phase_begin = timings == nullptr ?
                           Clock::time_point{} :
                           Clock::now();
    LifetimeProofProblem problem;
    auto target_id = graph.block_id(target);
    if (target_id >= graph.block_count() ||
        insertion_instruction == nullptr) {
        return problem;
    }
    problem.target_id = target_id;

    // Restrict the proof to the reverse slice from every pointer use to the
    // proposed lifetime start. Since target dominates every use, this slice
    // contains every executable path that can affect an observation before
    // target is reached again; target itself is a reset boundary.
    problem.active.assign(graph.block_count(), 0u);
    luisa::vector<size_t> worklist;
    for (auto *block : region.blocks) {
        if (!graph.dominates(target, block)) { return problem; }
        auto id = graph.block_id(block);
        if (id >= graph.block_count()) { return problem; }
        if (problem.active[id] == 0u) {
            problem.active[id] = 1u;
            worklist.emplace_back(id);
        }
    }
    if (problem.active[target_id] == 0u) {
        problem.active[target_id] = 1u;
        worklist.emplace_back(target_id);
    }
    for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
        auto block_id = worklist[cursor];
        if (block_id == target_id) { continue; }
        for (auto predecessor : graph.predecessors(block_id)) {
            auto *predecessor_block = graph.block(predecessor);
            if (!graph.dominates(target, predecessor_block)) {
                return problem;
            }
            if (problem.active[predecessor] == 0u) {
                problem.active[predecessor] = 1u;
                worklist.emplace_back(predecessor);
            }
        }
    }
    if (timings != nullptr) {
        auto now = Clock::now();
        timings->slice_ms +=
            std::chrono::duration<double, std::milli>{
                now - phase_begin}
                .count();
        phase_begin = now;
    }

    problem.layout = make_lifetime_fact_layout(atom_indices, region);
    if (problem.layout.fact_count == 0u) { return problem; }
    // The fixed-point equations are order independent. Sort the sparse
    // reverse slice by semantic block id to retain the previous deterministic
    // traversal order without materializing one empty vector per CFG block.
    std::sort(worklist.begin(), worklist.end());
    problem.active_blocks = std::move(worklist);
    problem.events.resize(problem.active_blocks.size());
    if (timings != nullptr) {
        auto now = Clock::now();
        timings->layout_ms +=
            std::chrono::duration<double, std::milli>{
                now - phase_begin}
                .count();
        phase_begin = now;
    }
    auto insertion_iter = instruction_locations.find(
        insertion_instruction);
    if (insertion_iter == instruction_locations.end() ||
        insertion_iter->second.block_id != target_id) {
        return problem;
    }
    luisa::vector<OrderedLifetimeUser> ordered_users;
    auto event_order_elapsed = 0.0;
    ordered_users.reserve(region.users.size());
    for (auto *instruction : region.users) {
        auto iter = instruction_locations.find(instruction);
        if (iter == instruction_locations.end()) { return problem; }
        ordered_users.emplace_back(
            OrderedLifetimeUser{instruction, iter->second});
    }
    std::sort(
        ordered_users.begin(), ordered_users.end(),
        [](auto lhs, auto rhs) noexcept {
            return lhs.location.block_id < rhs.location.block_id ||
                   (lhs.location.block_id == rhs.location.block_id &&
                    lhs.location.ordinal < rhs.location.ordinal);
        });
    if (timings != nullptr) {
        auto now = Clock::now();
        event_order_elapsed =
            std::chrono::duration<double, std::milli>{
                now - phase_begin}
                .count();
        timings->event_order_ms += event_order_elapsed;
        phase_begin = now;
    }
    for (auto user : ordered_users) {
        if (user.location.block_id == target_id &&
            user.location.ordinal < insertion_iter->second.ordinal) {
            return problem;
        }
        auto block_iter = std::lower_bound(
            problem.active_blocks.begin(),
            problem.active_blocks.end(),
            user.location.block_id);
        if (block_iter == problem.active_blocks.end() ||
            *block_iter != user.location.block_id) {
            return problem;
        }
        auto active_index = static_cast<size_t>(
            block_iter - problem.active_blocks.begin());
        if (!append_instruction_lifetime_events(
                user.instruction, region, reference_effects,
                problem.events[active_index])) {
            return problem;
        }
    }
    if (timings != nullptr) {
        auto elapsed = std::chrono::duration<double, std::milli>{
            Clock::now() - phase_begin}
                           .count();
        timings->event_transfer_ms += elapsed;
        timings->events_ms += event_order_elapsed + elapsed;
    }
    problem.valid = true;
    return problem;
}

[[nodiscard]] LifetimeProofResult prove_unconditional_fresh_lifetime(
    const LifetimeProofProblem &problem,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain) noexcept {
    LifetimeProofResult result;
    if (!problem.valid) { return result; }

    auto top = LifetimeFactState(
        problem.layout.fact_count, uint8_t{1u});
    auto bottom = LifetimeFactState(
        problem.layout.fact_count, uint8_t{0u});
    luisa::vector<LifetimeFactState> in_states(graph.block_count());
    luisa::vector<LifetimeFactState> out_states(graph.block_count());
    for (size_t active_index = 0u;
         active_index < problem.active_blocks.size(); ++active_index) {
        auto block_id = problem.active_blocks[active_index];
        in_states[block_id] = top;
        out_states[block_id] = top;
    }

    // This is the greatest fixed point of the forward Must equations:
    //   IN[target] = empty
    //   IN[b]      = intersection OUT[p]
    //   OUT[b]     = GEN[b] union (IN[b] - exact-GEP-redefinitions[b]).
    // Starting at top makes every update descending, so loop convergence is
    // finite and no traversal order can invent a definite initialization.
    for (;;) {
        auto changed = false;
        for (size_t active_index = 0u;
             active_index < problem.active_blocks.size(); ++active_index) {
            auto block_id = problem.active_blocks[active_index];
            ++result.block_evaluation_count;
            LifetimeFactState next_in;
            if (block_id == problem.target_id) {
                next_in = bottom;
            } else {
                auto first_predecessor = true;
                for (auto predecessor : graph.predecessors(block_id)) {
                    if (problem.active[predecessor] == 0u) {
                        return result;
                    }
                    if (first_predecessor) {
                        next_in = out_states[predecessor];
                        first_predecessor = false;
                    } else {
                        for (size_t i = 0u;
                             i < problem.layout.fact_count; ++i) {
                            next_in[i] &= out_states[predecessor][i];
                        }
                    }
                }
                if (first_predecessor) { return result; }
            }
            auto next_out = next_in;
            static_cast<void>(apply_lifetime_events(
                problem.events[active_index], next_out,
                problem.layout, domain, false));
            if (in_states[block_id] != next_in ||
                out_states[block_id] != next_out) {
                in_states[block_id] = std::move(next_in);
                out_states[block_id] = std::move(next_out);
                changed = true;
            }
        }
        if (!changed) { break; }
    }

    for (size_t active_index = 0u;
         active_index < problem.active_blocks.size(); ++active_index) {
        auto block_id = problem.active_blocks[active_index];
        auto state = in_states[block_id];
        if (!apply_lifetime_events(
                problem.events[active_index], state,
                problem.layout, domain, true,
                &result.failing_read)) {
            return result;
        }
    }
    result.succeeded = true;
    return result;
}

struct PredicateAssignment {
    size_t predicate;
    bool value;
};

using PredicateCube = luisa::vector<PredicateAssignment>;

struct GuardedLifetimeState {
    PredicateCube cube;
    LifetimeFactState facts;
};

constexpr auto max_predicates_per_cube = 12u;
constexpr auto max_guarded_states_per_block = 64u;

[[nodiscard]] bool cube_subsumes(
    const PredicateCube &general,
    const PredicateCube &specific) noexcept {
    auto j = size_t{0u};
    for (auto literal : general) {
        while (j < specific.size() &&
               specific[j].predicate < literal.predicate) {
            ++j;
        }
        if (j == specific.size() ||
            specific[j].predicate != literal.predicate ||
            specific[j].value != literal.value) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool intersect_facts(
    LifetimeFactState &target,
    const LifetimeFactState &incoming) noexcept {
    auto changed = false;
    LUISA_DEBUG_ASSERT(target.size() == incoming.size(),
                       "Mismatched lifetime fact domains.");
    for (size_t i = 0u; i < target.size(); ++i) {
        auto next = static_cast<uint8_t>(target[i] & incoming[i]);
        changed |= next != target[i];
        target[i] = next;
    }
    return changed;
}

void kill_predicates(PredicateCube &cube,
                     luisa::span<const size_t> killed) noexcept {
    if (cube.empty() || killed.empty()) { return; }
    cube.erase(
        std::remove_if(
            cube.begin(), cube.end(),
            [killed](auto literal) noexcept {
                return std::binary_search(
                    killed.begin(), killed.end(),
                    literal.predicate);
            }),
        cube.end());
}

[[nodiscard]] bool refine_cube(
    PredicateCube &cube,
    CoroPredicateLiteral literal) noexcept {
    auto iter = std::lower_bound(
        cube.begin(), cube.end(), literal.predicate,
        [](auto existing, size_t predicate) noexcept {
            return existing.predicate < predicate;
        });
    if (iter != cube.end() &&
        iter->predicate == literal.predicate) {
        return iter->value == literal.value;
    }
    // Forgetting a new literal is a conservative widening: both edges remain
    // feasible and later Must intersections can only lose facts.
    if (cube.size() >= max_predicates_per_cube) { return true; }
    cube.emplace(iter, PredicateAssignment{
                           literal.predicate, literal.value});
    return true;
}

[[nodiscard]] bool merge_guarded_state(
    luisa::vector<GuardedLifetimeState> &states,
    GuardedLifetimeState incoming,
    size_t &widening_count) noexcept {
    auto covered = false;
    auto changed = false;
    for (auto &state : states) {
        if (cube_subsumes(state.cube, incoming.cube)) {
            covered = true;
            changed |= intersect_facts(state.facts, incoming.facts);
        }
    }
    if (covered) { return changed; }

    for (size_t i = 0u; i < states.size();) {
        if (cube_subsumes(incoming.cube, states[i].cube)) {
            static_cast<void>(intersect_facts(
                incoming.facts, states[i].facts));
            states.erase(states.begin() + i);
        } else {
            ++i;
        }
    }
    if (states.size() >= max_guarded_states_per_block) {
        for (auto &&state : states) {
            static_cast<void>(intersect_facts(
                incoming.facts, state.facts));
        }
        states.clear();
        incoming.cube.clear();
        ++widening_count;
    }
    states.emplace_back(std::move(incoming));
    return true;
}

struct GuardedTransferEvent {
    Instruction *instruction;
    const LifetimeEvent *lifetimes{nullptr};
    size_t lifetime_count{0u};
};

[[nodiscard]] luisa::vector<luisa::vector<GuardedTransferEvent>>
make_guarded_transfer_events(
    const LifetimeProofProblem &problem,
    const CoroSemanticGraph &graph,
    const CoroPredicateAnalysis &predicates) noexcept {
    luisa::vector<luisa::vector<GuardedTransferEvent>> transfers(
        problem.active_blocks.size());
    for (size_t active_index = 0u;
         active_index < problem.active_blocks.size(); ++active_index) {
        auto block_id = problem.active_blocks[active_index];
        auto *block = graph.block(block_id);
        auto lifetime_index = size_t{0u};
        auto *first_lifetime_instruction =
            problem.events[active_index].empty() ?
                nullptr :
                problem.events[active_index].front().instruction;
        auto active = block_id != problem.target_id ||
                      first_lifetime_instruction == nullptr;
        for (auto *instruction : block->instructions()) {
            if (!active && instruction == first_lifetime_instruction) {
                active = true;
            }
            if (!active) { continue; }
            auto has_kills =
                !predicates.killed_predicates(instruction).empty();
            const LifetimeEvent *lifetimes = nullptr;
            auto lifetime_count = size_t{0u};
            if (lifetime_index < problem.events[active_index].size() &&
                problem.events[active_index][lifetime_index].instruction ==
                    instruction) {
                lifetimes = &problem.events[active_index][lifetime_index];
                do {
                    ++lifetime_index;
                    ++lifetime_count;
                } while (
                    lifetime_index < problem.events[active_index].size() &&
                    problem.events[active_index][lifetime_index].instruction ==
                        instruction);
            }
            if (has_kills || lifetime_count != 0u) {
                transfers[active_index].emplace_back(
                    GuardedTransferEvent{
                        instruction, lifetimes, lifetime_count});
            }
        }
        LUISA_DEBUG_ASSERT(
            lifetime_index == problem.events[active_index].size(),
            "Failed to order lifetime events in their XIR block.");
    }
    return transfers;
}

[[nodiscard]] bool apply_guarded_transfer(
    luisa::span<const GuardedTransferEvent> events,
    GuardedLifetimeState &state,
    const CoroPredicateAnalysis &predicates,
    const LifetimeFactLayout &layout,
    const CoroFrameAtomDomain &domain,
    bool validate_reads,
    Instruction **failing_read = nullptr) noexcept {
    for (auto event : events) {
        kill_predicates(
            state.cube,
            predicates.killed_predicates(event.instruction));
        if (event.lifetime_count != 0u &&
            !apply_lifetime_events(
                luisa::span<const LifetimeEvent>{
                    event.lifetimes, event.lifetime_count},
                state.facts,
                layout, domain, validate_reads,
                failing_read)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] LifetimeProofResult prove_guarded_fresh_lifetime(
    const LifetimeProofProblem &problem,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain,
    const CoroPredicateAnalysis &predicates) noexcept {
    LifetimeProofResult result;
    result.guarded = true;
    if (!problem.valid || predicates.predicate_count() == 0u) {
        return result;
    }
    auto transfers = make_guarded_transfer_events(
        problem, graph, predicates);
    constexpr auto invalid_active_index = ~size_t{0u};
    luisa::vector<size_t> active_indices(
        graph.block_count(), invalid_active_index);
    for (size_t i = 0u; i < problem.active_blocks.size(); ++i) {
        active_indices[problem.active_blocks[i]] = i;
    }
    luisa::vector<luisa::vector<GuardedLifetimeState>>
        in_states(graph.block_count());
    in_states[problem.target_id].emplace_back(
        GuardedLifetimeState{
            .facts = LifetimeFactState(
                problem.layout.fact_count, uint8_t{0u})});

    luisa::vector<size_t> worklist{problem.target_id};
    luisa::vector<uint8_t> queued(graph.block_count(), 0u);
    queued[problem.target_id] = 1u;
    for (size_t cursor = 0u; cursor < worklist.size(); ++cursor) {
        auto block_id = worklist[cursor];
        queued[block_id] = 0u;
        auto states = in_states[block_id];
        for (auto state : states) {
            ++result.guarded_state_evaluation_count;
            // The guarded domain is a forward May partition of paths whose
            // fact component is a Must property. Merging states only
            // intersects facts, predicate widening only admits more paths,
            // and no later fixed-point update can turn a missing fact at an
            // already reached read into a definite fact. Therefore a failed
            // read is a monotone proof failure and may terminate the analysis
            // immediately. This is not a concrete counterexample after
            // widening, but rejecting the contraction remains conservative.
            // Deferring the check until convergence needlessly explores the
            // complete downstream predicate product for known-bad lifetimes.
            if (!apply_guarded_transfer(
                    transfers[active_indices[block_id]], state, predicates,
                    problem.layout, domain, true,
                    &result.failing_read)) {
                result.failing_predicate_count = state.cube.size();
                return result;
            }
            for (auto successor : graph.successors(block_id)) {
                if (successor == problem.target_id ||
                    problem.active[successor] == 0u) {
                    continue;
                }
                auto next = state;
                if (auto literal = predicates.literal_on_edge(
                        graph.block(block_id),
                        graph.block(successor));
                    literal && !refine_cube(next.cube, *literal)) {
                    continue;
                }
                if (merge_guarded_state(
                        in_states[successor], std::move(next),
                        result.predicate_widening_count) &&
                    queued[successor] == 0u) {
                    queued[successor] = 1u;
                    worklist.emplace_back(successor);
                }
            }
        }
    }

    for (size_t active_index = 0u;
         active_index < problem.active_blocks.size(); ++active_index) {
        auto block_id = problem.active_blocks[active_index];
        for (auto state : in_states[block_id]) {
            if (!apply_guarded_transfer(
                    transfers[active_index], state, predicates,
                    problem.layout, domain, true,
                    &result.failing_read)) {
                result.failing_predicate_count =
                    state.cube.size();
                return result;
            }
        }
    }
    result.succeeded = true;
    return result;
}

[[nodiscard]] LifetimeProofResult prove_fresh_lifetime(
    BasicBlock *target, Instruction *insertion_instruction,
    const AllocaUseRegion &region,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain,
    const CoroPredicateAnalysis &predicates,
    ReferenceArgumentEffectAnalysis &reference_effects,
    const InstructionLocationMap &instruction_locations,
    luisa::span<const size_t> atom_indices,
    LifetimeProofTimings *timings) noexcept {
    using Clock = std::chrono::steady_clock;
    auto begin = timings == nullptr ?
                     Clock::time_point{} :
                     Clock::now();
    auto problem = make_lifetime_proof_problem(
        target, insertion_instruction, region, graph,
        domain, reference_effects, instruction_locations,
        atom_indices, timings);
    if (timings != nullptr) {
        auto now = Clock::now();
        timings->problem_ms +=
            std::chrono::duration<double, std::milli>{now - begin}.count();
        begin = now;
    }
    auto unconditional = prove_unconditional_fresh_lifetime(
        problem, graph, domain);
    if (timings != nullptr) {
        auto now = Clock::now();
        timings->unconditional_ms +=
            std::chrono::duration<double, std::milli>{now - begin}.count();
        begin = now;
    }
    if (unconditional.succeeded) { return unconditional; }
    auto guarded = prove_guarded_fresh_lifetime(
        problem, graph, domain, predicates);
    if (timings != nullptr) {
        timings->guarded_ms +=
            std::chrono::duration<double, std::milli>{
                Clock::now() - begin}
                .count();
    }
    guarded.block_evaluation_count =
        unconditional.block_evaluation_count;
    return guarded;
}

}

}// namespace detail

CoroAllocaScopeInfo coro_alloca_scope_pass_run_on_function(
    Function *function,
    const CoroAllocaScopeOptions &options) noexcept {
    CoroAllocaScopeInfo info;
    auto *definition =
        function == nullptr ? nullptr : function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        return info;
    }
    detail::CoroSemanticGraph graph{definition};
    if (!graph.valid()) {
        info.invalid_semantic_cfg_count = 1u;
        return info;
    }
    info.semantic_block_count = graph.block_count();
    info.semantic_edge_count = graph.edge_count();
    detail::CoroPredicateAnalysis predicates{graph};
    detail::ReferenceArgumentEffectAnalysis reference_effects;
    auto instruction_locations =
        detail::make_instruction_locations(definition, graph);
    const auto dump_scope_rejections = []() noexcept {
        if (auto value = std::getenv(
                "LUISA_CORO_DUMP_ALLOCA_SCOPE")) {
            return luisa::string_view{value} == "1";
        }
        return false;
    }();
    const auto profile_compilation = []() noexcept {
        if (auto value = std::getenv(
                "LUISA_CORO_PROFILE_COMPILATION")) {
            return luisa::string_view{value} == "1";
        }
        return false;
    }();
    using ProfileClock = std::chrono::steady_clock;
    using ProfileTick = ProfileClock::time_point;
    const auto profile_begin = [profile_compilation]() noexcept {
        return profile_compilation ?
                   ProfileClock::now() :
                   ProfileTick{};
    };
    const auto profile_elapsed_ms =
        [profile_compilation](ProfileTick begin) noexcept {
            return profile_compilation ?
                       std::chrono::duration<double, std::milli>{
                           ProfileClock::now() - begin}
                           .count() :
                       0.0;
        };
    auto collect_region_ms = 0.0;
    auto first_definition_ms = 0.0;
    auto placement_ms = 0.0;
    auto proof_ms = 0.0;
    auto mutation_ms = 0.0;
    detail::LifetimeProofTimings proof_timings;

    // Reuse the same type-shaped May/Must partition as coroutine liveness.
    // The definite-initialization proof and the eventual frame transfer must
    // agree on exactly which static subaggregates a store covers.
    detail::CoroFrameAtomDomain frame_domain{definition};
    luisa::unordered_map<AllocaInst *, luisa::vector<size_t>> alloca_atoms;
    for (size_t i = 0u; i < frame_domain.size(); ++i) {
        auto *root = frame_domain.atom(i).root;
        if (root != nullptr && root->isa<AllocaInst>()) {
            alloca_atoms[static_cast<AllocaInst *>(root)].emplace_back(i);
        }
    }

    // Freeze the candidate set before moving intrusive-list nodes.
    luisa::vector<AllocaInst *> allocas;
    for (auto *block : definition->basic_blocks()) {
        for (auto *instruction : block->instructions()) {
            if (instruction->isa<AllocaInst>() &&
                static_cast<AllocaInst *>(instruction)->is_local()) {
                allocas.emplace_back(
                    static_cast<AllocaInst *>(instruction));
            }
        }
    }

    for (auto *alloca : allocas) {
        ++info.scanned_local_alloca_count;
        auto phase_begin = profile_begin();
        auto region = detail::collect_alloca_use_region(
            alloca, definition, graph);
        collect_region_ms += profile_elapsed_ms(phase_begin);
        if (!region.valid) {
            ++info.rejected_unreachable_use_count;
            continue;
        }
        if (region.contains_phi_use) {
            ++info.rejected_phi_use_count;
            continue;
        }
        if (region.blocks.empty()) { continue; }

        phase_begin = profile_begin();
        auto first_definition =
            detail::plan_first_definition_delay(
                alloca, region, graph, instruction_locations,
                options.verify_instruction_order,
                info.instruction_order_query_count,
                info.placement_user_inspection_count);
        first_definition_ms += profile_elapsed_ms(phase_begin);
        if (first_definition.definition != nullptr) {
            auto *source = alloca->parent_block();
            phase_begin = profile_begin();
            detail::apply_first_definition_plan(
                alloca, first_definition);
            mutation_ms += profile_elapsed_ms(phase_begin);
            ++info.contracted_alloca_count;
            ++info.delayed_first_definition_count;
            if (source == first_definition.target) {
                ++info.intra_block_contraction_count;
                ++info.intra_block_first_definition_delay_count;
            } else {
                ++info.cross_block_contraction_count;
                ++info.cross_block_first_definition_delay_count;
            }
            continue;
        }

        phase_begin = profile_begin();
        auto *target = graph.nearest_common_dominator(
            luisa::span{region.blocks});
        auto *source = alloca->parent_block();
        if (target == nullptr || source == nullptr ||
            !graph.dominates(source, target)) {
            ++info.rejected_non_dominating_alloca_count;
            continue;
        }
        auto insertion = detail::find_latest_insertion_point(
            target, alloca, region.users,
            instruction_locations,
            options.verify_instruction_order,
            info.instruction_order_query_count,
            info.placement_user_inspection_count);
        placement_ms += profile_elapsed_ms(phase_begin);
        if (insertion.instruction == nullptr) {
            ++info.rejected_unreachable_use_count;
            continue;
        }
        if (source == target) {
            if (!insertion.follows_alloca) {
                ++info.rejected_non_dominating_alloca_count;
                continue;
            }
            if (!insertion.has_gap_after_alloca) { continue; }
        } else {
            auto atom_iter = alloca_atoms.find(alloca);
            auto atom_indices = atom_iter == alloca_atoms.end() ?
                                    luisa::span<const size_t>{} :
                                    luisa::span<const size_t>{
                                        atom_iter->second};
            phase_begin = profile_begin();
            auto proof = detail::prove_fresh_lifetime(
                target, insertion.instruction, region, graph,
                frame_domain, predicates, reference_effects,
                instruction_locations,
                atom_indices,
                profile_compilation ? &proof_timings : nullptr);
            proof_ms += profile_elapsed_ms(phase_begin);
            info.definite_initialization_block_evaluation_count +=
                proof.block_evaluation_count;
            info.guarded_initialization_state_evaluation_count +=
                proof.guarded_state_evaluation_count;
            info.predicate_widening_count +=
                proof.predicate_widening_count;
            if (!proof.succeeded) {
                phase_begin = profile_begin();
                auto prefix_proof = detail::
                    prove_initialized_prefix_fresh_lifetime(
                        alloca, target, insertion.instruction,
                        graph, frame_domain);
                proof_ms += profile_elapsed_ms(phase_begin);
                info.initialized_prefix_block_evaluation_count +=
                    prefix_proof.block_evaluation_count;
                if (prefix_proof.succeeded) {
                    proof.succeeded = true;
                    proof.guarded = false;
                    proof.failing_read = nullptr;
                    ++info.initialized_prefix_proof_count;
                } else if (prefix_proof.failing_read != nullptr) {
                    proof.failing_read = prefix_proof.failing_read;
                }
            }
            if (!proof.succeeded) {
                if (dump_scope_rejections) {
                    const auto alloca_name =
                        alloca->name().value_or("<unnamed>");
                    const auto read_name =
                        proof.failing_read == nullptr ?
                            luisa::string_view{"<none>"} :
                            proof.failing_read->name().value_or("<unnamed>");
                    const auto read_kind =
                        proof.failing_read == nullptr ?
                            luisa::string_view{"<none>"} :
                            to_string(proof.failing_read->derived_instruction_tag());
                    const auto callee_name =
                        proof.failing_read != nullptr &&
                                proof.failing_read->isa<CallInst>() &&
                                static_cast<CallInst *>(proof.failing_read)
                                        ->callee() != nullptr ?
                            static_cast<CallInst *>(proof.failing_read)
                                ->callee()
                                ->name()
                                .value_or("<unnamed>") :
                            luisa::string_view{"<none>"};
                    LUISA_INFO(
                        "Coroutine alloca lifetime rejection: name='{}' "
                        "type={} source_block={} target_block={} atoms={} "
                        "pointers={} users={} use_blocks={} failing_read='{}' "
                        "failing_kind={} callee='{}' guard_predicates={}.",
                        alloca_name,
                        alloca->type()->description(),
                        graph.block_id(source),
                        graph.block_id(target),
                        atom_indices.size(), region.pointers.size(),
                        region.users.size(), region.blocks.size(),
                        read_name, read_kind, callee_name,
                        proof.failing_predicate_count);
                }
                ++info.rejected_prior_lifetime_observation_count;
                continue;
            }
            ++info.definite_initialization_proof_count;
            if (proof.guarded) {
                ++info.guarded_initialization_proof_count;
            }
        }

        phase_begin = profile_begin();
        auto owned = alloca->remove_self();
        auto *moved = insertion.instruction->insert_before_self(
            std::move(owned));
        LUISA_DEBUG_ASSERT(moved == alloca,
                           "Alloca scope contraction changed identity.");
        mutation_ms += profile_elapsed_ms(phase_begin);
        ++info.contracted_alloca_count;
        if (source == target) {
            ++info.intra_block_contraction_count;
        } else {
            ++info.cross_block_contraction_count;
        }
    }
    if (profile_compilation) {
        LUISA_INFO(
            "Coroutine alloca lifetime timing: region={:.3f} ms "
            "first_definition={:.3f} ms placement={:.3f} ms "
            "order_queries={} user_inspections={} "
            "proof={:.3f} ms (problem={:.3f} ms: slice={:.3f} ms "
            "layout={:.3f} ms events={:.3f} ms (order={:.3f} ms "
            "transfer={:.3f} ms, reference_effects={} functions/{:.3f} ms); "
            "unconditional={:.3f} ms "
            "guarded={:.3f} ms) mutation={:.3f} ms.",
            collect_region_ms, first_definition_ms, placement_ms,
            info.instruction_order_query_count,
            info.placement_user_inspection_count,
            proof_ms, proof_timings.problem_ms, proof_timings.slice_ms,
            proof_timings.layout_ms, proof_timings.events_ms,
            proof_timings.event_order_ms,
            proof_timings.event_transfer_ms,
            reference_effects.analysis_count(),
            reference_effects.analysis_ms(),
            proof_timings.unconditional_ms, proof_timings.guarded_ms,
            mutation_ms);
    }
    return info;
}

}// namespace luisa::compute::xir
