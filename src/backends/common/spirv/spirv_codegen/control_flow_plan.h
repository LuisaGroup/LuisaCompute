#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {
class BasicBlock;
class FunctionDefinition;
class Instruction;
class Value;
class IfInst;
class LoopInst;
class PhiInst;
class SimpleLoopInst;
class SwitchInst;
}// namespace luisa::compute::xir

namespace lc::spirv {

enum class SpirvLoopPrepareKind : uint8_t {
    INVALID,
    UNCONDITIONAL,
    CONDITIONAL,
};

struct SpirvLoopPreparePlan {
    SpirvLoopPrepareKind kind{SpirvLoopPrepareKind::INVALID};
    luisa::string diagnostic;

    [[nodiscard]] bool succeeded() const noexcept {
        return kind != SpirvLoopPrepareKind::INVALID;
    }
};

[[nodiscard]] SpirvLoopPreparePlan plan_spirv_loop_prepare(
    const luisa::compute::xir::LoopInst *loop) noexcept;

/// Immutable, backend-facing description of normalized XIR control flow.
///
/// The plan contains no SPIR-V objects. It is built and validated before any
/// physical block IDs are allocated, so emission cannot discover or redirect
/// logical edges while it is mutating the target module.
class ControlFlowPlan final {
public:
    struct FunctionEntryBoundaryValidation {
        size_t logical_predecessor_count{0u};
        size_t phi_count{0u};

        [[nodiscard]] bool succeeded() const noexcept {
            return logical_predecessor_count == 0u && phi_count == 0u;
        }
    };

    // One record per reachable predecessor of a physical loop header: an XIR
    // Loop.prepare block, a synthetic SimpleLoop header, or a cyclic Switch
    // header. These facts deliberately describe the final physical graph,
    // after merge proxies and continue trampolines have been inserted.
    struct PhysicalLoopPredecessorFacts {
        bool dominated_by_header{false};
        bool dominated_by_continue_target{false};
        bool dominated_by_merge_target{false};
    };

    struct PhysicalLoopBoundaryValidation {
        size_t reachable_predecessor_count{0u};
        size_t entry_edge_count{0u};
        size_t backedge_edge_count{0u};
        bool backedge_dominated_by_continue_target{false};
        bool backedge_dominated_by_merge_target{false};

        [[nodiscard]] bool succeeded() const noexcept {
            return entry_edge_count == 1u && backedge_edge_count == 1u &&
                   backedge_dominated_by_continue_target &&
                   !backedge_dominated_by_merge_target;
        }
    };

    struct FunctionPhysicalLoopBoundaryValidation {
        luisa::string planning_diagnostic;
        luisa::vector<PhysicalLoopBoundaryValidation> loops;

        [[nodiscard]] bool planning_succeeded() const noexcept {
            return planning_diagnostic.empty();
        }
        [[nodiscard]] bool succeeded() const noexcept {
            if (!planning_succeeded()) { return false; }
            for (auto &&loop : loops) {
                if (!loop.succeeded()) { return false; }
            }
            return true;
        }
    };

    enum class BlockRole : uint32_t {
        NONE = 0u,
        FUNCTION_ENTRY = 1u << 0u,
        IF_HEADER = 1u << 1u,
        IF_TRUE_ENTRY = 1u << 2u,
        IF_FALSE_ENTRY = 1u << 3u,
        SELECTION_MERGE = 1u << 4u,
        LOOP_OWNER = 1u << 5u,
        LOOP_PREPARE = 1u << 6u,
        LOOP_BODY = 1u << 7u,
        LOOP_UPDATE = 1u << 8u,
        LOOP_MERGE = 1u << 9u,
        SIMPLE_LOOP_OWNER = 1u << 10u,
        SIMPLE_LOOP_BODY = 1u << 11u,
        SIMPLE_LOOP_MERGE = 1u << 12u,
        SWITCH_HEADER = 1u << 13u,
        SWITCH_CASE_ENTRY = 1u << 14u,
        SWITCH_DEFAULT_ENTRY = 1u << 15u,
        SWITCH_MERGE = 1u << 16u,
    };

    enum class SyntheticBlockKind : uint8_t {
        SIMPLE_LOOP_HEADER,
        SIMPLE_LOOP_CONTINUE,
        SWITCH_DISPATCH,
        SWITCH_CONTINUE,
        EDGE_TRAMPOLINE,
    };

    struct Target {
        enum class Kind : uint8_t {
            XIR_BLOCK,
            SYNTHETIC_BLOCK,
        };
        Kind kind{Kind::XIR_BLOCK};
        const luisa::compute::xir::BasicBlock *xir_block{nullptr};
        size_t synthetic_index{0u};

        [[nodiscard]] static Target xir(const luisa::compute::xir::BasicBlock *block) noexcept;
        [[nodiscard]] static Target synthetic(size_t index) noexcept;
        [[nodiscard]] bool operator==(const Target &) const noexcept = default;
    };

    struct BlockPlan {
        const luisa::compute::xir::BasicBlock *block{nullptr};
        uint32_t roles{0u};
        size_t schedule_index{0u};
        // A loop-boundary guard may bypass an exclusive chain of empty
        // Branch/Break/Continue proxy blocks. Keep those logical blocks in
        // the frozen XIR schedule for analysis, but terminate their unreachable
        // physical counterparts with OpUnreachable so they cannot introduce
        // stale predecessors into the structured SPIR-V graph.
        bool physically_pruned{false};

        [[nodiscard]] bool has_role(BlockRole role) const noexcept {
            return (roles & static_cast<uint32_t>(role)) != 0u;
        }
    };

    struct SyntheticBlockPlan {
        SyntheticBlockKind kind{SyntheticBlockKind::EDGE_TRAMPOLINE};
        const luisa::compute::xir::Instruction *owner{nullptr};
        size_t ordinal{0u};
        Target continuation{};
    };

    struct IfRegion {
        const luisa::compute::xir::IfInst *instruction{nullptr};
        const luisa::compute::xir::BasicBlock *header{nullptr};
        Target true_target{};
        Target false_target{};
        Target merge_target{};
        bool emit_selection_merge{true};
        const luisa::compute::xir::BasicBlock *
            loop_boundary_exit_target{nullptr};
        // Last logical proxy block on the bypassed exit path. A Phi incoming
        // attributed to this block is physically produced by the If header.
        const luisa::compute::xir::BasicBlock *
            loop_boundary_exit_predecessor{nullptr};
    };

    struct LoopRegion {
        const luisa::compute::xir::LoopInst *instruction{nullptr};
        const luisa::compute::xir::BasicBlock *owner{nullptr};
        const luisa::compute::xir::BasicBlock *prepare{nullptr};
        const luisa::compute::xir::BasicBlock *body{nullptr};
        const luisa::compute::xir::BasicBlock *update{nullptr};
        const luisa::compute::xir::BasicBlock *merge{nullptr};
        Target entry_target{};
        Target body_target{};
        Target continue_target{};
        Target merge_target{};
        SpirvLoopPrepareKind prepare_kind{
            SpirvLoopPrepareKind::INVALID};
        size_t physical_header_predecessor_count{0u};
        PhysicalLoopBoundaryValidation physical_boundary{};
    };

    struct SimpleLoopRegion {
        const luisa::compute::xir::SimpleLoopInst *instruction{nullptr};
        const luisa::compute::xir::BasicBlock *owner{nullptr};
        const luisa::compute::xir::BasicBlock *body{nullptr};
        const luisa::compute::xir::BasicBlock *merge{nullptr};
        Target merge_target{};
        size_t header_synthetic_index{0u};
        size_t continue_synthetic_index{0u};
        size_t physical_header_predecessor_count{0u};
        PhysicalLoopBoundaryValidation physical_boundary{};
    };

    struct SwitchRegion {
        const luisa::compute::xir::SwitchInst *instruction{nullptr};
        const luisa::compute::xir::BasicBlock *header{nullptr};
        luisa::vector<Target> case_targets;
        // OpSwitch case operands may be reordered without changing selector
        // semantics. The order below groups duplicate targets and satisfies the
        // SPIR-V case-fallthrough adjacency rules.
        luisa::vector<size_t> case_operand_order;
        // Logical operands that name an enclosing structured exit. Their
        // physical OpSwitch target is a one-way case-exit trampoline, so the
        // enclosing loop continue/merge block does not itself become a case
        // construct.
        luisa::unordered_set<const luisa::compute::xir::BasicBlock *>
            direct_exit_targets;
        Target default_target{};
        Target merge_target{};
        bool loop_wrapped{false};
        size_t dispatch_synthetic_index{0u};
        size_t continue_synthetic_index{0u};
        // A direct OpSwitch operand cannot name the continue block itself: that
        // would make the continue block a case construct whose backedge to the
        // loop header is not a legal selection exit. Route it through a case
        // exit trampoline first.
        bool has_header_case_target{false};
        size_t header_case_synthetic_index{0u};
        Target loop_merge_target{};
        size_t physical_header_predecessor_count{0u};
        PhysicalLoopBoundaryValidation physical_boundary{};
    };

    struct PhiIncomingPlan {
        const luisa::compute::xir::Value *value{nullptr};
        const luisa::compute::xir::BasicBlock *predecessor{nullptr};
        // Synthetic forwarding blocks traversed by this incoming edge, ordered
        // from the physical predecessor tail toward the result block.
        luisa::vector<size_t> forwarding_synthetic_indices;
    };

    struct PhiPlan {
        const luisa::compute::xir::PhiInst *instruction{nullptr};
        const luisa::compute::xir::BasicBlock *logical_block{nullptr};
        Target result_target{};
        luisa::vector<PhiIncomingPlan> incomings;
    };

    // Normalized XIR may represent a nested selection exit as
    // outer-merge -> inner-merge, with an inner arm reaching the outer merge
    // either directly or through a chain of one-way forwarding blocks. SPIR-V
    // must nest those physical merge roles in the opposite order. The logical
    // blocks keep their payloads and edge order; only the two declarations and
    // final entry edges exchange roles.
    struct NestedSelectionMergeRotation {
        const luisa::compute::xir::Instruction *outer_instruction{nullptr};
        const luisa::compute::xir::Instruction *inner_instruction{nullptr};
        const luisa::compute::xir::BasicBlock *outer_logical_merge{nullptr};
        const luisa::compute::xir::BasicBlock *inner_logical_merge{nullptr};
        Target outer_physical_merge{};
        Target inner_physical_merge{};
    };

private:
    const luisa::compute::xir::FunctionDefinition *_function{nullptr};
    luisa::vector<BlockPlan> _blocks;
    luisa::vector<SyntheticBlockPlan> _synthetic_blocks;
    luisa::vector<IfRegion> _if_regions;
    luisa::vector<LoopRegion> _loop_regions;
    luisa::vector<SimpleLoopRegion> _simple_loop_regions;
    luisa::vector<SwitchRegion> _switch_regions;
    luisa::vector<PhiPlan> _phi_plans;
    luisa::vector<NestedSelectionMergeRotation> _nested_selection_merge_rotations;
    luisa::unordered_map<const luisa::compute::xir::BasicBlock *, size_t> _block_indices;
    luisa::unordered_map<const luisa::compute::xir::IfInst *, size_t> _if_indices;
    luisa::unordered_map<const luisa::compute::xir::LoopInst *, size_t> _loop_indices;
    luisa::unordered_map<const luisa::compute::xir::SimpleLoopInst *, size_t> _simple_loop_indices;
    luisa::unordered_map<const luisa::compute::xir::SwitchInst *, size_t> _switch_indices;
    luisa::unordered_map<const luisa::compute::xir::PhiInst *, size_t> _phi_indices;
    luisa::unordered_map<const luisa::compute::xir::BasicBlock *, size_t> _loop_prepare_indices;
    luisa::unordered_map<const luisa::compute::xir::BasicBlock *, size_t> _loop_update_indices;
    luisa::unordered_map<const luisa::compute::xir::BasicBlock *, size_t> _simple_loop_body_indices;
    luisa::unordered_map<const luisa::compute::xir::BasicBlock *, size_t> _wrapped_switch_header_indices;
    luisa::unordered_map<
        const luisa::compute::xir::BasicBlock *,
        luisa::unordered_set<const luisa::compute::xir::BasicBlock *>>
        _wrapped_switch_backedge_sources;
    luisa::unordered_map<const luisa::compute::xir::BasicBlock *, const luisa::compute::xir::Instruction *> _merge_owners;
    luisa::unordered_map<const luisa::compute::xir::BasicBlock *, Target> _merge_targets;
    luisa::unordered_map<
        const luisa::compute::xir::BasicBlock *,
        luisa::unordered_set<const luisa::compute::xir::BasicBlock *>>
        _merge_scopes;
    luisa::unordered_map<const luisa::compute::xir::Instruction *, Target> _edge_targets;
    luisa::unordered_map<const luisa::compute::xir::Instruction *, size_t>
        _nested_selection_rotation_inner_indices;
    luisa::unordered_map<const luisa::compute::xir::Instruction *, Target>
        _nested_selection_rotation_entry_edge_targets;
    luisa::unordered_map<
        const luisa::compute::xir::BasicBlock *,
        const luisa::compute::xir::BasicBlock *>
        _nested_selection_merge_forward_targets;
    luisa::string _planning_diagnostic;

private:
    void _add_role(const luisa::compute::xir::BasicBlock *block, BlockRole role) noexcept;
    void _register_merge(const luisa::compute::xir::BasicBlock *block,
                         const luisa::compute::xir::Instruction *owner) noexcept;
    [[nodiscard]] size_t _add_synthetic(SyntheticBlockKind kind,
                                        const luisa::compute::xir::Instruction *owner,
                                        Target continuation) noexcept;
    [[nodiscard]] Target _resolve_loop_boundary_target(const luisa::compute::xir::BasicBlock *block) const noexcept;
    [[nodiscard]] Target _resolve_ordinary_target(
        const luisa::compute::xir::BasicBlock *source,
        const luisa::compute::xir::BasicBlock *target) const noexcept;
    [[nodiscard]] static ControlFlowPlan _create(
        const luisa::compute::xir::FunctionDefinition *function,
        bool enforce_physical_loop_boundaries) noexcept;

public:
    [[nodiscard]] static FunctionEntryBoundaryValidation
    validate_function_entry_boundary(
        const luisa::compute::xir::FunctionDefinition *function) noexcept;
    [[nodiscard]] static PhysicalLoopBoundaryValidation
    validate_physical_loop_boundary(
        luisa::span<const PhysicalLoopPredecessorFacts> predecessors) noexcept;
    // Builds the same resolved physical graph as create(), but reports planner
    // topology failures through planning_diagnostic and returns final
    // explicit-Loop, synthetic SimpleLoop, and cyclic-Switch boundary verdicts
    // instead of terminating. Loop.prepare shape is also classified
    // nonfatally; other raw XIR preconditions remain assertions, so callers
    // must first pass the dialect/generic verifier gate.
    [[nodiscard]] static FunctionPhysicalLoopBoundaryValidation
    validate_function_physical_loop_boundaries(
        const luisa::compute::xir::FunctionDefinition *function) noexcept;
    [[nodiscard]] static ControlFlowPlan create(const luisa::compute::xir::FunctionDefinition *function) noexcept;

    [[nodiscard]] const auto *function() const noexcept { return _function; }
    [[nodiscard]] const auto &blocks() const noexcept { return _blocks; }
    [[nodiscard]] const auto &synthetic_blocks() const noexcept { return _synthetic_blocks; }
    [[nodiscard]] const auto &if_regions() const noexcept { return _if_regions; }
    [[nodiscard]] const auto &loop_regions() const noexcept { return _loop_regions; }
    [[nodiscard]] const auto &simple_loop_regions() const noexcept { return _simple_loop_regions; }
    [[nodiscard]] const auto &switch_regions() const noexcept { return _switch_regions; }
    [[nodiscard]] const auto &phi_plans() const noexcept { return _phi_plans; }
    [[nodiscard]] const auto &nested_selection_merge_rotations() const noexcept {
        return _nested_selection_merge_rotations;
    }
    [[nodiscard]] const BlockPlan &block(const luisa::compute::xir::BasicBlock *block) const noexcept;
    [[nodiscard]] const IfRegion &if_region(const luisa::compute::xir::IfInst *instruction) const noexcept;
    [[nodiscard]] const LoopRegion &loop_region(const luisa::compute::xir::LoopInst *instruction) const noexcept;
    [[nodiscard]] const SimpleLoopRegion &simple_loop_region(const luisa::compute::xir::SimpleLoopInst *instruction) const noexcept;
    [[nodiscard]] const SwitchRegion &switch_region(const luisa::compute::xir::SwitchInst *instruction) const noexcept;
    [[nodiscard]] const PhiPlan &phi_plan(const luisa::compute::xir::PhiInst *instruction) const noexcept;
    [[nodiscard]] const LoopRegion *loop_with_prepare(const luisa::compute::xir::BasicBlock *prepare) const noexcept;
    [[nodiscard]] Target edge_target(const luisa::compute::xir::Instruction *instruction) const noexcept;
};

}// namespace lc::spirv
