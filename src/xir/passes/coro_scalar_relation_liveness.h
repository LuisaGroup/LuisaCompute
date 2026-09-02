#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class AllocaInst;
class Instruction;
class Value;

namespace detail {

class CoroSemanticGraph;

using CoroScalarSemanticUses = luisa::unordered_map<
    Instruction *, luisa::vector<AllocaInst *>>;

// Backward liveness for the scalar slots that may carry counted-array ticket
// relations. The analysis uses the coroutine semantic CFG and treats direct
// stores as definitions and direct loads as uses. Its only optimization role
// is to discard dead relational facts; an omitted death can cost memory but
// cannot create a proof.
class CoroScalarRelationLiveness {
private:
    luisa::vector<luisa::vector<AllocaInst *>> _live_in;
    luisa::unordered_map<
        Instruction *, luisa::vector<AllocaInst *>>
        _dead_after;

public:
    CoroScalarRelationLiveness(
        const CoroSemanticGraph &graph,
        luisa::span<const uint8_t> active_blocks,
        size_t lifetime_target,
        luisa::span<AllocaInst *const> slots,
        const CoroScalarSemanticUses &semantic_uses) noexcept;

    [[nodiscard]] luisa::span<AllocaInst *const>
    live_in(size_t block_id) const noexcept;

    [[nodiscard]] luisa::span<AllocaInst *const>
    dead_after(Instruction *instruction) const noexcept;
};

using CoroBooleanSemanticValues = luisa::unordered_map<
    Instruction *, luisa::vector<Value *>>;

// Backward liveness for Boolean predicates used by the guarded scalar-relation
// domain. A predicate is retained only while a later branch or Boolean copy
// can observe its current value. Projecting a dead predicate existentially
// from both feasible and unsafe valuation sets preserves soundness while
// preventing unrelated control-flow history from growing the ROBDD.
class CoroBooleanPredicateLiveness {
private:
    luisa::vector<luisa::vector<Value *>> _live_in;

public:
    CoroBooleanPredicateLiveness(
        const CoroSemanticGraph &graph,
        luisa::span<const uint8_t> active_blocks,
        size_t lifetime_target,
        luisa::span<Value *const> predicates,
        const CoroBooleanSemanticValues &uses,
        const CoroBooleanSemanticValues &definitions) noexcept;

    [[nodiscard]] luisa::span<Value *const>
    live_in(size_t block_id) const noexcept;
};

}// namespace detail
}// namespace luisa::compute::xir
