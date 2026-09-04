#pragma once

#include <cstddef>
#include <cstdint>

#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class AllocaInst;
class Value;

namespace detail {

struct CoroMaskedScalarWitness {
    AllocaInst *scalar;
    uint64_t mask;
};

// A companion capacity/resource scalar R whose zero-test guards extending
// the counted prefix. The tracked invariant is C>0 || R>0.
struct CoroCounterAvailabilityWitness {
    AllocaInst *scalar;
};

// Canonical reduced ordered Boolean decision diagrams. Sets denote possible
// valuations of the tracked Boolean memory/SSA predicates. Every operation is
// an over-approximation on budget exhaustion, so loss of precision can only
// reject an optimization.
class CoroBooleanSetManager {
public:
    using Set = uint32_t;

private:
    struct Node {
        uint32_t variable;
        Set low;
        Set high;
    };

    luisa::vector<Node> _nodes;
    luisa::unordered_map<Value *, uint32_t> _variables;
    luisa::vector<Value *> _variable_values;
    luisa::unordered_map<uint64_t, luisa::vector<Set>> _node_buckets;
    luisa::unordered_map<uint64_t, Set> _union_cache;
    luisa::unordered_map<uint64_t, Set> _intersection_cache;
    bool _widened{false};

private:
    [[nodiscard]] uint32_t _variable(Value *predicate) noexcept;
    [[nodiscard]] Set _make_node(
        uint32_t variable, Set low, Set high) noexcept;
    [[nodiscard]] Set _apply(
        bool is_union, Set lhs, Set rhs) noexcept;
    [[nodiscard]] Set _forget(Set set, uint32_t variable) noexcept;

public:
    CoroBooleanSetManager() noexcept;

    [[nodiscard]] static constexpr Set empty_set() noexcept { return 0u; }
    [[nodiscard]] static constexpr Set universe() noexcept { return 1u; }
    [[nodiscard]] static constexpr bool is_empty(Set set) noexcept {
        return set == empty_set();
    }

    [[nodiscard]] Set literal(Value *predicate, bool value) noexcept;
    [[nodiscard]] Set unite(Set lhs, Set rhs) noexcept;
    [[nodiscard]] Set intersect(Set lhs, Set rhs) noexcept;
    [[nodiscard]] Set forget(Set set, Value *predicate) noexcept;
    [[nodiscard]] Set assign(
        Set set, Value *destination, Value *source,
        bool true_when_source_is,
        luisa::optional<bool> copied_constant) noexcept;

    [[nodiscard]] bool widened() const noexcept { return _widened; }
    [[nodiscard]] size_t node_count() const noexcept { return _nodes.size(); }
    [[nodiscard]] luisa::string describe(Set set) const noexcept;
};

// Abstract value of one masked unsigned-scalar projection. `nonzero_possible`
// over-approximates states where (S & M) may be nonzero;
// `nonzero_without_positive_counter` over-approximates the subset where that
// witness may coexist with C == 0. The latter is therefore the exact safety
// obligation for (S & M) != 0 => C > 0.
struct CoroMaskedScalarProjection {
    CoroBooleanSetManager::Set nonzero_possible;
    CoroBooleanSetManager::Set nonzero_without_positive_counter;
};

// Abstract value of the predicate S==0. The second set is the possible
// violation of C>0 || S>0, i.e. S==0 && C==0.
struct CoroScalarZeroProjection {
    CoroBooleanSetManager::Set zero_possible;
    CoroBooleanSetManager::Set zero_without_positive_counter;
};

// For each scalar ticket I, `less_unsafe[I]` is the set of feasible Boolean
// valuations on which I<C has not been proved; missing entries mean the whole
// feasible set. Equality I=C, last-element identity I+1=C, and the stable
// physical fact Initialized(A,I) are represented identically. The last
// identity is admitted only when C>0, so unsigned addition cannot wrap.
// Initialized is materialized only while Prefix(A,C) holds; unlike a relation
// to C it then survives later counter changes. CFG join is set union of
// feasible and unsafe valuations, hence associative, commutative and
// idempotent. A relation is usable exactly when its unsafe set is empty.
class CoroGuardedScalarRelationDomain {
private:
    using Set = CoroBooleanSetManager::Set;

    CoroBooleanSetManager *_manager;
    Set _feasible{CoroBooleanSetManager::universe()};
    Set _counter_positive_unsafe{CoroBooleanSetManager::universe()};
    Set _tail_unsafe{CoroBooleanSetManager::universe()};
    luisa::unordered_map<AllocaInst *, Set> _less_unsafe;
    luisa::unordered_map<AllocaInst *, Set> _equal_unsafe;
    luisa::unordered_map<AllocaInst *, Set> _last_unsafe;
    luisa::unordered_map<AllocaInst *, Set> _initialized_unsafe;
    luisa::unordered_map<
        AllocaInst *, luisa::unordered_map<uint64_t, Set>>
        _masked_nonzero_possible;
    luisa::unordered_map<
        AllocaInst *, luisa::unordered_map<uint64_t, Set>>
        _masked_nonzero_positive_unsafe;
    luisa::unordered_map<AllocaInst *, Set> _scalar_zero_possible;
    luisa::unordered_map<AllocaInst *, Set>
        _scalar_zero_positive_unsafe;
    luisa::unordered_set<AllocaInst *> _tracked_indices;
    luisa::unordered_set<Value *> _tracked_predicates;

private:
    [[nodiscard]] Set _unsafe(
        const luisa::unordered_map<AllocaInst *, Set> &relations,
        AllocaInst *index) const noexcept;
    [[nodiscard]] bool _refine(Set selected) noexcept;
    void _canonicalize(
        luisa::unordered_map<AllocaInst *, Set> &relations) noexcept;
    [[nodiscard]] luisa::unordered_map<uint64_t, Set>
    _masked_possibilities(AllocaInst *scalar) const noexcept;
    [[nodiscard]] luisa::unordered_map<uint64_t, Set>
    _masked_witnesses(AllocaInst *scalar) const noexcept;
public:
    explicit CoroGuardedScalarRelationDomain(
        CoroBooleanSetManager &manager,
        luisa::span<const CoroMaskedScalarWitness>
            masked_scalar_witnesses = {},
        luisa::span<const CoroCounterAvailabilityWitness>
            counter_availability_witnesses = {}) noexcept;

    [[nodiscard]] bool operator==(
        const CoroGuardedScalarRelationDomain &) const noexcept = default;

    [[nodiscard]] bool feasible() const noexcept {
        return !CoroBooleanSetManager::is_empty(_feasible);
    }
    [[nodiscard]] bool knows_less(AllocaInst *index) const noexcept;
    [[nodiscard]] bool knows_equal(AllocaInst *index) const noexcept;
    [[nodiscard]] bool knows_last(AllocaInst *index) const noexcept;
    [[nodiscard]] bool knows_counter_positive() const noexcept;
    [[nodiscard]] bool knows_tail() const noexcept;
    [[nodiscard]] bool knows_initialized(AllocaInst *index) const noexcept;
    [[nodiscard]] bool tracks_masked_scalar(
        AllocaInst *scalar) const noexcept;
    [[nodiscard]] bool masked_nonzero_implies_counter_positive(
        AllocaInst *scalar, uint64_t mask) const noexcept;
    [[nodiscard]] bool refine_masked_scalar_nonzero(
        AllocaInst *scalar, uint64_t mask) noexcept;
    void materialize_initialized() noexcept;

    void add_less(AllocaInst *index) noexcept;
    void add_equal(AllocaInst *index) noexcept;
    void add_last(AllocaInst *index) noexcept;
    void erase_index(AllocaInst *index) noexcept;
    void retain_indices(
        luisa::span<AllocaInst *const> live_indices) noexcept;
    void clear_relations() noexcept;
    void add_counter_positive() noexcept;
    void clear_counter_positive() noexcept;
    void add_tail() noexcept;
    void clear_tail() noexcept;
    void advance_counter() noexcept;
    void retreat_counter() noexcept;
    void invalidate_counter_implications() noexcept;

    void assume_masked_scalar_zero(
        AllocaInst *scalar, uint64_t zero_mask) noexcept;
    [[nodiscard]] luisa::vector<uint64_t> masked_scalar_masks(
        AllocaInst *scalar) const noexcept;
    [[nodiscard]] CoroMaskedScalarProjection
    masked_scalar_unknown_projection() const noexcept;
    [[nodiscard]] CoroMaskedScalarProjection
    masked_scalar_constant_projection(
        uint64_t value, uint64_t mask) const noexcept;
    [[nodiscard]] CoroMaskedScalarProjection
    masked_scalar_load_projection(
        AllocaInst *source, uint64_t mask) const noexcept;
    [[nodiscard]] CoroMaskedScalarProjection
    masked_scalar_projection_union(
        CoroMaskedScalarProjection lhs,
        CoroMaskedScalarProjection rhs) const noexcept;
    [[nodiscard]] CoroMaskedScalarProjection
    masked_scalar_projection_intersection(
        CoroMaskedScalarProjection lhs,
        CoroMaskedScalarProjection rhs) const noexcept;
    void assign_masked_scalar_projection(
        AllocaInst *destination, uint64_t mask,
        CoroMaskedScalarProjection projection) noexcept;

    [[nodiscard]] bool tracks_counter_availability(
        AllocaInst *scalar) const noexcept;
    [[nodiscard]] bool scalar_zero_implies_counter_positive(
        AllocaInst *scalar) const noexcept;
    [[nodiscard]] bool refine_scalar_zero(
        AllocaInst *scalar) noexcept;
    void assume_scalar_nonzero(AllocaInst *scalar) noexcept;
    [[nodiscard]] CoroScalarZeroProjection
    scalar_zero_unknown_projection() const noexcept;
    [[nodiscard]] CoroScalarZeroProjection
    scalar_zero_constant_projection(uint64_t value) const noexcept;
    [[nodiscard]] CoroScalarZeroProjection
    scalar_zero_load_projection(AllocaInst *source) const noexcept;
    [[nodiscard]] CoroScalarZeroProjection
    scalar_zero_projection_union(
        CoroScalarZeroProjection lhs,
        CoroScalarZeroProjection rhs) const noexcept;
    [[nodiscard]] CoroScalarZeroProjection
    scalar_zero_projection_intersection(
        CoroScalarZeroProjection lhs,
        CoroScalarZeroProjection rhs) const noexcept;
    void assign_scalar_zero_projection(
        AllocaInst *destination,
        CoroScalarZeroProjection projection) noexcept;

    void forget_boolean(Value *predicate) noexcept;
    void retain_booleans(
        luisa::span<Value *const> live_predicates) noexcept;
    void assign_boolean(
        Value *destination, Value *source,
        bool true_when_source_is,
        luisa::optional<bool> copied_constant) noexcept;
    [[nodiscard]] bool refine_boolean(
        Value *predicate, bool value) noexcept;

    void assign_index_copy(
        AllocaInst *destination, AllocaInst *source,
        bool source_is_counter) noexcept;

    [[nodiscard]] bool merge(
        const CoroGuardedScalarRelationDomain &incoming) noexcept;
};

}// namespace detail
}// namespace luisa::compute::xir
