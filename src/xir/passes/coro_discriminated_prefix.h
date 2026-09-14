#pragma once

#include <cstddef>

#include <luisa/core/stl/memory.h>

namespace luisa::compute::xir {

class AllocaInst;
class BasicBlock;
class FunctionDefinition;
class Instruction;

namespace detail {

class CoroSemanticGraph;

struct CoroDiscriminatedPrefixProofResult {
    bool succeeded{false};
    Instruction *failing_read{nullptr};
    BasicBlock *placement_block{nullptr};
    Instruction *placement_instruction{nullptr};
    size_t candidate_count{0u};
    size_t rejected_missing_publication_count{0u};
    size_t block_evaluation_count{0u};
};

// Function-scoped analysis for conditionally initialized counted arrays.
//
// For a payload array P, a same-sized unsigned discriminant array T, and an
// unsigned published-record counter C, the abstract invariant is
//
//   U = { T[i] | 0 <= i < C and P[i] is not definitely initialized }.
//
// A payload read P[i] is legal only when i<C and the control-flow constraint
// on the co-indexed T[i] is disjoint from U. Record construction is modeled
// through the exact pre-increment ticket I=C, publication C:=C+1, last-record
// updates, rollback, and C:=0 reset transitions. Joins union possible unsafe
// tags and intersect definite ticket facts. Unsupported aliasing, counter
// mutation, or memory access fails closed.
//
// The implementation inspects existing XIR only. No source-language lifetime
// marker, renderer annotation, name convention, or additional IR entity is
// part of the proof contract.
class CoroDiscriminatedPrefixAnalysis {
private:
    class Impl;
    luisa::unique_ptr<Impl> _impl;

public:
    CoroDiscriminatedPrefixAnalysis(
        FunctionDefinition *definition,
        const CoroSemanticGraph &graph) noexcept;
    ~CoroDiscriminatedPrefixAnalysis() noexcept;
    CoroDiscriminatedPrefixAnalysis(
        CoroDiscriminatedPrefixAnalysis &&) noexcept;
    CoroDiscriminatedPrefixAnalysis &operator=(
        CoroDiscriminatedPrefixAnalysis &&) noexcept;
    CoroDiscriminatedPrefixAnalysis(
        const CoroDiscriminatedPrefixAnalysis &) = delete;
    CoroDiscriminatedPrefixAnalysis &operator=(
        const CoroDiscriminatedPrefixAnalysis &) = delete;

    [[nodiscard]] CoroDiscriminatedPrefixProofResult prove(
        AllocaInst *payload,
        BasicBlock *target,
        Instruction *insertion_instruction) noexcept;
};

}// namespace detail
}// namespace luisa::compute::xir
