#pragma once

#include <cstddef>

#include <luisa/core/stl/memory.h>

namespace luisa::compute::xir {

class AllocaInst;
class BasicBlock;
class FunctionDefinition;
class Instruction;

namespace detail {

class CoroFrameAtomDomain;
class CoroSemanticGraph;

struct CoroInitializedPrefixProofResult {
    bool succeeded{false};
    Instruction *failing_read{nullptr};
    BasicBlock *placement_block{nullptr};
    Instruction *placement_instruction{nullptr};
    size_t block_evaluation_count{0u};
};

class CoroInitializedPrefixAnalysis final {
private:
    class Impl;
    luisa::unique_ptr<Impl> _impl;

public:
    CoroInitializedPrefixAnalysis(
        FunctionDefinition *definition,
        const CoroSemanticGraph &graph,
        const CoroFrameAtomDomain &domain) noexcept;
    ~CoroInitializedPrefixAnalysis() noexcept;
    CoroInitializedPrefixAnalysis(
        CoroInitializedPrefixAnalysis &&) noexcept;
    CoroInitializedPrefixAnalysis &operator=(
        CoroInitializedPrefixAnalysis &&) noexcept;
    CoroInitializedPrefixAnalysis(
        const CoroInitializedPrefixAnalysis &) = delete;
    CoroInitializedPrefixAnalysis &operator=(
        const CoroInitializedPrefixAnalysis &) = delete;

    [[nodiscard]] CoroInitializedPrefixProofResult prove(
        AllocaInst *array,
        BasicBlock *target,
        Instruction *insertion_instruction) noexcept;

    void notify_target_mutated(
        BasicBlock *target,
        bool moved_store, bool moved_bool_store) noexcept;
};

// Proves a counted-array initialization invariant in the coroutine semantic
// CFG. The accepted transition system is deliberately small and closed:
//
//   Prefix(A, C) := every A[i] with 0 <= i < C is defined.
//
// C = 0 establishes Prefix vacuously. A full-element store to A[C] establishes
// a pending extension, and only a following, non-wrapping C = C + 1 consumes
// that extension while preserving Prefix. CFG joins intersect both facts.
// Reads are accepted only when ordinary aggregate facts already define the
// projection, or when their index is locally proved to select either a
// statically defined sentinel or a value less than C. Unsupported aliasing,
// arithmetic, counter mutation, and pointer escape fail closed.
[[nodiscard]] CoroInitializedPrefixProofResult
prove_initialized_prefix_fresh_lifetime(
    AllocaInst *array,
    BasicBlock *target,
    Instruction *insertion_instruction,
    const CoroSemanticGraph &graph,
    const CoroFrameAtomDomain &domain) noexcept;

}// namespace detail
}// namespace luisa::compute::xir
