#pragma once

#include <cstddef>

#include <luisa/core/stl/memory.h>

namespace luisa::compute::xir {

class BasicBlock;
class FunctionDefinition;

enum struct LoopBoundaryTargetKind {
    NONE,
    BREAK,
    CONTINUE,
    MIXED,
};

// Batch solution of every loop-boundary path query for one loop context.
// The immutable CFG is value-numbered once. Each evaluate() call solves only
// the structurally reachable loop region plus its explicit boundaries.
class LoopBoundaryPathDataflow final {
private:
    class Impl;
    luisa::unique_ptr<Impl> _impl;

public:
    explicit LoopBoundaryPathDataflow(
        FunctionDefinition *definition) noexcept;
    ~LoopBoundaryPathDataflow() noexcept;
    LoopBoundaryPathDataflow(
        LoopBoundaryPathDataflow &&) noexcept;
    LoopBoundaryPathDataflow &operator=(
        LoopBoundaryPathDataflow &&) noexcept;
    LoopBoundaryPathDataflow(
        const LoopBoundaryPathDataflow &) = delete;
    LoopBoundaryPathDataflow &operator=(
        const LoopBoundaryPathDataflow &) = delete;

    void evaluate(BasicBlock *body,
                  BasicBlock *continue_target,
                  BasicBlock *loop_entry,
                  BasicBlock *merge) noexcept;

    [[nodiscard]] LoopBoundaryTargetKind classify(
        BasicBlock *target) const noexcept;
    [[nodiscard]] size_t region_size() const noexcept;
    [[nodiscard]] BasicBlock *region_block(
        size_t index) const noexcept;
    [[nodiscard]] size_t active_block_count() const noexcept;
    [[nodiscard]] size_t edge_visit_count() const noexcept;
};

}// namespace luisa::compute::xir
