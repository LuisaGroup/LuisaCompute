#pragma once

#include <cstddef>

#include <luisa/core/stl/memory.h>

namespace luisa::compute::xir {

class BasicBlock;
class DomTree;
class FunctionDefinition;

enum struct LoopBoundaryTargetKind {
    NONE,
    BREAK,
    CONTINUE,
    MIXED,
};

enum struct LoopContinueRewriteKind {
    BREAK,
    CONTINUE,
};

struct LoopContinueRewrite {
    BasicBlock *block{nullptr};
    BasicBlock *from{nullptr};
    BasicBlock *target{nullptr};
    size_t site_index{0u};
    LoopContinueRewriteKind kind{
        LoopContinueRewriteKind::CONTINUE};
};

struct LoopContinueBatchStats {
    size_t region_block_visit_count{0u};
    size_t region_edge_visit_count{0u};
};

// Plans every loop-continue normalization query against one immutable CFG and
// dominance version. Applying the returned guarded rewrites is a separate
// phase; no analysis query observes a partially mutated graph.
class LoopContinueBatchAnalysis final {
private:
    class Impl;
    luisa::unique_ptr<Impl> _impl;

public:
    LoopContinueBatchAnalysis(
        FunctionDefinition *definition,
        const DomTree &dominance) noexcept;
    ~LoopContinueBatchAnalysis() noexcept;
    LoopContinueBatchAnalysis(
        LoopContinueBatchAnalysis &&) noexcept;
    LoopContinueBatchAnalysis &operator=(
        LoopContinueBatchAnalysis &&) noexcept;
    LoopContinueBatchAnalysis(
        const LoopContinueBatchAnalysis &) = delete;
    LoopContinueBatchAnalysis &operator=(
        const LoopContinueBatchAnalysis &) = delete;

    void plan(size_t site_index,
              BasicBlock *loop_entry,
              BasicBlock *body,
              BasicBlock *continue_target,
              BasicBlock *merge) noexcept;

    [[nodiscard]] size_t rewrite_count() const noexcept;
    [[nodiscard]] const LoopContinueRewrite &rewrite(
        size_t index) const noexcept;
    [[nodiscard]] const LoopContinueBatchStats &stats()
        const noexcept;
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
