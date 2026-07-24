#pragma once

#include <utility>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/function.h>
#include <luisa/xir/passes/dom_tree.h>

namespace luisa::compute::xir {

// A natural loop discovered on a plain (destructured) CFG: a header block
// with one or more back-edges from blocks dominated by the header.
struct LUISA_XIR_API NaturalLoop {
    BasicBlock *header{nullptr};
    // The unique predecessor of the header outside the loop body with no
    // other successors; nullptr when the header has no such preheader.
    BasicBlock *preheader{nullptr};
    // Back-edge source blocks. Passes that require a canonical form should
    // check latches.size() == 1u.
    luisa::vector<BasicBlock *> latches;
    // Blocks inside the loop, NOT including the header.
    luisa::vector<BasicBlock *> body_blocks;
    // Blocks outside the loop that are successors of in-loop blocks.
    luisa::vector<BasicBlock *> exit_blocks;
    luisa::vector<std::pair<BasicBlock *, BasicBlock *>> back_edges;

    [[nodiscard]] bool contains(BasicBlock *block) const noexcept;
    [[nodiscard]] bool is_innermost() const noexcept;
};

// Discover all natural loops in a plain-CFG function definition. Loops are
// ordered so that inner loops come before their parents (by ascending body
// size). The function must not contain structured control flow.
[[nodiscard]] LUISA_XIR_API luisa::vector<NaturalLoop> discover_natural_loops(
    FunctionDefinition *def, const DomTree &dom_tree) noexcept;

// A simple counted-loop pattern: an integer phi in the header with one
// incoming from the preheader (start) and one from a latch (start + stride),
// bounded by a comparison in the header's conditional branch.
struct LoopBoundsInfo {
    PhiInst *induction_phi{nullptr};
    Value *start_value{nullptr};
    Value *bound_value{nullptr};
    ArithmeticOp comparison{ArithmeticOp::BINARY_ADD};// actual comparison op found
    int64_t stride{0};
    bool stride_is_constant{false};
    // Valid only when start, bound, and stride are all constants.
    bool trip_count_is_constant{false};
    uint64_t constant_trip_count{0u};

    [[nodiscard]] bool is_valid() const noexcept { return induction_phi != nullptr; }
};

// Analyze a natural loop for a canonical induction pattern. The loop must
// have a preheader, exactly one latch, and a header terminated by a
// conditional branch whose condition is an integer comparison between the
// induction phi and a bound.
[[nodiscard]] LUISA_XIR_API LoopBoundsInfo analyze_loop_bounds(const NaturalLoop &loop) noexcept;

}// namespace luisa::compute::xir
