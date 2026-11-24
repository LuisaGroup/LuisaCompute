#pragma once

#include <cstddef>
#include <cstdint>

#include <SPIRV/spirv.hpp11>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {
class FunctionDefinition;
class LoopInst;
}// namespace luisa::compute::xir

namespace lc::spirv {

// Frozen loop-control decision of one structured `xir::LoopInst`.
//
// The default decision is `MaskNone` and is what every synthetic software loop
// (compare-exchange remainder, ray-query traversal, ...) keeps: those are owned
// by an emitter helper rather than by the XIR loop the planner inspects.
struct SpirvLoopControlDecision {
    spv::LoopControlMask mask{spv::LoopControlMask::MaskNone};
    // Exact constant trip count the proof established (0 unless accepted).
    uint64_t trip_count{0u};
    // Checked upper bound on the instructions the unrolled body expands into.
    uint64_t expanded_instruction_budget{0u};
    // Why the loop was (not) accepted. Diagnostic only; never hashed.
    luisa::string reason;

    [[nodiscard]] bool unroll() const noexcept {
        return mask != spv::LoopControlMask::MaskNone;
    }
};

// Backend-only control-flow optimization plan.
//
// It is built from the immutable XIR function *before* any physical block ID is
// allocated, stores nothing but literal decisions, and is never consulted for a
// second chance during emission. It cannot add or remove edges: only loop-control
// literals of an existing `OpLoopMerge` may be set.
//
// The only literals ever planned are `Unroll` for a loop whose trip count is an
// exact compile-time constant and whose body is provably safe to duplicate.
// `DontUnroll`, dependency lengths, peel/partial counts, min/max iteration
// hints, and `SelectionControlMask::Flatten` are deliberately out of scope: no
// exact proof for them exists in the current XIR.
class ControlFlowOptimizationPlan final {

public:
    struct Options {
        // A/B policy switch. `false` keeps every loop at MaskNone.
        bool enable_loop_unroll{true};
        // Largest exact trip count that may be unrolled.
        uint64_t max_trip_count{8u};
        // Largest expanded-instruction budget (trip count x body size).
        uint64_t max_expanded_instructions{512u};

        [[nodiscard]] static Options from_environment() noexcept;
    };

private:
    luisa::unordered_map<const luisa::compute::xir::LoopInst *,
                         SpirvLoopControlDecision>
        _loop_decisions;
    SpirvLoopControlDecision _default_decision;

public:
    // Never fails: a loop that the proof rejects simply keeps MaskNone.
    [[nodiscard]] static ControlFlowOptimizationPlan create(
        const luisa::compute::xir::FunctionDefinition *function,
        Options options = Options::from_environment()) noexcept;

    [[nodiscard]] const SpirvLoopControlDecision &loop_control(
        const luisa::compute::xir::LoopInst *loop) const noexcept;

    [[nodiscard]] size_t unrolled_loop_count() const noexcept {
        auto count = size_t{0u};
        for (auto &&[loop, decision] : _loop_decisions) {
            static_cast<void>(loop);
            count += decision.unroll() ? 1u : 0u;
        }
        return count;
    }
};

// Exposed for direct host tests: proves (or rejects) one loop independently of
// the per-function walk.
[[nodiscard]] SpirvLoopControlDecision plan_spirv_loop_control(
    const luisa::compute::xir::LoopInst *loop,
    ControlFlowOptimizationPlan::Options options) noexcept;

}// namespace lc::spirv