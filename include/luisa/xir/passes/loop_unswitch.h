#pragma once

#include <cstddef>

#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;
class ConditionalBranchInst;

using LoopUnswitchCandidateFilter = bool (*)(
    const ConditionalBranchInst *branch,
    const void *context) noexcept;

struct LoopUnswitchOptions {
    // The initial implementation deliberately bounds cloning and performs at
    // most one rewrite per function by default. The instruction count includes
    // Phis and terminators in the complete natural-loop region.
    size_t max_loop_instruction_count{64u};
    size_t max_unswitched_loop_count{1u};

    // Permit an unknown-trip canonical top-tested loop by cloning its pure
    // entry condition into a guard. The invariant selector is evaluated only
    // on the guard's entering edge, so the zero-trip path retains the source
    // behavior. This is opt-in because it adds one guard block and is intended
    // for targets that can amortize the extra control over repeated varying
    // branches.
    bool enable_guarded_dynamic_trip{false};

    // Optional target policy. The generic pass already requires an invariant
    // condition and a read-only, cohort-insensitive loop. A target may impose
    // a stricter uniformity or profitability rule here.
    LoopUnswitchCandidateFilter candidate_filter{nullptr};
    const void *candidate_filter_context{nullptr};
};

struct LoopUnswitchInfo {
    size_t unswitched_loop_count{0u};
    size_t guarded_dynamic_loop_count{0u};
    size_t cloned_block_count{0u};
    size_t cloned_instruction_count{0u};
    size_t created_preheader_count{0u};
    size_t merged_live_out_count{0u};
    size_t structured_cfg_error_count{0u};

    [[nodiscard]] bool changed() const noexcept {
        return unswitched_loop_count != 0u;
    }
    [[nodiscard]] bool succeeded() const noexcept {
        return structured_cfg_error_count == 0u;
    }
};

// Clones a positive constant-trip, innermost natural loop around one invariant
// internal conditional. The current fail-closed shape has one preheader,
// latch, and exit edge; no clock, volatile operation, or write is allowed in
// the loop. Live-out SSA values are merged in the unique exit block. Structured
// CFG is rejected without mutation and must first be lowered by destructure_cfg.
[[nodiscard]] LUISA_XIR_API LoopUnswitchInfo
loop_unswitch_pass_run_on_function(
    Function *function, LoopUnswitchOptions options = {}) noexcept;

[[nodiscard]] LUISA_XIR_API LoopUnswitchInfo
loop_unswitch_pass_run_on_module(
    Module *module, LoopUnswitchOptions options = {},
    PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
