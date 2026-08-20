#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "schedule_ir.h"

namespace luisa::compute::xir {
class BasicBlock;
class Function;
class Instruction;
class RayQueryPipelineInst;
}// namespace luisa::compute::xir

namespace luisa::compute::simd::schedule {

enum struct XIRToScheduleDiagnosticCode {
    invalid_source,
    invalid_warp_width,
    malformed_cfg,
    structured_control_flow,
    irreducible_control_flow,
    non_uniform_block_barrier,
    unsupported_instruction,
    unsupported_value,
    invalid_phi,
    schedule_verification,
};

struct XIRToScheduleDiagnostic {
    XIRToScheduleDiagnosticCode code{XIRToScheduleDiagnosticCode::malformed_cfg};
    std::string message{};
    const xir::BasicBlock *block{nullptr};
    const xir::Instruction *instruction{nullptr};
};

struct XIRToScheduleOptions {
    // Zero preserves symbolic width until target specialization.
    uint32_t logical_warp_width{0u};
    // Canonical early-exit loop induction values remain lane-wise state, but
    // a proven cohort-equal header predicate may route the whole active
    // continuation without constructing two successor masks.
    bool enable_cohort_uniform_induction{true};
    // Small loops normally do not repay the additional proof and lowering
    // machinery. Tests and diagnostic experiments may lower this threshold
    // without coupling the schedule projection to emitter-specific controls.
    uint32_t cohort_uniform_induction_min_loop_block_count{25u};
};

struct XIRToScheduleResult {
    std::optional<Function> function{};
    // Stable side table for high-level operations whose function operands are
    // intentionally not copied into dependency-light Schedule IR. The caller
    // consumes these pointers while the source XIR module is still alive.
    std::vector<const xir::RayQueryPipelineInst *> ray_query_pipelines{};
    std::vector<XIRToScheduleDiagnostic> diagnostics{};

    [[nodiscard]] bool succeeded() const noexcept {
        return function.has_value() && diagnostics.empty();
    }
};

// Projects optimized, destructured XIR into the backend-private scheduling
// dialect. This pass is non-mutating. Structured control, irreducible cycles,
// calls that escaped inlining, and not-yet-lowered high-level operations are
// rejected with actionable diagnostics rather than silently scalarized.
[[nodiscard]] XIRToScheduleResult lower_xir_to_schedule(
    const xir::Function *source,
    XIRToScheduleOptions options = {});

[[nodiscard]] const char *to_string(
    XIRToScheduleDiagnosticCode code) noexcept;

}// namespace luisa::compute::simd::schedule
