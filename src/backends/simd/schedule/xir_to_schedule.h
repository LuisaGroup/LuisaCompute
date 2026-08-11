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
}// namespace luisa::compute::xir

namespace luisa::compute::simd::schedule {

enum struct XIRToScheduleDiagnosticCode {
    invalid_source,
    invalid_warp_width,
    malformed_cfg,
    structured_control_flow,
    irreducible_control_flow,
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
};

struct XIRToScheduleResult {
    std::optional<Function> function{};
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
