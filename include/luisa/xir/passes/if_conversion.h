#pragma once

#include <cstddef>
#include <limits>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;
class Function;
class ConditionalBranchInst;

using IfConversionCandidateFilter = bool (*)(
    const ConditionalBranchInst *branch,
    const void *context) noexcept;

struct IfConversionOptions {
    // Structural limits. Defaults preserve the original pass boundary of at
    // most 16 instructions per arm (32 total).
    size_t max_arm_instruction_count{16u};
    size_t max_total_instruction_count{32u};

    // Cost/register limits are opt-in so target-independent callers retain
    // the historical behavior. One register unit is one 32-bit scalar value;
    // the cost includes both speculative arms and generated live-out selects.
    size_t max_live_out_register_units{
        std::numeric_limits<size_t>::max()};
    size_t max_speculation_cost{
        std::numeric_limits<size_t>::max()};

    // Floating-point division is non-trapping on the supported XIR targets,
    // but remains opt-in because it is substantially more expensive than the
    // total arithmetic accepted by the default target-independent policy.
    // Integer division and every remainder operation remain ineligible.
    bool allow_speculative_float_division{false};

    // A fully constant, verifier-valid aggregate extraction cannot address an
    // invalid element. Keep it opt-in because generic callers may intentionally
    // retain aggregate accesses behind control flow for target cost reasons.
    bool allow_speculative_static_extract{false};

    // Optional target policy. Structural and speculation-safety checks remain
    // inside the pass; the filter can reject otherwise valid candidates (for
    // example, a SIMD backend retaining warp-uniform scalar control flow).
    IfConversionCandidateFilter candidate_filter{nullptr};
    const void *candidate_filter_context{nullptr};
};

struct IfConversionInfo {
    size_t converted_diamond_count{0u};
    size_t hoisted_inst_count{0u};
    size_t replaced_phi_count{0u};
    size_t structured_cfg_error_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return converted_diamond_count != 0u ||
               hoisted_inst_count != 0u ||
               replaced_phi_count != 0u;
    }
    [[nodiscard]] bool succeeded() const noexcept { return structured_cfg_error_count == 0u; }
};

// Unstructured-CFG-only: structured functions are rejected without mutation.
// Metadata on the replaced parent terminator is cloned to the new branch.
// Annotated side blocks or arm-exit terminators are retained because deleting
// either arm provides no unique, verifier-valid metadata owner.
[[nodiscard]] LUISA_XIR_API IfConversionInfo if_conversion_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API IfConversionInfo if_conversion_pass_run_on_function(
    Function *function, IfConversionOptions options) noexcept;
[[nodiscard]] LUISA_XIR_API IfConversionInfo if_conversion_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API IfConversionInfo if_conversion_pass_run_on_module(
    Module *module, IfConversionOptions options,
    PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
