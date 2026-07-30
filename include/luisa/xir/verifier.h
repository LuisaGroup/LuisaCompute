#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class Module;
class Function;
class BasicBlock;
class Instruction;

struct XIRVerificationOptions {
    bool require_terminated_blocks{true};
    bool require_reachable_blocks{false};
    bool require_no_phi{false};
    bool require_no_unstructured_control_flow{false};
    bool require_unique_merge_blocks{false};
    bool require_canonical_break_continue_targets{false};
};

struct XIRVerificationError {
    const Function *function{nullptr};
    const BasicBlock *block{nullptr};
    const Instruction *instruction{nullptr};
    luisa::string message;
};

struct XIRVerificationResult {
    luisa::vector<XIRVerificationError> errors;
    struct Statistics {
        // Number of exact Use-node membership predicates evaluated while
        // checking instruction operands.
        size_t use_list_membership_queries{0u};
        // A verifier invocation materializes each distinct referenced
        // Value/use-list once. This is the operation count that guards the
        // membership check against quadratic high-fanout behavior.
        size_t distinct_use_lists_scanned{0u};
        size_t use_list_entries_scanned{0u};
    } statistics;
    [[nodiscard]] bool succeeded() const noexcept { return errors.empty(); }
};

[[nodiscard]] LUISA_XIR_API XIRVerificationResult
xir_verify_function(const Function *function,
                    const XIRVerificationOptions &options = {}) noexcept;

// Verifies a bounded set of functions with one verifier invocation. This is
// useful for transactional passes whose candidate output lives in shadow
// definitions: the complete output boundary can be checked once without
// re-verifying unrelated input definitions one by one.
[[nodiscard]] LUISA_XIR_API XIRVerificationResult
xir_verify_functions(luisa::span<const Function *const> functions,
                     const XIRVerificationOptions &options = {}) noexcept;

[[nodiscard]] LUISA_XIR_API XIRVerificationResult
xir_verify_module(const Module *module,
                  const XIRVerificationOptions &options = {}) noexcept;

}// namespace luisa::compute::xir
